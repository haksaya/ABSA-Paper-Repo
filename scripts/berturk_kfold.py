"""
BERTurk (dbmdz/bert-base-turkish-cased) — 5-Fold Cross-Validation
Dataset : data/dataset.csv
LR      : 2e-5  (previous grid-search best)
Output  : results/bert_kfold_results.json
"""
import sys, json, time, random
sys.stdout.reconfigure(encoding='utf-8')

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from transformers import (BertTokenizer, BertForSequenceClassification,
                          get_linear_schedule_with_warmup)
from sklearn.model_selection import KFold
from sklearn.metrics import (accuracy_score, f1_score,
                             classification_report, confusion_matrix)

# ── Config ────────────────────────────────────────────────────────────────────
MODEL_NAME   = "dbmdz/bert-base-turkish-cased"
SEED         = 42
N_FOLDS      = 5
VAL_RATIO    = 0.15
MAX_LEN      = 128
BATCH_SIZE   = 16
MAX_EPOCHS   = 10
PATIENCE     = 3
LR           = 2e-5
WARMUP_RATIO = 0.10
WEIGHT_DECAY = 0.01
GRAD_CLIP    = 1.0
RESULTS_FILE = Path("results/bert_kfold_results.json")
RESULTS_FILE.parent.mkdir(exist_ok=True)

def set_seed(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(s)

set_seed(SEED)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device : {device}")

# ── Data ──────────────────────────────────────────────────────────────────────
df = pd.read_csv("data/dataset.csv", encoding="utf-8-sig")
print(f"Dataset: {len(df)} pairs | {df['sentence_id'].nunique()} reviews")

tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)
print(f"Tokenizer loaded: {MODEL_NAME}")

# ── PyTorch Dataset ───────────────────────────────────────────────────────────
class ABSADataset(Dataset):
    def __init__(self, df, tokenizer, max_len):
        self.data      = df.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.max_len   = max_len

    def __len__(self): return len(self.data)

    def __getitem__(self, i):
        row = self.data.iloc[i]
        enc = self.tokenizer(
            str(row['aspect']), str(row['sentence']),
            max_length=self.max_len, padding='max_length',
            truncation=True, return_tensors='pt'
        )
        return {
            'input_ids':      enc['input_ids'].squeeze(0),
            'attention_mask': enc['attention_mask'].squeeze(0),
            'token_type_ids': enc['token_type_ids'].squeeze(0),
            'label':          torch.tensor(int(row['label']), dtype=torch.long),
        }

# ── Train / Eval epoch ────────────────────────────────────────────────────────
def run_epoch(model, loader, criterion, optimizer=None, scheduler=None):
    train = optimizer is not None
    model.train() if train else model.eval()
    total_loss, preds, labels = 0.0, [], []
    ctx = torch.enable_grad() if train else torch.no_grad()
    with ctx:
        for batch in loader:
            ids   = batch['input_ids'].to(device)
            mask  = batch['attention_mask'].to(device)
            ttype = batch['token_type_ids'].to(device)
            lbls  = batch['label'].to(device)
            if train: optimizer.zero_grad()
            out  = model(ids, attention_mask=mask, token_type_ids=ttype)
            loss = criterion(out.logits, lbls)
            if train:
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
                optimizer.step()
                if scheduler: scheduler.step()
            total_loss += loss.item()
            preds.extend(out.logits.argmax(-1).cpu().tolist())
            labels.extend(lbls.cpu().tolist())
    f1  = f1_score(labels, preds, average='macro', zero_division=0)
    acc = accuracy_score(labels, preds)
    return total_loss / len(loader), acc, f1, preds, labels

# ── 5-Fold CV ─────────────────────────────────────────────────────────────────
sids = df['sentence_id'].unique()
kf   = KFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)

fold_results = []
t_total = time.time()

for fold_idx, (train_idx, test_idx) in enumerate(kf.split(sids), 1):
    set_seed(SEED + fold_idx)
    t0 = time.time()
    print(f"\n{'='*60}")
    print(f"  BERTurk  Fold {fold_idx}/{N_FOLDS}  LR={LR}")
    print(f"{'='*60}")

    train_sids = sids[train_idx]
    test_sids  = sids[test_idx]

    rng = np.random.default_rng(SEED + fold_idx)
    rng.shuffle(train_sids)
    n_val    = int(len(train_sids) * VAL_RATIO)
    val_sids = train_sids[:n_val]
    tr_sids  = train_sids[n_val:]

    tr_df  = df[df['sentence_id'].isin(tr_sids)]
    val_df = df[df['sentence_id'].isin(val_sids)]
    te_df  = df[df['sentence_id'].isin(test_sids)]
    print(f"  Train: {len(tr_df)} | Val: {len(val_df)} | Test: {len(te_df)}")

    # Weighted loss
    neg_c = (tr_df['label'] == 0).sum()
    pos_c = (tr_df['label'] == 1).sum()
    w     = torch.tensor([pos_c / neg_c, 1.0], dtype=torch.float).to(device)
    crit  = nn.CrossEntropyLoss(weight=w)

    tr_loader  = DataLoader(ABSADataset(tr_df,  tokenizer, MAX_LEN),
                            batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(ABSADataset(val_df, tokenizer, MAX_LEN),
                            batch_size=BATCH_SIZE)
    te_loader  = DataLoader(ABSADataset(te_df,  tokenizer, MAX_LEN),
                            batch_size=BATCH_SIZE)

    model = BertForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)
    model.to(device)

    optimizer    = AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    total_steps  = len(tr_loader) * MAX_EPOCHS
    warmup_steps = int(total_steps * WARMUP_RATIO)
    scheduler    = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    history = {k: [] for k in
               ['train_loss','val_loss','train_f1','val_f1','train_acc','val_acc']}
    best_val_f1  = 0.0
    best_state   = None
    patience_cnt = 0

    for epoch in range(1, MAX_EPOCHS + 1):
        ep_t0 = time.time()
        tr_loss, tr_acc, tr_f1, _, _ = run_epoch(model, tr_loader, crit, optimizer, scheduler)
        vl_loss, vl_acc, vl_f1, _, _ = run_epoch(model, val_loader, crit)
        ep_min = (time.time() - ep_t0) / 60

        for k, v in zip(['train_loss','val_loss','train_f1','val_f1','train_acc','val_acc'],
                        [tr_loss, vl_loss, tr_f1, vl_f1, tr_acc, vl_acc]):
            history[k].append(round(float(v), 4))

        if vl_f1 > best_val_f1:
            best_val_f1  = vl_f1
            best_state   = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_cnt = 0
            mark = " *"
        else:
            patience_cnt += 1
            mark = f" ({patience_cnt}/{PATIENCE})"

        print(f"  Ep {epoch:2d} | tr={tr_loss:.4f}/{tr_f1:.4f}"
              f" | val={vl_loss:.4f}/{vl_f1:.4f} | {ep_min:.1f}dk{mark}", flush=True)

        if patience_cnt >= PATIENCE:
            print("  [Early Stop]")
            break

    # Test
    model.load_state_dict(best_state)
    model.to(device)
    _, te_acc, te_f1, te_preds, te_labels = run_epoch(model, te_loader, crit)
    f1_pos = f1_score(te_labels, te_preds, pos_label=1, average='binary')
    f1_neg = f1_score(te_labels, te_preds, pos_label=0, average='binary')
    cm     = confusion_matrix(te_labels, te_preds).tolist()
    elapsed = time.time() - t0

    print(f"\n  TEST  acc={te_acc:.4f}  MacF1={te_f1:.4f}"
          f"  F1-Pos={f1_pos:.4f}  F1-Neg={f1_neg:.4f}  ({elapsed/60:.1f} dk)")
    print(classification_report(te_labels, te_preds,
                                target_names=['Negative','Positive']))

    fold_results.append({
        'fold':        fold_idx,
        'accuracy':    round(float(te_acc), 4),
        'macro_f1':    round(float(te_f1),  4),
        'f1_positive': round(float(f1_pos), 4),
        'f1_negative': round(float(f1_neg), 4),
        'best_val_f1': round(float(best_val_f1), 4),
        'epochs_run':  len(history['train_f1']),
        'elapsed_sec': round(elapsed, 1),
        'history':     history,
        'predictions': [int(p) for p in te_preds],
        'true_labels': [int(l) for l in te_labels],
        'confusion_matrix': cm,
    })

    # Incremental save
    with open(RESULTS_FILE, 'w', encoding='utf-8') as f:
        json.dump({'BERTurk': fold_results}, f, indent=2, ensure_ascii=False)
    print(f"  [{fold_idx}/{N_FOLDS}] kaydedildi -> {RESULTS_FILE}")

    del model; torch.cuda.empty_cache() if torch.cuda.is_available() else None

# ── Final summary ─────────────────────────────────────────────────────────────
print(f"\nToplam sure: {(time.time()-t_total)/60:.1f} dk")
print("\n" + "="*60)
print("BERTurk 5-Fold Summary")
print("="*60)
for m in ['accuracy','macro_f1','f1_negative','f1_positive']:
    vals = [r[m] for r in fold_results]
    print(f"  {m:<15}: {np.mean(vals):.4f} ± {np.std(vals):.4f}")
print("="*60)
print("=== BERT K-FOLD TAMAM ===")
