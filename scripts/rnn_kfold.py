"""
BiLSTM & BiGRU — 5-Fold Cross-Validation
Dataset : data/dataset.csv
Config  : BiLSTM hidden=64 layers=2 | BiGRU hidden=256 layers=2
Output  : results/rnn_kfold_results.json
"""
import sys, json, time, random
sys.stdout.reconfigure(encoding='utf-8')

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import KFold
from sklearn.metrics import (accuracy_score, f1_score,
                             classification_report, confusion_matrix)

# ── Config ────────────────────────────────────────────────────────────────────
SEED        = 42
N_FOLDS     = 5
VAL_RATIO   = 0.15      # val split from each train fold (early stopping)
EMBED_DIM   = 128
DROPOUT     = 0.3
LR          = 1e-3
BATCH_SIZE  = 32
MAX_EPOCHS  = 30
PATIENCE    = 5
MAX_LEN     = 50        # max tokens per sentence (whitespace)

MODELS_CFG = {
    "BiLSTM": {"hidden_dim": 64,  "n_layers": 2},
    "BiGRU":  {"hidden_dim": 256, "n_layers": 2},
}

RESULTS_FILE = "results/rnn_kfold_results.json"

def set_seed(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s)

set_seed(SEED)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device : {device}")

# ── Data ──────────────────────────────────────────────────────────────────────
df = pd.read_csv("data/dataset.csv", encoding="utf-8-sig")
print(f"Dataset: {len(df)} pairs | {df['sentence_id'].nunique()} reviews")

# ── Tokeniser (whitespace) ────────────────────────────────────────────────────
def tokenise(text):
    return str(text).lower().split()

class Vocab:
    PAD, UNK = 0, 1
    def __init__(self):
        self.w2i = {"<PAD>": 0, "<UNK>": 1}
    def build(self, sentences):
        for s in sentences:
            for w in tokenise(s):
                if w not in self.w2i:
                    self.w2i[w] = len(self.w2i)
    def encode(self, text, max_len):
        ids = [self.w2i.get(w, self.UNK) for w in tokenise(text)]
        ids = ids[:max_len]
        ids += [self.PAD] * (max_len - len(ids))
        return ids
    def __len__(self): return len(self.w2i)

# ── PyTorch Dataset ───────────────────────────────────────────────────────────
class ABSADataset(Dataset):
    def __init__(self, df, vocab, max_len):
        self.data    = df.reset_index(drop=True)
        self.vocab   = vocab
        self.max_len = max_len
    def __len__(self): return len(self.data)
    def __getitem__(self, i):
        row  = self.data.iloc[i]
        text = str(row["sentence"]) + " " + str(row["aspect"])
        ids  = self.vocab.encode(text, self.max_len)
        return (torch.tensor(ids, dtype=torch.long),
                torch.tensor(int(row["label"]), dtype=torch.long))

# ── Model ─────────────────────────────────────────────────────────────────────
class BiRNN(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, n_layers,
                 dropout, rnn_type="LSTM"):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        RNN = nn.LSTM if rnn_type == "LSTM" else nn.GRU
        self.rnn = RNN(embed_dim, hidden_dim, num_layers=n_layers,
                       batch_first=True, bidirectional=True,
                       dropout=dropout if n_layers > 1 else 0.0)
        self.drop = nn.Dropout(dropout)
        self.fc   = nn.Linear(hidden_dim * 2, 2)

    def forward(self, x):
        emb = self.drop(self.embed(x))
        out, _ = self.rnn(emb)
        pooled = out.mean(dim=1)
        return self.fc(self.drop(pooled))

# ── Train / Eval epoch ────────────────────────────────────────────────────────
def run_epoch(model, loader, criterion, optimizer=None):
    train = optimizer is not None
    model.train() if train else model.eval()
    total_loss, preds, labels = 0.0, [], []
    ctx = torch.enable_grad() if train else torch.no_grad()
    with ctx:
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            if train: optimizer.zero_grad()
            logits = model(x)
            loss   = criterion(logits, y)
            if train:
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
            total_loss += loss.item()
            preds.extend(logits.argmax(-1).cpu().tolist())
            labels.extend(y.cpu().tolist())
    f1  = f1_score(labels, preds, average="macro", zero_division=0)
    acc = accuracy_score(labels, preds)
    return total_loss / len(loader), acc, f1, preds, labels

# ── K-Fold ────────────────────────────────────────────────────────────────────
sids  = df["sentence_id"].unique()
kf    = KFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)

all_results = {}   # model_name -> list of fold dicts

for model_name, cfg in MODELS_CFG.items():
    rnn_type = "LSTM" if "LSTM" in model_name else "GRU"
    print(f"\n{'='*60}")
    print(f"  {model_name}  hidden={cfg['hidden_dim']}  layers={cfg['n_layers']}")
    print(f"{'='*60}")
    fold_results = []

    for fold_idx, (train_idx, test_idx) in enumerate(kf.split(sids), 1):
        set_seed(SEED + fold_idx)
        t0 = time.time()
        print(f"\n  ── Fold {fold_idx}/{N_FOLDS} ──")

        train_sids = sids[train_idx]
        test_sids  = sids[test_idx]

        # Val split from train (sentence-level)
        rng = np.random.default_rng(SEED + fold_idx)
        rng.shuffle(train_sids)
        n_val       = int(len(train_sids) * VAL_RATIO)
        val_sids    = train_sids[:n_val]
        tr_sids     = train_sids[n_val:]

        tr_df  = df[df["sentence_id"].isin(tr_sids)]
        val_df = df[df["sentence_id"].isin(val_sids)]
        te_df  = df[df["sentence_id"].isin(test_sids)]

        # Vocab from train only
        vocab = Vocab()
        vocab.build(tr_df["sentence"].tolist() + tr_df["aspect"].tolist())

        # Weighted loss
        neg_c  = (tr_df["label"] == 0).sum()
        pos_c  = (tr_df["label"] == 1).sum()
        w      = torch.tensor([pos_c / neg_c, 1.0], dtype=torch.float).to(device)
        crit   = nn.CrossEntropyLoss(weight=w)

        tr_loader  = DataLoader(ABSADataset(tr_df,  vocab, MAX_LEN),
                                batch_size=BATCH_SIZE, shuffle=True)
        val_loader = DataLoader(ABSADataset(val_df, vocab, MAX_LEN),
                                batch_size=BATCH_SIZE)
        te_loader  = DataLoader(ABSADataset(te_df,  vocab, MAX_LEN),
                                batch_size=BATCH_SIZE)

        model = BiRNN(len(vocab), EMBED_DIM, cfg["hidden_dim"],
                      cfg["n_layers"], DROPOUT, rnn_type).to(device)
        optim = torch.optim.Adam(model.parameters(), lr=LR)

        history = {k: [] for k in
                   ["train_loss","val_loss","train_f1","val_f1","train_acc","val_acc"]}
        best_val_f1   = 0.0
        best_state    = None
        patience_cnt  = 0

        for epoch in range(1, MAX_EPOCHS + 1):
            tr_loss, tr_acc, tr_f1, _, _ = run_epoch(model, tr_loader, crit, optim)
            vl_loss, vl_acc, vl_f1, _, _ = run_epoch(model, val_loader, crit)
            for k, v in zip(["train_loss","val_loss","train_f1","val_f1",
                              "train_acc","val_acc"],
                             [tr_loss, vl_loss, tr_f1, vl_f1, tr_acc, vl_acc]):
                history[k].append(round(float(v), 4))

            if vl_f1 > best_val_f1:
                best_val_f1  = vl_f1
                best_state   = {k: v.cpu().clone()
                                for k, v in model.state_dict().items()}
                patience_cnt = 0
                mark = " *"
            else:
                patience_cnt += 1
                mark = f" ({patience_cnt}/{PATIENCE})"
            print(f"    Ep {epoch:2d} | tr_loss={tr_loss:.4f} tr_f1={tr_f1:.4f}"
                  f" | val_loss={vl_loss:.4f} val_f1={vl_f1:.4f}{mark}")
            if patience_cnt >= PATIENCE:
                print("    [Early Stop]")
                break

        # Test with best weights
        model.load_state_dict(best_state)
        _, te_acc, te_f1, te_preds, te_labels = run_epoch(model, te_loader, crit)
        f1_pos = f1_score(te_labels, te_preds, pos_label=1, average="binary")
        f1_neg = f1_score(te_labels, te_preds, pos_label=0, average="binary")
        cm     = confusion_matrix(te_labels, te_preds).tolist()
        elapsed = time.time() - t0

        print(f"    TEST  acc={te_acc:.4f}  MacF1={te_f1:.4f}"
              f"  F1-Pos={f1_pos:.4f}  F1-Neg={f1_neg:.4f}"
              f"  ({elapsed/60:.1f} dk)")

        fold_results.append({
            "fold":        fold_idx,
            "accuracy":    round(float(te_acc), 4),
            "macro_f1":    round(float(te_f1),  4),
            "f1_positive": round(float(f1_pos), 4),
            "f1_negative": round(float(f1_neg), 4),
            "best_val_f1": round(float(best_val_f1), 4),
            "epochs_run":  len(history["train_f1"]),
            "elapsed_sec": round(elapsed, 1),
            "history":     history,
            "predictions": [int(p) for p in te_preds],
            "true_labels": [int(l) for l in te_labels],
            "confusion_matrix": cm,
        })

    all_results[model_name] = fold_results

    # Fold summary
    macf1s = [r["macro_f1"] for r in fold_results]
    accs   = [r["accuracy"]  for r in fold_results]
    print(f"\n  {model_name} 5-Fold Summary:")
    print(f"    Macro-F1 : {np.mean(macf1s):.4f} ± {np.std(macf1s):.4f}")
    print(f"    Accuracy : {np.mean(accs):.4f}  ± {np.std(accs):.4f}")

# ── Save ──────────────────────────────────────────────────────────────────────
import pathlib
pathlib.Path("results").mkdir(exist_ok=True)
with open(RESULTS_FILE, "w", encoding="utf-8") as f:
    json.dump(all_results, f, indent=2, ensure_ascii=False)
print(f"\nSonuclar kaydedildi: {RESULTS_FILE}")

# ── Final table ───────────────────────────────────────────────────────────────
print("\n" + "="*65)
print(f"{'Model':<10} {'Acc':>8} {'MacF1':>8} {'F1-Neg':>8} {'F1-Pos':>8}")
print("-"*65)
for mname, folds in all_results.items():
    acc  = np.mean([r["accuracy"]    for r in folds])
    mf1  = np.mean([r["macro_f1"]    for r in folds])
    fn   = np.mean([r["f1_negative"] for r in folds])
    fp   = np.mean([r["f1_positive"] for r in folds])
    sacc = np.std([r["accuracy"]     for r in folds])
    smf1 = np.std([r["macro_f1"]     for r in folds])
    print(f"{mname:<10} {acc:.4f}±{sacc:.3f}  {mf1:.4f}±{smf1:.3f}"
          f"  {fn:.4f}  {fp:.4f}")
print("="*65)
print("=== RNN K-FOLD TAMAM ===")
