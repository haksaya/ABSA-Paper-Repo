"""
Classical baselines — 5-Fold Cross-Validation
Dataset : data/sample_dataset.csv
Models  : majority class | aspect-only | TF-IDF + Logistic Regression | TF-IDF + Linear SVM
Output  : results/baseline_kfold_results.json

Run under the same protocol as the neural models: folds are partitioned at the
review (sentence_id) level with KFold(5, shuffle=True, random_state=42), the input
text is "sentence + space + aspect", and class imbalance is handled with
class_weight="balanced" (the counterpart of the weighted loss used elsewhere).

The aspect-only predictor never sees the sentence; it returns the training-fold
majority polarity of each aspect category. It measures how much of the task is
solvable from the category label alone, which is the floor any model with access
to that label starts from.
"""
import sys, json, time
sys.stdout.reconfigure(encoding='utf-8')

import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

# ── Config ────────────────────────────────────────────────────────────────────
SEED    = 42
N_FOLDS = 5

df = pd.read_csv("data/sample_dataset.csv", encoding="utf-8-sig")
df["text"] = df["sentence"].astype(str) + " " + df["aspect"].astype(str)


class AspectMajority:
    """Ignores the sentence; predicts each aspect category's training majority class.

    A category unseen during training falls back to the corpus-level majority.
    """

    def fit(self, aspects, y):
        s = pd.DataFrame({"a": list(aspects), "y": list(y)})
        self.maj_ = s.groupby("a")["y"].agg(lambda v: v.mode().iat[0])
        self.global_ = s["y"].mode().iat[0]
        return self

    def predict(self, aspects):
        return pd.Series(list(aspects)).map(self.maj_).fillna(self.global_).astype(int).values


def make_models():
    """Fresh model instances for every fold."""
    tfidf = dict(ngram_range=(1, 2), min_df=2, sublinear_tf=True)
    return {
        "Majority": DummyClassifier(strategy="most_frequent"),
        "Aspect-only": AspectMajority(),
        "TF-IDF + LR": Pipeline([
            ("vec", TfidfVectorizer(**tfidf)),
            ("clf", LogisticRegression(max_iter=2000, class_weight="balanced",
                                       random_state=SEED)),
        ]),
        "TF-IDF + SVM": Pipeline([
            ("vec", TfidfVectorizer(**tfidf)),
            ("clf", LinearSVC(class_weight="balanced", random_state=SEED)),
        ]),
    }


# ── 5-fold cross-validation ───────────────────────────────────────────────────
sids = df["sentence_id"].unique()
kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)

results = {name: [] for name in make_models()}

for fold_idx, (train_idx, test_idx) in enumerate(kf.split(sids), 1):
    tr = df[df["sentence_id"].isin(sids[train_idx])]
    te = df[df["sentence_id"].isin(sids[test_idx])]
    print(f"Fold {fold_idx}: train={len(tr)} test={len(te)}")

    for name, model in make_models().items():
        t0 = time.time()
        if name == "Aspect-only":
            X_tr, X_te = tr["aspect"], te["aspect"]          # sentence text unused
        elif name == "Majority":
            X_tr = tr["text"].values.reshape(-1, 1)
            X_te = te["text"].values.reshape(-1, 1)
        else:
            X_tr, X_te = tr["text"], te["text"]
        model.fit(X_tr, tr["label"])
        pred = model.predict(X_te)
        rec = {
            "fold": fold_idx,
            "accuracy": round(accuracy_score(te["label"], pred), 4),
            "macro_f1": round(f1_score(te["label"], pred, average="macro", zero_division=0), 4),
            "f1_negative": round(f1_score(te["label"], pred, pos_label=0, zero_division=0), 4),
            "f1_positive": round(f1_score(te["label"], pred, pos_label=1, zero_division=0), 4),
            "elapsed_sec": round(time.time() - t0, 2),
        }
        results[name].append(rec)
        print(f"   {name:14} acc={rec['accuracy']:.4f} macroF1={rec['macro_f1']:.4f} "
              f"F1neg={rec['f1_negative']:.4f}")

# ── Summary ───────────────────────────────────────────────────────────────────
print()
print("=" * 72)
print(f"{'Model':15} {'Accuracy':>16} {'Macro-F1':>16} {'F1-Negative':>16}")
print("=" * 72)
summary = {}
for name, folds in results.items():
    row = {}
    for key in ("accuracy", "macro_f1", "f1_negative", "f1_positive"):
        vals = [f[key] for f in folds]
        row[key] = {"mean": round(float(np.mean(vals)), 4),
                    "std": round(float(np.std(vals)), 4)}
    summary[name] = row
    print(f"{name:15} "
          f"{row['accuracy']['mean']:.4f}±{row['accuracy']['std']:.3f}   "
          f"{row['macro_f1']['mean']:.4f}±{row['macro_f1']['std']:.3f}   "
          f"{row['f1_negative']['mean']:.4f}±{row['f1_negative']['std']:.3f}")

with open("results/baseline_kfold_results.json", "w", encoding="utf-8") as f:
    json.dump({"folds": results, "summary": summary}, f, ensure_ascii=False, indent=2)
print("\n-> results/baseline_kfold_results.json")
