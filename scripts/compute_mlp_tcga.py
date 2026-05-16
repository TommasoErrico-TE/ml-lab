"""
MLP (PyTorch) per Task 1 binario su TCGA Pan-Cancer.

Stessa architettura di compute_mlp.py (CCLE), allenata con il 3-way split
60/20/20 (train + val per early stopping informale, test per il report).

Auto-detect GPU se disponibile (per HPC); altrimenti CPU.

Output (in output_tp53/tcga/):
  - tcga_mlp_results.csv
  - tcga_mlp_confusion_matrix.csv
"""

import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, confusion_matrix

warnings.filterwarnings("ignore")

TCGA_MASTER = Path("output_tp53/datasets/tcga_master.parquet")
OUTPUT_DIR = Path("output_tp53/tcga")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 42
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

NON_FEATURE_TCGA = {
    "sample", "task1_tp53_mutated", "task2_mutation_type",
    "worst_effect", "worst_protein_change", "tcga_cancer_type",
}


def make_mlp(in_dim, n_classes, hidden=(256, 128), dropout=0.3):
    layers, prev = [], in_dim
    for h in hidden:
        layers += [nn.Linear(prev, h), nn.BatchNorm1d(h), nn.ReLU(), nn.Dropout(dropout)]
        prev = h
    layers.append(nn.Linear(prev, n_classes))
    return nn.Sequential(*layers)


def train(X_tr, y_tr, X_val, y_val, X_te, y_te, n_classes,
          epochs=60, batch_size=64, lr=1e-3):
    torch.manual_seed(RANDOM_STATE)
    np.random.seed(RANDOM_STATE)

    imp = SimpleImputer(strategy="median").fit(X_tr)
    sc = StandardScaler().fit(imp.transform(X_tr))

    def prep(X):
        return sc.transform(imp.transform(X)).astype(np.float32)

    X_tr_s = prep(X_tr); X_val_s = prep(X_val); X_te_s = prep(X_te)

    cls, counts = np.unique(y_tr, return_counts=True)
    weight_t = torch.tensor((counts.sum() / (len(cls) * counts)).astype(np.float32), device=DEVICE)

    model = make_mlp(X_tr_s.shape[1], n_classes).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    crit = nn.CrossEntropyLoss(weight=weight_t)

    ds = TensorDataset(torch.tensor(X_tr_s), torch.tensor(y_tr, dtype=torch.long))
    dl = DataLoader(ds, batch_size=batch_size, shuffle=True)

    t0 = time.time()
    for ep in range(epochs):
        model.train()
        for xb, yb in dl:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            opt.zero_grad()
            crit(model(xb), yb).backward()
            opt.step()
        if (ep + 1) % 10 == 0:
            print(f"  epoch {ep+1}/{epochs}  [{time.time()-t0:.0f}s]")

    model.eval()
    def predict(X):
        with torch.no_grad():
            logits = model(torch.tensor(X, device=DEVICE)).cpu().numpy()
        return logits.argmax(axis=1), logits

    yv_pred, _ = predict(X_val_s)
    yt_pred, t_logits = predict(X_te_s)
    t_proba = torch.softmax(torch.tensor(t_logits), dim=1).numpy()[:, 1]
    return yv_pred, yt_pred, t_proba


def main():
    print(f"Device: {DEVICE}")
    df = pd.read_parquet(TCGA_MASTER)
    feat = [c for c in df.columns if c not in NON_FEATURE_TCGA]
    X = df[feat].fillna(df[feat].median()).values.astype(np.float32)
    y = df["task1_tp53_mutated"].values.astype(int)
    print(f"X: {X.shape}   y counts: {dict(pd.Series(y).value_counts())}")

    X_tr, X_rest, y_tr, y_rest = train_test_split(
        X, y, train_size=0.6, random_state=RANDOM_STATE, stratify=y,
    )
    X_val, X_te, y_val, y_te = train_test_split(
        X_rest, y_rest, train_size=0.5, random_state=RANDOM_STATE, stratify=y_rest,
    )
    print(f"Train: {X_tr.shape}, Val: {X_val.shape}, Test: {X_te.shape}")

    yv_pred, yt_pred, yt_proba = train(
        X_tr, y_tr, X_val, y_val, X_te, y_te, n_classes=2,
    )

    val_acc = accuracy_score(y_val, yv_pred)
    val_f1 = f1_score(y_val, yv_pred, average="macro", zero_division=0)
    te_acc = accuracy_score(y_te, yt_pred)
    te_f1 = f1_score(y_te, yt_pred, average="macro", zero_division=0)
    te_auc = roc_auc_score(y_te, yt_proba)
    cm = confusion_matrix(y_te, yt_pred)
    pd.DataFrame(cm, index=["WT", "Mutated"], columns=["WT", "Mutated"]).to_csv(
        OUTPUT_DIR / "tcga_mlp_confusion_matrix.csv",
    )
    pd.DataFrame([{
        "task": "task1_tcga",
        "n_train": len(y_tr), "n_val": len(y_val), "n_test": len(y_te),
        "val_accuracy": val_acc, "val_macro_f1": val_f1,
        "test_accuracy": te_acc, "test_macro_f1": te_f1, "test_roc_auc": te_auc,
    }]).to_csv(OUTPUT_DIR / "tcga_mlp_results.csv", index=False)
    print(f"\n  test AUC = {te_auc:.4f}   test macro-F1 = {te_f1:.4f}")
    print(f"Saved CSVs in {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
