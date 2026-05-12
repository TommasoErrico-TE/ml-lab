"""
MLP (PyTorch) training for Task 1 and Task 2 on the full 19 214
expression features, with stratified 5-fold cross-validation.

Saves to output_tp53/:
  - task1_mlp_results.csv, task1_mlp_confusion_matrix.csv
  - task2_mlp_results_coding.csv, task2_mlp_confusion_matrix_coding.csv

Usage:
    python scripts/compute_mlp.py
"""
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

warnings.filterwarnings("ignore")

OUTPUT_DIR = Path("output_tp53")
RANDOM_STATE = 42
DEVICE = torch.device("cpu")

NON_FEATURES = {
    "Unnamed: 0",
    "SequencingID", "ModelConditionID", "ModelID",
    "IsDefaultEntryForMC", "IsDefaultEntryForModel", "IsDefaultEntryForModel_bool",
    "TP53_expression", "TP53_damaging_score",
    "task1_tp53_mutated",
    "task2_mutation_type", "task2_mutation_type_int",
    "task2_dna_type", "task2_dna_type_int",
    "CellLineName", "CCLEName",
    "OncotreeLineage", "OncotreePrimaryDisease", "OncotreeSubtype",
    "Sex", "AgeCategory", "PrimaryOrMetastasis",
}


def load():
    master = pd.read_csv(OUTPUT_DIR / "datasets/master.csv")
    master = master.drop(columns=["task2_mutation_type"], errors="ignore")
    labels = pd.read_csv(OUTPUT_DIR / "datasets/task2_labels.csv")
    feature_cols = [c for c in master.columns if c not in NON_FEATURES]
    X_full = master[feature_cols].fillna(master[feature_cols].median()).values
    return master, labels, feature_cols, X_full


def top_k_variance(X, k=None):
    """If k is None or k >= n_features, return X unchanged (all features)."""
    if k is None or k >= X.shape[1]:
        return X
    return X[:, np.argsort(X.var(axis=0))[::-1][:k]]


def make_mlp(in_dim, n_classes, hidden=(256, 128), dropout=0.3):
    layers, prev = [], in_dim
    for h in hidden:
        layers += [nn.Linear(prev, h), nn.BatchNorm1d(h), nn.ReLU(), nn.Dropout(dropout)]
        prev = h
    layers.append(nn.Linear(prev, n_classes))
    return nn.Sequential(*layers)


def train_one_fold(X_tr, y_tr, X_te, n_classes, epochs=60, batch_size=64, lr=1e-3):
    torch.manual_seed(RANDOM_STATE)
    np.random.seed(RANDOM_STATE)

    imp = SimpleImputer(strategy="median").fit(X_tr)
    sc = StandardScaler().fit(imp.transform(X_tr))
    X_tr_s = sc.transform(imp.transform(X_tr)).astype(np.float32)
    X_te_s = sc.transform(imp.transform(X_te)).astype(np.float32)

    cls, counts = np.unique(y_tr, return_counts=True)
    class_w = (counts.sum() / (len(cls) * counts)).astype(np.float32)
    weight_t = torch.tensor(class_w, device=DEVICE)

    model = make_mlp(X_tr_s.shape[1], n_classes).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    crit = nn.CrossEntropyLoss(weight=weight_t)

    ds = TensorDataset(torch.tensor(X_tr_s), torch.tensor(y_tr, dtype=torch.long))
    dl = DataLoader(ds, batch_size=batch_size, shuffle=True)

    for _ in range(epochs):
        model.train()
        for xb, yb in dl:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            opt.zero_grad()
            crit(model(xb), yb).backward()
            opt.step()

    model.eval()
    with torch.no_grad():
        logits = model(torch.tensor(X_te_s, device=DEVICE)).cpu().numpy()
    return logits.argmax(axis=1)


def cv_mlp(X, y, n_classes, label_name):
    print(f"\n  Task: {label_name} (n={len(y)}, classes={n_classes})")
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    y_pred = np.zeros(len(y), dtype=int)
    t0 = time.time()
    for fold, (tr, te) in enumerate(skf.split(X, y), 1):
        y_pred[te] = train_one_fold(X[tr], y[tr], X[te], n_classes)
        print(f"    fold {fold}/5 done [{time.time()-t0:.0f}s]")
    return y_pred


def main():
    OUTPUT_DIR.mkdir(exist_ok=True)
    (OUTPUT_DIR / 'task1').mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / 'task2').mkdir(parents=True, exist_ok=True)
    master, labels, feature_cols, X_full = load()
    print(f"Master: {master.shape}, features: {len(feature_cols)}, device: {DEVICE}")

    # ---------- Task 1 ----------
    y1 = master["task1_tp53_mutated"].values.astype(int)
    X1 = top_k_variance(X_full, None)  # full feature set
    y_pred1 = cv_mlp(X1, y1, n_classes=2, label_name="Task 1 (binary)")
    cm1 = confusion_matrix(y1, y_pred1)
    pd.DataFrame(cm1, index=["WT", "Mutated"], columns=["WT", "Mutated"]).to_csv(
        OUTPUT_DIR / "task1/mlp_confusion_matrix.csv")
    pd.DataFrame([{
        "task": "task1",
        "n_samples": len(y1),
        "accuracy": accuracy_score(y1, y_pred1),
        "macro_f1": f1_score(y1, y_pred1, average="macro", zero_division=0),
        "weighted_f1": f1_score(y1, y_pred1, average="weighted", zero_division=0),
    }]).to_csv(OUTPUT_DIR / "task1/mlp_results.csv", index=False)
    print(f"  Task 1 acc={accuracy_score(y1, y_pred1):.4f}  "
          f"f1={f1_score(y1, y_pred1, average='macro'):.4f}")

    # ---------- Task 2 (coding consequence, 3-class after dropping Inframe) ----------
    df2 = master.merge(
        labels[["ModelID", "task2_mutation_type"]], on="ModelID", how="inner"
    )
    df2 = df2[df2["task2_mutation_type"].isin(["WT", "Missense", "Truncating"])].copy()
    label_map = {"WT": 0, "Missense": 1, "Truncating": 2}
    y2 = df2["task2_mutation_type"].map(label_map).values
    X2_full = df2[feature_cols].fillna(df2[feature_cols].median()).values
    X2 = top_k_variance(X2_full, None)  # full feature set
    y_pred2 = cv_mlp(X2, y2, n_classes=3, label_name="Task 2 (coding, 3-class)")
    cm2 = confusion_matrix(y2, y_pred2)
    pd.DataFrame(cm2, index=["WT", "Missense", "Truncating"],
                  columns=["WT", "Missense", "Truncating"]).to_csv(
        OUTPUT_DIR / "task2/mlp_confusion_matrix.csv")
    pd.DataFrame([{
        "task": "task2_coding",
        "n_samples": len(y2),
        "accuracy": accuracy_score(y2, y_pred2),
        "macro_f1": f1_score(y2, y_pred2, average="macro", zero_division=0),
        "weighted_f1": f1_score(y2, y_pred2, average="weighted", zero_division=0),
    }]).to_csv(OUTPUT_DIR / "task2/mlp_results.csv", index=False)
    print(f"  Task 2 acc={accuracy_score(y2, y_pred2):.4f}  "
          f"macro_f1={f1_score(y2, y_pred2, average='macro'):.4f}")

    print("\nDone. CSVs saved in", OUTPUT_DIR.resolve())


if __name__ == "__main__":
    main()
