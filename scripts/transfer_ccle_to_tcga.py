"""
Transfer test: alleniamo i modelli su tutto CCLE e li testiamo su TCGA.

Pipeline:
  1) Carica CCLE master + TCGA master, intersezione geni
  2) Standardizza le feature usando media/std del train (CCLE)
  3) Fitta LogReg / RandomForest / XGBoost su CCLE intero
  4) Predice su TCGA, calcola AUC / F1 / accuracy
  5) Salva metriche e confusion matrix per ogni modello

Output (in output_tp53/tcga/):
  - transfer_metrics.csv
  - transfer_cm_<model>.csv
"""

import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score, confusion_matrix,
)

warnings.filterwarnings("ignore")

CCLE_MASTER = Path("output_tp53/datasets/master.csv")
TCGA_MASTER = Path("output_tp53/datasets/tcga_master.parquet")
OUTPUT_DIR = Path("output_tp53/tcga")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 42
TARGET_COL = "task1_tp53_mutated"

NON_FEATURE_CCLE = {
    "Unnamed: 0", "SequencingID", "ModelConditionID", "ModelID",
    "IsDefaultEntryForMC", "IsDefaultEntryForModel", "IsDefaultEntryForModel_bool",
    "TP53_expression", "TP53_damaging_score",
    "task1_tp53_mutated", "task2_mutation_type",
    "CellLineName", "CCLEName",
    "OncotreeLineage", "OncotreePrimaryDisease", "OncotreeSubtype",
    "Sex", "AgeCategory", "PrimaryOrMetastasis",
}

NON_FEATURE_TCGA = {
    "sample", "task1_tp53_mutated", "task2_mutation_type",
    "worst_effect", "worst_protein_change", "tcga_cancer_type",
}

try:
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False


def load_aligned():
    print("Loading CCLE master...")
    ccle = pd.read_csv(CCLE_MASTER)
    ccle_genes = [c for c in ccle.columns if c not in NON_FEATURE_CCLE]
    y_ccle = ccle[TARGET_COL].astype(int).values

    print("Loading TCGA master...")
    tcga = pd.read_parquet(TCGA_MASTER)
    tcga_genes = [c for c in tcga.columns if c not in NON_FEATURE_TCGA]
    y_tcga = tcga[TARGET_COL].astype(int).values

    common = sorted(set(ccle_genes) & set(tcga_genes))
    print(f"  CCLE: {ccle.shape},  TCGA: {tcga.shape},  geni in comune: {len(common)}")

    X_ccle = ccle[common].copy()
    X_tcga = tcga[common].copy()
    return X_ccle, y_ccle, X_tcga, y_tcga, common


def make_models():
    models = {}
    models["LogReg"] = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(
            max_iter=3000, random_state=RANDOM_STATE, class_weight="balanced",
        )),
    ])
    models["RandomForest"] = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("clf", RandomForestClassifier(
            n_estimators=300, random_state=RANDOM_STATE,
            n_jobs=-1, class_weight="balanced",
        )),
    ])
    if XGBOOST_AVAILABLE:
        models["XGBoost"] = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("clf", XGBClassifier(
                n_estimators=300, max_depth=6, learning_rate=0.05,
                subsample=0.8, colsample_bytree=0.8,
                objective="binary:logistic", eval_metric="logloss",
                random_state=RANDOM_STATE, n_jobs=-1,
            )),
        ])
    return models


def main():
    X_ccle, y_ccle, X_tcga, y_tcga, _ = load_aligned()
    print(f"  CCLE label dist: {dict(pd.Series(y_ccle).value_counts())}")
    print(f"  TCGA label dist: {dict(pd.Series(y_tcga).value_counts())}")

    rows = []
    for name, model in make_models().items():
        print(f"\nFitting {name} su CCLE intero...")
        model.fit(X_ccle, y_ccle)

        y_pred = model.predict(X_tcga)
        y_proba = model.predict_proba(X_tcga)[:, 1]

        auc = roc_auc_score(y_tcga, y_proba)
        f1 = f1_score(y_tcga, y_pred, zero_division=0)
        acc = accuracy_score(y_tcga, y_pred)
        cm = confusion_matrix(y_tcga, y_pred)
        cm_df = pd.DataFrame(cm, index=["WT", "Mutated"], columns=["WT", "Mutated"])
        cm_df.to_csv(OUTPUT_DIR / f"transfer_cm_{name}.csv")

        print(f"  CCLE->TCGA  AUC = {auc:.4f}   F1 = {f1:.4f}   Acc = {acc:.4f}")
        rows.append({
            "model": name, "n_ccle_train": len(y_ccle), "n_tcga_test": len(y_tcga),
            "accuracy": acc, "f1": f1, "roc_auc": auc,
        })

    out = pd.DataFrame(rows)
    out.to_csv(OUTPUT_DIR / "transfer_metrics.csv", index=False)
    print("\n" + out.round(4).to_string(index=False))
    print(f"\nSaved {OUTPUT_DIR / 'transfer_metrics.csv'}")


if __name__ == "__main__":
    main()
