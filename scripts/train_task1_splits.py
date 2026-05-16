"""
Task 1 — train / val / test split design.

Sostituisce il vecchio 80/20 train/val con un 3-way split stratificato.
Inoltre confronta diverse proporzioni (50/25/25, 60/20/20, 70/15/15, 80/10/10)
per giustificare la scelta dei numeri.

Outputs (in output_tp53/task1/):
  - baseline_split.csv      metriche dei 3 modelli sul 60/20/20 (val + test)
  - split_ratios.csv        ratio experiment (LogReg, 5 seed per ratio)
"""

import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score,
)

warnings.filterwarnings("ignore")

# =========================================================
# CONFIG
# =========================================================
DATA_FILE = Path("output_tp53") / "datasets/master.csv"
OUTPUT_DIR = Path("output_tp53") / "task1"
RANDOM_STATE = 42
TARGET_COL = "task1_tp53_mutated"

# Splitting principale: 60/20/20
TRAIN_RATIO = 0.6
VAL_RATIO = 0.2
TEST_RATIO = 0.2

# Esperimento sui ratio (train, val, test)
RATIOS = [
    (0.5, 0.25, 0.25),
    (0.6, 0.20, 0.20),
    (0.7, 0.15, 0.15),
    (0.8, 0.10, 0.10),
]
N_SEEDS = 5

NON_FEATURE_COLS = {
    "Unnamed: 0", "SequencingID", "ModelConditionID", "ModelID",
    "IsDefaultEntryForMC", "IsDefaultEntryForModel", "IsDefaultEntryForModel_bool",
    "TP53_expression", "TP53_damaging_score",
    "task1_tp53_mutated", "task2_mutation_type",
    "CellLineName", "CCLEName",
    "OncotreeLineage", "OncotreePrimaryDisease", "OncotreeSubtype",
    "Sex", "AgeCategory", "PrimaryOrMetastasis",
}

try:
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False


# =========================================================
# HELPERS
# =========================================================
def load_xy():
    df = pd.read_csv(DATA_FILE)
    feature_cols = [c for c in df.columns if c not in NON_FEATURE_COLS]
    X = df[feature_cols].copy()
    y = df[TARGET_COL].astype(int).copy()
    return X, y, feature_cols


def three_way_split(X, y, train_r, val_r, test_r, seed):
    """Stratified split in tre parti, mantenendo le proporzioni di classe."""
    assert abs(train_r + val_r + test_r - 1.0) < 1e-9
    # primo split: train vs rest (val+test)
    X_tr, X_rest, y_tr, y_rest = train_test_split(
        X, y,
        train_size=train_r,
        random_state=seed,
        stratify=y,
    )
    # secondo split: val vs test all'interno del rest
    val_share = val_r / (val_r + test_r)
    X_val, X_te, y_val, y_te = train_test_split(
        X_rest, y_rest,
        train_size=val_share,
        random_state=seed,
        stratify=y_rest,
    )
    return X_tr, X_val, X_te, y_tr, y_val, y_te


def build_models():
    models = {}
    models["Logistic Regression"] = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(
            max_iter=2000, random_state=RANDOM_STATE, class_weight="balanced",
        )),
    ])
    models["Random Forest"] = Pipeline([
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


def eval_split(model, X_eval, y_eval):
    y_pred = model.predict(X_eval)
    y_proba = model.predict_proba(X_eval)[:, 1] if hasattr(model, "predict_proba") else None
    return {
        "accuracy":  accuracy_score(y_eval, y_pred),
        "precision": precision_score(y_eval, y_pred, zero_division=0),
        "recall":    recall_score(y_eval, y_pred, zero_division=0),
        "f1":        f1_score(y_eval, y_pred, zero_division=0),
        "roc_auc":   roc_auc_score(y_eval, y_proba) if y_proba is not None else np.nan,
    }


# =========================================================
# MAIN: baseline 60/20/20
# =========================================================
def run_baseline(X, y):
    print("\n" + "=" * 60)
    print(f"BASELINE 3-way split: {int(TRAIN_RATIO*100)}/{int(VAL_RATIO*100)}/{int(TEST_RATIO*100)}")
    print("=" * 60)
    X_tr, X_val, X_te, y_tr, y_val, y_te = three_way_split(
        X, y, TRAIN_RATIO, VAL_RATIO, TEST_RATIO, seed=RANDOM_STATE,
    )
    print(f"Train: {X_tr.shape}, Val: {X_val.shape}, Test: {X_te.shape}")

    rows = []
    for name, model in build_models().items():
        print(f"\nTraining {name}...")
        t0 = time.time()
        model.fit(X_tr, y_tr)
        dt = time.time() - t0
        m_val = eval_split(model, X_val, y_val)
        m_te = eval_split(model, X_te, y_te)
        rows.append({
            "model": name,
            "fit_time_s": round(dt, 1),
            **{f"val_{k}": v for k, v in m_val.items()},
            **{f"test_{k}": v for k, v in m_te.items()},
        })
        print(f"  val AUC = {m_val['roc_auc']:.4f}   test AUC = {m_te['roc_auc']:.4f}")
    return pd.DataFrame(rows)


# =========================================================
# RATIO EXPERIMENT
# =========================================================
def run_ratio_experiment(X, y):
    print("\n" + "=" * 60)
    print("RATIO EXPERIMENT (LogReg, 5 seed per ratio)")
    print("=" * 60)
    rows = []
    for tr_r, va_r, te_r in RATIOS:
        for seed in range(N_SEEDS):
            X_tr, X_val, X_te, y_tr, y_val, y_te = three_way_split(X, y, tr_r, va_r, te_r, seed)
            model = Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                ("clf", LogisticRegression(
                    max_iter=2000, random_state=RANDOM_STATE, class_weight="balanced",
                )),
            ])
            model.fit(X_tr, y_tr)
            val_auc = roc_auc_score(y_val, model.predict_proba(X_val)[:, 1])
            te_auc = roc_auc_score(y_te, model.predict_proba(X_te)[:, 1])
            rows.append({
                "train_pct": int(tr_r * 100),
                "val_pct": int(va_r * 100),
                "test_pct": int(te_r * 100),
                "seed": seed,
                "n_train": len(y_tr),
                "n_val": len(y_val),
                "n_test": len(y_te),
                "val_auc": val_auc,
                "test_auc": te_auc,
            })
        last = rows[-N_SEEDS:]
        m_val = np.mean([r["val_auc"] for r in last])
        s_val = np.std([r["val_auc"] for r in last])
        m_te = np.mean([r["test_auc"] for r in last])
        s_te = np.std([r["test_auc"] for r in last])
        print(f"  {int(tr_r*100)}/{int(va_r*100)}/{int(te_r*100)}  "
              f"val={m_val:.4f}+-{s_val:.4f}  test={m_te:.4f}+-{s_te:.4f}")
    return pd.DataFrame(rows)


# =========================================================
# MAIN
# =========================================================
def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print("Loading dataset...")
    X, y, feature_cols = load_xy()
    print(f"X shape: {X.shape}   y: {dict(y.value_counts())}")

    baseline = run_baseline(X, y)
    baseline.to_csv(OUTPUT_DIR / "baseline_split.csv", index=False)
    print(f"\nSaved {OUTPUT_DIR / 'baseline_split.csv'}")
    print(baseline.round(4).to_string(index=False))

    ratios = run_ratio_experiment(X, y)
    ratios.to_csv(OUTPUT_DIR / "split_ratios.csv", index=False)
    print(f"\nSaved {OUTPUT_DIR / 'split_ratios.csv'}")


if __name__ == "__main__":
    main()
