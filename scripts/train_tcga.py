"""
Train + valuta i modelli direttamente su TCGA, con la stessa pipeline di CCLE:
  - 60/20/20 train/val/test stratificato
  - top-2000 geni per varianza (calcolata su train)
  - RandomizedSearchCV (30 iter, 5-fold) per ogni modello
  - eval finale su val e test

Output (in output_tp53/tcga/):
  - tcga_hp_search_best.csv     come per CCLE, ma su TCGA
  - tcga_baseline_split.csv     metriche dei tre modelli con HP di default

CLI:
  --n-iter   (default 30)
  --top-k    (default 2000)
  --jobs     (default -1)
"""

import argparse
import json
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import loguniform, uniform, randint

from sklearn.model_selection import (
    train_test_split, StratifiedKFold, RandomizedSearchCV,
)
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score

warnings.filterwarnings("ignore")

TCGA_MASTER = Path("output_tp53/datasets/tcga_master.parquet")
OUTPUT_DIR = Path("output_tp53/tcga")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 42
TARGET_COL = "task1_tp53_mutated"

NON_FEATURE_TCGA = {
    "sample", "task1_tp53_mutated", "task2_mutation_type",
    "worst_effect", "worst_protein_change", "tcga_cancer_type",
}

try:
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False


def load_xy():
    df = pd.read_parquet(TCGA_MASTER)
    feat = [c for c in df.columns if c not in NON_FEATURE_TCGA]
    X = df[feat].copy()
    y = df[TARGET_COL].astype(int).to_numpy()
    return X, y, feat


def three_way_split(X, y, seed=RANDOM_STATE):
    X_tr, X_rest, y_tr, y_rest = train_test_split(
        X, y, train_size=0.6, random_state=seed, stratify=y,
    )
    X_val, X_te, y_val, y_te = train_test_split(
        X_rest, y_rest, train_size=0.5, random_state=seed, stratify=y_rest,
    )
    return X_tr, X_val, X_te, y_tr, y_val, y_te


def select_top_var(X_train, X_val, X_test, k):
    var = X_train.var(axis=0)
    top = var.values.argsort()[::-1][:k]
    cols = X_train.columns[top]
    return X_train[cols], X_val[cols], X_test[cols], list(cols)


def space_logreg():
    pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(
            max_iter=3000, random_state=RANDOM_STATE, class_weight="balanced",
        )),
    ])
    params = {
        "clf__C": loguniform(1e-3, 1e2),
        "clf__penalty": ["l1", "l2"],
        "clf__solver": ["liblinear"],
    }
    return pipe, params


def space_rf(jobs):
    pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("clf", RandomForestClassifier(
            random_state=RANDOM_STATE, n_jobs=jobs, class_weight="balanced",
        )),
    ])
    params = {
        "clf__n_estimators": randint(200, 800),
        "clf__max_depth": [None, 5, 10, 20, 40],
        "clf__min_samples_split": randint(2, 10),
        "clf__min_samples_leaf": randint(1, 5),
        "clf__max_features": ["sqrt", "log2", 0.1, 0.3],
    }
    return pipe, params


def space_xgb(jobs):
    pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("clf", XGBClassifier(
            objective="binary:logistic", eval_metric="logloss",
            random_state=RANDOM_STATE, n_jobs=jobs,
        )),
    ])
    params = {
        "clf__n_estimators": randint(200, 800),
        "clf__max_depth": randint(3, 10),
        "clf__learning_rate": loguniform(1e-2, 3e-1),
        "clf__subsample": uniform(0.6, 0.4),
        "clf__colsample_bytree": uniform(0.6, 0.4),
        "clf__reg_alpha": loguniform(1e-3, 1e1),
        "clf__reg_lambda": loguniform(1e-3, 1e1),
    }
    return pipe, params


def run_one(name, pipe, params, splits, n_iter, jobs):
    X_tr, X_val, X_te, y_tr, y_val, y_te = splits
    print("\n" + "=" * 60)
    print(f"HP SEARCH (TCGA): {name}")
    print("=" * 60)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    search = RandomizedSearchCV(
        pipe, params, n_iter=n_iter, cv=cv, scoring="roc_auc",
        random_state=RANDOM_STATE, n_jobs=jobs, refit=True, verbose=1,
    )
    t0 = time.time()
    search.fit(X_tr, y_tr)
    dt = time.time() - t0
    best = search.best_estimator_

    val_auc = roc_auc_score(y_val, best.predict_proba(X_val)[:, 1])
    te_auc = roc_auc_score(y_te, best.predict_proba(X_te)[:, 1])
    print(f"  best CV AUC = {search.best_score_:.4f}")
    print(f"  val AUC = {val_auc:.4f}   test AUC = {te_auc:.4f}")
    return {
        "model": name, "n_iter": n_iter, "search_time_s": round(dt, 1),
        "cv_auc": search.best_score_, "val_auc": val_auc,
        "val_f1": f1_score(y_val, best.predict(X_val), zero_division=0),
        "val_acc": accuracy_score(y_val, best.predict(X_val)),
        "test_auc": te_auc,
        "test_f1": f1_score(y_te, best.predict(X_te), zero_division=0),
        "test_acc": accuracy_score(y_te, best.predict(X_te)),
        "best_params": json.dumps(search.best_params_, default=str),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-iter", type=int, default=30)
    ap.add_argument("--top-k", type=int, default=2000)
    ap.add_argument("--jobs", type=int, default=-1)
    args = ap.parse_args()

    print("Loading TCGA master...")
    X, y, feat = load_xy()
    print(f"X: {X.shape}  y counts: {dict(pd.Series(y).value_counts())}")

    X_tr, X_val, X_te, y_tr, y_val, y_te = three_way_split(X, y)
    print(f"Split  train: {X_tr.shape}  val: {X_val.shape}  test: {X_te.shape}")

    print(f"\nSelecting top-{args.top_k} genes by variance (train)...")
    X_tr, X_val, X_te, _ = select_top_var(X_tr, X_val, X_te, args.top_k)
    print(f"After variance filter: {X_tr.shape[1]} features")

    splits = (X_tr, X_val, X_te, y_tr, y_val, y_te)
    rows = []
    pipe, params = space_logreg()
    rows.append(run_one("LogReg", pipe, params, splits, args.n_iter, args.jobs))
    pipe, params = space_rf(args.jobs)
    rows.append(run_one("RandomForest", pipe, params, splits, args.n_iter, args.jobs))
    if XGBOOST_AVAILABLE:
        pipe, params = space_xgb(args.jobs)
        rows.append(run_one("XGBoost", pipe, params, splits, args.n_iter, args.jobs))

    out = pd.DataFrame(rows)
    out.to_csv(OUTPUT_DIR / "tcga_hp_search_best.csv", index=False)
    print("\n" + out[["model", "cv_auc", "val_auc", "test_auc"]].round(4).to_string(index=False))
    print(f"\nSaved {OUTPUT_DIR / 'tcga_hp_search_best.csv'}")


if __name__ == "__main__":
    main()
