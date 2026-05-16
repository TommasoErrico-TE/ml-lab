"""
Task 1 — hyperparameter search esteso su LogReg, RandomForest, XGBoost.

Pipeline:
  1) 60/20/20 train/val/test stratificato (seed fisso)
  2) Top-K geni per varianza calcolati sul train (default K=2000)
  3) RandomizedSearchCV con 5-fold CV interna sul train, scoring=roc_auc
  4) Riallenamento del best model su train, eval su val e test

Output (in output_tp53/task1/):
  - hp_search_best.csv     riga per modello: best_params, cv_auc, val_auc, test_auc
  - hp_search_<model>.csv  dettaglio per modello (top 20 combinazioni)

CLI:
  --n-iter   numero campionamenti RandomizedSearch (default 30)
  --top-k    geni per varianza usati come feature pool (default 2000)
  --jobs     n_jobs per la search e per i modelli (default -1)
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

DATA_FILE = Path("output_tp53") / "datasets/master.csv"
OUTPUT_DIR = Path("output_tp53") / "task1"
RANDOM_STATE = 42
TARGET_COL = "task1_tp53_mutated"

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
# DATA
# =========================================================
def load_xy():
    df = pd.read_csv(DATA_FILE)
    feature_cols = [c for c in df.columns if c not in NON_FEATURE_COLS]
    X = df[feature_cols].copy()
    y = df[TARGET_COL].astype(int).to_numpy()
    return X, y, feature_cols


def three_way_split(X, y, seed=RANDOM_STATE):
    X_tr, X_rest, y_tr, y_rest = train_test_split(
        X, y, train_size=0.6, random_state=seed, stratify=y,
    )
    X_val, X_te, y_val, y_te = train_test_split(
        X_rest, y_rest, train_size=0.5, random_state=seed, stratify=y_rest,
    )
    return X_tr, X_val, X_te, y_tr, y_val, y_te


def select_top_var(X_train, X_val, X_test, k):
    """Seleziona top-K geni per varianza calcolata SOLO sul train."""
    var = X_train.var(axis=0)
    top_idx = var.values.argsort()[::-1][:k]
    cols = X_train.columns[top_idx]
    return X_train[cols], X_val[cols], X_test[cols], list(cols)


# =========================================================
# SEARCH SPACES
# =========================================================
def search_space_logreg():
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


def search_space_rf(n_jobs):
    pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("clf", RandomForestClassifier(
            random_state=RANDOM_STATE, n_jobs=n_jobs, class_weight="balanced",
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


def search_space_xgb(n_jobs):
    pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("clf", XGBClassifier(
            objective="binary:logistic", eval_metric="logloss",
            random_state=RANDOM_STATE, n_jobs=n_jobs,
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


# =========================================================
# RUN
# =========================================================
def run_one(name, pipe, params, X_tr, y_tr, X_val, y_val, X_te, y_te, n_iter, jobs):
    print("\n" + "=" * 60)
    print(f"HP SEARCH: {name}")
    print("=" * 60)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    search = RandomizedSearchCV(
        pipe, params,
        n_iter=n_iter, cv=cv,
        scoring="roc_auc",
        random_state=RANDOM_STATE,
        n_jobs=jobs,
        refit=True,
        return_train_score=True,
        verbose=1,
    )
    t0 = time.time()
    search.fit(X_tr, y_tr)
    dt = time.time() - t0
    print(f"\n  best CV AUC = {search.best_score_:.4f}")
    print(f"  best params = {search.best_params_}")
    print(f"  search time = {dt:.1f}s")

    best = search.best_estimator_
    val_auc = roc_auc_score(y_val, best.predict_proba(X_val)[:, 1])
    te_auc = roc_auc_score(y_te, best.predict_proba(X_te)[:, 1])
    val_f1 = f1_score(y_val, best.predict(X_val), zero_division=0)
    te_f1 = f1_score(y_te, best.predict(X_te), zero_division=0)
    val_acc = accuracy_score(y_val, best.predict(X_val))
    te_acc = accuracy_score(y_te, best.predict(X_te))
    print(f"  val AUC = {val_auc:.4f}   test AUC = {te_auc:.4f}")

    cvres = pd.DataFrame(search.cv_results_).sort_values("mean_test_score", ascending=False)
    keep = ["mean_test_score", "std_test_score", "mean_train_score", "params"]
    cvres = cvres[keep].head(20).reset_index(drop=True)
    cvres["params"] = cvres["params"].apply(lambda d: json.dumps(d, default=str))
    cvres.to_csv(OUTPUT_DIR / f"hp_search_{name}.csv", index=False)

    return {
        "model": name,
        "n_iter": n_iter,
        "search_time_s": round(dt, 1),
        "cv_auc": search.best_score_,
        "val_auc": val_auc,
        "val_f1": val_f1,
        "val_acc": val_acc,
        "test_auc": te_auc,
        "test_f1": te_f1,
        "test_acc": te_acc,
        "best_params": json.dumps(search.best_params_, default=str),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-iter", type=int, default=30)
    ap.add_argument("--top-k", type=int, default=2000)
    ap.add_argument("--jobs", type=int, default=-1)
    args = ap.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print("Loading dataset...")
    X, y, feature_cols = load_xy()
    print(f"X: {X.shape}   y counts: {dict(pd.Series(y).value_counts())}")

    X_tr, X_val, X_te, y_tr, y_val, y_te = three_way_split(X, y)
    print(f"Split  train: {X_tr.shape}   val: {X_val.shape}   test: {X_te.shape}")

    print(f"\nSelecting top-{args.top_k} genes by variance (from train)...")
    X_tr, X_val, X_te, kept = select_top_var(X_tr, X_val, X_te, args.top_k)
    print(f"After variance filter: {X_tr.shape[1]} features")

    rows = []

    def save_partial():
        if rows:
            pd.DataFrame(rows).to_csv(OUTPUT_DIR / "hp_search_best.csv", index=False)

    pipe, params = search_space_logreg()
    rows.append(run_one("LogReg", pipe, params, X_tr, y_tr, X_val, y_val, X_te, y_te, args.n_iter, args.jobs))
    save_partial()

    pipe, params = search_space_rf(args.jobs)
    rows.append(run_one("RandomForest", pipe, params, X_tr, y_tr, X_val, y_val, X_te, y_te, args.n_iter, args.jobs))
    save_partial()

    if XGBOOST_AVAILABLE:
        pipe, params = search_space_xgb(args.jobs)
        rows.append(run_one("XGBoost", pipe, params, X_tr, y_tr, X_val, y_val, X_te, y_te, args.n_iter, args.jobs))
        save_partial()

    out = pd.DataFrame(rows)
    out.to_csv(OUTPUT_DIR / "hp_search_best.csv", index=False)
    print("\n" + "=" * 60)
    print("HP SEARCH SUMMARY")
    print("=" * 60)
    print(out[["model", "cv_auc", "val_auc", "test_auc", "search_time_s"]].round(4).to_string(index=False))
    print(f"\nSaved {OUTPUT_DIR / 'hp_search_best.csv'}")


if __name__ == "__main__":
    main()
