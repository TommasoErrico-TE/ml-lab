"""Helper: carica i best hyperparameter dalla HP search e instanzia i modelli."""
import json
from pathlib import Path

import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

try:
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False


def load_best_hp(csv_path):
    """Legge hp_search_best.csv e restituisce dict {model_name: params_dict}."""
    import pandas as pd
    if not Path(csv_path).exists():
        return {}
    df = pd.read_csv(csv_path)
    out = {}
    for _, row in df.iterrows():
        params = json.loads(row["best_params"])
        # rimuove il prefisso 'clf__' (era usato nel pipeline della search)
        params = {k.replace("clf__", ""): v for k, v in params.items()}
        out[row["model"]] = params
    return out


def make_tuned_logreg(best_hp, random_state=42, default_C=1.0):
    """Restituisce un Pipeline con il LogReg tunato (o default se HP search manca)."""
    p = best_hp.get("LogReg", {})
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(
            C=p.get("C", default_C),
            penalty=p.get("penalty", "l2"),
            solver=p.get("solver", "lbfgs" if p.get("penalty", "l2") == "l2" else "liblinear"),
            max_iter=3000,
            random_state=random_state,
            class_weight="balanced",
        )),
    ])


def make_tuned_rf(best_hp, random_state=42, n_jobs=-1):
    """RandomForest tunato (o default)."""
    p = best_hp.get("RandomForest", {})
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("clf", RandomForestClassifier(
            n_estimators=int(p.get("n_estimators", 300)),
            max_depth=p.get("max_depth", None),
            min_samples_split=int(p.get("min_samples_split", 2)),
            min_samples_leaf=int(p.get("min_samples_leaf", 1)),
            max_features=p.get("max_features", "sqrt"),
            random_state=random_state,
            n_jobs=n_jobs,
            class_weight="balanced",
        )),
    ])


def make_tuned_xgb(best_hp, random_state=42, n_jobs=-1, objective="binary:logistic"):
    """XGBoost tunato (o default)."""
    if not XGBOOST_AVAILABLE:
        return None
    p = best_hp.get("XGBoost", {})
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("clf", XGBClassifier(
            n_estimators=int(p.get("n_estimators", 300)),
            max_depth=int(p.get("max_depth", 6)),
            learning_rate=p.get("learning_rate", 0.05),
            subsample=p.get("subsample", 0.8),
            colsample_bytree=p.get("colsample_bytree", 0.8),
            reg_alpha=p.get("reg_alpha", 0.0),
            reg_lambda=p.get("reg_lambda", 1.0),
            objective=objective,
            eval_metric="logloss",
            random_state=random_state,
            n_jobs=n_jobs,
        )),
    ])
