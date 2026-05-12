import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

warnings.filterwarnings("ignore")

# =========================================================
# CONFIG
# =========================================================
DATA_FILE = Path("output_tp53") / "datasets/master.csv"
RANDOM_STATE = 42
TEST_SIZE = 0.2

TARGET_COL = "task1_tp53_mutated"

NON_FEATURE_COLS = {
    "SequencingID",
    "ModelConditionID",
    "ModelID",
    "IsDefaultEntryForMC",
    "IsDefaultEntryForModel",
    "IsDefaultEntryForModel_bool",
    "TP53_expression",         # da escludere per evitare leakage
    "TP53_damaging_score",     # da escludere per evitare leakage
    "task1_tp53_mutated",
    "task2_mutation_type",
    "CellLineName",
    "CCLEName",
    "OncotreeLineage",
    "OncotreePrimaryDisease",
    "OncotreeSubtype",
    "Sex",
    "AgeCategory",
    "PrimaryOrMetastasis",
}

# =========================================================
# OPTIONAL XGBOOST
# =========================================================
try:
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False


# =========================================================
# HELPERS
# =========================================================
def load_dataset(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Dataset non trovato: {path}")
    df = pd.read_csv(path)
    return df


def build_features_and_target(df: pd.DataFrame):
    if TARGET_COL not in df.columns:
        raise ValueError(f"Column target not found: {TARGET_COL}")

    feature_cols = [c for c in df.columns if c not in NON_FEATURE_COLS]
    X = df[feature_cols].copy()
    y = df[TARGET_COL].astype(int).copy()
    return X, y, feature_cols


def evaluate_model(model, X_train, X_test, y_train, y_test, model_name: str):
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    if hasattr(model, "predict_proba"):
        y_proba = model.predict_proba(X_test)[:, 1]
    else:
        y_proba = None

    results = {
        "model": model_name,
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1": f1_score(y_test, y_pred, zero_division=0),
    }

    if y_proba is not None:
        results["roc_auc"] = roc_auc_score(y_test, y_proba)
    else:
        results["roc_auc"] = np.nan

    cm = confusion_matrix(y_test, y_pred)

    return results, y_pred, y_proba, cm


def print_model_report(model_name, y_test, y_pred, cm):
    print(f"\n{'=' * 60}")
    print(f"MODEL: {model_name}")
    print(f"{'=' * 60}")

    print("\nConfusion Matrix:")
    print(cm)

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, digits=4, zero_division=0))


# =========================================================
# MAIN
# =========================================================
def main():
    print("Loading dataset...")
    df = load_dataset(DATA_FILE)

    print(f"Shape dataset: {df.shape}")

    print("\nTarget distribution:")
    print(df[TARGET_COL].value_counts(dropna=False))

    X, y, feature_cols = build_features_and_target(df)

    print(f"\nNumber of features used: {len(feature_cols)}")
    print(f"First 10 features: {feature_cols[:10]}")

    # split stratificato per mantenere le proporzioni delle classi
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    print(f"\nTrain shape: {X_train.shape}")
    print(f"Test shape:  {X_test.shape}")

    # =====================================================
    # MODELS
    # =====================================================
    models = {}

    # 1) Logistic Regression
    models["Logistic Regression"] = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(
            max_iter=2000,
            random_state=RANDOM_STATE,
            class_weight="balanced"
        ))
    ])

    # 2) Random Forest
    models["Random Forest"] = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("clf", RandomForestClassifier(
            n_estimators=300,
            max_depth=None,
            min_samples_split=2,
            min_samples_leaf=1,
            random_state=RANDOM_STATE,
            n_jobs=-1,
            class_weight="balanced"
        ))
    ])

    # 3) XGBoost
    if XGBOOST_AVAILABLE:
        models["XGBoost"] = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("clf", XGBClassifier(
                n_estimators=300,
                max_depth=6,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                objective="binary:logistic",
                eval_metric="logloss",
                random_state=RANDOM_STATE,
                n_jobs=-1
            ))
        ])
    else:
        print("\n[INFO] XGBoost non installato. Lo salto.")
        print("Per installarlo: pip install xgboost")

    # =====================================================
    # TRAIN + EVAL
    # =====================================================
    all_results = []
    fitted_models = {}

    for model_name, model in models.items():
        print(f"\nTraining {model_name}...")
        results, y_pred, y_proba, cm = evaluate_model(
            model, X_train, X_test, y_train, y_test, model_name
        )
        all_results.append(results)
        fitted_models[model_name] = model
        print_model_report(model_name, y_test, y_pred, cm)

    # =====================================================
    # RESULTS TABLE
    # =====================================================
    results_df = pd.DataFrame(all_results)
    results_df = results_df.sort_values(by="roc_auc", ascending=False)

    print(f"\n{'=' * 60}")
    print("FINAL MODEL COMPARISON")
    print(f"{'=' * 60}")
    print(results_df.round(4))

    # salva risultati
    output_dir = Path("output_tp53")
    output_dir.mkdir(exist_ok=True)
    (output_dir / 'task1').mkdir(parents=True, exist_ok=True)

    results_df.to_csv(output_dir / "task1/model_comparison.csv", index=False)

    print("\nResults saved to:")
    print(output_dir / "task1/model_comparison.csv")

    # =====================================================
    # FEATURE IMPORTANCE
    # =====================================================
    # Random Forest
    if "Random Forest" in fitted_models:
        rf_clf = fitted_models["Random Forest"].named_steps["clf"]
        rf_importances = pd.DataFrame({
            "feature": feature_cols,
            "importance": rf_clf.feature_importances_
        }).sort_values("importance", ascending=False)

        rf_importances.to_csv(output_dir / "task1/rf_feature_importance.csv", index=False)

        print("\nTop 20 features - Random Forest:")
        print(rf_importances.head(20).to_string(index=False))

    # XGBoost
    if "XGBoost" in fitted_models:
        xgb_clf = fitted_models["XGBoost"].named_steps["clf"]
        xgb_importances = pd.DataFrame({
            "feature": feature_cols,
            "importance": xgb_clf.feature_importances_
        }).sort_values("importance", ascending=False)

        xgb_importances.to_csv(output_dir / "task1/xgb_feature_importance.csv", index=False)

        print("\nTop 20 features - XGBoost:")
        print(xgb_importances.head(20).to_string(index=False))


if __name__ == "__main__":
    main()