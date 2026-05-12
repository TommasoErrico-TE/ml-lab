"""
Task 2 — multi-class classification of TP53 mutation type.

Trains Logistic Regression, Random Forest, XGBoost on two parallel labels:
  - coding consequence: WT / Missense / Truncating  (3-class)
  - DNA-level:          WT / SNV / Insertion / Deletion  (4-class)

Rare classes (n < MIN_SAMPLES_PER_CLASS = 30) are dropped to avoid
training on too few samples (Inframe: 24, MNV: 14).

Evaluation: stratified 5-fold cross-validation.
Outputs (in output_tp53/):
  - task2_taxonomy_comparison.csv
  - task2_cm_<label>_<model>.csv and .png
"""

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

warnings.filterwarnings("ignore")

# =========================================================
# CONFIG
# =========================================================
MASTER_FILE = Path("output_tp53") / "datasets/master.csv"
LABELS_FILE = Path("output_tp53") / "datasets/task2_labels.csv"
OUTPUT_DIR = Path("output_tp53")

RANDOM_STATE = 42
N_SPLITS = 5
MIN_SAMPLES_PER_CLASS = 30
N_ESTIMATORS = 300

LABEL_CODING = "task2_mutation_type"
LABEL_DNA = "task2_dna_type"

NON_FEATURE_COLS = {
    "Unnamed: 0", "SequencingID", "ModelConditionID", "ModelID",
    "IsDefaultEntryForMC", "IsDefaultEntryForModel", "IsDefaultEntryForModel_bool",
    "TP53_expression",         # leakage
    "TP53_damaging_score",     # Task 1 label
    "task1_tp53_mutated",
    "task2_mutation_type", "task2_mutation_type_int",
    "task2_dna_type", "task2_dna_type_int",
    "n_tp53_mutations", "worst_variant_type",
    "worst_variant_info", "worst_protein_change",
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
# LOAD
# =========================================================
def load_data():
    print("Loading master + Task 2 labels...")
    master = pd.read_csv(MASTER_FILE)
    # data.py inserts a task2_mutation_type=NA placeholder in master; drop it before merging the real label.
    if "task2_mutation_type" in master.columns:
        master = master.drop(columns=["task2_mutation_type"])

    labels = pd.read_csv(LABELS_FILE)
    df = master.merge(
        labels[["ModelID", LABEL_CODING, LABEL_DNA]],
        on="ModelID",
        how="inner",
    )
    print(f"  Merged shape: {df.shape}")

    feature_cols = [c for c in df.columns if c not in NON_FEATURE_COLS]
    print(f"  Expression features: {len(feature_cols)}")
    return df, feature_cols


def filter_rare_classes(df, label_col, min_n):
    counts = df[label_col].value_counts()
    dropped = counts[counts < min_n]
    if not dropped.empty:
        n_before = len(df)
        df = df[~df[label_col].isin(dropped.index)].copy()
        print(f"  [filter] {label_col}: drop {dropped.to_dict()}  "
              f"({n_before} → {len(df)} sample)")
    classes_left = sorted(df[label_col].unique())
    print(f"  classes left: {classes_left}")
    return df


# =========================================================
# PIPELINES
# =========================================================
def build_pipelines(n_classes):
    pipelines = {}

    pipelines["LogReg"] = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(
            max_iter=3000,
            multi_class="multinomial",
            solver="lbfgs",
            class_weight="balanced",
            random_state=RANDOM_STATE,
        )),
    ])

    pipelines["RandomForest"] = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("clf", RandomForestClassifier(
            n_estimators=N_ESTIMATORS,
            random_state=RANDOM_STATE,
            n_jobs=-1,
            class_weight="balanced",
        )),
    ])

    if XGBOOST_AVAILABLE:
        pipelines["XGBoost"] = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("clf", XGBClassifier(
                n_estimators=N_ESTIMATORS,
                max_depth=6,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                objective="multi:softprob",
                num_class=n_classes,
                eval_metric="mlogloss",
                random_state=RANDOM_STATE,
                n_jobs=-1,
            )),
        ])
    else:
        print("  [WARN] XGBoost not available, skipping.")

    return pipelines


# =========================================================
# EVALUATE
# =========================================================
def plot_confusion_matrix(cm_df, title, filepath):
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm_df, annot=True, fmt="d", cmap="Blues", cbar=False,
                ax=ax, square=True)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)
    plt.tight_layout()
    plt.savefig(filepath, dpi=150)
    plt.close()


def evaluate_cv(pipelines, X, y_str, label_name):
    """Encode string labels as contiguous integers (required by XGBoost when rare classes have been dropped), run CV, save metrics and confusion matrix."""
    le = LabelEncoder()
    y_int = le.fit_transform(y_str)
    classes_str = list(le.classes_)
    n_classes = len(classes_str)

    cv = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    results = []

    for model_name, pipe in pipelines.items():
        print(f"  CV {model_name}...")
        y_pred_int = cross_val_predict(pipe, X, y_int, cv=cv, n_jobs=1)

        acc = accuracy_score(y_int, y_pred_int)
        macro_f1 = f1_score(y_int, y_pred_int, average="macro", zero_division=0)
        weighted_f1 = f1_score(y_int, y_pred_int, average="weighted", zero_division=0)
        f1_per_class = f1_score(
            y_int, y_pred_int, average=None,
            labels=range(n_classes), zero_division=0,
        )
        cm = confusion_matrix(y_int, y_pred_int, labels=range(n_classes))
        cm_df = pd.DataFrame(cm, index=classes_str, columns=classes_str)

        row = {
            "label": label_name,
            "model": model_name,
            "n_classes": n_classes,
            "n_samples": len(y_int),
            "accuracy": acc,
            "macro_f1": macro_f1,
            "weighted_f1": weighted_f1,
        }
        for cls, score in zip(classes_str, f1_per_class):
            row[f"f1_{cls}"] = score
        results.append(row)

        cm_path_csv = OUTPUT_DIR / f"task2/cm_{label_name}_{model_name}.csv"
        cm_path_png = OUTPUT_DIR / f"task2/cm_{label_name}_{model_name}.png"
        cm_df.to_csv(cm_path_csv)
        plot_confusion_matrix(
            cm_df,
            title=f"Task 2 - {label_name} - {model_name} (5-fold CV)",
            filepath=cm_path_png,
        )

        print(f"    accuracy={acc:.4f}  macro_f1={macro_f1:.4f}  "
              f"weighted_f1={weighted_f1:.4f}")
        for cls, score in zip(classes_str, f1_per_class):
            print(f"      F1[{cls}] = {score:.4f}")

    return pd.DataFrame(results)


# =========================================================
# MAIN
# =========================================================
def main():
    OUTPUT_DIR.mkdir(exist_ok=True)
    (OUTPUT_DIR / 'task2').mkdir(parents=True, exist_ok=True)

    df, feature_cols = load_data()

    print()
    print("Class distribution (before rare-class drop):")
    print(f"  {LABEL_CODING}: {df[LABEL_CODING].value_counts().to_dict()}")
    print(f"  {LABEL_DNA}:    {df[LABEL_DNA].value_counts().to_dict()}")

    all_results = []

    for label_name, label_col in [("coding", LABEL_CODING),
                                   ("dna", LABEL_DNA)]:
        print("\n" + "=" * 60)
        print(f"LABEL: {label_name}  ({label_col})")
        print("=" * 60)

        df_filt = filter_rare_classes(df, label_col, MIN_SAMPLES_PER_CLASS)
        X = df_filt[feature_cols].values
        y = df_filt[label_col].values

        pipelines = build_pipelines(n_classes=df_filt[label_col].nunique())
        results_df = evaluate_cv(pipelines, X, y, label_name)
        all_results.append(results_df)

    # =====================================================
    # COMBINE & SAVE
    # =====================================================
    all_df = pd.concat(all_results, ignore_index=True)
    out_csv = OUTPUT_DIR / "task2/taxonomy_comparison.csv"
    all_df.to_csv(out_csv, index=False)

    print("\n" + "=" * 60)
    print("CONFRONTO TASSONOMIE (5-fold CV)")
    print("=" * 60)
    summary_cols = ["label", "model", "n_classes", "n_samples",
                    "accuracy", "macro_f1", "weighted_f1"]
    print(all_df[summary_cols].round(4).to_string(index=False))

    print("\nFiles saved:")
    print(f"  - {out_csv}")
    print(f"  - confusion matrices in {OUTPUT_DIR}/task2_cm_*.{{csv,png}}")


if __name__ == "__main__":
    main()
