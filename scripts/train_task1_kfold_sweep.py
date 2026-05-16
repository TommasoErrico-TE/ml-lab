"""
Task 1 — k-fold sweep per giustificare la scelta di 5-fold.

Confronta StratifiedKFold con k=3, 5, 10. Per ogni k usa 5 seed diversi
per stimare la varianza della media. Reporta AUC media, std e tempo.

Output: output_tp53/task1/kfold_sweep.csv
"""

import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

warnings.filterwarnings("ignore")

DATA_FILE = Path("output_tp53") / "datasets/master.csv"
OUTPUT_DIR = Path("output_tp53") / "task1"
RANDOM_STATE = 42
TARGET_COL = "task1_tp53_mutated"

K_VALUES = [3, 5, 10]
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


def build_pipeline():
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(
            max_iter=2000, random_state=RANDOM_STATE, class_weight="balanced",
        )),
    ])


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(DATA_FILE)
    feature_cols = [c for c in df.columns if c not in NON_FEATURE_COLS]
    X = df[feature_cols].values
    y = df[TARGET_COL].astype(int).to_numpy()
    print(f"X: {X.shape}   y counts: {dict(pd.Series(y).value_counts())}")

    rows = []
    for k in K_VALUES:
        per_seed_means = []
        per_seed_times = []
        all_fold_aucs = []
        for seed in range(N_SEEDS):
            cv = StratifiedKFold(n_splits=k, shuffle=True, random_state=seed)
            model = build_pipeline()
            t0 = time.time()
            aucs = cross_val_score(model, X, y, cv=cv, scoring="roc_auc", n_jobs=1)
            dt = time.time() - t0
            per_seed_means.append(np.mean(aucs))
            per_seed_times.append(dt)
            all_fold_aucs.extend(aucs.tolist())
            print(f"  k={k:2d}  seed={seed}  mean_auc={np.mean(aucs):.4f}  "
                  f"std_per_fold={np.std(aucs):.4f}  time={dt:.1f}s")
        rows.append({
            "k": k,
            "mean_auc_across_seeds": np.mean(per_seed_means),
            "std_auc_across_seeds": np.std(per_seed_means),
            "mean_std_per_fold": np.mean([np.std(all_fold_aucs[i*k:(i+1)*k]) for i in range(N_SEEDS)]),
            "mean_time_s": np.mean(per_seed_times),
            "total_time_s": np.sum(per_seed_times),
        })

    out = pd.DataFrame(rows)
    out.to_csv(OUTPUT_DIR / "kfold_sweep.csv", index=False)
    print("\n" + out.round(4).to_string(index=False))
    print(f"\nSaved {OUTPUT_DIR / 'kfold_sweep.csv'}")


if __name__ == "__main__":
    main()
