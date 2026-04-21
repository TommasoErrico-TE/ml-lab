"""
Task 1 con tre scenari di cross-validation, per capire quanto del
segnale in `train_task1.py` è davvero p53 e quanto è confounding di tessuto.

Scenari:
  A) Standard StratifiedKFold (5-fold). Ogni fold vede tutti i tessuti.
     Riproduce la situazione di `train_task1.py` ma in CV → numero piu'
     stabile (media +- std) rispetto a un singolo train/test split.

  B) GroupKFold per tessuto (5-fold). I tessuti nel test NON sono nel train.
     Testa la vera generalizzazione a tessuti mai visti.
     Se AUC crolla -> il modello stava usando il tessuto come proxy di p53.

  C) Tissue-only baseline. Feature: solo le dummy del tessuto, nessun gene.
     Mostra quanto AUC si ottiene "gratis" dal tessuto da solo.
     Se e' gia' alto, gran parte del segnale in A non e' p53.

Output: tabella di confronto dei tre scenari + breakdown per tessuto
nel caso piu' severo (LeaveOneGroupOut).
"""
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.model_selection import (
    StratifiedKFold,
    GroupKFold,
    LeaveOneGroupOut,
    cross_val_score,
    cross_val_predict,
)
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

warnings.filterwarnings("ignore")

# =========================================================
# CONFIG
# =========================================================
DATA_FILE = Path("output_tp53") / "tp53_master_dataset.csv"
OUTPUT_DIR = Path("output_tp53")
RANDOM_STATE = 42
N_SPLITS = 5

TARGET_COL = "task1_tp53_mutated"
TISSUE_COL = "OncotreeLineage"

NON_FEATURE_COLS = {
    "SequencingID",
    "ModelConditionID",
    "ModelID",
    "IsDefaultEntryForMC",
    "IsDefaultEntryForModel",
    "IsDefaultEntryForModel_bool",
    "TP53_expression",
    "TP53_damaging_score",
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
# HELPERS
# =========================================================
def load_data():
    df = pd.read_csv(DATA_FILE)
    print(f"Dataset shape: {df.shape}")

    before = len(df)
    df = df.dropna(subset=[TISSUE_COL, TARGET_COL]).copy()
    print(f"Righe dopo drop missing su {TISSUE_COL}/{TARGET_COL}: "
          f"{len(df)} (scartate {before - len(df)})")

    feature_cols = [c for c in df.columns if c not in NON_FEATURE_COLS]
    X = df[feature_cols].copy()
    y = df[TARGET_COL].astype(int).to_numpy()
    groups = df[TISSUE_COL].astype(str).to_numpy()

    return df, X, y, groups, feature_cols


def tissue_summary(df):
    summary = df.groupby(TISSUE_COL).agg(
        n=(TARGET_COL, "size"),
        n_mut=(TARGET_COL, "sum"),
    )
    summary["mut_rate_%"] = (summary["n_mut"] / summary["n"] * 100).round(1)
    summary = summary.sort_values("n", ascending=False)
    return summary


def build_pipeline():
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(
            max_iter=2000,
            random_state=RANDOM_STATE,
            class_weight="balanced",
        )),
    ])


def fmt_aucs(aucs):
    return f"{np.mean(aucs):.4f} +- {np.std(aucs):.4f}"


# =========================================================
# SCENARIO A: StratifiedKFold standard
# =========================================================
def scenario_A_standard_cv(X, y):
    print("\n" + "=" * 60)
    print("SCENARIO A: StratifiedKFold standard (5-fold)")
    print("=" * 60)
    print("Ogni fold contiene tutti i tessuti. Questo e' cio' che farebbe")
    print("una CV 'normale'. Numero atteso ~ single-split AUC.")
    cv = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    model = build_pipeline()
    aucs = cross_val_score(model, X, y, cv=cv, scoring="roc_auc", n_jobs=1)
    print(f"AUC per fold: {np.round(aucs, 4).tolist()}")
    print(f"Media +- std: {fmt_aucs(aucs)}")
    return aucs


# =========================================================
# SCENARIO B: GroupKFold per tessuto
# =========================================================
def scenario_B_group_cv(X, y, groups):
    print("\n" + "=" * 60)
    print("SCENARIO B: GroupKFold per tessuto (5-fold)")
    print("=" * 60)
    print("I tessuti nel fold di test NON compaiono nel train.")
    print("Testa la vera generalizzazione a tessuti mai visti.")
    cv = GroupKFold(n_splits=N_SPLITS)
    model = build_pipeline()
    aucs = cross_val_score(
        model, X, y,
        groups=groups,
        cv=cv,
        scoring="roc_auc",
        n_jobs=1,
    )
    print(f"AUC per fold: {np.round(aucs, 4).tolist()}")
    print(f"Media +- std: {fmt_aucs(aucs)}")
    return aucs


# =========================================================
# SCENARIO C: solo tessuto, senza geni
# =========================================================
def scenario_C_tissue_only(df, y):
    print("\n" + "=" * 60)
    print("SCENARIO C: Tissue-only baseline (solo dummy tessuto)")
    print("=" * 60)
    print("Feature: solo one-hot del tessuto. Nessun gene.")
    print("Se AUC qui e' gia' alto -> buona parte del segnale e' gratis dal tessuto.")
    tissue_dummies = pd.get_dummies(df[TISSUE_COL].astype(str), drop_first=False)
    X_tissue = tissue_dummies.values
    print(f"Feature shape: {X_tissue.shape}")
    cv = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    model = LogisticRegression(
        max_iter=2000,
        random_state=RANDOM_STATE,
        class_weight="balanced",
    )
    aucs = cross_val_score(model, X_tissue, y, cv=cv, scoring="roc_auc", n_jobs=1)
    print(f"AUC per fold: {np.round(aucs, 4).tolist()}")
    print(f"Media +- std: {fmt_aucs(aucs)}")
    return aucs


# =========================================================
# BONUS: per-tissue AUC con LeaveOneGroupOut
# =========================================================
def per_tissue_breakdown(X, y, groups, min_samples=15):
    print("\n" + "=" * 60)
    print("BONUS: AUC per tessuto (LeaveOneGroupOut)")
    print("=" * 60)
    print(f"Per ogni tessuto con >= {min_samples} cell lines e almeno")
    print("entrambe le classi, alleniamo su tutto il resto e testiamo su lui.")

    logo = LeaveOneGroupOut()
    model = build_pipeline()
    y_proba = cross_val_predict(
        model, X, y,
        groups=groups,
        cv=logo,
        method="predict_proba",
        n_jobs=1,
    )[:, 1]

    groups_s = pd.Series(groups)
    rows = []
    for tissue in sorted(groups_s.unique()):
        mask = groups_s.values == tissue
        y_t = y[mask]
        p_t = y_proba[mask]
        n = int(mask.sum())
        n_mut = int(y_t.sum())
        if n < min_samples:
            continue
        if len(np.unique(y_t)) < 2:
            auc = np.nan
        else:
            auc = roc_auc_score(y_t, p_t)
        rows.append({
            "tissue": tissue,
            "n": n,
            "n_mut": n_mut,
            "mut_rate_%": round(n_mut / n * 100, 1),
            "auc": auc,
        })

    res = pd.DataFrame(rows).sort_values("auc", ascending=False)
    return res


# =========================================================
# MAIN
# =========================================================
def main():
    OUTPUT_DIR.mkdir(exist_ok=True)

    df, X, y, groups, feature_cols = load_data()
    print(f"N feature geniche: {len(feature_cols)}")
    print(f"N tessuti distinti: {len(set(groups))}")

    print("\nDistribuzione TP53 per tessuto (top 15):")
    summary = tissue_summary(df)
    print(summary.head(15).to_string())

    aucs_A = scenario_A_standard_cv(X, y)
    aucs_B = scenario_B_group_cv(X, y, groups)
    aucs_C = scenario_C_tissue_only(df, y)

    print("\n" + "=" * 60)
    print("RIEPILOGO CONFRONTO")
    print("=" * 60)
    compare = pd.DataFrame({
        "scenario": [
            "A) Standard 5-fold CV",
            "B) GroupKFold per tessuto",
            "C) Tissue-only baseline",
        ],
        "mean_auc": [np.mean(aucs_A), np.mean(aucs_B), np.mean(aucs_C)],
        "std_auc": [np.std(aucs_A), np.std(aucs_B), np.std(aucs_C)],
    })
    print(compare.round(4).to_string(index=False))
    compare.to_csv(OUTPUT_DIR / "task1_cv_comparison.csv", index=False)

    print("\nInterpretazione:")
    delta_B_A = np.mean(aucs_A) - np.mean(aucs_B)
    print(f"  A - B = {delta_B_A:+.4f}  (drop quando i tessuti sono out-of-training)")
    print(f"  C     = {np.mean(aucs_C):.4f}  (AUC spiegato dal solo tessuto)")

    # breakdown per tessuto
    per_tissue = per_tissue_breakdown(X, y, groups)
    print("\nAUC per tessuto (LeaveOneGroupOut, ordinato per AUC):")
    print(per_tissue.to_string(index=False))
    per_tissue.to_csv(OUTPUT_DIR / "task1_per_tissue_auc.csv", index=False)

    print("\nFile salvati:")
    print(f"  - {OUTPUT_DIR / 'task1_cv_comparison.csv'}")
    print(f"  - {OUTPUT_DIR / 'task1_per_tissue_auc.csv'}")


if __name__ == "__main__":
    main()
