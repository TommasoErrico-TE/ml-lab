"""
Per-gene correlation con la label TP53 — Spearman + confronto con Pearson.

Motivazione: Pearson cattura solo associazioni lineari ed e' sensibile
a outlier; Spearman lavora sui rank, quindi e' piu' robusto e cattura
relazioni monotone non lineari.

Trick implementativo: Spearman = Pearson sui rank delle colonne.
Cosi' possiamo calcolarlo in un colpo solo su 19k geni invece di
chiamare scipy.stats.spearmanr in loop.

Outputs (in output_tp53/task1/):
  - gene_correlation_pearson.csv
  - gene_correlation_spearman.csv
  - gene_correlation_comparison.csv   (entrambe le metriche + delta)
"""

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata

warnings.filterwarnings("ignore")

DATA_FILE = Path("output_tp53") / "datasets/master.csv"
OUTPUT_DIR = Path("output_tp53") / "task1"
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


def pearson_columns_vs_y(X, y):
    """Pearson r tra ogni colonna di X (matrice) e y (vettore). Vettorizzato."""
    Xc = X - X.mean(axis=0)
    yc = y - y.mean()
    num = (Xc * yc[:, None]).sum(axis=0)
    den = np.sqrt((Xc ** 2).sum(axis=0) * (yc ** 2).sum())
    return num / (den + 1e-12)


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(DATA_FILE)
    feature_cols = [c for c in df.columns if c not in NON_FEATURE_COLS]
    Xdf = df[feature_cols].fillna(df[feature_cols].median())
    X = Xdf.values.astype(np.float64)
    y = df[TARGET_COL].values.astype(np.float64)
    print(f"X: {X.shape}   y mean: {y.mean():.3f}")

    print("Computing Pearson...")
    pearson_r = pearson_columns_vs_y(X, y)

    print("Computing Spearman (Pearson on column ranks)...")
    X_rank = np.apply_along_axis(rankdata, 0, X)
    # y e' binaria: il rank rompe i tie ma il risultato e' equivalente
    # al point-biserial sui rank, che e' la definizione standard.
    y_rank = rankdata(y)
    spearman_r = pearson_columns_vs_y(X_rank, y_rank)

    pear_df = pd.DataFrame({"gene": feature_cols, "pearson_r": pearson_r})
    spear_df = pd.DataFrame({"gene": feature_cols, "spearman_r": spearman_r})
    pear_df.sort_values("pearson_r").to_csv(OUTPUT_DIR / "gene_correlation_pearson.csv", index=False)
    spear_df.sort_values("spearman_r").to_csv(OUTPUT_DIR / "gene_correlation_spearman.csv", index=False)

    comp = pear_df.merge(spear_df, on="gene")
    comp["delta"] = comp["spearman_r"] - comp["pearson_r"]
    comp["abs_delta"] = comp["delta"].abs()
    comp.sort_values("abs_delta", ascending=False).to_csv(
        OUTPUT_DIR / "gene_correlation_comparison.csv", index=False,
    )

    # Stampe di sanita'
    agreement = np.corrcoef(pearson_r, spearman_r)[0, 1]
    rank_top100_overlap = len(
        set(pear_df.nsmallest(100, "pearson_r")["gene"]) &
        set(spear_df.nsmallest(100, "spearman_r")["gene"])
    )
    print(f"\nCorrelazione tra Pearson r e Spearman r (su tutti i geni): {agreement:.4f}")
    print(f"Overlap top-100 negativi (Pearson vs Spearman): {rank_top100_overlap}/100")
    print(f"\nTop 10 differenze |spearman - pearson|:")
    print(comp.sort_values("abs_delta", ascending=False).head(10).round(4).to_string(index=False))

    print(f"\nSaved:")
    print(f"  - {OUTPUT_DIR / 'gene_correlation_pearson.csv'}")
    print(f"  - {OUTPUT_DIR / 'gene_correlation_spearman.csv'}")
    print(f"  - {OUTPUT_DIR / 'gene_correlation_comparison.csv'}")


if __name__ == "__main__":
    main()
