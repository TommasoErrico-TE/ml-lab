import pandas as pd
from pathlib import Path

# =========================================================
# CONFIG
# =========================================================
DATA_DIR = Path("data")

MUT_MAF_FILE = DATA_DIR / "OmicsSomaticMutations.csv"
MUT_MATRIX_FILE = DATA_DIR / "OmicsSomaticMutationsMatrixDamaging.csv"

OUTPUT_DIR = Path("output_tp53")
OUTPUT_DIR.mkdir(exist_ok=True)

TP53_GENE = "TP53"


# =========================================================
# HELPERS
# =========================================================
def clean_gene_columns(columns):
    metadata_cols = {
        "SequencingID",
        "ProfileID",
        "ModelConditionID",
        "ModelID",
        "IsDefaultEntryForMC",
        "IsDefaultEntryForModel",
        "is_default_entry",
    }

    cleaned = []
    for c in columns:
        if c in metadata_cols:
            cleaned.append(c)
        elif " (" in c and c.endswith(")"):
            cleaned.append(c.split(" (")[0])
        else:
            cleaned.append(c)
    return cleaned


def normalize_yes_no(series):
    return series.astype(str).str.strip().str.lower().map({"yes": True, "no": False})


# =========================================================
# LOAD
# =========================================================
print("Caricamento file...")
maf = pd.read_csv(MUT_MAF_FILE)
matrix = pd.read_csv(MUT_MATRIX_FILE)

print(f"MAF-like shape: {maf.shape}")
print(f"Matrix shape:   {matrix.shape}")

# pulizia nomi colonne matrix
matrix.columns = clean_gene_columns(matrix.columns)

if TP53_GENE not in matrix.columns:
    raise ValueError(f"Colonna {TP53_GENE} non trovata in OmicsSomaticMutationsMatrixDamaging.csv")

# =========================================================
# FILTER DEFAULT ENTRIES
# =========================================================
if "IsDefaultEntryForModel" in maf.columns:
    maf["IsDefaultEntryForModel_bool"] = normalize_yes_no(maf["IsDefaultEntryForModel"])
    maf = maf[maf["IsDefaultEntryForModel_bool"] == True].copy()

if "IsDefaultEntryForModel" in matrix.columns:
    matrix["IsDefaultEntryForModel_bool"] = normalize_yes_no(matrix["IsDefaultEntryForModel"])
    matrix = matrix[matrix["IsDefaultEntryForModel_bool"] == True].copy()

print(f"MAF-like dopo filtro default: {maf.shape}")
print(f"Matrix dopo filtro default:   {matrix.shape}")

# =========================================================
# 1) PRENDI SOLO TP53 DAL FILE MAF-LIKE
# =========================================================
maf_tp53 = maf[maf["HugoSymbol"].astype(str).str.upper() == TP53_GENE].copy()

print(f"\nRighe TP53 nel file MAF-like: {maf_tp53.shape[0]}")

# una riga = una variante; per ModelID vogliamo sapere se esiste almeno una mutazione TP53
# in questo file, se una riga TP53 esiste, la consideriamo 'mutated_in_maf = 1'
maf_tp53["mutated_in_maf"] = 1

# riassunto per ModelID
maf_tp53_summary = (
    maf_tp53.groupby("ModelID", as_index=False)
    .agg(
        maf_n_tp53_variants=("HugoSymbol", "size"),
        maf_dna_changes=("DNAChange", lambda x: "; ".join(sorted(set(x.dropna().astype(str))))),
        maf_protein_changes=("ProteinChange", lambda x: "; ".join(sorted(set(x.dropna().astype(str))))),
        maf_variant_types=("VariantType", lambda x: "; ".join(sorted(set(x.dropna().astype(str))))),
        maf_molecular_consequences=("MolecularConsequence", lambda x: "; ".join(sorted(set(x.dropna().astype(str))))),
        maf_likely_lof_any=("LikelyLoF", lambda x: int(pd.Series(x).fillna(False).astype(bool).any())),
        mutated_in_maf=("mutated_in_maf", "max"),
    )
)

# =========================================================
# 2) PRENDI TP53 DALLA MATRIX DANNEGGIANTE
# =========================================================
matrix_tp53 = matrix[["ModelID", TP53_GENE]].copy()
matrix_tp53 = matrix_tp53.rename(columns={TP53_GENE: "matrix_tp53_score"})
matrix_tp53["mutated_in_matrix"] = (matrix_tp53["matrix_tp53_score"] > 0).astype(int)

print("\nDistribuzione TP53 nella matrix:")
print(matrix_tp53["matrix_tp53_score"].value_counts(dropna=False).sort_index())

# =========================================================
# 3) CONFRONTO TRA I DUE FILE
# =========================================================
comparison = matrix_tp53.merge(maf_tp53_summary, on="ModelID", how="outer")

comparison["matrix_tp53_score"] = comparison["matrix_tp53_score"].fillna(0)
comparison["mutated_in_matrix"] = comparison["mutated_in_matrix"].fillna(0).astype(int)
comparison["mutated_in_maf"] = comparison["mutated_in_maf"].fillna(0).astype(int)
comparison["maf_n_tp53_variants"] = comparison["maf_n_tp53_variants"].fillna(0).astype(int)

# stessa etichetta sì/no
comparison["same_label"] = (comparison["mutated_in_matrix"] == comparison["mutated_in_maf"]).astype(int)

# categoria confronto
def compare_status(row):
    if row["mutated_in_matrix"] == 1 and row["mutated_in_maf"] == 1:
        return "both_mutated"
    elif row["mutated_in_matrix"] == 0 and row["mutated_in_maf"] == 0:
        return "both_wt"
    elif row["mutated_in_matrix"] == 1 and row["mutated_in_maf"] == 0:
        return "matrix_only_mutated"
    else:
        return "maf_only_mutated"

comparison["comparison_status"] = comparison.apply(compare_status, axis=1)

# =========================================================
# 4) OUTPUT
# =========================================================
comparison = comparison.sort_values(["same_label", "comparison_status", "ModelID"])

comparison.to_csv(OUTPUT_DIR / "tp53_maf_vs_matrix_comparison.csv", index=False)

discordant = comparison[comparison["same_label"] == 0].copy()
discordant.to_csv(OUTPUT_DIR / "tp53_maf_vs_matrix_discordant_only.csv", index=False)

maf_tp53_summary.to_csv(OUTPUT_DIR / "tp53_maf_summary_only.csv", index=False)

# =========================================================
# 5) REPORT
# =========================================================
print("\n=== CONFRONTO TP53 TRA MAF-LIKE E MATRIX ===")
print(f"Numero ModelID totali nel confronto: {comparison.shape[0]}")

print("\nComparison status:")
print(comparison["comparison_status"].value_counts(dropna=False))

print("\nSame label:")
print(comparison["same_label"].value_counts(dropna=False))

print("\nEsempio righe discordanti:")
print(
    discordant[
        [
            "ModelID",
            "matrix_tp53_score",
            "mutated_in_matrix",
            "maf_n_tp53_variants",
            "mutated_in_maf",
            "maf_dna_changes",
            "maf_protein_changes",
            "maf_variant_types",
            "maf_molecular_consequences",
            "maf_likely_lof_any",
            "comparison_status",
        ]
    ].head(20).to_string(index=False)
)

print("\nFile salvati in:", OUTPUT_DIR.resolve())
print("- tp53_maf_vs_matrix_comparison.csv")
print("- tp53_maf_vs_matrix_discordant_only.csv")
print("- tp53_maf_summary_only.csv")