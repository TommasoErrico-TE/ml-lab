import pandas as pd
from pathlib import Path

# =========================================================
# CONFIG
# =========================================================
DATA_DIR = Path("data")

EXPR_FILE = DATA_DIR / "OmicsExpressionTPMLogp1HumanProteinCodingGenes.csv"
MUT_FILE = DATA_DIR / "OmicsSomaticMutationsMatrixDamaging.csv"
MODEL_FILE = DATA_DIR / "Model.csv"
PROFILES_FILE = DATA_DIR / "OmicsProfiles.csv"  # caricato ma non indispensabile qui

OUTPUT_DIR = Path("output_tp53")
OUTPUT_DIR.mkdir(exist_ok=True)

TP53_GENE = "TP53"


# =========================================================
# HELPERS
# =========================================================
def clean_gene_columns(columns):
    """
    Trasforma colonne tipo 'TP53 (7157)' in 'TP53'.
    Le colonne metadata restano invariate.
    """
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


def find_exact_gene_column(df, gene_name):
    matches = [c for c in df.columns if c.strip().upper() == gene_name.upper()]
    if len(matches) == 0:
        raise ValueError(f"Colonna {gene_name} non trovata.")
    if len(matches) > 1:
        raise ValueError(f"Trovate più colonne chiamate {gene_name}: {matches}")
    return matches[0]


def normalize_yes_no(series):
    return series.astype(str).str.strip().str.lower().map({"yes": True, "no": False})


# =========================================================
# LOAD
# =========================================================
print("Caricamento file...")

expr = pd.read_csv(EXPR_FILE)
mut = pd.read_csv(MUT_FILE)
model = pd.read_csv(MODEL_FILE)

print(f"Expression shape: {expr.shape}")
print(f"Mutation shape:   {mut.shape}")
print(f"Model shape:      {model.shape}")

# =========================================================
# CLEAN COLUMN NAMES
# =========================================================
expr.columns = clean_gene_columns(expr.columns)
mut.columns = clean_gene_columns(mut.columns)

tp53_expr_col = find_exact_gene_column(expr, TP53_GENE)
tp53_mut_col = find_exact_gene_column(mut, TP53_GENE)

print(f"Colonna TP53 expression: {tp53_expr_col}")
print(f"Colonna TP53 mutation:   {tp53_mut_col}")

# =========================================================
# FILTER DEFAULT ENTRIES
# =========================================================
if "IsDefaultEntryForModel" in expr.columns:
    expr["IsDefaultEntryForModel_bool"] = normalize_yes_no(expr["IsDefaultEntryForModel"])
    expr = expr[expr["IsDefaultEntryForModel_bool"] == True].copy()

if "IsDefaultEntryForModel" in mut.columns:
    mut["IsDefaultEntryForModel_bool"] = normalize_yes_no(mut["IsDefaultEntryForModel"])
    mut = mut[mut["IsDefaultEntryForModel_bool"] == True].copy()

print(f"Expression dopo filtro default: {expr.shape}")
print(f"Mutation dopo filtro default:   {mut.shape}")

# =========================================================
# KEEP ONE ROW PER ModelID
# =========================================================
expr = expr.drop_duplicates(subset=["ModelID"]).copy()
mut = mut.drop_duplicates(subset=["ModelID"]).copy()

print(f"Expression unique ModelID: {expr.shape}")
print(f"Mutation unique ModelID:   {mut.shape}")

# =========================================================
# BUILD MASTER DATASET
# =========================================================

# teniamo solo la colonna TP53 dalla mutational matrix e la rinominiamo
mut_tp53 = mut[["ModelID", tp53_mut_col]].copy()
mut_tp53 = mut_tp53.rename(columns={tp53_mut_col: "TP53_damaging_score"})

# metadata modello
model_meta_cols = [
    c for c in [
        "ModelID",
        "CellLineName",
        "CCLEName",
        "OncotreeLineage",
        "OncotreePrimaryDisease",
        "OncotreeSubtype",
        "Sex",
        "AgeCategory",
        "PrimaryOrMetastasis"
    ] if c in model.columns
]
model_meta = model[model_meta_cols].drop_duplicates(subset=["ModelID"]).copy()

# merge base: expression + tp53 damaging + metadata
master = expr.merge(mut_tp53, on="ModelID", how="inner")
master = master.merge(model_meta, on="ModelID", how="left")

# =========================================================
# LABELS
# =========================================================

# TASK 1 ufficiale: mutant vs WT
master["task1_tp53_mutated"] = (master["TP53_damaging_score"] > 0).astype(int)

# Colonna utile informativa, non è la label ufficiale ma la teniamo
master = master.rename(columns={tp53_expr_col: "TP53_expression"})

# TASK 2: placeholder
# Non è definibile con questo file. La lasciamo vuota per ora.
master["task2_mutation_type"] = pd.NA

# =========================================================
# FEATURE COLUMNS
# =========================================================
metadata_and_label_cols = {
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

feature_cols = [c for c in master.columns if c not in metadata_and_label_cols]

# =========================================================
# SAVE
# =========================================================
master.to_csv(OUTPUT_DIR / "tp53_master_dataset.csv", index=False)

labels_only_cols = [
    c for c in [
        "ModelID",
        "CellLineName",
        "CCLEName",
        "OncotreeLineage",
        "OncotreePrimaryDisease",
        "TP53_expression",
        "TP53_damaging_score",
        "task1_tp53_mutated",
        "task2_mutation_type",
    ] if c in master.columns
]
master[labels_only_cols].to_csv(OUTPUT_DIR / "tp53_labels_only.csv", index=False)

with open(OUTPUT_DIR / "tp53_feature_columns.txt", "w") as f:
    for col in feature_cols:
        f.write(col + "\n")

# =========================================================
# REPORT
# =========================================================
print("\n=== DATASET MASTER ===")
print(f"Shape master: {master.shape}")

print("\nDistribuzione task1_tp53_mutated:")
print(master["task1_tp53_mutated"].value_counts(dropna=False))

print("\nDistribuzione TP53_damaging_score:")
print(master["TP53_damaging_score"].value_counts(dropna=False).sort_index())

print("\nStatistiche TP53_expression:")
print(master["TP53_expression"].describe())

print("\nNumero feature di espressione usabili:")
print(len(feature_cols))

print("\nPrime colonne TP53-related:")
print([c for c in master.columns if "TP53" in c])

print("\nFile salvati in:", OUTPUT_DIR.resolve())
print("- tp53_master_dataset.csv")
print("- tp53_labels_only.csv")
print("- tp53_feature_columns.txt")