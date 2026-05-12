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
(OUTPUT_DIR / 'datasets').mkdir(parents=True, exist_ok=True)

TP53_GENE = "TP53"


# =========================================================
# HELPERS
# =========================================================
def clean_gene_columns(columns):
    """
    Cleans column names like 'TP53 (7157)' in 'TP53'.
    Metadata columns stay unchanged.
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
        raise ValueError(f"Column {gene_name} not found.")
    if len(matches) > 1:
        raise ValueError(f"Multiple columns called {gene_name}: {matches}")
    return matches[0]


def normalize_yes_no(series):
    return series.astype(str).str.strip().str.lower().map({"yes": True, "no": False})


# =========================================================
# LOAD
# =========================================================
print("Loading files...")

expr = pd.read_csv(EXPR_FILE)
mut = pd.read_csv(MUT_FILE)
model = pd.read_csv(MODEL_FILE)

# Drop unnamed index columns (es. "Unnamed: 0") che pandas creates when
# the CSV has a numeric index column. Otherwise ends up as a feature.
expr = expr.loc[:, ~expr.columns.str.startswith("Unnamed")]
mut = mut.loc[:, ~mut.columns.str.startswith("Unnamed")]

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

print(f"Column TP53 expression: {tp53_expr_col}")
print(f"Column TP53 mutation:   {tp53_mut_col}")

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
print(f"Mutation after default filter:   {mut.shape}")

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

# keep only the TP53 column from the mutation matrix e rename it
mut_tp53 = mut[["ModelID", tp53_mut_col]].copy()
mut_tp53 = mut_tp53.rename(columns={tp53_mut_col: "TP53_damaging_score"})

# cell-line metadata
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

# base merge: expression + TP53 damaging + metadata
master = expr.merge(mut_tp53, on="ModelID", how="inner")
master = master.merge(model_meta, on="ModelID", how="left")

# =========================================================
# LABELS
# =========================================================

# TASK 1 label: mutant vs WT
master["task1_tp53_mutated"] = (master["TP53_damaging_score"] > 0).astype(int)

# Informational column, kept for reference
master = master.rename(columns={tp53_expr_col: "TP53_expression"})

# TASK 2: placeholder (real labels in tp53_task2_labels.csv)
# Cannot be derived from this file; left empty.
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
master.to_csv(OUTPUT_DIR / "datasets/master.csv", index=False)

