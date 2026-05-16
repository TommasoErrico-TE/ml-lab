"""
Costruisce il master dataset TCGA Pan-Cancer:
  righe = sample (primary tumor)
  colonne = geni (intersezione TCGA expression vs CCLE master)
  + label task1/task2 derivate dal MC3 MAF
  + lineage da Survival_SupplementalTable

Output:
  output_tp53/datasets/tcga_master.parquet
  output_tp53/datasets/tcga_master_labels.csv   (solo metadati + label, comodo da ispezionare)
"""

import gzip
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

TCGA_DIR = Path("data/tcga")
EXPR_FILE = TCGA_DIR / "EB++AdjustPANCAN_IlluminaHiSeq_RNASeqV2.geneExp.xena.gz"
MAF_FILE = TCGA_DIR / "mc3.v0.2.8.PUBLIC.xena.gz"
CLIN_FILE = TCGA_DIR / "Survival_SupplementalTable_S1_20171025_xena_sp"
CCLE_MASTER = Path("output_tp53/datasets/master.csv")
OUTPUT_DIR = Path("output_tp53/datasets")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Tassonomia identica a CCLE Task 2
TRUNCATING_EFFECTS = {
    "Nonsense_Mutation",
    "Frame_Shift_Del",
    "Frame_Shift_Ins",
    "Splice_Site",
    "Translation_Start_Site",
    "Nonstop_Mutation",
}
INFRAME_EFFECTS = {"In_Frame_Del", "In_Frame_Ins"}
MISSENSE_EFFECTS = {"Missense_Mutation"}
# Tutto il resto (Silent, 3'UTR, ...) e' trattato come non-impattante (WT-like)


def primary_tumor(sample_id: str) -> bool:
    """TCGA barcode: i char 14-15 sono il sample type. 01-09 = tumor."""
    # 'TCGA-XX-YYYY-01'  ->  prendi gli ultimi 2 caratteri dopo l'ultimo '-'
    parts = sample_id.split("-")
    if len(parts) < 4:
        return False
    return parts[3].startswith("0") and parts[3][:2] != "10" and int(parts[3][:2]) < 10


def build_tp53_labels(maf_path: Path, expr_samples: list) -> pd.DataFrame:
    """Per ogni sample in expr_samples, calcola Task1 (binary) e Task2 (coding)."""
    print(f"Loading MC3 MAF {maf_path}...")
    maf = pd.read_csv(maf_path, sep="\t", compression="gzip", low_memory=False)
    print(f"  MAF shape: {maf.shape}")

    tp53 = maf[maf["gene"] == "TP53"].copy()
    print(f"  TP53 rows: {len(tp53)}  ({tp53['sample'].nunique()} sample unici)")

    # priorita': Truncating > Inframe > Missense > altro (silent, ecc.)
    def categorize(eff):
        if eff in TRUNCATING_EFFECTS:
            return "Truncating"
        if eff in INFRAME_EFFECTS:
            return "Inframe"
        if eff in MISSENSE_EFFECTS:
            return "Missense"
        return "Other"

    tp53["category"] = tp53["effect"].apply(categorize)
    priority = {"Truncating": 3, "Inframe": 2, "Missense": 1, "Other": 0}
    tp53["prio"] = tp53["category"].map(priority)
    # tieni la mutazione con priorita' massima per ogni sample
    tp53 = tp53.sort_values("prio", ascending=False).drop_duplicates("sample")
    tp53 = tp53[["sample", "category", "effect", "Amino_Acid_Change"]].rename(
        columns={"Amino_Acid_Change": "worst_protein_change",
                 "effect": "worst_effect"}
    )

    # build label DataFrame allineato a expr_samples
    out = pd.DataFrame({"sample": expr_samples})
    out = out.merge(tp53, on="sample", how="left")
    # WT = nessuna mutazione, oppure solo mutazioni "Other"
    out["task2_mutation_type"] = out["category"].fillna("WT")
    out.loc[out["task2_mutation_type"] == "Other", "task2_mutation_type"] = "WT"
    out["task1_tp53_mutated"] = (out["task2_mutation_type"] != "WT").astype(int)
    out = out.drop(columns=["category"])
    return out


def main():
    # 1) lista geni CCLE per intersezione (il transfer ha senso solo sui geni in comune)
    print("Loading CCLE master per gene list...")
    ccle_header = pd.read_csv(CCLE_MASTER, nrows=0)
    NON_FEATURE = {
        "Unnamed: 0", "SequencingID", "ModelConditionID", "ModelID",
        "IsDefaultEntryForMC", "IsDefaultEntryForModel", "IsDefaultEntryForModel_bool",
        "TP53_expression", "TP53_damaging_score",
        "task1_tp53_mutated", "task2_mutation_type",
        "CellLineName", "CCLEName",
        "OncotreeLineage", "OncotreePrimaryDisease", "OncotreeSubtype",
        "Sex", "AgeCategory", "PrimaryOrMetastasis",
    }
    ccle_genes = [c for c in ccle_header.columns if c not in NON_FEATURE]
    print(f"  CCLE feature genes: {len(ccle_genes)}")

    # 2) leggi expression TCGA
    print(f"\nLoading TCGA expression {EXPR_FILE}...")
    expr = pd.read_csv(EXPR_FILE, sep="\t", compression="gzip", index_col=0)
    print(f"  Raw expression shape: {expr.shape}  (genes x samples)")

    # 3) filtra a primary tumor
    samples_all = expr.columns.tolist()
    samples_pt = [s for s in samples_all if primary_tumor(s)]
    print(f"  Primary tumor samples: {len(samples_pt)} / {len(samples_all)}")
    expr = expr[samples_pt]

    # 4) intersezione geni con CCLE
    common_genes = [g for g in ccle_genes if g in expr.index]
    print(f"  Geni in comune CCLE-TCGA: {len(common_genes)} / {len(ccle_genes)}")
    expr = expr.loc[common_genes]

    # 5) trasponi: righe = sample, colonne = gene
    X = expr.T  # (samples x genes)
    X.index.name = "sample"
    print(f"  After transpose: {X.shape}")

    # 6) label TP53
    labels = build_tp53_labels(MAF_FILE, X.index.tolist())
    print("\nClass distribution TCGA:")
    print(labels["task1_tp53_mutated"].value_counts())
    print(labels["task2_mutation_type"].value_counts())

    # 7) lineage da Survival table
    print(f"\nLoading clinical {CLIN_FILE}...")
    clin = pd.read_csv(CLIN_FILE, sep="\t")
    # Survival sample id e' "TCGA-XX-YYYY-01" come l'expression
    clin = clin[["sample", "cancer type abbreviation"]].rename(
        columns={"cancer type abbreviation": "tcga_cancer_type"}
    )
    labels = labels.merge(clin, on="sample", how="left")

    # 8) merge labels + expression -> master parquet
    master = labels.set_index("sample").join(X)
    print(f"\nMaster TCGA shape: {master.shape}")

    out_parquet = OUTPUT_DIR / "tcga_master.parquet"
    master.to_parquet(out_parquet, engine="pyarrow", compression="snappy")
    print(f"  saved {out_parquet}  ({out_parquet.stat().st_size/1e6:.1f} MB)")

    labels_out = OUTPUT_DIR / "tcga_master_labels.csv"
    labels.to_csv(labels_out, index=False)
    print(f"  saved {labels_out}")

    # piccolo summary
    print("\nTop cancer types:")
    print(labels["tcga_cancer_type"].value_counts().head(10).to_string())


if __name__ == "__main__":
    main()
