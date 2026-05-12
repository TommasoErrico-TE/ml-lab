"""
Costruisce le label per il Task 2 in DUE tassonomie parallele:

(1) Coding consequence (livello proteico, standard SO/VEP):
    0) WT          - no functional TP53 mutation (or only silent)
    1) Missense    - cambio di un singolo aminoacido
    2) Inframe     - inframe insertion/deletion (mantiene reading frame)
    3) Truncating  - nonsense / frameshift / splice site (perdita di funzione)

(2) DNA-level (livello nucleotidico, dalla colonna VariantType di DepMap):
    0) WT          - no TP53 mutation
    1) SNV         - Single Nucleotide Variant
    2) MNV         - Multi-Nucleotide Variant (DepMap: 'substitution')
    3) Insertion   - insertion di nucleotidi
    4) Deletion    - delezione di nucleotidi

Le due tassonomie servono per il confronto richiesto dal progetto: la slide
del corso elenca *due* tabelle (DNA-level + coding consequence) e ci chiede
to evaluate which one is meaningful to classify. We keep both and
confrontiamo in `compare_task2_taxonomies.py`.

Per la coding consequence: regola di priorita' "worst-wins":
  Truncating > Inframe > Missense > WT (motivata biologicamente).

For DNA-level: we use the same anchor mutation chosen by the
coding-consequence e ne riportiamo il VariantType. Cosi' la stessa
physical mutation is labelled in two ways and the comparison
fra le tassonomie e' apples-to-apples.

Input:
  - data/OmicsSomaticMutations.csv            (catalogo completo mutazioni)
  - output_tp53/tp53_master_dataset.csv       (solo per lista ModelID)

Output:
  - output_tp53/tp53_task2_labels.csv         (ModelID + entrambe le label + dettaglio)
"""
from pathlib import Path
import pandas as pd

# =========================================================
# CONFIG
# =========================================================
DATA_DIR = Path("data")
OUTPUT_DIR = Path("output_tp53")

MUT_FULL_FILE = DATA_DIR / "OmicsSomaticMutations.csv"
MASTER_FILE = OUTPUT_DIR / "datasets/master.csv"

TP53_GENE = "TP53"

# keyword che triggerano ciascuna classe (cercate dentro VariantInfo,
# che puo' essere una catena di annotazioni separate da '&')
TRUNCATING_KEYWORDS = {
    "stop_gained",
    "frameshift_variant",
    "splice_acceptor_variant",
    "splice_donor_variant",
    "start_lost",
    "stop_lost",
    "transcript_ablation",
}
INFRAME_KEYWORDS = {
    "inframe_deletion",
    "inframe_insertion",
    "inframe_indel",
    "protein_altering_variant",  # alteration che mantiene frame
}
MISSENSE_KEYWORDS = {
    "missense_variant",
    "initiator_codon_variant",
    "initiatior_codon_variant",  # typo presente in DepMap, lo includiamo
}

# priority: higher rank = more severe. If a cell line has multiple mutations,
# keep the class with the highest rank.
CLASS_RANK = {
    "WT": 0,
    "Silent": 0,      # trattato come WT nell'output finale
    "Missense": 1,
    "Inframe": 2,
    "Truncating": 3,
}

# Mapping VariantType (DepMap MAF) -> tassonomia DNA-level della slide.
# DepMap usa 'substitution' per i multi-nucleotide variant (MNV): es. 2-3
# basi adiacenti che cambiano insieme.
DNA_TYPE_MAP = {
    "SNV": "SNV",
    "substitution": "MNV",
    "insertion": "Insertion",
    "deletion": "Deletion",
}

DNA_TYPE_TO_INT = {
    "WT": 0,
    "SNV": 1,
    "MNV": 2,
    "Insertion": 3,
    "Deletion": 4,
}


# =========================================================
# SINGLE-MUTATION CLASSIFIER
# =========================================================
def classify_variant_info(variant_info):
    """
    Restituisce 'Truncating' / 'Inframe' / 'Missense' / 'Silent' / 'Unknown'
    guardando le keyword SO (Sequence Ontology) in VariantInfo.

    VariantInfo puo' essere tipo:
      - 'missense_variant'
      - 'stop_gained&splice_region_variant'
      - 'splice_donor_variant&coding_sequence_variant&intron_variant'
    """
    if pd.isna(variant_info) or not isinstance(variant_info, str):
        return "Unknown"
    parts = set(variant_info.split("&"))

    if parts & TRUNCATING_KEYWORDS:
        return "Truncating"
    if parts & INFRAME_KEYWORDS:
        return "Inframe"
    if parts & MISSENSE_KEYWORDS:
        return "Missense"
    # tutto il resto (synonymous, splice_region alone, intron_variant,
    # non_coding_transcript_variant, UTR, ecc.) -> Silent
    return "Silent"


# =========================================================
# LOAD DATI
# =========================================================
def load_tp53_mutations():
    print("Loading OmicsSomaticMutations.csv (only useful columns)...")
    usecols = [
        "ModelID",
        "HugoSymbol",
        "VariantType",
        "VariantInfo",
        "ProteinChange",
        "IsDefaultEntryForModel",
    ]
    df = pd.read_csv(MUT_FULL_FILE, usecols=usecols, low_memory=False)
    print(f"  Totale righe: {len(df)}")

    tp53 = df[df["HugoSymbol"] == TP53_GENE].copy()
    print(f"  Mutazioni su {TP53_GENE}: {len(tp53)}")

    # coerenza con data.py: solo default entry per ModelID
    is_default = tp53["IsDefaultEntryForModel"].astype(str).str.strip().str.lower().eq("yes")
    tp53 = tp53[is_default].copy()
    print(f"  Dopo filtro IsDefaultEntryForModel=Yes: {len(tp53)}")
    return tp53


def load_master():
    print("\nLoading master dataset (solo ModelID e task1)...")
    df = pd.read_csv(
        MASTER_FILE,
        usecols=["ModelID", "task1_tp53_mutated"],
    )
    print(f"  Cell lines nel master: {len(df)}")
    return df


# =========================================================
# AGGREGATE PER ModelID
# =========================================================
def aggregate_per_model(tp53):
    """
    Per ogni ModelID con almeno una mutazione TP53, assegna la classe
    with the highest rank (= most severe) for the coding consequence, and
    riporta il VariantType (DNA-level) della *stessa* mutazione "ancora".

    Cosi' le due tassonomie etichettano la stessa mutazione fisica:
    confronto apples-to-apples.

    Restituisce un DataFrame con:
      ModelID, task2_mutation_class (coding), worst_variant_type (DNA),
      worst_variant_info, worst_protein_change, n_tp53_mutations.
    """
    tp53 = tp53.copy()
    tp53["mutation_class"] = tp53["VariantInfo"].map(classify_variant_info)
    tp53["mutation_rank"] = tp53["mutation_class"].map(CLASS_RANK).fillna(0).astype(int)

    # ordina per rank decrescente: la prima occorrenza per ModelID e' la peggiore
    tp53_sorted = tp53.sort_values(["ModelID", "mutation_rank"], ascending=[True, False])
    worst = tp53_sorted.groupby("ModelID", as_index=False).first()
    worst = worst.rename(columns={
        "mutation_class": "task2_mutation_class",
        "VariantInfo": "worst_variant_info",
        "VariantType": "worst_variant_type",
        "ProteinChange": "worst_protein_change",
    })

    # conta anche quante mutazioni per ModelID
    counts = tp53.groupby("ModelID").size().rename("n_tp53_mutations")
    worst = worst.merge(counts, on="ModelID", how="left")

    return worst[[
        "ModelID",
        "task2_mutation_class",
        "worst_variant_type",
        "worst_variant_info",
        "worst_protein_change",
        "n_tp53_mutations",
    ]]


# =========================================================
# MERGE CON MASTER E ASSEGNA WT AI MANCANTI
# =========================================================
def assemble_labels(master, worst):
    merged = master.merge(worst, on="ModelID", how="left")

    # ---- Tassonomia (1): coding consequence ---------------------------
    # cell lines senza mutazione TP53 nel file -> WT
    merged["task2_mutation_class"] = merged["task2_mutation_class"].fillna("WT")
    # Silent -> collapsed into WT for the final output (no protein-level change)
    merged["task2_mutation_type"] = merged["task2_mutation_class"].replace({"Silent": "WT"})

    # encoding numerico utile per sklearn
    type_to_int = {"WT": 0, "Missense": 1, "Inframe": 2, "Truncating": 3}
    merged["task2_mutation_type_int"] = merged["task2_mutation_type"].map(type_to_int).astype(int)

    # ---- Tassonomia (2): DNA-level ------------------------------------
    # mappiamo VariantType (es. 'SNV', 'substitution', 'insertion', 'deletion')
    # nelle classi della slide del corso (SNV, MNV, Insertion, Deletion).
    # Cell lines senza mutazione (Silent collassato a WT) -> WT.
    is_wt_coding = merged["task2_mutation_type"] == "WT"
    dna_type = merged["worst_variant_type"].map(DNA_TYPE_MAP)
    dna_type = dna_type.where(~is_wt_coding, "WT")
    # Eventuali VariantType non mappati (caso raro): segnaliamo con 'Other'
    # per non perdere il dato silenziosamente.
    dna_type = dna_type.fillna("Other")
    merged["task2_dna_type"] = dna_type
    merged["task2_dna_type_int"] = merged["task2_dna_type"].map(DNA_TYPE_TO_INT).fillna(-1).astype(int)

    merged["n_tp53_mutations"] = merged["n_tp53_mutations"].fillna(0).astype(int)
    return merged


# =========================================================
# REPORT DI COERENZA CON TASK 1
# =========================================================
def _format_distribution(series, class_order):
    out = []
    counts = series.value_counts().reindex(class_order, fill_value=0)
    total = counts.sum()
    for cls, n in counts.items():
        pct = 100 * n / total if total else 0.0
        out.append(f"  {cls:<11s}: {n:>5d}  ({pct:5.1f}%)")
    out.append(f"  {'TOTALE':<11s}: {total:>5d}")
    return out


def build_summary(labels):
    lines = []
    lines.append("=" * 60)
    lines.append("TASK 2 (1) CODING CONSEQUENCE - DISTRIBUZIONE")
    lines.append("=" * 60)
    lines += _format_distribution(
        labels["task2_mutation_type"],
        ["WT", "Missense", "Inframe", "Truncating"],
    )

    lines.append("")
    lines.append("=" * 60)
    lines.append("TASK 2 (2) DNA-LEVEL - DISTRIBUZIONE")
    lines.append("=" * 60)
    lines += _format_distribution(
        labels["task2_dna_type"],
        ["WT", "SNV", "MNV", "Insertion", "Deletion", "Other"],
    )

    lines.append("")
    lines.append("=" * 60)
    lines.append("CROSSTAB: DNA-LEVEL x CODING CONSEQUENCE")
    lines.append("=" * 60)
    lines.append("(quanto le due tassonomie si sovrappongono)")
    lines.append("")
    crosstab_dna_coding = pd.crosstab(
        labels["task2_dna_type"],
        labels["task2_mutation_type"],
        margins=True,
    )
    lines.append(crosstab_dna_coding.to_string())

    lines.append("")
    lines.append("=" * 60)
    lines.append("COERENZA TASK 1 vs TASK 2 (coding consequence)")
    lines.append("=" * 60)
    crosstab_t1_t2 = pd.crosstab(
        labels["task1_tp53_mutated"],
        labels["task2_mutation_type"],
        margins=True,
    )
    lines.append(crosstab_t1_t2.to_string())

    lines.append("")
    n_conflict_mutated_but_WT = int(((labels["task1_tp53_mutated"] == 1) &
                                     (labels["task2_mutation_type"] == "WT")).sum())
    n_conflict_WT_but_typed = int(((labels["task1_tp53_mutated"] == 0) &
                                   (labels["task2_mutation_type"] != "WT")).sum())
    lines.append(f"task1=mutato ma task2=WT (solo mutazioni silent): {n_conflict_mutated_but_WT}")
    lines.append(f"task1=WT ma task2!=WT (mutazioni non-damaging): {n_conflict_WT_but_typed}")
    lines.append("")
    lines.append("Nota: non sono veri conflitti, sono coerenti con la definizione")
    lines.append("diversa delle due label. Task 1 usa la matrice 'damaging' di DepMap,")
    lines.append("Task 2 (coding) usa qualsiasi mutazione con effetto sulla proteina.")
    return "\n".join(lines)


# =========================================================
# MAIN
# =========================================================
def main():
    OUTPUT_DIR.mkdir(exist_ok=True)
    (OUTPUT_DIR / 'datasets').mkdir(parents=True, exist_ok=True)
    tp53 = load_tp53_mutations()
    master = load_master()

    print("\nClassificazione di ogni mutazione...")
    worst = aggregate_per_model(tp53)
    print(f"  Cell lines con almeno 1 mutazione TP53: {len(worst)}")
    print("  Distribution della classe 'peggiore' per cell line:")
    print(worst["task2_mutation_class"].value_counts().to_string())

    labels = assemble_labels(master, worst)

    out_cols = [
        "ModelID",
        "task1_tp53_mutated",
        "task2_mutation_type",            # coding consequence (str)
        "task2_mutation_type_int",        # coding consequence (int)
        "task2_dna_type",                 # DNA-level (str)
        "task2_dna_type_int",             # DNA-level (int)
        "n_tp53_mutations",
        "worst_variant_type",             # SNV/insertion/.. dal MAF (raw)
        "worst_variant_info",             # annotazione SO della mutazione "ancora"
        "worst_protein_change",
    ]
    out_path = OUTPUT_DIR / "datasets/task2_labels.csv"
    labels[out_cols].to_csv(out_path, index=False)

    summary = build_summary(labels)
    print("\n" + summary)

    print("\nFile salvati:")
    print(f"  - {out_path}")


if __name__ == "__main__":
    main()
