"""
Costruisce le label per il Task 2: classificazione a 4 classi del tipo di
mutazione TP53 per ogni cell line.

Classi:
  0) WT          - nessuna mutazione TP53 funzionale (o solo silent)
  1) Missense    - cambio di un singolo aminoacido
  2) Inframe     - inframe insertion/deletion (mantiene reading frame)
  3) Truncating  - nonsense / frameshift / splice site (perdita di funzione)

Priorita' quando una cell line ha piu' mutazioni TP53:
  Truncating > Inframe > Missense > WT
Cioe' vince la peggiore (funzionalmente piu' distruttiva).

Input:
  - data/OmicsSomaticMutations.csv            (catalogo completo mutazioni)
  - output_tp53/tp53_master_dataset.csv       (solo per lista ModelID)

Output:
  - output_tp53/tp53_task2_labels.csv         (ModelID + task2 + dettaglio)
  - output_tp53/tp53_task2_summary.txt        (statistiche + coerenza con Task 1)
"""
from pathlib import Path
import pandas as pd

# =========================================================
# CONFIG
# =========================================================
DATA_DIR = Path("data")
OUTPUT_DIR = Path("output_tp53")

MUT_FULL_FILE = DATA_DIR / "OmicsSomaticMutations.csv"
MASTER_FILE = OUTPUT_DIR / "tp53_master_dataset.csv"

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

# priorita': piu' alto = piu' grave. Se una cell line ha piu' mutazioni,
# teniamo la classe con rank piu' alto.
CLASS_RANK = {
    "WT": 0,
    "Silent": 0,      # trattato come WT nell'output finale
    "Missense": 1,
    "Inframe": 2,
    "Truncating": 3,
}


# =========================================================
# CLASSIFICATORE DI UNA SINGOLA MUTAZIONE
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
    print("Caricamento OmicsSomaticMutations.csv (solo colonne utili)...")
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
    print("\nCaricamento master dataset (solo ModelID e task1)...")
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
    con rank piu' alto (= piu' grave).
    Restituisce un DataFrame con ModelID, task2_mutation_type,
    n_mutations, worst_variant_info, worst_protein_change.
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
        "ProteinChange": "worst_protein_change",
    })

    # conta anche quante mutazioni per ModelID
    counts = tp53.groupby("ModelID").size().rename("n_tp53_mutations")
    worst = worst.merge(counts, on="ModelID", how="left")

    return worst[[
        "ModelID",
        "task2_mutation_class",
        "worst_variant_info",
        "worst_protein_change",
        "n_tp53_mutations",
    ]]


# =========================================================
# MERGE CON MASTER E ASSEGNA WT AI MANCANTI
# =========================================================
def assemble_labels(master, worst):
    merged = master.merge(worst, on="ModelID", how="left")

    # cell lines senza mutazione TP53 nel file -> WT
    merged["task2_mutation_class"] = merged["task2_mutation_class"].fillna("WT")
    # Silent -> lo collassiamo dentro WT per l'output finale (nessuna alterazione proteica)
    merged["task2_mutation_type"] = merged["task2_mutation_class"].replace({"Silent": "WT"})

    # encoding numerico utile per sklearn
    type_to_int = {"WT": 0, "Missense": 1, "Inframe": 2, "Truncating": 3}
    merged["task2_mutation_type_int"] = merged["task2_mutation_type"].map(type_to_int).astype(int)

    merged["n_tp53_mutations"] = merged["n_tp53_mutations"].fillna(0).astype(int)
    return merged


# =========================================================
# REPORT DI COERENZA CON TASK 1
# =========================================================
def build_summary(labels):
    lines = []
    lines.append("=" * 60)
    lines.append("TASK 2 - DISTRIBUZIONE DELLE CLASSI")
    lines.append("=" * 60)
    counts = labels["task2_mutation_type"].value_counts().reindex(
        ["WT", "Missense", "Inframe", "Truncating"]
    )
    total = counts.sum()
    for cls, n in counts.items():
        pct = 100 * n / total
        lines.append(f"  {cls:<11s}: {n:>5d}  ({pct:5.1f}%)")
    lines.append(f"  {'TOTALE':<11s}: {total:>5d}")

    lines.append("")
    lines.append("=" * 60)
    lines.append("COERENZA TASK 1 vs TASK 2")
    lines.append("=" * 60)
    # tabella incrociata
    crosstab = pd.crosstab(
        labels["task1_tp53_mutated"],
        labels["task2_mutation_type"],
        margins=True,
    )
    lines.append(crosstab.to_string())

    lines.append("")
    # casi "sospetti": task1 != 0 e task2 == WT, oppure task1 == 0 e task2 != WT
    n_conflict_mutated_but_WT = int(((labels["task1_tp53_mutated"] == 1) &
                                     (labels["task2_mutation_type"] == "WT")).sum())
    n_conflict_WT_but_typed = int(((labels["task1_tp53_mutated"] == 0) &
                                   (labels["task2_mutation_type"] != "WT")).sum())
    lines.append(f"task1=mutato ma task2=WT (solo mutazioni silent): {n_conflict_mutated_but_WT}")
    lines.append(f"task1=WT ma task2!=WT (mutazioni non-damaging): {n_conflict_WT_but_typed}")
    lines.append("")
    lines.append("Nota: non sono veri conflitti, sono coerenti con la definizione")
    lines.append("diversa delle due label. Task 1 usa la matrice 'damaging' di DepMap,")
    lines.append("Task 2 usa qualsiasi mutazione con effetto sulla proteina.")
    return "\n".join(lines)


# =========================================================
# MAIN
# =========================================================
def main():
    OUTPUT_DIR.mkdir(exist_ok=True)
    tp53 = load_tp53_mutations()
    master = load_master()

    print("\nClassificazione di ogni mutazione...")
    worst = aggregate_per_model(tp53)
    print(f"  Cell lines con almeno 1 mutazione TP53: {len(worst)}")
    print("  Distribuzione della classe 'peggiore' per cell line:")
    print(worst["task2_mutation_class"].value_counts().to_string())

    labels = assemble_labels(master, worst)

    out_cols = [
        "ModelID",
        "task1_tp53_mutated",
        "task2_mutation_type",
        "task2_mutation_type_int",
        "n_tp53_mutations",
        "worst_variant_info",
        "worst_protein_change",
    ]
    out_path = OUTPUT_DIR / "tp53_task2_labels.csv"
    labels[out_cols].to_csv(out_path, index=False)

    summary = build_summary(labels)
    print("\n" + summary)
    (OUTPUT_DIR / "tp53_task2_summary.txt").write_text(summary)

    print("\nFile salvati:")
    print(f"  - {out_path}")
    print(f"  - {OUTPUT_DIR / 'tp53_task2_summary.txt'}")


if __name__ == "__main__":
    main()
