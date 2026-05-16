# Predicting TP53 mutation status from gene expression

Course project — **Machine Learning Lab** (Prof. Francesca M. Buffa), A.Y. 2025–2026.

Team: Tommaso Errico, Edoardo Paccagnella, Federico Ferrari, Rebecca Rinero.

## Objective

Train machine learning models that, starting from a gene-expression profile
(RNA-seq, log2(TPM+1)), predict the mutation status of the **TP53** gene.

TP53 is the most frequently mutated tumor suppressor in human cancers
(~50% of cases). When p53 is mutated, its target genes — the p53 *regulon*
— show altered expression patterns. The hypothesis is that these patterns
are recognizable by an ML model.

The project covers two tasks:

- **Task 1** — binary classification: `mutated` vs `wild-type`
- **Task 2** — multi-class classification of mutation type, comparing two
  parallel taxonomies (coding consequence vs DNA-level)

## Deliverable

The deliverable is the Jupyter notebook **`ML_project.ipynb`**, together
with its rendered HTML version **`ML_project.html`**. All analyses,
results and conclusions are documented there.

## Quick start

### 1. Environment

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Download the data

The DepMap CSVs are not versioned (~1 GB total). See
[`data/README.md`](data/README.md) for how to download them.

### 3. Reproduce the pipeline (optional)

All scripts are run from the repo root.

```bash
# Build the master dataset (Task 1 label included)
python scripts/data.py

# Build the Task 2 labels (both taxonomies)
python scripts/build_task2_labels.py

# Task 1 — train/val/test design + split ratio sweep
python scripts/train_task1_splits.py

# Task 1 — k-fold sweep (3, 5, 10)
python scripts/train_task1_kfold_sweep.py

# Task 1 — RandomizedSearchCV su LogReg/RF/XGBoost (heavy)
python scripts/train_task1_hpsearch.py --n-iter 30 --top-k 2000

# Task 1 — feature importance from RF and XGBoost
python scripts/train_task1.py

# Task 1 — 5-fold CV + GroupKFold-by-tissue
python scripts/train_task1_cv.py

# Task 2 multi-class (both taxonomies)
python scripts/train_task2.py

# Per-gene Spearman + Pearson correlation with TP53 status
python scripts/gene_correlation.py

# MLP (Task 1 + Task 2)  - meglio su HPC con GPU
python scripts/compute_mlp.py

# TCGA extension
python scripts/download_tcga.py
python scripts/build_tcga_master.py
python scripts/transfer_ccle_to_tcga.py
python scripts/train_tcga.py --n-iter 30 --top-k 2000   # heavy, meglio su HPC
python scripts/compute_mlp_tcga.py                       # idem
```

### HPC (Bocconi)

Per gli step pesanti (HP search, training TCGA, MLP) sono pronti job SLURM
in `slurm/jobs/`:

```bash
sbatch slurm/jobs/hp_search_ccle.slurm
sbatch slurm/jobs/train_tcga.slurm
sbatch slurm/jobs/mlp.slurm
```

Workflow completo (clone, librerie, rsync dati, submit) in
[`slurm/README.md`](slurm/README.md). Niente venv: si installano le
librerie minime con `pip install --user`.

The notebook loads pre-computed results from `output_tp53/` so it executes
in seconds without re-running the heavy training. Set `RUN_HEAVY = True`
in the setup cell to recompute everything from scratch.

## Repository structure

```
ml-lab/
├── ML_project.ipynb          # main deliverable (notebook)
├── ML_project.html           # rendered HTML export
├── README.md
├── LICENSE
├── requirements.txt
├── .gitignore
│
├── data/                     # raw DepMap CSVs (gitignored, see data/README.md)
│
├── output_tp53/              # generated outputs consumed by the notebook
│   ├── datasets/             #   master CCLE + Task 2 labels + master TCGA
│   ├── task1/                #   Task 1 results (splits, k-fold, HP search, importances, MLP, correlations)
│   ├── task2/                #   Task 2 results, confusion matrices, MLP
│   └── tcga/                 #   TCGA transfer + retraining + MLP
│
└── scripts/                  # pipeline code
    ├── data.py                       # CCLE master dataset + Task 1 label
    ├── build_task2_labels.py         # Task 2 labels (coding + DNA-level)
    ├── train_task1.py                # Task 1 80/20 reference (feature importance)
    ├── train_task1_splits.py         # Task 1 60/20/20 + split ratio sweep   [P1]
    ├── train_task1_kfold_sweep.py    # k=3/5/10 comparison                   [P2]
    ├── train_task1_hpsearch.py       # RandomizedSearchCV (LR/RF/XGB)        [P3]
    ├── train_task1_cv.py             # 5-fold CV + GroupKFold-by-tissue
    ├── train_task2.py                # Task 2 multi-class on both taxonomies
    ├── gene_correlation.py           # Spearman + Pearson per-gene           [P4]
    ├── compute_mlp.py                # PyTorch MLP CCLE Task 1 + Task 2
    ├── download_tcga.py              # Scarica TCGA Pan-Cancer da Xena       [P5]
    ├── build_tcga_master.py          # TCGA master (sample x gene + label)   [P5]
    ├── transfer_ccle_to_tcga.py      # Train CCLE -> test TCGA               [P5]
    ├── train_tcga.py                 # Retrain pipeline on TCGA              [P5]
    ├── compute_mlp_tcga.py           # PyTorch MLP TCGA Task 1               [P5]

slurm/                               # SLURM jobs per Bocconi HPC
├── README.md                        # workflow completo (clone, rsync, submit)
├── jobs/
│   ├── hp_search_ccle.slurm
│   ├── train_tcga.slurm
│   └── mlp.slurm
└── logs/                            # output dei job (gitignored)
```

## Data

The project uses two cohorts:

- **CCLE** — DepMap Public 26Q1 release of the Cancer Cell Line
  Encyclopedia. The five raw files required by `scripts/data.py`
  and `scripts/build_task2_labels.py` are listed in
  [`data/README.md`](data/README.md).
- **TCGA Pan-Cancer Atlas** — from the UCSC Xena Pan-Cancer hub:
  - `EB++AdjustPANCAN_IlluminaHiSeq_RNASeqV2.geneExp.xena.gz` (batch-corrected RNA-seq)
  - `mc3.v0.2.8.PUBLIC.xena.gz` (somatic mutations, MC3)
  - `Survival_SupplementalTable_S1_20171025_xena_sp` (cancer type per sample)
  
  Scaricabili automaticamente con `python scripts/download_tcga.py` (~395 MB).

## License

See [LICENSE](LICENSE) (MIT).
