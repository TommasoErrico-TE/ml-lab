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

# Task 1 baseline (LR + RF + XGBoost on 80/20 split)
python scripts/train_task1.py

# Task 1 with stratified CV + tissue control (GroupKFold)
python scripts/train_task1_cv.py

# Task 2 multi-class (both taxonomies)
python scripts/train_task2.py

# MLP (Task 1 + Task 2)
python scripts/compute_mlp.py
```

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
│   ├── datasets/             #   merged master + Task 2 labels
│   ├── task1/                #   Task 1 model results, feature importance, MLP, correlations
│   └── task2/                #   Task 2 results, confusion matrices, MLP
│
└── scripts/                  # pipeline code
    ├── data.py                  # master dataset + Task 1 label
    ├── build_task2_labels.py    # Task 2 labels (coding consequence + DNA-level)
    ├── train_task1.py           # Task 1 baseline: LR + RF + XGBoost
    ├── train_task1_cv.py        # Task 1 5-fold CV + GroupKFold-by-tissue
    ├── train_task2.py           # Task 2 multi-class on both taxonomies
    └── compute_mlp.py           # PyTorch MLP for Task 1 and Task 2
```

## Data

The project uses the **DepMap Public 26Q1** release of the Cancer Cell
Line Encyclopedia (CCLE). The five raw files required by `scripts/data.py`
and `scripts/build_task2_labels.py` are listed in
[`data/README.md`](data/README.md).

## License

See [LICENSE](LICENSE) (MIT).
