# ml-lab — Predicting TP53 mutation status from gene expression

Progetto per il corso **Machine Learning Lab** (prof. Francesca M. Buffa), A.A. 2025-2026.

Team: **Tommaso Errico**, **Edoardo Paccagnella**, **Federico Ferrari**.

## Obiettivo

Costruire modelli di ML che, a partire dal profilo di espressione genica (RNA-seq,
log2(TPM+1)) di un campione, predicano lo stato mutazionale del gene **TP53**.

TP53 è il tumor suppressor più frequentemente mutato nei tumori umani (~50% dei
casi). Quando p53 è mutato, i suoi geni target — il suo "regulon" — mostrano
pattern di espressione alterati. L'ipotesi è che questi pattern siano
riconoscibili da un modello di ML.

Il progetto è diviso in due task:

- **Task 1** — classificazione binaria: `mutated` vs `wild-type`
- **Task 2** — classificazione a 4 classi: `WT` / `Missense` / `Inframe` / `Truncating`

La strategia generale prevede sviluppo su **CCLE** (cell lines da DepMap) e poi
transfer a **TCGA** (pazienti reali) con fine-tuning. Vedi
[`docs/pipeline.md`](docs/pipeline.md) per tutti i dettagli.

## Quick start

### 1. Ambiente

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Scaricare i dati

I dati DepMap non sono versionati (sono troppo pesanti, ~1 GB totali). Per
scaricarli vedi [`data/README.md`](data/README.md).

### 3. Costruire il master dataset

```bash
python data.py
```

Genera `output_tp53/tp53_master_dataset.csv` (1643 cell lines × ~19k feature).

### 4. Lanciare i modelli

```bash
# Task 1 - baseline
python train_task1.py

# Task 1 - cross-validation + controllo confounding tessuto
python train_task1_cv.py

# Task 2 - costruzione label 4-class
python build_task2_labels.py
```

## Struttura del repo

```
ml-lab/
├── README.md                      # questo file
├── LICENSE
├── requirements.txt               # dipendenze Python
├── .gitignore
│
├── docs/                          # documentazione
│   ├── pipeline.md                # strategia completa del progetto
│   ├── task1.md                   # Task 1 - dettagli e risultati
│   ├── task2.md                   # Task 2 - dettagli e risultati
│   └── team.md                    # autorship degli script
│
├── data/                          # raw data da DepMap (gitignored)
│   └── README.md                  # come scaricare i file
│
├── output_tp53/                   # output generati dagli script
│   └── README.md                  # descrizione dei file generati
│
│ ── SCRIPTS ─────────────────────────────────────
│
├── data.py                        # [Task 1+2] costruisce master dataset
├── train_task1.py                 # [Task 1] baseline: LR + RF + XGBoost
├── train_task1_cv.py              # [Task 1] CV stratificata + per tessuto
├── check_differences.py           # [Task 1+2] confronta MAF vs Matrix
└── build_task2_labels.py          # [Task 2] costruisce label a 4 classi
```

## Separazione Task 1 / Task 2

La divisione tra i due task è logicamente questa:

| Fase | Task 1 | Task 2 |
|---|---|---|
| Master dataset | `data.py` (condiviso) | `data.py` (condiviso) |
| Costruzione label | automatica in `data.py` | `build_task2_labels.py` |
| File label | `task1_tp53_mutated` dentro master | `output_tp53/tp53_task2_labels.csv` |
| Training | `train_task1.py`, `train_task1_cv.py` | *(in arrivo: `train_task2.py`)* |

Il **master dataset è condiviso**: stesso insieme di cell lines (1643) e stesse
feature di espressione (~19k geni). Cambiano solo le label. Quindi il merge per
il Task 2 è: `tp53_master_dataset.csv` ⟕ `tp53_task2_labels.csv` su `ModelID`,
e si predice `task2_mutation_type_int` invece di `task1_tp53_mutated`.

Per dettagli vedi [`docs/task1.md`](docs/task1.md) e
[`docs/task2.md`](docs/task2.md).

## Status

- [x] Download dati DepMap 26Q1
- [x] Master dataset (1643 cell lines × 19214 feature)
- [x] Task 1 baseline (Logistic, Random Forest, XGBoost) — AUC ~0.90
- [x] Task 1 con cross-validation stratificata per tessuto
- [x] Task 2 label construction (4 classi)
- [ ] Task 2 training
- [ ] Cross-validation + tuning iperparametri
- [ ] MLP in PyTorch
- [ ] Download TCGA
- [ ] Harmonization CCLE ↔ TCGA (ComBat)
- [ ] Transfer learning + fine-tuning
- [ ] Interpretabilità (SHAP) + mappatura hotspot p53

## Licenza

Vedi [LICENSE](LICENSE) (MIT).

## Risorse esterne

- **DepMap Portal** — https://depmap.org/portal/
- **TCGA / GDC** — https://portal.gdc.cancer.gov/
- **p53.fr** — https://p53.fr/ (database TP53, successore di Thierry Soussa)
- **TP53 Database (NCI/IARC)** — https://tp53.cancer.gov/
- **Berkeley mutation types** — https://evolution.berkeley.edu/dna-and-mutations/types-of-mutations/
