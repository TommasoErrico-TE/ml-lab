# Task 1 — Classificazione binaria *mutated* vs *WT*

## Definizione

Dato il profilo di espressione di una cell line (log2(TPM+1) per ~19k geni
protein-coding), predire se TP53 e' **mutato** (presenza di almeno una
mutazione *damaging* nel locus TP53) oppure **wild-type**.

## Label

La label `task1_tp53_mutated` (0/1) e' calcolata dentro `data.py` con la regola:

```python
task1_tp53_mutated = (TP53_damaging_score > 0).astype(int)
```

dove `TP53_damaging_score` viene dalla colonna `TP53` del file
`OmicsSomaticMutationsMatrixDamaging.csv` (0 = WT, 1 = monoallelica,
2 = biallelica). DepMap applica internamente il filtro per "damaging".

**Distribuzione della label (release 26Q1):**

| | n | % |
|---|---|---|
| WT (0) | 670 | 40.8% |
| Mutato (1) | 973 | 59.2% |
| **Totale** | **1643** | |

Di cui 113 mono-allelici e 860 bi-allelici — la stragrande maggioranza dei
mutati ha hit entrambi gli alleli, coerente con TP53 come tumor suppressor che
per essere inattivato richiede LOH o seconda mutazione.

## Script coinvolti

- **`data.py`** — costruisce il master dataset (feature + label + metadata).
  Input: i 3 CSV di DepMap. Output: `output_tp53/tp53_master_dataset.csv`.
- **`train_task1.py`** — baseline con 3 modelli classici su singolo split 80/20.
  Output: `output_tp53/task1_model_comparison.csv` + feature importance.
- **`train_task1_cv.py`** — cross-validation stratificata + controllo
  confounding tessuto. Output: `output_tp53/task1_cv_comparison.csv` +
  `task1_per_tissue_auc.csv`.

## Feature

~19214 geni protein-coding, valori log2(TPM+1). Escluse esplicitamente:

- `TP53_expression` (livello di espressione di TP53 stesso — esclusa per evitare
  leakage diretto)
- `TP53_damaging_score` (e' la label stessa)
- Tutte le colonne di metadata (ModelID, SequencingID, CellLineName, lineage,
  ecc.)

**Nota sul bug `Unnamed: 0`:** il file di espressione CCLE ha una prima colonna
senza nome (indice numerico 0,1,2,...) che pandas legge come `Unnamed: 0`.
Originariamente finiva come feature, introducendo rumore. Fixato nel commit
`03c987b` droppando qualsiasi colonna che inizi con `Unnamed` subito dopo la
`read_csv`.

## Risultati

### Baseline (train_task1.py, split singolo 80/20)

| Modello | Accuracy | ROC-AUC | F1 (mut) |
|---|---|---|---|
| XGBoost | 88.2% | **0.928** | 0.904 |
| Logistic Regression | 86.9% | 0.921 | 0.889 |
| Random Forest | 83.3% | 0.898 | 0.871 |

**Observazione:** XGBoost ha leggermente il meglio. Random Forest e' sbilanciato
verso il "mutato" (recall WT = 66%, recall mut = 95%), XGBoost molto piu'
bilanciato (79% / 94%).

### Feature importance

**Random Forest — firma p53 da manuale:**

```
MDM2, EDA2R, ZMAT3, AEN, CDKN1A, FDXR, SPATA18, SESN1,
PHLDA3, DDB2, BBC3, TNFRSF10B, RPS27L, BAX, RRM2B, PPM1D
```

Sono tutti bersagli diretti di p53 (+ PPM1D, la fosfatasi del feedback loop
p53/MDM2). Il modello ha imparato la firma trascrizionale canonica di p53:
quando p53 e' mutato non riesce ad attivare questi geni, e la loro bassa
espressione e' il segnale che il modello legge.

**XGBoost — firma meno canonica:**

```
EDA2R, OTX1, HEATR4, C4orf46, CADM3, CPSF4, GNB2, GRHL2, GAB1,
CFAP299, KLHL36, ESRP2, ACTR8, CARD14, KIAA1549L, CENPS, ZMAT3, PRMT5
```

Qui oltre a EDA2R e ZMAT3 (p53 target) compaiono geni di lineage (OTX1 neurale,
GRHL2 epiteliale, CADM3/ESRP2 lineage-specifici). Sospetto di confounding
di tessuto.

### Cross-validation + controllo tessuto (train_task1_cv.py)

Tre scenari a confronto (Logistic Regression, 5-fold):

| Scenario | Mean AUC |
|---|---|
| A) StratifiedKFold standard | **0.898 ± 0.017** |
| B) GroupKFold per tessuto (test su tessuti mai visti) | **0.849 ± 0.056** |
| C) Tissue-only baseline (solo dummy tessuto) | **0.753 ± 0.028** |

**Lettura:**

- **Drop A → B = 0.049.** Piccolo. Anche quando si nascondono i tessuti di
  test al training, il modello mantiene gran parte del suo potere
  predittivo. C'e' quindi una vera firma p53 che generalizza a tessuti nuovi.
- **Baseline C = 0.753.** Il solo tessuto (one-hot encoding di
  `OncotreeLineage`) da gia' 0.75 di AUC. Questo e' il livello sotto cui
  non si puo' parlare di merito del modello.
- **Valore aggiunto del modello sopra il baseline:** +0.145 (in CV standard)
  e +0.096 (in out-of-tissue CV). Quei ~0.10 AUC sono la "vera" firma p53.

### Per-tessuto (LeaveOneGroupOut)

Tessuti dove il modello funziona (AUC > 0.90): Skin, Bladder, Uterus, Ovary,
CNS, Thyroid, Bowel.

Tessuti problematici:

- **Pancreas: AUC 0.586** — unico vero outlier. 54 cell lines, 80% mutate,
  AUC quasi casuale. Possibile spiegazione: il pancreas ha mutazioni KRAS
  sistematiche che dominano il trascrittoma e mascherano la firma p53.
- **Lung (0.74), Head&Neck (0.73)** — mutation rate > 85%, poca varianza
  per separare. Artefatto statistico piu' che difetto del modello.

## Open issues

- [ ] Logistic Regression con **L1 penalty** per feature selection esplicita
  (oggi `penalty="l2"` default)
- [ ] Hyperparameter tuning (oggi hard-coded)
- [ ] MLP in PyTorch (non ancora implementato)
- [ ] Bilanciamento per tessuto in fase di training (sampling stratificato o
  pesi) per ridurre il confounding
- [ ] Ablation: togliere le feature che escono come top di XGBoost
  (OTX1, GRHL2, ecc.) e vedere se la differenza A−B si restringe ulteriormente
- [ ] Indagine specifica sul Pancreas (perche' crolla?)

## Come citare i risultati nella relazione

Una formulazione onesta: "Il nostro modello (Logistic Regression) raggiunge
AUC 0.90 in cross-validation 5-fold sul dataset CCLE. Contro un baseline
'solo tessuto' che gia' fornisce AUC 0.75, il segnale aggiuntivo e' di ~0.15.
Quando si testa su tessuti out-of-training, il modello mantiene AUC 0.85,
dimostrando che la firma di espressione appresa e' associata a p53 e non solo
al tessuto di origine."

I geni piu' importanti del Random Forest (MDM2, CDKN1A, BAX, BBC3, ecc.) sono
bersagli diretti del programma trascrizionale di p53, a supporto della
validita' biologica del modello.
