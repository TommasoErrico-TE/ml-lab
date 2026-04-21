# Pipeline del progetto — Predicting TP53 mutation status from gene expression

> **Autore:** Federico Ferrari
> **Corso:** ML Lab (Francesca M. Buffa)
> **Data inizio:** 21 Aprile 2026

> Documento di strategia complessiva. Per lo stato di implementazione attuale
> vedi [`task1.md`](task1.md), [`task2.md`](task2.md) e [`team.md`](team.md).

---

## Obiettivo

Costruire modelli di machine learning che, a partire dal profilo di espressione genica (RNA-seq) di un campione, predicano lo stato mutazionale del gene TP53.

Il gene TP53 codifica la proteina p53, un fattore di trascrizione ("guardian of the genome") che regola cell cycle arrest, apoptosi e DNA repair. E' il gene piu' frequentemente mutato nei tumori umani (~50%). Quando p53 e' mutato, i suoi geni target — il suo "regulon" — dovrebbero mostrare pattern di espressione alterati. L'ipotesi di lavoro e' che questi pattern siano riconoscibili da un modello di ML.

## Due task

- **Task 1 — Classificazione binaria:** *mutant* vs *wild-type (WT)*
- **Task 2 — Classificazione multi-classe (4 classi):**
  - `WT` — nessuna mutazione non-sinonima
  - `Missense` — singola sostituzione amminoacidica
  - `Truncating` — nonsense + frameshift + splice site (proteina tronca, tipicamente LoF)
  - `Inframe` — insertion/deletion che non rompono il reading frame

## Strategia generale (transfer learning)

1. **Blocco principale su CCLE:** sviluppo, training e valutazione interna su linee cellulari (dataset piu' pulito e gestibile).
2. **Blocco finale su TCGA:** transfer del miglior modello a pazienti reali, con fine-tuning su una porzione di TCGA e valutazione su un'altra porzione hold-out.

---

## Stack tecnico

- **Linguaggio:** Python 3.10+
- **Ambiente:** script `.py` al root del repo per il codice riutilizzabile, notebook Jupyter per esplorazione puntuale
- **Librerie principali:**
  - `pandas`, `numpy` — manipolazione dati
  - `scikit-learn` — modelli classici (logistic regression, random forest)
  - `xgboost` — gradient boosting
  - `pytorch` — deep learning
  - `matplotlib`, `seaborn` — visualizzazione
  - `shap` — interpretabilita'
- **Risorse esterne:** DepMap (CCLE), GDC Data Portal (TCGA), DoRothEA (regulon), p53.fr, tp53.cancer.gov (annotazioni funzionali)

---

## Pipeline — dettaglio per step

### Step 1 — Acquisizione dati CCLE

Scaricare da DepMap (https://depmap.org/portal/download/):
- `OmicsExpressionTPMLogp1HumanProteinCodingGenes.csv` — matrice espressione (righe = linee cellulari, colonne = geni, valori = log2(TPM+1))
- `OmicsSomaticMutations.csv` — file mutazioni (MAF-like: una riga per ogni mutazione osservata)
- `OmicsSomaticMutationsMatrixDamaging.csv` — matrice binaria/score di mutazioni damaging (usata dal Task 1)
- `Model.csv` — metadati delle linee cellulari (nome, lineage, tumor type)
- `OmicsProfiles.csv` — metadati dei profili di sequenziamento

Release usata: **DepMap Public 26Q1**. Dettagli in [`../data/README.md`](../data/README.md).

**Output:** master dataset in `output_tp53/tp53_master_dataset.csv`.

### Step 2 — Costruzione etichette

Per ciascuna linea cellulare, estrarre le mutazioni TP53 (`HugoSymbol == "TP53"`) e derivare:

- **`task1_tp53_mutated`** (binaria): 1 se `TP53_damaging_score > 0` nella matrice DepMap damaging, 0 altrimenti. Prodotta da `data.py`.

- **`task2_mutation_type`** (4 classi, WT/Missense/Inframe/Truncating): classificazione a livello di conseguenza proteica leggendo `VariantInfo` dal file MAF. Prodotta da `build_task2_labels.py`. Regola di priorita': **Truncating > Inframe > Missense > WT**.

Decisioni prese e documentate:
- Silent mutations non contano come "Mutant" (finiscono in WT per Task 2).
- Splice site donor/acceptor → Truncating (LoF).
- Stop-gained / frameshift → Truncating.
- Inframe insertion/deletion tenute come classe separata.
- Copy number loss senza mutazione nucleotidica NON conta come Mutant (in prima approssimazione).

Dettagli e distribuzioni in [`task1.md`](task1.md) e [`task2.md`](task2.md).

### Step 3 — Preprocessing delle features

- Rimuovere colonne di indice senza nome (`Unnamed:*`) — fix in `data.py` al commit `03c987b`.
- Rinominare colonne geniche da `HGNC (EntrezID)` a `HGNC`.
- Filtrare `IsDefaultEntryForModel == "Yes"` e deduplicare per ModelID.
- Rimuovere TP53 stesso dalle features (`TP53_expression`) per evitare leakage.
- Rimuovere `TP53_damaging_score` (e' la label).
- (Opzionale) Rimuovere geni a bassa varianza.
- (Opzionale) Standardizzazione per gene (media 0, varianza 1) con `StandardScaler` fittato solo sul train.

Due varianti di feature set da confrontare in futuro:
- **Full transcriptome** — tutti i ~19k geni protein-coding (attuale)
- **TP53 regulon only** — solo i geni regolati da p53 (da DoRothEA, tipicamente 100-300 geni)

### Step 4 — Split train/validation/test su CCLE

- Baseline: 80/20 stratificato singolo split (`train_task1.py`).
- CV: 5-fold stratificato (`train_task1_cv.py`, scenario A).
- **Controllo confounding tessuto:** GroupKFold con `OncotreeLineage` come gruppo (`train_task1_cv.py`, scenario B).
- Baseline solo tessuto: dummy encoding di `OncotreeLineage` → logistic regression (`train_task1_cv.py`, scenario C). Quantifica l'AUC "gratuita" data solo dal tessuto.
- `random_state = 42` per riproducibilita'.

### Step 5 — Modellazione su CCLE (progressione dal semplice al complesso)

**5.a — Baseline: Logistic Regression** ✅ *(Fatto, `train_task1.py`)*
- Accuracy 86.9%, ROC-AUC 0.921
- L1 penalty per feature selection — *TODO*
- Tuning di `C` via CV — *TODO*

**5.b — Random Forest** ✅ *(Fatto, `train_task1.py`)*
- Accuracy 83.3%, ROC-AUC 0.898
- Feature importance: firma p53 canonica (MDM2, CDKN1A, BAX, BBC3, ...)

**5.c — XGBoost** ✅ *(Fatto, `train_task1.py`)*
- Accuracy 88.2%, ROC-AUC 0.928 (miglior modello al momento)
- Feature importance: firma meno canonica, sospetto confounding tessuto

**5.d — Multi-Layer Perceptron (PyTorch)** ⬜ *(TODO, `train_mlp.py`)*
- 2-3 layer fully connected con dropout e batch norm
- Ottimizzatore Adam, early stopping su validation loss
- Confronto con i modelli classici — aspettativa: su dataset di queste dimensioni, i boosted trees tendono a vincere, ma il DL puo' aiutare se usato con smart regularization

**5.e — (Opzionale) Autoencoder + classifier** ⬜
- Autoencoder non supervisionato per ridurre i 19k geni a uno spazio latente di ~64-128 dimensioni
- Classificatore sullo spazio latente
- Utile se il regulon approach non basta

Per il **Task 2** ⬜ *(TODO, `train_task2.py`)*: stessi modelli in versione multi-class. Attenzione al class imbalance (Inframe 1.5%): usare `class_weight='balanced'` o campionamento bilanciato.

### Step 6 — Valutazione su CCLE test set

Metriche per Task 1:
- Accuracy, Precision, Recall, F1 ✅
- AUC-ROC ✅, AUC-PR ⬜
- Confusion matrix ⬜
- Per-tissue AUC (LeaveOneGroupOut) ✅ — vedi `task1.md`

Metriche per Task 2 ⬜:
- Accuracy globale e per classe
- Macro/weighted F1
- Confusion matrix

Analisi degli errori: ci sono tipi cellulari o lineage dove il modello sbaglia di piu'? ✅ *(Pancreas e' l'outlier principale — AUC 0.586)*

### Step 7 — Interpretazione biologica

Estrarre i geni piu' importanti per il modello migliore:
- Coefficienti (logistic regression) ⬜
- Feature importance (tree-based) ✅
- SHAP values (per qualunque modello) ⬜

Confrontare con:
- **Target noti di p53** (DoRothEA, TRRUST): MDM2, CDKN1A/p21, BAX, PUMA/BBC3, GADD45, ecc. ✅ *(RF trova questi)*
- **Pathway analysis** ⬜ — i top geni si arricchiscono in pathway p53-related?
- **Database TP53-specifici** ⬜ — p53.fr, tp53.cancer.gov

Se il modello sta imparando biologia corretta → ottimo segno da riportare nella relazione.

### — fine blocco CCLE —

### Step 8 — Acquisizione e harmonization TCGA ⬜

Download da GDC (https://portal.gdc.cancer.gov/):
- RNA-seq expression (tipicamente STAR-counts o FPKM-UQ, da decidere)
- MAF files per le mutazioni TP53 (aggregate MAF o per-study)

**Challenge chiave: harmonization CCLE ↔ TCGA**

CCLE e TCGA usano pipeline diverse di quantificazione. Per poter trasferire un modello dobbiamo mettere i dati nello stesso spazio:
- Intersecare i geni comuni
- Assicurarsi che l'unita' di misura sia confrontabile (log2-TPM per entrambi)
- Applicare correzione batch (es. ComBat via `neuroCombat` o `pyComBat`) per rimuovere effetti tecnici di pipeline
- Ri-standardizzare le features usando lo scaler fittato su CCLE training (o ri-fittare se ComBat ha fatto gia' il grosso)

Stesse regole dello Step 2 per costruire le etichette TP53 su TCGA.

### Step 9 — Zero-shot evaluation su TCGA ⬜

Applicare il modello CCLE direttamente a TCGA, senza re-training.
Baseline di generalizzazione del modello.

### Step 10 — Fine-tuning su TCGA ⬜

Split TCGA in:
- **TCGA-finetune** (~40-50%)
- **TCGA-holdout** (~50-60%)

Tecnica di fine-tuning dipende dal modello:
- **MLP PyTorch:** sblocco solo gli ultimi layer, learning rate ridotto (1e-4 o meno), pochi epochs, early stopping
- **XGBoost:** `xgb.train(..., xgb_model=base_model)` per continuare il boosting
- **Logistic Regression:** warm start con `model.coef_` e `model.intercept_` dal modello CCLE, poi fit su TCGA-finetune

Stratificare per tumor type nello split, per evitare che il modello veda solo alcuni tipi di tumore durante il fine-tuning.

### Step 11 — Valutazione finale su TCGA hold-out ⬜

Confronto diretto:
- Zero-shot (Step 9)
- Fine-tuned (Step 10)

Metriche per tumor type — il modello funziona meglio su alcuni tipi di tumore?

### Step 12 — Interpretazione finale e confronto ⬜

- I geni importanti in CCLE sono gli stessi importanti in TCGA post fine-tuning?
- Quali differenze biologiche emergono?
- Il modello cattura hotspot p53 noti (R175H, R248W, R273H)?
- Confronto con IARC TP53 Database (tp53.cancer.gov) e p53.fr per annotazione funzionale degli hotspot identificati.

### Step 13 — Report / presentazione finale ⬜

Struttura suggerita:
1. Motivazione biologica e task
2. Dati e scelte di etichettatura (Task 1 + Task 2)
3. Pipeline di preprocessing
4. Confronto modelli su CCLE
5. Controllo confounding tessuto (questo e' un pezzo forte della nostra analisi)
6. Interpretazione biologica
7. Transfer a TCGA (zero-shot vs fine-tuned)
8. Conclusioni e limitazioni

---

## Struttura del progetto (reale, aggiornata 21/04/2026)

```
ml-lab/
├── README.md                         # overview del repo
├── requirements.txt                  # dipendenze Python
├── data.py                           # [Tommaso] build master dataset + Task 1 label
├── train_task1.py                    # [Edoardo] baseline LR + RF + XGB su Task 1
├── train_task1_cv.py                 # [Federico] CV + controllo confounding tessuto
├── build_task2_labels.py             # [Federico] costruzione label Task 2 (4 classi)
├── check_differences.py              # [Edoardo] diagnostica MAF vs Matrix damaging
├── data/
│   ├── README.md                     # descrizione dei 5 file DepMap 26Q1
│   └── *.csv                         # raw data (gitignored)
├── output_tp53/
│   ├── README.md                     # descrizione output generati
│   └── *.csv                         # generated data (gitignored)
└── docs/
    ├── pipeline.md                   # questo documento (strategia completa)
    ├── task1.md                      # Task 1 — definizione, risultati, open issues
    ├── task2.md                      # Task 2 — tassonomia, label, open issues
    └── team.md                       # authorship, regole di convivenza
```

Nota: la struttura e' piatta (script al root, non in `src/`, non in `notebooks/`) perche' il progetto e' piccolo e in fase iniziale. Si potra' refattorizzare in `src/` se gli script diventano grandi o se emergono duplicazioni significative.

---

## Decisioni ancora aperte (da affrontare quando pertinenti)

- Deadline del progetto e formato richiesto di consegna (codice solo? report scritto? presentazione?)
- Quale fonte specifica per il regulon TP53 (DoRothEA level A-B-C? TRRUST? combinazione?)
- Se includere anche tumor type come feature/covariate o come stratificazione (attualmente stratificazione via GroupKFold)
- Correzione batch con ComBat vs altri metodi
- Usare Colab o ambiente locale (attualmente locale)
- Se collassare la classe Inframe dentro Missense per il Task 2 (24 sample sono pochi per una classe a se')

---

## Log delle decisioni prese

- **21/04/2026:** Dataset principale: CCLE (DepMap 26Q1). Transfer finale su TCGA.
- **21/04/2026:** Task 2 raggruppato in 4 classi (WT / Missense / Truncating / Inframe) a livello di conseguenza proteica, con regola di priorita' "worst-wins".
- **21/04/2026:** Stack: Python + Jupyter + scikit-learn + xgboost + PyTorch.
- **21/04/2026:** Baseline Task 1 completata: LogReg 0.921, RF 0.898, XGBoost 0.928 (ROC-AUC, split 80/20).
- **21/04/2026:** Fix bug `Unnamed: 0` in `data.py` (commit `03c987b`): da 19215 a 19214 feature.
- **21/04/2026:** CV multi-scenario completata (`train_task1_cv.py`): AUC standard 0.898, out-of-tissue 0.849, tissue-only baseline 0.753. Il segnale "vera firma p53" oltre il tessuto e' ~0.10 AUC. Pancreas e' l'outlier (AUC 0.586).
- **21/04/2026:** Label Task 2 prodotte (`build_task2_labels.py`): 654 WT / 628 Missense / 337 Truncating / 24 Inframe su 1643 cell lines.
