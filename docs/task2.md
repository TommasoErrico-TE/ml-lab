# Task 2 — Classificazione multi-classe del tipo di mutazione TP53

## Definizione

Dato il profilo di espressione di una cell line, predire **il tipo di mutazione
TP53** presente. 4 classi.

## Tassonomia scelta

Abbiamo scelto una tassonomia a livello di **conseguenza proteica**, che e'
lo standard in oncologia clinica e in tutti i principali database di
mutazioni (DepMap, IARC, COSMIC, ClinVar):

| Classe | Encoding | Definizione |
|---|---|---|
| `WT` | 0 | Nessuna mutazione funzionale. Include anche cell lines con solo mutazioni silenti (synonymous) o UTR/intronic isolate. |
| `Missense` | 1 | Cambio di un singolo amminoacido. La proteina viene prodotta a lunghezza normale ma con una sostituzione. Esempi: `p.R175H`, `p.R273H`, `p.R248Q` (hotspot tipici di TP53). |
| `Inframe` | 2 | Inserzione o delezione di un multiplo di 3 basi. La proteina perde o guadagna uno o pochi amminoacidi ma mantiene il reading frame. |
| `Truncating` | 3 | Mutazioni che troncano la proteina: nonsense (`stop_gained`), frameshift, disruzione dello splice site (donor/acceptor). Tipicamente loss-of-function. |

### Regola di priorita'

Una stessa cell line puo' avere piu' mutazioni TP53. In questo caso assegniamo
la classe *piu' grave* (rank piu' alto):

```
Truncating (3) > Inframe (2) > Missense (1) > WT (0)
```

Motivazione: se una cell line ha sia una missense che una truncating, il
fenotipo funzionale e' guidato dalla truncating (proteina distrutta) e non
dalla missense (proteina alterata ma prodotta). E' la convenzione standard
in MAF-level annotation.

### Confronto con la tassonomia Berkeley

Il link del corso (https://evolution.berkeley.edu/dna-and-mutations/types-of-mutations/)
definisce 4 categorie a **livello di DNA**:

- Substitution
- Insertion
- Deletion
- Frameshift

La nostra tassonomia non corrisponde letteralmente, ma mappa su di essa:

| Nostra classe | Corrisponde a (Berkeley) |
|---|---|
| Missense | Substitution → singolo AA change |
| Truncating | Substitution → nonsense **+** Frameshift **+** splice site |
| Inframe | Insertion / Deletion non-frameshift |
| WT | nessuna |

Abbiamo scelto la tassonomia proteica per due motivi:

1. **Biologicamente piu' significativa**: per p53 conta l'effetto funzionale
   (la proteina funziona, e' alterata, o e' distrutta), non il meccanismo
   DNA sottostante.
2. **Class imbalance piu' gestibile**: con la tassonomia Berkeley, Substitution
   raccoglierebbe circa l'85% delle mutazioni osservate, rendendo le altre
   classi quasi vuote.

Queste motivazioni vanno scritte esplicitamente nella relazione.

## Script coinvolti

- **`build_task2_labels.py`** — costruisce le label Task 2 da
  `data/OmicsSomaticMutations.csv` (file MAF-like) + master dataset.
  Output: `output_tp53/tp53_task2_labels.csv`.
- *(In arrivo)* **`train_task2.py`** — multinomial logistic + RF + XGBoost per
  la classificazione a 4 classi.

## Pipeline delle label

1. Leggi `OmicsSomaticMutations.csv` (1.17M righe, 69 colonne).
2. Filtra `HugoSymbol == "TP53"` → 2138 mutazioni.
3. Filtra `IsDefaultEntryForModel == Yes` → 1324 mutazioni su 1181 cell lines.
4. Per ogni mutazione, classifica in `Truncating` / `Inframe` / `Missense` /
   `Silent` leggendo `VariantInfo` (formato Sequence Ontology).
5. Per ogni ModelID, aggrega tenendo la classe con rank piu' alto.
6. Merge col master dataset (1643 cell lines): i ModelID senza mutazione TP53
   (o con solo silenti) diventano `WT`.

## Distribuzione risultante (release 26Q1)

| Classe | n | % |
|---|---|---|
| WT | 654 | 39.8% |
| Missense | 628 | 38.2% |
| Truncating | 337 | 20.5% |
| Inframe | 24 | 1.5% |
| **Totale** | **1643** | |

Proporzioni in linea con letteratura: nei tumori, ~75% delle mutazioni TP53
sono missense, ~25% truncating, una coda piccola di inframe.

**Class imbalance:** la classe `Inframe` e' rara (1.5%). In training useremo
`class_weight="balanced"` e/o report per-class della F1 per non mascherare
il problema.

## Coerenza Task 1 ↔ Task 2

Tabella incrociata:

```
task2            Inframe  Missense  Truncating   WT   Totale
task1
  0 (WT)              4        20           0   646      670
  1 (Mutato)         20       608         337     8      973
```

- Tutte le 337 `Truncating` sono `task1=1`. Perfettamente coerente: le
  troncanti sono sempre damaging.
- **8 cell lines** con `task1=1` ma `task2=WT` (solo mutazioni silent nel file
  MAF). Edge case, 0.5%.
- **24 cell lines** con `task1=0` ma `task2!=WT` (missense/inframe non
  classificate come damaging dalla matrix DepMap). Edge case, 1.5%.

Questi 32 conflitti apparenti (2% del totale) sono coerenti con la diversa
definizione dei due label: Task 1 usa il filtro damaging di DepMap, Task 2
usa qualsiasi mutazione con effetto sulla proteina.

## Fonti di validazione esterna

Oltre a DepMap, esistono database TP53-specifici curati manualmente:

- **TP53 Database NCI/IARC** — https://tp53.cancer.gov/get_tp53data
  Ha un dataset dedicato "TP53 variant status of human cell-lines" che
  puo' servire come gold standard indipendente. Uso futuro:
  confronto con le nostre label per stimare l'accuracy della nostra
  classificazione.
- **p53.fr** — https://p53.fr/ (successore del database di Thierry Soussa,
  Jean-Luc Beroud et al.). Contiene oltre 80k mutanti TP53 con annotazioni
  funzionali (LoF, Dominant-Negative, Gain-of-Function).

Queste fonti saranno usate anche per **l'interpretazione** dei risultati: per
ogni hotspot p53 che il modello identifica (es. R175H, R248W, R273H),
possiamo andare a vedere la sua annotazione funzionale in IARC/p53.fr e
discuterla nella relazione.

## Open issues

- [ ] Training effettivo (4-class) non ancora iniziato
- [ ] Valutazione: accuracy globale + per-class F1 + confusion matrix
- [ ] Cross-validation stratificata (la classe Inframe e' rara)
- [ ] Validazione con dataset cell lines di tp53.cancer.gov
- [ ] Considerare 3-classe alternativa collassando Inframe dentro Missense
  se l'Inframe risulta impossibile da imparare con 24 sample
- [ ] Interpretabilita': quali geni separano Missense da Truncating? Ci si
  aspetta che le truncanti abbiano una firma piu' forte (LoF completa),
  le missense piu' variabile (gain of function per alcuni hotspot).

## Come si usano le label

Dal proprio script di training Task 2:

```python
import pandas as pd
master = pd.read_csv("output_tp53/tp53_master_dataset.csv")
task2 = pd.read_csv("output_tp53/tp53_task2_labels.csv")

df = master.merge(
    task2[["ModelID", "task2_mutation_type", "task2_mutation_type_int"]],
    on="ModelID",
    how="inner",
)

# target: task2_mutation_type_int (0/1/2/3)
# features: stesse ~19k colonne di espressione del Task 1
```

La stessa X (feature matrix) del Task 1 e' riusabile: cambia solo la y.
