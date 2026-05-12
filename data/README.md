# `data/` — raw data da DepMap 26Q1

Questa cartella contiene i file originali scaricati da
[DepMap Portal](https://depmap.org/portal/download/) per il release
**DepMap Public 26Q1** (rilasciato 2026-04-01). I file sono **gitignored**
perche' troppo pesanti (~1.1 GB totali).

## File necessari

Per far girare `data.py`, `train_task1.py`, `train_task1_cv.py`,
`build_task2_labels.py` e `check_differences.py` servono i seguenti 5 file:

| File | Dimensione | Contenuto |
|---|---|---|
| `OmicsExpressionTPMLogp1HumanProteinCodingGenes.csv` | ~290 MB | Matrice espressione: righe = cell line, colonne = geni (protein-coding), valori = log2(TPM+1). Colonne con formato `HGNC (EntrezID)`. Include anche 5 colonne di metadata (`SequencingID`, `ModelConditionID`, `ModelID`, `IsDefaultEntryForMC`, `IsDefaultEntryForModel`). |
| `OmicsSomaticMutationsMatrixDamaging.csv` | ~230 MB | Matrice binaria/score di mutazioni *damaging*: righe = cell line, colonne = geni. Valori: 0 (WT), 1 (mono-allelica), 2 (bi-allelica). Usato dal Task 1 per la label `task1_tp53_mutated`. |
| `OmicsSomaticMutations.csv` | ~550 MB | File MAF-like: **una riga per ogni mutazione** osservata. Colonne chiave: `HugoSymbol`, `VariantType`, `VariantInfo`, `DNAChange`, `ProteinChange`, `MolecularConsequence`, `VepImpact`, `LikelyLoF`. Usato dal Task 2 per determinare il *tipo* di mutazione TP53. |
| `Model.csv` | ~0.7 MB | Metadati delle cell lines: `ModelID`, `CellLineName`, `OncotreeLineage`, `OncotreePrimaryDisease`, `OncotreeSubtype`, `Sex`, `AgeCategory`, `PrimaryOrMetastasis`, ecc. |
| `OmicsProfiles.csv` | ~0.6 MB | Metadati dei profili di sequenziamento (relazione `SequencingID` ↔ `ModelID`). Caricato da `data.py` ma non strettamente necessario per il merge finale. |

`Model.csv` e `OmicsProfiles.csv` sono abbastanza piccoli da essere committati;
i 3 grossi sono gitignored.

## Come scaricare

Tre opzioni, in ordine di comodita':

### Opzione 1 — download manuale dal portale

1. Andare su https://depmap.org/portal/download/all/
2. Selezionare **DepMap Public 26Q1** come release
3. Scaricare i 5 file elencati sopra
4. Metterli dentro `data/`

### Opzione 2 — script automatico

Esiste uno script pronto che legge l'indice dei file via API DepMap e scarica
solo quelli che servono. Nel repo non e' ancora integrato; il codice completo
e' in `../scripts/download_depmap.py` nel workspace temporaneo. Da integrare
quando serve.

### Opzione 3 — da un'altra release

Se 26Q1 non e' piu' disponibile, **ogni release successiva** dovrebbe contenere
gli stessi 5 file con lo stesso schema. I risultati possono variare leggermente
ma la pipeline funziona identica.

## Schema dei file (dettaglio)

### `OmicsExpressionTPMLogp1HumanProteinCodingGenes.csv`

```
Unnamed: 0,SequencingID,ModelConditionID,ModelID,IsDefaultEntryForMC,IsDefaultEntryForModel,TSPAN6 (7105),TNMD (64102),...
0,CDS-010xbm,MC-001113-k2lR,ACH-001113,Yes,Yes,4.9565773,0.0,...
```

Note:
- La prima colonna senza nome (`Unnamed: 0`) e' un indice numerico di riga che
  pandas legge come colonna. **E' rumore** e viene droppato dentro `data.py`.
- Le colonne geniche hanno il formato `HGNC_SYMBOL (ENTREZ_ID)`. `data.py`
  le rinomina in `HGNC_SYMBOL` (strip dell'ID tra parentesi).

### `OmicsSomaticMutations.csv` — MAF-like

Una riga per ogni mutazione osservata. 69 colonne. Quelle chiave per noi:

- `ModelID` — cell line
- `HugoSymbol` — gene (filtriamo == `TP53`)
- `VariantInfo` — annotazione Sequence Ontology (es. `missense_variant`,
  `stop_gained`, `frameshift_variant`, `splice_donor_variant`, `inframe_deletion`)
- `VariantType` — `SNV` / `insertion` / `deletion` / `substitution`
- `ProteinChange` — es. `p.R273H` per missense, `p.R196Ter` per nonsense
- `IsDefaultEntryForModel` — `Yes` / `No` per deduplicazione
- `LikelyLoF`, `VepImpact` — annotazioni di predicted impact

### `OmicsSomaticMutationsMatrixDamaging.csv`

Stessa struttura di OmicsExpression (metadata + colonne geniche) ma i valori
sono score di mutazione *damaging*: 0 = WT, 1 = monoallelica, 2 = biallelica.
La soglia di "damaging" e' applicata internamente da DepMap.

## Release usata e riproducibilita'

**Release:** DepMap Public 26Q1 (2026-04-01)

I file scaricati in data/ sono quelli di questa specifica release. Cambiare
release puo' modificare leggermente i numeri (nuove cell lines aggiunte,
ri-annotazioni di mutazioni) ma la pipeline funziona identica.

Per riprodurre esattamente gli stessi numeri riportati in
`docs/task1.md` e `docs/task2.md` bisogna usare 26Q1.
