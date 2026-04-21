# Authorship degli script

Il repo e' sviluppato a 3 mani. Questo documento tiene traccia di chi ha
scritto cosa, principalmente per dividersi il lavoro successivo senza
sovrapporci. Non e' un documento di credito, e' un promemoria operativo.

## Divisione attuale

| Script | Autore principale | Quando | Cosa fa |
|---|---|---|---|
| `data.py` | Tommaso Errico | commit `4de1ecb` | Costruisce il master dataset: carica i 5 CSV DepMap, pulisce i nomi di colonna, filtra `IsDefaultEntryForModel`, deduplica per ModelID, deriva la label Task 1, salva in `output_tp53/`. |
| `train_task1.py` | Edoardo Paccagnella | commit `8c3911b` | Allena 3 modelli baseline (Logistic Regression, Random Forest, XGBoost) sul Task 1 con split singolo 80/20 stratificato. Genera tabella di confronto e feature importance per RF e XGB. |
| `check_differences.py` | Edoardo Paccagnella | commit `18216d4` | Diagnostica: confronta il file MAF (`OmicsSomaticMutations.csv`) con la matrice danneggiante (`OmicsSomaticMutationsMatrixDamaging.csv`) per ogni cell line, produce report dei discordanti. Utile per capire edge case. |
| `train_task1_cv.py` | Federico Ferrari | commit `ee7cfd2` | Cross-validation 5-fold in tre scenari (StratifiedKFold, GroupKFold per tessuto, tissue-only baseline) per valutare quanto del segnale Task 1 sia vera firma p53 vs confounding tessuto. Include LeaveOneGroupOut per per-tissue AUC. |
| `build_task2_labels.py` | Federico Ferrari | *(untracked al momento)* | Costruisce le label Task 2 a 4 classi (WT / Missense / Inframe / Truncating) leggendo `OmicsSomaticMutations.csv`. Aggrega per ModelID con regola di priorita' "vince la piu' grave". |

## Bug fix condivisi

| Commit | Autore | Cosa |
|---|---|---|
| `03c987b` | Federico Ferrari | `data.py`: drop delle colonne `Unnamed:*` che pandas creava leggendo i CSV, e che finivano come feature rumore. |

## Regole di convivenza

Per evitare conflitti mentre tutti e tre stiamo sviluppando:

1. **Non rinominare ne' spostare file degli altri** senza prima avvertire in
   chat.
2. **Fix chiari di bug** (tipo l'`Unnamed: 0`) si possono fare direttamente,
   con un commit message esplicito che spiega cosa.
3. **Nuovo lavoro** → sempre in un file nuovo, non dentro file altrui, per
   non triggerare merge conflicts.
4. **Output in `output_tp53/`** sono gitignored per default. Se serve
   committarne uno piccolo (es. una tabella di risultati per la relazione)
   lo si aggiunge con `git add -f`.
5. **`requirements.txt`**: se qualcuno aggiunge una dipendenza nuova, la
   aggiunge anche qui. E' la nostra unica fonte di verita' per le versioni.

## Cosa resta da fare, chi lo prende

Per ora la divisione del lavoro rimanente e' aperta. Lista indicativa dei
prossimi pezzi, da assegnarsi in chat:

- `train_task2.py` — training multinomial per le 4 classi del Task 2
- `train_task1_tuned.py` — hyperparameter tuning + L1 Logistic per Task 1
- `train_mlp.py` — MLP in PyTorch (Task 1 e/o Task 2)
- Download + preprocessing TCGA (Step 8 della pipeline)
- Harmonization CCLE ↔ TCGA con ComBat (Step 8)
- Zero-shot + fine-tuning su TCGA (Step 9-10)
- Interpretabilita': SHAP + confronto con database TP53 (Step 7, 12)

Vedi [`pipeline.md`](pipeline.md) per la strategia completa.
