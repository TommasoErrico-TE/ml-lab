# HPC (Bocconi) — submission guide

3 job pronti in `slurm/jobs/`:

| Job | Tempo | Risorse | Cosa produce |
|---|---|---|---|
| `hp_search_ccle.slurm` | ~30–60 min | 16 CPU / 32 GB | `output_tp53/task1/hp_search_best.csv` |
| `train_tcga.slurm` | ~2–4 h | 16 CPU / 48 GB | `output_tp53/tcga/tcga_hp_search_best.csv` |
| `mlp.slurm` | ~30–60 min | 8 CPU / 24 GB / 1 GPU | `output_tp53/task1/mlp_*.csv`, `task2/mlp_*.csv`, `tcga/tcga_mlp_*.csv` |

## 1) Clonare il repo sull'HPC

Prima di tutto su locale (Mac) commit + push delle modifiche, poi sull'HPC:

```bash
# sull'HPC (login node)
cd ~
git clone https://github.com/TommasoErrico-TE/ml-lab.git
cd ml-lab
```

## 2) Installare le librerie (una volta sola, niente venv)

Le librerie minime per i 3 job:

```bash
# sull'HPC, login node
module load python/3.11           # adatta al modulo Python disponibile (es. python/3.11, anaconda3, ...)

pip install --user pandas numpy scikit-learn xgboost scipy pyarrow
pip install --user torch          # solo per i job MLP
```

`--user` installa in `~/.local/`, niente venv, niente env.sh. Tutti i job lo vedono automaticamente.

> Se `module load python` non funziona, prova `module avail python` o `module avail anaconda` per vedere quali Python sono disponibili.

## 3) Caricare i dati grezzi sull'HPC

I file CSV di CCLE (~1.1 GB) sono gitignored, quindi non arrivano col `git clone`. Vanno trasferiti via rsync dal Mac:

```bash
# da locale (Mac)
rsync -avz --progress \
    /Users/edoardopaccagnella/ml-lab/data/ \
    user@bocconi-hpc:~/ml-lab/data/
```

I file TCGA (`data/tcga/`) sono ~395 MB, ma si possono anche scaricare direttamente sull'HPC (più veloce, niente upload):

```bash
# sull'HPC
cd ~/ml-lab
python scripts/download_tcga.py
```

## 4) Costruire i master dataset (una volta sola)

```bash
# sull'HPC
cd ~/ml-lab
python scripts/data.py                  # genera output_tp53/datasets/master.csv
python scripts/build_task2_labels.py    # genera output_tp53/datasets/task2_labels.csv
python scripts/build_tcga_master.py     # genera output_tp53/datasets/tcga_master.parquet
```

In alternativa, se hai già questi 3 file in `output_tp53/datasets/` sul Mac, puoi anche rsync-arli direttamente:

```bash
# da locale (Mac)
rsync -avz \
    /Users/edoardopaccagnella/ml-lab/output_tp53/datasets/ \
    user@bocconi-hpc:~/ml-lab/output_tp53/datasets/
```

## 5) Sottomettere i job

```bash
# sull'HPC
cd ~/ml-lab
mkdir -p slurm/logs

sbatch slurm/jobs/hp_search_ccle.slurm
sbatch slurm/jobs/train_tcga.slurm
sbatch slurm/jobs/mlp.slurm
```

Monitora con `squeue -u $USER`. Log in `slurm/logs/<jobname>_<jobid>.out` e `.err`.

## 6) Scaricare i risultati a fine job

```bash
# da locale (Mac)
rsync -avz \
    user@bocconi-hpc:~/ml-lab/output_tp53/ \
    /Users/edoardopaccagnella/ml-lab/output_tp53/
```

Poi rilancia il notebook in locale:

```bash
jupyter nbconvert --to notebook --execute ML_project.ipynb \
    --output ML_project.ipynb
jupyter nbconvert --to html ML_project.ipynb
```

## Cosa adattare nei `.slurm` per Bocconi

I file usano queste impostazioni "tipo Bocconi":

- `--partition=medium_cpu` (per HP search + TCGA) e `long_gpu` (per MLP)
- `--qos=normal`
- log in `slurm/logs/<jobname>_<jobid>.out`

Se i nomi partition / qos sul cluster Bocconi sono diversi, modificali nei 3 file. Comandi utili:

```bash
sinfo                    # vede i partition disponibili
sacctmgr show qos        # vede le QoS disponibili
```
