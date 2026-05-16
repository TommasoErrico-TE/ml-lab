#!/bin/bash
#SBATCH --job-name=tp53_train_tcga
#SBATCH --partition=medium_cpu
#SBATCH --qos=normal
#SBATCH --cpus-per-task=16
#SBATCH --mem=48G
#SBATCH --time=04:00:00
#SBATCH --output=slurm/logs/%x_%j.out
#SBATCH --error=slurm/logs/%x_%j.err

# ============================================================================
# Retraining della stessa pipeline su TCGA Pan-Cancer.
# 60/20/20 stratified split + top-2000 geni per varianza + RandomizedSearchCV
# 30 iter * 5-fold CV su LogReg / RandomForest / XGBoost.
#
# Input:   output_tp53/datasets/tcga_master.parquet
#          (genera con scripts/download_tcga.py + build_tcga_master.py)
# Output:  output_tp53/tcga/tcga_hp_search_best.csv
# ============================================================================

set -uo pipefail
cd "$SLURM_SUBMIT_DIR"

date
echo ""

# Se manca il master TCGA, lo costruiamo
if [ ! -f output_tp53/datasets/tcga_master.parquet ]; then
    echo ">>> tcga_master.parquet mancante, lo costruisco"
    if [ ! -f data/tcga/mc3.v0.2.8.PUBLIC.xena.gz ]; then
        python -u scripts/download_tcga.py
    fi
    python -u scripts/build_tcga_master.py
fi

python -u scripts/train_tcga.py \
    --n-iter 30 --top-k 2000 --jobs "$SLURM_CPUS_PER_TASK"
rc=$?

date
echo ""
echo "exit: $rc"
exit $rc
