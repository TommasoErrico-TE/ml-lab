#!/bin/bash
#SBATCH --job-name=tp53_hp_search_ccle
#SBATCH --partition=medium_cpu
#SBATCH --qos=normal
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=slurm/logs/%x_%j.out
#SBATCH --error=slurm/logs/%x_%j.err

# ============================================================================
# Task 1 — RandomizedSearchCV su LogReg / RandomForest / XGBoost.
# Top-2000 geni per varianza calcolati su train.  30 iter * 5-fold CV.
# Input:   output_tp53/datasets/master.csv  (genera con scripts/data.py)
# Output:  output_tp53/task1/hp_search_best.csv
#          output_tp53/task1/hp_search_<model>.csv
# ============================================================================

set -uo pipefail
cd "$SLURM_SUBMIT_DIR"

date
echo ""

python -u scripts/train_task1_hpsearch.py \
    --n-iter 30 --top-k 2000 --jobs "$SLURM_CPUS_PER_TASK"
rc=$?

date
echo ""
echo "exit: $rc"
exit $rc
