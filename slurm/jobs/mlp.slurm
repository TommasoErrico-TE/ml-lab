#!/bin/bash
#SBATCH --job-name=tp53_mlp
#SBATCH --partition=medium_gpuh200
#SBATCH --qos=normal
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --output=slurm/logs/%x_%j.out
#SBATCH --error=slurm/logs/%x_%j.err

# ============================================================================
# MLP (PyTorch) per Task 1 + Task 2 su CCLE, e Task 1 su TCGA se il master
# TCGA e' gia' presente.  Auto-detect GPU (CUDA se disponibile).
#
# Input:   output_tp53/datasets/master.csv + task2_labels.csv
#          (opzionale) output_tp53/datasets/tcga_master.parquet
# Output:  output_tp53/task1/mlp_*.csv, task2/mlp_*.csv, tcga/tcga_mlp_*.csv
# ============================================================================

set -uo pipefail
cd "$SLURM_SUBMIT_DIR"

date
echo ""

declare -a STEPS=(
    "scripts/compute_mlp.py"
)
if [ -f output_tp53/datasets/tcga_master.parquet ]; then
    STEPS+=("scripts/compute_mlp_tcga.py")
else
    echo "tcga_master.parquet non presente: skip compute_mlp_tcga.py"
fi

declare -a RESULTS=()
overall=0
for step in "${STEPS[@]}"; do
    echo "================================================================"
    echo ">>> $step"
    echo "================================================================"
    date
    python -u "$step"
    rc=$?
    date
    RESULTS+=("$rc  $step")
    if [ "$rc" -ne 0 ]; then
        overall=$rc
        echo "!! step failed (exit $rc) — continuing"
    fi
    echo ""
done

echo "================================================================"
echo "Summary (exit_code  step)"
echo "================================================================"
for r in "${RESULTS[@]}"; do
    echo "  $r"
done
echo ""
echo "overall exit: $overall"
exit $overall
