#!/bin/bash
#SBATCH --job-name=regime_sim
#SBATCH --partition=single
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --output=logs/sim_%j.out
#SBATCH --error=logs/sim_%j.err

# ---- Environment (Helix: miniforge) ----
module load devel/miniforge
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate regime_sim

# ---- Run ----
mkdir -p logs
cd "$(dirname "$0")"

echo "Starting simulation: $(date)"
python sim_main.py
echo "Done: $(date)"
