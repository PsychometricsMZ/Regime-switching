#!/bin/bash -l
#SBATCH --job-name=regime_sim
#SBATCH --output=logs/sim_%A_%a.out
#SBATCH --error=logs/sim_%A_%a.err
#SBATCH --partition=cpu-single
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=0-12:00:00
#SBATCH --array=0-3
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=kenoku.ychu.ym421@gmail.com

# Four array tasks, one per (N, NT_TRAIN) condition of the 2x2 grid:
#   task 0 -> N=50,  NT_TRAIN=25
#   task 1 -> N=50,  NT_TRAIN=50
#   task 2 -> N=100, NT_TRAIN=25
#   task 3 -> N=100, NT_TRAIN=50
# sim_main.py reads SLURM_ARRAY_TASK_ID and runs only that condition, so each task
# does ~1/4 of the work and finishes well within the 12h wall time.

module load devel/python/3.13.1
source ~/venv_regime/bin/activate

cd $HOME/Codes/Simulation
mkdir -p output logs

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

echo "Job started: $(date)   array task ${SLURM_ARRAY_TASK_ID}"
echo "Node: $(hostname)"
echo "Working dir: $(pwd)"
python sim_main.py
echo "Job finished: $(date)"
