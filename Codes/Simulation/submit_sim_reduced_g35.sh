#!/bin/bash -l
#SBATCH --job-name=regime_red_g35
#SBATCH --output=logs/simredg35_%A_%a.out
#SBATCH --error=logs/simredg35_%A_%a.err
#SBATCH --partition=cpu-single
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=0-12:00:00
#SBATCH --array=0-3
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=kenoku.ychu.ym421@gmail.com

# Reduced-model run under the gamma1 = 3.5 calibration, so that it can be
# compared with the main gamma1 = 3.5 run. The estimator holds
# gamma3 = gamma4 = 0 while the data-generating process keeps the true
# non-zero values, which is the comparison the reviewer asked for.
#
# The earlier reduced run (output_reduced/) used gamma1 = 1.48 and is therefore
# not comparable with the new main results.
#
# Results go to output_reduced_g35/.
#
# Four array tasks, one per (N, NT_TRAIN) condition:
#   task 0 -> N=50,  NT_TRAIN=25      task 2 -> N=100, NT_TRAIN=25
#   task 1 -> N=50,  NT_TRAIN=50      task 3 -> N=100, NT_TRAIN=50

module load devel/python/3.13.1
source ~/venv_regime/bin/activate

cd $HOME/Codes/Simulation
mkdir -p output_reduced_g35 logs

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export SIM_REDUCED=1
export SIM_TAG=g35

echo "Job started: $(date)   array task ${SLURM_ARRAY_TASK_ID}   (REDUCED, gamma1 = 3.5)"
echo "Node: $(hostname)"
grep '^gamma1' parameter_estimates_kelava.csv
python sim_main.py
echo "Job finished: $(date)"
