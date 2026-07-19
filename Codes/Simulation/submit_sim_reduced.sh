#!/bin/bash -l
#SBATCH --job-name=regime_sim_red
#SBATCH --output=logs/simred_%A_%a.out
#SBATCH --error=logs/simred_%A_%a.err
#SBATCH --partition=cpu-single
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=0-12:00:00
#SBATCH --array=0-3
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=kenoku.ychu.ym421@gmail.com

# Reduced-model run: the estimator holds gamma3 = gamma4 = 0 (the within-person
# predictors are removed from the transition equation), while the data-generating
# process still uses the true non-zero gamma3/gamma4. This provides the comparison
# requested by the reviewer (omitting eta_{1i,t-1} from the transition model).
#
# Results are written to output_reduced/ so the full-model results in output/
# are left untouched.
#
# Four array tasks, one per (N, NT_TRAIN) condition:
#   task 0 -> N=50,  NT_TRAIN=25      task 2 -> N=100, NT_TRAIN=25
#   task 1 -> N=50,  NT_TRAIN=50      task 3 -> N=100, NT_TRAIN=50

module load devel/python/3.13.1
source ~/venv_regime/bin/activate

cd $HOME/Codes/Simulation
mkdir -p output_reduced logs

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export SIM_REDUCED=1

echo "Job started: $(date)   array task ${SLURM_ARRAY_TASK_ID}   (REDUCED model)"
echo "Node: $(hostname)"
python sim_main.py
echo "Job finished: $(date)"
