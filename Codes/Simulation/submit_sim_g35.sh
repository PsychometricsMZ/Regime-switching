#!/bin/bash -l
#SBATCH --job-name=regime_sim_g35
#SBATCH --output=logs/simg35_%A_%a.out
#SBATCH --error=logs/simg35_%A_%a.err
#SBATCH --partition=cpu-single
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=0-12:00:00
#SBATCH --array=0-3
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=kenoku.ychu.ym421@gmail.com

# Re-run of the full design with the transition intercept set to gamma1 = 3.5
# instead of the 1.48 reported by Kelava et al. (2022).
#
# Reason: with gamma1 = 1.48 the per-time-point switching probability is 18.5%,
# so under the near-absorbing regime 2 essentially every person has switched by
# t = 25. Regime 2 then accounts for 79-89% of person-time points in the
# estimation window and 97-99% in the forecast window, and a constant
# "always regime 2" predictor beats the model on accuracy. At gamma1 = 3.5 the
# switching probability is about 3% and both regimes are represented throughout
# (regime 2: 29-45% observed, 54-74% forecast).
#
# Results go to output_g35/ so the existing output/ is untouched.
#
# Four array tasks, one per (N, NT_TRAIN) condition:
#   task 0 -> N=50,  NT_TRAIN=25      task 2 -> N=100, NT_TRAIN=25
#   task 1 -> N=50,  NT_TRAIN=50      task 3 -> N=100, NT_TRAIN=50

module load devel/python/3.13.1
source ~/venv_regime/bin/activate

cd $HOME/Codes/Simulation
mkdir -p output_g35 logs

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export SIM_TAG=g35

echo "Job started: $(date)   array task ${SLURM_ARRAY_TASK_ID}   (gamma1 = 3.5)"
echo "Node: $(hostname)"
grep '^gamma1' parameter_estimates_kelava.csv
python sim_main.py
echo "Job finished: $(date)"
