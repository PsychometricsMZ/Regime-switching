#!/bin/bash -l
#SBATCH --job-name=regime_sim_redintg
#SBATCH --output=logs/simredintg_%A_%a.out
#SBATCH --error=logs/simredintg_%A_%a.err
#SBATCH --partition=cpu-single
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=0-6:00:00
#SBATCH --array=0-3
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=kenoku.ychu.ym421@gmail.com

# Restricted-model run under the INTEGRATED estimator (gamma1 = 3.5 calibration).
#
# The estimator holds gamma3 = gamma4 = 0 while the DGP keeps the true non-zero
# values (same comparison as the two-stage restricted run), but zeta_{2i} is
# marginalized in the state with Q2 estimated by ML (sim_filtering_integrated.py).
# Together with output_integrated_g35/ this gives the full-vs-restricted
# comparison under the integrated estimator, mirroring the two-stage pair
# output_g35/ vs output_reduced_g35/.
#
# Results go to output_reduced_integrated_g35/. Existing outputs are untouched.
# Wall time based on the observed integrated full-model run (~30-55 min/task).
#
# Four array tasks, one per (N, NT_TRAIN) condition:
#   task 0 -> N=50,  NT_TRAIN=25      task 2 -> N=100, NT_TRAIN=25
#   task 1 -> N=50,  NT_TRAIN=50      task 3 -> N=100, NT_TRAIN=50

module load devel/python/3.13.1
source ~/venv_regime/bin/activate

cd $HOME/Codes/Simulation
mkdir -p output_reduced_integrated_g35 logs

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export SIM_REDUCED=1
export SIM_INTEGRATED=1
export SIM_TAG=g35

echo "Job started: $(date)   array task ${SLURM_ARRAY_TASK_ID}   (reduced + integrated, gamma1 = 3.5)"
echo "Node: $(hostname)"
grep '^gamma1' parameter_estimates_kelava.csv
python sim_main.py
echo "Job finished: $(date)"
