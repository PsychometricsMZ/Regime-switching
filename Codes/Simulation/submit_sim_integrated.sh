#!/bin/bash -l
#SBATCH --job-name=regime_sim_intg
#SBATCH --output=logs/simintg_%A_%a.out
#SBATCH --error=logs/simintg_%A_%a.err
#SBATCH --partition=cpu-single
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=0-18:00:00
#SBATCH --array=0-3
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=kenoku.ychu.ym421@gmail.com

# Integrated (marginalized) estimator run under the gamma1 = 3.5 calibration.
#
# zeta_{2i} is carried in the state vector with its proper N(0, Q2) prior as
# initial condition; Q2 is estimated by ML through the marginal likelihood
# (see sim_filtering_integrated.py). The two-stage between-level step (step 7)
# is not used. The DGP is identical to the main g35 run, so results are
# directly comparable with output_g35/ (full two-stage model).
#
# Purpose: test whether the persistence/heterogeneity confound (diag(B3)
# biased up, B2/Q2 collapsed) disappears when the likelihood weighs the
# autoregressive and between-person explanations of stable individual levels
# jointly rather than sequentially.
#
# Results go to output_integrated_g35/. Existing outputs are untouched.
# Wall time is set higher than the g35 run: the augmented state doubles the
# state dimension (roughly 4-8x the flops in the covariance recursions).
#
# Four array tasks, one per (N, NT_TRAIN) condition:
#   task 0 -> N=50,  NT_TRAIN=25      task 2 -> N=100, NT_TRAIN=25
#   task 1 -> N=50,  NT_TRAIN=50      task 3 -> N=100, NT_TRAIN=50

module load devel/python/3.13.1
source ~/venv_regime/bin/activate

cd $HOME/Codes/Simulation
mkdir -p output_integrated_g35 logs

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export SIM_INTEGRATED=1
export SIM_TAG=g35

echo "Job started: $(date)   array task ${SLURM_ARRAY_TASK_ID}   (integrated, gamma1 = 3.5)"
echo "Node: $(hostname)"
grep '^gamma1' parameter_estimates_kelava.csv
python sim_main.py
echo "Job finished: $(date)"
