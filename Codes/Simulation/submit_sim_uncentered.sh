#!/bin/bash -l
#SBATCH --job-name=regime_sim_unc
#SBATCH --output=logs/simunc_%A_%a.out
#SBATCH --error=logs/simunc_%A_%a.err
#SBATCH --partition=cpu-single
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=0-18:00:00
#SBATCH --array=2
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=kenoku.ychu.ym421@gmail.com

# UNCENTERED check for the integrated estimator (Assumption 6 disabled).
#
# CENTER_MS_PREDICTOR=0 makes the transition predictor the RAW regime-1
# conditional filtered mean of eta1 (no person-mean centering), which matches
# the data-generating process in sim_data_generation.py exactly and the
# Kelava (2022) parameterization. Everything else is identical to the main
# integrated g35 run, so results are directly comparable with
# output_integrated_g35/.
#
# Motivation: the DGP does not center, so under the centered estimator the
# effective estimands for gamma1 and gamma2 are shifted (approx. 3.77 and
# -0.69 instead of 3.50 and -1.17); the uncentered estimator removes this
# mismatch. The original motivation for centering (collinearity of gamma3
# with gamma2 under AR ~ 0.9) is expected to be much weaker now that the
# estimated persistence is moderate and the level is carried by the zeta2
# block. This run decides it empirically.
#
# One array task only: task 2 = (N=100, NT_TRAIN=25), i.e. D_{100,25}.
# Results -> output_integrated_g35_uncent/. Existing outputs untouched.
# Decision rule: if gamma1 recovers ~3.5 and gamma2 ~ -0.40 (Bartlett
# attenuation x0.34 of -1.17) with stable SEs and unchanged classification,
# drop Assumption 6; extend to the remaining conditions before merging.

module load devel/python/3.13.1
source ~/venv_regime/bin/activate

cd $HOME/Codes/Simulation
mkdir -p output_integrated_g35_uncent logs

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export SIM_INTEGRATED=1
export SIM_TAG=g35_uncent
export CENTER_MS_PREDICTOR=0

echo "Job started: $(date)   array task ${SLURM_ARRAY_TASK_ID}   (integrated, UNCENTERED, gamma1 = 3.5)"
echo "Node: $(hostname)"
grep '^gamma1' parameter_estimates_kelava.csv
python sim_main.py
echo "Job finished: $(date)"
