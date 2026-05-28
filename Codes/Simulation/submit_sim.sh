#!/bin/bash -l
#SBATCH --job-name=regime_sim
#SBATCH --output=logs/sim_%j.out
#SBATCH --error=logs/sim_%j.err
#SBATCH --partition=cpu-single
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=0-12:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=kenoku.ychu.ym421@gmail.com

module load devel/python/3.13.1
source ~/venv_regime/bin/activate

cd $HOME/Codes/Simulation
mkdir -p output logs

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

echo "Job started: $(date)"
echo "Node: $(hostname)"
echo "Working dir: $(pwd)"
python sim_main.py
echo "Job finished: $(date)"
