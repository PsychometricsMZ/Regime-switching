#!/bin/bash -l
#SBATCH --job-name=regime_test
#SBATCH --output=logs/test_%j.out
#SBATCH --error=logs/test_%j.err
#SBATCH --partition=cpu-single
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=0-00:30:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=kenoku.ychu.ym421@gmail.com

module load devel/python/3.13.1
source ~/venv_regime/bin/activate

cd $HOME/Codes/Simulation
mkdir -p output_test logs sim_summary_test

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

echo "Test job started: $(date)"
echo "Node: $(hostname)"
python run_test.py
echo "Test job finished: $(date)"
