#!/bin/bash -l
#SBATCH --job-name=regime_post
#SBATCH --output=logs/post_%j.out
#SBATCH --error=logs/post_%j.err
#SBATCH --partition=cpu-single
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=0-01:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=kenoku.ychu.ym421@gmail.com

module load devel/python/3.13.1
source ~/venv_regime/bin/activate
cd $HOME/Codes/Simulation
python sim_summarize.py
python sim_evaluation.py
