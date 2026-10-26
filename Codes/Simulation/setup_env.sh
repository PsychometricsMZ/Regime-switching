#!/bin/bash
# Run once: set up the Python virtual environment on bwHelix

module load devel/python/3.13.1

# Create the virtual environment
python -m venv ~/venv_regime
source ~/venv_regime/bin/activate

# Install packages (CPU-only torch keeps the environment small)
pip install --upgrade pip
pip install numpy pandas scipy scikit-learn matplotlib pyreadr
pip install torch --index-url https://download.pytorch.org/whl/cpu

echo "Environment setup complete."
python --version
python -c "import torch; print('torch:', torch.__version__)"
