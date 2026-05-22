#!/bin/bash
# 初回のみ実行: Python仮想環境のセットアップ

module load devel/python/3.13.1

# 仮想環境を作成
python -m venv ~/venv_regime
source ~/venv_regime/bin/activate

# パッケージインストール（torch はCPU版を指定して軽量化）
pip install --upgrade pip
pip install numpy pandas scipy scikit-learn matplotlib pyreadr
pip install torch --index-url https://download.pytorch.org/whl/cpu

echo "Environment setup complete."
python --version
python -c "import torch; print('torch:', torch.__version__)"
