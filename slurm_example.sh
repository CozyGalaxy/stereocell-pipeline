#!/bin/bash
#SBATCH --job-name=stereocell_seg
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=logs/stereocell_%j.out

set -euo pipefail

# 环境: 建议 conda
#   conda create -n scell python=3.11 -y && conda activate scell
#   pip install -r requirements.txt
#   pip install torch --index-url https://download.pytorch.org/whl/cu121
#   pip install cellpose   # 可选: Cellpose-SAM 核分割后端
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate scell

DATA=/data/stereocell/chipA
OUT=/results/stereocell/chipA
mkdir -p "$OUT" logs

# 部署后先跑冒烟自检(约2-5分钟, CPU即可):
# python run_pipeline.py --smoke-test --outdir "$OUT/smoke"

python run_pipeline.py \
  --ssdna      "$DATA/ssDNA.tif" \
  --matrix     "$DATA/matrix.csv.gz" \
  --reliable   "$DATA/reliable_cells.txt" \
  --outdir     "$OUT" \
  --seed-backend cellpose \
  --device cuda \
  --tile 4096 \
  --chunk 4000000
