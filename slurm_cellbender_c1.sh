#!/bin/bash
#SBATCH --job-name=cb_c1
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=logs/cb_c1_%j.out
#SBATCH --error=logs/cb_c1_%j.err
#
# CellBender 对照实验 (C1 = Worker-Pupal.Y40364C1, 扩散污染最重芯片):
#   阶段1: cell-segment (复用 tier1 训练参数与核 mask) + --cellbender 生成伪空液滴
#   阶段2: cellbender remove-background 去 ambient
#   阶段3: 去除前后特征基因表达比例对比 (compare_cb.py)
#
# 用法:
#   1) 编辑下方 TIER1_DIR / PIPE_DIR / OUT / MARKERS 四个变量
#   2) sbatch slurm_cellbender_c1.sh

set -euo pipefail

# ===== 需按集群实际路径修改 =====
TIER1_DIR=/data/Apis_Dev_Stereocell/CellSegTest/v1.2.0/tier1/Worker-Pupal.Y40364C1
PIPE_DIR=/opt/stereocell-pipeline                    # v1.4.0+ 仓库路径
OUT=/results/Apis_Dev_Stereocell/cellbender_test/Worker-Pupal.Y40364C1
MARKERS=""                                           # 可选: marker 基因列表文件 (每行一个基因名)
SAMPLE=Worker-Pupal.Y40364C1
# ===============================

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate scell          # v1.4.0: pip install -U 后含 scell/pseudo.py
mkdir -p "$OUT" logs

SSDNA="$TIER1_DIR/../../tier0/$SAMPLE/$SAMPLE.ssDNA_regist.tif"
MATRIX="$TIER1_DIR/matrix.gz"                       # tier0 输出矩阵 (cell_id 列, v1.3.0+ 直接兼容)
RELIABLE="$TIER1_DIR/solid.cell.list"
NUCMASK="$TIER1_DIR/out_nuclei/nuclei_mask.tif"
PARAMS="$TIER1_DIR/params_cell.json"

# ---------- 阶段1: 分割 + CellBender 输入 ----------
python "$PIPE_DIR/cell_segment.py" \
  --ssdna "$SSDNA" --matrix "$MATRIX" --reliable "$RELIABLE" \
  --nuclei-mask "$NUCMASK" --params "$PARAMS" \
  --sample "$SAMPLE" --cellbender --device cuda \
  --outdir "$OUT/seg"

# ---------- 阶段2: CellBender ----------
N_REAL=$(python -c "
import json; r=json.load(open('$OUT/seg/qc_report.json')); print(r['n_seeds'])")
N_PSEUDO=$(($(wc -l < "$OUT/seg/pseudo_cells.csv") - 1))
echo "真细胞=$N_REAL 伪空液滴=$N_PSEUDO"

conda activate cellbender   # 或同一环境; 需 cellbender>=0.3
cellbender remove-background \
  --input "$OUT/seg/cellbender_raw.h5ad" \
  --output "$OUT/cb_clean.h5ad" \
  --expected-cells "$N_REAL" \
  --total-droplets-included $((N_REAL + N_PSEUDO)) \
  --cuda
conda activate scell

# ---------- 阶段3: 去除前后对比 ----------
MK_ARG=()
[[ -n "$MARKERS" && -s "$MARKERS" ]] && MK_ARG=(--markers "$MARKERS")
python "$PIPE_DIR/compare_cb.py" \
  --before "$OUT/seg/cells_x_genes.h5ad" \
  --after  "$OUT/cb_clean.h5ad" \
  "${MK_ARG[@]}" \
  --outdir "$OUT/compare"

echo "[$(date '+%F %T')] DONE -> $OUT"
