#!/bin/bash
#SBATCH --job-name=scell16
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=logs/scell16_%A_%a.out
#SBATCH --error=logs/scell16_%A_%a.err
#
# v1.2.0 全量回归: 16 张 StereoCell 芯片端到端 (Step0 ROI + 核识别 + EM 分割 + QC + h5ad)
#
# 用法:
#   1) 编辑下方 DATA_ROOT / OUT_ROOT / PIPE_DIR / CHIPS 四个变量
#   2) sbatch --array=1-16 slurm_array_16chips.sh
#   可选: sbatch --array=1-16%4 ...   (最多并发 4 个, 视 GPU 配额)
#
# 每个数组任务处理一张芯片; 失败芯片可单独重跑:
#   sbatch --array=7 slurm_array_16chips.sh

set -euo pipefail

# ===== 需按集群实际路径修改 =====
DATA_ROOT=/data/Apis_Dev_Stereocell/CellSegTest      # 芯片数据根目录 (每芯片一个子目录)
OUT_ROOT=/results/Apis_Dev_Stereocell/v1.2.0         # 结果根目录
PIPE_DIR=/opt/stereocell-pipeline                    # 本仓库克隆/安装路径
CHIPS="${PIPE_DIR}/chips_16.tsv"                     # 芯片清单 (每行一个芯片目录名)
# ===============================

# 环境
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate scell   # 含 numpy/scipy/scikit-image/tifffile/pandas/matplotlib/anndata/torch(cuda)

CHIP=$(sed -n "${SLURM_ARRAY_TASK_ID}p" "$CHIPS")
DATA="${DATA_ROOT}/${CHIP}"
OUT="${OUT_ROOT}/${CHIP}"
mkdir -p "$OUT" logs

echo "[$(date '+%F %T')] chip=${CHIP} task=${SLURM_ARRAY_TASK_ID} host=$(hostname)"
echo "[$(date '+%F %T')] data=${DATA}"

# 输入文件命名约定: <芯片>/<芯片>.ssDNA_regist.tif, <芯片>.raw.matrix.gz, solid.cell.list
SSDNA="${DATA}/${CHIP}.ssDNA_regist.tif"
MATRIX="${DATA}/${CHIP}.raw.matrix.gz"
RELIABLE="${DATA}/solid.cell.list"

for f in "$SSDNA" "$MATRIX"; do
  [[ -s "$f" ]] || { echo "FATAL: 输入缺失 $f" >&2; exit 2; }
done
REL_ARG=()
[[ -s "$RELIABLE" ]] && REL_ARG=(--reliable "$RELIABLE") || echo "WARN: 无可靠细胞列表, 跳过参数校准"

# 端到端: Step0(ROI+伪影排除) -> 核识别(自适应校准) -> EM 分割 -> QC -> 合胞体 -> h5ad
# 训练参数可选: 有 params_cell.json 时追加 --params
PARAMS_ARG=()
[[ -s "${PIPE_DIR}/params_cell.json" ]] && PARAMS_ARG=(--params "${PIPE_DIR}/params_cell.json")

python "${PIPE_DIR}/cell_segment.py" \
  --ssdna   "$SSDNA" \
  --matrix  "$MATRIX" \
  "${REL_ARG[@]}" \
  "${PARAMS_ARG[@]}" \
  --outdir  "$OUT" \
  --device  cuda \
  --seed-backend skimage

echo "[$(date '+%F %T')] chip=${CHIP} DONE -> ${OUT}"
ls -lh "$OUT"
