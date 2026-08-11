# 工具操作手册（Tool Playbooks）

## 目录
1. SAW / CellBin
2. StereoMap 4（手动配准与第三方 mask 闭环）
3. Cellpose / Cellpose-SAM
4. DeepCell Mesmer
5. Baysor
6. BIDCell
7. STalign

---

## 1. SAW / CellBin

**流程**：FASTQ + 芯片 mask → `saw count` → .gef 表达矩阵 +（有图像时）自动 CellBin → `.cellbin.gef` / `.cellbin_1.0.h5ad`。

CellBin 内部步骤：MFWS 频域拼接 → trackline 刚体配准（误差 <5 μm）→ 组织分割 → 核分割（自研 DL，支持 ssDNA/DAPI/H&E(FF)/IF/CFW）→ MLCG 分子标记（核 mask 内 GMM 拟合，核外分子按后验分配）。

**关键命令**：
```bash
# 第三方 mask 后重算 CellBin（SAW >= 8.0）
saw realign \
  --id=<SN>_realign \
  --sn=<SN> \
  --count-data=<原 saw count 输出目录> \
  --realigned-image-tar=<StereoMap 导出的图像 tar.gz> \
  --threads-num=24
# 产物：outs/analysis/*.cellbin*.h5ad, outs/feature_expression/*.cellbin.gef
```

**何时降级**：bin20 中位基因数 <200 或图像质量差 → 用 bin50/bin100 分析，不强行 CellBin。

## 2. StereoMap 4

- **Image QC**：判断图像能否走 SAW 自动流程。
- **Image Processing**：手动配准（Morphology=形态学对齐；Feature Point=特征点），Step 5 Export 导出 `*_regist.tif`（与空间矩阵逐像素对齐）和图像 `.tar.gz`。
- **导入第三方 mask**：在 Cell Segmentation 步骤替换自动 mask（要求 uint8 单通道二值 TIFF，与 regist.tif 同形同向），导出后交 SAW realign。
- **Visual Explore**：`.gef` + `.rpi`；图层 Image / CellMask / CellMask_adjusted（外扩 10 px）叠加检查分割。
- 若 SAW 已做自动图像分析，优先上传 `.stereo` 文件（自带配准信息）。

## 3. Cellpose / Cellpose-SAM

```python
from cellpose import models
model = models.CellposeModel(gpu=True, model_type="cyto3")  # v4: pretrained_model="cpsam"
masks, flows, styles = model.eval(img, diameter=None, channels=[0, 0])
```
- v4 (Cellpose-SAM)：SAM ViT 主干 + 梯度流 tracking；抗噪声/模糊/降采样，跨域泛化最强——**ssDNA 图像分割首选基线**。
- 关键参数：`diameter`（像素，可自动估计）、`cellprob_threshold` / `flow_threshold`（欠/过分割调节）。
- 联合表达数据的简易做法：第二通道放"全基因表达密度图"（分子坐标高斯核栅格化），`channels=[cyto, nuclei]`。
- 弱染色图像：先用 Cellpose 3 的 restoration（去噪/去模糊）再分割；仍差则 GUI 人工标注几十细胞做微调。
- 大图的内存：分块 + 边缘重叠（tile overlap ≥ 预期细胞直径），拼接处按 mask 重叠投票去重。

## 4. DeepCell Mesmer

```python
from deepcell.applications import Mesmer
app = Mesmer()
mask = app.predict(img, image_mpp=0.5, compartment="whole-cell")  # 或 "nuclear"
```
- 输入：双通道 [核(ssDNA/DAPI), 膜/胞质]；只有核染时 `compartment="nuclear"`。
- 训练分辨率 0.5 μm/px（20x）；其他分辨率必须设 `image_mpp`，必要时重采样。
- 优势：快（后处理轻）、组织上人类水平精度、自动核-细胞配对（可算 N/C 比）。

## 5. Baysor

```bash
baysor run -x x -y y -g gene --min-molecules-per-cell 15 \
  [--prior-segmentation nuclei_mask.tif] [--scale <半径, 慎用>] transcripts.csv
```
- 三种模式：纯分子 / +图像先验（`prior_segmentation_confidence` 默认 ~0.2，越高越贴图像）/ +scRNA 细胞类型先验。
- **参数经验**：细胞大小差异大时不要设 `--scale`（会产生不真实的均匀细胞）；大细胞过分割 → 增大半径重跑"噪声分子+超大对象"两轮策略；输出含 assignment confidence，边界低置信分子建议过滤。
-  Stereo-seq 分子密度极高 → 按组织区域切片跑或用于关键区域精修/验证，不建议全芯片默认跑。

## 6. BIDCell

- 输入：表达栅格图（基因=通道，48×48×n_genes patch，两套半重叠 patch 推理）+ DAPI 核分割（其官方用 Cellpose）+ 参考细胞类型正负 marker（Human Cell Atlas 等）。
- 自监督 6 损失：核包裹 / cell-calling（按核离心率区分 elongated 类型）/ 过分割 / 重叠 / 正 marker / 负 marker。无需人工分割标注。
- 注意： elongated 细胞类型需按组织先验指定；推理+形态学后处理较慢；12 GB GPU 验证至 960 基因。
- 移植到 Stereo-seq 时：ssDNA 核分割替代 DAPI；Stereo-seq 全转录组基因数 >>960，需选 HVG/marker 子集作通道。

## 7. STalign

- 用途：切片间、跨技术（MERFISH↔Visium）、到 Allen CCF 的 2D/3D 对齐。**不是**图像↔矩阵的芯片内配准（那是 trackline/StereoMap 的活）。
- 流程：细胞坐标高斯核栅格化 → 人工 landmark 初始化（部分匹配时必要）→ 仿射 + LDDMM 梯度下降 → 变换应用到原始坐标。
- 经验：选更完整的切片作 source（GMM 伪影项作用于 target）；σ²_R/σ²_M 调光滑度与匹配精度平衡；局部极小风险 → 多初始化。
