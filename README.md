# StereoCell 细胞分割流程（集群版）

[![CI](https://github.com/CozyGalaxy/stereocell-pipeline/actions/workflows/ci.yml/badge.svg)](https://github.com/CozyGalaxy/stereocell-pipeline/actions/workflows/ci.yml)
[![release](https://img.shields.io/github/v/release/CozyGalaxy/stereocell-pipeline)](https://github.com/CozyGalaxy/stereocell-pipeline/releases/tag/v1.0.0)

输入（均来自 StereoCell 技术）：
1. **ssDNA 灰度图像**（核染色，鉴定细胞核位置）
2. **逐像素表达矩阵**：`geneID, x, y, MIDCount, ExonCount, label`（首行为 title；`label=0/0.0` 表示无细胞；其余为预分配细胞 ID——纯数字或带前缀字符串如 `Sample.chip.2803`，**不准确**，需用 ssDNA 校准）
3. **可靠细胞列表**：经下游验证的少量准确细胞 ID（作为锚点；可与 label 同格式）

流程由**两个可独立运行的模块**组成，均支持"训练参数模式"与"无督导模式"：

```
模块一 (核识别):  nuclei_train.py → params_nuclei.json → nuclei_segment.py
                        │ (训练: 多组 ssDNA+矩阵+可靠细胞, 网格搜索)
                        ▼
模块二 (细胞分割): cell_train.py → params_cell.json → cell_segment.py
                        │ (训练: 可靠细胞锚定, 网格搜索 EM 阈值)
                        ▼
        细胞 mask / 像素映射 / 更新矩阵 / QC / 合胞体概率 / h5ad
```

## 模块一：ssDNA 细胞核识别

```bash
# 训练 (manifest: 每行  ssdna<TAB>matrix<TAB>reliable, 可多行多数据集)
python nuclei_train.py --manifest train.tsv --out params_nuclei.json [--quick]

# 推理-训练参数模式
python nuclei_segment.py --ssdna ssDNA.tif --outdir out_nuclei/ --params params_nuclei.json

# 推理-无督导模式 (自适应 Otsu + 致密组织回退)
python nuclei_segment.py --ssdna ssDNA.tif --outdir out_nuclei/ [--nuc-radius 6]
```

输出：
- `nuclei_pixel_map.tsv.gz` — 文本：**每个像素的 x, y 及所属细胞核 ID**
- `nuclei_overlay.png` — ssDNA 图 + 1 像素红圈（核）
- `nuclei_mask.tif`、`nuclei_summary.csv`（质心/面积/强度）、`params_used.json`

训练目标（可靠细胞为 StereoCell 预分割+扩散后的细胞范围，大于核、可含多核）：
`score = recall − w·oversplit`，recall = 可靠细胞覆盖半径内含 ≥1 核质心的比例；oversplit = 每可靠细胞核数超过 2 的均值惩罚。核质心判定在连续坐标系进行（分子位于离散捕获点，不做像素栅格化）。

## 模块二：细胞分割（RNA 扩散模型 + QC + 合胞体评估）

```bash
# 训练 (复用模块一参数, 网格搜索 conf_thr/margin/dense_margin)
python cell_train.py --manifest train.tsv --out params_cell.json \
    --params-nuclei params_nuclei.json [--quick] [--em-iter 8]

# 推理-训练参数模式
python cell_segment.py --ssdna ssDNA.tif --matrix matrix.txt \
    --reliable solid.cell.list --params params_cell.json --outdir out_cells/

# 推理-无督导模式
python cell_segment.py --ssdna ssDNA.tif --matrix matrix.txt --outdir out_cells/ [--kappa 1.0]

# 复用模块一已生成的核 mask (跳过核分割)
python cell_segment.py ... --nuclei-mask out_nuclei/nuclei_mask.tif
```

算法（模拟原型实测结论，详见 skill `st-cell-segmentation/references/stereocell.md`）：
- **细胞 > 核**：核为种子向外扩展；**转录本范围 > 细胞范围**：扩散长度 λ 由孤立细胞径向 UMI 衰减拟合 `exp(-r/λ)`（密度护栏：λ ≤ 0.45·最近邻距离中位数，防致密区拟合爆炸）；
- **聚集细胞中线划分**：领地半径上限 `R_i = min(r95, κ·d_i/2)`，κ=1.0 即严格中线；
- **不硬截断候选半径**（会推走自己的分子、升高近邻污染）；"密度大则缩小范围"用**后验边际拒绝**——拥挤细胞 margin_i 上调（默认 0.20→0.35），两细胞后验接近的分子整体拒收；
- **背景污染**：GMM-EM 内置空间均匀背景类（w_b·b(gene)/Area），低置信分子 label 置 0；
- **QC 过滤**：线粒体基因（`mito-` 前缀）比例 > 10% 的细胞过滤（比例见 `cell_qc.csv`，同时给出 `Ribo-` 核糖体蛋白比例）；被滤细胞分子在输出矩阵中置 0；
- **合胞体（多核细胞）评估**：质心距离 < 2.5×最近邻间距中位数的细胞对，计算 log1p-CPM 余弦相似度、top-50 高表达基因共享率、距离衰减分，logistic 组合为同源概率。

输出：
- `cell_pixel_map.tsv.gz` — 文本1a：**每个像素的 x, y 及所属细胞 ID**（像素级）
- `matrix_cell_id.txt.gz` — 文本1b：与输入同构矩阵，`cell_id` 列为更新后细胞 ID（0 = 背景/拒收/被过滤）
- `syncytium_pairs.tsv` — 文本2：相近细胞对来自同一多核细胞的概率（含距离/相似度/共享率特征）
- `cells_overlay.png` — ssDNA 图 + 1 像素红圈（最终细胞）；`seeds_overlay.png`（种子）
- `cell_qc.csv` — 逐细胞 UMI/基因数/mito_frac/ribo_frac/置信度/是否通过
- `cells_x_genes.h5ad` — 过滤后细胞×基因稀疏矩阵（scanpy 直接可读）
- 中间结果：`01_nuclei_mask.tif`、`03_diffusion_fit.png`、`04_cell_params.csv`、`qc_report.json`

## 算法文档

流程背后的算法调研与实测结论（来自 `st-cell-segmentation` skill 知识库）：

- [docs/stereocell.md](docs/stereocell.md) — StereoCell 分割策略与自适应 EM 原型的模拟实测（边际拒绝 vs 硬截断等关键决策依据）
- [docs/segmentation-strategies.md](docs/segmentation-strategies.md) — 三类输入（仅图像 / 仅表达 / 联合）分割算法对比
- [docs/rna-diffusion.md](docs/rna-diffusion.md) — RNA 扩散建模与背景污染处理
- [docs/registration.md](docs/registration.md) — 图像-矩阵配准算法（STalign / trackline / imreg_dft）
- [docs/tool-playbooks.md](docs/tool-playbooks.md) — CellBin2 / Cellpose-SAM / Baysor / BIDCell / DeepCell 等工具适用场景

## 依赖

- Python ≥3.10；numpy、scipy、scikit-image、tifffile、pandas、matplotlib
- 可选 GPU：**PyTorch (CUDA)** —— EM 归属 GPU 加速（`--device cuda`）
- 可选 GPU：**cellpose** —— ssDNA 核分割 GPU 后端（`--seed-backend cellpose`），默认 skimage tiled（CPU 即可）
- 可选：anndata（输出 .h5ad；缺失自动降级 .mtx）、pyarrow（parquet 输入）

```
pip install numpy scipy scikit-image tifffile pandas matplotlib anndata
pip install torch --index-url https://download.pytorch.org/whl/cu121   # GPU 可选
pip install cellpose                                                    # GPU 分割后端可选
```

## 安装（v1.0.0）

```bash
pip install .            # 或 pip install git+https://github.com/<user>/stereocell-pipeline.git
pip install .[h5ad]      # 带 anndata 输出
pip install -e .         # 开发模式
```

安装后可直接使用命令行入口（与 `python xxx.py` 等价）：

```bash
stereocell-nuclei-train    --manifest train.tsv --out params_nuclei.json
stereocell-nuclei-segment  --ssdna ssDNA.tif --outdir out_nuclei/
stereocell-cell-train      --manifest train.tsv --out params_cell.json
stereocell-cell-segment    --ssdna ssDNA.tif --matrix matrix.txt --outdir out_cells/
stereocell-run             --smoke-test --outdir /tmp/scell_smoke
```

## 集群运行

- **先试跑再全量**：用 `crop_region.py` 从全尺寸数据裁剪子区域（自动选最密/中位密度窗口，坐标重定基，可靠列表同步裁剪，输出可直接用于训练的 manifest）：

```bash
python crop_region.py --ssdna big.tif --matrix big.txt.gz \
    --reliable solid.cell.list --outdir pilot/ --size 6000 --auto dense
python cell_segment.py --ssdna pilot/crop_ssdna.tif --matrix pilot/crop_matrix.txt.gz \
    --reliable pilot/crop_reliable.list --outdir pilot/cells/
```

- 大数据建议：内存 ≈ 分子数 × 40 B + 候选对 × 24 B；1 亿分子约需 8–16 GB RAM；EM 已全向量化（候选对排序 + reduceat）。23500² / 8000 细胞 / ~3 亿分子规模资源评估：CPU 64 GB 节点 1.5–3 h；CUDA（≥24 GB 显存）1–2 h。
- 冒烟自检（部署后先跑）：`python run_pipeline.py --smoke-test --outdir /tmp/scell_smoke`。
- 传统单步入口 `run_pipeline.py`（无训练/QC/合胞体功能）保留兼容；SLURM 示例见 `slurm_example.sh`。

## 参数速查

| 参数 | 默认 | 说明 |
|---|---|---|
| NucleiParams.thr_factor | 1.0 | Otsu 阈值倍率（训练学习） |
| NucleiParams.min_distance | 6 | 相邻核最小间距 px（训练学习） |
| NucleiParams.dense_cov | 0.35 | 前景覆盖率超过则启用致密回退（强度峰种子） |
| CellParams.kappa | 0.9 | R_i = min(r95, κ·d_i/2)；1.0 = 严格中线划分 |
| CellParams.conf_thr | 0.55 | 分子归属最低后验（训练学习） |
| CellParams.margin / dense_margin | 0.20 / 0.35 | top1−top2 最小后验差，拥挤区自动上调（训练学习） |
| CellParams.w_b | 0.15 | 背景类先验权重 |
| CellParams.mito_max | 0.10 | 线粒体比例过滤阈值 |

## 真实数据验证（Y40360CC, 果蝇 0.5d 雌幼虫, 2000×2000 示例窗）

| 指标 | 无督导 | 训练后 |
|---|---|---|
| 检出核 / 可靠细胞核召回 | 189 / 91.7% | 190 / 91.7%（oversplit 6.7%↓） |
| λ / r95 | 13.4px / 40.3px | 13.4px / 40.2px |
| 可靠分子归属正确率 (EM) | — | **86.3%**（污染 1.0%） |
| 分子归属率 | 28.9% | 28.9% |
| 细胞 QC | 189 通过 / 0 过滤 | 190 通过 / 0 过滤（mito max 2.9%） |
| 合胞体候选对 (prob≥0.5) | 16 | 16 |

## 目录结构

```
stereocell_pipeline/
├── README.md            ← 本文档
├── requirements.txt
├── nuclei_train.py      ← 模块一-训练
├── nuclei_segment.py    ← 模块一-推理 (训练参数/无督导)
├── cell_train.py        ← 模块二-训练
├── cell_segment.py      ← 模块二-推理 (分割+QC+合胞体)
├── run_pipeline.py      ← 传统单步入 口(兼容) + 冒烟测试
├── slurm_example.sh
└── scell/
    ├── io.py            ← 矩阵读写(label 兼容数字/前缀字符串)、h5ad/mtx 导出
    ├── seeds.py         ← ssDNA 核分割 (分块; 致密回退; 连续重标号) + 红线叠加
    ├── nuclei_model.py  ← 模块一参数模型/训练/锚定评分
    ├── diffusion.py     ← 密度、λ 估计(子采样+密度护栏)、R_i/margin_i
    ├── assign.py        ← 边际拒绝 GMM-EM (全向量化 numpy / torch GPU)
    ├── cell_model.py    ← 模块二参数模型/核心分割/训练/锚定评分
    ├── qc.py            ← mito-/Ribo- 比例与过滤
    ├── syncytium.py     ← 多核细胞(合胞体)概率评估
    ├── export.py        ← 领地栅格化、QC 报告
    └── simulate.py      ← 合成 StereoCell 数据 (冒烟测试)
```
