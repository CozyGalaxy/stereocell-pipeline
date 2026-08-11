# 图像配准（Registration）

## 目录
1. 芯片内配准：图像 ↔ 表达矩阵（Stereo-seq）
2. 通用配准算法谱系（切片间 / 跨模态 / 到图谱）
3. 配准质量评估
4. 常见问题与对策

---

## 1. 芯片内配准（图像 ↔ 表达矩阵）

这是 CellBin 的前提，目标是让 ssDNA 图像像素与芯片 DNB 坐标一一对应。

- **自动（SAW/CellBin）**：基于芯片 track lines（基准线）做刚体变换（平移/缩放/翻转/旋转），官方误差 <5 μm。前提：track lines 在图像中清晰可辨 → 拍照参数要保基准线可见。
- **手动（StereoMap Image Processing）**：
  - Morphology 模式：形态学特征对齐；
  - Feature Point 模式：人工/自动特征点；
  - 导出 `*_regist.tif`（与矩阵同尺寸同方向）+ 图像 tar.gz。
- **工程红线**：
  - 第三方分割只用 `*_regist.tif`，不用原始图；
  - mask 与 regist.tif 尺寸/分辨率/方向严格一致（uint8 二值）；
  - SAW realign 按像素坐标一对一映射，错位会直接错分细胞。
- **拼接（拍照 tile → 大图）**：CellBin 用 MFWS（频域）拼接，benchmark 优于 ASHLAR/MIST；拼接错位会造成细胞错位，配准前先检查拼接质量。

## 2. 通用配准算法谱系

| 方法 | 变换模型 | 特点 | 适用场景 |
|---|---|---|---|
| Landmark 仿射 | 线性（最小二乘） | 快、可解释 | 形变小的同源切片 |
| **STalign** | 仿射 + LDDMM 微分同胚 | 非线性；坐标栅格化（varifold+高斯核）避免二次复杂度；GMM 匹配/背景/伪影三分量容忍部分匹配与撕裂；2D/3D；可到 Allen CCF | 连续切片堆叠、MERFISH↔Visium 跨技术、图谱 lift-over |
| PASTE / PASTE2 | 最优传输 | 保表达相似性；PASTE2 支持部分重叠 | Visium 级 spot 数据 |
| SLAT | 图匹配 | 严格一一对应 | 同源同质切片 |
| GPSA | 高斯过程空间对齐 | 概率框架 | 连续形变 |
| GALA (2025) | 遗传算法全局 + EM-LDDMM 局部耦合 | landmark-free；基因通道+H&E 作引导；支持部分重叠、跨分辨率 | 复杂跨模态粗到细对齐 |

STalign 使用要点：
- 选更完整的切片作 source（伪影 GMM 作用于 target 侧）；
- 部分匹配/撕裂多时务必人工 landmark 初始化；
- σ²_R（正则/光滑）与 σ²_M（匹配精度）决定形变自由度，先默认后按 landmark RMSE 调；
- LDDMM 双射约束不能产生拓扑差异（洞/撕裂），这类差异靠 GMM 伪影项吸收；
- 梯度下降有局部极小风险 → 多组初始化取 landmark RMSE 最小。

## 3. 配准质量评估

- **Landmark RMSE**：人工/自动特征点配准后距离（STalign 论文：MERFISH 切片 RMSE 202→113 μm vs 仿射）。
- **表达一致性**：空间模式基因在配准后匹配像素/spots 的余弦相似度（对空间模式基因敏感、对非模式基因应无虚假提升）。
- **图像-表达重合**：核 mask 边缘与表达密度图的互信息/边缘重合率；核内 UMI 占比应显著高于随机。
- **CellBin 场景硬指标**：配准误差 << 细胞直径（<5 μm 达标）；在 StereoMap 中叠加 Image + 表达热图目检边缘区。

## 4. 常见问题与对策

- 自动配准失败（QC fail / trackline 不可见）→ StereoMap 手动配准，后续一切照旧。
- H&E FFPE：SAW 默认不做自动细胞分割 → 手动配准 + 第三方分割 + realign。
- 图像与矩阵方向不一致（翻转/旋转）→ 在 StereoMap Feature Point 模式下显式指定；mask 导出后务必叠加验证。
- 大图分块分割后 mask 拼缝处细胞被切 → 分块重叠 ≥ 细胞直径，拼缝区按实例重叠投票合并。
