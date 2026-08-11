# 按输入模态的细胞分割策略

## 目录
- A. 仅 ssDNA/核染图像
- B. 仅表达矩阵（分子坐标）
- C. 图像 + 表达联合
- D. 选型速查

---

## A. 仅 ssDNA/核染图像

| 方法 | 核心算法 | 优势 | 风险/边界 |
|---|---|---|---|
| Cellpose-SAM (v4) | SAM ViT 主干 + 梯度流场 tracking | 跨域泛化最强，抗噪声/模糊/降采样 | 只得核；需 GPU |
| Cellpose 1–3 | U-Net 梯度流场 | 成熟、可人在环微调 | 分布外组织欠分割 |
| Mesmer (nuclear 模式) | ResNet50+FPN 质心/边界 + watershed | 快、组织上人类水平 | 全细胞模式需膜/胞质通道 |
| CellBin 核分割 | 自研 DL（多染色类型训练） | 与 SAW 无缝 | 训练域外组织性能下降 |
| StarDist/watershed | 凸星形多边形/形态学 | 轻量基线 | 异形、密集细胞差 |

要点：纯图像路径的输出是**核**；要得全细胞，必须做边界扩展（EDM/距离外扩）或转入 C 路径。StereoMap 的 `CellMask_adjusted`（外扩 10 px）就是官方的距离外扩实现。

## B. 仅表达矩阵（无图像）

| 方法 | 核心算法 | 优势 | 风险/边界 |
|---|---|---|---|
| Baysor (no-prior) | 贝叶斯混合模型（细胞=空间高斯×组成多项式）+ MRF 空间约束 | 边界分子概率归属；细胞数比图像法多近一倍 | 慢；大细胞过分割；scale 参数敏感；转录同质区不可靠 |
| ClusterMap | 邻域基因组成 + 密度峰聚类 | 可再分核/胞质亚结构 | 密度参数敏感 |
| BOMS | 空间-NGE 联合域 mean-shift | 快（MERFISH 4.4 min vs Baysor 30 min）、3 参数 | 同质区弱 |
| SSAM/Sparcle | 核密度 / Dirichlet 过程混合 | 无图像可用 | 边界粗糙 |

要点：纯表达路径的边界是概率性的；务必输出/使用分子 assignment 置信度，下游过滤低置信分子。

## C. ssDNA 图像 + 表达联合（推荐主路径）

| 方法 | 联合机制 | 优势 | 风险/边界 |
|---|---|---|---|
| CellBin 全流程 (MLCG) | 图像核分割为锚点；核 mask 内 GMM 拟合分子，核外分子按后验归入最可能细胞 | 一站式、固定超参、大芯片、官方生态 | 依赖配准与核分割质量 |
| SCS（STOmics 经典） | ssDNA 图像梯度 + 表达信号自适应 watershed | Stereo-seq 原生、可作对照 | 已被 CellBin 取代 |
| Baysor + 图像先验 | 图像分割作 prior，BMM+MRF 优化边界 | 定量置信度、边界精修 | 区域级使用；先验置信度需调 |
| BIDCell | 表达栅格图（基因=通道）+ 核 mask + marker 先验，自监督 UNet3+，6 个生物学损失 | 细胞类型特异形态（elongated vs 圆形）；表达纯度最高 | 需核分割与 marker 先验；推理慢；基因数受限 |
| Cellpose 双通道 | ssDNA=核通道，全基因表达密度图=胞质通道 | 最简单的联合 | 表达密度受 RNA 扩散模糊边界 |
| Bering / Segger | 图嵌入 / GNN 联合分割+注释 | 端到端、可扩展 | 需训练或迁移学习 |

### 组合范式（研究级推荐）
1. Cellpose-SAM/Mesmer 在 `*_regist.tif` 上出**核 mask**；
2a. 快速路线：核 mask 直接进 StereoMap → SAW realign（MLCG 完成边界扩展）；
2b. 精修路线：核 mask 作 Baysor prior（confidence 0.2 起步），或 BIDCell 自监督训练，输出全细胞 mask 再走 realign；
3. 用评估清单定量对比各路线。

## D. 选型速查

- 只想快速出结果：SAW 自动 CellBin。
- 图像差/自动失败：StereoMap 手动配准 + Cellpose-SAM + realign。
- 无图像：Baysor（预算足）/ BOMS（要快）。
- 追求论文级边界：核分割 + Baysor prior / BIDCell，配扩散校正（见 rna-diffusion.md）。
- 跨切片/跨平台/到图谱对齐：STalign（见 registration.md）。
