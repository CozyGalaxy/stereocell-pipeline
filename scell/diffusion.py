"""局部密度、扩散长度估计与逐细胞自适应参数。

策略(经模拟原型验证):
- d_i = 核质心 1-NN 距离; 孤立细胞: d_i > D_ISO = 6 × 核半径中位数
- 扩散长度 λ: 孤立细胞(优先可靠细胞)径向 UMI 衰减拟合 exp(-r/λ), r95 = λ·ln20
- 领地半径上限 R_i = min(r95, κ·d_i/2) —— 只用于 mask 栅格化, 不用于 EM 候选截断
- 拥挤区通过更大的 margin_i 拒收模糊分子, 而非缩小候选半径
"""
import numpy as np
from scipy.spatial import cKDTree


def local_density(centers):
    tree = cKDTree(centers)
    d = tree.query(centers, k=2)[0][:, 1]
    return tree, d


def estimate_diffusion(umi, centers, r_nuc_med, reliable_idx=None, r_max=None, out_png=None):
    """径向衰减拟合。返回 (lam, r95, D_ISO, isolated_mask, r_mid, radial)。"""
    D_ISO = 6 * r_nuc_med
    tree, d_nn = local_density(centers)
    isolated = d_nn > D_ISO
    if reliable_idx is not None and len(reliable_idx):
        use = isolated.copy()
        use[reliable_idx] = True   # 可靠细胞即使略拥挤也可用于拟合(其谱可信)
        use &= isolated            # 但仍要求几何孤立, 防止污染衰减曲线
        if use.sum() < 10:         # 可靠孤立太少则退回全部孤立
            use = isolated
    else:
        use = isolated
    if r_max is None:
        r_max = int(max(40, 2 * np.median(d_nn)))
    H, W = umi.shape
    bins = np.arange(0, r_max + 2, 2)
    rad_sum = np.zeros(len(bins) - 1); rad_cnt = np.zeros(len(bins) - 1)
    fit_cells = centers[use]
    if len(fit_cells) > 300:   # 子采样, 防止大规模芯片径向扫描过慢
        sel = np.random.default_rng(0).choice(len(fit_cells), 300, replace=False)
        fit_cells = fit_cells[sel]
    for c in fit_cells:
        x0, y0 = int(c[0]), int(c[1])
        y_lo, y_hi = max(0, y0 - r_max), min(H, y0 + r_max + 1)
        x_lo, x_hi = max(0, x0 - r_max), min(W, x0 + r_max + 1)
        sub = umi[y_lo:y_hi, x_lo:x_hi]
        yy_s, xx_s = np.mgrid[y_lo:y_hi, x_lo:x_hi]
        rr = np.sqrt((xx_s - c[0]) ** 2 + (yy_s - c[1]) ** 2)
        idx = np.clip(np.digitize(rr, bins) - 1, 0, len(bins) - 2)
        np.add.at(rad_sum, idx, sub); np.add.at(rad_cnt, idx, 1)
    radial = rad_sum / np.maximum(rad_cnt, 1)
    r_mid = (bins[:-1] + bins[1:]) / 2
    # 环境背景地板: 远区径向密度的中位数, 扣除后再拟合, 防止平台期拉平斜率
    n_tail = max(1, len(radial) // 4)
    floor = np.median(radial[-n_tail:][radial[-n_tail:] > 0]) if (radial[-n_tail:] > 0).any() else 0.0
    radial_c = radial - floor
    fit = (r_mid > r_nuc_med) & (radial_c > 0.2 * np.nanmax(radial_c)) & np.isfinite(radial_c)
    if fit.sum() < 3:
        fit = (r_mid > r_nuc_med) & (radial > 0) & np.isfinite(radial)
    lam = -np.polyfit(r_mid[fit], np.log(radial_c[fit] if radial_c[fit].min() > 0 else radial[fit]), 1)[0] ** -1
    # 密度护栏: 扩散尾不应超过典型细胞间距的一半, 防止致密区拟合被邻居拉平
    d_med = float(np.median(d_nn))
    lam = float(min(lam, 0.45 * d_med))
    lam = max(lam, 1.0)
    r95 = float(lam * np.log(20))
    if out_png:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.semilogy(r_mid, np.maximum(radial, 1e-3), "o", ms=3, label="radial UMI")
        ax.semilogy(r_mid[fit], np.exp(np.polyval(np.polyfit(r_mid[fit], np.log(radial[fit]), 1), r_mid[fit])),
                    "-", label=f"exp(-r/λ), λ={lam:.1f}px, r95={r95:.1f}px")
        ax.legend(); ax.set_xlabel("r (px)"); ax.set_ylabel("UMI density")
        fig.savefig(out_png, bbox_inches="tight"); plt.close(fig)
    return float(lam), r95, D_ISO, isolated, r_mid, radial


def cell_params(d_nn, r95, r_nuc_med, D_ISO, kappa=0.9, margin0=0.20, dense_margin=0.35):
    """逐细胞参数: R_i(栅格化上限), sigma_i, margin_i, crowded 标记。"""
    R_i = np.minimum(r95, kappa * d_nn / 2)
    R_i = np.maximum(R_i, r_nuc_med * 1.2)
    sigma = r95 / 2.45
    crowded = d_nn <= D_ISO
    margin_i = np.where(crowded, dense_margin, margin0)
    return R_i, np.full_like(R_i, sigma, dtype=float), margin_i, crowded
