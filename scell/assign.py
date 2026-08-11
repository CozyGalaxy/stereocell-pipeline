"""边际拒绝 GMM-EM 分子归属 (全向量化 numpy 后端; torch CUDA 后端可选)。

模型: 分子 m 来自细胞 j 的似然 = N(x_m; c_j, σ²I) · π_j(gene_m);
      背景类 = w_b · b(gene) / Area (空间均匀)。
接受规则: conf > conf_thr 且 conf > P(背景) 且 conf − 第二候选后验 > margin_i(所属最优细胞)。
σ 取全局估计(不逐细胞截断候选) —— "缩小扩散因子"由 margin_i 实现。
"""
import numpy as np
from scipy.spatial import cKDTree


def build_candidates(mol_xy, centers, r):
    """返回 (cand_mol, cand_cell): 候选 (分子, 细胞) 对, 按分子排序。"""
    tree = cKDTree(centers)
    lists = tree.query_ball_point(mol_xy, r=r)
    counts = np.fromiter(map(len, lists), dtype=np.int64, count=len(lists))
    cand_mol = np.repeat(np.arange(len(lists)), counts)
    cand_cell = np.concatenate([np.asarray(l, dtype=np.int32) for l in lists]) \
        if counts.sum() else np.zeros(0, np.int32)
    order = np.argsort(cand_mol, kind="stable")
    return cand_mol[order], cand_cell[order]


def init_pi(nuclei_mask, ix, iy, gene_codes, n_genes, n_cells, ambient_gene, min_nuc_mols=5):
    """逐细胞初始基因谱: 核内分子基因分布; 核内分子太少的细胞用背景谱。"""
    pi = np.tile(ambient_gene, (n_cells, 1)).astype(np.float64)
    lab = nuclei_mask[iy, ix]
    cnt = np.zeros((n_cells, n_genes))
    m = lab > 0
    if m.any():
        np.add.at(cnt, (lab[m] - 1, gene_codes[m]), 1.0)
    has = cnt.sum(1) >= min_nuc_mols
    pi[has] = cnt[has] + 0.5
    pi[has] /= pi[has].sum(1, keepdims=True)
    return pi


def estimate_ambient(gene_codes, mid, in_territory):
    """背景基因谱: 领地外分子的基因分布。"""
    out = ~in_territory
    if out.sum() < 1000:
        out = np.ones_like(in_territory)   # 退化时用全体
    b = np.bincount(gene_codes[out], weights=mid[out], minlength=int(gene_codes.max()) + 1) + 0.5
    return b / b.sum()


def em_assign(mol_xy, gene_codes, mid, centers, pi, ambient_gene, sigma, r_cand,
              margin_i, conf_thr=0.55, w_b=0.15, n_iter=12, device="cpu",
              cand_mol=None, cand_cell=None, chunk=4_000_000, log=None):
    """返回 assign(int32, 0=背景/拒收), conf(float32)。

    E 步全向量化: 候选对按 (分子, -似然) 排序后 reduceat 得 top1/top2 与分母;
    M 步一次 np.add.at 聚合全部细胞的基因计数。
    """
    n_mol, n_cells = len(mol_xy), len(centers)
    if cand_mol is None:
        cand_mol, cand_cell = build_candidates(mol_xy, centers, r_cand)
        if log:
            log(f"候选对 {len(cand_mol)} (r_cand={r_cand:.1f}px, "
                f"平均 {len(cand_mol) / max(n_mol, 1):.2f} 候选/分子)")
    area = float(mol_xy[:, 0].max() + 1) * float(mol_xy[:, 1].max() + 1)
    cm_xy = mol_xy[cand_mol]
    d2 = ((cm_xy - centers[cand_cell]) ** 2).sum(1).astype(np.float64)
    spat = np.exp(-d2 / (2 * sigma ** 2)) / (2 * np.pi * sigma ** 2)
    gg_c = gene_codes[cand_mol]

    use_torch = device == "cuda"
    if use_torch:
        import torch
        dev = torch.device("cuda")
        t_spat = torch.tensor(spat, dtype=torch.float64, device=dev)
        t_cm = torch.tensor(cand_mol, dtype=torch.int64, device=dev)
        t_cc = torch.tensor(cand_cell, dtype=torch.int64, device=dev)
        t_gg = torch.tensor(gg_c, dtype=torch.int64, device=dev)

    assign = np.zeros(n_mol, np.int32)
    conf = np.zeros(n_mol, np.float32)
    norm = 2 * np.pi * sigma ** 2

    for it in range(n_iter):
        num_b = w_b * ambient_gene[gene_codes] / area
        if use_torch:
            t_pi = torch.tensor(pi, dtype=torch.float64, device=dev)
            lj = t_spat * t_pi[t_cc, t_gg]
            order = torch.argsort(-lj)          # 按似然降序
            # 同分子内按似然降序: 先按分子稳定排序再按 -lj 稳定排序
            lj_s = lj[order]
            cm_s = t_cm[order]
            cc_s = t_cc[order]
            o2 = torch.argsort(cm_s, stable=True)
            lj_s, cm_s, cc_s = lj_s[o2], cm_s[o2], cc_s[o2]
            cm_np = cm_s.cpu().numpy()
            bound = np.flatnonzero(np.r_[True, cm_np[1:] != cm_np[:-1]])
            lj_np = lj_s.cpu().numpy()
            cc_np = cc_s.cpu().numpy()
        else:
            lj = spat * pi[cand_cell, gg_c]
            order = np.lexsort((-lj, cand_mol))  # 分子升序, 同分子内似然降序
            cm_np = cand_mol[order]
            lj_np = lj[order]
            cc_np = cand_cell[order]
            bound = np.flatnonzero(np.r_[True, cm_np[1:] != cm_np[:-1]])
        seg_len = np.diff(np.r_[bound, len(cm_np)])
        den = num_b[cm_np[bound]] + np.add.reduceat(lj_np, bound)
        m1 = lj_np[bound]
        a1 = cc_np[bound]
        has2 = seg_len > 1
        m2 = np.zeros(len(bound))
        m2[has2] = lj_np[bound[has2] + 1]
        mol_idx = cm_np[bound]

        conf_blk = m1 / den
        post_b = num_b[mol_idx] / den
        second = m2 / den
        mg = margin_i[a1]
        keep = (conf_blk > conf_thr) & (conf_blk > post_b) & ((conf_blk - second) > mg)
        assign[mol_idx] = np.where(keep, a1 + 1, 0)
        conf[mol_idx] = np.where(keep, conf_blk, 0)

        # M 步: 更新 pi (向量化)
        m = assign > 0
        cnt = np.zeros((n_cells, pi.shape[1]))
        np.add.at(cnt, (assign[m] - 1, gene_codes[m]), mid[m].astype(np.float64))
        has = cnt.sum(1) >= 10
        pi[has] = cnt[has] + 0.5
        pi[has] /= pi[has].sum(1, keepdims=True)
        if log:
            log(f"  EM iter {it + 1}/{n_iter}: 归属 {(assign > 0).sum()}/{n_mol}")
    return assign, conf
