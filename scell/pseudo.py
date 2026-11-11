"""CellBender remove-background 输入构建: 非细胞区伪细胞 (pseudo empty droplets)。

adaptive 模式:
  1) 尺寸匹配真实细胞领地面积中位数;
  2) 仅在合法芯片 ROI 且非伪影区采样, 排除细胞领地及安全边界;
  3) 安全边界默认 0.25*lambda, 保留近端扩散晕;
  4) 超上限时按空间窗口 x 细胞距离 x 局部密度分层抽样, 近端晕区权重较高。
legacy 模式保留 v1.4--v1.6 行为: 固定 margin=5、全图候选、全局随机子采样。
"""
import numpy as np


def _stratified_choice(bins, strata, dist_cat, max_cells, rng):
    """按 strata 分组、距离加权且无放回抽样; 每组至少保留一个候选。"""
    groups, inv = np.unique(strata, return_inverse=True)
    counts = np.bincount(inv)
    # 近端/中间/远端权重 2/1/0.5; 空间窗口和密度只分层, 不额外加权。
    weights = np.asarray([2.0, 1.0, 0.5], float)[dist_cat]
    group_weight = np.bincount(inv, weights=weights)
    ng = len(groups)
    quotas = np.zeros(ng, np.int64)
    if max_cells < ng:
        p = group_weight / group_weight.sum()
        quotas[rng.choice(ng, max_cells, replace=False, p=p)] = 1
    else:
        quotas[:] = 1
        target = max_cells * group_weight / group_weight.sum()
        for _ in range(max_cells - ng):
            deficit = np.where(quotas < counts, target - quotas, -np.inf)
            g = int(np.argmax(deficit))
            if not np.isfinite(deficit[g]):
                break
            quotas[g] += 1
    selected = []
    for g, quota in enumerate(quotas):
        if quota <= 0:
            continue
        idx = np.flatnonzero(inv == g)
        if quota < len(idx):
            idx = rng.choice(idx, quota, replace=False)
        selected.append(bins[idx])
    return np.sort(np.concatenate(selected)) if selected else np.empty(0, np.int64)


def build_pseudo_cells(cell_mask, x, y, mid, gene_codes, n_genes,
                       size=0, margin=0, max_cells=50000, min_umi=1,
                       seed=0, log=print, roi_mask=None, excl_mask=None,
                       lam=None, r95=None, mode="adaptive", grid=4):
    """非细胞区网格化伪细胞, 聚合其分子为 pseudo×gene CSR。

    cell_mask: int32 领地 label 图 (0=非细胞); x,y,mid,gene_codes: 分子级数组;
    size: bin 边长 px (0=自动: sqrt(领地面积中位数), 限制在 [15,200]);
    margin: 领地外扩 px; adaptive 下 0=自动(0.25*lambda), legacy 下 0=5;
    roi_mask: 合法芯片区; excl_mask: 伪影排除区 (仅 adaptive 使用);
    mode: adaptive / legacy; grid: 空间分层每维窗口数。
    返回 (pseudo_csr, meta); meta 含逐伪细胞空间 QC 列及采样参数摘要。
    """
    from scipy import ndimage as ndi, sparse

    if mode not in ("adaptive", "legacy"):
        raise ValueError("pseudo mode 必须为 adaptive 或 legacy")
    grid = max(1, int(grid))
    H, W = cell_mask.shape
    # ---- 1) 自动尺寸: 领地面积中位数的平方根 ----
    if size <= 0:
        n_lab = int(cell_mask.max())
        if n_lab > 0:
            areas = np.bincount(cell_mask.ravel(), minlength=n_lab + 1)[1:]
            areas = areas[areas > 0]
            size = int(round(np.sqrt(np.median(areas)))) if len(areas) else 25
        else:
            size = 25
        size = int(np.clip(size, 15, 200))
    if margin <= 0:
        margin = int(np.clip(round(0.25 * lam), 3, 50)) \
            if mode == "adaptive" and lam is not None else 5
    log(f"  伪细胞 bin 边长: {size}px (mode={mode}, margin={margin}px)")

    ph, pw = -H % size, -W % size
    nbins_y, nbins_x = (H + ph) // size, (W + pw) // size
    cell_any = cell_frac = None
    if mode == "adaptive":
        # bin 级细胞距离/密度; 避免全分辨率 EDT 的额外数 GB 内存。
        cpad = np.pad(cell_mask > 0, ((0, ph), (0, pw)))
        cblock = cpad.reshape(nbins_y, size, nbins_x, size)
        cell_any = cblock.any(axis=(1, 3))
        cell_frac = cblock.mean(axis=(1, 3), dtype=np.float32)
        del cpad, cblock

    # ---- 2) 禁选 bin: 细胞领地+安全边界; adaptive 另排除片外/伪影 ----
    forbidden = ndi.binary_dilation(cell_mask > 0, iterations=max(1, margin))
    roi_frac = 1.0
    if mode == "adaptive":
        legal = np.ones((H, W), bool) if roi_mask is None else np.asarray(roi_mask, bool)
        if excl_mask is not None:
            legal = legal & ~np.asarray(excl_mask, bool)
        roi_frac = float(legal.mean())
        forbidden |= ~legal
        log(f"  伪细胞合法区域占全图 {roi_frac:.1%}")
    # adaptive 下图外 padding 同样禁选, 防止右/下边缘 partial bin 混入;
    # legacy 保留旧版 False padding 行为。
    fpad = np.pad(forbidden, ((0, ph), (0, pw)),
                  constant_values=(mode == "adaptive"))
    fbin = fpad.reshape(nbins_y, size, nbins_x, size).any(axis=(1, 3))
    del forbidden, fpad

    # ---- 3) 候选分子: 只取合法 bin; 不限分子原 assign ----
    bx, by = x // size, y // size
    keep = ~fbin[by, bx]
    bid = by[keep].astype(np.int64) * nbins_x + bx[keep]
    gc = gene_codes[keep].astype(np.int64)
    m = mid[keep].astype(np.float64)

    # ---- 4) 按 (bin, gene) 聚合 MIDCount ----
    key = bid * np.int64(n_genes) + gc
    order = np.argsort(key, kind="mergesort")
    key_s, m_s = key[order], m[order]
    del key, bid, gc, m
    if len(key_s):
        starts = np.flatnonzero(np.r_[True, key_s[1:] != key_s[:-1]])
        ukey = key_s[starts]
        umi = np.add.reduceat(m_s, starts).astype(np.float32)
    else:
        ukey = np.empty(0, np.int64)
        umi = np.empty(0, np.float32)
    del key_s, m_s, order
    ubid, ugene = ukey // n_genes, ukey % n_genes

    # ---- 5) bin 级过滤与空间分层抽样 ----
    bin_umi = np.zeros(nbins_y * nbins_x, np.float64)
    np.add.at(bin_umi, ubid, umi)
    candidate_bins = np.nonzero(bin_umi >= min_umi)[0]
    ok_bins = candidate_bins
    n_candidates = len(candidate_bins)
    dist_px = dist_cat = local_density = spatial = dense = None
    if mode == "adaptive" and n_candidates:
        by_c, bx_c = candidate_bins // nbins_x, candidate_bins % nbins_x
        dist_bin = ndi.distance_transform_edt(~cell_any)
        dist_px = dist_bin[by_c, bx_c] * size
        near_end = max(float(r95 or 3 * size), float(size))
        dist_cat = np.where(dist_px <= near_end, 0,
                            np.where(dist_px <= 2 * near_end, 1, 2)).astype(np.int8)
        local_bin = ndi.uniform_filter(cell_frac, size=3)
        local_density = local_bin[by_c, bx_c]
        gy = np.minimum(by_c * grid // nbins_y, grid - 1)
        gx = np.minimum(bx_c * grid // nbins_x, grid - 1)
        spatial = (gy * grid + gx).astype(np.int16)
        positive = local_density[local_density > 0]
        dense_cut = np.quantile(positive, 0.75) if len(positive) >= 20 else np.inf
        dense = (local_density > dense_cut).astype(np.int8)
        strata = ((spatial.astype(np.int64) * 3 + dist_cat) * 2 + dense)
        if n_candidates > max_cells:
            ok_bins = _stratified_choice(
                candidate_bins, strata, dist_cat, max_cells, np.random.default_rng(seed))
            keep_idx = np.searchsorted(candidate_bins, ok_bins)
            dist_px, dist_cat = dist_px[keep_idx], dist_cat[keep_idx]
            local_density, spatial, dense = (local_density[keep_idx],
                                             spatial[keep_idx], dense[keep_idx])
            log(f"  候选伪细胞 {n_candidates} 个, 空间分层抽样至 {len(ok_bins)}")
    elif n_candidates > max_cells:
        rng = np.random.default_rng(seed)
        ok_bins = np.sort(rng.choice(candidate_bins, max_cells, replace=False))
        log(f"  候选伪细胞 {n_candidates} 个, legacy 全局随机抽样至 {max_cells}")

    bset = np.zeros(nbins_y * nbins_x, bool)
    bset[ok_bins] = True
    sel = bset[ubid]
    ubid, ugene, umi = ubid[sel], ugene[sel], umi[sel]
    lut = np.zeros(nbins_y * nbins_x, np.int64)
    lut[ok_bins] = np.arange(len(ok_bins))
    rows = lut[ubid]
    mat = sparse.csr_matrix((umi, (rows, ugene)), shape=(len(ok_bins), n_genes))

    by_c, bx_c = ok_bins // nbins_x, ok_bins % nbins_x
    n = len(ok_bins)
    if mode == "legacy" or n == 0:
        dist_px = np.full(n, np.nan)
        dist_cat = np.full(n, -1, np.int8)
        local_density = np.full(n, np.nan)
        spatial = np.full(n, -1, np.int16)
        dense = np.full(n, -1, np.int8)
    dist_name = np.asarray(["near", "mid", "far"])
    meta = {
        "bin_x": np.minimum(bx_c * size + size // 2, W - 1).astype(np.int32),
        "bin_y": np.minimum(by_c * size + size // 2, H - 1).astype(np.int32),
        "n_umi": np.asarray(mat.sum(1)).ravel(),
        "bin_size": size,
        "dist_to_cell_px": np.asarray(dist_px, np.float32),
        "distance_stratum": (dist_name[np.clip(dist_cat, 0, 2)] if mode == "adaptive"
                             else np.full(n, "legacy")),
        "local_cell_frac": np.asarray(local_density, np.float32),
        "spatial_window": np.asarray(spatial),
        "density_stratum": np.asarray(dense),
        "mode": mode, "margin_px": int(margin), "roi_frac": roi_frac,
        "n_candidates": int(n_candidates), "n_selected": int(n),
    }
    log(f"  伪细胞 {n} 个, UMI 中位数 {np.median(meta['n_umi']) if n else 0:.0f}")
    return mat, meta


def write_cellbender_input(out_h5ad, real_mat, real_names, var_names,
                           pseudo_mat, sample, real_conf=None):
    """真细胞 + 伪细胞合并为 CellBender raw 输入 (h5ad; anndata 缺失时降级 mtx+tsv)。
    obs: droplet_type = cell / empty; 伪细胞命名 sample.pseudoK。
    """
    from scipy import sparse
    n_real, n_pseudo = real_mat.shape[0], pseudo_mat.shape[0]
    mat = sparse.vstack([real_mat.tocsr(), pseudo_mat.tocsr()], format="csr")
    names = list(real_names) + [
        f"{sample}.pseudo{k}" if sample else f"pseudo{k}"
        for k in range(1, n_pseudo + 1)]
    try:
        import anndata as ad
        adata = ad.AnnData(X=mat)
        adata.obs_names = names
        adata.var_names = list(var_names)
        adata.obs["droplet_type"] = ["cell"] * n_real + ["empty"] * n_pseudo
        adata.obs["is_pseudo"] = [False] * n_real + [True] * n_pseudo
        if real_conf is not None:
            adata.obs["mean_conf"] = np.concatenate(
                [np.asarray(real_conf, np.float32), np.zeros(n_pseudo, np.float32)])
        adata.write_h5ad(out_h5ad)
        return out_h5ad
    except ImportError:
        base = out_h5ad[:-5] if out_h5ad.endswith(".h5ad") else out_h5ad
        sparse.save_npz(base + ".npz", mat)
        with open(base + "_barcodes.tsv", "w") as f:
            f.write("\n".join(names) + "\n")
        with open(base + "_droplet_type.tsv", "w") as f:
            f.write("\n".join(["cell"] * n_real + ["empty"] * n_pseudo) + "\n")
        with open(base + "_features.tsv", "w") as f:
            f.write("\n".join(var_names) + "\n")
        return base + ".npz"
