"""CellBender remove-background 输入构建: 非细胞区伪细胞 (pseudo empty droplets)。

CellBender 假设空液滴与真细胞液滴捕获面积/效率一致, 空液滴计数 = 纯 ambient 采样。
因此伪细胞策略:
  1) 尺寸自动匹配真实细胞领地面积中位数 (边长 = sqrt(area_med)), 不默认固定 25px;
  2) 只排除与细胞领地 (膨胀 margin 后) 重叠的 bin —— 扩散晕(halo)内的 bin 保留,
     因为扩散 RNA 本身就是污染细胞的 ambient 来源;
  3) 候选 bin 超过 max_cells 时随机子采样 (可复现 seed)。
"""
import numpy as np


def build_pseudo_cells(cell_mask, x, y, mid, gene_codes, n_genes,
                       size=0, margin=5, max_cells=50000, min_umi=1,
                       seed=0, log=print):
    """非细胞区网格化伪细胞, 聚合其分子为 pseudo×gene CSR。

    cell_mask: int32 领地 label 图 (0=非细胞); x,y,mid,gene_codes: 分子级数组;
    size: bin 边长 px (0=自动: sqrt(领地面积中位数), 限制在 [15,200]);
    margin: 领地向外膨胀 px, 防止真细胞 RNA 漏入 ambient 估计;
    min_umi: 伪细胞最少 UMI (全 0 的 bin 对 CellBender 无信息)。
    返回 (pseudo_csr (n_pseudo, n_genes) float32, meta: dict of arrays)。
    """
    from scipy import ndimage as ndi, sparse

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
    log(f"  伪细胞 bin 边长: {size}px (margin={margin}px)")

    # ---- 2) 领地膨胀后的禁选 bin (bin 分辨率块归约, 避免全尺寸 bid 图) ----
    forbidden = ndi.binary_dilation(cell_mask > 0, iterations=max(1, margin))
    ph, pw = -H % size, -W % size
    fpad = np.pad(forbidden, ((0, ph), (0, pw)))
    fbin = fpad.reshape(fpad.shape[0] // size, size,
                        fpad.shape[1] // size, size).any(axis=(1, 3))
    del forbidden, fpad
    nbins_y, nbins_x = fbin.shape

    # ---- 3) 候选分子: bin 未被领地占据 (分子是否已归属不限 —
    #    领地外 assign>0 的分子同样是扩散 ambient 的组成部分, 但本实现只取领地外,
    #    这些分子绝大多数 assign==0) ----
    bx, by = x // size, y // size
    keep = ~fbin[by, bx]
    bid = by[keep].astype(np.int64) * nbins_x + bx[keep]
    gc = gene_codes[keep].astype(np.int64)
    m = mid[keep].astype(np.float64)

    # ---- 4) 按 (bin, gene) 聚合 MIDCount: 排序 + reduceat ----
    key = bid * np.int64(n_genes) + gc
    order = np.argsort(key, kind="mergesort")
    key_s, m_s = key[order], m[order]
    del key, bid, gc, m
    starts = np.flatnonzero(np.r_[True, key_s[1:] != key_s[:-1]])
    ukey = key_s[starts]
    umi = np.add.reduceat(m_s, starts).astype(np.float32)
    del key_s, m_s, order
    ubid, ugene = ukey // n_genes, ukey % n_genes

    # ---- 5) bin 级过滤: min_umi + 上限子采样 ----
    bin_umi = np.zeros(nbins_y * nbins_x, np.float64)
    np.add.at(bin_umi, ubid, umi)
    ok_bins = np.nonzero(bin_umi >= min_umi)[0]
    if len(ok_bins) > max_cells:
        rng = np.random.default_rng(seed)
        ok_bins = np.sort(rng.choice(ok_bins, max_cells, replace=False))
        log(f"  候选伪细胞超上限, 子采样至 {max_cells}")
    bset = np.zeros(nbins_y * nbins_x, bool)
    bset[ok_bins] = True
    sel = bset[ubid]
    ubid, ugene, umi = ubid[sel], ugene[sel], umi[sel]

    lut = np.zeros(nbins_y * nbins_x, np.int64)
    lut[ok_bins] = np.arange(len(ok_bins))
    rows = lut[ubid]
    mat = sparse.csr_matrix((umi, (rows, ugene)),
                            shape=(len(ok_bins), n_genes))

    by_c, bx_c = ok_bins // nbins_x, ok_bins % nbins_x
    meta = {
        # 边缘 bin 中心可能超出图像边界, 截断到合法范围
        "bin_x": np.minimum(bx_c * size + size // 2, W - 1).astype(np.int32),
        "bin_y": np.minimum(by_c * size + size // 2, H - 1).astype(np.int32),
        "n_umi": np.asarray(mat.sum(1)).ravel(),
        "bin_size": size,
    }
    log(f"  伪细胞 {len(ok_bins)} 个, UMI 中位数 "
        f"{np.median(meta['n_umi']) if len(ok_bins) else 0:.0f}")
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
