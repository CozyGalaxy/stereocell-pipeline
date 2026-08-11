"""细胞质量指标: 线粒体/核糖体基因比例与过滤。

基因命名约定(表达矩阵 geneID 列): 线粒体基因以 'mito-' 起始, 核糖体蛋白基因以 'Ribo-' 起始。
过滤规则: 线粒体表达比例 > mito_max (默认 10%) 的细胞判为低质量, 予以过滤。
"""
import numpy as np


def gene_class_masks(genes):
    g = np.asarray([str(x) for x in genes])
    return g, np.char.startswith(g, "mito-"), np.char.startswith(g, "Ribo-")


def cell_metrics(genes, gene_codes, mid, assign, cell_ids):
    """逐细胞指标。assign: 每分子细胞序号(1..N 对应 cell_ids, 0=未归属)。
    返回 dict 数组字段: cell_id, total_umi, n_genes, mito_frac, ribo_frac。
    """
    from scipy import sparse
    _, is_mito, is_ribo = gene_class_masks(genes)
    n_cells = len(cell_ids)
    keep = assign > 0
    rows = assign[keep] - 1
    tot = np.bincount(rows, weights=mid[keep], minlength=n_cells).astype(float)
    mito = np.bincount(rows[is_mito[gene_codes[keep]]],
                       weights=mid[keep][is_mito[gene_codes[keep]]],
                       minlength=n_cells).astype(float)
    ribo = np.bincount(rows[is_ribo[gene_codes[keep]]],
                       weights=mid[keep][is_ribo[gene_codes[keep]]],
                       minlength=n_cells).astype(float)
    # n_genes: 每细胞检测到的基因数
    ck = sparse.csr_matrix((np.ones(keep.sum(), np.int8),
                            (rows, gene_codes[keep])),
                           shape=(n_cells, len(genes)))
    ck.data[:] = 1
    n_genes = np.asarray((ck > 0).sum(1)).ravel()
    return {
        "cell_id": np.asarray(cell_ids),
        "total_umi": tot,
        "n_genes": n_genes,
        "mito_frac": mito / np.maximum(tot, 1),
        "ribo_frac": ribo / np.maximum(tot, 1),
    }


def filter_cells(metrics, mito_max=0.10, min_umi=0, min_genes=0):
    """返回 pass_mask(bool)。规则: mito_frac <= mito_max 且 UMI/基因数达标。"""
    return ((metrics["mito_frac"] <= mito_max)
            & (metrics["total_umi"] >= min_umi)
            & (metrics["n_genes"] >= min_genes))


def write_qc_csv(path, metrics, pass_mask, conf_mean=None):
    with open(path, "w") as f:
        hdr = "cell_id,total_umi,n_genes,mito_frac,ribo_frac"
        if conf_mean is not None:
            hdr += ",mean_conf"
        hdr += ",pass\n"
        f.write(hdr)
        for k, cid in enumerate(metrics["cell_id"]):
            row = f"{cid},{metrics['total_umi'][k]:.0f},{metrics['n_genes'][k]}," \
                  f"{metrics['mito_frac'][k]:.4f},{metrics['ribo_frac'][k]:.4f}"
            if conf_mean is not None:
                row += f",{conf_mean[k]:.4f}"
            row += f",{int(pass_mask[k])}\n"
            f.write(row)
