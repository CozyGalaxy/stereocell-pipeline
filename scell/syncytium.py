"""多核细胞(合胞体)评估: 相近细胞对来自同一多核细胞的概率。

判据(用户定义): ssDNA 上距离极近的细胞, 在表达矩阵上共享一个高表达基因 pool,
且表达模式相似 → 可能来自同一多核细胞。

特征:
  d         质心距离 (px)
  dist_sc   exp(-d / d0), d0 = 细胞半径中位数
  cos       log1p-CPM 表达向量的余弦相似度 (表达模式相似性)
  shared    top-N (默认 50) 高表达基因交集比例 (共享高表达 pool)
概率: 无标注数据, 采用无督导 logistic 组合
  z = w0 + w_cos·cos + w_sh·shared + w_d·dist_sc;  p = sigmoid(z)
默认权重由合成/经验设定, 可用 --weights 覆盖。
"""
import numpy as np
from scipy import sparse
from scipy.spatial import cKDTree


DEFAULT_WEIGHTS = dict(w0=-3.0, w_cos=3.5, w_sh=2.0, w_d=1.5)


def cell_gene_matrix(gene_codes, mid, assign, n_cells, n_genes):
    keep = assign > 0
    return sparse.csr_matrix(
        (mid[keep].astype(np.float64), (assign[keep] - 1, gene_codes[keep])),
        shape=(n_cells, n_genes))


def pair_probabilities(cg, centers, cell_ids, top_n=50, dist_factor=2.5,
                       weights=None, prob_min=0.05):
    """返回候选细胞对列表 (按概率降序):
    dict(cell_i, cell_j, dist, cos, shared, dist_sc, prob)。
    cg: cells×genes 稀疏矩阵; centers: 与 cell_ids 对齐的质心 (x,y)。
    """
    w = dict(DEFAULT_WEIGHTS)
    if weights:
        w.update(weights)
    n = cg.shape[0]
    # log1p-CPM 归一化
    sums = np.asarray(cg.sum(1)).ravel()
    norm = cg.multiply(1e4 / np.maximum(sums, 1)[:, None]).tocsr()
    norm.data = np.log1p(norm.data)
    # 余弦相似度 (L2 归一化后点积)
    l2 = np.sqrt(np.asarray(norm.multiply(norm).sum(1)).ravel())
    l2[l2 == 0] = 1.0
    normed = norm.multiply(1.0 / l2[:, None]).tocsr()
    # top-N 基因集合
    top_sets = []
    for i in range(n):
        row = cg.getrow(i)
        if row.nnz == 0:
            top_sets.append(set()); continue
        k = min(top_n, row.nnz)
        idx = row.indices[np.argpartition(-row.data, k - 1)[:k]]
        top_sets.append(set(idx.tolist()))
    # 距离尺度: 最近邻距离中位数估计典型细胞间距
    tree = cKDTree(centers)
    d_nn = tree.query(centers, k=2)[0][:, 1]
    d0 = float(np.median(d_nn)) / 2.0
    d_max = dist_factor * float(np.median(d_nn))
    pairs = tree.query_pairs(d_max, output_type="ndarray")
    out = []
    if len(pairs) == 0:
        return out
    for i, j in pairs:
        d = float(np.linalg.norm(centers[i] - centers[j]))
        cos = float(normed.getrow(i).multiply(normed.getrow(j)).sum())
        ti, tj = top_sets[i], top_sets[j]
        shared = len(ti & tj) / max(min(len(ti), len(tj)), 1)
        dist_sc = float(np.exp(-d / max(d0, 1e-6)))
        z = w["w0"] + w["w_cos"] * cos + w["w_sh"] * shared + w["w_d"] * dist_sc
        p = 1.0 / (1.0 + np.exp(-z))
        if p >= prob_min:
            out.append(dict(cell_i=str(cell_ids[i]), cell_j=str(cell_ids[j]),
                            dist=d, cos=cos, shared=shared, dist_sc=dist_sc, prob=p))
    out.sort(key=lambda r: -r["prob"])
    return out


def write_pairs(path, pairs):
    with open(path, "w") as f:
        f.write("cell_i\tcell_j\tdist_px\tcos_sim\tshared_top_frac\tdist_score\tsame_cell_prob\n")
        for r in pairs:
            f.write(f"{r['cell_i']}\t{r['cell_j']}\t{r['dist']:.1f}\t{r['cos']:.4f}\t"
                    f"{r['shared']:.4f}\t{r['dist_sc']:.4f}\t{r['prob']:.4f}\n")
