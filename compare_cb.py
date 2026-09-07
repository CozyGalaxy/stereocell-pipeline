#!/usr/bin/env python
"""CellBender 去除 ambient RNA 前后对比。

输入:
  --before  分割流程输出的 cells_x_genes.h5ad (CellBender 前)
  --after   cellbender remove-background 输出的 cb_clean.h5ad (后)
  --markers 可选: marker 基因列表 (每行一个), 计算特征基因表达比例变化
输出 (outdir):
  cb_compare_summary.json   总体指标
  cb_per_cell.csv           逐细胞 UMI/marker 比例 前后对照
  cb_per_gene_removed.csv   逐基因被去除的 UMI 比例 (ambient 程度排序)
"""
import argparse, json, os
import numpy as np
import pandas as pd


def load_h5ad(path):
    import anndata as ad
    a = ad.read_h5ad(path)
    return a


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--before", required=True)
    ap.add_argument("--after", required=True)
    ap.add_argument("--markers", default=None)
    ap.add_argument("--outdir", required=True)
    a = ap.parse_args()
    os.makedirs(a.outdir, exist_ok=True)

    b = load_h5ad(a.before)
    cb = load_h5ad(a.after)

    # CellBender 输出可能只保留判定为细胞的 barcode; 按 obs_names 对齐
    # before 的细胞名带 sample 前缀; after 的 barcode 来自 cellbender_raw.h5ad,
    # 命名一致 (sample.ID / sample.pseudoK)
    common = b.obs_names.intersection(cb.obs_names)
    if len(common) == 0:
        # 兼容: cellbender 可能去掉前缀或改写 barcode, 尝试末段数字匹配
        b_tail = pd.Index([s.rsplit(".", 1)[-1] for s in b.obs_names])
        cb_tail = pd.Index([s.rsplit(".", 1)[-1] for s in cb.obs_names])
        common_tail = b_tail.intersection(cb_tail)
        if len(common_tail) == 0:
            raise SystemExit("前后结果无共同细胞, 请检查 barcode 命名")
        b_sub = b[b_tail.isin(common_tail)]
        cb_sub = cb[cb_tail.isin(common_tail)]
        # 按 tail 对齐顺序
        b_sub = b_sub[pd.Index([s.rsplit(".", 1)[-1] for s in b_sub.obs_names])
                      .get_indexer(common_tail)]
        cb_sub = cb_sub[pd.Index([s.rsplit(".", 1)[-1] for s in cb_sub.obs_names])
                        .get_indexer(common_tail)]
    else:
        b_sub = b[common]
        cb_sub = cb[common]
    n = b_sub.shape[0]
    print(f"共同细胞: {n}")

    Xb = b_sub.X.tocsr()
    Xa = cb_sub.X.tocsr()
    # 基因对齐: cellbender 输出基因集与输入一致 (raw h5ad 的 var)
    # before (cells_x_genes) 与 after (cellbender_raw) 基因集应相同; 若不同则按名对齐
    if list(b_sub.var_names) != list(cb_sub.var_names):
        idx_b = pd.Index(b_sub.var_names).get_indexer(cb_sub.var_names)
        Xb = Xb[:, idx_b]

    umi_b = np.asarray(Xb.sum(1)).ravel()
    umi_a = np.asarray(Xa.sum(1)).ravel()

    # 逐基因去除比例 (衡量各基因 ambient 程度)
    g_b = np.asarray(Xb.sum(0)).ravel()
    g_a = np.asarray(Xa.sum(0)).ravel()
    removed_frac = np.where(g_b > 0, 1 - g_a / np.maximum(g_b, 1e-9), 0)
    per_gene = pd.DataFrame({
        "gene": list(cb_sub.var_names), "umi_before": g_b, "umi_after": g_a,
        "removed_frac": removed_frac,
    }).sort_values("removed_frac", ascending=False)
    per_gene.to_csv(os.path.join(a.outdir, "cb_per_gene_removed.csv"), index=False)

    # marker 基因比例
    summary = {
        "n_cells_common": int(n),
        "umi_before_median": float(np.median(umi_b)),
        "umi_after_median": float(np.median(umi_a)),
        "umi_removed_total_frac": float(1 - umi_a.sum() / max(umi_b.sum(), 1)),
        "genes_most_ambient_top10": per_gene.head(10)["gene"].tolist(),
    }
    df = pd.DataFrame({"cell": list(cb_sub.obs_names),
                       "umi_before": umi_b, "umi_after": umi_a})

    if a.markers:
        with open(a.markers) as f:
            markers = {ln.strip() for ln in f if ln.strip()}
        vnames = pd.Index(cb_sub.var_names)
        mk = vnames.isin(markers)
        n_hit = int(mk.sum())
        print(f"marker 基因命中 {n_hit}/{len(markers)}")
        if n_hit:
            mb = np.asarray(Xb[:, mk].sum(1)).ravel()
            ma = np.asarray(Xa[:, mk].sum(1)).ravel()
            df["marker_umi_before"] = mb
            df["marker_umi_after"] = ma
            df["marker_frac_before"] = mb / np.maximum(umi_b, 1)
            df["marker_frac_after"] = ma / np.maximum(umi_a, 1)
            summary["markers_hit"] = n_hit
            summary["marker_frac_before_median"] = float(np.median(df["marker_frac_before"]))
            summary["marker_frac_after_median"] = float(np.median(df["marker_frac_after"]))
            summary["marker_frac_gain"] = (
                summary["marker_frac_after_median"]
                - summary["marker_frac_before_median"])

    df.to_csv(os.path.join(a.outdir, "cb_per_cell.csv"), index=False)
    with open(os.path.join(a.outdir, "cb_compare_summary.json"), "w") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
