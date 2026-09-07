#!/usr/bin/env python
"""模块二-推理: 基于核种子 + RNA 扩散模型的细胞分割, 含 QC 过滤与多核细胞评估。

模式:
  训练参数模式: --params params_cell.json
  无督导模式:   不传 --params, 全部参数自适应估计

用法:
  python cell_segment.py --ssdna ssDNA.tif --matrix matrix.txt \
      [--reliable solid.cell.list] [--params params_cell.json] \
      [--nuclei-mask out_nuclei/nuclei_mask.tif] --outdir out_cells/ [--device cuda]

输出 (outdir):
  01_nuclei_mask.tif / seeds_overlay.png   核种子与叠加图
  03_diffusion_fit.png                     扩散衰减拟合曲线
  04_cell_params.csv                       逐细胞参数 (R_i, margin, crowded)
  cell_mask.tif                            最终细胞领地 mask
  cells_overlay.png                        ssDNA + 1 像素红圈(最终细胞)
  cell_pixel_map.tsv.gz                    文本1a: x, y, cell_id (像素级)
  matrix_cell_id.txt.gz                    文本1b: 原矩阵 label 列更新为 cell_id
  syncytium_pairs.tsv                      文本2: 相近细胞对同源概率
  cell_qc.csv                              逐细胞 UMI/基因数/mito/ribo/过滤
  cells_x_genes.h5ad                       细胞×基因稀疏矩阵 (过滤后)
  qc_report.json                           汇总报告
"""
import argparse
import gzip
import json
import os
import time

import numpy as np
import tifffile


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def main():
    ap = argparse.ArgumentParser(description="StereoCell 细胞分割")
    ap.add_argument("--ssdna", required=True)
    ap.add_argument("--matrix", required=True)
    ap.add_argument("--reliable", default=None)
    ap.add_argument("--params", default=None, help="训练参数 JSON; 缺省=无督导")
    ap.add_argument("--nuclei-mask", default=None, help="复用模块一的核 mask")
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    ap.add_argument("--seed-backend", default="skimage", choices=["skimage", "cellpose"])
    ap.add_argument("--kappa", type=float, default=None,
                    help="无督导模式下覆盖 kappa (1.0 = 严格中线划分)")
    ap.add_argument("--sample", default=None,
                    help="文库名: 输出矩阵/h5ad/5NN 中细胞命名为 sample.ID (非细胞=0)")
    # ---- CellBender 输入 (伪空液滴) ----
    ap.add_argument("--cellbender", action="store_true",
                    help="输出 CellBender raw 输入 h5ad (真细胞+伪空液滴)")
    ap.add_argument("--pseudo-size", type=int, default=0,
                    help="伪细胞 bin 边长 px (0=自动: sqrt(细胞领地面积中位数))")
    ap.add_argument("--pseudo-margin", type=int, default=5,
                    help="细胞领地外扩 px, 该范围内不生成伪细胞")
    ap.add_argument("--pseudo-max", type=int, default=50000, help="伪细胞数上限")
    ap.add_argument("--pseudo-min-umi", type=int, default=1, help="伪细胞最少 UMI")
    args = ap.parse_args()

    from scell import io as scio, seeds, export
    from scell import cell_model as cm, qc as scqc, syncytium, pseudo

    os.makedirs(args.outdir, exist_ok=True)
    if args.params:
        p = cm.CellParams.load(args.params)
        log(f"模式: 训练参数 ({args.params})")
    else:
        p = cm.CellParams()
        if args.kappa:
            p.kappa = args.kappa
        log("模式: 无督导 (自适应默认参数)")

    log("加载表达矩阵 ...")
    expr = scio.load_expression(args.matrix)
    reliable = scio.load_reliable(args.reliable)
    log(f"分子行数={expr['n_rows']} 基因数={len(expr['genes'])} 可靠细胞数={len(reliable)}")

    img = tifffile.imread(args.ssdna)
    nuclei = None
    if args.nuclei_mask:
        nuclei = tifffile.imread(args.nuclei_mask).astype(np.int32)
        log(f"复用核 mask: {args.nuclei_mask} ({nuclei.max()} 核)")

    log("Step1-3: 核种子 → 扩散估计 → 边际拒绝 EM")
    res = cm.segment_cells(img, expr, p, device=args.device,
                           seed_backend=args.seed_backend, nuclei=nuclei, log=log,
                           fit_png=os.path.join(args.outdir, "03_diffusion_fit.png"))
    ids, cents = res["ids"], res["cents"]
    n_assigned = int((res["assign"] > 0).sum())
    log(f"归属 {n_assigned}/{expr['n_rows']} 分子 ({n_assigned / expr['n_rows']:.1%})")

    # 中间产物
    tifffile.imwrite(os.path.join(args.outdir, "01_nuclei_mask.tif"),
                     res["nuclei"].astype(np.int32))
    # 核一致性质量评估: 同物种核大小应均一、近似椭圆; 异常核分色标注
    morph = seeds.nucleus_morphology(res["nuclei"].astype(np.int32), img)
    flags, (alo, ahi) = seeds.classify_nuclei(morph)
    morph["morph_flag"] = flags
    morph.to_csv(os.path.join(args.outdir, "02_nucleus_morphology.csv"), index=False)
    log(f"核形态 QC: 合法面积 {alo:.0f}-{ahi:.0f}px, 异常 {int((flags != 'ok').sum())}/{len(flags)} ("
        + ", ".join(f"{t}={int((flags == t).sum())}" for t in sorted(set(flags)) if t != "ok")
        + ")")
    colors = {int(l): seeds.MORPH_FLAG_COLORS[f]
              for l, f in zip(morph["label"].values, flags) if f != "ok"}
    seeds.write_overlay(img, res["nuclei"], os.path.join(args.outdir, "seeds_overlay.png"),
                        label_colors=colors)
    tifffile.imwrite(os.path.join(args.outdir, "cell_mask.tif"), res["cell_mask"])
    seeds.write_overlay(img, res["cell_mask"], os.path.join(args.outdir, "cells_overlay.png"),
                        label_colors=colors)
    with open(os.path.join(args.outdir, "04_cell_params.csv"), "w") as f:
        f.write("seed_id,x,y,d_nn,R_i,margin_i,crowded\n")
        for k, sid in enumerate(ids):
            f.write(f"{sid},{cents[k][0]:.1f},{cents[k][1]:.1f},{res['d_nn'][k]:.1f},"
                    f"{res['R_i'][k]:.1f},{res['margin_i'][k]:.2f},{int(res['crowded'][k])}\n")

    # ---------- QC: mito / ribo / 过滤 ----------
    log("Step4: 细胞 QC (mito-/Ribo- 比例)")
    metrics = scqc.cell_metrics(expr["genes"], expr["gene_codes"], expr["mid"],
                                res["assign"], ids)
    conf_mean = np.bincount(np.clip(res["assign"], 0, len(ids)),
                            weights=res["conf"], minlength=len(ids) + 1)[1:] / \
        np.maximum(np.bincount(np.clip(res["assign"], 0, len(ids)),
                               minlength=len(ids) + 1)[1:], 1)
    pass_mask = scqc.filter_cells(metrics, mito_max=p.mito_max)
    scqc.write_qc_csv(os.path.join(args.outdir, "cell_qc.csv"),
                      metrics, pass_mask, conf_mean=conf_mean)
    log(f"QC: {pass_mask.sum()}/{len(ids)} 细胞通过 (mito>{p.mito_max:.0%} 过滤 "
        f"{(~pass_mask).sum()} 个)")

    # 过滤: 被滤细胞的分子重新标记为 0
    assign_f = res["assign"].copy()
    filtered_seeds = np.nonzero(~pass_mask)[0] + 1
    if len(filtered_seeds):
        assign_f[np.isin(assign_f, filtered_seeds)] = 0

    # ---------- 文本输出 ----------
    log("Step5: 写出文本结果")
    # 文本1a: 像素级 cell_id
    with gzip.open(os.path.join(args.outdir, "cell_pixel_map.tsv.gz"), "wt") as f:
        f.write("x\ty\tcell_id\n")
        cm_mask = res["cell_mask"]
        ys, xs = np.nonzero(cm_mask)
        for k in range(0, len(ys), 1_000_000):
            e = min(k + 1_000_000, len(ys))
            f.write("".join(f"{xs[i]}\t{ys[i]}\t{cm_mask[ys[i], xs[i]]}\n"
                            for i in range(k, e)))
    # 文本1b: 更新矩阵
    out_matrix = os.path.join(args.outdir, "matrix_cell_id.txt.gz")
    scio.write_updated_matrix(args.matrix, out_matrix, assign_f, sample=args.sample)

    # 文本2: 多核细胞概率
    log("Step6: 多核细胞(合胞体)概率评估")
    cg = syncytium.cell_gene_matrix(expr["gene_codes"], expr["mid"],
                                    res["assign"], len(ids), len(expr["genes"]))
    pairs = syncytium.pair_probabilities(cg, cents, ids)
    syncytium.write_pairs(os.path.join(args.outdir, "syncytium_pairs.tsv"), pairs)
    n_high = sum(1 for r in pairs if r["prob"] >= 0.5)
    log(f"合胞体候选对 {len(pairs)} 个, 其中概率≥0.5: {n_high}")

    # 细胞×基因矩阵 (过滤后)
    made = scio.save_cell_by_gene(
        os.path.join(args.outdir, "cells_x_genes"),
        [sid for k, sid in enumerate(ids) if pass_mask[k]],
        expr["genes"],
        expr["gene_codes"][assign_f > 0],
        expr["mid"][assign_f > 0],
        _reindex(assign_f, pass_mask)[assign_f > 0],
        res["conf"][assign_f > 0], sample=args.sample)

    # ---------- 5NN 距离稀疏矩阵 ----------
    knn_npz = os.path.join(args.outdir, "cell_5nn_dist.npz")
    knn_tsv = os.path.join(args.outdir, "cell_5nn_dist.tsv.gz")
    export.knn_distance_matrix(cents, ids, k=5, out_npz=knn_npz,
                               out_tsv=knn_tsv, sample=args.sample)
    log(f"5NN 距离矩阵写出 {knn_npz} / {knn_tsv}")

    # ---------- CellBender 输入: 真细胞(未过滤) + 伪空液滴 ----------
    cb_h5ad = None
    if args.cellbender:
        log("Step7: CellBender 输入构建 (伪空液滴)")
        from scipy import sparse as _sp
        a_all = res["assign"]
        keep_m = a_all > 0
        real_mat = _sp.csr_matrix(
            (expr["mid"][keep_m].astype(np.float32),
             (a_all[keep_m] - 1, expr["gene_codes"][keep_m])),
            shape=(len(ids), len(expr["genes"])))
        pmat, pmeta = pseudo.build_pseudo_cells(
            res["cell_mask"], expr["x"], expr["y"], expr["mid"],
            expr["gene_codes"], len(expr["genes"]),
            size=args.pseudo_size, margin=args.pseudo_margin,
            max_cells=args.pseudo_max, min_umi=args.pseudo_min_umi, log=log)
        real_names = [f"{args.sample}.{int(c)}" if args.sample else str(int(c))
                      for c in ids]
        cb_h5ad = os.path.join(args.outdir, "cellbender_raw.h5ad")
        pseudo.write_cellbender_input(cb_h5ad, real_mat, real_names,
                                      expr["genes"], pmat, args.sample)
        # 伪细胞坐标表
        with open(os.path.join(args.outdir, "pseudo_cells.csv"), "w") as f:
            f.write("pseudo_id,cx,cy,n_umi,bin_size\n")
            for k in range(len(pmeta["n_umi"])):
                nm = f"{args.sample}.pseudo{k+1}" if args.sample else f"pseudo{k+1}"
                f.write(f"{nm},{pmeta['bin_x'][k]},{pmeta['bin_y'][k]},"
                        f"{pmeta['n_umi'][k]:.0f},{pmeta['bin_size']}\n")
        log(f"CellBender 输入: {cb_h5ad} ({len(ids)} 真细胞 + "
            f"{len(pmeta['n_umi'])} 伪空液滴)")

    export.qc_report(os.path.join(args.outdir, "qc_report.json"),
                     mode="trained" if args.params else "unsupervised",
                     n_molecules=expr["n_rows"], n_genes=len(expr["genes"]),
                     n_seeds=int(len(ids)), n_reliable=len(reliable),
                     lambda_px=res["lam"], r95_px=res["r95"], D_ISO_px=res["D_ISO"],
                     r_cand_px=res["r_cand"],
                     frac_isolated=float(res["isolated"].mean()),
                     frac_mols_assigned=n_assigned / expr["n_rows"],
                     n_cells_pass=int(pass_mask.sum()),
                     n_cells_filtered=int((~pass_mask).sum()),
                     mito_max=p.mito_max,
                     n_syncytium_pairs=len(pairs), n_syncytium_high=n_high,
                     n_nuclei_flagged=int((flags != "ok").sum()),
                     sample=args.sample,
                     cellbender_raw=cb_h5ad,
                     outputs={"matrix": out_matrix, "cell_by_gene": made,
                              "nucleus_morphology": os.path.join(args.outdir, "02_nucleus_morphology.csv"),
                              "knn5_npz": knn_npz, "knn5_tsv": knn_tsv})
    log(f"全部完成。输出目录: {args.outdir}")


def _reindex(assign, pass_mask):
    """把过滤后的 assign 重新编号为 1..n_pass (与输出细胞顺序一致)。"""
    remap = np.zeros(len(pass_mask) + 1, np.int32)
    remap[1:][pass_mask] = np.arange(1, pass_mask.sum() + 1)
    return remap[assign]


if __name__ == "__main__":
    main()
