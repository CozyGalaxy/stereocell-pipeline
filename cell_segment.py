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
    ap.add_argument("--pseudo-mode", choices=["adaptive", "legacy"], default="adaptive",
                    help="adaptive=ROI+自适应边界+空间分层; legacy=v1.4-v1.6 旧行为")
    ap.add_argument("--pseudo-margin", type=int, default=0,
                    help="细胞领地外扩 px; adaptive 默认 0=0.25*lambda, legacy 默认 0=5")
    ap.add_argument("--pseudo-max", type=int, default=50000, help="伪细胞数上限")
    ap.add_argument("--pseudo-min-umi", type=int, default=1, help="伪细胞最少 UMI")
    ap.add_argument("--pseudo-grid", type=int, default=4, help="空间分层每维窗口数")
    ap.add_argument("--roi-masks", default=None,
                    help="00_roi_masks.npz; 缺省尝试核 mask 同目录, 再自动重建")
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
    # v1.6.0: 并入核形态/亮度 QC 列 (peak_norm/snr/contrast/morph_flag 等)
    import pandas as pd
    _cq = pd.read_csv(os.path.join(args.outdir, "cell_qc.csv"))
    _mq = morph.rename(columns={"label": "cell_id"})[
        ["cell_id"] + [c for c in ("area", "circularity", "eccentricity",
                                   "max_intensity", "mean_intensity", "peak_norm",
                                   "bg_local", "snr", "contrast", "morph_flag")
                       if c in morph.columns]]
    _cq = _cq.merge(_mq, on="cell_id", how="left")
    _cq.to_csv(os.path.join(args.outdir, "cell_qc.csv"), index=False)
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
    # v1.6.1: assign 是位置序号 (1..N 对应 ids), 映射回真实核 label,
    # 保证 matrix_cell_id 的 cell_id 与 cell_qc/cells_x_genes/cell_pixel_map 一致
    _pos2label = np.zeros(len(ids) + 1, np.int32)
    _pos2label[1:] = ids
    scio.write_updated_matrix(args.matrix, out_matrix, _pos2label[assign_f],
                              sample=args.sample)

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
    cb_h5ad = cb_pseudo_summary = None
    if args.cellbender:
        log("Step7: CellBender 输入构建 (伪空液滴)")
        from scipy import sparse as _sp
        a_all = res["assign"]
        keep_m = a_all > 0
        real_mat = _sp.csr_matrix(
            (expr["mid"][keep_m].astype(np.float32),
             (a_all[keep_m] - 1, expr["gene_codes"][keep_m])),
            shape=(len(ids), len(expr["genes"])))
        roi_full = excl = None
        if args.pseudo_mode == "adaptive":
            roi_full, excl = _pseudo_region_masks(args, img, expr, log)
        pmat, pmeta = pseudo.build_pseudo_cells(
            res["cell_mask"], expr["x"], expr["y"], expr["mid"],
            expr["gene_codes"], len(expr["genes"]),
            size=args.pseudo_size, margin=args.pseudo_margin,
            max_cells=args.pseudo_max, min_umi=args.pseudo_min_umi, log=log,
            roi_mask=roi_full, excl_mask=excl, lam=res["lam"], r95=res["r95"],
            mode=args.pseudo_mode, grid=args.pseudo_grid)
        real_names = [f"{args.sample}.{int(c)}" if args.sample else str(int(c))
                      for c in ids]
        cb_h5ad = os.path.join(args.outdir, "cellbender_raw.h5ad")
        pseudo.write_cellbender_input(cb_h5ad, real_mat, real_names,
                                      expr["genes"], pmat, args.sample)
        # 伪细胞坐标及空间 QC 表
        import pandas as pd
        n_pseudo = len(pmeta["n_umi"])
        names = [f"{args.sample}.pseudo{k+1}" if args.sample else f"pseudo{k+1}"
                 for k in range(n_pseudo)]
        pd.DataFrame({
            "pseudo_id": names, "cx": pmeta["bin_x"], "cy": pmeta["bin_y"],
            "n_umi": pmeta["n_umi"], "bin_size": pmeta["bin_size"],
            "dist_to_cell_px": pmeta["dist_to_cell_px"],
            "distance_stratum": pmeta["distance_stratum"],
            "local_cell_frac": pmeta["local_cell_frac"],
            "spatial_window": pmeta["spatial_window"],
            "density_stratum": pmeta["density_stratum"],
        }).to_csv(os.path.join(args.outdir, "pseudo_cells.csv"), index=False)
        cb_pseudo_summary = {
            k: pmeta[k] for k in ("mode", "margin_px", "bin_size", "roi_frac",
                                  "n_candidates", "n_selected")}
        cb_pseudo_summary["distance_counts"] = {
            s: int((pmeta["distance_stratum"] == s).sum()) for s in ("near", "mid", "far")}
        log(f"CellBender 输入: {cb_h5ad} ({len(ids)} 真细胞 + {n_pseudo} 伪空液滴)")

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
                     cellbender_pseudo=cb_pseudo_summary,
                     outputs={"matrix": out_matrix, "cell_by_gene": made,
                              "nucleus_morphology": os.path.join(args.outdir, "02_nucleus_morphology.csv"),
                              "knn5_npz": knn_npz, "knn5_tsv": knn_tsv})
    log(f"全部完成。输出目录: {args.outdir}")


def _pseudo_region_masks(args, img, expr, log):
    """读取核分割 ROI 缓存; 旧结果无缓存时从当前矩阵/图像重建。"""
    from scell import roi as scroi
    path = args.roi_masks
    if path is None and args.nuclei_mask:
        candidate = os.path.join(os.path.dirname(args.nuclei_mask), "00_roi_masks.npz")
        if os.path.exists(candidate):
            path = candidate
    if path is not None:
        if not os.path.exists(path):
            raise FileNotFoundError(f"ROI 掩膜不存在: {path}")
        z = np.load(path, allow_pickle=False)
        roi_small = z["roi_small"]
        bin_size = int(z["bin_size"]) if "bin_size" in z.files else 48
        roi_full = scroi.upsample_mask(roi_small, img.shape, bin_size)
        excl = z["excl"] if "excl" in z.files else np.zeros(img.shape, bool)
        if excl.shape == roi_small.shape:
            excl = scroi.upsample_mask(excl, img.shape, bin_size)
        if excl.shape != img.shape:
            raise ValueError(f"ROI 缓存伪影掩膜形状不符: {excl.shape} vs {img.shape}")
        log(f"伪细胞区域: 复用 {path} (ROI={roi_full.mean():.1%})")
        return roi_full, excl

    log("伪细胞区域: 无 ROI 缓存, 从当前表达矩阵重建")
    bin_size = 48
    H, W = img.shape
    nb_y, nb_x = (H + bin_size - 1) // bin_size, (W + bin_size - 1) // bin_size
    hist = np.zeros((nb_y, nb_x), np.float64)
    bx = np.clip(expr["x"] // bin_size, 0, nb_x - 1)
    by = np.clip(expr["y"] // bin_size, 0, nb_y - 1)
    np.add.at(hist, (by, bx), expr["mid"])
    roi_small = scroi.chip_roi(hist)
    roi_small = scroi.erode_roi_small(roi_small, scroi.roi_erosion_px(bin_size), bin_size)
    roi_full = scroi.upsample_mask(roi_small, img.shape, bin_size)
    excl = scroi.artifact_mask(img, roi_full)
    np.savez_compressed(os.path.join(args.outdir, "00_pseudo_roi_masks.npz"),
                        roi_small=roi_small, excl=excl, bin_size=bin_size)
    log(f"伪细胞区域: ROI={roi_full.mean():.1%}, 伪影占ROI="
        f"{excl.sum() / max(roi_full.sum(), 1):.1%}")
    return roi_full, excl


def _reindex(assign, pass_mask):
    """把过滤后的 assign 重新编号为 1..n_pass (与输出细胞顺序一致)。"""
    remap = np.zeros(len(pass_mask) + 1, np.int32)
    remap[1:][pass_mask] = np.arange(1, pass_mask.sum() + 1)
    return remap[assign]


if __name__ == "__main__":
    main()
