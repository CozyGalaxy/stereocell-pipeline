#!/usr/bin/env python
"""StereoCell 细胞分割流程主入口。

用法:
  python run_pipeline.py --ssdna ssDNA.tif --matrix matrix.csv.gz \
      --reliable reliable.txt --outdir out/ [--device cuda] [--seed-backend cellpose]
  python run_pipeline.py --smoke-test --outdir /tmp/scell_smoke

分步: seeds -> diffuse -> assign -> export; 中间产物存在即跳过 (--force 重算)。
"""
import argparse
import json
import os
import time

import numpy as np
import tifffile


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def main():
    ap = argparse.ArgumentParser(description="StereoCell 细胞分割流程")
    ap.add_argument("--ssdna")
    ap.add_argument("--matrix")
    ap.add_argument("--reliable", default=None)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--smoke-test", action="store_true")
    ap.add_argument("--seed-backend", default="skimage", choices=["skimage", "cellpose"])
    ap.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    ap.add_argument("--tile", type=int, default=4096)
    ap.add_argument("--nuc-radius", type=float, default=5)
    ap.add_argument("--kappa", type=float, default=0.9)
    ap.add_argument("--conf-thr", type=float, default=0.55)
    ap.add_argument("--margin", type=float, default=0.20)
    ap.add_argument("--dense-margin", type=float, default=0.35)
    ap.add_argument("--min-nuc-mols", type=int, default=5)
    ap.add_argument("--em-iter", type=int, default=12)
    ap.add_argument("--chunk", type=int, default=2_000_000)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    if args.smoke_test:
        from scell import simulate
        pre = os.path.join(args.outdir, "smoke")
        simulate.make_synth(pre)
        args.ssdna = pre + "_ssDNA.tif"
        args.matrix = pre + "_matrix.csv"
        args.reliable = pre + "_reliable.txt"
        log("冒烟测试: 已生成合成三件套")

    from scell import io as scio, seeds, diffusion, assign as scassign, export

    # ---------- 读入 ----------
    log("加载表达矩阵 ...")
    expr = scio.load_expression(args.matrix)
    reliable = scio.load_reliable(args.reliable)
    log(f"分子行数={expr['n_rows']}, 基因数={len(expr['genes'])}, 可靠细胞数={len(reliable)}")

    p_nucmask = os.path.join(args.outdir, "01_nuclei_mask.tif")
    p_seedqc = os.path.join(args.outdir, "02_seed_qc.csv")
    p_fit = os.path.join(args.outdir, "03_diffusion_fit.png")
    p_params = os.path.join(args.outdir, "04_cell_params.csv")
    p_assign = os.path.join(args.outdir, "05_assign.npz")
    p_cellmask = os.path.join(args.outdir, "06_cell_mask.tif")

    # ---------- Step 1: 核分割 ----------
    if os.path.exists(p_nucmask) and not args.force:
        nuclei = tifffile.imread(p_nucmask)
        log("Step1 跳过: 核 mask 已存在")
    else:
        log(f"Step1: ssDNA 核分割 (backend={args.seed_backend}, device={args.device})")
        img = tifffile.imread(args.ssdna)
        nuclei = seeds.segment_nuclei(img, backend=args.seed_backend, tile=args.tile,
                                      nuc_radius=args.nuc_radius, device=args.device)
        tifffile.imwrite(p_nucmask, nuclei.astype(np.int32))
        seeds.write_overlay(img, nuclei, os.path.join(args.outdir, "seeds_overlay.png"))
        log(f"Step1 完成: 检出核 {nuclei.max()} 个, seeds_overlay 已输出")
    nuclei = nuclei.astype(np.int32)
    ids, areas, cents = seeds.nuclei_stats(nuclei)
    r_nuc_med = float(np.median(np.sqrt(areas / np.pi)))

    # 种子 QC: 核内分子数 / 匹配预分配 label / 是否可靠
    ix = np.clip(expr["x"], 0, nuclei.shape[1] - 1)
    iy = np.clip(expr["y"], 0, nuclei.shape[0] - 1)
    mol_nuc = nuclei[iy, ix]
    if not (os.path.exists(p_seedqc) and not args.force):
        with open(p_seedqc, "w") as f:
            f.write("seed_id,area_px,n_mols_in_nucleus,matched_prelab,reliable\n")
            for k, sid in enumerate(ids):
                m = mol_nuc == sid
                labs = expr["label"][m]
                labs = labs[labs > 0]
                pre = ""
                rel = 0
                if len(labs):
                    vals, cnt = np.unique(labs, return_counts=True)
                    pre = f"{vals[cnt.argmax()]:.0f}"
                    rel = int(str(int(vals[cnt.argmax()])) in reliable)
                f.write(f"{sid},{areas[k]},{int(m.sum())},{pre},{rel}\n")
        log(f"种子 QC 写出: {p_seedqc}")

    # ---------- Step 2: 扩散估计 + 逐细胞参数 ----------
    umi = np.zeros(nuclei.shape, np.float32)
    np.add.at(umi, (iy, ix), expr["mid"])
    lam, r95, D_ISO, isolated, _, _ = diffusion.estimate_diffusion(
        umi, cents, r_nuc_med, out_png=p_fit)
    log(f"Step2: λ={lam:.1f}px, r95={r95:.1f}px, D_ISO={D_ISO:.1f}px, "
        f"孤立核 {isolated.sum()}/{len(ids)}")
    tree, d_nn = diffusion.local_density(cents)
    R_i, sigma_arr, margin_i, crowded = diffusion.cell_params(
        d_nn, r95, r_nuc_med, D_ISO, kappa=args.kappa,
        margin0=args.margin, dense_margin=args.dense_margin)
    with open(p_params, "w") as f:
        f.write("seed_id,x,y,d_nn,R_i,sigma,margin_i,crowded\n")
        for k, sid in enumerate(ids):
            f.write(f"{sid},{cents[k][0]:.1f},{cents[k][1]:.1f},{d_nn[k]:.1f},"
                    f"{R_i[k]:.1f},{sigma_arr[k]:.2f},{margin_i[k]:.2f},{int(crowded[k])}\n")

    # ---------- Step 3: EM 归属 ----------
    # 候选半径密度护栏 (候选截断仅作效率护栏, 分割决策仍由 margin_i 控制)
    d_med = float(np.median(d_nn)) if len(d_nn) else r95
    r_cand = float(min(r95, max(4 * r_nuc_med, 1.5 * d_med)))
    if os.path.exists(p_assign) and not args.force:
        z = np.load(p_assign)
        assign, conf = z["assign"], z["conf"]
        log("Step3 跳过: 归属结果已存在")
    else:
        log(f"Step3: 边际拒绝 EM (device={args.device}, iter={args.em_iter})")
        ambient = scassign.estimate_ambient(expr["gene_codes"], expr["mid"],
                                            np.isin(mol_nuc, ids))
        pi = scassign.init_pi(nuclei, ix, iy, expr["gene_codes"],
                              len(expr["genes"]), len(ids), ambient,
                              min_nuc_mols=args.min_nuc_mols)
        assign, conf = scassign.em_assign(
            np.stack([expr["x"], expr["y"]], 1).astype(np.float32),
            expr["gene_codes"], expr["mid"], cents, pi, ambient,
            sigma=float(sigma_arr[0]), r_cand=r_cand,
            margin_i=margin_i, conf_thr=args.conf_thr,
            n_iter=args.em_iter, device=args.device, chunk=args.chunk, log=log)
        np.savez_compressed(p_assign, assign=assign, conf=conf)
    n_assigned = int((assign > 0).sum())
    log(f"Step3 完成: 归属 {n_assigned}/{expr['n_rows']} 分子 "
        f"({n_assigned / expr['n_rows']:.1%}), 中位置信 {np.median(conf[conf > 0]) if (conf > 0).any() else 0:.3f}")

    # ---------- Step 4: 领地栅格化 + 输出 ----------
    log("Step4: 领地栅格化与导出")
    cell_mask = export.rasterize_cells(umi, nuclei, cents, R_i)
    tifffile.imwrite(p_cellmask, cell_mask)
    img = tifffile.imread(args.ssdna)
    seeds.write_overlay(img, cell_mask, os.path.join(args.outdir, "cells_overlay.png"))

    out_matrix = os.path.join(args.outdir, "matrix_updated.csv.gz")
    scio.write_updated_matrix(args.matrix, out_matrix, assign)
    cell_ids = ids
    prefix = os.path.join(args.outdir, "cells_x_genes")
    made = scio.save_cell_by_gene(prefix, cell_ids, expr["genes"],
                                  expr["gene_codes"], expr["mid"], assign, conf)

    export.qc_report(os.path.join(args.outdir, "qc_report.json"),
                     n_molecules=expr["n_rows"], n_genes=len(expr["genes"]),
                     n_seeds=int(len(ids)), n_reliable=len(reliable),
                     lambda_px=lam, r95_px=r95, D_ISO_px=D_ISO,
                     frac_isolated=float(isolated.mean()),
                     frac_mols_assigned=n_assigned / expr["n_rows"],
                     median_conf=float(np.median(conf[conf > 0])) if (conf > 0).any() else 0.0,
                     outputs={"updated_matrix": out_matrix, "cell_by_gene": made,
                              "cell_mask": p_cellmask})
    log(f"全部完成。输出目录: {args.outdir}")


if __name__ == "__main__":
    main()
