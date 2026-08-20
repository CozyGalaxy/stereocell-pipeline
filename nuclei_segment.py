#!/usr/bin/env python
"""模块一-推理: 在 ssDNA 图上识别细胞核。

支持两种模式:
  训练参数模式: --params params_nuclei.json (由 nuclei_train.py 生成)
  无督导模式:   不传 --params, 使用自适应默认参数 (Otsu + 致密回退)

用法:
  python nuclei_segment.py --ssdna ssDNA.tif --outdir out_nuclei/ [--params p.json]

输出 (outdir):
  nuclei_mask.tif        核 label mask (int32)
  nuclei_pixel_map.tsv.gz  文本: x, y, nucleus_id  (每个属于核的像素一行)
  nuclei_summary.csv     每个核: id, 质心, 面积, 平均强度
  nuclei_overlay.png     ssDNA 图 + 1 像素红圈
  params_used.json       实际使用的参数
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
    ap = argparse.ArgumentParser(description="ssDNA 细胞核识别")
    ap.add_argument("--ssdna", required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--params", default=None, help="训练参数 JSON; 缺省=无督导")
    ap.add_argument("--nuc-radius", type=float, default=None,
                    help="无督导模式下覆盖 min_distance")
    ap.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    ap.add_argument("--seed-backend", default="skimage", choices=["skimage", "cellpose"])
    ap.add_argument("--matrix", default=None,
                    help="表达矩阵 (可选): 提供时先做芯片 ROI 识别与伪影排除")
    ap.add_argument("--reliable", default=None,
                    help="可靠细胞列表 (可选): 校准峰值强度阈值")
    ap.add_argument("--no-roi", action="store_true", help="关闭芯片区域识别")
    ap.add_argument("--no-excl", action="store_true", help="关闭伪影排除")
    args = ap.parse_args()

    from scipy import ndimage as ndi
    from scell import nuclei_model as nm, seeds, roi as scroi, io as scio

    os.makedirs(args.outdir, exist_ok=True)
    if args.params:
        p = nm.NucleiParams.load(args.params)
        log(f"模式: 训练参数 ({args.params})")
    else:
        p = nm.NucleiParams()
        if args.nuc_radius:
            p.min_distance = max(1, int(args.nuc_radius))
        log("模式: 无督导 (自适应默认参数)")

    img = tifffile.imread(args.ssdna)
    if img.ndim > 2:
        img = img.squeeze()
    log(f"图像 {img.shape} dtype={img.dtype}")

    # 可选: 芯片 ROI + 伪影排除 + 可靠细胞校准
    roi_mask = excl_mask = reliable_xy = None
    if args.matrix and not args.no_roi:
        log("芯片区域识别 (表达密度) ...")
        hist = scroi.density_histogram(args.matrix, img.shape)
        roi_small = scroi.chip_roi(hist)
        roi_small = scroi.erode_roi_small(roi_small, scroi.roi_erosion_px(48), 48)
        roi_mask = scroi.upsample_mask(roi_small, img.shape, 48)
        log(f"  ROI 占全图 {roi_mask.mean():.1%}")
        if not args.no_excl:
            excl_mask = scroi.artifact_mask(img, roi_mask, nuc_radius=p.min_distance)
            log(f"  伪影排除区占 ROI {excl_mask.sum() / max(roi_mask.sum(), 1):.1%}")
    if args.reliable and args.matrix:
        expr = scio.load_expression(args.matrix)
        rel = scio.reliable_label_ids(scio.load_reliable(args.reliable))
        sel = np.isin(expr["label"], list(rel))
        if sel.any():
            labs = expr["label"][sel]
            _, inv = np.unique(labs, return_inverse=True)
            w = expr["mid"][sel].astype(np.float64)
            cx = np.bincount(inv, weights=expr["x"][sel] * w) / np.bincount(inv, weights=w)
            cy = np.bincount(inv, weights=expr["y"][sel] * w) / np.bincount(inv, weights=w)
            reliable_xy = np.stack([cx, cy], 1)
            log(f"可靠细胞质心: {len(reliable_xy)} 个 (峰值阈值校准)")

    mask = nm.apply(img, p, backend=args.seed_backend, device=args.device,
                    roi_mask=roi_mask, excl_mask=excl_mask, reliable_xy=reliable_xy,
                    log=log)
    tifffile.imwrite(os.path.join(args.outdir, "nuclei_mask.tif"), mask.astype(np.int32))

    ids, areas, cents = seeds.nuclei_stats(mask)
    log(f"检出核 {len(ids)} 个")

    # 文本输出: 每个属于核的像素 (流式, 内存安全)
    p_map = os.path.join(args.outdir, "nuclei_pixel_map.tsv.gz")
    import gzip
    with gzip.open(p_map, "wt") as f:
        f.write("x\ty\tnucleus_id\n")
        ys, xs = np.nonzero(mask)
        for k in range(0, len(ys), 1_000_000):
            sl = slice(k, k + 1_000_000)
            f.write("".join(f"{xs[i]}\t{ys[i]}\t{mask[ys[i], xs[i]]}\n"
                            for i in range(sl.start, min(sl.stop, len(ys)))))
    # 汇总
    with open(os.path.join(args.outdir, "nuclei_summary.csv"), "w") as f:
        f.write("nucleus_id,cx,cy,area_px,mean_intensity\n")
        for k, sid in enumerate(ids):
            m = mask == sid
            f.write(f"{sid},{cents[k][0]:.1f},{cents[k][1]:.1f},{areas[k]},"
                    f"{img[m].mean():.1f}\n")
    seeds.write_overlay(img, mask, os.path.join(args.outdir, "nuclei_overlay.png"),
                        excl_mask=excl_mask)
    p.save(os.path.join(args.outdir, "params_used.json"))
    log(f"完成。输出: {args.outdir}")


if __name__ == "__main__":
    main()
