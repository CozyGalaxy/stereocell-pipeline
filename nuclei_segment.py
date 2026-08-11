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
    args = ap.parse_args()

    from scell import nuclei_model as nm, seeds

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
    log(f"图像 {img.shape} dtype={img.dtype}")
    mask = nm.apply(img, p, backend=args.seed_backend, device=args.device)
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
    seeds.write_overlay(img, mask, os.path.join(args.outdir, "nuclei_overlay.png"))
    p.save(os.path.join(args.outdir, "params_used.json"))
    log(f"完成。输出: {args.outdir}")


if __name__ == "__main__":
    main()
