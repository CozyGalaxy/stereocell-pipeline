#!/usr/bin/env python
"""子区域切分工具: 从全尺寸芯片数据裁剪试跑窗口 (ssDNA + 矩阵 + 可靠细胞列表)。

用途: 集群全尺寸运行前, 先用子区域校准参数与资源评估。

用法:
  # 自动选择最密窗口 (推荐, 覆盖聚集/稀疏各种场景可用 --mode median)
  python crop_region.py --ssdna big.tif --matrix big.txt.gz \
      --reliable solid.cell.list --outdir pilot/ --size 6000 --auto dense

  # 手动指定窗口左上角
  python crop_region.py --ssdna big.tif --matrix big.txt.gz \
      --outdir pilot/ --x 8000 --y 8000 --size 6000

输出 (outdir):
  crop_ssdna.tif / crop_matrix.txt.gz / crop_reliable.list / crop_manifest.tsv / crop_info.json
矩阵坐标以窗口左上角为原点重定基; 可靠细胞仅保留窗口内分子数 >= --min-mols 的细胞。
"""
import argparse
import gzip
import json
import os
import time

import numpy as np
import pandas as pd
import tifffile


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def _open(path):
    return gzip.open(path, "rt") if str(path).endswith(".gz") else open(path)


def scan_hist(matrix, bin_size=500, chunk=5_000_000):
    """第一遍流式扫描: 粗直方图 + 坐标范围。"""
    with _open(matrix) as fh:
        header = fh.readline()
    sep = "\t" if "\t" in header else ","
    hist = None
    xmax = ymax = 0
    for df in pd.read_csv(matrix, sep=sep, engine="c", compression="infer",
                          chunksize=chunk, usecols=["x", "y"]):
        xmax = max(xmax, int(df.x.max())); ymax = max(ymax, int(df.y.max()))
        h, _, _ = np.histogram2d(df.y, df.x,
                                 bins=[np.arange(0, ymax + bin_size, bin_size),
                                       np.arange(0, xmax + bin_size, bin_size)])
        hist = h if hist is None else hist + h
    return hist, xmax + 1, ymax + 1, sep


def pick_window(hist, W, H, size, mode):
    """按模式选窗口左上角。dense=最密 bin; median=非零 bin 密度中位; random=随机非零。"""
    n_bins_y, n_bins_x = hist.shape
    bs_y, bs_x = H / n_bins_y, W / n_bins_x
    if mode == "dense":
        iy, ix = np.unravel_index(np.argmax(hist), hist.shape)
    else:
        nz = hist[hist > 0]
        target = np.median(nz) if mode == "median" else nz[np.random.default_rng(0).integers(len(nz))]
        cand = np.argwhere(np.abs(hist - target) <= 0.05 * target)
        iy, ix = cand[len(cand) // 2]
    x0 = int(np.clip(ix * bs_x + bs_x / 2 - size / 2, 0, max(0, W - size)))
    y0 = int(np.clip(iy * bs_y + bs_y / 2 - size / 2, 0, max(0, H - size)))
    return x0, y0


def main():
    ap = argparse.ArgumentParser(description="子区域切分 (试跑校准)")
    ap.add_argument("--ssdna", required=True)
    ap.add_argument("--matrix", required=True)
    ap.add_argument("--reliable", default=None)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--size", type=int, default=6000)
    ap.add_argument("--x", type=int, default=None)
    ap.add_argument("--y", type=int, default=None)
    ap.add_argument("--auto", choices=["dense", "median", "random"], default=None)
    ap.add_argument("--min-mols", type=int, default=20,
                    help="可靠细胞在窗口内最少分子数 (低于则移出列表)")
    ap.add_argument("--chunk", type=int, default=5_000_000)
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    # ---- ssDNA 裁剪 (优先 memmap; 压缩 TIFF 回退整图读入) ----
    log("读取 ssDNA 尺寸 ...")
    try:
        img = tifffile.memmap(args.ssdna)
    except ValueError:
        img = tifffile.imread(args.ssdna)
    H, W = img.shape[-2], img.shape[-1]
    log(f"ssDNA: {W}x{H}")

    # ---- 确定窗口 ----
    if args.x is not None and args.y is not None:
        x0, y0 = args.x, args.y
    else:
        mode = args.auto or "dense"
        log(f"扫描矩阵粗直方图 (mode={mode}) ...")
        hist, Wm, Hm, sep = scan_hist(args.matrix, chunk=args.chunk)
        x0, y0 = pick_window(hist, Wm, Hm, args.size, mode)
    x0 = int(np.clip(x0, 0, max(0, W - args.size)))
    y0 = int(np.clip(y0, 0, max(0, H - args.size)))
    s = min(args.size, W - x0, H - y0)
    log(f"窗口: x [{x0},{x0+s}) y [{y0},{y0+s})")

    crop_img = np.asarray(img[y0:y0 + s, x0:x0 + s])
    p_img = os.path.join(args.outdir, "crop_ssdna.tif")
    tifffile.imwrite(p_img, crop_img)
    log(f"ssDNA 裁剪写出: {p_img}")

    # ---- 矩阵裁剪 (第二遍流式) + 窗口内逐 label 分子计数 ----
    with _open(args.matrix) as fh:
        header = fh.readline()
    sep = "\t" if "\t" in header else ","
    p_mat = os.path.join(args.outdir, "crop_matrix.txt.gz")
    label_counts = {}
    n_in = 0
    with gzip.open(p_mat, "wt") as fout:
        first = True
        # keep_default_na=False: "nan"/"null" 是合法基因名, 不得转 NaN 后写成空
        for df in pd.read_csv(args.matrix, sep=sep, engine="c", compression="infer",
                              chunksize=args.chunk, dtype={"geneID": str},
                              keep_default_na=False):
            m = (df.x >= x0) & (df.x < x0 + s) & (df.y >= y0) & (df.y < y0 + s)
            if not m.any():
                continue
            sub = df[m].copy()
            sub["x"] -= x0
            sub["y"] -= y0
            lab = sub["label"].astype(str).str.rsplit(".", n=1).str[-1]
            for k, v in lab.value_counts().items():
                if k not in ("0", "0.0", "nan"):
                    label_counts[k] = label_counts.get(k, 0) + int(v)
            sub.to_csv(fout, sep=sep, index=False, header=first)
            first = False
            n_in += len(sub)
    log(f"矩阵裁剪: {n_in} 行 -> {p_mat}")

    # ---- 可靠细胞列表裁剪 ----
    p_rel = None
    if args.reliable:
        p_rel = os.path.join(args.outdir, "crop_reliable.list")
        kept, dropped = 0, 0
        with open(args.reliable) as fin, open(p_rel, "w") as fout:
            for ln in fin:
                ln_s = ln.strip()
                if not ln_s or ln_s.startswith("#"):
                    continue
                tail = ln_s.rsplit(".", 1)[-1]
                if label_counts.get(tail, 0) >= args.min_mols:
                    fout.write(ln_s + "\n")
                    kept += 1
                else:
                    dropped += 1
        log(f"可靠细胞: 保留 {kept}, 移出 {dropped} -> {p_rel}")

    # ---- manifest + 信息 ----
    p_man = os.path.join(args.outdir, "crop_manifest.tsv")
    with open(p_man, "w") as f:
        f.write(f"{p_img}\t{p_mat}\t{p_rel or ''}\n")
    info = dict(x0=x0, y0=y0, size=s, mode=args.auto or "manual",
                rows=n_in, labels_in_window=len(label_counts),
                src_ssdna=args.ssdna, src_matrix=args.matrix)
    with open(os.path.join(args.outdir, "crop_info.json"), "w") as f:
        json.dump(info, f, ensure_ascii=False, indent=2)
    log(f"完成。manifest: {p_man} (可直接用于 nuclei_train/cell_train --manifest)")


if __name__ == "__main__":
    main()
