#!/usr/bin/env python
"""模块一-训练: 在多组 (ssDNA, 表达矩阵, 可靠细胞列表) 上训练核识别参数。

用法:
  python nuclei_train.py --manifest train.tsv --out params_nuclei.json [--device cuda]

manifest 格式: 每行一个数据集, 三列以制表符分隔: ssdna图  表达矩阵  可靠细胞列表
输出: params_nuclei.json (最优参数) + train_report.json (全部网格得分)。
"""
import argparse
import json
import os
import time


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def read_manifest(path):
    ds = []
    with open(path) as f:
        for ln in f:
            ln = ln.strip()
            if not ln or ln.startswith("#"):
                continue
            parts = ln.split("\t")
            if len(parts) < 3:
                parts = ln.split()
            ds.append((parts[0], parts[1], parts[2]))
    return ds


def main():
    ap = argparse.ArgumentParser(description="ssDNA 核识别参数训练")
    ap.add_argument("--manifest", required=True, help="训练清单 (ssdna<TAB>matrix<TAB>reliable 每行一组)")
    ap.add_argument("--out", required=True, help="输出参数 JSON")
    ap.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    ap.add_argument("--seed-backend", default="skimage", choices=["skimage", "cellpose"])
    ap.add_argument("--quick", action="store_true", help="小网格快速训练")
    args = ap.parse_args()

    from scell import nuclei_model as nm

    datasets = read_manifest(args.manifest)
    log(f"训练数据集: {len(datasets)} 组")
    grid = None
    if args.quick:
        grid = [nm.NucleiParams(thr_factor=tf, min_distance=md)
                for tf in (0.85, 1.0) for md in (5, 8)]
    best, records = nm.train(datasets, grid=grid, backend=args.seed_backend,
                             device=args.device, log=log)
    best.save(args.out)
    rep = os.path.splitext(args.out)[0] + "_report.json"
    with open(rep, "w") as f:
        json.dump(records, f, indent=2)
    log(f"最优参数: {best} -> {args.out}; 网格报告 -> {rep}")


if __name__ == "__main__":
    main()
