#!/usr/bin/env python
"""模块二-训练: 在多组数据上训练细胞分割参数 (EM 置信/边际阈值)。

用法:
  python cell_train.py --manifest train.tsv --out params_cell.json [--device cuda]

manifest 同 nuclei_train.py (ssdna<TAB>matrix<TAB>reliable 每行一组)。
可用 --params-nuclei 传入模块一训练好的核参数作为基础。
"""
import argparse
import json
import os
import time
from dataclasses import asdict


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def main():
    ap = argparse.ArgumentParser(description="细胞分割参数训练")
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--params-nuclei", default=None, help="模块一核参数 JSON")
    ap.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    ap.add_argument("--seed-backend", default="skimage", choices=["skimage", "cellpose"])
    ap.add_argument("--quick", action="store_true", help="小网格快速训练")
    ap.add_argument("--em-iter", type=int, default=None, help="覆盖 EM 迭代数(提速)")
    args = ap.parse_args()

    from scell import cell_model as cm
    from nuclei_train import read_manifest

    base = cm.CellParams()
    if args.params_nuclei:
        import json as j
        base.nuclei = j.load(open(args.params_nuclei))
    if args.em_iter:
        base.em_iter = args.em_iter
    datasets = read_manifest(args.manifest)
    log(f"训练数据集: {len(datasets)} 组")
    best, records = cm.train(datasets, base=base, backend=args.seed_backend,
                             device=args.device, log=log, quick=args.quick)
    best.save(args.out)
    rep = os.path.splitext(args.out)[0] + "_report.json"
    with open(rep, "w") as f:
        json.dump(records, f, indent=2)
    log(f"最优参数: conf_thr={best.conf_thr} margin={best.margin} "
        f"dense_margin={best.dense_margin} -> {args.out}")


if __name__ == "__main__":
    main()
