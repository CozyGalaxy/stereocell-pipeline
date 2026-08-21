#!/usr/bin/env python
"""汇总 v1.2.0 批量回归的逐芯片 QC 报告为一张 TSV 对照表。

用法:
    python collect_qc.py --root /results/Apis_Dev_Stereocell/v1.2.0 --out qc_summary.tsv

遍历 root 下每个芯片子目录, 读取 qc_report.json / cell_qc.csv / syncytium_pairs.tsv,
输出每芯片一行: 核数、细胞数、分子归属率、ROI/排除占比、mito 过滤数、合胞体对数等。
"""
import argparse, json, os, sys
import pandas as pd


def chip_row(d):
    row = {"chip": os.path.basename(d)}
    qr = os.path.join(d, "qc_report.json")
    if os.path.isfile(qr):
        with open(qr) as f:
            r = json.load(f)
        for k in ("n_nuclei", "n_cells", "n_molecules", "assigned_frac",
                  "roi_frac", "excl_frac_of_roi", "lambda", "r95"):
            if k in r:
                row[k] = r[k]
    qc = os.path.join(d, "cell_qc.csv")
    if os.path.isfile(qc):
        df = pd.read_csv(qc)
        row["cells_pass"] = int(df.get("pass", pd.Series(dtype=bool)).sum()) if "pass" in df else len(df)
        row["cells_fail_mito"] = int((~df["pass"]).sum()) if "pass" in df else 0
        if "mito_frac" in df:
            row["mito_frac_max"] = round(float(df["mito_frac"].max()), 4)
        if "n_umi" in df:
            row["umi_median"] = float(df["n_umi"].median())
    syn = os.path.join(d, "syncytium_pairs.tsv")
    if os.path.isfile(syn):
        sdf = pd.read_csv(syn, sep="\t")
        pcol = [c for c in sdf.columns if "prob" in c]
        if pcol:
            row["syncytium_pairs_p>=0.5"] = int((sdf[pcol[0]] >= 0.5).sum())
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="批量结果根目录 (每芯片一个子目录)")
    ap.add_argument("--out", default="qc_summary.tsv")
    a = ap.parse_args()
    rows = []
    for name in sorted(os.listdir(a.root)):
        d = os.path.join(a.root, name)
        if os.path.isdir(d) and (os.path.isfile(os.path.join(d, "qc_report.json"))
                                 or os.path.isfile(os.path.join(d, "cell_qc.csv"))):
            rows.append(chip_row(d))
    if not rows:
        sys.exit(f"未在 {a.root} 下找到任何芯片结果 (qc_report.json/cell_qc.csv)")
    df = pd.DataFrame(rows)
    df.to_csv(a.out, sep="\t", index=False)
    print(df.to_string(index=False))
    print(f"\n-> {a.out} ({len(df)} 张芯片)")


if __name__ == "__main__":
    main()
