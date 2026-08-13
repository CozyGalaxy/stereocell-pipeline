"""表达矩阵与辅助文件的读写。

矩阵格式: 首行为 title, 列 = geneID, x, y, MIDCount, ExonCount, label
支持 .csv/.tsv/.txt(.gz) 与 .parquet。label 中 0.0 表示无细胞。
"""
import os
import gzip
import numpy as np
import pandas as pd

COLS = ["geneID", "x", "y", "MIDCount", "ExonCount", "label"]


def _open(path):
    return gzip.open(path, "rt") if str(path).endswith(".gz") else open(path)


def load_expression(path):
    """读取表达矩阵, 返回 dict:
    gene_codes(int32, 基因→词汇索引), genes(索引→geneID 数组),
    x, y(int32), mid, exon(int32), label(int64, 预分配细胞ID, 0=无细胞),
    label_str(原始 label 字符串数组)。
    label 兼容纯数字与带前缀字符串(如 'Sample.chip.2803', 取末段数字)。
    """
    p = str(path)
    if p.endswith(".parquet"):
        df = pd.read_parquet(p)
    else:
        # 从首行判定分隔符, 用 C 引擎 (python 引擎在亿级行上慢 10 倍以上)
        # keep_default_na=False: "nan"/"null"/"NA" 是合法基因名 (如果蝇 nan=nanos), 不得转为 NaN
        with _open(p) as fh:
            header = fh.readline()
        sep = "\t" if "\t" in header else ","
        df = pd.read_csv(p, sep=sep, engine="c", compression="infer",
                         dtype={"geneID": str}, keep_default_na=False)
    # 容忍列顺序/多余空白: 按名称取列
    df.columns = [c.strip() for c in df.columns]
    genes, gene_codes = np.unique(df["geneID"].fillna("NA").map(str).values,
                                  return_inverse=True)
    lab_raw = df["label"].astype(str).str.strip()
    lab_str = lab_raw.values
    tail = np.char.array(lab_str)
    # 取最后一段(以 . 分隔)并转数字; 失败 → 0
    tail = np.array([s.rsplit(".", 1)[-1] for s in lab_str])
    with np.errstate(invalid="ignore"):
        lab_num = pd.to_numeric(pd.Series(tail), errors="coerce").fillna(0).values
    return {
        "genes": genes,
        "gene_codes": gene_codes.astype(np.int32),
        "x": df["x"].values.astype(np.int32),
        "y": df["y"].values.astype(np.int32),
        "mid": df["MIDCount"].values.astype(np.int32),
        "exon": df["ExonCount"].values.astype(np.int32),
        "label": lab_num.astype(np.int64),
        "label_str": lab_str,
        "n_rows": len(df),
    }


def load_reliable(path):
    """可靠细胞列表: 每行一个细胞 ID。缺失时返回空集。
    支持两种命名: 纯数字(与矩阵 label 一致) 或 带前缀如 'Sample.chip.2803'
    (取最后一段数字映射到 label 命名空间)。返回字符串集合(原样)。"""
    if path is None or not os.path.exists(path):
        return set()
    with _open(path) as f:
        return {ln.strip() for ln in f if ln.strip() and not ln.startswith("#")}


def reliable_label_ids(reliable):
    """从可靠细胞 ID 集合提取数字 label ID 集合(int)。"""
    ids = set()
    for s in reliable:
        tail = str(s).rsplit(".", 1)[-1]
        try:
            ids.add(int(float(tail)))
        except ValueError:
            continue
    return ids


def write_updated_matrix(src_path, out_path, new_cell_id, compression="gzip",
                         chunk=5_000_000):
    """按原始行序把 label 列替换为 new_cell_id, 分块写出(内存安全)。
    new_cell_id: int32 数组, 长度 = 矩阵行数, 0 = 背景/拒收。
    """
    p = str(src_path)
    if p.endswith(".parquet"):
        df = pd.read_parquet(p)
        df["label"] = new_cell_id
        df.rename(columns={"label": "cell_id"}).to_parquet(out_path, index=False)
        return
    with _open(p) as fh:
        header = fh.readline()
    sep = "\t" if "\t" in header else ","
    cols = [c.strip() for c in header.strip().split(sep)]
    out_cols = cols[:-1] + ["cell_id"]
    # newline="": csv 写出器自带 \r\n, 文本模式默认换行转换会在 Windows 写成 \r\r\n
    if str(out_path).endswith(".gz"):
        fout = gzip.open(out_path, "wt", newline="")
    else:
        fout = open(out_path, "w", newline="")
    i, first, total = 0, True, 0
    # keep_default_na=False: 同上, "nan"/"null" 等是合法基因名, 必须原样保留
    for df in pd.read_csv(p, sep=sep, engine="c", compression="infer",
                          chunksize=chunk, dtype={"geneID": str},
                          keep_default_na=False):
        n = len(df)
        df = df.iloc[:, :-1].copy()
        df["cell_id"] = new_cell_id[i:i + n]
        df.to_csv(fout, sep=sep, index=False, header=first, columns=out_cols)
        first = False
        i += n
        total += n
    fout.close()
    assert total == len(new_cell_id), f"行数不匹配: 写入了{total}, 赋值数组{len(new_cell_id)}"


def save_cell_by_gene(out_prefix, cell_ids, genes, gene_codes, mid, assign, conf=None):
    """聚合 cell×gene 稀疏矩阵。优先 .h5ad, 降级 .mtx。
    cell_ids: 输出细胞 ID 序列(int, 从1开始按 assign 值); assign: 每分子细胞索引(0=背景)。
    """
    from scipy import sparse
    keep = assign > 0
    rows = assign[keep] - 1
    mat = sparse.csr_matrix(
        (mid[keep].astype(np.float32), (rows, gene_codes[keep])),
        shape=(int(assign.max()), len(genes)),
    )
    try:
        import anndata as ad
        adata = ad.AnnData(X=mat)
        adata.obs_names = [str(c) for c in cell_ids]
        adata.var_names = list(genes)
        if conf is not None:
            adata.obs["mean_conf"] = np.bincount(rows, weights=conf[keep],
                                                 minlength=len(cell_ids)) / np.maximum(
                np.bincount(rows, minlength=len(cell_ids)), 1)
        adata.write_h5ad(out_prefix + ".h5ad")
        return out_prefix + ".h5ad"
    except ImportError:
        from scipy import io as sio
        sio.mmwrite(out_prefix + ".mtx", mat)
        with open(out_prefix + "_barcodes.tsv", "w") as f:
            f.write("\n".join(str(c) for c in cell_ids) + "\n")
        with open(out_prefix + "_features.tsv", "w") as f:
            f.write("\n".join(genes) + "\n")
        return out_prefix + ".mtx"
