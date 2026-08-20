"""模块一: ssDNA 细胞核识别 —— 参数模型、训练与评估。

参数模型 NucleiParams:
  blur_sigma   高斯去噪 σ
  thr_factor   Otsu 阈值倍率 (>1 更严格, 减少假核; <1 更敏感)
  min_distance 相邻核最小间距(像素), 即 peak_local_max 的 min_distance
  min_size     最小核面积(像素)
  dense_cov    前景覆盖率超过该值判定为致密组织
  dense_mode   auto / peaks / dist (致密区种子策略)

训练目标(可靠细胞为 StereoCell 预分割+扩散后的细胞范围, 覆盖大于核, 可含多核):
  recall    = 可靠细胞足迹内含 ≥1 个核质心的比例 (越高越好)
  oversplit = 可靠细胞足迹内平均核数 (理想 1~2; 过多说明过分割)
  score     = recall - w_over · max(0, mean_n - 2) - w_sparse · max(0, 1 - mean_n) · (1-recall)
训练数据可含多个数据集(manifest), 对网格参数在各数据集上分别评分取平均。
"""
import json
from dataclasses import dataclass, asdict

import numpy as np
import tifffile

from . import seeds, io as scio


@dataclass
class NucleiParams:
    blur_sigma: float = 1.0
    thr_factor: float = 1.0
    min_distance: int = 6
    min_size: int = 20
    dense_cov: float = 0.35
    dense_mode: str = "auto"     # auto | peaks | dist
    tile: int = 4096
    max_cov: float = 0.35        # 块内前景覆盖率护栏 (防空块噪声过割)
    max_size: int = 1500         # 核面积上限 (聚集体剔除)
    min_circ: float = 0.5        # 圆度下限 (允许少量多核聚集)

    def save(self, path):
        with open(path, "w") as f:
            json.dump(asdict(self), f, indent=2)

    @staticmethod
    def load(path):
        with open(path) as f:
            return NucleiParams(**json.load(f))


def apply(img, p: NucleiParams, backend="skimage", device="cpu",
          roi_mask=None, excl_mask=None, reliable_xy=None, log=print):
    """按参数分割细胞核, 返回 int32 label mask。"""
    return seeds.segment_nuclei(
        img, backend=backend, tile=p.tile,
        nuc_radius=p.min_distance, min_size=p.min_size,
        device=device, thr_factor=p.thr_factor,
        dense_cov=p.dense_cov, dense_mode=p.dense_mode,
        roi_mask=roi_mask, excl_mask=excl_mask,
        max_cov=p.max_cov, max_size=p.max_size, min_circ=p.min_circ,
        reliable_xy=reliable_xy, log=log)


def reliable_geometry(expr, reliable_ids):
    """可靠细胞几何: 每个可靠细胞的分子云质心与覆盖半径 (p95 分子距离)。
    分子位于离散捕获点上, 不做像素栅格化, 直接在连续坐标系判定。
    返回 rid(int 数组), mu((n,2) xy), R(float 数组)。
    """
    lab = expr["label"]
    rids, mus, Rs = [], [], []
    for r in sorted(reliable_ids):
        m = lab == r
        if m.sum() < 5:
            continue
        xs = expr["x"][m].astype(float); ys = expr["y"][m].astype(float)
        mx, my = xs.mean(), ys.mean()
        d = np.sqrt((xs - mx) ** 2 + (ys - my) ** 2)
        rids.append(int(r)); mus.append((mx, my))
        Rs.append(float(max(np.percentile(d, 95), 10.0)))
    return (np.array(rids, np.int64), np.array(mus, float).reshape(-1, 2),
            np.array(Rs, float))


def score_mask(nuclei_mask, rel_geo):
    """对核分割结果评分。rel_geo = reliable_geometry 返回的三元组。
    核质心距可靠细胞质心 < 其覆盖半径 → 记为该细胞的核。
    返回 dict(recall, mean_n_nuclei, oversplit, n_reliable, n_nuclei)。"""
    rid, mu, R = rel_geo
    ids, areas, cents = seeds.nuclei_stats(nuclei_mask)
    if len(ids) == 0 or len(rid) == 0:
        return dict(recall=0.0, mean_n_nuclei=np.inf, oversplit=np.inf,
                    n_reliable=len(rid), n_nuclei=len(ids))
    d = np.sqrt(((cents[:, None, :] - mu[None, :, :]) ** 2).sum(-1))  # (n_nuclei, n_rel)
    within = d < R[None, :]
    dd = np.where(within, d, np.inf)
    nearest = dd.argmin(1)
    hit_rel = np.where(np.isfinite(dd[np.arange(len(ids)), nearest]), nearest, -1)
    n_per_cell = np.zeros(len(rid))
    for h in hit_rel:
        if h >= 0:
            n_per_cell[h] += 1
    recall = float((n_per_cell >= 1).mean())
    return dict(recall=recall, mean_n_nuclei=float(n_per_cell.mean()),
                oversplit=float(np.maximum(n_per_cell - 2, 0).mean()),
                n_reliable=len(rid), n_nuclei=len(ids),
                frac_nuclei_in_reliable=float((hit_rel >= 0).mean()))


def _objective(sc, w_over=1.0):
    if not np.isfinite(sc["mean_n_nuclei"]):
        return -1e9
    return sc["recall"] - w_over * sc["oversplit"]


def train(datasets, grid=None, w_over=1.0, backend="skimage", device="cpu", log=print):
    """网格搜索训练。datasets: [(ssdna_path, matrix_path, reliable_path), ...]
    返回 (best_params, records[list of dict])。
    """
    if grid is None:
        grid = [NucleiParams(thr_factor=tf, min_distance=md, min_size=ms)
                for tf in (0.7, 0.85, 1.0, 1.2)
                for md in (4, 6, 8, 10)
                for ms in (10, 20, 40)]
    # 预载数据
    loaded = []
    for ssdna, matrix, reliable_p in datasets:
        img = tifffile.imread(ssdna)
        expr = scio.load_expression(matrix)
        rel = scio.reliable_label_ids(scio.load_reliable(reliable_p))
        geo = reliable_geometry(expr, rel)
        loaded.append((ssdna, img, geo))
        log(f"  数据集 {ssdna}: 图像 {img.shape}, 可靠细胞 {len(geo[0])}")
    best, best_obj, records = None, -1e18, []
    for p in grid:
        objs, recalls, overs = [], [], []
        for ssdna, img, geo in loaded:
            mask = apply(img, p, backend=backend, device=device)
            sc = score_mask(mask, geo)
            objs.append(_objective(sc, w_over))
            recalls.append(sc["recall"]); overs.append(sc["oversplit"])
        obj = float(np.mean(objs))
        records.append(dict(params=asdict(p), objective=obj,
                            recall=float(np.mean(recalls)),
                            oversplit=float(np.mean(overs))))
        log(f"  thr={p.thr_factor} md={p.min_distance} ms={p.min_size}: "
            f"obj={obj:.4f} recall={np.mean(recalls):.3f} oversplit={np.mean(overs):.3f}")
        if obj > best_obj:
            best, best_obj = p, obj
    return best, records
