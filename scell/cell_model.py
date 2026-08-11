"""模块二: 细胞分割 —— 参数模型、核心分割流程与训练。

CellParams:
  nuclei       嵌套 NucleiParams (核识别参数)
  kappa        领地半径上限系数: R_i = min(r95, kappa·d_i/2); kappa=1.0 即严格中线划分
  conf_thr     EM 置信阈值
  margin       常规区边际拒绝阈值
  dense_margin 聚集区边际拒绝阈值 (拥挤时增大 → 等效缩小 RNA 扩散因子)
  w_b          背景类先验权重
  mito_max     线粒体比例过滤阈值 (默认 0.10)
  em_iter      EM 迭代数

训练(多数据集): 以可靠细胞为锚点, 网格搜索 (conf_thr, margin, dense_margin),
目标 = 可靠分子归属正确率 - 0.5·污染率, 平手时取归属率高者。
"""
import json
from dataclasses import dataclass, field, asdict

import numpy as np

from . import io as scio, seeds, diffusion, assign as scassign, export, nuclei_model as nm


@dataclass
class CellParams:
    nuclei: dict = field(default_factory=lambda: asdict(nm.NucleiParams()))
    kappa: float = 0.9
    conf_thr: float = 0.55
    margin: float = 0.20
    dense_margin: float = 0.35
    w_b: float = 0.15
    mito_max: float = 0.10
    em_iter: int = 12

    def nuclei_params(self):
        return nm.NucleiParams(**self.nuclei)

    def save(self, path):
        with open(path, "w") as f:
            json.dump(asdict(self), f, indent=2)

    @staticmethod
    def load(path):
        with open(path) as f:
            return CellParams(**json.load(f))


def segment_cells(img, expr, p: CellParams, device="cpu", seed_backend="skimage",
                  nuclei=None, log=print, fit_png=None):
    """核心分割: 核种子 → 扩散估计 → 边际拒绝 EM → 领地栅格化。
    返回 dict(所有中间结果)。
    """
    if nuclei is None:
        nuclei = nm.apply(img, p.nuclei_params(), backend=seed_backend, device=device)
    nuclei = nuclei.astype(np.int32)
    ids, areas, cents = seeds.nuclei_stats(nuclei)
    r_nuc_med = float(np.median(np.sqrt(areas / np.pi))) if len(areas) else 5.0
    log(f"核种子 {len(ids)} 个, 核半径中位数 {r_nuc_med:.1f}px")

    ix = np.clip(expr["x"], 0, nuclei.shape[1] - 1)
    iy = np.clip(expr["y"], 0, nuclei.shape[0] - 1)
    mol_nuc = nuclei[iy, ix]
    umi = np.zeros(nuclei.shape, np.float32)
    np.add.at(umi, (iy, ix), expr["mid"])

    lam, r95, D_ISO, isolated, _, _ = diffusion.estimate_diffusion(
        umi, cents, r_nuc_med, out_png=fit_png)
    tree, d_nn = diffusion.local_density(cents)
    R_i, sigma_arr, margin_i, crowded = diffusion.cell_params(
        d_nn, r95, r_nuc_med, D_ISO, kappa=p.kappa,
        margin0=p.margin, dense_margin=p.dense_margin)
    log(f"λ={lam:.1f}px r95={r95:.1f}px D_ISO={D_ISO:.1f}px "
        f"孤立 {isolated.sum()}/{len(ids)} 拥挤 {crowded.sum()}")

    # 候选半径密度护栏: 致密区防止候选爆炸 (候选截断仅作效率护栏,
    # 分割决策仍由 margin_i 控制, 不做硬性半径截断)
    d_med = float(np.median(d_nn)) if len(d_nn) else r95
    r_cand = float(min(r95, max(4 * r_nuc_med, 1.5 * d_med)))

    ambient = scassign.estimate_ambient(expr["gene_codes"], expr["mid"],
                                        mol_nuc > 0)
    pi = scassign.init_pi(nuclei, ix, iy, expr["gene_codes"],
                          len(expr["genes"]), len(ids), ambient)
    assign, conf = scassign.em_assign(
        np.stack([expr["x"], expr["y"]], 1).astype(np.float32),
        expr["gene_codes"], expr["mid"], cents, pi, ambient,
        sigma=float(sigma_arr[0]) if len(sigma_arr) else r95 / 2.45,
        r_cand=r_cand, margin_i=margin_i, conf_thr=p.conf_thr, w_b=p.w_b,
        n_iter=p.em_iter, device=device, log=log)

    cell_mask = export.rasterize_cells(umi, nuclei, cents, R_i)
    return dict(nuclei=nuclei, ids=ids, areas=areas, cents=cents,
                r_nuc_med=r_nuc_med, lam=lam, r95=r95, D_ISO=D_ISO,
                isolated=isolated, d_nn=d_nn, R_i=R_i, margin_i=margin_i,
                crowded=crowded, r_cand=r_cand, assign=assign, conf=conf,
                cell_mask=cell_mask, umi=umi)


def _seed_reliable_map(cents, rel_geo):
    """种子 → 可靠细胞映射: 每个种子质心距哪个可靠细胞质心最近且在其覆盖半径内。
    返回 seed_to_rel (len=n_seeds, -1=无) 与 rel_to_seeds (dict rel_idx -> [seed_idx])。
    """
    rid, mu, R = rel_geo
    if len(rid) == 0 or len(cents) == 0:
        return np.full(len(cents), -1), {}
    d = np.sqrt(((cents[:, None, :] - mu[None, :, :]) ** 2).sum(-1))
    within = np.where(d < R[None, :], d, np.inf)
    nearest = within.argmin(1)
    ok = np.isfinite(within[np.arange(len(cents)), nearest])
    seed_to_rel = np.where(ok, nearest, -1)
    rel_to_seeds = {}
    for s, r in enumerate(seed_to_rel):
        if r >= 0:
            rel_to_seeds.setdefault(int(r), []).append(s)
    return seed_to_rel, rel_to_seeds


def score_segmentation(res, expr, rel_geo):
    """以可靠细胞为锚评估分割: 可靠分子的归属正确率/污染率/未归属率。"""
    rid, mu, R = rel_geo
    if len(rid) == 0:
        return dict(agreement=np.nan, contam=np.nan, unassigned=np.nan)
    seed_to_rel, rel_to_seeds = _seed_reliable_map(res["cents"], rel_geo)
    lab = expr["label"]
    rel_index = {int(r): k for k, r in enumerate(rid)}
    m = np.isin(lab, rid)
    if m.sum() == 0:
        return dict(agreement=np.nan, contam=np.nan, unassigned=np.nan)
    a = res["assign"][m]
    lrel = np.array([rel_index[int(v)] for v in lab[m]])
    seed_rel = np.full(len(res["cents"]), -1)
    for r, ss in rel_to_seeds.items():
        for s in ss:
            seed_rel[s] = r
    a_rel = np.where(a > 0, seed_rel[np.clip(a - 1, 0, len(seed_rel) - 1)], -1)
    n_correct = int(((a > 0) & (a_rel == lrel)).sum())
    n_contam = int(((a > 0) & (a_rel != lrel)).sum())
    n_un = int((a == 0).sum())
    n = len(a)
    return dict(agreement=n_correct / n, contam=n_contam / n,
                unassigned=n_un / n, n_reliable_mols=n)


def train(datasets, base: CellParams = None, backend="skimage", device="cpu",
          log=print, quick=False):
    """网格搜索 (conf_thr, margin, dense_margin)。datasets 同 nuclei_model.train。"""
    import tifffile
    base = base or CellParams()
    loaded = []
    for ssdna, matrix, reliable_p in datasets:
        img = tifffile.imread(ssdna)
        expr = scio.load_expression(matrix)
        rel = scio.reliable_label_ids(scio.load_reliable(reliable_p))
        geo = nm.reliable_geometry(expr, rel)
        # 核分割与扩散估计只做一次 (与被调参数无关)
        res0 = segment_cells(img, expr, base, device=device,
                             seed_backend=backend, log=log)
        loaded.append((ssdna, expr, geo, res0))
    if quick:
        grid = [(0.50, 0.15, 0.35), (0.50, 0.20, 0.35),
                (0.60, 0.15, 0.35), (0.60, 0.20, 0.35)]
    else:
        grid = [(ct, mg, dm) for ct in (0.45, 0.55, 0.65)
                for mg in (0.15, 0.20, 0.25) for dm in (0.30, 0.40)]
    best, best_obj, records = None, -1e18, []
    for ct, mg, dm in grid:
        p = CellParams(**{**asdict(base),
                          "conf_thr": ct, "margin": mg, "dense_margin": dm})
        objs, ags, cts_, uns = [], [], [], []
        for ssdna, expr, geo, res0 in loaded:
            # 仅重跑 EM (复用核/扩散结果)
            nuclei = res0["nuclei"]
            ix = np.clip(expr["x"], 0, nuclei.shape[1] - 1)
            iy = np.clip(expr["y"], 0, nuclei.shape[0] - 1)
            ambient = scassign.estimate_ambient(
                expr["gene_codes"], expr["mid"], nuclei[iy, ix] > 0)
            pi = scassign.init_pi(nuclei, ix, iy, expr["gene_codes"],
                                  len(expr["genes"]), len(res0["ids"]), ambient)
            margin_i = np.where(res0["crowded"], dm, mg)
            a, c = scassign.em_assign(
                np.stack([expr["x"], expr["y"]], 1).astype(np.float32),
                expr["gene_codes"], expr["mid"], res0["cents"], pi,
                ambient, sigma=res0["r95"] / 2.45, r_cand=res0["r_cand"],
                margin_i=margin_i, conf_thr=ct, w_b=p.w_b,
                n_iter=p.em_iter, device=device)
            res = dict(res0); res["assign"] = a
            sc = score_segmentation(res, expr, geo)
            if np.isnan(sc["agreement"]):
                continue
            obj = sc["agreement"] - 0.5 * sc["contam"]
            objs.append(obj); ags.append(sc["agreement"])
            cts_.append(sc["contam"]); uns.append(sc["unassigned"])
        if not objs:
            continue
        obj = float(np.mean(objs))
        records.append(dict(conf_thr=ct, margin=mg, dense_margin=dm,
                            objective=obj, agreement=float(np.mean(ags)),
                            contam=float(np.mean(cts_)),
                            unassigned=float(np.mean(uns))))
        log(f"  ct={ct} mg={mg} dm={dm}: obj={obj:.4f} "
            f"agree={np.mean(ags):.3f} contam={np.mean(cts_):.3f} un={np.mean(uns):.3f}")
        if obj > best_obj:
            best, best_obj = p, obj
    return best, records
