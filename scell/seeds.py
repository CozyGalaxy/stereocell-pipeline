"""ssDNA 核分割(种子)与红线叠加图。

后端:
- skimage (默认, CPU): 局部背景扣除 + 分块自适应阈值 + 距离变换 watershed + 形态 QC
- cellpose (可选, GPU): Cellpose-SAM, 分块推理 + 重叠区投票合并

鲁棒性设计 (针对跨文库差异):
- 局部背景扣除 (大 sigma 高斯): 消除曝光不均/辉光梯度, 暗区细胞可检出;
- 分块 Otsu + 全局稳健下限 (median+k*MAD): 暗块不漏检, 空块不过割;
- 前景覆盖率护栏: 块内前景占比超 max_cov 时自动提高阈值, 防噪声爆量分割;
- 形态 QC: 面积/圆度/峰值强度三维过滤, 聚集体与刮擦结构判为伪影;
- 峰值强度可用可靠细胞位置校准 (每芯片自适应)。
"""
import numpy as np
import tifffile
from scipy import ndimage as ndi
from skimage import filters, measure, segmentation, feature, morphology


def _remove_small(arr, min_size):
    """morphology.remove_small_objects 的替代实现。

    skimage 0.26 起 min_size 参数弃用 (FutureWarning), 且 label 图只有 1 个
    对象时会触发 "Only one label was provided" UserWarning。本实现:
    bool 图 → 删除面积 < min_size 的连通域; int label 图 → 小对象像素置 0,
    保留其余对象的原始编号。
    """
    if arr.dtype == bool:
        lab = measure.label(arr)
    else:
        lab = arr
    if lab.max() == 0:
        return arr
    sizes = np.bincount(lab.ravel())
    small = np.flatnonzero(sizes < min_size)
    small = small[small > 0]
    if len(small) == 0:
        return arr
    out = arr.copy()
    if arr.dtype == bool:
        out[np.isin(lab, small)] = False
    else:
        out[np.isin(lab, small)] = 0
    return out


def _tiles(shape, tile, overlap):
    H, W = shape
    for y0 in range(0, H, tile - overlap):
        for x0 in range(0, W, tile - overlap):
            y1, x1 = min(y0 + tile, H), min(x0 + tile, W)
            yield y0, y1, x0, x1


def calibrate_peak_thr(norm_small, reliable_xy, nuc_radius, step=1):
    """用可靠细胞位置校准核峰值强度阈值。

    norm_small: 背景扣除后的 (降采样) 图像; reliable_xy: (N,2) 全分辨率坐标;
    step: norm_small 相对全图的降采样步长。
    返回 p20 峰值强度; 样本不足返回 None (调用方回退到稳健统计下限)。
    """
    if reliable_xy is None or len(reliable_xy) < 30:
        return None
    H, W = norm_small.shape
    r = max(2, int(np.ceil(2 * nuc_radius / step)))
    peaks = []
    for x, y in reliable_xy:
        sx, sy = int(x / step), int(y / step)
        y0, y1 = max(0, sy - r), min(H, sy + r + 1)
        x0, x1 = max(0, sx - r), min(W, sx + r + 1)
        if y1 > y0 and x1 > x0:
            peaks.append(float(norm_small[y0:y1, x0:x1].max()))
    peaks = np.asarray(peaks)
    if len(peaks) < 30:
        return None
    return float(np.percentile(peaks, 20))


def _morpho_qc(m, norm, min_size, max_size, min_circ, peak_thr):
    """形态 QC: 面积 ∈ [min_size, max_size], 圆度 ≥ min_circ, 峰值 ≥ peak_thr。

    允许少量聚集: 圆度阈值放宽到 0.5 时, 2~3 个核融合的近圆对象仍保留;
    大而狭长/不规则的聚集体与刮擦被剔除。返回剔除后的 label 图 (原编号保留)。
    """
    if m.max() == 0:
        return m
    props = measure.regionprops(m, intensity_image=norm)
    bad = []
    for r in props:
        if r.area < min_size or r.area > max_size:
            bad.append(r.label)
            continue
        per = max(r.perimeter, 1e-6)
        circ = 4 * np.pi * r.area / per ** 2
        if circ < min_circ:
            bad.append(r.label)
            continue
        if peak_thr is not None and r.intensity_max < peak_thr:
            bad.append(r.label)
    if bad:
        m = m.copy()
        m[np.isin(m, np.asarray(bad))] = 0
    return m


def segment_nuclei(img, backend="skimage", tile=4096, overlap=128,
                   nuc_radius=5, min_size=30, device="cpu", cellpose_model=None,
                   thr_factor=1.0, dense_cov=0.35, dense_mode="auto",
                   roi_mask=None, excl_mask=None, bg_sigma=None,
                   max_cov=0.35, max_size=1500, min_circ=0.5,
                   reliable_xy=None, log=print):
    """返回 int32 label mask (0=背景)。

    thr_factor: Otsu 阈值倍率 (>1 更严格); dense_cov: 前景占比超过该值视为致密组织;
    dense_mode: auto(按覆盖率选择) / peaks(强制强度峰) / dist(强制距离变换);
    roi_mask: 芯片区域 (None=全图); excl_mask: 伪影排除区 (True=排除);
    bg_sigma: 局部背景高斯 sigma (None=自适应 6*nuc_radius, 下限20);
    max_cov: 块内前景覆盖率护栏; max_size/min_circ: 形态 QC 上限;
    reliable_xy: 可靠细胞质心 (N,2), 用于校准峰值强度阈值。
    """
    img = img.astype(np.float32)
    H, W = img.shape
    if roi_mask is None:
        roi_mask = np.ones((H, W), bool)
    if bg_sigma is None:
        bg_sigma = max(20.0, 6 * nuc_radius)

    # ---- 全分辨率预处理 (一次): 局部背景扣除 + 轻平滑 ----
    # bg 在 4 倍降采样上估计再双线性回升 (平滑场, 无损且快 16 倍);
    # 校准/阈值/分割共用同一 norm, 避免分块背景不一致与尺度失配
    ds_bg = 4
    img_s = img[::ds_bg, ::ds_bg]
    bg_s = ndi.gaussian_filter(img_s, bg_sigma / ds_bg)
    bg = np.kron(bg_s, np.ones((ds_bg, ds_bg), np.float32))[:H, :W]
    norm_full = ndi.gaussian_filter(img - bg, 1.0)
    del img_s, bg_s, bg

    # ---- 全局统计 (降采样视图, ROI 内): 半正态噪声 sigma + 峰值校准 ----
    step = max(1, H // 1024)
    norm_small = norm_full[::step, ::step]
    roi_small = roi_mask[::step, ::step]
    vals = norm_small[roi_small]
    neg = vals[vals < 0]
    if len(neg) > 100:
        # 半正态噪声估计: 负值镜像, 对 uint8 量化噪声比 MAD 稳健
        sig_g = float(np.sqrt((np.concatenate([neg, -neg]) ** 2).mean()))
    else:
        sig_g = float(np.std(vals)) if len(vals) else 1.0
    sig_g = max(sig_g, 1e-3)
    floor = 3 * sig_g           # 前景阈值下限
    peak_thr = calibrate_peak_thr(norm_full, reliable_xy, nuc_radius, 1)
    if peak_thr is None:
        peak_thr = 5 * sig_g    # 无可靠细胞时的回退峰值阈值
    log(f"  全局统计: bg_sigma={bg_sigma:.0f} sigma={sig_g:.2f} "
        f"floor={floor:.2f} peak_thr={peak_thr:.2f}")

    # 致密组织回退: 前景占比过高时距离变换 watershed 退化为少数巨块,
    # 改用强度局部极大值作为种子 (UMI-core 思路, 与 CellBin Stereo-cell 一致)
    cover = float((vals > 3 * sig_g).mean()) if len(vals) else 0.0
    if dense_mode == "peaks":
        dense = True
    elif dense_mode == "dist":
        dense = False
    else:
        dense = cover > dense_cov

    labels = np.zeros((H, W), np.int32)
    offset = 0
    use_cellpose = backend == "cellpose"
    if use_cellpose:
        from cellpose import models
        model = models.CellposeModel(gpu=(device == "cuda"),
                                     pretrained_model=cellpose_model or "cpsam")
    for y0, y1, x0, x1 in _tiles((H, W), tile, overlap):
        roi_t = roi_mask[y0:y1, x0:x1]
        if roi_t.mean() < 0.01:
            continue  # 块几乎全在片外
        sub = img[y0:y1, x0:x1]
        if use_cellpose:
            m, _, _ = model.eval(sub, diameter=2 * nuc_radius)
            m = m.astype(np.int32)
        else:
            # 切片共用全分辨率 norm; 块内半正态噪声 -> 3σ 前景阈值
            norm = norm_full[y0:y1, x0:x1]
            v = norm[roi_t]
            neg_t = v[v < 0]
            sig_t = float(np.sqrt((np.concatenate([neg_t, -neg_t]) ** 2).mean())) if len(neg_t) > 100 else sig_g
            thr = max(3 * sig_t * thr_factor, 0.5 * floor)
            thr_pk = max(4 * sig_t * thr_factor, peak_thr)
            fg = (norm > thr) & roi_t
            # 前景覆盖率护栏: 防空块/噪声块过割
            cov = fg.sum() / max(int(roi_t.sum()), 1)
            if cov > max_cov:
                thr = max(thr, float(np.quantile(v, 1 - max_cov * roi_t.sum() / len(v))))
                fg = (norm > thr) & roi_t
            if excl_mask is not None:
                fg &= ~excl_mask[y0:y1, x0:x1]
            fg = _remove_small(fg, 20)
            fg = ndi.binary_fill_holes(fg)
            # 极稀疏块 (cov<0.05): 强度峰已天然分离, 跳过 EDT 提速 ~70%
            if dense or cov < 0.05:
                pk = feature.peak_local_max(
                    norm, min_distance=max(1, int(nuc_radius)),
                    threshold_abs=thr_pk, labels=fg)
                seed = np.zeros_like(fg, bool)
                if len(pk):
                    seed[tuple(pk.T)] = True
                m = segmentation.watershed(-norm, markers=measure.label(seed), mask=fg)
            else:
                dist = ndi.distance_transform_edt(fg)
                pk = feature.peak_local_max(dist, min_distance=max(1, int(nuc_radius)), labels=fg)
                seed = np.zeros_like(fg, bool)
                if len(pk):
                    seed[tuple(pk.T)] = True
                m = segmentation.watershed(-dist, markers=measure.label(seed), mask=fg)
            m = _remove_small(m.astype(np.int32), min_size)
            # 形态 QC: 面积/圆度/峰值 (可靠细胞校准)
            m = _morpho_qc(m, norm, min_size, max_size, min_circ, peak_thr)
        if m.max() == 0:
            continue
        # 只保留块中心区域的结果, 消除分块边缘重复
        oy0 = 0 if y0 == 0 else overlap // 2
        ox0 = 0 if x0 == 0 else overlap // 2
        oy1 = m.shape[0] if y1 == H else m.shape[0] - overlap // 2
        ox1 = m.shape[1] if x1 == W else m.shape[1] - overlap // 2
        core = m[oy0:oy1, ox0:ox1]
        keep = core > 0
        core = core.copy()
        core[keep] += offset
        offset += int(m.max())
        labels[y0 + oy0:y0 + oy1, x0 + ox0:x0 + ox1][keep] = core[keep]
    # remove_small / 形态 QC 会在 label 上留下空洞编号, 重排为连续 1..N
    labels, _, _ = segmentation.relabel_sequential(labels)
    return labels.astype(np.int32)


def nuclei_stats(mask, min_radius_px=1.0):
    ids = np.unique(mask)[1:]
    areas = np.bincount(mask.ravel(), minlength=int(mask.max()) + 1)[ids]
    cents = ndi.center_of_mass(mask > 0, mask, ids)
    cents = np.array(cents)[:, ::-1] if len(ids) else np.zeros((0, 2))  # (x, y)
    return ids, areas, cents


def write_overlay(img, label_mask, out_path, max_png_px=16000, excl_mask=None):
    """ssDNA 灰度图 + 1 像素红色实例轮廓 (+ 伪影排除区蓝色轮廓)。大图自动写 TIFF。"""
    from skimage.segmentation import find_boundaries
    gray = img.astype(np.float32)
    gray = (gray - gray.min()) / max(float(np.ptp(gray)), 1e-6)
    rgb = np.stack([gray, gray, gray], -1)
    if excl_mask is not None:
        bd_x = find_boundaries(excl_mask.astype(np.uint8), mode="outer")
        rgb[bd_x] = [0.0, 0.4, 1.0]
    bd = find_boundaries(label_mask, mode="outer") & (label_mask >= 0)
    # 实例间边界也要画出: 对 label 图求 boundaries
    bd |= find_boundaries(label_mask, mode="thick") & (label_mask > 0)
    rgb[bd] = [1.0, 0.0, 0.0]
    rgb8 = (rgb * 255).astype(np.uint8)
    if max(img.shape) > max_png_px or not out_path.endswith(".png"):
        tifffile.imwrite(out_path if out_path.endswith((".tif", ".tiff")) else out_path + ".tif", rgb8)
    else:
        from PIL import Image
        Image.fromarray(rgb8).save(out_path)
    return out_path
