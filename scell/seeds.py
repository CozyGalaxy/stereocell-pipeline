"""ssDNA 核分割(种子)与红线叠加图。

后端:
- skimage (默认, CPU): 分块高斯去噪 + 全局 Otsu + 距离变换 watershed
- cellpose (可选, GPU): Cellpose-SAM, 分块推理 + 重叠区投票合并
"""
import numpy as np
import tifffile
from scipy import ndimage as ndi
from skimage import filters, measure, segmentation, feature, morphology


def _tiles(shape, tile, overlap):
    H, W = shape
    for y0 in range(0, H, tile - overlap):
        for x0 in range(0, W, tile - overlap):
            y1, x1 = min(y0 + tile, H), min(x0 + tile, W)
            yield y0, y1, x0, x1


def segment_nuclei(img, backend="skimage", tile=4096, overlap=128,
                   nuc_radius=5, min_size=30, device="cpu", cellpose_model=None,
                   thr_factor=1.0, dense_cov=0.35, dense_mode="auto"):
    """返回 int32 label mask (0=背景)。

    thr_factor: Otsu 阈值倍率 (>1 更严格); dense_cov: 前景占比超过该值视为致密组织;
    dense_mode: auto(按覆盖率选择) / peaks(强制强度峰) / dist(强制距离变换)。
    """
    img = img.astype(np.float32)
    H, W = img.shape
    # 全局阈值在下采样图上估计, 避免分块阈值不一致
    small = img[:: max(1, H // 1024), :: max(1, W // 1024)]
    blur_small = ndi.gaussian_filter(small, 1.0)
    thr = filters.threshold_otsu(blur_small) * thr_factor
    # 致密组织回退: 前景占比过高时距离变换 watershed 退化为少数巨块,
    # 改用强度局部极大值作为种子 (UMI-core 思路, 与 CellBin Stereo-cell 一致)
    cover = float((blur_small > thr).mean())
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
        sub = img[y0:y1, x0:x1]
        if use_cellpose:
            m, _, _ = model.eval(sub, diameter=2 * nuc_radius)
            m = m.astype(np.int32)
        else:
            blur = ndi.gaussian_filter(sub, 1.0)
            fg = blur > thr
            fg = morphology.remove_small_objects(fg, 20)
            fg = ndi.binary_fill_holes(fg)
            if dense:
                pk = feature.peak_local_max(
                    blur, min_distance=max(1, int(nuc_radius)),
                    threshold_abs=thr, labels=fg)
                seed = np.zeros_like(fg, bool)
                if len(pk):
                    seed[tuple(pk.T)] = True
                m = segmentation.watershed(-blur, markers=measure.label(seed), mask=fg)
            else:
                dist = ndi.distance_transform_edt(fg)
                pk = feature.peak_local_max(dist, min_distance=max(1, int(nuc_radius)), labels=fg)
                seed = np.zeros_like(fg, bool)
                if len(pk):
                    seed[tuple(pk.T)] = True
                m = segmentation.watershed(-dist, markers=measure.label(seed), mask=fg)
            m = morphology.remove_small_objects(m.astype(np.int32), min_size)
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
    # remove_small_objects 会在 label 上留下空洞编号, 重排为连续 1..N
    labels, _, _ = segmentation.relabel_sequential(labels)
    return labels.astype(np.int32)


def nuclei_stats(mask, min_radius_px=1.0):
    ids = np.unique(mask)[1:]
    areas = np.bincount(mask.ravel(), minlength=int(mask.max()) + 1)[ids]
    cents = ndi.center_of_mass(mask > 0, mask, ids)
    cents = np.array(cents)[:, ::-1] if len(ids) else np.zeros((0, 2))  # (x, y)
    return ids, areas, cents


def write_overlay(img, label_mask, out_path, max_png_px=16000):
    """ssDNA 灰度图 + 1 像素红色实例轮廓。大图自动写 TIFF, 小图写 PNG。"""
    from skimage.segmentation import find_boundaries
    gray = img.astype(np.float32)
    gray = (gray - gray.min()) / max(float(np.ptp(gray)), 1e-6)
    rgb = np.stack([gray, gray, gray], -1)
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
