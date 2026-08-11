"""领地栅格化、最终 mask、更新矩阵与 QC 报告。"""
import json
import numpy as np
from scipy import ndimage as ndi
from skimage import segmentation


def rasterize_cells(umi, nuclei_mask, centers, R_i, sigma_blur=3.0):
    """逐细胞 R_i 约束的 watershed 生成最终细胞领地 mask。
    高程图 = -UMI密度(高斯平滑); 允许区域 = 距最近种子 < 该种子 R_i 的像素;
    markers = 核 label。保证每细胞领地包含其核且不越出自适应半径。
    """
    H, W = umi.shape
    yy, xx = np.mgrid[0:H, 0:W]
    # 逐像素: 最近种子索引与距离 (分块防内存爆炸)
    region = np.zeros((H, W), bool)
    elev = -ndi.gaussian_filter(umi, sigma_blur)
    chunk = 1024
    nearest_lab = np.zeros((H, W), np.int32)
    for y0 in range(0, H, chunk):
        for x0 in range(0, W, chunk):
            y1, x1 = min(y0 + chunk, H), min(x0 + chunk, W)
            ys, xs = np.mgrid[y0:y1, x0:x1]
            d2 = ((ys[..., None] - centers[None, None, :, 1]) ** 2 +
                  (xs[..., None] - centers[None, None, :, 0]) ** 2)
            j = d2.argmin(-1)
            d = np.sqrt(d2.min(-1))
            ok = d < R_i[j]
            region[y0:y1, x0:x1][ok] = True
            nearest_lab[y0:y1, x0:x1][ok] = j[ok] + 1
    mask = segmentation.watershed(elev, markers=nuclei_mask.astype(np.int32), mask=region)
    # watershed 可能把像素分给非最近种子, 用 nearest_lab 约束修正
    bad = (mask > 0) & (mask != nearest_lab) & (nearest_lab > 0)
    mask[bad] = nearest_lab[bad]
    return mask.astype(np.int32)


def qc_report(path, **kw):
    with open(path, "w") as f:
        json.dump(kw, f, ensure_ascii=False, indent=2, default=float)
