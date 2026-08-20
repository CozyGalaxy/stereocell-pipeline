"""芯片区域识别 (ROI) 与伪影排除。

StereoCell 图像结构 (从外到内):
  1. 最外缘黑色背景 (不一定每个方向都有; Y40364FA 为白色照片区)
  2. 较亮的照片/芯片边框区 (ssDNA 高亮聚集, 会污染阈值估计)
  3. 主体芯片区域 (唯一合法分析区, 表达密度高出边缘数十倍)

策略:
  - ROI 以表达密度为准 (捕获探针只存在于芯片上), 对图像亮度不敏感,
    天然兼容黑/白两种片外背景与轻微偏转;
  - 伪影排除在 ssDNA 上做: 局部背景扣除后检测大面积/高偏心率的高亮结构
    (聚集体、刮擦亮线) 与前景覆盖率异常的严重聚集区。
"""
import numpy as np
import pandas as pd
from scipy import ndimage as ndi
from skimage import filters, measure, morphology

from .io import _open


# ---------------------------------------------------------------- ROI ----
def density_histogram(matrix_path, shape, bin_size=48, chunk=5_000_000):
    """流式扫描表达矩阵, 返回 MIDCount 加权的粗粒度密度直方图 (bin_size 像素/格)。"""
    H, W = shape
    nb_y, nb_x = (H + bin_size - 1) // bin_size, (W + bin_size - 1) // bin_size
    hist = np.zeros((nb_y, nb_x), np.float64)
    with _open(str(matrix_path)) as fh:
        header = fh.readline()
    sep = "\t" if "\t" in header else ","
    for df in pd.read_csv(matrix_path, sep=sep, engine="c", compression="infer",
                          chunksize=chunk, usecols=["x", "y", "MIDCount"],
                          keep_default_na=False):
        # 坐标兼容浮点写法 (如 2803.0), 统一取整
        bx = np.clip(df["x"].values.astype(np.float64) // bin_size, 0, nb_x - 1).astype(np.int64)
        by = np.clip(df["y"].values.astype(np.float64) // bin_size, 0, nb_y - 1).astype(np.int64)
        np.add.at(hist, (by, bx), df["MIDCount"].values)
    return hist


def chip_roi(hist, close_radius=3):
    """表达密度直方图 -> 芯片区域 bool 掩膜 (直方图分辨率)。

    非芯片区密度接近 0, 芯片内高出数十倍; 对非零 bin 的 log 密度取 Otsu,
    闭运算 + 填洞 + 最大连通域。
    """
    if not (hist > 0).any():
        return np.zeros(hist.shape, bool)
    logd = np.log1p(hist)
    vals = logd[hist > 0]
    thr = filters.threshold_otsu(vals)
    mask = logd >= thr
    mask = morphology.closing(mask, morphology.disk(close_radius))
    mask = ndi.binary_fill_holes(mask)
    lab = measure.label(mask)
    if lab.max() > 1:
        sizes = np.bincount(lab.ravel())
        sizes[0] = 0
        mask = lab == sizes.argmax()
    return mask


def roi_erosion_px(bin_size):
    """ROI 默认内缩像素数: 去掉芯片边框亮带与边界扩散晕。"""
    return 2 * bin_size


def erode_roi_small(roi_small, erode_px, bin_size):
    """在直方图分辨率上做 ROI 内缩 (EDT 距离 > erode_px)。

    全分辨率 EDT 在 23520² 上要数分钟; 掩膜本身就是 bin_size 量化的,
    在低分辨率做不损失精度, 快数千倍。
    """
    r_bins = erode_px / bin_size
    return ndi.distance_transform_edt(roi_small) > r_bins


def upsample_mask(mask_small, shape, bin_size):
    """直方图分辨率掩膜 -> 全分辨率 (最近邻)。"""
    H, W = shape
    full = np.kron(mask_small, np.ones((bin_size, bin_size), bool))
    return full[:H, :W]


# ------------------------------------------------------- 伪影排除 ----
def artifact_mask(img, roi_full, nuc_radius=5, z_thr=8.0, max_nuc_area=400,
                  area_factor=8, ecc_thr=0.97, cov_win=201, cov_thr=0.55,
                  dilate_px=None, downsample=4):
    """检测需排除的伪影区域, 返回 bool 掩膜 (True=排除)。

    两类伪影:
      (1) 大面积/狭长的高亮结构: 细胞聚集体、刮擦亮线、边缘沉积。
          在局部背景扣除后的图像上做稳健阈值 (median + z_thr*MAD),
          连通域面积 > area_factor*max_nuc_area 或偏心率 > ecc_thr 即判伪影;
      (2) 严重聚集区: 窗口内前景覆盖率 > cov_thr (正常组织远低于此),
          该区域 watershed 不可靠, 整窗排除。
    roi_full: 全分辨率 ROI 掩膜 (片外不检测)。
    downsample: 内部降采样因子 (伪影均为大结构, 无需全分辨率, 提速 ~16 倍)。
    """
    H, W = img.shape
    ds = max(1, int(downsample))
    small = img[::ds, ::ds].astype(np.float32)
    roi_s = roi_full[::ds, ::ds]
    if roi_s.sum() == 0:
        return np.zeros((H, W), bool)
    sigma_bg = max(20.0, 6 * nuc_radius) / ds
    bg = ndi.gaussian_filter(small, sigma_bg)
    norm = small - bg
    roi_vals = norm[roi_s]
    med = np.median(roi_vals)
    mad = np.median(np.abs(roi_vals - med)) * 1.4826
    hi = med + z_thr * max(mad, 1e-3)

    # (1) 高亮大结构 (偏心率判据需配面积下限: 1-2 像素小点天然狭长, 不能误判)
    bright = (norm > hi) & roi_s
    lab = measure.label(bright)
    excl = np.zeros(small.shape, bool)
    max_area = area_factor * max_nuc_area / (ds * ds)
    min_ecc_area = max_nuc_area / (ds * ds)  # 狭长判据只用于至少一个核大小的对象
    for r in measure.regionprops(lab):
        if r.area >= max_area or (r.eccentricity >= ecc_thr and r.area >= min_ecc_area):
            excl[lab == r.label] = True

    # (2) 严重聚集: 前景覆盖率异常
    fg = norm > med + 2 * max(mad, 1e-3)
    fg &= roi_s
    cov = ndi.uniform_filter(fg.astype(np.float32), size=max(3, cov_win // ds))
    crowded = (cov > cov_thr) & roi_s

    excl |= crowded
    if excl.any():
        d = (dilate_px if dilate_px is not None else 2 * nuc_radius) // ds
        excl = morphology.dilation(excl, morphology.disk(max(1, d)))
    excl &= roi_s
    # 升回全分辨率
    full = np.kron(excl, np.ones((ds, ds), bool))[:H, :W]
    if full.shape != (H, W):  # 边角不足时补齐
        out = np.zeros((H, W), bool)
        out[:full.shape[0], :full.shape[1]] = full
        full = out
    return full & roi_full


# ------------------------------------------------------- 每芯片参数 ----
def estimate_nuc_radius(img, roi_full, reliable_xy=None, default=5.0,
                        clip=(3.0, 15.0)):
    """估计本芯片典型细胞核半径 (像素)。

    优先用可靠细胞位置: 以每个可靠细胞质心为圆心测强度衰减半径;
    无可靠细胞时, 用局部背景扣除后高亮连通域的中位等效半径。
    结果 clip 到合理范围。
    """
    img = img.astype(np.float32)
    sigma_bg = max(20.0, 6 * default)
    bg = ndi.gaussian_filter(img, sigma_bg)
    norm = img - bg
    roi_vals = norm[roi_full]
    if len(roi_vals) == 0:
        return default
    med = np.median(roi_vals)
    mad = np.median(np.abs(roi_vals - med)) * 1.4826
    fg = norm > med + 3 * max(mad, 1e-3)
    fg &= roi_full
    lab = measure.label(fg)
    sizes = np.bincount(lab.ravel())
    sizes = sizes[(sizes >= 20) & (sizes <= 2000)]
    if len(sizes) < 10:
        return default
    r = float(np.median(np.sqrt(sizes / np.pi)))
    return float(np.clip(r, clip[0], clip[1]))
