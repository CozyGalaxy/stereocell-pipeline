"""合成 StereoCell 数据生成器 (冒烟测试): 输出三件套输入文件。"""
import numpy as np
import tifffile


def make_synth(out_prefix, H=1500, W=1500, n_iso=200, n_pair=40, nuc_r=5,
               sigma_d=14.0, n_genes=60, mol_per_cell=300, ambient_frac=0.10, seed=7):
    rng = np.random.default_rng(seed)
    centers, types = [], []
    while len(centers) < n_iso:
        p = rng.uniform(60, H - 60, 2)
        if not centers or np.min(np.linalg.norm(np.array(centers) - p, axis=1)) > 55:
            centers.append(p); types.append(len(centers) % 3)
    np_ = 0
    while np_ < n_pair:
        p = rng.uniform(60, H - 60, 2)
        if np.min(np.linalg.norm(np.array(centers) - p, axis=1)) < 60:
            continue
        a = rng.uniform(0, 2 * np.pi)
        q = p + 2.2 * 2 * nuc_r * np.array([np.cos(a), np.sin(a)])
        if (q < 60).any() or (q > H - 60).any():
            continue
        centers += [p, q]; types += [np_ % 3, (np_ + 1) % 3]; np_ += 1
    centers = np.array(centers); types = np.array(types)
    NC = len(centers)

    prof = np.full((3, n_genes), 0.01)
    for t in range(3):
        prof[t, t * 10:(t + 1) * 10] = 0.06
    prof /= prof.sum(1, keepdims=True)
    amb = prof.mean(0); amb /= amb.sum()

    rows = []
    for i in range(NC):
        n = rng.poisson(mol_per_cell)
        xy = centers[i] + rng.normal(0, sigma_d, (n, 2))
        g = rng.choice(n_genes, n, p=prof[types[i]])
        for (x, y), gg in zip(xy, g):
            rows.append((gg, x, y, 1, 1, float(i + 1)))
    n_amb = int(len(rows) * ambient_frac)
    for x, y in rng.uniform(0, H, (n_amb, 2)):
        rows.append((int(rng.choice(n_genes, p=amb)), x, y, 1, 1, 0.0))

    yy, xx = np.mgrid[0:H, 0:W]
    img = rng.gamma(1.2, 8, (H, W)).astype(np.float32)
    for c in centers:
        img += 220 * np.exp(-((xx - c[0]) ** 2 + (yy - c[1]) ** 2) / (2 * (nuc_r / 1.5) ** 2))
    img = np.clip(img, 0, 255).astype(np.uint8)
    tifffile.imwrite(out_prefix + "_ssDNA.tif", img)

    # 预分配 label 注入误差: 5% 细胞分子错标为邻细胞, 8% 细胞整体丢失(标 0)
    genes = [f"Gene{k:03d}" for k in range(n_genes)]
    lost = rng.random(NC) < 0.08
    with open(out_prefix + "_matrix.csv", "w") as f:
        f.write("geneID,x,y,MIDCount,ExonCount,label\n")
        for gg, x, y, mid, ex, lab in rows:
            if lab > 0:
                ci = int(lab) - 1
                if lost[ci]:
                    lab = 0.0
                elif rng.random() < 0.05:
                    lab = float(rng.integers(1, NC + 1))   # 错标
            f.write(f"{genes[gg]},{x:.1f},{y:.1f},{mid},{ex},{lab}\n")
    reliable = [str(i + 1) for i in range(NC) if rng.random() < 0.15 and not lost[i]]
    with open(out_prefix + "_reliable.txt", "w") as f:
        f.write("\n".join(reliable) + "\n")
    truth = {"centers": centers, "types": types}
    np.save(out_prefix + "_truth.npy", centers)
    print(f"synth: {NC} cells, {len(rows)} molecules -> {out_prefix}_ssDNA.tif / _matrix.csv / _reliable.txt")
    return truth
