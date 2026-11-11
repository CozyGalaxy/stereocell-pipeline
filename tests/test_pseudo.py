import unittest

import numpy as np

from scell.pseudo import build_pseudo_cells


class PseudoCellTest(unittest.TestCase):
    def test_adaptive_roi_exclusion_and_margin(self):
        mask = np.zeros((100, 100), np.int32)
        mask[40:50, 40:50] = 1
        roi = np.zeros_like(mask, bool)
        roi[10:90, 10:90] = True
        excl = np.zeros_like(mask, bool)
        excl[10:20, 10:20] = True
        x = np.array([5, 12, 45, 65, 75, 85])
        y = x.copy()
        mid = np.ones(len(x))
        genes = np.zeros(len(x), np.int32)
        mat, meta = build_pseudo_cells(
            mask, x, y, mid, genes, 2, size=10, margin=0,
            roi_mask=roi, excl_mask=excl, lam=20, r95=60,
            mode="adaptive", log=lambda _: None)
        self.assertEqual(meta["margin_px"], 5)
        self.assertEqual(mat.shape[1], 2)
        self.assertEqual(len(meta["bin_x"]), mat.shape[0])
        self.assertTrue(np.all(meta["bin_x"] >= 60))
        self.assertTrue(np.all(meta["distance_stratum"] == "near"))

        # legacy 不应用 ROI，保留旧版片外候选和固定 margin=5。
        _, old = build_pseudo_cells(
            mask, x, y, mid, genes, 2, size=10, margin=0,
            roi_mask=roi, excl_mask=excl, lam=20, r95=60,
            mode="legacy", log=lambda _: None)
        self.assertEqual(old["margin_px"], 5)
        self.assertTrue(np.any(old["bin_x"] < 10))
        self.assertTrue(np.all(old["distance_stratum"] == "legacy"))

    def test_stratified_sampling_reproducible(self):
        rng = np.random.default_rng(7)
        mask = np.zeros((400, 400), np.int32)
        mask[180:220, 180:220] = 1
        x = rng.integers(0, 400, 5000, dtype=np.int32)
        y = rng.integers(0, 400, 5000, dtype=np.int32)
        mid = np.ones(5000)
        genes = rng.integers(0, 5, 5000, dtype=np.int32)
        outputs = []
        for _ in range(2):
            mat, meta = build_pseudo_cells(
                mask, x, y, mid, genes, 5, size=20, max_cells=40,
                lam=20, r95=80, mode="adaptive", grid=4,
                seed=11, log=lambda _: None)
            outputs.append((mat, meta))
        a, b = outputs
        self.assertEqual(a[0].shape[0], 40)
        self.assertEqual(a[0].shape, b[0].shape)
        np.testing.assert_array_equal(a[1]["bin_x"], b[1]["bin_x"])
        np.testing.assert_array_equal(a[1]["bin_y"], b[1]["bin_y"])
        np.testing.assert_array_equal(a[1]["distance_stratum"],
                                      b[1]["distance_stratum"])
        self.assertGreater(np.sum(a[1]["distance_stratum"] == "near"), 0)
        self.assertEqual(a[0].sum(), a[1]["n_umi"].sum())

    def test_partial_edge_bin_is_excluded(self):
        mask = np.zeros((105, 105), np.int32)
        mask[40:50, 40:50] = 1
        x = np.array([65, 102])
        y = x.copy()
        mid = np.ones(2)
        genes = np.zeros(2, np.int32)
        _, meta = build_pseudo_cells(
            mask, x, y, mid, genes, 2, size=20,
            roi_mask=np.ones_like(mask, bool), mode="adaptive",
            log=lambda _: None)
        self.assertEqual(len(meta["bin_x"]), 1)
        self.assertEqual(meta["bin_x"][0], 70)

    def test_empty_input(self):
        mask = np.zeros((40, 40), np.int32)
        empty_i = np.empty(0, np.int32)
        empty_f = np.empty(0, float)
        mat, meta = build_pseudo_cells(
            mask, empty_i, empty_i, empty_f, empty_i, 3,
            size=10, mode="adaptive", log=lambda _: None)
        self.assertEqual(mat.shape, (0, 3))
        for key in ("bin_x", "bin_y", "n_umi", "dist_to_cell_px",
                    "distance_stratum", "local_cell_frac"):
            self.assertEqual(len(meta[key]), 0)


if __name__ == "__main__":
    unittest.main()
