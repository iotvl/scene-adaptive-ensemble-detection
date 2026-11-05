"""Stage-4/5 tests: the grid scorer reproduces compute_map exactly; ASR is 0
on identical configurations
mixed (rho<1) assembly is consistent
npz
round-trip of ClassRun is lossless."""

import numpy as np

from exp36.eval.bootstrap import make_resamples
from exp36.eval.matching import ClassRun, ConfigStats, match_frame_class
from exp36.eval.target_metrics import attack_success_rate, fppi_delta, target_recall
from exp36.fusion.methods import fuse_all_classes
from exp36.tests.test_fusion import synth
from src.evaluation.metrics import compute_map


def _stats_from(fused, gts, num_classes):
    per_class = []
    for c in range(num_classes):
        per_class.append(
            [
                match_frame_class(
                    p["boxes"][p["labels"] == c],
                    p["scores"][p["labels"] == c],
                    g["boxes"][g["labels"] == c],
                )
                for p, g in zip(fused, gts)
            ]
        )
    return ConfigStats(per_class)


def _scene(seed, n_img=12, n_cls=3):
    rng = np.random.default_rng(seed)
    fused, gts = [], []
    for _ in range(n_img):
        preds, wh = synth(rng, n_models=3, n_obj=10, n_cls=n_cls)
        gt_boxes = preds[0]["boxes"] + rng.normal(0, 3, size=preds[0]["boxes"].shape)
        gts.append({"boxes": gt_boxes, "labels": preds[0]["labels"]})
        fused.append(fuse_all_classes("wbf", preds, wh, n_cls, 3, 0.55, 0.01, 0.5))
    return fused, gts


def test_grid_scorer_equals_compute_map():
    fused, gts = _scene(1)
    st = _stats_from(fused, gts, 3)
    ref = compute_map(
        [{"image_id": i, **p} for i, p in enumerate(fused)],
        [{"image_id": i, **g} for i, g in enumerate(gts)],
        num_classes=3,
    )
    assert abs(st.map50() - ref["mAP50"]) < 1e-9
    assert abs(st.map5095() - ref["mAP50-95"]) < 1e-9


def test_asr_zero_on_clean_and_recall_range():
    fused, gts = _scene(2)
    a = _stats_from(fused, gts, 3)
    b = _stats_from(fused, gts, 3)
    assert attack_success_rate(a, b, 0, 0.25) == 0.0
    assert fppi_delta(a, b, 0, 0.25) == 0.0
    r = target_recall(a, 0, 0.25)
    assert 0.0 <= r <= 1.0


def test_asr_after_dropping_detections():
    fused, gts = _scene(3)
    clean = _stats_from(fused, gts, 3)
    dropped = [
        {
            "boxes": p["boxes"][p["labels"] != 0],
            "scores": p["scores"][p["labels"] != 0],
            "labels": p["labels"][p["labels"] != 0],
        }
        for p in fused
    ]
    pois = _stats_from(dropped, gts, 3)
    assert attack_success_rate(clean, pois, 0, 0.25) == 1.0
    assert target_recall(pois, 0, 0.25) == 0.0


def test_classrun_roundtrip(tmp_path):
    fused, gts = _scene(4)
    frames = [
        match_frame_class(
            p["boxes"][p["labels"] == 1],
            p["scores"][p["labels"] == 1],
            g["boxes"][g["labels"] == 1],
        )
        for p, g in zip(fused, gts)
    ]
    run = ClassRun(frames)
    run.save(tmp_path / "r.npz")
    back = ClassRun.load(tmp_path / "r.npz")
    np.testing.assert_array_equal(run.scores, back.scores)
    np.testing.assert_array_equal(run.tp10, back.tp10)
    np.testing.assert_array_equal(run.gt_idx, back.gt_idx)
    np.testing.assert_array_equal(run.n_gt, back.n_gt)
    f = back.frame(3)
    assert f.n == frames[3].n and f.n_gt == frames[3].n_gt


def test_bootstrap_unit_counts_identity_and_cluster():
    fused, gts = _scene(5)
    st = _stats_from(fused, gts, 3)
    ones = np.ones(len(gts), dtype=int)
    assert st.map50(ones) == st.map50()

    class DS:  # minimal stand-in for make_resamples
        pass

    class SD:
        def __init__(self, n, seq):
            self.n_images, self.seq = n, seq

    ds = DS()
    ds.scenes = {"a": SD(10, None), "b": SD(9, ["s1"] * 3 + ["s2"] * 3 + ["s3"] * 3)}
    res = make_resamples(ds, 50, 7)
    assert res["a"].shape == (50, 10) and (res["a"].sum(axis=1) == 10).all()
    # cluster resample: whole sequences drawn together (equal counts within a sequence)
    b = res["b"]
    assert (b[:, 0] == b[:, 1]).all() and (b[:, 3] == b[:, 5]).all() and (b.sum(axis=1) == 9).all()
