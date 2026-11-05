"""Paired bootstrap on shared resamples (same protocol as article TIT /
scripts/14 for VisDrone; cluster bootstrap over sequences for UAVDT as in
scripts/08). Resamples are drawn ONCE from seed_bootstrap and applied to every
configuration, so replicate-wise deltas are paired."""

from __future__ import annotations

import numpy as np

from exp36.loaders.cache import Dataset


def make_resamples(ds: Dataset, n_boot: int, seed_bootstrap: int) -> dict[str, np.ndarray]:
    """scene -> (n_boot, n_images) resample-count matrix."""
    rng = np.random.default_rng(seed_bootstrap)
    out = {}
    for scene, sd in ds.scenes.items():
        n = sd.n_images
        counts = np.empty((n_boot, n), dtype=np.int32)
        if sd.seq is None:  # image-level resampling
            for b in range(n_boot):
                counts[b] = np.bincount(rng.integers(0, n, n), minlength=n)
        else:  # cluster (sequence-level) resampling
            seqs = sorted(set(sd.seq))
            seq_arr = np.array(sd.seq)
            for b in range(n_boot):
                draw = rng.choice(len(seqs), size=len(seqs), replace=True)
                times = dict(zip(seqs, np.bincount(draw, minlength=len(seqs))))
                counts[b] = np.array([times[s] for s in seq_arr], dtype=np.int32)
        out[scene] = counts
    return out


def replicates(
    stats_by_scene: dict, resamples: dict[str, np.ndarray], target_cls: int
) -> np.ndarray:
    """(n_boot, n_scenes, 2) replicates for one configuration:
    [..., 0] = scene mAP@50, [..., 1] = AP@50 of the attacked class."""
    scenes = list(stats_by_scene)
    n_boot = next(iter(resamples.values())).shape[0]
    out = np.empty((n_boot, len(scenes), 2))
    for j, s in enumerate(scenes):
        st, cnt = stats_by_scene[s], resamples[s]
        for b in range(n_boot):
            aps = st.per_class_ap50(cnt[b])
            out[b, j, 0] = aps.mean()
            out[b, j, 1] = aps[target_cls]
    return out


def summarize(values: np.ndarray, alpha: float = 0.05) -> dict[str, float]:
    lo, hi = np.percentile(values, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return {
        "mean": round(float(values.mean()), 4),
        "std": round(float(values.std(ddof=1)), 4),
        "ci_lo": round(float(lo), 4),
        "ci_hi": round(float(hi), 4),
        "p_le_zero": round(float((values <= 0).mean()), 3),
    }
