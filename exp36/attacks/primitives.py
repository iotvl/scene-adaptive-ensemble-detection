"""Attack primitives on the OUTPUT of a pool member (spec §3, stage 3).

Rationale (goes into the thesis text, not as an excuse): data poisoning and
weight backdoors manifest exclusively through the model's output, and the
ensemble consumes exactly that output; simulating at the level of cached
predictions is therefore a faithful model of the threat as seen by the fusion
stage. The trigger condition of a backdoor is modelled by the fraction rho of
frames on which the attack is active.

Determinism: every random choice depends only on `seed_attack` plus the
frame / member identity, never on iteration order, so any two configurations
that share a member and a frame see the same phantoms, and every fusion method
sees the same triggered frames.
"""

from __future__ import annotations

import hashlib

import numpy as np


def _rng(seed: int, *parts) -> np.random.Generator:
    h = hashlib.sha256(("|".join(str(p) for p in (seed, *parts))).encode()).digest()
    return np.random.default_rng(int.from_bytes(h[:8], "little"))


def triggered_frames(
    n_images: int, rho: float, seed_attack: int, dataset: str, scene: str
) -> np.ndarray:
    """Boolean mask of frames on which the backdoor fires. rho=1 -> all frames.
    Shared by every member, every fusion method and every poisoned set (one
    trigger condition per frame)."""
    if rho >= 1.0:
        return np.ones(n_images, dtype=bool)
    rng = _rng(seed_attack, "trigger", dataset, scene, rho)
    n_on = int(round(rho * n_images))
    mask = np.zeros(n_images, dtype=bool)
    mask[rng.choice(n_images, size=n_on, replace=False)] = True
    return mask


# ---------------------------------------------------------------- A1 suppression
def attack_suppress(pred: dict, cls: int, active: bool) -> dict:
    """A1: drop every detection of class `cls` from the member's output on a
    triggered frame. Never adds boxes; on inactive frames returns the input."""
    if not active:
        return pred
    keep = pred["labels"] != cls
    return {
        "boxes": pred["boxes"][keep],
        "scores": pred["scores"][keep],
        "labels": pred["labels"][keep],
    }


# ---------------------------------------------------------------- A2 injection
def sample_phantoms(
    n: int,
    box_prior: np.ndarray,
    img_wh: tuple[int, int],
    conf_range: tuple[float, float],
    seed_attack: int,
    dataset: str,
    scene: str,
    member: str,
    frame: int,
) -> tuple[np.ndarray, np.ndarray]:
    """n phantom boxes of the target class for one (member, frame): sizes drawn
    from the empirical TRAIN (w, h) prior, positions uniform inside the frame,
    confidence ~ U(conf_range). Sampled from the prior, NOT optimised against
    the fusion — a lower bound on the damage an adaptive adversary could do."""
    rng = _rng(seed_attack, "inject", dataset, scene, member, frame)
    w, h = img_wh
    wh = box_prior[rng.integers(0, len(box_prior), size=n)]
    bw = np.minimum(wh[:, 0], w - 1)
    bh = np.minimum(wh[:, 1], h - 1)
    x1 = rng.uniform(0, w - bw)
    y1 = rng.uniform(0, h - bh)
    boxes = np.stack([x1, y1, x1 + bw, y1 + bh], axis=1)
    scores = rng.uniform(conf_range[0], conf_range[1], size=n)
    return boxes, scores


def attack_inject(pred: dict, cls: int, phantoms: tuple[np.ndarray, np.ndarray]) -> dict:
    """A2: append phantom boxes of class `cls`. Never removes boxes."""
    pb, ps = phantoms
    if len(ps) == 0:
        return pred
    return {
        "boxes": np.concatenate([pred["boxes"], pb]),
        "scores": np.concatenate([pred["scores"], ps]),
        "labels": np.concatenate([pred["labels"], np.full(len(ps), cls, dtype=int)]),
    }


# ---------------------------------------------------------------- A3 inflation
def attack_inflate(pred: dict, factor: float = 1.5) -> dict:
    """A3: multiply every confidence by `factor`, clip to 1. Box set and
    coordinates untouched; only the ranking (and WBF's weighting) is corrupted."""
    return {
        "boxes": pred["boxes"],
        "scores": np.minimum(pred["scores"] * factor, 1.0),
        "labels": pred["labels"],
    }
