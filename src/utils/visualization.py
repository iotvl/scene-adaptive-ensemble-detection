"""
Visualization utilities for detection results and comparisons.
"""

from __future__ import annotations

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from src.constants import VISDRONE_CLASSES

CLASS_COLORS = [
    (255, 0, 0),
    (0, 255, 0),
    (0, 0, 255),
    (255, 255, 0),
    (255, 0, 255),
    (0, 255, 255),
    (128, 0, 0),
    (0, 128, 0),
    (0, 0, 128),
    (128, 128, 0),
]


def plot_detections(
    image: np.ndarray,
    boxes: np.ndarray,
    labels: np.ndarray,
    scores: np.ndarray,
    class_names: list[str] = VISDRONE_CLASSES,
    score_threshold: float = 0.25,
    save_path: str | None = None,
    title: str = "",
) -> np.ndarray:
    """
    Draw detection boxes on image.

    Args:
        image:  BGR image (H, W, 3)
        boxes:  (N, 4) in xyxy format
        labels: (N,) class indices
        scores: (N,) confidence scores
        class_names: list of class names
        score_threshold: minimum score to display
        save_path: optional path to save the image
        title: title for the plot

    Returns:
        Annotated image
    """
    img = image.copy()

    for box, label, score in zip(boxes, labels, scores):
        if score < score_threshold:
            continue

        x1, y1, x2, y2 = map(int, box)
        label_idx = int(label)
        color = CLASS_COLORS[label_idx % len(CLASS_COLORS)]

        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)

        text = f"{class_names[label_idx]}: {score:.2f}"
        text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
        cv2.rectangle(img, (x1, y1 - text_size[1] - 4), (x1 + text_size[0], y1), color, -1)
        cv2.putText(img, text, (x1, y1 - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    if save_path:
        cv2.imwrite(save_path, img)

    return img


def plot_comparison(
    results_df: pd.DataFrame,
    metric: str = "mAP50",
    group_by: str = "scene",
    save_path: str | None = None,
):
    """
    Plot comparison of models across scenes.

    Args:
        results_df: DataFrame with columns [model, scene, metric_value]
        metric: which metric to plot
        group_by: grouping column
        save_path: optional path to save
    """
    fig, ax = plt.subplots(figsize=(12, 6))

    pivot = results_df.pivot_table(values=metric, index="model", columns=group_by, aggfunc="mean")

    pivot.plot(kind="bar", ax=ax, rot=45)
    ax.set_ylabel(metric)
    ax.set_title(f"Model Comparison: {metric} by {group_by}")
    ax.legend(title=group_by, bbox_to_anchor=(1.05, 1), loc="upper left")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()


def plot_resource_comparison(results: list[dict], save_path: str | None = None):
    """
    Plot resource comparison: mAP vs FPS scatter, params bar chart.
    """
    df = pd.DataFrame(results)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # 1) mAP vs FPS
    ax = axes[0]
    for _, row in df.iterrows():
        ax.scatter(row["fps"], row["mAP50"], s=row["params_m"] * 10, alpha=0.7)
        ax.annotate(row["model"], (row["fps"], row["mAP50"]), fontsize=8)
    ax.set_xlabel("FPS")
    ax.set_ylabel("mAP@50")
    ax.set_title("Accuracy vs Speed (size = #params)")
    ax.grid(True, alpha=0.3)

    # 2) Params comparison
    ax = axes[1]
    ax.barh(df["model"], df["params_m"], color=sns.color_palette("viridis", len(df)))
    ax.set_xlabel("Parameters (M)")
    ax.set_title("Model Size")

    # 3) Latency comparison
    ax = axes[2]
    ax.barh(df["model"], df["latency_ms"], color=sns.color_palette("magma", len(df)))
    ax.set_xlabel("Latency (ms)")
    ax.set_title("Inference Latency")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()


def plot_scene_distribution(scene_counts: dict[str, int], save_path: str | None = None):
    """Plot distribution of scenes in the dataset."""
    fig, ax = plt.subplots(figsize=(8, 5))

    scenes = list(scene_counts.keys())
    counts = list(scene_counts.values())
    colors = ["#FFD700", "#191970", "#4682B4"]

    bars = ax.bar(scenes, counts, color=colors[: len(scenes)])

    for bar, count in zip(bars, counts):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 10,
            str(count),
            ha="center",
            fontsize=12,
            fontweight="bold",
        )

    ax.set_ylabel("Number of images")
    ax.set_title("Scene Distribution in Dataset")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()
