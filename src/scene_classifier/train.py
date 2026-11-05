"""
Training pipeline for the scene classifier CNN.
"""

import json
import logging
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm import tqdm

from src.constants import IMAGE_EXTENSIONS, IMAGENET_MEAN, IMAGENET_STD

from .classifier import SCENE_TO_IDX, SceneClassifierCNN

logger = logging.getLogger(__name__)


class SceneDataset(Dataset[tuple[Any, int]]):
    """Dataset for scene classification training."""

    def __init__(
        self,
        root_dir: str,
        split: str = "train",
        transform=None,
        exclude_synthetic: bool = False,
    ):
        """
        Args:
            root_dir: root of scene-split directory
                      (with day/images/train, night/images/train, etc.)
            split: "train", "val", or "test"
            transform: torchvision transforms
            exclude_synthetic: drop "aug_*" images (synthetic rain/fog rendered
                onto day images) — they teach overlay textures the real
                heuristic-labelled eval images never show
        """
        self.root = Path(root_dir)
        self.split = split
        self.transform = transform

        self.samples = []  # (image_path, label_idx)

        for scene_name, label_idx in SCENE_TO_IDX.items():
            images_dir = self.root / scene_name / "images" / split
            if not images_dir.exists():
                continue

            for img_path in sorted(
                p
                for p in images_dir.iterdir()
                if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
            ):
                if exclude_synthetic and img_path.name.startswith("aug_"):
                    continue
                self.samples.append((str(img_path), label_idx))

        logger.info("SceneDataset [%s]: %d images", split, len(self.samples))
        for scene, idx in SCENE_TO_IDX.items():
            count = sum(1 for _, label in self.samples if label == idx)
            logger.info("  %s: %d", scene, count)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        img_path, label = self.samples[index]

        img = cv2.imread(img_path)
        if img is None:
            raise FileNotFoundError(f"Could not read image: {img_path}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        if self.transform:
            img = self.transform(img)

        return img, label


def get_transforms(input_size: int = 224, is_train: bool = True, color_jitter: bool = True):
    """Get data transforms.

    color_jitter=False matters when the LABELS are functions of brightness /
    saturation (the heuristic scene split): jittering those statistics moves a
    day image across the day/rain decision boundary while keeping its label,
    actively erasing the class-defining signal.
    """
    if is_train:
        aug: list[Any] = [
            transforms.ToPILImage(),
            transforms.RandomResizedCrop(input_size, scale=(0.8, 1.0)),
            transforms.RandomHorizontalFlip(),
        ]
        if color_jitter:
            aug.append(transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3))
        aug += [
            transforms.ToTensor(),
            transforms.Normalize(list(IMAGENET_MEAN), list(IMAGENET_STD)),
        ]
        return transforms.Compose(aug)
    else:
        return transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.Resize(int(input_size * 1.14)),
                transforms.CenterCrop(input_size),
                transforms.ToTensor(),
                transforms.Normalize(list(IMAGENET_MEAN), list(IMAGENET_STD)),
            ]
        )


def train_scene_classifier(
    data_root: str,
    output_dir: str = "experiments/scene_classifier",
    backbone: str = "mobilenet_v3_small",
    input_size: int = 224,
    batch_size: int = 32,
    epochs: int = 30,
    lr: float = 0.001,
    weight_decay: float = 0.0001,
    device: str = "auto",
    num_workers: int = 4,
    color_jitter: bool = True,
    exclude_synthetic: bool = False,
    select_by: str = "acc",
) -> str:
    """
    Train scene classifier.

    Args:
        data_root: path to scene-split data root
        output_dir: where to save checkpoints
        backbone: backbone architecture
        input_size: input image size
        batch_size: batch size
        epochs: number of epochs
        lr: learning rate
        weight_decay: optimizer weight decay
        device: compute device
        num_workers: dataloader workers
        color_jitter: apply ColorJitter in training (disable when labels are
            brightness/saturation functions — see get_transforms)
        exclude_synthetic: drop synthetic "aug_*" rain/fog training images
        select_by: checkpoint-selection metric, "acc" (overall val accuracy)
            or "macro" (mean per-class val accuracy — resists day dominating
            the tiny val split)

    Returns:
        Path to best model checkpoint
    """
    if select_by not in ("acc", "macro"):
        raise ValueError(f"select_by must be 'acc' or 'macro', got {select_by!r}")
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Device
    from src.utils.general import get_device

    dev = get_device(device)

    logger.info("Training Scene Classifier")
    logger.info("  Backbone: %s", backbone)
    logger.info("  Device:   %s", dev)
    logger.info("  Epochs:   %d", epochs)

    # Datasets
    train_dataset = SceneDataset(
        data_root,
        "train",
        get_transforms(input_size, is_train=True, color_jitter=color_jitter),
        exclude_synthetic=exclude_synthetic,
    )
    val_dataset = SceneDataset(data_root, "val", get_transforms(input_size, is_train=False))

    # Handle class imbalance with weighted sampling
    labels = [s[1] for s in train_dataset.samples]
    class_counts = np.bincount(labels, minlength=len(SCENE_TO_IDX))
    class_weights = 1.0 / (class_counts + 1e-6)
    sample_weights = [class_weights[lbl] for lbl in labels]
    sampler = torch.utils.data.WeightedRandomSampler(sample_weights, len(sample_weights))

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    # Model
    model = SceneClassifierCNN(backbone=backbone, pretrained=True)
    model.to(dev)

    # Loss with class weights
    loss_weights = torch.FloatTensor(class_weights / class_weights.sum()).to(dev)
    criterion = nn.CrossEntropyLoss(weight=loss_weights)

    # Optimizer and scheduler
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    # Training loop
    best_val_acc = 0.0
    best_model_path = out_dir / "best_scene_classifier.pth"
    history = []

    for epoch in range(1, epochs + 1):
        # Train
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{epochs} [train]")
        for images, targets in pbar:
            images = images.to(dev)
            targets = targets.to(dev)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * images.size(0)
            _, predicted = outputs.max(1)
            train_correct += predicted.eq(targets).sum().item()
            train_total += targets.size(0)

            pbar.set_postfix(
                {
                    "loss": f"{loss.item():.4f}",
                    "acc": f"{100 * train_correct / train_total:.1f}%",
                }
            )

        train_loss /= train_total
        train_acc = train_correct / train_total

        # Validate
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        cls_correct = np.zeros(len(SCENE_TO_IDX))
        cls_total = np.zeros(len(SCENE_TO_IDX))

        with torch.no_grad():
            for images, targets in val_loader:
                images = images.to(dev)
                targets = targets.to(dev)

                outputs = model(images)
                loss = criterion(outputs, targets)

                val_loss += loss.item() * images.size(0)
                _, predicted = outputs.max(1)
                val_correct += predicted.eq(targets).sum().item()
                val_total += targets.size(0)
                for c in range(len(SCENE_TO_IDX)):
                    mask = targets == c
                    cls_total[c] += int(mask.sum().item())
                    cls_correct[c] += int(predicted[mask].eq(targets[mask]).sum().item())

        val_loss /= max(val_total, 1)
        val_acc = val_correct / max(val_total, 1)
        present = cls_total > 0
        val_macro_acc = float((cls_correct[present] / cls_total[present]).mean())

        scheduler.step()

        epoch_info = {
            "epoch": epoch,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_loss": val_loss,
            "val_acc": val_acc,
            "val_macro_acc": val_macro_acc,
            "lr": optimizer.param_groups[0]["lr"],
        }
        history.append(epoch_info)

        logger.info(
            "  Epoch %d: train_loss=%.4f, train_acc=%.3f, val_loss=%.4f, "
            "val_acc=%.3f, val_macro_acc=%.3f",
            epoch,
            train_loss,
            train_acc,
            val_loss,
            val_acc,
            val_macro_acc,
        )

        # Save best model by the configured selection metric
        select_metric = val_acc if select_by == "acc" else val_macro_acc
        if select_metric > best_val_acc:
            best_val_acc = select_metric
            torch.save(model.state_dict(), best_model_path)
            logger.info("  -> Saved best model (%s=%.3f)", select_by, select_metric)

    # Save training history
    with open(out_dir / "training_history.json", "w") as f:
        json.dump(history, f, indent=2)

    # Save final model
    torch.save(model.state_dict(), out_dir / "final_scene_classifier.pth")

    logger.info("Training complete! Best val_acc: %.3f", best_val_acc)
    logger.info("Best model: %s", best_model_path)

    return str(best_model_path)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train scene classifier")
    parser.add_argument("--data", required=True, help="Scene-split data root")
    parser.add_argument("--output", default="experiments/scene_classifier")
    parser.add_argument("--backbone", default="mobilenet_v3_small")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    train_scene_classifier(
        data_root=args.data,
        output_dir=args.output,
        backbone=args.backbone,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        device=args.device,
    )
