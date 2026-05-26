"""
Dataset loader for HAM10000 / ISIC skin lesion data.

Expected folder layout:
    data/
      HAM10000_metadata.csv
      HAM10000_images_part1/   (*.jpg)
      HAM10000_images_part2/   (*.jpg)

The 7 classes:
    akiec - Actinic keratoses / intraepithelial carcinoma
    bcc   - Basal cell carcinoma
    bkl   - Benign keratosis (often mixed up with melanoma)
    df    - Dermatofibroma
    mel   - Melanoma
    nv    - Melanocytic nevi  (most samples, big imbalance issue)
    vasc  - Vascular lesions
"""

import os
import numpy as np
import pandas as pd
from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms

IMG_SIZE    = 224
NUM_CLASSES = 7

CLASS_NAMES = ["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"]
CLASS_TO_IDX = {c: i for i, c in enumerate(CLASS_NAMES)}

# MobileNetV2 was pretrained on ImageNet, so we use ImageNet normalization stats
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]


def get_train_transform() -> transforms.Compose:
    return transforms.Compose([
        transforms.Resize((IMG_SIZE + 20, IMG_SIZE + 20)),
        transforms.RandomCrop(IMG_SIZE),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(30),
        transforms.ColorJitter(brightness=0.3, contrast=0.3,
                               saturation=0.2, hue=0.05),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


def get_val_transform() -> transforms.Compose:
    return transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


class HAM10000Dataset(Dataset):
    """
    Reads HAM10000_metadata.csv and loads images from one or more image dirs.
    Supports an explicit list of image_ids for train/val splitting.
    """

    def __init__(
        self,
        csv_path: str,
        image_dirs: list[str],
        transform=None,
        ids: list[str] | None = None,
    ):
        meta = pd.read_csv(csv_path)
        if ids is not None:
            meta = meta[meta["image_id"].isin(ids)].reset_index(drop=True)

        # build a lookup from image_id to file path
        path_map: dict[str, str] = {}
        for d in image_dirs:
            for fname in os.listdir(d):
                if fname.lower().endswith((".jpg", ".jpeg", ".png")):
                    iid = os.path.splitext(fname)[0]
                    path_map[iid] = os.path.join(d, fname)

        # drop any rows where the image file is missing
        meta = meta[meta["image_id"].isin(path_map)].reset_index(drop=True)

        self.paths   = [path_map[iid] for iid in meta["image_id"]]
        self.labels  = [CLASS_TO_IDX[dx] for dx in meta["dx"]]
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img = Image.open(self.paths[idx]).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, self.labels[idx]


def compute_class_weights(labels: list[int], num_classes: int) -> torch.Tensor:
    """Inverse-frequency weights for nn.CrossEntropyLoss(weight=...)."""
    counts = np.bincount(labels, minlength=num_classes).astype(float)
    counts = np.where(counts == 0, 1, counts)
    weights = 1.0 / counts
    weights = weights / weights.sum() * num_classes   # normalize so they sum to num_classes
    return torch.tensor(weights, dtype=torch.float)


def make_weighted_sampler(labels: list[int], num_classes: int) -> WeightedRandomSampler:
    """Per-sample weights so each class gets sampled roughly equally."""
    counts = np.bincount(labels, minlength=num_classes).astype(float)
    counts = np.where(counts == 0, 1, counts)
    class_w = 1.0 / counts
    sample_w = torch.tensor([class_w[l] for l in labels], dtype=torch.float)
    return WeightedRandomSampler(sample_w, num_samples=len(sample_w), replacement=True)


def get_dataloaders(
    csv_path: str,
    image_dirs: list[str],
    val_split: float = 0.2,
    batch_size: int = 32,
    num_workers: int = 2,
    seed: int = 42,
    use_sampler: bool = True,
) -> tuple[DataLoader, DataLoader, torch.Tensor]:
    """
    Returns (train_loader, val_loader, class_weights).
    class_weights can be passed to CrossEntropyLoss as a backup to the sampler.
    """
    meta = pd.read_csv(csv_path)

    # stratified split - sample from each class separately
    rng = np.random.default_rng(seed)
    val_ids: list[str] = []
    for cls in CLASS_NAMES:
        cls_ids = meta.loc[meta["dx"] == cls, "image_id"].tolist()
        n_val = max(1, int(len(cls_ids) * val_split))
        chosen = rng.choice(cls_ids, size=n_val, replace=False).tolist()
        val_ids.extend(chosen)
    val_set  = set(val_ids)
    train_ids = [iid for iid in meta["image_id"] if iid not in val_set]

    train_ds = HAM10000Dataset(csv_path, image_dirs, get_train_transform(), ids=train_ids)
    val_ds   = HAM10000Dataset(csv_path, image_dirs, get_val_transform(),   ids=val_ids)

    class_weights = compute_class_weights(train_ds.labels, NUM_CLASSES)

    if use_sampler:
        sampler = make_weighted_sampler(train_ds.labels, NUM_CLASSES)
        train_loader = DataLoader(
            train_ds, batch_size=batch_size, sampler=sampler,
            num_workers=num_workers, pin_memory=True, drop_last=True,
        )
    else:
        train_loader = DataLoader(
            train_ds, batch_size=batch_size, shuffle=True,
            num_workers=num_workers, pin_memory=True, drop_last=True,
        )

    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
    )

    return train_loader, val_loader, class_weights


if __name__ == "__main__":
    import sys
    csv   = sys.argv[1] if len(sys.argv) > 1 else "data/HAM10000_metadata.csv"
    dirs  = sys.argv[2:] if len(sys.argv) > 2 else [
        "data/HAM10000_images_part1",
        "data/HAM10000_images_part2",
    ]
    tr, va, cw = get_dataloaders(csv, dirs, batch_size=8, num_workers=0)
    imgs, lbls = next(iter(tr))
    print(f"Train batches: {len(tr)} | Val batches: {len(va)}")
    print(f"Batch shape  : {imgs.shape}")
    print(f"Class weights: {cw.round(decimals=3).tolist()}")
