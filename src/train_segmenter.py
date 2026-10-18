import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
from pathlib import Path
import cv2
import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm
import sys
sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.model_segmenter import DRSegmenter
from src.config import IDRiD_DIR, DDR_SEG_DIR, SEGMENTER_PATH, LESION_NAMES, IMG_SIZE_SEGMENTER


class SegmentationDataset(Dataset):
    def __init__(self, image_paths, mask_paths, transform=None):
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = cv2.imread(str(self.image_paths[idx]))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        masks = []
        for mp in self.mask_paths[idx]:
            mask = cv2.imread(str(mp), cv2.IMREAD_GRAYSCALE)
            if mask is None:
                mask = np.zeros(image.shape[:2], dtype=np.uint8)
            masks.append(mask)

        mask = np.stack(masks, axis=-1).astype(np.float32) / 255.0

        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented["image"]
            mask = augmented["mask"].permute(2, 0, 1).float()

        return image, mask


def get_transforms(img_size=512):
    train_transform = A.Compose([
        A.Resize(img_size, img_size),
        A.RandomRotate90(p=0.5),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.3),
        A.Transpose(p=0.3),
        A.OneOf([
            A.RandomBrightnessContrast(p=0.5),
            A.HueSaturationValue(p=0.5),
        ], p=0.3),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])

    val_transform = A.Compose([
        A.Resize(img_size, img_size),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])

    return train_transform, val_transform


def load_idrid_data(data_dir, val_split=0.15):
    data_dir = Path(data_dir)
    image_dir = data_dir / "images"
    mask_dir = data_dir / "masks"

    if not image_dir.exists():
        image_dir = data_dir / "train" / "images"
        mask_dir = data_dir / "train" / "masks"

    if not image_dir.exists():
        for subdir in ["1. Original Images", "A. Fundus Images"]:
            candidate = data_dir / subdir
            if candidate.exists():
                image_dir = candidate
                break

    assert image_dir.exists(), f"Image directory not found in {data_dir}"

    if not mask_dir.exists():
        for subdir in ["2. All Segmentation Groundtruths", "B. Lesion Segmentation"]:
            candidate = data_dir / subdir
            if candidate.exists():
                mask_dir = candidate
                break

    if not mask_dir.exists():
        mask_dir = data_dir
        for lesion_type in ["MA", "HE", "EX", "SE"]:
            candidate = data_dir / lesion_type
            if candidate.exists():
                mask_dir = data_dir
                break

    lesion_subdirs = {
        "Microaneurysms": ["MA", "Microaneurysms", "microaneurysms"],
        "Hemorrhages": ["HE", "Hemorrhages", "hemorrhages"],
        "Hard Exudates": ["EX", "Exudates", "exudates", "HardExudates"],
        "Cotton Wool Spots": ["SE", "SoftExudates", "CWS", "cotton_wool_spots"],
    }

    extensions = ["*.jpg", "*.jpeg", "*.png", "*.tif", "*.bmp"]
    image_paths = []
    for ext in extensions:
        image_paths.extend(sorted(image_dir.glob(ext)))
        image_paths.extend(sorted(image_dir.glob(ext.upper())))

    assert len(image_paths) > 0, f"No images found in {image_dir}"

    mask_paths_list = []
    for img_path in image_paths:
        img_name = img_path.stem
        lesion_masks = []
        for lesion_name, subdir_names in lesion_subdirs.items():
            found = False
            for sd in subdir_names:
                mask_candidate = mask_dir / sd / f"{img_name}.png"
                if not mask_candidate.exists():
                    mask_candidate = mask_dir / sd / f"{img_name}.tif"
                if not mask_candidate.exists():
                    mask_candidate = mask_dir / f"{sd}" / f"{img_name}.png"
                if not mask_candidate.exists():
                    for ext in [".png", ".tif", ".jpg", ".bmp"]:
                        mc = mask_dir / sd / f"{img_name}{ext}"
                        if mc.exists():
                            mask_candidate = mc
                            break
                if mask_candidate.exists():
                    lesion_masks.append(mask_candidate)
                    found = True
                    break
            if not found:
                lesion_masks.append(None)
        mask_paths_list.append(lesion_masks)

    items = [(p, m) for p, m in zip(image_paths, mask_paths_list)]
    np.random.RandomState(42).shuffle(items)
    split = int(len(items) * (1 - val_split))

    train_items = items[:split]
    val_items = items[split:]

    train_paths = [it[0] for it in train_items]
    train_masks = [it[1] for it in train_items]
    val_paths = [it[0] for it in val_items]
    val_masks = [it[1] for it in val_items]

    print(f"Loaded {len(train_paths)} training, {len(val_paths)} validation images")
    return train_paths, train_masks, val_paths, val_masks


def load_ddr_data(data_dir, img_size=512):
    data_dir = Path(data_dir)
    lesion_dirs = ["EX", "HE", "MA", "SE"]

    ann_test = "tet"
    if not (data_dir / "annotations" / "tet").exists():
        ann_test = "test"

    splits = {
        "train": ("train", "train"),
        "val": ("val", "val"),
        "test": ("test", ann_test),
    }

    train_items, val_items = [], []
    for split_name, (img_sub, ann_sub) in splits.items():
        img_dir = data_dir / "images" / img_sub
        ann_base = data_dir / "annotations" / ann_sub

        if not img_dir.exists():
            continue

        images = sorted(img_dir.glob("*"))
        for img_path in images:
            stem = img_path.stem
            mask_paths = []
            for ld in lesion_dirs:
                candidates = list((ann_base / ld).glob(f"{stem}.*"))
                if candidates:
                    mask_paths.append(candidates[0])
                else:
                    mask_paths.append(None)

            if any(m is not None for m in mask_paths):
                item = (img_path, mask_paths)
                if split_name == "train":
                    train_items.append(item)
                else:
                    val_items.append(item)

    print(f"DDR: Loaded {len(train_items)} training, {len(val_items)} validation images")
    train_paths = [it[0] for it in train_items]
    train_masks = [it[1] for it in train_items]
    val_paths = [it[0] for it in val_items]
    val_masks = [it[1] for it in val_items]
    return train_paths, train_masks, val_paths, val_masks


def train_segmenter(data_dir=None, num_epochs=30, batch_size=4, lr=1e-4):
    if data_dir is None:
        data_dir = IDRiD_DIR

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    data_dir = Path(data_dir)
    if (data_dir / "images").exists() and (data_dir / "annotations").exists():
        train_paths, train_masks, val_paths, val_masks = load_ddr_data(data_dir)
    else:
        train_paths, train_masks, val_paths, val_masks = load_idrid_data(data_dir)

    train_transform, val_transform = get_transforms(IMG_SIZE_SEGMENTER)

    train_dataset = SegmentationDataset(train_paths, train_masks, train_transform)
    val_dataset = SegmentationDataset(val_paths, val_masks, val_transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    model = DRSegmenter(num_lesions=4)
    model.to(device)

    start_epoch = 0
    best_dice = 0.0
    checkpoint_path = Path(SEGMENTER_PATH)
    if checkpoint_path.exists():
        ckpt = torch.load(str(checkpoint_path), map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        start_epoch = ckpt.get("epoch", 0) + 1
        best_dice = ckpt.get("val_dice", 0.0)
        print(f"Resumed from checkpoint: epoch {start_epoch}, val_dice={best_dice:.4f}")

    def focal_loss(pred, target, gamma=2.0, alpha=0.25):
        bce = F.binary_cross_entropy_with_logits(pred, target, reduction="none")
        pt = torch.exp(-bce)
        focal = alpha * (1 - pt) ** gamma * bce
        return focal.mean()

    def dice_loss(pred, target, smooth=1.0):
        pred = torch.sigmoid(pred)
        intersection = (pred * target).sum(dim=(2, 3))
        union = pred.sum(dim=(2, 3)) + target.sum(dim=(2, 3))
        return 1 - (2.0 * intersection + smooth) / (union + smooth)

    def combined_loss(pred, target):
        return focal_loss(pred, target) + dice_loss(pred, target).mean()

    criterion = combined_loss
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)

    for epoch in range(start_epoch, num_epochs):
        model.train()
        train_loss = 0.0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")

        for images, masks in pbar:
            images, masks = images.to(device), masks.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        scheduler.step()

        model.eval()
        val_dice = 0.0
        val_count = 0
        with torch.no_grad():
            for images, masks in val_loader:
                images, masks = images.to(device), masks.to(device)
                outputs = torch.sigmoid(model(images))
                preds = (outputs > 0.3).float()

                intersection = (preds * masks).sum(dim=(2, 3))
                union = preds.sum(dim=(2, 3)) + masks.sum(dim=(2, 3))
                dice = (2 * intersection) / (union + 1e-6)
                val_dice += dice.mean().item()
                val_count += 1

        avg_dice = val_dice / val_count
        print(f"Epoch {epoch+1}: Loss={train_loss/len(train_loader):.4f}, Val Dice={avg_dice:.4f}")

        if avg_dice > best_dice:
            best_dice = avg_dice
            torch.save({
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_dice": avg_dice,
                "epoch": epoch,
            }, SEGMENTER_PATH)
            print(f"  Model saved with val_dice: {avg_dice:.4f}")

    print(f"Training complete. Best validation Dice: {best_dice:.4f}")
    return SEGMENTER_PATH


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default=str(DDR_SEG_DIR))
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    args = parser.parse_args()

    train_segmenter(args.data_dir, args.epochs, args.batch_size, args.lr)
