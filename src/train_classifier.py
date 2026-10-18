import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
import numpy as np
from pathlib import Path
import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2
from tqdm import tqdm
import pandas as pd
import sys
sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.model_classifier import DRClassifier
from src.config import APTOS_DIR, CLASSIFIER_PATH, DR_CLASSES_SHORT, IMG_SIZE_CLASSIFIER


class FundusDataset(Dataset):
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img = cv2.imread(str(self.image_paths[idx]))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        label = self.labels[idx]

        if self.transform:
            augmented = self.transform(image=img)
            img = augmented["image"]

        return img, label


def get_transforms(img_size=224):
    train_transform = A.Compose([
        A.Resize(img_size, img_size),
        A.RandomRotate90(p=0.5),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.1),
        A.Transpose(p=0.3),
        A.OneOf([
            A.GaussianBlur(blur_limit=3, p=0.5),
            A.MedianBlur(blur_limit=3, p=0.5),
        ], p=0.3),
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


def load_aptos_data(data_dir, val_split=0.15):
    data_dir = Path(data_dir)
    image_paths = []
    labels = []

    for class_idx, class_name in enumerate(DR_CLASSES_SHORT):
        class_dir = data_dir / class_name
        if not class_dir.exists():
            class_dir = data_dir / class_name.lower().replace(" ", "_")
        if not class_dir.exists():
            class_dir = data_dir / class_name.lower()
        if not class_dir.exists():
            print(f"Warning: {class_dir} not found, trying train/{class_name}...")
            class_dir = data_dir / "train" / class_name

        if class_dir.exists():
            for ext in ["*.jpg", "*.jpeg", "*.png"]:
                for img_path in class_dir.glob(ext):
                    image_paths.append(img_path)
                    labels.append(class_idx)
        else:
            print(f"Warning: Directory {class_dir} does not exist. Skipping class {class_name}.")

    if len(image_paths) == 0:
        csv_candidates = list(data_dir.glob("train*.csv"))
        csv_file = csv_candidates[0] if csv_candidates else data_dir / "train.csv"

        if csv_file.exists():
            import pandas as pd
            df = pd.read_csv(csv_file)

            img_dir = data_dir / "train_images"
            if not img_dir.exists():
                img_dir = data_dir / "train"
            nested = img_dir / "train_images"
            if nested.exists():
                img_dir = nested

            id_col = "id_code" if "id_code" in df.columns else df.columns[0]
            label_col = "diagnosis" if "diagnosis" in df.columns else df.columns[1]

            for _, row in df.iterrows():
                img_name = str(row[id_col]).split(".")[0]
                found = False
                for ext in [".png", ".jpg", ".jpeg"]:
                    img_path = img_dir / f"{img_name}{ext}"
                    if img_path.exists():
                        image_paths.append(img_path)
                        labels.append(int(row[label_col]))
                        found = True
                        break
                if not found:
                    img_path = img_dir / img_name
                    if img_path.exists():
                        image_paths.append(img_path)
                        labels.append(int(row[label_col]))

    assert len(image_paths) > 0, f"No images found in {data_dir}"

    indices = np.random.RandomState(42).permutation(len(image_paths))
    split = int(len(indices) * (1 - val_split))

    train_paths = [image_paths[i] for i in indices[:split]]
    train_labels = [labels[i] for i in indices[:split]]
    val_paths = [image_paths[i] for i in indices[split:]]
    val_labels = [labels[i] for i in indices[split:]]

    print(f"Loaded {len(train_paths)} training, {len(val_paths)} validation images")
    return train_paths, train_labels, val_paths, val_labels


def train_classifier(data_dir=None, num_epochs=20, batch_size=16, lr=1e-4):
    if data_dir is None:
        data_dir = APTOS_DIR

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    train_paths, train_labels, val_paths, val_labels = load_aptos_data(data_dir)

    train_transform, val_transform = get_transforms(IMG_SIZE_CLASSIFIER)

    train_dataset = FundusDataset(train_paths, train_labels, train_transform)
    val_dataset = FundusDataset(val_paths, val_labels, val_transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)

    model = DRClassifier(num_classes=5)
    model.to(device)

    class_counts = [0] * 5
    for label in train_labels:
        class_counts[label] += 1
    total = sum(class_counts)
    class_weights = [total / (5 * c) for c in class_counts]
    class_weights_tensor = torch.tensor(class_weights, dtype=torch.float32).to(device)
    print(f"Class weights: {class_weights}")

    criterion = nn.CrossEntropyLoss(weight=class_weights_tensor, label_smoothing=0.1)
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=3, verbose=True)

    best_acc = 0.0
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")
        for images, labels in pbar:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            _, predicted = torch.max(outputs, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()
            train_loss += loss.item()

            pbar.set_postfix({
                "loss": f"{loss.item():.4f}",
                "acc": f"{train_correct/train_total:.4f}"
            })

        model.eval()
        val_correct = 0
        val_total = 0
        val_loss = 0.0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()

        val_acc = val_correct / val_total
        scheduler.step(val_acc)
        print(f"Epoch {epoch+1}: Train Acc: {train_correct/train_total:.4f}, Val Acc: {val_acc:.4f}, LR: {optimizer.param_groups[0]['lr']:.2e}")

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save({
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_accuracy": val_acc,
                "epoch": epoch,
            }, CLASSIFIER_PATH)
            print(f"  Model saved with val_acc: {val_acc:.4f}")

    print(f"Training complete. Best validation accuracy: {best_acc:.4f}")
    return CLASSIFIER_PATH


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default=None)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    args = parser.parse_args()

    train_classifier(args.data_dir, args.epochs, args.batch_size, args.lr)
