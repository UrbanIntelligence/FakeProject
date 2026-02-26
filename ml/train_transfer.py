#!/usr/bin/env python3
import argparse
import csv
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import List

import numpy as np
import torch
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms


@dataclass
class Sample:
    rel_path: str
    label: int


class FaceDataset(Dataset):
    def __init__(self, samples: List[Sample], root: Path, transform):
        self.samples = samples
        self.root = root
        self.transform = transform

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        s = self.samples[idx]
        image_path = self.root / s.rel_path
        image = Image.open(image_path).convert("RGB")
        image = self.transform(image)
        label = torch.tensor(s.label, dtype=torch.long)
        return image, label


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def read_split(csv_path: Path, per_class_limit: int = 0) -> List[Sample]:
    real: List[Sample] = []
    fake: List[Sample] = []

    with csv_path.open("r", newline="") as f:
        reader = csv.reader(f)
        next(reader, None)
        for row in reader:
            if len(row) < 6:
                continue
            label_str = row[4].strip()
            rel_path = row[5].strip()
            if not rel_path:
                continue

            label = 1 if label_str == "real" else 0
            item = Sample(rel_path=rel_path, label=label)

            if per_class_limit > 0:
                if label == 1 and len(real) < per_class_limit:
                    real.append(item)
                elif label == 0 and len(fake) < per_class_limit:
                    fake.append(item)
                if len(real) >= per_class_limit and len(fake) >= per_class_limit:
                    break
            else:
                if label == 1:
                    real.append(item)
                else:
                    fake.append(item)

    samples = real + fake
    random.shuffle(samples)
    return samples


def build_model(num_classes: int = 2, freeze_backbone: bool = True):
    try:
        weights = models.ResNet18_Weights.IMAGENET1K_V1
        model = models.resnet18(weights=weights)
    except Exception:
        model = models.resnet18(weights=None)

    if freeze_backbone:
        for name, param in model.named_parameters():
            if not name.startswith("fc."):
                param.requires_grad = False

    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    return model


def evaluate(model, loader, device):
    model.eval()
    total = 0
    correct = 0
    total_loss = 0.0
    criterion = nn.CrossEntropyLoss()

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            logits = model(x)
            loss = criterion(logits, y)
            preds = torch.argmax(logits, dim=1)
            correct += (preds == y).sum().item()
            total += y.size(0)
            total_loss += loss.item() * y.size(0)

    acc = correct / total if total else 0.0
    avg_loss = total_loss / total if total else 0.0
    return avg_loss, acc


def train(args):
    set_seed(args.seed)

    data_root = Path(args.data_root)
    train_csv = Path(args.train_csv)
    valid_csv = Path(args.valid_csv)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    train_samples = read_split(train_csv, args.train_per_class)
    valid_samples = read_split(valid_csv, args.valid_per_class)

    train_tfm = transforms.Compose(
        [
            transforms.Resize((256, 256)),
            transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.1, hue=0.02),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    valid_tfm = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    train_ds = FaceDataset(train_samples, data_root, train_tfm)
    valid_ds = FaceDataset(valid_samples, data_root, valid_tfm)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)
    valid_loader = DataLoader(valid_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(num_classes=2, freeze_backbone=args.freeze_backbone).to(device)

    criterion = nn.CrossEntropyLoss()
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_acc = 0.0
    best_epoch = -1
    history = []

    for epoch in range(args.epochs):
        model.train()
        running_loss = 0.0
        running_correct = 0
        seen = 0

        for x, y in train_loader:
            x = x.to(device)
            y = y.to(device)

            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()

            preds = torch.argmax(logits, dim=1)
            running_correct += (preds == y).sum().item()
            seen += y.size(0)
            running_loss += loss.item() * y.size(0)

        scheduler.step()

        train_loss = running_loss / seen if seen else 0.0
        train_acc = running_correct / seen if seen else 0.0
        valid_loss, valid_acc = evaluate(model, valid_loader, device)

        row = {
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "valid_loss": valid_loss,
            "valid_acc": valid_acc,
            "lr": optimizer.param_groups[0]["lr"],
            "freeze_backbone": args.freeze_backbone,
        }
        history.append(row)

        print(f"epoch={epoch+1}/{args.epochs} train_acc={train_acc:.4f} valid_acc={valid_acc:.4f} train_loss={train_loss:.4f} valid_loss={valid_loss:.4f}")

        if valid_acc > best_acc:
            best_acc = valid_acc
            best_epoch = epoch + 1
            ckpt_path = out_dir / "best_resnet18.pt"
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "best_acc": best_acc,
                    "best_epoch": best_epoch,
                    "class_map": {"fake": 0, "real": 1},
                    "input_size": 224,
                },
                ckpt_path,
            )

    summary = {
        "best_valid_acc": best_acc,
        "best_epoch": best_epoch,
        "epochs": args.epochs,
        "train_samples": len(train_samples),
        "valid_samples": len(valid_samples),
        "device": str(device),
    }

    with (out_dir / "history.json").open("w") as f:
        json.dump(history, f, indent=2)
    with (out_dir / "summary.json").open("w") as f:
        json.dump(summary, f, indent=2)

    print("training complete")
    print(json.dumps(summary, indent=2))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", default="archive/rvf10k")
    parser.add_argument("--train-csv", default="archive/train.csv")
    parser.add_argument("--valid-csv", default="archive/valid.csv")
    parser.add_argument("--out-dir", default="ml/artifacts")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--train-per-class", type=int, default=1500)
    parser.add_argument("--valid-per-class", type=int, default=600)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--freeze-backbone", action="store_true", default=True)
    parser.add_argument("--unfreeze-backbone", dest="freeze_backbone", action="store_false")
    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
