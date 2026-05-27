import os
import sys
import time
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.amp import autocast, GradScaler
from tqdm import tqdm

from config import *
from model import NestedUNetV2
from dataset import ISIC2018Dataset


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class DiceLoss(nn.Module):
    def __init__(self, smooth=1e-5):
        super().__init__()
        self.smooth = smooth

    def forward(self, pred, target):
        pred = torch.sigmoid(pred)
        pred = pred.contiguous().view(-1)
        target = target.contiguous().view(-1)
        intersection = (pred * target).sum()
        return 1 - (2. * intersection + self.smooth) / (pred.sum() + target.sum() + self.smooth)


class ComboLoss(nn.Module):
    def __init__(self, bce_weight=0.5, dice_weight=0.5):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()
        self.dice = DiceLoss()
        self.bce_w = bce_weight
        self.dice_w = dice_weight

    def forward(self, pred, target):
        return self.bce_w * self.bce(pred, target) + self.dice_w * self.dice(pred, target)


def dice_score(pred, target, smooth=1e-5):
    pred = torch.sigmoid(pred)
    pred = (pred > 0.5).float()
    pred = pred.contiguous().view(-1)
    target = target.contiguous().view(-1)
    intersection = (pred * target).sum()
    return (2. * intersection + smooth) / (pred.sum() + target.sum() + smooth)


def iou_score(pred, target, smooth=1e-5):
    pred = torch.sigmoid(pred)
    pred = (pred > 0.5).float()
    pred = pred.contiguous().view(-1)
    target = target.contiguous().view(-1)
    intersection = (pred * target).sum()
    union = pred.sum() + target.sum() - intersection
    return (intersection + smooth) / (union + smooth)


def validate(model, loader, device):
    """Validate using the final output (L4) only."""
    model.eval()
    dices, ious = [], []
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)
            with autocast("cuda"):
                outputs = model(images)
            preds = outputs[-1]  # L4 (final output)
            dices.append(dice_score(preds, labels).item())
            ious.append(iou_score(preds, labels).item())
    model.train()
    return np.mean(dices), np.mean(ious)


def train():
    set_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    train_ds = ISIC2018Dataset("train", augment=True)
    test_ds = ISIC2018Dataset("test", augment=False)

    train_loader = DataLoader(train_ds, BATCH_SIZE, shuffle=True,
                              num_workers=NUM_WORKERS, pin_memory=True, drop_last=True)
    test_loader = DataLoader(test_ds, BATCH_SIZE, shuffle=False,
                            num_workers=NUM_WORKERS, pin_memory=True)

    model = NestedUNetV2(in_ch=3, out_ch=1).to(device)
    print(f"Params: {sum(p.numel() for p in model.parameters()) / 1e6:.2f} M")

    criterion = ComboLoss()
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS)
    scaler = GradScaler("cuda") if USE_AMP else None

    best_dice = 0.0
    best_path = os.path.join(SAVE_DIR, "best.pth")

    for epoch in range(NUM_EPOCHS):
        model.train()
        epoch_loss = 0.0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS}")

        for images, labels in pbar:
            images = images.to(device)
            labels = labels.to(device)

            if USE_AMP:
                with autocast("cuda"):
                    outputs = model(images)
                    loss = sum(
                        w * criterion(out, labels)
                        for w, out in zip(DS_WEIGHTS, outputs)
                    )
                optimizer.zero_grad()
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                outputs = model(images)
                loss = sum(
                    w * criterion(out, labels)
                    for w, out in zip(DS_WEIGHTS, outputs)
                )
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            epoch_loss += loss.item()
            pbar.set_postfix(loss=f"{loss.item():.4f}")

        scheduler.step()

        test_dice, test_iou = validate(model, test_loader, device)
        avg_loss = epoch_loss / len(train_loader)
        print(f"  Train Loss: {avg_loss:.4f} | Val DSC: {test_dice:.4f} | Val IoU: {test_iou:.4f}")

        if test_dice > best_dice:
            best_dice = test_dice
            torch.save(model.state_dict(), best_path)
            print(f"  Saved best model (DSC={best_dice:.4f})")

    print(f"Training done. Best Val DSC: {best_dice:.4f}")


if __name__ == "__main__":
    train()
