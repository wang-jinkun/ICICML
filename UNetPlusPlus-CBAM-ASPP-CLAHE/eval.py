import os
import random
import numpy as np
import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torch.amp import autocast
from tqdm import tqdm

from config import *
from model import NestedUNetV2
from dataset import ISIC2018Dataset


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def dice_score(pred, target, smooth=1e-5):
    pred = torch.sigmoid(pred)
    pred = (pred > 0.5).float().contiguous().view(-1)
    target = target.contiguous().view(-1)
    intersection = (pred * target).sum()
    return ((2. * intersection + smooth) / (pred.sum() + target.sum() + smooth)).item()


def iou_score(pred, target, smooth=1e-5):
    pred = torch.sigmoid(pred)
    pred = (pred > 0.5).float().contiguous().view(-1)
    target = target.contiguous().view(-1)
    intersection = (pred * target).sum()
    union = pred.sum() + target.sum() - intersection
    return ((intersection + smooth) / (union + smooth)).item()


def visualize(model, loader, device, save_dir, num_samples=5):
    os.makedirs(save_dir, exist_ok=True)
    model.eval()

    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

    with torch.no_grad():
        for idx, (images, labels) in enumerate(loader):
            if idx >= num_samples:
                break
            img = images[0:1].to(device)
            label = labels[0:1].to(device)
            with autocast("cuda"):
                outputs = model(img)
            pred = torch.sigmoid(outputs[-1])  # L4

            img_cpu = img.cpu()[0] * std + mean
            img_cpu = img_cpu.permute(1, 2, 0).clamp(0, 1).numpy()
            pred_cpu = (pred.cpu()[0, 0] > 0.5).float().numpy()
            label_cpu = label.cpu()[0, 0].numpy()

            fig, axes = plt.subplots(1, 3, figsize=(12, 4))
            axes[0].imshow(img_cpu)
            axes[0].set_title("Image")
            axes[0].axis("off")
            axes[1].imshow(label_cpu, cmap="gray")
            axes[1].set_title("Ground Truth")
            axes[1].axis("off")
            axes[2].imshow(pred_cpu, cmap="gray")
            axes[2].set_title("Prediction")
            axes[2].axis("off")
            plt.tight_layout()
            plt.savefig(os.path.join(save_dir, f"sample_{idx+1}.png"), dpi=150)
            plt.close()


def evaluate():
    set_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    test_ds = ISIC2018Dataset("test", augment=False)
    test_loader = DataLoader(test_ds, BATCH_SIZE, shuffle=False,
                             num_workers=NUM_WORKERS, pin_memory=True)

    model = NestedUNetV2(in_ch=3, out_ch=1).to(device)
    ckpt = os.path.join(SAVE_DIR, "best.pth")
    model.load_state_dict(torch.load(ckpt, map_location=device))
    model.eval()

    dices, ious = [], []
    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc="Evaluating"):
            images = images.to(device)
            labels = labels.to(device)
            with autocast("cuda"):
                outputs = model(images)
            preds = outputs[-1] if isinstance(outputs, (list, tuple)) else outputs
            dices.append(dice_score(preds, labels))
            ious.append(iou_score(preds, labels))

    mean_dice = np.mean(dices)
    mean_iou = np.mean(ious)
    print(f"
Test Results: DSC = {mean_dice:.4f}, IoU = {mean_iou:.4f}")

    vis_loader = DataLoader(test_ds, BATCH_SIZE, shuffle=True)
    vis_dir = os.path.join(os.path.dirname(__file__), "visualizations")
    visualize(model, vis_loader, device, vis_dir)
    print(f"Visualizations saved to {vis_dir}")

    return mean_dice, mean_iou
