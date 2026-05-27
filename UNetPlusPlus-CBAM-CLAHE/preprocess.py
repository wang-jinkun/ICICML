"""预计算 CLAHE 增强图像，保存到本地 preprocessed/ 目录。"""
import os, torch, numpy as np
from tqdm import tqdm
from skimage.color import rgb2lab, lab2rgb
from skimage.exposure import equalize_adapthist

INPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "preprocessed")
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "preprocessed")
os.makedirs(OUTPUT_DIR, exist_ok=True)


def apply_clahe(img_np):
    lab = rgb2lab(img_np)
    l_ch = lab[:, :, 0] / 100.0
    l_ch = equalize_adapthist(l_ch, kernel_size=None, clip_limit=0.02)
    lab[:, :, 0] = l_ch * 100.0
    return lab2rgb(lab)


def process_split(split_name):
    img_path = os.path.join(INPUT_DIR, f"{split_name}_images.pt")
    lbl_path = os.path.join(INPUT_DIR, f"{split_name}_labels.pt")
    images = torch.load(img_path, map_location="cpu", weights_only=True).float()
    labels = torch.load(lbl_path, map_location="cpu", weights_only=True)

    print(f"{split_name}: {len(images)} images")
    out_images = images.clone()
    for i in tqdm(range(len(images)), desc=f"CLAHE {split_name}"):
        img_np = images[i].permute(1, 2, 0).numpy()
        enhanced = apply_clahe(img_np).astype(np.float32)
        out_images[i] = torch.from_numpy(enhanced).permute(2, 0, 1)

    torch.save(out_images.half(), os.path.join(OUTPUT_DIR, f"{split_name}_images.pt"))
    torch.save(labels, os.path.join(OUTPUT_DIR, f"{split_name}_labels.pt"))
    mb = out_images.element_size() * out_images.numel() / 1024**2
    print(f"  Saved ({out_images.shape}, {mb:.1f} MB)")


if __name__ == "__main__":
    for split in ["train", "val", "test"]:
        process_split(split)
    print("Done.")
