import os
import torch
import random
import torchvision.transforms.functional as TF

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
PREPROCESSED = os.path.join(os.path.dirname(__file__), "preprocessed")


class ISIC2018Dataset(torch.utils.data.Dataset):
    def __init__(self, split="train", augment=False):
        images = torch.load(os.path.join(PREPROCESSED, f"{split}_images.pt"),
                            map_location="cpu", weights_only=True).float()
        labels = torch.load(os.path.join(PREPROCESSED, f"{split}_labels.pt"),
                            map_location="cpu", weights_only=True).float()
        self.images = images
        self.labels = labels
        self.augment = augment

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]

        if self.augment:
            if random.random() < 0.5:
                image = TF.hflip(image); label = TF.hflip(label)
            if random.random() < 0.5:
                image = TF.vflip(image); label = TF.vflip(label)
            angle = random.uniform(-15, 15)
            image = TF.rotate(image, angle); label = TF.rotate(label, angle)
            if random.random() < 0.3:
                image = TF.adjust_brightness(image, random.uniform(0.8, 1.2))
            if random.random() < 0.3:
                image = TF.adjust_contrast(image, random.uniform(0.8, 1.2))

        image = TF.normalize(image, IMAGENET_MEAN, IMAGENET_STD)
        return image, label
