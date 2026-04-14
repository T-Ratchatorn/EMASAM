import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
import numpy as np
from .cutout import Cutout

import torchvision.transforms.functional as F

class SquarePad:
    def __call__(self, image):
        # Calculate padding
        w, h = image.size
        max_wh = np.max([w, h])
        hp = int((max_wh - w) / 2)
        vp = int((max_wh - h) / 2)
        padding = (hp, vp, hp, vp)

        # Apply padding
        image = F.pad(image, padding, 0, 'constant')
        return image


class imagenet_1k:
    def __init__(self, batch_size, threads, is_DDP=False, tsubame_id=None):
        self.tsubame_id = tsubame_id

        mean =  [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]

        train_transform = transforms.Compose([
            transforms.RandomResizedCrop(224, scale=(0.08, 1.0), ratio=(0.75, 1.33)), #crop with scale/ratio, then resize to 224x224
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        test_transform = transforms.Compose([
            transforms.Resize(256),                  # Resize the shorter side to 256 pixels
            transforms.CenterCrop(224),              # Center crop the image to 224x224 pixels
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        self.train_set = torchvision.datasets.ImageNet(root=f'/local/{self.tsubame_id}/ImageNet-1k', split="train", transform=train_transform)
        self.test_set = torchvision.datasets.ImageNet(root=f'/local/{self.tsubame_id}/ImageNet-1k', split="val", transform=test_transform)

        if is_DDP:
            self.train = DataLoader(self.train_set, batch_size=batch_size, shuffle=False, num_workers=threads, pin_memory=True, sampler=DistributedSampler(self.train_set))
            self.test = DataLoader(self.test_set, batch_size=batch_size, shuffle=False, num_workers=threads, pin_memory=True, sampler=DistributedSampler(self.test_set, shuffle=False, drop_last=True))
        else:
            self.train = DataLoader(self.train_set, batch_size=batch_size, shuffle=True, num_workers=threads)
            self.test = DataLoader(self.test_set, batch_size=batch_size, shuffle=False, num_workers=threads)
