import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from .cutout import Cutout

class flowers102:
    def __init__(self, batch_size, threads, is_DDP=False, tsubame_id=None):
        self.tsubame_id = tsubame_id
        mean, std = self._get_statistics()

        train_transform = transforms.Compose([
            transforms.RandomResizedCrop(224, scale=(0.08, 1.0), ratio=(0.75, 1.33)), #crop with scale/ratio, then resize to 224x224
            torchvision.transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])

        test_transform = transforms.Compose([
            transforms.Resize(256),                  # Resize the shorter side to 256 pixels
            transforms.CenterCrop(224),              # Center crop the image to 224x224 pixels
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])

        if self.tsubame_id:
            self.train_set = torchvision.datasets.Flowers102(root=f'/local/{self.tsubame_id}/data_flowers102', split='train', download=True, transform=train_transform)
            self.test_set  = torchvision.datasets.Flowers102(root=f'/local/{self.tsubame_id}/data_flowers102', split='test',  download=True, transform=test_transform)
        else:
            self.train_set = torchvision.datasets.Flowers102(root='./data/data_flowers102', split='train', download=True, transform=train_transform)
            self.test_set  = torchvision.datasets.Flowers102(root='./data/data_flowers102', split='test',  download=True, transform=test_transform)

        if is_DDP:
            self.train = DataLoader(self.train_set, batch_size=batch_size, shuffle=False, num_workers=threads, pin_memory=True, sampler=DistributedSampler(self.train_set))
            self.test  = DataLoader(self.test_set,  batch_size=batch_size, shuffle=False, num_workers=threads, pin_memory=True, sampler=DistributedSampler(self.test_set, shuffle=False, drop_last=True))
        else:
            self.train = DataLoader(self.train_set, batch_size=batch_size, shuffle=True,  num_workers=threads)
            self.test  = DataLoader(self.test_set,  batch_size=batch_size, shuffle=False, num_workers=threads)

    def _get_statistics(self):
        stat_transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor()
        ])
        if self.tsubame_id:
            train_set = torchvision.datasets.Flowers102(root=f'/local/{self.tsubame_id}/data_flowers102', split='train', download=True, transform=stat_transform)
        else:
            train_set = torchvision.datasets.Flowers102(root='./data/data_flowers102', split='train', download=True, transform=stat_transform)

        loader = DataLoader(train_set, batch_size=64, shuffle=False, num_workers=4)
        data = torch.cat([imgs for imgs, _ in loader], dim=0)
        
        return data.mean(dim=[0, 2, 3]), data.std(dim=[0, 2, 3])