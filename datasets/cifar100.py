import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from .cutout import Cutout

class cifar100:
    def __init__(self, batch_size, threads, is_DDP=False, tsubame_id=None):
        self.tsubame_id = tsubame_id
        mean, std = self._get_statistics()

        train_transform = transforms.Compose([
            torchvision.transforms.RandomCrop(size=(32, 32), padding=4),
            torchvision.transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
            Cutout()
        ])

        test_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])

        if self.tsubame_id:
            self.train_set = torchvision.datasets.CIFAR100(root=f'/local/{self.tsubame_id}/data_cifar100', train=True, download=True, transform=train_transform)
            self.test_set = torchvision.datasets.CIFAR100(root=f'/local/{self.tsubame_id}/data_cifar100', train=False, download=True, transform=test_transform)
        else:
            self.train_set = torchvision.datasets.CIFAR100(root='./data/data_cifar100', train=True, download=True, transform=train_transform)
            self.test_set = torchvision.datasets.CIFAR100(root='./data/data_cifar100', train=False, download=True, transform=test_transform)

        if is_DDP:
            self.train = DataLoader(self.train_set, batch_size=batch_size, shuffle=False, num_workers=threads, pin_memory=True, sampler=DistributedSampler(self.train_set))
            self.test = DataLoader(self.test_set, batch_size=batch_size, shuffle=False, num_workers=threads, pin_memory=True, sampler=DistributedSampler(self.test_set, shuffle=False, drop_last=True))
        else:
            self.train = DataLoader(self.train_set, batch_size=batch_size, shuffle=True, num_workers=threads)
            self.test = DataLoader(self.test_set, batch_size=batch_size, shuffle=False, num_workers=threads)

    def _get_statistics(self):
        if self.tsubame_id:
            train_set = torchvision.datasets.CIFAR100(root=f'/local/{self.tsubame_id}/data_cifar100', train=True, download=True, transform=transforms.ToTensor())
        else:
            train_set = torchvision.datasets.CIFAR100(root='./data/data_cifar100', train=True, download=True, transform=transforms.ToTensor())

        data = torch.cat([d[0] for d in DataLoader(train_set)])
        return data.mean(dim=[0, 2, 3]), data.std(dim=[0, 2, 3])