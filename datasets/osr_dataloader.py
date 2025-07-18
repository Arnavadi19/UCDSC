import os
import torch
import numpy as np
from PIL import Image
from torchvision import transforms
from torchvision.datasets import ImageFolder
import torch.utils.data as data
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
import sys
import os

# medmnist_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'datasets/MedMNIST')
# sys.path.insert(0, medmnist_path)
# from medmnist.dataset import BloodMNIST, OCTMNIST, DermaMNIST, TissueMNIST
from medmnist import BloodMNIST, OCTMNIST, DermaMNIST, TissueMNIST

class Random300K_Images(torch.utils.data.Dataset):
    def __init__(self, file_path, transform=None, extendable=0):
        self.transform = transform
        self.extendable = extendable
        self.offset = 0
        
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
        
        self.data = np.load(file_path)
        if extendable > 0:
            self.data = np.repeat(self.data, extendable + 1, axis=0)
            
        if transform is None:
            self.transform = transforms.Compose([
                transforms.Resize((32,32)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
            ])
    
    def __getitem__(self, index):
        index = (index + self.offset) % len(self)
        img = self.data[index]
        img = Image.fromarray(img)
        if self.transform is not None:
            img = self.transform(img)
        return img, 0
    
    def __len__(self):
        return len(self.data)

class BloodMNIST_OSR(object):
    """Open Set Recognition wrapper for BloodMNIST dataset"""
    def __init__(self, known, dataroot='./data', use_gpu=True, num_workers=4, batch_size=128):
        self.num_classes = len(known)
        if isinstance(known, dict):
            self.known = known['known']
        else:
            self.known = known
            
        self.unknown = list(set(range(0, 8)) - set(self.known))

        print('BloodMNIST_OSR Known classes:', self.known)
        print('BloodMNIST_OSR Unknown classes:', self.unknown)

        # Define transforms
        transform = transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomAffine(degrees=0, translate=(0.05, 0.05), scale=(0.95, 1.05)),
            transforms.RandomApply([transforms.RandomRotation(15)], p=0.5),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])

        test_transform = transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])

        # Load dataset
        train_dataset = BloodMNIST(split='train', transform=transform, download=True, root=dataroot)
        test_dataset = BloodMNIST(split='test', transform=test_transform, download=True, root=dataroot)

        train_labels = train_dataset.labels.squeeze()
        print("BloodMNIST_OSR Training set class distribution:", np.bincount(train_labels))

        # Filter dataset
        train_mask = np.isin(train_dataset.labels.squeeze(), self.known)
        known_test_mask = np.isin(test_dataset.labels.squeeze(), self.known)
        unknown_test_mask = np.isin(test_dataset.labels.squeeze(), self.unknown)

        # Create data loaders
        self.train_loader = DataLoader(
            FilteredDataset(train_dataset, train_mask, self.known), 
            batch_size=batch_size, shuffle=True,
            num_workers=num_workers, pin_memory=use_gpu
        )

        self.test_loader = DataLoader(
            FilteredDataset(test_dataset, known_test_mask, self.known),
            batch_size=batch_size, shuffle=False,
            num_workers=num_workers, pin_memory=use_gpu
        )

        self.out_loader = DataLoader(
            FilteredDataset(test_dataset, unknown_test_mask, self.unknown),
            batch_size=batch_size, shuffle=False,
            num_workers=num_workers, pin_memory=use_gpu
        )

        print(f'BloodMNIST_OSR Train samples: {len(self.train_loader.dataset)}')
        print(f'BloodMNIST_OSR Test samples (known): {len(self.test_loader.dataset)}')
        print(f'BloodMNIST_OSR Test samples (unknown): {len(self.out_loader.dataset)}')

        
class FilteredDataset(Dataset):
    """Helper class to filter and remap labels"""
    def __init__(self, dataset, mask, known_classes):
        self.dataset = dataset
        self.indices = np.where(mask)[0]
        self.target_map = {label: idx for idx, label in enumerate(known_classes)}
        
    def __getitem__(self, index):
        img, label = self.dataset[self.indices[index]]
        return img, self.target_map[int(label)]

    def __len__(self):
        return len(self.indices)
 

class OCTMnist_OSR(object):
    def __init__(self, known, unknown, dataroot='./data', use_gpu=True, num_workers=4, batch_size=128):
        self.known = known
        self.unknown = unknown  
        self.num_classes = len(known)

        print('OCTMnist_OSR Known classes:', self.known)
        print('OCTMnist_OSR Unknown classes:', self.unknown)

        transform = transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.Grayscale(3),
            transforms.RandomCrop(32, padding=4),  
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomAffine(degrees=0, translate=(0.05, 0.05), scale=(0.95, 1.05)), 
            transforms.RandomApply([transforms.RandomRotation(15)], p=0.5),     
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])

        test_transform = transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.Grayscale(3),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])

        train_dataset = OCTMNIST(split='train', transform=transform, 
                                download=True, root=dataroot)
        test_dataset = OCTMNIST(split='test', transform=test_transform,
                               download=True, root=dataroot)

        train_labels = train_dataset.labels.squeeze()
        print("OCTMnist_OSR Training set class distribution:", np.bincount(train_labels))

        train_mask = np.isin(train_dataset.labels.squeeze(), self.known)
        known_test_mask = np.isin(test_dataset.labels.squeeze(), self.known)
        unknown_test_mask = np.isin(test_dataset.labels.squeeze(), self.unknown)

        self.train_loader = DataLoader(
            FilteredDataset(train_dataset, train_mask, self.known),
            batch_size=batch_size, shuffle=True,
            num_workers=num_workers, pin_memory=use_gpu
        )

        self.test_loader = DataLoader(
            FilteredDataset(test_dataset, known_test_mask, self.known),
            batch_size=batch_size, shuffle=False,
            num_workers=num_workers, pin_memory=use_gpu
        )

        self.out_loader = DataLoader(
            FilteredDataset(test_dataset, unknown_test_mask, self.unknown),
            batch_size=batch_size, shuffle=False,
            num_workers=num_workers, pin_memory=use_gpu
        )

        print(f'OCTMnist_OSR Train samples: {len(self.train_loader.dataset)}')
        print(f'OCTMnist_OSR Test samples (known): {len(self.test_loader.dataset)}')
        print(f'OCTMnist_OSR Test samples (unknown): {len(self.out_loader.dataset)}')


class DermaMNIST_OSR(object):
    """Open Set Recognition wrapper for DermaMNIST dataset"""
    def __init__(self, known,unknown = None, dataroot='./data', use_gpu=True, num_workers=4, batch_size=128):
        self.num_classes = len(known)
        if isinstance(known, dict):
            self.known = known['known']
        else:
            self.known = known
            
        if unknown is not None:
            self.unknown = unknown
        else:
            self.unknown = list(set(range(0, 7)) - set(self.known))


        print('DermaMNIST_OSR Known classes:', self.known)
        print('DermaMNIST_OSR Unknown classes:', self.unknown)

        transform = transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomAffine(degrees=0, translate=(0.05, 0.05), scale=(0.95, 1.05)), 
            transforms.RandomApply([transforms.RandomRotation(15)], p=0.5),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])

        test_transform = transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])

        # Load dataset
        train_dataset = DermaMNIST(split='train', transform=transform, download=True, root=dataroot)
        test_dataset = DermaMNIST(split='test', transform=test_transform, download=True, root=dataroot)

        train_labels = train_dataset.labels.squeeze()
        print("DermaMNIST_OSR Training set class distribution:", np.bincount(train_labels))

        # Filter dataset
        train_mask = np.isin(train_dataset.labels.squeeze(), self.known)
        known_test_mask = np.isin(test_dataset.labels.squeeze(), self.known)
        unknown_test_mask = np.isin(test_dataset.labels.squeeze(), self.unknown)

        # Create data loaders
        self.train_loader = DataLoader(
            FilteredDataset(train_dataset, train_mask, self.known), 
            batch_size=batch_size, shuffle=True,
            num_workers=num_workers, pin_memory=use_gpu
        )

        self.test_loader = DataLoader(
            FilteredDataset(test_dataset, known_test_mask, self.known),
            batch_size=batch_size, shuffle=False,
            num_workers=num_workers, pin_memory=use_gpu
        )

        self.out_loader = DataLoader(
            FilteredDataset(test_dataset, unknown_test_mask, self.unknown),
            batch_size=batch_size, shuffle=False,
            num_workers=num_workers, pin_memory=use_gpu
        )

        print(f'DermaMNIST_OSR Train samples: {len(self.train_loader.dataset)}')
        print(f'DermaMNIST_OSR Test samples (known): {len(self.test_loader.dataset)}')
        print(f'DermaMNIST_OSR Test samples (unknown): {len(self.out_loader.dataset)}')


class TissueMNIST_OSR(object):
    """Open Set Recognition wrapper for TissueMNIST dataset"""
    def __init__(self, known,unknown = None, dataroot='./data', use_gpu=True, num_workers=4, batch_size=128):
        self.num_classes = len(known)
        if isinstance(known, dict):
            self.known = known['known']
        else:
            self.known = known
            
        if unknown is not None:
            self.unknown = unknown
        else:
            self.unknown = list(set(range(0, 8)) - set(self.known))


        print('TissueMNIST_OSR Known classes:', self.known)
        print('TissueMNIST_OSR Unknown classes:', self.unknown)

        transform = transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.Grayscale(3),
            transforms.RandomCrop(32, padding=4),  
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomAffine(degrees=0, translate=(0.05, 0.05), scale=(0.95, 1.05)),
            transforms.RandomApply([transforms.RandomRotation(15)], p=0.5),     
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])


        test_transform = transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.Grayscale(3),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])

        train_dataset = TissueMNIST(split='train', transform=transform, download=True, root=dataroot)
        test_dataset = TissueMNIST(split='test', transform=test_transform, download=True, root=dataroot)

        train_labels = train_dataset.labels.squeeze()
        print("TissueMNIST_OSR Training set class distribution:", np.bincount(train_labels))

        # Filter datasets
        train_mask = np.isin(train_dataset.labels.squeeze(), self.known)
        known_test_mask = np.isin(test_dataset.labels.squeeze(), self.known)
        unknown_test_mask = np.isin(test_dataset.labels.squeeze(), self.unknown)

        self.train_loader = DataLoader(
            FilteredDataset(train_dataset, train_mask, self.known), 
            batch_size=batch_size, shuffle=True,
            num_workers=num_workers, pin_memory=use_gpu
        )

        self.test_loader = DataLoader(
            FilteredDataset(test_dataset, known_test_mask, self.known),
            batch_size=batch_size, shuffle=False,
            num_workers=num_workers, pin_memory=use_gpu
        )

        self.out_loader = DataLoader(
            FilteredDataset(test_dataset, unknown_test_mask, self.unknown),
            batch_size=batch_size, shuffle=False,
            num_workers=num_workers, pin_memory=use_gpu
        )

        print(f'TissueMNIST_OSR Train samples: {len(self.train_loader.dataset)}')
        print(f'TissueMNIST_OSR Test samples (known): {len(self.test_loader.dataset)}')
        print(f'TissueMNIST_OSR Test samples (unknown): {len(self.out_loader.dataset)}')
