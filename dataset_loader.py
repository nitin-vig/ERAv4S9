"""
Dataset loader module for multiple ImageNet variants
"""

import os
import torch
import torchvision
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np
from PIL import Image
import zipfile
import requests
from config import Config

class TinyImageNetDataset(Dataset):
    """Custom dataset class for Tiny ImageNet"""
    
    def __init__(self, root_dir, split='train', transform=None):
        self.root_dir = root_dir
        self.split = split
        self.transform = transform
        
        # Set up paths
        if split == 'train':
            self.data_dir = os.path.join(root_dir, 'train')
        else:
            self.data_dir = os.path.join(root_dir, 'val')
        
        # Load class names
        self.classes = self._load_class_names()
        self.class_to_idx = {cls_name: idx for idx, cls_name in enumerate(self.classes)}
        
        # Load all image paths and labels
        self.samples = []
        if split == 'train':
            for class_name in self.classes:
                class_dir = os.path.join(self.data_dir, class_name, 'images')
                if os.path.exists(class_dir):
                    for img_name in os.listdir(class_dir):
                        if img_name.lower().endswith(('.jpg', '.jpeg', '.png')):
                            img_path = os.path.join(class_dir, img_name)
                            label = self.class_to_idx[class_name]
                            self.samples.append((img_path, label))
        else:
            # For validation, load from val_annotations.txt
            val_annotations = os.path.join(root_dir, 'val', 'val_annotations.txt')
            if os.path.exists(val_annotations):
                with open(val_annotations, 'r') as f:
                    for line in f:
                        parts = line.strip().split('\t')
                        img_name = parts[0]
                        class_name = parts[1]
                        img_path = os.path.join(self.data_dir, 'images', img_name)
                        if os.path.exists(img_path) and class_name in self.class_to_idx:
                            label = self.class_to_idx[class_name]
                            self.samples.append((img_path, label))
    
    def _load_class_names(self):
        """Load Tiny ImageNet class names"""
        words_file = os.path.join(self.root_dir, 'words.txt')
        if os.path.exists(words_file):
            with open(words_file, 'r') as f:
                classes = [line.strip().split('\t')[0] for line in f]
            return sorted(classes)
        else:
            # Fallback: use directory names
            train_dir = os.path.join(self.root_dir, 'train')
            if os.path.exists(train_dir):
                return sorted(os.listdir(train_dir))
            return []
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        
        # Load image
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            if isinstance(self.transform, A.Compose):
                # Albumentations transform
                image = np.array(image)
                image = self.transform(image=image)["image"]
            else:
                # Torchvision transform
                image = self.transform(image)
        
        return image, label

class ImageNetteDataset(Dataset):
    """Custom dataset class for ImageNette"""
    
    def __init__(self, root_dir, split='train', transform=None):
        self.root_dir = root_dir
        self.split = split
        self.transform = transform
        
        # Set up paths
        self.data_dir = os.path.join(root_dir, split)
        
        # ImageNette has 10 classes
        self.classes = [
            'n01440764', 'n02102040', 'n02979186', 'n03000684', 'n03028079',
            'n03394916', 'n03417042', 'n03425413', 'n03445777', 'n03888257'
        ]
        self.class_to_idx = {cls_name: idx for idx, cls_name in enumerate(self.classes)}
        
        # Load all image paths and labels
        self.samples = []
        for class_name in self.classes:
            class_dir = os.path.join(self.data_dir, class_name)
            if os.path.exists(class_dir):
                for img_name in os.listdir(class_dir):
                    if img_name.lower().endswith(('.jpg', '.jpeg', '.png')):
                        img_path = os.path.join(class_dir, img_name)
                        label = self.class_to_idx[class_name]
                        self.samples.append((img_path, label))
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        
        # Load image
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            if isinstance(self.transform, A.Compose):
                # Albumentations transform
                image = np.array(image)
                image = self.transform(image=image)["image"]
            else:
                # Torchvision transform
                image = self.transform(image)
        
        return image, label

class ImageNetDataset(Dataset):
    """Custom dataset class for full ImageNet"""
    
    def __init__(self, root_dir, split='train', transform=None):
        self.root_dir = root_dir
        self.split = split
        self.transform = transform
        
        # Set up paths
        self.data_dir = os.path.join(root_dir, split)
        
        # Load class names from synsets
        self.classes = self._load_class_names()
        self.class_to_idx = {cls_name: idx for idx, cls_name in enumerate(self.classes)}
        
        # Load all image paths and labels
        self.samples = []
        for class_name in self.classes:
            class_dir = os.path.join(self.data_dir, class_name)
            if os.path.exists(class_dir):
                for img_name in os.listdir(class_dir):
                    if img_name.lower().endswith(('.jpg', '.jpeg', '.png')):
                        img_path = os.path.join(class_dir, img_name)
                        label = self.class_to_idx[class_name]
                        self.samples.append((img_path, label))
    
    def _load_class_names(self):
        """Load ImageNet class names"""
        # For full ImageNet, we'll use the standard 1000 classes
        # In practice, you would load from synsets.txt or similar
        return [f"class_{i:03d}" for i in range(1000)]
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        
        # Load image
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            if isinstance(self.transform, A.Compose):
                # Albumentations transform
                image = np.array(image)
                image = self.transform(image=image)["image"]
            else:
                # Torchvision transform
                image = self.transform(image)
        
        return image, label

class ImageNetMiniDataset(Dataset):
    """Custom dataset class for ImageNet Mini"""
    
    def __init__(self, root_dir, split='train', transform=None):
        self.root_dir = root_dir
        self.split = split
        self.transform = transform
        
        # Set up paths
        self.data_dir = os.path.join(root_dir, split)
        self.classes = sorted(os.listdir(self.data_dir))
        self.class_to_idx = {cls_name: idx for idx, cls_name in enumerate(self.classes)}
        
        # Load all image paths and labels
        self.samples = []
        for class_name in self.classes:
            class_dir = os.path.join(self.data_dir, class_name)
            for img_name in os.listdir(class_dir):
                if img_name.lower().endswith(('.jpg', '.jpeg', '.png')):
                    img_path = os.path.join(class_dir, img_name)
                    label = self.class_to_idx[class_name]
                    self.samples.append((img_path, label))
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        
        # Load image
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            if isinstance(self.transform, A.Compose):
                # Albumentations transform
                image = np.array(image)
                image = self.transform(image=image)["image"]
            else:
                # Torchvision transform
                image = self.transform(image)
        
        return image, label
def download_imagenet():
    """Download full ImageNet dataset"""
    print("ImageNet dataset download instructions:")
    print("1. Register at https://image-net.org/")
    print("2. Download ILSVRC2012_img_train.tar and ILSVRC2012_img_val.tar")
    print("3. Extract to ./data/imagenet/")
    print("4. Ensure folder structure:")
    print("   imagenet/")
    print("   ├── train/")
    print("   │   ├── n01440764/")
    print("   │   ├── n01443537/")
    print("   │   └── ...")
    print("   └── val/")
    print("       ├── n01440764/")
    print("       ├── n01443537/")
    print("       └── ...")
    
    imagenet_path = os.path.join(Config.DATA_ROOT, "imagenet")
    os.makedirs(imagenet_path, exist_ok=True)
    
    return imagenet_path

def download_tiny_imagenet():
    """Download Tiny ImageNet dataset"""
    print("Tiny ImageNet dataset download instructions:")
    print("1. Download from: http://cs231n.stanford.edu/tiny-imagenet-200.zip")
    print("2. Extract to ./data/tiny-imagenet-200/")
    print("3. Ensure folder structure:")
    print("   tiny-imagenet-200/")
    print("   ├── train/")
    print("   │   ├── n01440764/")
    print("   │   └── ...")
    print("   ├── val/")
    print("   │   ├── images/")
    print("   │   └── val_annotations.txt")
    print("   └── words.txt")
    
    tiny_imagenet_path = os.path.join(Config.DATA_ROOT, "tiny-imagenet-200")
    os.makedirs(tiny_imagenet_path, exist_ok=True)
    
    return tiny_imagenet_path

def download_imagenette():
    """Download ImageNette dataset"""
    print("ImageNette dataset download instructions:")
    print("1. Download from: https://s3.amazonaws.com/fast-ai-imageclas/imagenette2.tgz")
    print("2. Extract to ./data/imagenette2/")
    print("3. Ensure folder structure:")
    print("   imagenette2/")
    print("   ├── train/")
    print("   │   ├── n01440764/")
    print("   │   └── ...")
    print("   └── val/")
    print("       ├── n01440764/")
    print("       └── ...")
    
    imagenette_path = os.path.join(Config.DATA_ROOT, "imagenette2")
    os.makedirs(imagenette_path, exist_ok=True)
    
    return imagenette_path

def download_imagenet_mini():
    """Download ImageNet Mini dataset"""
    print("Downloading ImageNet Mini dataset...")
    
    # Create data directory
    os.makedirs(Config.DATA_ROOT, exist_ok=True)
    
    # Note: In a real scenario, you would need to download from Kaggle
    # For now, we'll create a placeholder structure
    imagenet_path = os.path.join(Config.DATA_ROOT, "imagenet-mini")
    os.makedirs(imagenet_path, exist_ok=True)
    
    print(f"ImageNet Mini dataset will be downloaded to: {imagenet_path}")
    print("Please manually download the dataset from Kaggle and extract it to this location.")
    
    return imagenet_path

def get_albumentations_transforms(dataset_config, is_training=True):
    """Get Albumentations transforms for data augmentation"""
    
    if is_training:
        transform = A.Compose([
            A.Resize(dataset_config["image_size"], dataset_config["image_size"]),
            A.PadIfNeeded(min_height=dataset_config["image_size"] + 32, 
                         min_width=dataset_config["image_size"] + 32, 
                         border_mode=0, p=1.0),
            A.RandomCrop(dataset_config["image_size"], dataset_config["image_size"], p=1.0),
            A.HorizontalFlip(p=0.5),
            A.OneOf([
                A.CoarseDropout(
                    max_holes=1,
                    max_height=16,
                    max_width=16,
                    min_height=8,
                    min_width=8,
                    fill_value=tuple([int(x * 255) for x in dataset_config["mean"]]),
                    p=0.75
                ),
                A.GaussNoise(var_limit=(10.0, 50.0), p=0.25),
            ], p=0.5),
            A.Normalize(mean=dataset_config["mean"], std=dataset_config["std"]),
            ToTensorV2(),
        ])
    else:
        transform = A.Compose([
            A.Resize(dataset_config["image_size"], dataset_config["image_size"]),
            A.Normalize(mean=dataset_config["mean"], std=dataset_config["std"]),
            ToTensorV2(),
        ])
    
    return transform

def get_torchvision_transforms(dataset_config, is_training=True):
    """Get torchvision transforms for data augmentation"""
    
    if is_training:
        transform = transforms.Compose([
            transforms.Resize((dataset_config["image_size"], dataset_config["image_size"])),
            transforms.RandomCrop(dataset_config["image_size"], padding=4),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),
            transforms.Normalize(mean=dataset_config["mean"], std=dataset_config["std"])
        ])
    else:
        transform = transforms.Compose([
            transforms.Resize((dataset_config["image_size"], dataset_config["image_size"])),
            transforms.ToTensor(),
            transforms.Normalize(mean=dataset_config["mean"], std=dataset_config["std"])
        ])
    
    return transform


def get_tiny_imagenet_dataset():
    """Get Tiny ImageNet dataset"""
    dataset_config = Config.get_dataset_config()
    
    # Download dataset if not exists
    tiny_imagenet_path = os.path.join(Config.DATA_ROOT, "tiny-imagenet-200")
    if not os.path.exists(tiny_imagenet_path):
        download_tiny_imagenet()
    
    # Get transforms
    train_transform = get_torchvision_transforms(dataset_config, is_training=True)
    test_transform = get_torchvision_transforms(dataset_config, is_training=False)
    
    # Load datasets
    train_dataset = TinyImageNetDataset(
        tiny_imagenet_path, split='train', transform=train_transform
    )
    test_dataset = TinyImageNetDataset(
        tiny_imagenet_path, split='val', transform=test_transform
    )
    
    return train_dataset, test_dataset

def get_imagenette_dataset():
    """Get ImageNette dataset"""
    dataset_config = Config.get_dataset_config()
    
    # Download dataset if not exists
    imagenette_path = os.path.join(Config.DATA_ROOT, "imagenette2")
    if not os.path.exists(imagenette_path):
        download_imagenette()
    
    # Get transforms
    train_transform = get_torchvision_transforms(dataset_config, is_training=True)
    test_transform = get_torchvision_transforms(dataset_config, is_training=False)
    
    # Load datasets
    train_dataset = ImageNetteDataset(
        imagenette_path, split='train', transform=train_transform
    )
    test_dataset = ImageNetteDataset(
        imagenette_path, split='val', transform=test_transform
    )
    
    return train_dataset, test_dataset
def get_imagenet_dataset():
    """Get full ImageNet dataset"""
    dataset_config = Config.get_dataset_config()
    
    # Download dataset if not exists
    imagenet_path = os.path.join(Config.DATA_ROOT, "imagenet")
    if not os.path.exists(imagenet_path):
        download_imagenet()
    
    # Get transforms
    train_transform = get_torchvision_transforms(dataset_config, is_training=True)
    test_transform = get_torchvision_transforms(dataset_config, is_training=False)
    
    # Load datasets
    train_dataset = ImageNetDataset(
        imagenet_path, split='train', transform=train_transform
    )
    test_dataset = ImageNetDataset(
        imagenet_path, split='val', transform=test_transform
    )
    
    return train_dataset, test_dataset

def get_imagenet_mini_dataset():
    """Get ImageNet Mini dataset"""
    dataset_config = Config.get_dataset_config()
    
    # Download dataset if not exists
    imagenet_path = os.path.join(Config.DATA_ROOT, "imagenet-mini")
    if not os.path.exists(imagenet_path):
        download_imagenet_mini()
    
    # Get transforms
    train_transform = get_torchvision_transforms(dataset_config, is_training=True)
    test_transform = get_torchvision_transforms(dataset_config, is_training=False)
    
    # Load datasets
    train_dataset = ImageNetMiniDataset(
        imagenet_path, split='train', transform=train_transform
    )
    test_dataset = ImageNetMiniDataset(
        imagenet_path, split='val', transform=test_transform
    )
    
    return train_dataset, test_dataset

def get_data_loaders(dataset_name="imagenette"):
    """Get data loaders for the specified dataset"""
    
    # Set random seed for reproducibility
    torch.manual_seed(Config.SEED if hasattr(Config, 'SEED') else 1)
    
    # Get dataset
    if dataset_name == "imagenet":
        train_dataset, test_dataset = get_imagenet_dataset()
    elif dataset_name == "imagenet_mini":
        train_dataset, test_dataset = get_imagenet_mini_dataset()
    elif dataset_name == "tiny_imagenet":
        train_dataset, test_dataset = get_tiny_imagenet_dataset()
    elif dataset_name == "imagenette":
        train_dataset, test_dataset = get_imagenette_dataset()
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    
    # DataLoader arguments
    dataloader_args = {
        'shuffle': True,
        'batch_size': Config.BATCH_SIZE,
        'num_workers': Config.NUM_WORKERS,
        'pin_memory': Config.PIN_MEMORY
    }
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, **dataloader_args)
    test_loader = DataLoader(test_dataset, **dataloader_args)
    
    print(f"Dataset: {dataset_name}")
    print(f"Train samples: {len(train_dataset)}")
    print(f"Test samples: {len(test_dataset)}")
    print(f"Number of classes: {len(train_dataset.classes) if hasattr(train_dataset, 'classes') else 'Unknown'}")
    
    return train_loader, test_loader

def visualize_samples(data_loader, num_samples=12):
    """Visualize sample images from the dataset"""
    import matplotlib.pyplot as plt
    
    batch_data, batch_label = next(iter(data_loader))
    
    fig = plt.figure(figsize=(15, 10))
    
    for i in range(min(num_samples, len(batch_data))):
        plt.subplot(3, 4, i+1)
        plt.tight_layout()
        
        # Denormalize for visualization
        img = batch_data[i]
        if img.min() < 0:  # If normalized
            img = img * torch.tensor(Config.get_dataset_config()["std"]).view(3, 1, 1) + \
                  torch.tensor(Config.get_dataset_config()["mean"]).view(3, 1, 1)
            img = torch.clamp(img, 0, 1)
        
        plt.imshow(img.permute(1, 2, 0))
        plt.title(f"Label: {batch_label[i].item()}")
        plt.xticks([])
        plt.yticks([])
    
    plt.show()
