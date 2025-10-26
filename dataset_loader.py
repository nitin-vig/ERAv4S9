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
import tarfile
import subprocess
import sys
from tqdm import tqdm
from config import Config

def download_file_with_progress(url, filename):
    """Download file with progress bar"""
    print(f"Downloading {filename} from {url}...")
    
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()
        
        total_size = int(response.headers.get('content-length', 0))
        
        with open(filename, 'wb') as file, tqdm(
            desc=filename,
            total=total_size,
            unit='iB',
            unit_scale=True,
            unit_divisor=1024,
        ) as progress_bar:
            for chunk in response.iter_content(chunk_size=8192):
                size = file.write(chunk)
                progress_bar.update(size)
        
        print(f"✅ Downloaded {filename}")
        return True
        
    except Exception as e:
        print(f"❌ Error downloading {filename}: {e}")
        return False

def extract_tar_file(tar_path, extract_to):
    """Extract tar file"""
    print(f"Extracting {tar_path} to {extract_to}...")
    try:
        with tarfile.open(tar_path, 'r:gz') as tar:
            tar.extractall(extract_to)
        print(f"✅ Extracted {tar_path}")
        return True
    except Exception as e:
        print(f"❌ Error extracting {tar_path}: {e}")
        return False

def extract_zip_file(zip_path, extract_to):
    """Extract zip file"""
    print(f"Extracting {zip_path} to {extract_to}...")
    try:
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_to)
        print(f"✅ Extracted {zip_path}")
        return True
    except Exception as e:
        print(f"❌ Error extracting {zip_path}: {e}")
        return False

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
    """Download Tiny ImageNet dataset automatically"""
    print("🔄 Downloading Tiny ImageNet dataset...")
    
    # Create data directory
    os.makedirs(Config.DATA_ROOT, exist_ok=True)
    
    tiny_imagenet_path = os.path.join(Config.DATA_ROOT, "tiny-imagenet-200")
    
    # Check if already exists
    if os.path.exists(tiny_imagenet_path) and os.listdir(tiny_imagenet_path):
        print(f"✅ Tiny ImageNet dataset already exists at {tiny_imagenet_path}")
        return tiny_imagenet_path
    
    # Download URL
    url = "http://cs231n.stanford.edu/tiny-imagenet-200.zip"
    zip_filename = os.path.join(Config.DATA_ROOT, "tiny-imagenet-200.zip")
    
    # Download the file
    if not download_file_with_progress(url, zip_filename):
        print("❌ Failed to download Tiny ImageNet dataset")
        print("Manual download instructions:")
        print("1. Download from: http://cs231n.stanford.edu/tiny-imagenet-200.zip")
        print("2. Extract to ./data/tiny-imagenet-200/")
        return tiny_imagenet_path
    
    # Extract the file
    if not extract_zip_file(zip_filename, Config.DATA_ROOT):
        print("❌ Failed to extract Tiny ImageNet dataset")
        return tiny_imagenet_path
    
    # Clean up zip file
    try:
        os.remove(zip_filename)
        print(f"🗑️  Cleaned up {zip_filename}")
    except:
        pass
    
    print(f"✅ Tiny ImageNet dataset ready at {tiny_imagenet_path}")
    return tiny_imagenet_path

def download_imagenette():
    """Download ImageNette dataset automatically"""
    print("🔄 Downloading ImageNette dataset...")
    
    # Create data directory
    os.makedirs(Config.DATA_ROOT, exist_ok=True)
    
    imagenette_path = os.path.join(Config.DATA_ROOT, "imagenette2")
    
    # Check if already exists
    if os.path.exists(imagenette_path) and os.listdir(imagenette_path):
        print(f"✅ ImageNette dataset already exists at {imagenette_path}")
        return imagenette_path
    
    # Download URL
    url = "https://s3.amazonaws.com/fast-ai-imageclas/imagenette2.tgz"
    tar_filename = os.path.join(Config.DATA_ROOT, "imagenette2.tgz")
    
    # Download the file
    if not download_file_with_progress(url, tar_filename):
        print("❌ Failed to download ImageNette dataset")
        print("Manual download instructions:")
        print("1. Download from: https://s3.amazonaws.com/fast-ai-imageclas/imagenette2.tgz")
        print("2. Extract to ./data/imagenette2/")
        return imagenette_path
    
    # Extract the file
    if not extract_tar_file(tar_filename, Config.DATA_ROOT):
        print("❌ Failed to extract ImageNette dataset")
        return imagenette_path
    
    # Clean up tar file
    try:
        os.remove(tar_filename)
        print(f"🗑️  Cleaned up {tar_filename}")
    except:
        pass
    
    print(f"✅ ImageNette dataset ready at {imagenette_path}")
    return imagenette_path

def download_imagenet_mini():
    """Download ImageNet Mini dataset (requires Kaggle setup)"""
    print("🔄 Setting up ImageNet Mini dataset...")
    
    # Create data directory
    os.makedirs(Config.DATA_ROOT, exist_ok=True)
    
    imagenet_path = os.path.join(Config.DATA_ROOT, "imagenet-mini")
    
    # Check if already exists
    if os.path.exists(imagenet_path) and os.listdir(imagenet_path):
        print(f"✅ ImageNet Mini dataset already exists at {imagenet_path}")
        return imagenet_path
    
    print("📋 ImageNet Mini requires Kaggle authentication:")
    print("1. Install Kaggle API: pip install kaggle")
    print("2. Get API credentials from: https://www.kaggle.com/account")
    print("3. Place kaggle.json in ~/.kaggle/")
    print("4. Run: kaggle datasets download -d ifigotin/imagenetmini-1000")
    print("5. Extract to ./data/imagenet-mini/")
    print("\nAlternatively, use ImageNette or Tiny ImageNet for testing.")
    
    # Create placeholder directory
    os.makedirs(imagenet_path, exist_ok=True)
    
    return imagenet_path

def get_albumentations_transforms(dataset_config, is_training=True):
    """Get Albumentations transforms for data augmentation"""
    
    # Get mean and std from AUGMENTATION config
    norm_key = "train" if is_training else "val"
    mean = Config.AUGMENTATION[norm_key]["normalize"]["mean"]
    std = Config.AUGMENTATION[norm_key]["normalize"]["std"]
    
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
                    fill_value=tuple([int(x * 255) for x in mean]),
                    p=0.75
                ),
                A.GaussNoise(var_limit=(10.0, 50.0), p=0.25),
            ], p=0.5),
            A.Normalize(mean=mean, std=std),
            ToTensorV2(),
        ])
    else:
        transform = A.Compose([
            A.Resize(dataset_config["image_size"], dataset_config["image_size"]),
            A.Normalize(mean=mean, std=std),
            ToTensorV2(),
        ])
    
    return transform

def get_torchvision_transforms(dataset_config, is_training=True):
    """Get torchvision transforms for data augmentation"""
    
    # Get mean and std from AUGMENTATION config
    norm_key = "train" if is_training else "val"
    mean = Config.AUGMENTATION[norm_key]["normalize"]["mean"]
    std = Config.AUGMENTATION[norm_key]["normalize"]["std"]
    
    if is_training:
        transform = transforms.Compose([
            transforms.Resize((dataset_config["image_size"], dataset_config["image_size"])),
            transforms.RandomCrop(dataset_config["image_size"], padding=4),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)
        ])
    else:
        transform = transforms.Compose([
            transforms.Resize((dataset_config["image_size"], dataset_config["image_size"])),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)
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
    
    # Try to get dataset
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
    
    # Check if datasets are empty
    if len(train_dataset) == 0 or len(test_dataset) == 0:
        raise RuntimeError(f"Dataset '{dataset_name}' is empty. Please download the dataset first.")
    
    # # DataLoader arguments
    # dataloader_args = {
    #     'shuffle': True,
    #     'batch_size': Config.BATCH_SIZE,
    #     'num_workers': Config.NUM_WORKERS,
    #     'pin_memory': Config.PIN_MEMORY
    # }
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, **Config.DATA_LOADING)
    test_loader = DataLoader(test_dataset, **Config.DATA_LOADING)
    
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
            # Get mean and std from AUGMENTATION config (use val config for visualization)
            mean = Config.AUGMENTATION["val"]["normalize"]["mean"]
            std = Config.AUGMENTATION["val"]["normalize"]["std"]
            img = img * torch.tensor(std).view(3, 1, 1) + \
                  torch.tensor(mean).view(3, 1, 1)
            img = torch.clamp(img, 0, 1)
        
        plt.imshow(img.permute(1, 2, 0))
        plt.title(f"Label: {batch_label[i].item()}")
        plt.xticks([])
        plt.yticks([])
    
    plt.show()
