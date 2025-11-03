"""
Dataset Loader for ImageNet Training
Provides data loading utilities for ImageNet-1k, Tiny ImageNet, and other datasets
"""

import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2
from config import Config, ProgressiveConfig


class ImageNetDataset(Dataset):
    """
    ImageNet-1k Dataset Loader
    Expected structure:
        imagenet1k/
          train/
            n01440764/
              image1.JPEG
              ...
            n01443537/
              ...
          val/
            n01440764/
              image1.JPEG
              ...
    """
    
    def __init__(self, root, split='train', transform=None):
        """
        Args:
            root: Path to imagenet1k directory (not train/val subdirs)
            split: 'train' or 'val'
            transform: Albumentations transform
        """
        self.root = root
        self.split = split
        self.transform = transform
        
        # Path to split directory
        split_dir = os.path.join(root, split)
        
        if not os.path.exists(split_dir):
            raise ValueError(f"Split directory not found: {split_dir}")
        
        # Get all class folders
        self.classes = sorted([d for d in os.listdir(split_dir) 
                              if os.path.isdir(os.path.join(split_dir, d))])
        
        # Build list of (image_path, class_idx) tuples
        self.samples = []
        for class_idx, class_name in enumerate(self.classes):
            class_dir = os.path.join(split_dir, class_name)
            images = [f for f in os.listdir(class_dir) 
                     if f.lower().endswith(('.jpg', '.jpeg', '.png', '.JPEG', '.JPG'))]
            
            for img_name in images:
                img_path = os.path.join(class_dir, img_name)
                self.samples.append((img_path, class_idx))
        
        print(f"📊 ImageNet {split} dataset:")
        print(f"   Classes: {len(self.classes)}")
        print(f"   Samples: {len(self.samples)}")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        
        # Load image
        try:
            image = Image.open(img_path).convert('RGB')
            image = np.array(image)
        except Exception as e:
            print(f"Error loading {img_path}: {e}")
            # Return a black image if loading fails
            image = np.zeros((224, 224, 3), dtype=np.uint8)
        
        # Apply transforms
        if self.transform:
            transformed = self.transform(image=image)
            image = transformed['image']
        
        return image, label


class TinyImageNetDataset(Dataset):
    """
    Tiny ImageNet Dataset Loader
    Expected structure:
        tiny-imagenet-200/
          train/
            n01443537/
              images/
                xxx.JPEG
              boxes.txt
            ...
          val/
            images/
              xxx.JPEG
            val_annotations.txt
          wnids.txt
          words.txt
    """
    
    def __init__(self, root, split='train', transform=None):
        """
        Args:
            root: Path to tiny-imagenet-200 directory
            split: 'train' or 'val'
            transform: Albumentations transform
        """
        self.root = root
        self.split = split
        self.transform = transform
        
        # Path to split directory
        if split == 'train':
            split_dir = os.path.join(root, 'train')
            self.samples = []
            self.classes = sorted([d for d in os.listdir(split_dir) 
                                  if os.path.isdir(os.path.join(split_dir, d))])
            
            for class_idx, class_name in enumerate(self.classes):
                class_dir = os.path.join(split_dir, class_name, 'images')
                if not os.path.exists(class_dir):
                    continue
                images = [f for f in os.listdir(class_dir) 
                         if f.lower().endswith(('.jpg', '.jpeg', '.png', '.JPEG', '.JPG'))]
                
                for img_name in images:
                    img_path = os.path.join(class_dir, img_name)
                    self.samples.append((img_path, class_idx))
        
        elif split == 'val':
            val_dir = os.path.join(root, 'val')
            val_images_dir = os.path.join(val_dir, 'images')
            val_annotations = os.path.join(val_dir, 'val_annotations.txt')
            
            # Read class mappings from val_annotations.txt
            self.class_to_idx = {}
            if os.path.exists(val_annotations):
                with open(val_annotations, 'r') as f:
                    for line in f:
                        parts = line.strip().split('\t')
                        if len(parts) >= 2:
                            img_name = parts[0]
                            class_name = parts[1]
                            if class_name not in self.class_to_idx:
                                self.class_to_idx[class_name] = len(self.class_to_idx)
            
            # Get all unique classes
            if not self.class_to_idx:
                # Fallback: use train classes
                train_dir = os.path.join(root, 'train')
                self.classes = sorted([d for d in os.listdir(train_dir) 
                                      if os.path.isdir(os.path.join(train_dir, d))])
                self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
            else:
                self.classes = sorted(self.class_to_idx.keys())
            
            # Build samples list
            self.samples = []
            if os.path.exists(val_annotations):
                with open(val_annotations, 'r') as f:
                    for line in f:
                        parts = line.strip().split('\t')
                        if len(parts) >= 2:
                            img_name = parts[0]
                            class_name = parts[1]
                            img_path = os.path.join(val_images_dir, img_name)
                            if os.path.exists(img_path):
                                class_idx = self.class_to_idx[class_name]
                                self.samples.append((img_path, class_idx))
            else:
                # Fallback: load all images from val/images
                if os.path.exists(val_images_dir):
                    images = [f for f in os.listdir(val_images_dir) 
                             if f.lower().endswith(('.jpg', '.jpeg', '.png', '.JPEG', '.JPG'))]
                    # Assign to first class as fallback
                    for img_name in images:
                        img_path = os.path.join(val_images_dir, img_name)
                        self.samples.append((img_path, 0))
        
        else:
            raise ValueError(f"Invalid split: {split}. Must be 'train' or 'val'")
        
        print(f"📊 Tiny ImageNet {split} dataset:")
        print(f"   Classes: {len(self.classes)}")
        print(f"   Samples: {len(self.samples)}")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        
        # Load image
        try:
            image = Image.open(img_path).convert('RGB')
            image = np.array(image)
        except Exception as e:
            print(f"Error loading {img_path}: {e}")
            # Return a black image if loading fails
            image = np.zeros((64, 64, 3), dtype=np.uint8)
        
        # Apply transforms
        if self.transform:
            transformed = self.transform(image=image)
            image = transformed['image']
        
        return image, label


def get_albumentations_transforms(dataset_name, split='train'):
    """
    Get Albumentations transforms for dataset
    
    Args:
        dataset_name: Name of dataset (imagenet1k, tiny_imagenet, etc.)
        split: 'train' or 'val'
    
    Returns:
        Albumentations Compose transform
    """
    # Get dataset config
    dataset_config = Config.get_dataset_config(dataset_name)
    image_size = dataset_config.get('image_size', 224)
    
    # Get augmentation config
    aug_config = Config.AUGMENTATION.get(split, Config.AUGMENTATION.get('val', {}))
    
    transforms = []
    
    if split == 'train':
        # Training augmentations
        resize = aug_config.get('resize', image_size)
        crop = aug_config.get('crop', image_size)
        
        # Resize
        if resize != crop:
            transforms.append(A.Resize(resize, resize))
            transforms.append(A.RandomCrop(crop, crop))
        else:
            transforms.append(A.Resize(image_size, image_size))
        
        # Horizontal flip
        if aug_config.get('horizontal_flip', False):
            transforms.append(A.HorizontalFlip(p=0.5))
        
        # Color jitter
        color_jitter = aug_config.get('color_jitter', {})
        if color_jitter:
            transforms.append(A.ColorJitter(
                brightness=color_jitter.get('brightness', 0),
                contrast=color_jitter.get('contrast', 0),
                saturation=color_jitter.get('saturation', 0),
                hue=color_jitter.get('hue', 0),
                p=0.8
            ))
        
        # Normalize
        normalize = aug_config.get('normalize', {})
        mean = normalize.get('mean', [0.485, 0.456, 0.406])
        std = normalize.get('std', [0.229, 0.224, 0.225])
        transforms.append(A.Normalize(mean=mean, std=std))
    
    else:
        # Validation/test augmentations
        resize = aug_config.get('resize', image_size)
        crop = aug_config.get('crop', image_size)
        
        # Resize and center crop
        if resize != crop:
            transforms.append(A.Resize(resize, resize))
            transforms.append(A.CenterCrop(crop, crop))
        else:
            transforms.append(A.Resize(image_size, image_size))
        
        # Normalize
        normalize = aug_config.get('normalize', {})
        mean = normalize.get('mean', [0.485, 0.456, 0.406])
        std = normalize.get('std', [0.229, 0.224, 0.225])
        transforms.append(A.Normalize(mean=mean, std=std))
    
    # Convert to tensor
    transforms.append(ToTensorV2())
    
    return A.Compose(transforms)


def get_data_loaders(dataset_name, data_root=None, batch_size=None, num_workers=None):
    """
    Get train and validation data loaders for specified dataset
    
    Args:
        dataset_name: Name of dataset (imagenet1k, tiny_imagenet, imagenette, imagenet_mini)
        data_root: Root directory for datasets (defaults to Config.DATA_ROOT)
        batch_size: Batch size (defaults to dataset config)
        num_workers: Number of data loading workers (defaults to Config.DATA_LOADING)
    
    Returns:
        tuple: (train_loader, val_loader)
    """
    # Get dataset config
    dataset_config = Config.get_dataset_config(dataset_name)
    
    # Use provided data_root or config default
    if data_root is None:
        data_root = Config.DATA_ROOT
    
    # Get batch size
    if batch_size is None:
        batch_size = dataset_config.get('batch_size', 256)
    
    # Get num_workers
    if num_workers is None:
        num_workers = Config.DATA_LOADING.get('num_workers', 4)
    
    # Get image size
    image_size = dataset_config.get('image_size', 224)
    
    # Get transforms
    train_transform = get_albumentations_transforms(dataset_name, split='train')
    val_transform = get_albumentations_transforms(dataset_name, split='val')
    
    # Create datasets based on dataset name
    if dataset_name == 'imagenet1k' or dataset_name == 'imagenet_mini':
        # ImageNet-1k dataset
        imagenet_path = os.path.join(data_root, "imagenet1k")
        if not os.path.exists(imagenet_path):
            raise ValueError(f"ImageNet dataset not found at {imagenet_path}. Please download it first.")
        
        train_dataset = ImageNetDataset(imagenet_path, split='train', transform=train_transform)
        val_dataset = ImageNetDataset(imagenet_path, split='val', transform=val_transform)
    
    elif dataset_name == 'tiny_imagenet':
        # Tiny ImageNet dataset
        # Try common paths
        tiny_imagenet_paths = [
            os.path.join(data_root, "tiny-imagenet-200"),
            os.path.join(data_root, "tiny_imagenet"),
            data_root if "tiny-imagenet" in data_root.lower() else None
        ]
        
        tiny_imagenet_path = None
        for path in tiny_imagenet_paths:
            if path and os.path.exists(path):
                tiny_imagenet_path = path
                break
        
        if tiny_imagenet_path is None:
            raise ValueError(f"Tiny ImageNet dataset not found. Tried: {tiny_imagenet_paths}")
        
        train_dataset = TinyImageNetDataset(tiny_imagenet_path, split='train', transform=train_transform)
        val_dataset = TinyImageNetDataset(tiny_imagenet_path, split='val', transform=val_transform)
    
    elif dataset_name == 'imagenette':
        # ImageNette dataset (using ImageFolder style)
        imagenette_path = os.path.join(data_root, "imagenette2")
        if not os.path.exists(imagenette_path):
            raise ValueError(f"ImageNette dataset not found at {imagenette_path}. Please download it first.")
        
        # Use torchvision ImageFolder for ImageNette
        # Create wrapper transform for albumentations
        class AlbumentationsTransform:
            def __init__(self, transform):
                self.transform = transform
            
            def __call__(self, image):
                # Convert PIL to numpy
                if isinstance(image, Image.Image):
                    image = np.array(image)
                # Apply albumentations
                transformed = self.transform(image=image)
                return transformed['image']
        
        from torchvision.datasets import ImageFolder
        train_dataset = ImageFolder(
            root=os.path.join(imagenette_path, 'train'),
            transform=AlbumentationsTransform(train_transform)
        )
        val_dataset = ImageFolder(
            root=os.path.join(imagenette_path, 'val'),
            transform=AlbumentationsTransform(val_transform)
        )
    
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}. Supported: imagenet1k, tiny_imagenet, imagenette, imagenet_mini")
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=Config.DATA_LOADING.get('pin_memory', True),
        persistent_workers=Config.DATA_LOADING.get('persistent_workers', False) if num_workers > 0 else False,
        prefetch_factor=Config.DATA_LOADING.get('prefetch_factor', 8) if num_workers > 0 else None,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=Config.DATA_LOADING.get('pin_memory', True),
        persistent_workers=Config.DATA_LOADING.get('persistent_workers', False) if num_workers > 0 else False,
        prefetch_factor=Config.DATA_LOADING.get('prefetch_factor', 8) if num_workers > 0 else None,
        drop_last=False
    )
    
    print(f"\n📦 Dataset: {dataset_name}")
    print(f"   Train samples: {len(train_dataset)}")
    print(f"   Val samples: {len(val_dataset)}")
    print(f"   Batch size: {batch_size}")
    print(f"   Number of classes: {dataset_config.get('classes', 'unknown')}")
    
    return train_loader, val_loader

