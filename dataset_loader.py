"""
Dataset Loader for ImageNet Training
Provides data loading utilities for ImageNet-1k, Tiny ImageNet, and other datasets
"""

import os
import json
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2
from pathlib import Path
from config import Config, ProgressiveConfig
import matplotlib.pyplot as plt


def load_imagenet_standard_order():
    """
    Load standard ImageNet class ordering from imagenet_class_index.json.
    Returns the ordered list of WNID class names (e.g., ['n01440764', 'n01443537', ...])
    """
    # Try multiple possible locations for the JSON file
    possible_paths = [
        Path(__file__).parent.parent / "hf_app" / "imagenet_class_index.json",
        Path(__file__).parent / "hf_app" / "imagenet_class_index.json",
        Path("hf_app") / "imagenet_class_index.json",
        Path("imagenet_class_index.json"),
    ]
    
    for path in possible_paths:
        if path.exists():
            try:
                with open(path, 'r') as f:
                    class_index = json.load(f)
                # Extract WNIDs in standard order: [idx_map["0"][0], idx_map["1"][0], ...]
                standard_order = [class_index[str(i)][0] for i in range(len(class_index))]
                print(f"✅ Loaded standard ImageNet ordering from {path} ({len(standard_order)} classes)")
                return standard_order
            except Exception as e:
                print(f"⚠️  Error loading {path}: {e}")
    
    print("⚠️  imagenet_class_index.json not found, falling back to alphabetical sorting")
    return None


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
        
        # Get all class folders found in the directory
        available_classes = [d for d in os.listdir(split_dir) 
                            if os.path.isdir(os.path.join(split_dir, d))]
        
        # Use standard ImageNet ordering if available, otherwise fall back to alphabetical
        standard_order = load_imagenet_standard_order()
        if standard_order is not None:
            # Filter standard_order to only include classes that exist in the dataset
            # and maintain the standard order
            self.classes = [cls for cls in standard_order if cls in available_classes]
            # Add any extra classes that might be in the dataset but not in standard order
            extra_classes = set(available_classes) - set(self.classes)
            if extra_classes:
                print(f"⚠️  Found {len(extra_classes)} classes not in standard order, appending alphabetically")
                self.classes.extend(sorted(extra_classes))
        else:
            # Fallback to alphabetical sorting
            self.classes = sorted(available_classes)
            print(f"⚠️  Using alphabetical sorting (standard order not available)")
        
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
            
            # Get all class folders found in the directory
            available_classes = [d for d in os.listdir(split_dir) 
                              if os.path.isdir(os.path.join(split_dir, d))]
            
            # For Tiny ImageNet, try to use wnids.txt if available for standard ordering
            wnids_file = os.path.join(root, 'wnids.txt')
            if os.path.exists(wnids_file):
                try:
                    with open(wnids_file, 'r') as f:
                        standard_order = [line.strip() for line in f if line.strip()]
                    # Filter to only include classes that exist in the dataset
                    self.classes = [cls for cls in standard_order if cls in available_classes]
                    # Add any extra classes
                    extra_classes = set(available_classes) - set(self.classes)
                    if extra_classes:
                        print(f"⚠️  Found {len(extra_classes)} classes not in wnids.txt, appending alphabetically")
                        self.classes.extend(sorted(extra_classes))
                    print(f"✅ Using Tiny ImageNet ordering from wnids.txt")
                except Exception as e:
                    print(f"⚠️  Error reading wnids.txt: {e}, falling back to alphabetical")
                    self.classes = sorted(available_classes)
            else:
                # Fallback to alphabetical sorting if wnids.txt not available
                self.classes = sorted(available_classes)
                print(f"⚠️  wnids.txt not found, using alphabetical sorting")
            
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
                # Fallback: use train classes with standard ordering if available
                train_dir = os.path.join(root, 'train')
                available_classes = [d for d in os.listdir(train_dir) 
                                  if os.path.isdir(os.path.join(train_dir, d))]
                
                # Try to use wnids.txt for standard ordering
                wnids_file = os.path.join(root, 'wnids.txt')
                if os.path.exists(wnids_file):
                    try:
                        with open(wnids_file, 'r') as f:
                            standard_order = [line.strip() for line in f if line.strip()]
                        self.classes = [cls for cls in standard_order if cls in available_classes]
                        extra_classes = set(available_classes) - set(self.classes)
                        if extra_classes:
                            self.classes.extend(sorted(extra_classes))
                    except Exception:
                        self.classes = sorted(available_classes)
                else:
                    self.classes = sorted(available_classes)
                
                self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
            else:
                # Use wnids.txt ordering if available, otherwise alphabetical
                wnids_file = os.path.join(root, 'wnids.txt')
                if os.path.exists(wnids_file):
                    try:
                        with open(wnids_file, 'r') as f:
                            standard_order = [line.strip() for line in f if line.strip()]
                        # Filter to only include classes in class_to_idx
                        self.classes = [cls for cls in standard_order if cls in self.class_to_idx]
                        extra_classes = set(self.class_to_idx.keys()) - set(self.classes)
                        if extra_classes:
                            self.classes.extend(sorted(extra_classes))
                    except Exception:
                        self.classes = sorted(self.class_to_idx.keys())
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
    if dataset_name == 'imagenet' or dataset_name == 'imagenet1k' or dataset_name == 'imagenet_mini':
        # ImageNet-1k dataset
        imagenet_path = os.path.join(data_root, "imagenet1k")
        if not os.path.exists(imagenet_path):
            raise ValueError(f"ImageNet dataset not found at {imagenet_path}. Please download it first.")
        
        train_dataset = ImageNetDataset(imagenet_path, split='train', transform=train_transform)
        
        # Check if validation directory exists, otherwise use val
        val_split = 'validation' if os.path.exists(os.path.join(imagenet_path, 'validation')) else 'val'
        val_dataset = ImageNetDataset(imagenet_path, split=val_split, transform=val_transform)
    
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
        prefetch_factor=Config.DATA_LOADING.get('prefetch_factor', 2) if num_workers > 0 else None,
        drop_last=False
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=Config.DATA_LOADING.get('pin_memory', True),
        persistent_workers=Config.DATA_LOADING.get('persistent_workers', False) if num_workers > 0 else False,
        prefetch_factor=Config.DATA_LOADING.get('prefetch_factor', 2) if num_workers > 0 else None,
        drop_last=False
    )
    
    print(f"\n📦 Dataset: {dataset_name}")
    print(f"   Train samples: {len(train_dataset)}")
    print(f"   Val samples: {len(val_dataset)}")
    print(f"   Batch size: {batch_size}")
    print(f"   Number of classes: {dataset_config.get('classes', 'unknown')}")
    
    return train_loader, val_loader


def visualize_samples(
    data_loader,
    num_samples=8,
    cols=4,
    mean=None,
    std=None,
    figsize=(12, 8),
    class_map=None,
):
    """
    Visualize a grid of samples from a DataLoader, showing label ids and names.

    Args:
        data_loader: PyTorch DataLoader yielding (images, labels)
        num_samples: Total number of images to display
        cols: Number of columns in the grid
        mean: Per-channel mean used for normalization (defaults to ImageNet)
        std: Per-channel std used for normalization (defaults to ImageNet)
        figsize: Matplotlib figure size
        class_map: Optional mapping for label → name. Can be dict[int,str] or
                   list/tuple indexed by label id. If None, will try to infer
                   from dataset (prefers `class_to_idx`, falls back to `classes`).
    """
    if num_samples <= 0:
        return
    if mean is None:
        mean = [0.485, 0.456, 0.406]
    if std is None:
        std = [0.229, 0.224, 0.225]

    images, labels = next(iter(data_loader))
    images = images[:num_samples]
    labels = labels[:num_samples]

    rows = int(np.ceil(num_samples / cols))
    fig, axes = plt.subplots(rows, cols, figsize=figsize)
    axes = np.array(axes).reshape(rows, cols)

    mean_tensor = torch.tensor(mean).view(3, 1, 1)
    std_tensor = torch.tensor(std).view(3, 1, 1)

    # Build index → class name map
    idx_to_name = None
    if class_map is not None:
        if isinstance(class_map, dict):
            idx_to_name = class_map
        elif isinstance(class_map, (list, tuple)):
            idx_to_name = {i: name for i, name in enumerate(class_map)}
    else:
        dataset = getattr(data_loader, 'dataset', None)
        if dataset is not None and hasattr(dataset, 'class_to_idx') and isinstance(dataset.class_to_idx, dict):
            idx_to_name = {v: k for k, v in dataset.class_to_idx.items()}
        elif dataset is not None and hasattr(dataset, 'classes') and isinstance(dataset.classes, (list, tuple)):
            idx_to_name = {i: name for i, name in enumerate(dataset.classes)}

    idx = 0
    for r in range(rows):
        for c in range(cols):
            ax = axes[r, c]
            if idx < len(images):
                img = images[idx].detach().cpu()
                img = img * std_tensor + mean_tensor
                img = torch.clamp(img, 0.0, 1.0)
                img_np = img.permute(1, 2, 0).numpy()
                ax.imshow(img_np)
                label_id = int(labels[idx])
                if idx_to_name is not None and label_id in idx_to_name:
                    ax.set_title(f"{idx_to_name[label_id]} ({label_id})")
                else:
                    ax.set_title(f"label: {label_id}")
            ax.axis('off')
            idx += 1

    plt.tight_layout()
    plt.show()

