"""
Dataset Utilities for ImageNet Training
Includes functions for computing normalization statistics and dataset validation
"""

import os
import torch
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm
from PIL import Image
import logging


def compute_dataset_mean_std(dataset, num_samples=10000, batch_size=64, num_workers=4):
    """
    Compute mean and standard deviation of a dataset for normalization
    
    Args:
        dataset: PyTorch dataset
        num_samples: Number of samples to use for computation (default: 10000)
        batch_size: Batch size for computation
        num_workers: Number of workers for data loading
    
    Returns:
        tuple: (mean, std) as lists of 3 values (RGB channels)
    """
    print(f"📊 Computing mean/std from {min(num_samples, len(dataset))} samples...")
    
    # Create a subset if needed
    if num_samples < len(dataset):
        indices = torch.randperm(len(dataset))[:num_samples]
        subset = torch.utils.data.Subset(dataset, indices)
    else:
        subset = dataset
    
    # Create dataloader without normalization transforms
    # We need to temporarily disable normalization to get raw pixel values
    loader = DataLoader(
        subset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True,
        shuffle=False
    )
    
    mean = torch.zeros(3)
    std = torch.zeros(3)
    total_samples = 0
    
    # Compute mean and std
    with torch.no_grad():
        for images, _ in tqdm(loader, desc="Computing statistics"):
            # Convert to float and scale to [0, 1] if needed
            if images.dtype != torch.float32:
                images = images.float()
            if images.max() > 1.0:
                images = images / 255.0
            
            # Reshape to (batch, channels, -1) to compute over spatial dimensions
            batch_size = images.size(0)
            images = images.view(batch_size, 3, -1)
            
            # Accumulate mean and std
            mean += images.mean(2).sum(0)
            std += images.std(2).sum(0)
            total_samples += batch_size
    
    mean /= total_samples
    std /= total_samples
    
    mean_list = mean.tolist()
    std_list = std.tolist()
    
    print(f"✅ Computed statistics:")
    print(f"   Mean: [{mean_list[0]:.4f}, {mean_list[1]:.4f}, {mean_list[2]:.4f}]")
    print(f"   Std:  [{std_list[0]:.4f}, {std_list[1]:.4f}, {std_list[2]:.4f}]")
    
    return mean_list, std_list


def verify_imagenet_structure(data_root):
    """
    Verify ImageNet dataset structure and provide download instructions if missing
    
    Args:
        data_root: Root directory for ImageNet data
    
    Returns:
        tuple: (is_valid, imagenet_path, message)
    """
    imagenet_path = os.path.join(data_root, "full_dataset")
    train_path = os.path.join(imagenet_path, "train")
    
    # Check for both val and validation directories
    val_path = os.path.join(imagenet_path, "val")
    validation_path = os.path.join(imagenet_path, "validation")
    
    # Use whichever validation directory exists
    if os.path.exists(validation_path):
        val_path = validation_path
        val_dir_name = "validation"
    else:
        val_dir_name = "val"
    
    # Check if paths exist
    if not os.path.exists(imagenet_path):
        message = (
            f"❌ ImageNet dataset not found at {imagenet_path}\n"
            f"\n📥 To download ImageNet:\n"
            f"1. Register at https://image-net.org/\n"
            f"2. Download:\n"
            f"   - ILSVRC2012_img_train.tar (~138 GB)\n"
            f"   - ILSVRC2012_img_val.tar (~6.3 GB)\n"
            f"3. Extract:\n"
            f"   mkdir -p {imagenet_path}\n"
            f"   cd {imagenet_path}\n"
            f"   tar -xf ILSVRC2012_img_train.tar\n"
            f"   tar -xf ILSVRC2012_img_val.tar\n"
            f"4. Organize validation set into class folders:\n"
            f"   # See organize_imagenet_val.py or use provided script\n"
            f"\nExpected structure:\n"
            f"  {imagenet_path}/\n"
            f"  ├── train/\n"
            f"  │   ├── n01440764/\n"
            f"  │   ├── n01443537/\n"
            f"  │   └── ... (1000 class folders)\n"
            f"  └── validation/ (or val/)\n"
            f"      ├── n01440764/\n"
            f"      ├── n01443537/\n"
            f"      └── ... (1000 class folders)"
        )
        return False, imagenet_path, message
    
    # Check train and val directories
    if not os.path.exists(train_path):
        message = f"❌ Train directory not found: {train_path}"
        return False, imagenet_path, message
    
    if not os.path.exists(val_path):
        message = f"❌ Validation directory not found: {val_path} or {validation_path}"
        return False, imagenet_path, message
    
    # Check for class folders
    train_classes = [d for d in os.listdir(train_path) 
                    if os.path.isdir(os.path.join(train_path, d))]
    val_classes = [d for d in os.listdir(val_path) 
                  if os.path.isdir(os.path.join(val_path, d))]
    
    if len(train_classes) == 0:
        message = f"❌ No class folders found in {train_path}"
        return False, imagenet_path, message
    
    if len(val_classes) == 0:
        message = f"❌ No class folders found in {val_path}"
        return False, imagenet_path, message
    
    # Count images
    train_images = sum(len([f for f in os.listdir(os.path.join(train_path, cls)) 
                           if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
                      for cls in train_classes[:10])  # Sample check
    
    if train_images == 0:
        message = f"❌ No images found in train folders"
        return False, imagenet_path, message
    
    message = (
        f"✅ ImageNet dataset found at {imagenet_path}\n"
        f"   Train classes: {len(train_classes)}\n"
        f"   Validation classes: {len(val_classes)} (using {val_dir_name}/ directory)\n"
        f"   Expected: 1000 classes each"
    )
    
    if len(train_classes) != 1000 or len(val_classes) != 1000:
        message += f"\n   ⚠️  Warning: Expected 1000 classes, found {len(train_classes)} train and {len(val_classes)} val"
    
    return True, imagenet_path, message


def create_imagenet_val_structure(val_images_dir, val_labels_file=None):
    """
    Organize ImageNet validation images into class folders
    
    Args:
        val_images_dir: Directory containing validation images
        val_labels_file: Optional file with image-to-class mappings
    """
    print("📁 Organizing ImageNet validation set into class folders...")
    
    val_path = val_images_dir
    
    # If val_labels_file is provided, use it
    # Otherwise, try to infer from filename patterns
    # This is a simplified version - full implementation would parse synsets
    
    # For now, just check if already organized
    if os.path.exists(val_path):
        subdirs = [d for d in os.listdir(val_path) 
                  if os.path.isdir(os.path.join(val_path, d))]
        if len(subdirs) > 0:
            print(f"✅ Validation set appears to be organized ({len(subdirs)} folders found)")
            return True
    
    print("⚠️  Validation set organization not implemented automatically.")
    print("   Please organize manually or use ImageNet preparation scripts.")
    return False

