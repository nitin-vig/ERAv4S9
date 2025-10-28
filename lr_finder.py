"""
Learning Rate Finder for One Cycle LR Optimization

This utility helps find the optimal max_lr for One Cycle LR scheduling
by running a short training session with exponentially increasing LR.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

def find_optimal_lr(model, train_loader, device, start_lr=1e-7, end_lr=1.0, num_iter=200):
    """
    Learning Rate Finder to determine optimal max_lr for One Cycle LR
    
    Trains the model with exponentially increasing learning rate and tracks the loss.
    The optimal max_lr is typically where the loss decreases most rapidly.
    
    Args:
        model: Model to train
        train_loader: Training data loader
        device: Device to use
        start_lr: Starting learning rate (default: 1e-7)
        end_lr: Ending learning rate (default: 1.0)
        num_iter: Number of iterations to run (default: 200)
        
    Returns:
        tuple: (learning_rates, losses) arrays
    """
    model.train()
    model = model.to(device)
    
    # Use the same optimizer as your actual training
    optimizer = optim.SGD(model.parameters(), lr=start_lr, weight_decay=1e-4, momentum=0.9)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    
    lrs = []
    losses = []
    
    # Create exponential learning rate range
    lr_mult = (end_lr / start_lr) ** (1 / num_iter)
    
    print(f"🔍 Running LR Finder...")
    print(f"   LR range: {start_lr:.2e} → {end_lr:.2e}")
    print(f"   Iterations: {num_iter}")
    
    pbar = tqdm(enumerate(train_loader), total=num_iter, desc="Finding optimal LR")
    
    for batch_idx, (data, target) in pbar:
        if batch_idx >= num_iter:
            break
            
        # Update LR exponentially
        current_lr = start_lr * (lr_mult ** batch_idx)
        for param_group in optimizer.param_groups:
            param_group['lr'] = current_lr
        
        data, target = data.to(device), target.to(device)
        
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        
        # Optional: Gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        lrs.append(current_lr)
        losses.append(loss.item())
        
        pbar.set_description(f"LR: {current_lr:.6f}, Loss: {loss.item():.4f}")
        
        # Stop early if loss explodes
        if loss.item() > 10 or np.isnan(loss.item()):
            print(f"\n⚠️  Loss exploded at LR={current_lr:.6f}")
            break
    
    return np.array(lrs), np.array(losses)

def plot_lr_finder(lrs, losses, save_path=None):
    """
    Plot LR finder results and suggest optimal max_lr
    
    Args:
        lrs: Array of learning rates
        losses: Array of losses
        save_path: Optional path to save the plot
    """
    plt.figure(figsize=(12, 6))
    
    # Plot 1: Loss vs LR
    plt.subplot(1, 2, 1)
    plt.plot(lrs, losses)
    plt.xlabel('Learning Rate (log scale)')
    plt.ylabel('Loss')
    plt.title('LR Finder: Loss vs Learning Rate')
    plt.xscale('log')
    plt.grid(True)
    
    # Find points of interest
    # Point of steepest descent (good for max_lr)
    if len(losses) > 1:
        gradients = np.gradient(losses)
        min_slope_idx = np.argmin(gradients)
        min_slope_lr = lrs[min_slope_idx]
        
        # Minimum loss point
        min_loss_idx = np.argmin(losses)
        min_loss_lr = lrs[min_loss_idx]
        
        plt.axvline(x=min_slope_lr, color='green', linestyle='--', label=f'Steepest descent: {min_slope_lr:.6f}')
        plt.axvline(x=min_loss_lr, color='red', linestyle='--', label=f'Min loss: {min_loss_lr:.6f}')
        plt.legend()
    
    # Plot 2: Smoothed loss
    plt.subplot(1, 2, 2)
    if len(losses) > 10:
        # Smooth the loss curve
        window = min(10, len(losses) // 10)
        smoothed = np.convolve(losses, np.ones(window)/window, mode='valid')
        smoothed_lrs = lrs[:len(smoothed)]
        plt.plot(smoothed_lrs, smoothed)
    else:
        plt.plot(lrs, losses)
    
    plt.xlabel('Learning Rate (log scale)')
    plt.ylabel('Smoothed Loss')
    plt.title('Smoothed Loss vs Learning Rate')
    plt.xscale('log')
    plt.grid(True)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"📊 Plot saved to: {save_path}")
    
    plt.show()
    
    # Print recommendations
    print("\n" + "="*60)
    print("📊 LR FINDER RESULTS")
    print("="*60)
    
    if len(losses) > 1:
        gradients = np.gradient(losses)
        min_slope_idx = np.argmin(gradients)
        min_slope_lr = lrs[min_slope_idx]
        min_slope_loss = losses[min_slope_idx]
        
        # Find where loss starts increasing
        increasing_idx = None
        for i in range(len(gradients)):
            if gradients[i] > 0 and losses[i] > np.min(losses) * 1.1:
                increasing_idx = i
                break
        
        if increasing_idx:
            optimal_lr = lrs[increasing_idx]
            print(f"\n✅ Recommended max_lr for One Cycle: {optimal_lr:.6f}")
            print(f"   (Point where loss stops decreasing)")
        else:
            optimal_lr = min_slope_lr
            print(f"\n✅ Recommended max_lr for One Cycle: {optimal_lr:.6f}")
            print(f"   (Steepest descent point)")
        
        print(f"\n📈 Steepest descent LR: {min_slope_lr:.6f}")
        print(f"📉 Minimum loss LR: {lrs[np.argmin(losses)]:.6f}")
        print(f"\n💡 For One Cycle LR, use max_lr ≈ {optimal_lr:.4f}")
        print(f"   This will give you:")
        print(f"   - Initial LR: {optimal_lr/25:.6f}")
        print(f"   - Peak LR: {optimal_lr:.6f}")
        print(f"   - Final LR: {optimal_lr/10000:.9f}")
        
        return optimal_lr
    else:
        print("⚠️  Not enough data to recommend optimal LR")
        return None

def run_lr_finder_for_stage(stage_name, dataset_name, num_iter=200):
    """
    Convenience function to run LR finder for a specific stage
    
    Args:
        stage_name: Name of the stage (e.g., "tiny_imagenet")
        dataset_name: Dataset to use
        num_iter: Number of iterations
    """
    from models import get_model
    from dataset_loader import get_data_loaders
    from config import Config
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Get config for this stage
    Config.update_for_dataset(dataset_name)
    dataset_config = Config.get_dataset_config(dataset_name)
    
    print(f"\n{'='*60}")
    print(f"LR Finder for {stage_name}")
    print(f"{'='*60}")
    print(f"Dataset: {dataset_name}")
    print(f"Image size: {dataset_config['image_size']}")
    print(f"Classes: {dataset_config['classes']}")
    print(f"Batch size: {dataset_config['batch_size']}")
    
    # Get data loaders
    print("\nLoading dataset...")
    train_loader, _ = get_data_loaders(dataset_name)
    
    # Create model
    print("Creating model...")
    model = get_model(model_name="resnet50", dataset_name=dataset_name)
    
    # Run LR finder
    lrs, losses = find_optimal_lr(model, train_loader, device, num_iter=num_iter)
    
    # Plot and get recommendation
    optimal_lr = plot_lr_finder(lrs, losses, save_path=f"lr_finder_{dataset_name}.png")
    
    return optimal_lr

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        stage = sys.argv[1]
        print(f"Running LR finder for stage: {stage}")
        run_lr_finder_for_stage(stage, stage)
    else:
        print("Usage: python lr_finder.py <stage_name>")
        print("Example: python lr_finder.py tiny_imagenet")

