"""
Learning Rate Finder for One Cycle LR Optimization

Uses pytorch-lr-finder library for finding optimal max_lr.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np

try:
    from torch_lr_finder import LRFinder
    HAS_LRFINDER = True
except ImportError:
    print("⚠️ pytorch-lr-finder not installed.")
    print("Install with: pip install torch-lr-finder")
    HAS_LRFINDER = False
    LRFinder = None


def find_optimal_lr(model, train_loader, device, dataset_name="tiny_imagenet", weight_decay=1e-3):
    """
    Find optimal learning rate for One Cycle LR using pytorch-lr-finder
    
    Args:
        model: Model to train
        train_loader: Training data loader
        device: Device (cuda/cpu)
        dataset_name: Name of dataset (for logging)
        weight_decay: Weight decay for the optimizer
        
    Returns:
        Optimal learning rate for One Cycle LR
    """
    if not HAS_LRFINDER:
        raise ImportError("pytorch-lr-finder is required. Install with: pip install torch-lr-finder")
    
    print(f"🔍 Finding optimal LR for {dataset_name} using pytorch-lr-finder")
    print("="*60)
    
    # Use SGD optimizer with CORRECT weight_decay
    optimizer = optim.SGD(model.parameters(), lr=1e-8, weight_decay=weight_decay, momentum=0.9)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    
    # Create LR finder
    lr_finder = LRFinder(model, optimizer, criterion, device=device)
    
    # Run LR range test (linear mode for better results)
    print("Running LR range test (200 iterations)...")
    
    # More conservative end_lr to prevent excessive divergence
    # For transfer learning and One Cycle LR, we typically don't need LR > 0.1
    lr_finder.range_test(
        train_loader, 
        start_lr=1e-7, 
        end_lr=0.1,  # More conservative end limit
        num_iter=200, 
        step_mode="linear"
    )
    
    # Extract history
    learning_rates = lr_finder.history['lr']
    losses = lr_finder.history['loss']
    
    # Find optimal LR (point where loss decreases fastest)
    # This is typically 1 order of magnitude before the minimum
    
    # Find where loss starts increasing (inflection point)
    min_loss_idx = np.argmin(losses)
    min_loss_lr = learning_rates[min_loss_idx]
    
    # Calculate gradients (derivative of loss w.r.t. LR)
    # This tells us where loss decreases fastest
    gradients = np.gradient(losses)
    
    # Find the point where gradient is most negative (steepest descent)
    # but before the loss starts diverging
    # We want to avoid the region right before min_loss where curve flattens
    # and the region after where loss increases
    
    # Look for steepest descent in the first 80% of the range
    search_range = int(0.8 * len(losses))
    
    # Find the most negative gradient before the minimum
    steepest_idx = np.argmin(gradients[:min(search_range, min_loss_idx)])
    
    # Now look for a stable point before divergence
    # Find where loss starts increasing significantly (divergence detection)
    optimal_idx = steepest_idx
    
    # Starting from steepest descent, move forward looking for where
    # loss starts to stabilize (gradient becomes less negative)
    # We want to stop before loss starts increasing
    for i in range(steepest_idx, min(min_loss_idx, len(losses)-5)):
        # Check if loss is still decreasing smoothly
        window_losses = losses[i:min(i+10, len(losses))]
        if len(window_losses) >= 3:
            window_gradient = np.mean(np.gradient(window_losses))
            # If gradient becomes positive or near zero, we've hit the sweet spot
            if window_gradient >= -0.01:  # Near zero or positive (stabilizing)
                optimal_idx = max(i-2, steepest_idx)  # Back up a bit for safety
                break
        else:
            optimal_idx = i
            break
    
    suggested_lr = learning_rates[optimal_idx]
    
    # Additional diagnostics
    suggested_loss = losses[optimal_idx]
    min_loss = min(losses)
    
    # Check for divergence warning
    last_losses = losses[-10:]
    loss_increase_count = sum(1 for i in range(len(last_losses)-1) if last_losses[i+1] > last_losses[i])
    if loss_increase_count >= 7:  # Loss mostly increasing at the end
        print("\n⚠️  WARNING: Loss diverging at high LR. Model may need lower learning rate!")
    
    print(f"\n📊 LR Finder Results:")
    print(f"   📉 Minimum loss: {min_loss:.4f} at LR={min_loss_lr:.6f}")
    print(f"   📊 Suggested loss: {suggested_loss:.4f} at LR={suggested_lr:.6f}")
    print(f"   ✅ Suggested max_lr: {suggested_lr:.6f}")
    print(f"   📈 LR range: {min(learning_rates):.2e} → {max(learning_rates):.2e}")
    print(f"   📍 Suggested LR position: {optimal_idx}/{len(losses)} ({100*optimal_idx/len(losses):.1f}% through range)")
    
    # Warn if suggested LR seems too high
    if suggested_lr > 0.05:
        print(f"\n⚠️  WARNING: Suggested LR ({suggested_lr:.4f}) seems high.")
        print(f"   Consider starting with lower LR (e.g., {suggested_lr/2:.4f}) and monitor for stability.")
    
    # Plot results
    print("\n📊 Plotting LR finder results...")
    lr_finder.plot(skip_end=0, skip_start=0)
    plt.tight_layout()
    plt.show()
    
    # Reset the finder
    lr_finder.reset()
    
    # Calculate One Cycle LR parameters
    print("\n💡 For One Cycle LR scheduler, use:")
    print(f"   max_lr = {suggested_lr:.6f}")
    print(f"   This gives you:")
    print(f"   - Initial LR: {suggested_lr/25:.6f} (warmup starts, div_factor=25)")
    print(f"   - Peak LR: {suggested_lr:.6f} (at 30% through training)")
    print(f"   - Final LR: {suggested_lr/250000:.9f} (end of training, final_div_factor=10000)")
    
    return suggested_lr


def run_lr_finder(model, dataset_name, num_iter=200):
    """
    Convenience function to run LR finder for a specific dataset with the given model
    
    Args:
        model: Model to use for LR finder (can be fresh or pretrained)
        dataset_name: Name of the dataset (e.g., "tiny_imagenet")
        num_iter: Number of iterations
        
    Returns:
        Optimal learning rate for One Cycle LR
    """
    if not HAS_LRFINDER:
        raise ImportError(
            "pytorch-lr-finder is required. Install with:\n"
            "  !pip install torch-lr-finder\n"
            "  # Then run this cell again"
        )
    
    from dataset_loader import get_data_loaders
    from config import Config
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Get config for this dataset
    dataset_config = Config.get_dataset_config(dataset_name)
    
    print(f"\n{'='*60}")
    print(f"LR Finder for: {dataset_name}")
    print(f"{'='*60}")
    print(f"Dataset: {dataset_name}")
    print(f"Image size: {dataset_config['image_size']}")
    print(f"Classes: {dataset_config['classes']}")
    print(f"Batch size: {dataset_config['batch_size']}")
    
    # Get data loaders
    print("\nLoading dataset...")
    train_loader, _ = get_data_loaders(dataset_name)
    
    # Run LR finder
    weight_decay = dataset_config.get("weight_decay", 1e-3)
    optimal_lr = find_optimal_lr(model, train_loader, device, dataset_name, weight_decay=weight_decay)
    
    print(f"\n✅ Recommended max_lr for One Cycle LR: {optimal_lr:.6f}")
    print(f"💡 Update config.py with: 'lr': {optimal_lr:.6f}")
    
    return optimal_lr


if __name__ == "__main__":
    import sys
    from models import get_model
    
    if len(sys.argv) > 1:
        dataset_name = sys.argv[1]
        print(f"Running LR finder for dataset: {dataset_name}")
        
        # Get model and device
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        from config import Config
        Config.update_for_dataset(dataset_name)
        model = get_model(model_name="resnet50", dataset_name=dataset_name)
        model = model.to(device)
        
        run_lr_finder(model, dataset_name)
    else:
        print("Usage: python lr_finder.py <dataset_name>")
        print("Example: python lr_finder.py tiny_imagenet")
        print("\nOr use in notebook:")
        print("  from lr_finder import run_lr_finder")
        print("  from models import get_model")
        print("  model = get_model('resnet50', 'tiny_imagenet')")
        print("  optimal_lr = run_lr_finder(model, 'tiny_imagenet')")
