"""
Learning Rate Finder for One Cycle LR Optimization

Uses pytorch-lr-finder library for finding optimal max_lr.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

try:
    from torch_lr_finder import LRFinder
    HAS_LRFINDER = True
except ImportError:
    print("⚠️ pytorch-lr-finder not installed.")
    print("Install with: pip install torch-lr-finder")
    HAS_LRFINDER = False
    LRFinder = None


def find_optimal_lr_for_stage(model, train_loader, device, dataset_name="tiny_imagenet"):
    """
    Find optimal learning rate for One Cycle LR using pytorch-lr-finder
    
    Args:
        model: Model to train
        train_loader: Training data loader
        device: Device (cuda/cpu)
        dataset_name: Name of dataset (for logging)
        
    Returns:
        tuple: (suggested_min_lr, suggested_max_lr) for One Cycle LR
    """
    if not HAS_LRFINDER:
        raise ImportError("pytorch-lr-finder is required. Install with: pip install torch-lr-finder")
    
    print(f"🔍 Finding optimal LR for {dataset_name} using pytorch-lr-finder")
    print("="*60)
    
    # Use SGD optimizer (same as actual training)
    optimizer = optim.SGD(model.parameters(), lr=1e-8, weight_decay=1e-4, momentum=0.9)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    
    # Create LR finder
    lr_finder = LRFinder(model, optimizer, criterion, device=device)
    
    # Run LR range test (linear mode for better results)
    print("Running LR range test (200 iterations)...")
    lr_finder.range_test(
        train_loader, 
        start_lr=1e-7, 
        end_lr=1.0, 
        num_iter=200, 
        step_mode="linear"
    )
    
    # Get suggestion
    suggested_lr = lr_finder.suggestion()
    
    # Extract history
    learning_rates = lr_finder.history['lr']
    losses = lr_finder.history['loss']
    
    print(f"\n📊 LR Finder Results:")
    print(f"   ✅ Suggested LR: {suggested_lr:.6f}")
    print(f"   📉 Minimum loss: {min(losses):.4f}")
    print(f"   📈 LR range: {min(learning_rates):.2e} → {max(learning_rates):.2e}")
    
    # Plot results
    print("\n📊 Plotting LR finder results...")
    lr_finder.plot(skip_end=0, skip_start=0)
    plt.tight_layout()
    plt.show()
    
    # Calculate One Cycle LR parameters
    print("\n💡 For One Cycle LR scheduler:")
    print(f"   max_lr = {suggested_lr:.6f}")
    print(f"   This gives you:")
    print(f"   - Initial LR: {suggested_lr/25:.6f} (warmup starts)")
    print(f"   - Peak LR: {suggested_lr:.6f} (at 30% through training)")
    print(f"   - Final LR: {suggested_lr/10000:.9f} (end of training)")
    
    return suggested_lr


def run_lr_finder_for_stage(stage_name, dataset_name=None, num_iter=200):
    """
    Convenience function to run LR finder for a specific stage
    
    Args:
        stage_name: Name of the stage (e.g., "tiny_imagenet")
        dataset_name: Dataset to use (defaults to stage_name)
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
    
    if dataset_name is None:
        dataset_name = stage_name
    
    from models import get_model
    from dataset_loader import get_data_loaders
    from config import Config
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Get config for this stage
    Config.update_for_dataset(dataset_name)
    dataset_config = Config.get_dataset_config(dataset_name)
    
    print(f"\n{'='*60}")
    print(f"LR Finder for: {stage_name}")
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
    optimal_lr = find_optimal_lr_for_stage(model, train_loader, device, dataset_name)
    
    print(f"\n✅ Recommended max_lr for One Cycle LR: {optimal_lr:.6f}")
    print(f"💡 Update config.py with: 'lr': {optimal_lr:.6f}")
    
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
        print("\nOr use in notebook:")
        print("  from lr_finder import run_lr_finder_for_stage")
        print("  optimal_lr = run_lr_finder_for_stage('tiny_imagenet')")
