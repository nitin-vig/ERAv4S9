"""
Example: Progressive Transfer Learning Across Stages
=====================================================

This example shows how to train progressively across stages,
transferring weights from each stage to the next.

Stages: ImageNette → Tiny ImageNet → ImageNet Mini → Full ImageNet
"""

import torch
from config import Config
from dataset_loader import get_data_loaders
from models import get_model
from training_utils import train_model_with_transfer

def progressive_transfer_learning():
    """Train progressively across multiple stages with transfer learning"""
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create model save directory
    import os
    os.makedirs(Config.SAVE_MODEL_PATH, exist_ok=True)
    
    stages = [
        {
            "name": "imagenette",
            "next": "tiny_imagenet",
            "description": "Stage 1: Quick warmup on ImageNette (10 classes)"
        },
        {
            "name": "tiny_imagenet", 
            "next": "imagenet_mini",
            "description": "Stage 2: Medium complexity with Tiny ImageNet (200 classes)"
        },
        {
            "name": "imagenet_mini",
            "next": "imagenet",
            "description": "Stage 3: Full complexity with ImageNet Mini (1000 classes)"
        },
        {
            "name": "imagenet",
            "next": None,
            "description": "Stage 4: Final training on Full ImageNet (1000 classes)"
        }
    ]
    
    pretrained_weights_path = None
    
    for i, stage in enumerate(stages, 1):
        print(f"\n{'='*80}")
        print(f"STAGE {i}/{len(stages)}: {stage['name'].upper()}")
        print(f"{'='*80}")
        print(stage['description'])
        print(f"Previous weights: {pretrained_weights_path}")
        print()
        
        # Update config for this stage
        Config.update_for_dataset(stage['name'])
        
        # Get data loaders
        print(f"Loading {stage['name']} dataset...")
        train_loader, test_loader = get_data_loaders(stage['name'])
        
        # Create model
        print(f"Creating model for {stage['name']}...")
        model = get_model(
            model_name="resnet50",
            dataset_name=stage['name'],
            pretrained=False
        )
        model = model.to(device)
        
        # Train with transfer learning
        print(f"\nTraining {stage['name']}...")
        metrics_tracker, weights = train_model_with_transfer(
            model=model,
            train_loader=train_loader,
            test_loader=test_loader,
            device=device,
            config=Config,
            pretrained_weights_path=pretrained_weights_path,
            next_stage_name=stage['next']
        )
        
        # Set weights path for next stage
        if stage['next']:
            pretrained_weights_path = f"{Config.SAVE_MODEL_PATH}/weights_for_{stage['next']}.pth"
        
        # Plot metrics
        metrics_tracker.plot_metrics(save_path=f"{Config.SAVE_MODEL_PATH}/metrics_stage_{i}_{stage['name']}.png")
        
        print(f"\n✅ Stage {i} completed!")
    
    print("\n" + "="*80)
    print("🎉 ALL STAGES COMPLETE!")
    print("="*80)
    print("\nFinal model saved at:")
    print(f"  {Config.SAVE_MODEL_PATH}/best_model_{stages[-1]['name']}.pth")


if __name__ == "__main__":
    # Optional: Enable only specific stages for testing
    # For now, run all stages
    progressive_transfer_learning()

