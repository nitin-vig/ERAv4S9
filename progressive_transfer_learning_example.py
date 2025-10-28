"""
Example: Progressive Transfer Learning Across Stages
=====================================================

This example shows how to train progressively across stages,
transferring weights from each stage to the next.

Stages: ImageNette → Tiny ImageNet → ImageNet Mini → Full ImageNet
"""

import torch
import os
from config import Config
from dataset_loader import get_data_loaders
from models import get_model
from training_utils import train_model_with_transfer

def progressive_transfer_learning(resume_from_stage=None):
    """Train progressively across multiple stages with transfer learning
    
    Args:
        resume_from_stage: Name of stage to resume from (e.g., "tiny_imagenet", 
                          "imagenet_mini", "imagenet"). If None, starts from Stage 1.
    """
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create model save directory
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
    
    # Handle resuming from a specific stage
    start_idx = 0
    pretrained_weights_path = None
    total_stages = len(stages)  # Total number of stages
    
    if resume_from_stage:
        # Find the stage index
        stage_names = [s["name"] for s in stages]
        if resume_from_stage not in stage_names:
            raise ValueError(f"Unknown stage: {resume_from_stage}. Available stages: {stage_names}")
        
        start_idx = stage_names.index(resume_from_stage)
        
        # Find pretrained weights from previous stage
        if start_idx > 0:
            prev_stage_name = stages[start_idx - 1]["name"]
            weights_file = f"{Config.SAVE_MODEL_PATH}/weights_for_{resume_from_stage}.pth"
            
            if os.path.exists(weights_file):
                pretrained_weights_path = weights_file
                print(f"✅ Resuming from {resume_from_stage} with weights from {prev_stage_name}")
            else:
                print(f"⚠️  Weights file not found: {weights_file}")
                print(f"   Starting {resume_from_stage} without pretrained weights")
        
        # Slice stages to start from resume point
        stages = stages[start_idx:]
        print(f"\n📌 Resuming training from Stage {start_idx + 1}/{total_stages}: {resume_from_stage}")
    else:
        print(f"\n🚀 Starting training from Stage 1/{total_stages}")
    
    for i, stage in enumerate(stages, start=start_idx + 1):
        print(f"\n{'='*80}")
        print(f"STAGE {i}/{total_stages}: {stage['name'].upper()}")
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
            next_stage_name=stage['next'],
            dataset_name=stage['name']
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
    import sys
    
    # Parse command line arguments
    if len(sys.argv) > 1:
        resume_stage = sys.argv[1]
        print(f"Resuming from stage: {resume_stage}")
        progressive_transfer_learning(resume_from_stage=resume_stage)
    else:
        # Run all stages from the beginning
        print("Running all stages from Stage 1")
        print("\nTo resume from a specific stage, run:")
        print("  python progressive_transfer_learning_example.py tiny_imagenet")
        print("  python progressive_transfer_learning_example.py imagenet_mini")
        print("  python progressive_transfer_learning_example.py imagenet")
        print()
        progressive_transfer_learning()

