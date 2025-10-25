"""
Practical Example: Progressive ImageNet Training
Demonstrates the complete workflow from smaller datasets to full ImageNet
"""

import torch
import torch.nn as nn
import torchvision.models as models
from progressive_training_strategy import ProgressiveTrainingStrategy, DatasetManager
import matplotlib.pyplot as plt
import os

def create_resnet50_model():
    """Create a ResNet50 model for progressive training"""
    # Use torchvision's ResNet50 as base
    model = models.resnet50(pretrained=False)  # No pretrained weights
    
    # Modify the final layer to be adaptable
    model.fc = nn.Linear(model.fc.in_features, 1000)  # Default to 1000 classes
    
    return model

def quick_demo():
    """Quick demo with available datasets"""
    print("Progressive ImageNet Training - Quick Demo")
    print("=" * 50)
    
    # Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create model
    base_model = create_resnet50_model()
    print(f"Model created: {sum(p.numel() for p in base_model.parameters()):,} parameters")
    
    # Initialize strategy
    strategy = ProgressiveTrainingStrategy(base_model, device, save_dir="./demo_models")
    
    # Initialize dataset manager
    dataset_manager = DatasetManager(data_root="./data")
    
    # Load available datasets
    print("\nLoading datasets...")
    dataset_loaders = dataset_manager.load_all_datasets()
    
    if not dataset_loaders:
        print("No datasets found. Please ensure datasets are available in ./data/")
        print("Required structure:")
        print("data/")
        print("├── imagenette2/")
        print("├── tiny-imagenet-200/")
        print("├── imagenet-mini/")
        print("└── imagenet/")
        return
    
    # Execute progressive training
    print(f"\nStarting progressive training with {len(dataset_loaders)} datasets...")
    training_history = strategy.progressive_train(dataset_loaders)
    
    # Generate reports
    print("\nGenerating reports and visualizations...")
    strategy.plot_training_progress()
    strategy.generate_training_report()
    
    print("\nDemo completed! Check ./demo_models/ for results.")

def single_stage_example():
    """Example of training a single stage"""
    print("Single Stage Training Example")
    print("=" * 30)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    base_model = create_resnet50_model()
    
    # Initialize strategy
    strategy = ProgressiveTrainingStrategy(base_model, device)
    
    # Load only ImageNette for quick example
    dataset_manager = DatasetManager()
    try:
        train_loader, val_loader = dataset_manager.load_imagenette(batch_size=32)
        
        # Train only ImageNette stage
        print("Training ImageNette stage...")
        weights = strategy.train_stage("imagenette", train_loader, val_loader)
        
        print("Single stage training completed!")
        print(f"Model weights saved with {len(weights)} parameters")
        
    except Exception as e:
        print(f"Error loading ImageNette: {e}")
        print("Please ensure ImageNette dataset is available in ./data/imagenette2/")

def custom_stage_example():
    """Example of adding a custom training stage"""
    print("Custom Stage Example")
    print("=" * 20)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    base_model = create_resnet50_model()
    
    # Initialize strategy
    strategy = ProgressiveTrainingStrategy(base_model, device)
    
    # Add custom stage
    strategy.stages["cifar10"] = {
        "dataset": "cifar10",
        "classes": 10,
        "image_size": 32,
        "epochs": 15,
        "batch_size": 128,
        "lr": 0.001,
        "optimizer": "adamw",
        "scheduler": "cosine",
        "description": "CIFAR-10 warmup stage"
    }
    
    print("Custom stage added: CIFAR-10")
    print("Configuration:")
    for key, value in strategy.stages["cifar10"].items():
        print(f"  {key}: {value}")

def analyze_training_results():
    """Analyze and visualize training results"""
    print("Training Results Analysis")
    print("=" * 25)
    
    # Check if training history exists
    history_file = "./progressive_models/training_history.json"
    if not os.path.exists(history_file):
        print("No training history found. Please run training first.")
        return
    
    import json
    with open(history_file, 'r') as f:
        history = json.load(f)
    
    print("Training Summary:")
    print("-" * 20)
    
    total_time = 0
    for stage, data in history.items():
        print(f"\n{stage.upper()}:")
        print(f"  Best Accuracy: {data['best_val_acc']:.2f}%")
        print(f"  Training Time: {data['training_time']/60:.1f} minutes")
        print(f"  Epochs: {data['config']['epochs']}")
        print(f"  Classes: {data['config']['classes']}")
        total_time += data['training_time']
    
    print(f"\nTotal Training Time: {total_time/3600:.2f} hours")
    
    # Create accuracy progression plot
    stages = list(history.keys())
    accuracies = [history[stage]['best_val_acc'] for stage in stages]
    
    plt.figure(figsize=(10, 6))
    plt.bar(stages, accuracies, color=['skyblue', 'lightgreen', 'orange', 'red'])
    plt.title('Best Validation Accuracy by Stage')
    plt.ylabel('Accuracy (%)')
    plt.xlabel('Training Stage')
    plt.ylim(0, 100)
    
    # Add value labels
    for i, v in enumerate(accuracies):
        plt.text(i, v + 1, f'{v:.1f}%', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig('./training_accuracy_progression.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("\nAccuracy progression plot saved as 'training_accuracy_progression.png'")

def compare_strategies():
    """Compare progressive training vs direct training"""
    print("Strategy Comparison")
    print("=" * 20)
    
    print("Progressive Training Benefits:")
    print("✓ Faster initial convergence")
    print("✓ Better feature learning")
    print("✓ Early architecture validation")
    print("✓ Reduced overfitting risk")
    print("✓ Flexible stopping points")
    print("✓ Resource efficiency")
    
    print("\nDirect Training Limitations:")
    print("✗ Slower convergence")
    print("✗ Higher overfitting risk")
    print("✗ Late architecture validation")
    print("✗ Resource intensive")
    print("✗ Less flexible")
    
    print("\nWhen to Use Progressive Training:")
    print("• Architecture experimentation")
    print("• Resource-constrained environments")
    print("• Educational purposes")
    print("• Hyperparameter optimization")
    print("• Transfer learning studies")

def main():
    """Main function with menu options"""
    print("Progressive ImageNet Training Examples")
    print("=" * 40)
    print("1. Quick Demo (all available datasets)")
    print("2. Single Stage Example (ImageNette only)")
    print("3. Custom Stage Example")
    print("4. Analyze Training Results")
    print("5. Strategy Comparison")
    print("6. Exit")
    
    while True:
        try:
            choice = input("\nSelect an option (1-6): ").strip()
            
            if choice == "1":
                quick_demo()
            elif choice == "2":
                single_stage_example()
            elif choice == "3":
                custom_stage_example()
            elif choice == "4":
                analyze_training_results()
            elif choice == "5":
                compare_strategies()
            elif choice == "6":
                print("Goodbye!")
                break
            else:
                print("Invalid choice. Please select 1-6.")
                
        except KeyboardInterrupt:
            print("\nExiting...")
            break
        except Exception as e:
            print(f"Error: {e}")

if __name__ == "__main__":
    main()
