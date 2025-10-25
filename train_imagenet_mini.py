"""
Main training script for ImageNet Mini with ResNet50 in Colab
"""

import torch
import torch.nn as nn
import os
import sys
from google.colab import drive

# Import our modules
from config import Config
from dataset_loader import get_data_loaders, visualize_samples
from models import get_model, count_parameters, get_model_summary, save_model
from training_utils import train_model, evaluate_model, MetricsTracker

def setup_colab_environment():
    """Setup Colab environment"""
    print("Setting up Colab environment...")
    
    # Mount Google Drive
    if Config.MOUNT_DRIVE:
        drive.mount('/content/drive')
        print("Google Drive mounted successfully")
    
    # Create necessary directories
    os.makedirs(Config.DATA_ROOT, exist_ok=True)
    os.makedirs(Config.SAVE_MODEL_PATH, exist_ok=True)
    
    if Config.MOUNT_DRIVE:
        os.makedirs(Config.DRIVE_MODEL_PATH, exist_ok=True)
    
    # Update configuration for Colab
    Config.update_for_colab()
    
    print("Colab environment setup complete!")

def check_gpu_availability():
    """Check GPU availability and setup device"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    return device

def main():
    """Main training function"""
    
    # Setup environment
    setup_colab_environment()
    device = check_gpu_availability()
    
    # Set random seed for reproducibility
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)
    
    print(f"\n{'='*50}")
    print("ImageNet Mini Training with ResNet50")
    print(f"{'='*50}")
    
    # Load dataset
    print("\nLoading dataset...")
    train_loader, test_loader = get_data_loaders(Config.DATASET_NAME)
    
    # Visualize some samples
    print("\nVisualizing sample images...")
    visualize_samples(train_loader, num_samples=12)
    
    # Create model
    print(f"\nCreating {Config.MODEL_NAME} model...")
    model = get_model(
        model_name=Config.MODEL_NAME,
        dataset_name=Config.DATASET_NAME,
        pretrained=Config.PRETRAINED
    )
    
    # Move model to device
    model = model.to(device)
    
    # Print model info
    print(f"Model parameters: {count_parameters(model):,}")
    
    # Get model summary
    dataset_config = Config.get_dataset_config()
    input_size = (3, dataset_config["image_size"], dataset_config["image_size"])
    get_model_summary(model, input_size=input_size)
    
    # Train model
    print(f"\nStarting training for {Config.NUM_EPOCHS} epochs...")
    print(f"Batch size: {Config.BATCH_SIZE}")
    print(f"Learning rate: {Config.LEARNING_RATE}")
    print(f"Weight decay: {Config.WEIGHT_DECAY}")
    
    metrics_tracker = train_model(model, train_loader, test_loader, device, Config)
    
    # Plot training metrics
    print("\nPlotting training metrics...")
    metrics_tracker.plot_metrics(save_path=f"{Config.SAVE_MODEL_PATH}/training_metrics.png")
    
    # Final evaluation
    print("\nFinal evaluation...")
    test_loss, test_acc, test_top5_acc = evaluate_model(model, test_loader, device)
    
    # Save final model
    final_model_path = f"{Config.SAVE_MODEL_PATH}/final_model.pth"
    save_model(model, final_model_path, epoch=Config.NUM_EPOCHS, loss=test_loss)
    
    # Save to Google Drive if mounted
    if Config.MOUNT_DRIVE:
        drive_model_path = f"{Config.DRIVE_MODEL_PATH}/final_model.pth"
        save_model(model, drive_model_path, epoch=Config.NUM_EPOCHS, loss=test_loss)
        print(f"Model also saved to Google Drive: {drive_model_path}")
    
    print(f"\nTraining completed!")
    print(f"Final Test Accuracy: {test_acc:.2f}%")
    print(f"Final Top-5 Accuracy: {test_top5_acc:.2f}%")
    
    return model, metrics_tracker

def quick_test():
    """Quick test function to verify everything works"""
    print("Running quick test...")
    
    # Setup
    setup_colab_environment()
    device = check_gpu_availability()
    
    # Load small subset of data
    train_loader, test_loader = get_data_loaders(Config.DATASET_NAME)
    
    # Create model
    model = get_model(
        model_name=Config.MODEL_NAME,
        dataset_name=Config.DATASET_NAME,
        pretrained=Config.PRETRAINED
    ).to(device)
    
    # Test forward pass
    model.eval()
    with torch.no_grad():
        for data, target in train_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            print(f"Input shape: {data.shape}")
            print(f"Output shape: {output.shape}")
            print(f"Target shape: {target.shape}")
            break
    
    print("Quick test completed successfully!")

if __name__ == "__main__":
    # Check if running in Colab
    try:
        import google.colab
        IN_COLAB = True
    except ImportError:
        IN_COLAB = False
    
    if IN_COLAB:
        print("Running in Google Colab environment")
        main()
    else:
        print("Not running in Colab. Use quick_test() for local testing.")
        quick_test()
