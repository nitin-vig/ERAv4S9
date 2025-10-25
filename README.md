# Multi-Dataset ImageNet Training with ResNet50 - Modular Version

This project provides a modular implementation for training ResNet50 on multiple ImageNet variants, supporting comprehensive experiments across different dataset sizes and complexities.

## Supported Datasets

| Dataset | Classes | Image Size | Samples | Complexity | Training Time |
|---------|---------|------------|---------|------------|---------------|
| **ImageNette** | 10 | 224×224 | ~13k | Low | ~30 epochs |
| **Tiny ImageNet** | 200 | 64×64 | ~100k | Medium | ~50 epochs |
| **ImageNet Mini** | 1000 | 224×224 | ~100k | High | ~50 epochs |
| **Full ImageNet** | 1000 | 224×224 | ~1.2M | Very High | ~90 epochs |

## Project Structure

```
Imagenet/
├── config.py              # Multi-dataset configuration
├── dataset_loader.py      # Data loading for all datasets
├── models.py             # ResNet50 architecture
├── training_utils.py     # Training utilities and metrics
├── train_imagenet_mini.py # Main training script
├── ImageNet_Tiny_with_Resnet_50.ipynb # Multi-dataset notebook
└── README.md             # This file
```

## Features

- **Multi-Dataset Support**: Easy switching between ImageNette, Tiny ImageNet, ImageNet Mini, and Full ImageNet
- **Dataset-Specific Optimization**: Each dataset has optimized hyperparameters and training strategies
- **Modular Design**: Separate modules for configuration, data loading, models, and training utilities
- **Custom ResNet50**: No pretrained weights, pure custom implementation
- **Comprehensive Metrics**: Tracks Top-1 and Top-5 accuracy with visualization
- **Flexible Configuration**: Easy parameter modification and dataset switching
- **Model Saving**: Automatic checkpointing with dataset-specific naming

## Quick Start

### Option 1: Using the Notebook (Recommended)

1. Upload all Python files to your environment
2. Open `ImageNet_Tiny_with_Resnet_50.ipynb`
3. Modify `DATASET_NAME` in the configuration cell:
   ```python
   DATASET_NAME = "imagenette"  # Options: "imagenette", "tiny_imagenet", "imagenet_mini", "imagenet"
   ```
4. Run all cells sequentially

### Option 2: Using the Training Script

1. Upload all Python files to your environment
2. Modify the dataset in `train_imagenet_mini.py`:
   ```python
   Config.DATASET_NAME = "imagenette"  # Change dataset here
   ```
3. Run the training script:
   ```python
   python train_imagenet_mini.py
   ```

## Configuration

The main configuration is in `config.py`. Key parameters:

```python
# Dataset Configuration
DATASET_NAME = "imagenet_mini"  # Options: "cifar100", "imagenet_mini"
IMAGE_SIZE = 224  # ImageNet standard size
NUM_CLASSES = 1000  # ImageNet has 1000 classes

# Training Configuration
BATCH_SIZE = 32  # Reduced for Colab memory constraints
NUM_EPOCHS = 50
LEARNING_RATE = 0.001
WEIGHT_DECAY = 1e-4

# Model Configuration
MODEL_NAME = "resnet50"
PRETRAINED = True  # Use ImageNet pretrained weights
```

## Dataset Setup

### ImageNet Mini

1. Download ImageNet Mini from Kaggle: https://www.kaggle.com/datasets/ifigotin/imagenetmini-1000
2. Extract the dataset to `./data/imagenet-mini/`
3. The dataset should have the following structure:
```
imagenet-mini/
├── train/
│   ├── class1/
│   ├── class2/
│   └── ...
└── val/
    ├── class1/
    ├── class2/
    └── ...
```

### CIFAR100

CIFAR100 will be automatically downloaded when first used.

## Usage Examples

### Basic Training

```python
from config import Config
from dataset_loader import get_data_loaders
from models import get_model
from training_utils import train_model

# Load data
train_loader, test_loader = get_data_loaders("imagenet_mini")

# Create model
model = get_model("resnet50", "imagenet_mini", pretrained=True)

# Train
metrics_tracker = train_model(model, train_loader, test_loader, device, Config)
```

### Switching to CIFAR100

```python
# Change configuration
Config.DATASET_NAME = "cifar100"
Config.PRETRAINED = False

# Load CIFAR100 data
train_loader, test_loader = get_data_loaders("cifar100")

# Create model for CIFAR100
model = get_model("resnet50", "cifar100", pretrained=False)
```

### Custom Configuration

```python
# Modify training parameters
Config.BATCH_SIZE = 16
Config.NUM_EPOCHS = 30
Config.LEARNING_RATE = 0.0005

# Update for Colab environment
Config.update_for_colab()
```

## Model Architectures

The project supports:

- **ResNet50**: Both custom implementation and torchvision's pretrained version
- **Custom ResNet**: Optimized for CIFAR100 (no initial 7x7 conv)
- **ImageNet ResNet**: Standard ResNet50 for ImageNet (224x224 input)

## Training Features

- **Data Augmentation**: Albumentations for ImageNet, custom transforms for CIFAR100
- **Optimizers**: AdamW, Adam, SGD
- **Schedulers**: ReduceLROnPlateau, CosineAnnealingLR, StepLR
- **Loss Functions**: CrossEntropyLoss with label smoothing
- **Metrics**: Top-1 and Top-5 accuracy tracking
- **Visualization**: Training curves and sample images

## Colab Optimization

The code is optimized for Google Colab:

- Reduced batch size for memory constraints
- Automatic Google Drive mounting
- Model saving to Drive
- Memory-efficient data loading
- GPU detection and utilization

## Requirements

```
torch>=1.9.0
torchvision>=0.10.0
albumentations>=1.0.0
torchsummary>=1.5.0
matplotlib>=3.3.0
numpy>=1.19.0
tqdm>=4.60.0
```

## Installation

```bash
pip install torch torchvision albumentations torchsummary matplotlib numpy tqdm
```

## Troubleshooting

### Memory Issues in Colab

- Reduce batch size: `Config.BATCH_SIZE = 8`
- Use gradient checkpointing
- Clear cache: `torch.cuda.empty_cache()`

### Dataset Not Found

- Ensure ImageNet Mini is downloaded and extracted correctly
- Check the path in `Config.DATA_ROOT`
- Verify the folder structure matches the expected format

### Import Errors

- Make sure all Python files are uploaded to Colab
- Check that all required packages are installed
- Restart the runtime if needed

## Contributing

Feel free to contribute by:
- Adding new model architectures
- Implementing additional datasets
- Improving data augmentation strategies
- Adding new training techniques

## License

This project is open source and available under the MIT License.
