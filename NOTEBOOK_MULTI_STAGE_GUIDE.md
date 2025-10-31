# Notebook Multi-Stage Training Guide

## What Was Added

### 1. Updated Import (Cell 3)
Added `train_model_with_transfer` to the imports:
```python
from training_utils import train_model, train_model_with_transfer, evaluate_model, MetricsTracker
```

### 2. New Cell for Multi-Stage Training (Cells 20-21)

A new section **"Progressive Transfer Learning (Multi-Stage Training)"** has been added with a complete multi-stage training cell.

## How to Use

### Basic Usage

Simply modify the `STAGES_TO_RUN` list in the new cell:

```python
# Choose which stages to run
STAGES_TO_RUN = ["imagenette", "tiny_imagenet"]

# Or run all stages in sequence:
STAGES_TO_RUN = ["imagenette", "tiny_imagenet", "imagenet_mini", "imagenet1k"]
```

### Key Configuration Options

```python
STAGES_TO_RUN = ["imagenette", "tiny_imagenet"]  # Which stages to run
USE_PRETRAINED_FOR_FIRST_STAGE = False  # Use ImageNet pretrained?
SAVE_RESULTS_AT_EACH_STAGE = True  # Save models/metrics at each stage
```

## What Happens at Each Stage

### 1. **Weight Loading** (if applicable)
- Loads weights from previous stage automatically
- Adjusts final layer for different class counts
- Shows which weights were loaded

### 2. **Training**
- Trains with stage-specific hyperparameters
- Uses optimized batch size (256 for ImageNette, 512 for Tiny ImageNet, etc.)
- Albumentations augmentation with adaptive cutout (15-30% of image size)

### 3. **Model & Results Saving** (if enabled)
- **Model**: `./models/{stage_name}_stage_{i}/final_model.pth`
- **Metrics**: `./models/{stage_name}_stage_{i}/metrics.json`
- **Plot**: `./models/{stage_name}_stage_{i}/training_metrics.png`
- **Weights for next**: `./models/weights_for_{next_stage}.pth`

### 4. **Weight Transfer**
- Automatically saves weights for the next stage
- Loads those weights when starting the next stage

## File Structure After Training

```
./models/
├── imagenette_stage_1/
│   ├── final_model.pth
│   ├── metrics.json
│   └── training_metrics.png
├── tiny_imagenet_stage_2/
│   ├── final_model.pth
│   ├── metrics.json
│   └── training_metrics.png
├── weights_for_tiny_imagenet.pth
├── weights_for_imagenet_mini.pth
└── weights_for_imagenet.pth
```

## Example: Running Two Stages

```python
# Configure stages
STAGES_TO_RUN = ["imagenette", "tiny_imagenet"]
USE_PRETRAINED_FOR_FIRST_STAGE = False
SAVE_RESULTS_AT_EACH_STAGE = True

# Run the cell - it will:
# 1. Train ImageNette from scratch
# 2. Save model and results for ImageNette
# 3. Save weights for Tiny ImageNet
# 4. Load weights from ImageNette into Tiny ImageNet model
# 5. Train Tiny ImageNet with transferred weights
# 6. Save model and results for Tiny ImageNet
```

## Progressive Training Orders

### Quick Test (Fastest)
```python
STAGES_TO_RUN = ["imagenette"]  # ~1-2 minutes
```

### Medium Training
```python
STAGES_TO_RUN = ["imagenette", "tiny_imagenet"]  # ~10-15 minutes
```

### Full Progressive Training
```python
STAGES_TO_RUN = ["imagenette", "tiny_imagenet", "imagenet_mini", "imagenet1k"]  # Hours
```

## Benefits

✅ **Automatic transfer learning** between stages  
✅ **Organized results** with dedicated folders per stage  
✅ **Selective training** - run only the stages you need  
✅ **Easy comparison** - metrics saved for each stage  
✅ **Resume capability** - can load weights from any stage  
✅ **GPU optimized** - batch sizes 256-512 for maximum utilization  

## Tips

1. **Start small**: Test with just ImageNette first
2. **Monitor GPU**: Check GPU utilization during training
3. **Save frequently**: Enable `SAVE_RESULTS_AT_EACH_STAGE = True`
4. **Use pretrained**: Set `USE_PRETRAINED_FOR_FIRST_STAGE = True` for better first-stage accuracy
5. **Compare results**: Check metrics.json files to compare stage performance

## Troubleshooting

### Out of memory?
```python
# Reduce to one stage at a time
STAGES_TO_RUN = ["imagenette"]
```

### Want to skip a stage?
```python
# Just don't include it in the list
STAGES_TO_RUN = ["imagenette", "imagenet_mini"]  # Skip tiny_imagenet
```

### Resume from a saved stage?
```python
# Manually set pretrained_weights_path
pretrained_weights_path = "./models/weights_for_imagenet_mini.pth"
# Then run from that stage forward
STAGES_TO_RUN = ["imagenet_mini", "imagenet1k"]
```

Happy training! 🚀

