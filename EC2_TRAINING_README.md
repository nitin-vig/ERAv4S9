# ImageNet Training on EC2 - Multi-GPU Setup

This guide explains how to use `train_imagenet_ec2.py` for training ResNet50 on ImageNet with multi-GPU support and checkpoint resumption.

## Features

✅ **Multi-GPU Support**: Works with both DataParallel and DistributedDataParallel  
✅ **Checkpoint Resumption**: Resume training from any checkpoint  
✅ **Transfer Learning**: Starts from tiny_imagenet weights  
✅ **Mixed Precision**: Uses AMP for faster training  
✅ **Automatic Checkpointing**: Saves best model and periodic checkpoints  
✅ **EC2 Optimized**: No Colab-specific dependencies  

## Prerequisites

1. **ImageNet Dataset**: Ensure ImageNet is downloaded and extracted to the expected structure:
   ```
   /data/imagenet/
   ├── train/
   │   ├── n01440764/
   │   ├── n01443537/
   │   └── ... (1000 class folders)
   └── val/
       ├── n01440764/
       ├── n01443537/
       └── ... (1000 class folders)
   ```
   
   The script will automatically check if the dataset exists and provide download instructions if missing.

2. **Tiny ImageNet Weights**: Have `tiny_imagenet_stage_2/final_model.pth` ready (or specify custom path)

3. **Python Environment**: Install dependencies:
   ```bash
   pip install torch torchvision albumentations tqdm numpy matplotlib
   ```

## Usage

### Basic Training (Single Node, Multiple GPUs)

```bash
python train_imagenet_ec2.py \
    --tiny-imagenet-weights ./tiny_imagenet_stage_2/final_model.pth \
    --data-root /data/imagenet \
    --checkpoint-dir ./checkpoints
```

### Resume from Checkpoint

```bash
python train_imagenet_ec2.py \
    --resume ./checkpoints/checkpoint_epoch_50.pth \
    --data-root /data/imagenet \
    --checkpoint-dir ./checkpoints
```

### Distributed Training (Multi-Node)

Use `torchrun` for distributed training across multiple nodes:

```bash
torchrun --nproc_per_node=4 \
    --nnodes=1 \
    --node_rank=0 \
    --master_addr=localhost \
    --master_port=29500 \
    train_imagenet_ec2.py \
    --distributed \
    --tiny-imagenet-weights ./tiny_imagenet_stage_2/final_model.pth \
    --data-root /data/imagenet \
    --checkpoint-dir ./checkpoints
```

### Custom Configuration

```bash
python train_imagenet_ec2.py \
    --tiny-imagenet-weights ./tiny_imagenet_stage_2/final_model.pth \
    --data-root /data/imagenet \
    --checkpoint-dir ./checkpoints \
    --batch-size 128 \
    --epochs 100 \
    --lr 0.1 \
    --num-workers 16
```

### Compute Dataset Statistics

Compute actual mean/std from your ImageNet dataset:

```bash
python train_imagenet_ec2.py \
    --data-root /data/imagenet \
    --checkpoint-dir ./checkpoints \
    --compute-stats
```

This will:
- Sample 10,000 images from the training set
- Compute mean and standard deviation for RGB channels
- Save results to `checkpoint_dir/dataset_stats.json`
- Display suggested normalization values

### Find Optimal Learning Rate

Automatically find the optimal learning rate before training:

```bash
python train_imagenet_ec2.py \
    --tiny-imagenet-weights ./tiny_imagenet_stage_2/final_model.pth \
    --data-root /data/imagenet \
    --find-lr
```

This will:
- Run LR range test on your model
- Find the optimal max learning rate
- Save it to `checkpoint_dir/optimal_lr.txt`
- Automatically use it for training (unless `--lr` is explicitly set)

### Combined: Find LR and Compute Stats

```bash
python train_imagenet_ec2.py \
    --tiny-imagenet-weights ./tiny_imagenet_stage_2/final_model.pth \
    --data-root /data/imagenet \
    --compute-stats \
    --find-lr
```

## Command Line Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--tiny-imagenet-weights` | `./tiny_imagenet_stage_2/final_model.pth` | Path to tiny_imagenet weights |
| `--resume` | `None` | Resume from checkpoint path |
| `--checkpoint-dir` | `./checkpoints` | Directory to save checkpoints |
| `--data-root` | `/data/imagenet` | Root directory for ImageNet data |
| `--distributed` | `False` | Use distributed training (torchrun) |
| `--batch-size` | `None` | Batch size (overrides config) |
| `--epochs` | `None` | Number of epochs (overrides config) |
| `--lr` | `None` | Learning rate (overrides config) |
| `--num-workers` | `8` | Number of data loading workers |
| `--compute-stats` | `False` | Compute dataset mean/std from actual data |
| `--find-lr` | `False` | Run LR finder before training to find optimal LR |
| `--skip-dataset-check` | `False` | Skip dataset existence check |

## Checkpoint Structure

The script saves three types of checkpoints:

1. **Latest Checkpoint** (`checkpoint_latest.pth`): Updated every epoch
2. **Periodic Checkpoints** (`checkpoint_epoch_N.pth`): Saved every 10 epochs
3. **Best Model** (`checkpoint_best.pth`): Saved when validation loss improves

Each checkpoint contains:
- Model state dict
- Optimizer state dict
- Scheduler state dict (if applicable)
- Epoch number
- Best loss and accuracy
- Timestamp

## Multi-GPU Setup

### Option 1: DataParallel (Simpler, Single Node)

Automatically detects multiple GPUs and uses DataParallel. No extra arguments needed:

```bash
# Uses all available GPUs
python train_imagenet_ec2.py --data-root /data/imagenet
```

### Option 2: DistributedDataParallel (Recommended for Multi-Node)

Use `torchrun` for true distributed training (better performance):

```bash
# Single node, 4 GPUs
torchrun --nproc_per_node=4 train_imagenet_ec2.py \
    --distributed \
    --data-root /data/imagenet

# Multi-node example (adjust for your setup)
torchrun \
    --nproc_per_node=8 \
    --nnodes=2 \
    --node_rank=0 \
    --master_addr=<master-node-ip> \
    --master_port=29500 \
    train_imagenet_ec2.py \
    --distributed \
    --data-root /data/imagenet
```

## EC2 Setup Tips

1. **Instance Type**: Use GPU instances (e.g., `p3.2xlarge`, `p3.8xlarge`, `g4dn.12xlarge`)

2. **Storage**: 
   - Use EBS volumes or instance store for ImageNet data
   - Ensure sufficient space (~150GB for ImageNet)

3. **NVIDIA Drivers**: 
   ```bash
   # Check GPU availability
   nvidia-smi
   
   # Install CUDA if needed (usually pre-installed on Deep Learning AMIs)
   ```

4. **Data Loading**: 
   - Use `--num-workers 16` or higher for faster data loading
   - Ensure data is on fast storage (NVMe SSD)

5. **Screen/Tmux**: Run in screen session for long training:
   ```bash
   screen -S imagenet_training
   python train_imagenet_ec2.py --data-root /data/imagenet
   # Detach: Ctrl+A, then D
   # Reattach: screen -r imagenet_training
   ```

## Dataset Statistics

### Computing Normalization Statistics

The script can compute mean and standard deviation from your actual ImageNet dataset:

```bash
python train_imagenet_ec2.py --compute-stats --data-root /data/imagenet
```

**Output:**
- Computes statistics from 10,000 sample images
- Displays mean and std values
- Saves to `dataset_stats.json`

**Example output:**
```
📊 Computed statistics:
   Mean: [0.4854, 0.4562, 0.4061]
   Std:  [0.2290, 0.2241, 0.2252]
```

These values can be used to update `config.py` if your dataset differs from standard ImageNet.

### Learning Rate Finder

Automatically find the optimal learning rate before training:

```bash
python train_imagenet_ec2.py --find-lr --data-root /data/imagenet
```

**What it does:**
1. Runs LR range test (200 iterations)
2. Finds steepest descent point in loss curve
3. Suggests optimal max_lr for OneCycleLR scheduler
4. Saves result to `optimal_lr.txt`
5. Automatically uses it for training (unless overridden)

**Typical output:**
```
✅ LR Finder suggests max_lr = 0.037500
💡 For One Cycle LR scheduler, use:
   max_lr = 0.037500
   - Initial LR: 0.001500 (warmup starts)
   - Peak LR: 0.037500 (at 30% through training)
   - Final LR: 0.000000 (end of training)
```

## Monitoring Training

1. **Logs**: Check `checkpoint_dir/training_TIMESTAMP.log`

2. **Progress**: Look for output like:
   ```
   Train Epoch 1: Loss=4.5234, Top-1 Acc=12.34%, Top-5 Acc=28.90%
   Validation: Loss=4.1234, Top-1 Acc=15.67%, Top-5 Acc=32.45%
   Best:  Loss=4.1234, Top-1 Acc=15.67%
   💾 Saved checkpoint: ./checkpoints/checkpoint_latest.pth
   ```

3. **GPU Usage**: Monitor with:
   ```bash
   watch -n 1 nvidia-smi
   ```

## Troubleshooting

### Out of Memory
- Reduce `--batch-size` (default is 256, try 128 or 64)
- Reduce `--num-workers`

### Slow Training
- Increase `--num-workers` (try 16-32)
- Ensure data is on fast storage
- Use DistributedDataParallel instead of DataParallel

### Checkpoint Not Loading
- Verify checkpoint file exists and is not corrupted
- Check that model architecture matches (should be ResNet50 for ImageNet)

### Dataset Not Found
- Verify ImageNet is at the path specified by `--data-root`
- Check folder structure matches expected format

## Expected Outputs

After training starts, you should see:
- Model initialization
- Dataset loading statistics
- Training progress with per-epoch metrics
- Checkpoint saves
- Best model tracking

Typical ImageNet training on ResNet50:
- Training time: ~2-4 hours per epoch (depending on hardware)
- Target accuracy: Top-1 ~76-78%, Top-5 ~93-94% (after 100 epochs)

## Next Steps

After training completes:
1. Best model: `./checkpoints/checkpoint_best.pth`
2. Final model: `./checkpoints/checkpoint_latest.pth`
3. Training logs: `./checkpoints/training_*.log`

You can evaluate the model separately or use it for inference.

