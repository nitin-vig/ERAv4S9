"""
ImageNet Training Script for EC2 Multi-GPU Setup
================================================

This script trains ResNet50 on ImageNet dataset with:
- Multi-GPU support (DataParallel or DistributedDataParallel)
- Resume from checkpoint capability
- Loading weights from tiny_imagenet stage
- Automatic checkpoint saving
- EC2-friendly configuration

Usage:
    # Single node, multiple GPUs
    python train_imagenet_ec2.py --tiny-imagenet-weights ./tiny_imagenet_stage_2/final_model.pth
    
    # Resume from checkpoint
    python train_imagenet_ec2.py --resume ./checkpoints/checkpoint_epoch_50.pth
    
    # Distributed training (if using torchrun)
    torchrun --nproc_per_node=4 train_imagenet_ec2.py --distributed
"""

import argparse
import os
import sys
import json
import time
import logging
from datetime import datetime
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.parallel import DataParallel, DistributedDataParallel
from torch.utils.data.distributed import DistributedSampler
from torch.cuda.amp import GradScaler, autocast
import torch.backends.cudnn as cudnn

# Import project modules
from config import Config
from dataset_loader import get_data_loaders, ImageNetDataset, get_albumentations_transforms
from models import get_model, save_model, load_model, get_id_to_class_mapping
from training_utils import topk_accuracy, get_optimizer, get_scheduler, get_criterion
from dataset_utils import verify_imagenet_structure, compute_dataset_mean_std
from lr_finder import find_optimal_lr
import albumentations as A
from albumentations.pytorch import ToTensorV2


def setup_logging(log_dir):
    """Setup logging to both file and console"""
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    
    log_file = log_dir / f"training_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    return logging.getLogger(__name__)


def setup_distributed():
    """Setup distributed training if applicable"""
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ['RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        local_rank = int(os.environ.get('LOCAL_RANK', 0))
        
        torch.cuda.set_device(local_rank)
        torch.distributed.init_process_group(
            backend='nccl',
            init_method='env://',
            rank=rank,
            world_size=world_size
        )
        
        return True, rank, world_size, local_rank
    else:
        return False, 0, 1, 0


def cleanup_distributed():
    """Cleanup distributed training"""
    if torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()


def load_tiny_imagenet_weights(model, weights_path, logger, device):
    """Load weights from tiny_imagenet stage, handling class mismatch"""
    if not os.path.exists(weights_path):
        logger.error(f"❌ Weights file not found: {weights_path}")
        return False
    
    try:
        logger.info(f"📦 Loading tiny_imagenet weights from: {weights_path}")
        checkpoint = torch.load(weights_path, map_location=device)
        
        # Handle both state_dict formats
        if isinstance(checkpoint, dict):
            if 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
            elif 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
            else:
                state_dict = checkpoint
        else:
            state_dict = checkpoint
        
        # Get current model output size (1000 for ImageNet)
        current_fc_size = model.fc.out_features
        
        # Check pretrained FC size
        pretrained_fc_size = None
        if 'fc.weight' in state_dict:
            pretrained_fc_size = state_dict['fc.weight'].shape[0]
        
        logger.info(f"   Current model (ImageNet): {current_fc_size} classes")
        logger.info(f"   Pretrained model (Tiny ImageNet): {pretrained_fc_size} classes" if pretrained_fc_size else "   Pretrained model: no FC layer found")
        
        # Filter out final layer if sizes don't match
        if pretrained_fc_size and current_fc_size != pretrained_fc_size:
            logger.info(f"   ⚠️  Final layer size mismatch! Filtering out fc layer")
            state_dict = {k: v for k, v in state_dict.items() if 'fc' not in k}
        else:
            logger.info(f"   ✅ Final layer sizes match")
        
        # Remove DataParallel/DistributedDataParallel prefix if present
        model_state_dict = model.state_dict()
        cleaned_state_dict = {}
        for k, v in state_dict.items():
            # Remove 'module.' prefix if present
            key = k.replace('module.', '') if k.startswith('module.') else k
            if key in model_state_dict and model_state_dict[key].shape == v.shape:
                cleaned_state_dict[key] = v
        
        # Load with strict=False to allow FC layer mismatch
        missing_keys, unexpected_keys = model.load_state_dict(cleaned_state_dict, strict=False)
        
        if missing_keys:
            logger.info(f"   ⚠️  Missing keys (expected for FC layer): {len(missing_keys)}")
        if unexpected_keys:
            logger.info(f"   ⚠️  Unexpected keys: {len(unexpected_keys)}")
        
        logger.info(f"✅ Successfully loaded tiny_imagenet weights")
        return True
        
    except Exception as e:
        logger.error(f"❌ Error loading weights: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False


def train_epoch_distributed(model, train_loader, optimizer, criterion, epoch, 
                             scaler, scheduler, logger, device, is_distributed):
    """Train for one epoch with distributed/multi-GPU support"""
    model.train()
    
    total_loss = 0.0
    correct = 0
    correct_top5 = 0
    total_samples = 0
    
    # Progress bar only on rank 0
    if (not is_distributed) or (is_distributed and torch.distributed.get_rank() == 0):
        from tqdm import tqdm
        pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}')
    else:
        pbar = train_loader
    
    for batch_idx, (data, target) in enumerate(pbar):
        data, target = data.to(device, non_blocking=True), target.to(device, non_blocking=True)
        
        optimizer.zero_grad()
        
        # Mixed precision training
        with autocast():
            output = model(data)
            loss = criterion(output, target)
        
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        # Step scheduler per batch for One Cycle LR
        if scheduler is not None and isinstance(scheduler, optim.lr_scheduler.OneCycleLR):
            scheduler.step()
        
        # Calculate metrics
        total_loss += loss.item()
        pred = output.argmax(dim=1, keepdim=True)
        correct += pred.eq(target.view_as(pred)).sum().item()
        correct_top5 += topk_accuracy(output, target, k=5)
        total_samples += len(data)
        
        # Update progress bar
        if (not is_distributed) or (is_distributed and torch.distributed.get_rank() == 0):
            batch_accuracy = 100. * correct / total_samples
            avg_loss = total_loss / (batch_idx + 1)
            current_lr = optimizer.param_groups[0]['lr']
            pbar.set_postfix({
                'Loss': f'{avg_loss:.4f}',
                'Acc': f'{batch_accuracy:.2f}%',
                'LR': f'{current_lr:.6f}'
            })
    
    # Average across all processes for distributed training
    if is_distributed:
        # Gather metrics from all processes
        metrics = torch.tensor([total_loss, correct, correct_top5, total_samples], 
                              device=device, dtype=torch.float32)
        torch.distributed.all_reduce(metrics, op=torch.distributed.ReduceOp.SUM)
        total_loss = metrics[0].item()
        correct = int(metrics[1].item())
        correct_top5 = int(metrics[2].item())
        total_samples = int(metrics[3].item())
    
    epoch_loss = total_loss / len(train_loader)
    epoch_accuracy = 100. * correct / total_samples
    epoch_top5_accuracy = 100. * correct_top5 / total_samples
    
    if (not is_distributed) or (is_distributed and torch.distributed.get_rank() == 0):
        logger.info(f"Train Epoch {epoch+1}: Loss={epoch_loss:.4f}, "
                   f"Top-1 Acc={epoch_accuracy:.2f}%, Top-5 Acc={epoch_top5_accuracy:.2f}%")
    
    return epoch_loss, epoch_accuracy, epoch_top5_accuracy


def validate(model, test_loader, criterion, logger, device, is_distributed):
    """Validate the model"""
    model.eval()
    
    total_loss = 0.0
    correct = 0
    correct_top5 = 0
    total_samples = 0
    
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device, non_blocking=True), target.to(device, non_blocking=True)
            
            with autocast():
                output = model(data)
                loss = criterion(output, target)
            
            total_loss += loss.item() * len(data)
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            correct_top5 += topk_accuracy(output, target, k=5)
            total_samples += len(data)
    
    # Average across all processes for distributed training
    if is_distributed:
        metrics = torch.tensor([total_loss, correct, correct_top5, total_samples], 
                              device=device, dtype=torch.float32)
        torch.distributed.all_reduce(metrics, op=torch.distributed.ReduceOp.SUM)
        total_loss = metrics[0].item()
        correct = int(metrics[1].item())
        correct_top5 = int(metrics[2].item())
        total_samples = int(metrics[3].item())
    
    test_loss = total_loss / total_samples
    accuracy = 100. * correct / total_samples
    top5_accuracy = 100. * correct_top5 / total_samples
    
    if (not is_distributed) or (is_distributed and torch.distributed.get_rank() == 0):
        logger.info(f"Validation: Loss={test_loss:.4f}, "
                   f"Top-1 Acc={accuracy:.2f}%, Top-5 Acc={top5_accuracy:.2f}%")
    
    return test_loss, accuracy, top5_accuracy


def save_checkpoint(model, optimizer, scheduler, epoch, loss, accuracy, checkpoint_dir, 
                   is_distributed, logger, is_best=False, id_to_class=None):
    """Save training checkpoint with required id-to-class mapping"""
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    # Get model state dict (handle DataParallel/DistributedDataParallel)
    if isinstance(model, (DataParallel, DistributedDataParallel)):
        model_state_dict = model.module.state_dict()
    else:
        model_state_dict = model.state_dict()
    
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model_state_dict,
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
        'accuracy': accuracy,
        'timestamp': datetime.now().isoformat()
    }
    
    if scheduler is not None:
        checkpoint['scheduler_state_dict'] = scheduler.state_dict()
    
    # id_to_class mapping is required
    if id_to_class is None:
        raise ValueError("id_to_class mapping is required when saving checkpoint. Provide dataset/loader or mapping explicitly.")
    
    checkpoint['id_to_class'] = id_to_class
    logger.info(f"✅ Class mapping included in checkpoint ({len(id_to_class)} classes)")
    
    # Save latest checkpoint
    latest_path = checkpoint_dir / 'checkpoint_latest.pth'
    torch.save(checkpoint, latest_path)
    
    # Save periodic checkpoint
    if (epoch + 1) % 10 == 0:
        periodic_path = checkpoint_dir / f'checkpoint_epoch_{epoch+1}.pth'
        torch.save(checkpoint, periodic_path)
        logger.info(f"💾 Saved periodic checkpoint: {periodic_path}")
    
    # Save best model
    if is_best:
        best_path = checkpoint_dir / 'checkpoint_best.pth'
        torch.save(checkpoint, best_path)
        logger.info(f"🏆 Saved best model checkpoint: {best_path}")
    
    if (not is_distributed) or (is_distributed and torch.distributed.get_rank() == 0):
        logger.info(f"💾 Saved checkpoint: {latest_path}")


def load_checkpoint(model, optimizer, scheduler, checkpoint_path, device, logger):
    """Load training checkpoint"""
    if not os.path.exists(checkpoint_path):
        logger.error(f"❌ Checkpoint not found: {checkpoint_path}")
        return None, 0
    
    try:
        logger.info(f"📦 Loading checkpoint from: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        
        # Load model state
        model_state_dict = checkpoint['model_state_dict']
        # Handle DataParallel/DistributedDataParallel
        if isinstance(model, (DataParallel, DistributedDataParallel)):
            model.module.load_state_dict(model_state_dict)
        else:
            model.load_state_dict(model_state_dict)
        
        # Load optimizer state
        if 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        # Load scheduler state
        if scheduler is not None and 'scheduler_state_dict' in checkpoint:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        start_epoch = checkpoint.get('epoch', 0)
        best_loss = checkpoint.get('loss', float('inf'))
        best_accuracy = checkpoint.get('accuracy', 0.0)
        
        logger.info(f"✅ Loaded checkpoint: epoch={start_epoch}, loss={best_loss:.4f}, acc={best_accuracy:.2f}%")
        
        return checkpoint, start_epoch + 1  # Resume from next epoch
        
    except Exception as e:
        logger.error(f"❌ Error loading checkpoint: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return None, 0


def main():
    parser = argparse.ArgumentParser(description='ImageNet Training on EC2 with Multi-GPU')
    parser.add_argument('--tiny-imagenet-weights', type=str, 
                       default='./tiny_imagenet_stage_2/final_model.pth',
                       help='Path to tiny_imagenet weights')
    parser.add_argument('--resume', type=str, default=None,
                       help='Resume from checkpoint path')
    parser.add_argument('--checkpoint-dir', type=str, default='./checkpoints',
                       help='Directory to save checkpoints')
    parser.add_argument('--data-root', type=str, default='/data/imagenet',
                       help='Root directory for ImageNet data')
    parser.add_argument('--dataset', type=str, default='imagenet1k', 
                       choices=['imagenet', 'imagenet1k', 'tiny_imagenet'],
                       help='Dataset to train on (imagenet, imagenet1k or tiny_imagenet)')
    parser.add_argument('--distributed', action='store_true',
                       help='Use distributed training (torchrun)')
    parser.add_argument('--batch-size', type=int, default=None,
                       help='Batch size (overrides config)')
    parser.add_argument('--epochs', type=int, default=None,
                       help='Number of epochs (overrides config)')
    parser.add_argument('--lr', type=float, default=None,
                       help='Learning rate (overrides config)')
    parser.add_argument('--num-workers', type=int, default=8,
                       help='Number of data loading workers')
    parser.add_argument('--compute-stats', action='store_true',
                       help='Compute dataset mean/std from actual data')
    parser.add_argument('--find-lr', action='store_true',
                       help='Run LR finder before training to find optimal learning rate')
    parser.add_argument('--skip-dataset-check', action='store_true',
                       help='Skip dataset existence check')
    parser.add_argument('--quick-fix-labels', action='store_true',
                       help='Quick fix: Retrain only FC layer with correct labels (freezes backbone, sets epochs to 5). Use with --resume to fix existing model.')
    parser.add_argument('--fix-epochs', type=int, default=5,
                       help='Number of epochs for quick fix (default: 5, only used with --quick-fix-labels)')
    
    args = parser.parse_args()
    
    # Setup distributed training if applicable
    is_distributed, rank, world_size, local_rank = setup_distributed()
    use_cuda = torch.cuda.is_available()
    device = torch.device(f'cuda:{local_rank}' if use_cuda else 'cpu')
    
    # Setup logging (only rank 0 writes)
    if (not is_distributed) or (is_distributed and rank == 0):
        logger = setup_logging(args.checkpoint_dir)
        logger.info("="*80)
        logger.info("ImageNet Training on EC2 - Multi-GPU Setup")
        logger.info("="*80)
        logger.info(f"Device: {device}")
        logger.info(f"Distributed: {is_distributed}, Rank: {rank}, World Size: {world_size}")
        logger.info(f"CUDA Available: {use_cuda}, GPU Count: {torch.cuda.device_count()}")
    else:
        # Dummy logger for other ranks
        logger = logging.getLogger(__name__)
        logger.setLevel(logging.WARNING)
    
    # Set cudnn benchmark for better performance
    if use_cuda:
        cudnn.benchmark = True
    
    # Update config for specified dataset
    Config.update_for_dataset(args.dataset)
    Config.DATA_ROOT = args.data_root
    
    # Verify dataset structure
    if not args.skip_dataset_check and ((not is_distributed) or (is_distributed and rank == 0)):
        if args.dataset == 'imagenet' or args.dataset == 'imagenet1k':
            is_valid, imagenet_path, message = verify_imagenet_structure(args.data_root)
            logger.info(message)
            if not is_valid:
                logger.error("Please download and organize ImageNet dataset before training")
                return
        elif args.dataset == 'tiny_imagenet':
            # For tiny_imagenet, check if the dataset directory exists
            tiny_imagenet_path = args.data_root
            if not os.path.exists(tiny_imagenet_path):
                logger.error(f"❌ Tiny ImageNet dataset not found at {tiny_imagenet_path}")
                return
            
            # Check for required subdirectories
            required_dirs = ['train', 'val']
            missing_dirs = [d for d in required_dirs if not os.path.exists(os.path.join(tiny_imagenet_path, d))]
            if missing_dirs:
                logger.error(f"❌ Missing required directories in Tiny ImageNet: {missing_dirs}")
                logger.error(f"Expected structure: {tiny_imagenet_path}/{{train,val}}/")
                return
            
            logger.info(f"✅ Tiny ImageNet dataset found at {tiny_imagenet_path}")
            imagenet_path = tiny_imagenet_path
    
    # Override config if provided
    if args.batch_size:
        Config.BATCH_SIZE = args.batch_size
    if args.epochs:
        Config.NUM_EPOCHS = args.epochs
    
    dataset_config = Config.get_dataset_config(args.dataset)
    
    if (not is_distributed) or (is_distributed and rank == 0):
        logger.info(f"Dataset: {args.dataset.title()}")
        logger.info(f"Batch size: {Config.BATCH_SIZE}")
        logger.info(f"Epochs: {Config.NUM_EPOCHS}")
        logger.info(f"Learning rate: {dataset_config.get('lr', Config.LEARNING_RATE)}")
    
    # Get data loaders
    if (not is_distributed) or (is_distributed and rank == 0):
        logger.info(f"📂 Loading {args.dataset.title()} dataset from: {args.data_root}...")
    
    train_loader, test_loader = get_data_loaders(args.dataset)
    
    # Extract id-to-class mapping from dataset (required)
    id_to_class = None
    if (not is_distributed) or (is_distributed and rank == 0):
        id_to_class = get_id_to_class_mapping(train_loader)
        if id_to_class is None:
            raise ValueError("Could not extract class mapping from dataset. Dataset must have 'classes' or 'class_to_idx' attribute.")
        logger.info(f"✅ Extracted class mapping: {len(id_to_class)} classes")
    else:
        # For non-rank-0 processes, we still need id_to_class for save_checkpoint
        # but it won't be used. Set to empty dict to avoid None error.
        id_to_class = {}
    
    # Compute dataset statistics if requested
    if args.compute_stats and ((not is_distributed) or (is_distributed and rank == 0)):
        logger.info("📊 Computing dataset mean/std statistics...")
        # Create a temporary dataset without normalization for stats computation
        if args.dataset == 'imagenet' or args.dataset == 'imagenet1k':
            dataset_path = os.path.join(args.data_root, "full_dataset")
        else:  # tiny_imagenet
            dataset_path = args.data_root
            
        dataset_config = Config.get_dataset_config(args.dataset)
        
        # Create transforms without normalization
        train_transform_no_norm = A.Compose([
            A.Resize(dataset_config["image_size"], dataset_config["image_size"]),
            ToTensorV2(),
        ])
        
        if args.dataset == 'imagenet' or args.dataset == 'imagenet1k':
            stats_dataset = ImageNetDataset(dataset_path, split='train', transform=train_transform_no_norm)
        else:  # tiny_imagenet
            from dataset_loader import TinyImageNetDataset
            stats_dataset = TinyImageNetDataset(dataset_path, split='train', transform=train_transform_no_norm)
        mean, std = compute_dataset_mean_std(stats_dataset, num_samples=10000, 
                                             batch_size=64, num_workers=args.num_workers)
        
        logger.info(f"\n💡 Suggested normalization values:")
        logger.info(f"   mean = {mean}")
        logger.info(f"   std = {std}")
        logger.info(f"\n   Update config.py AUGMENTATION section with these values")
        
        # Save to file
        stats_file = os.path.join(args.checkpoint_dir, 'dataset_stats.json')
        os.makedirs(args.checkpoint_dir, exist_ok=True)
        import json
        with open(stats_file, 'w') as f:
            json.dump({'mean': mean, 'std': std}, f, indent=2)
        logger.info(f"   Saved to: {stats_file}")
    
    # Override batch size and num_workers in loaders if needed
    if args.batch_size and train_loader.batch_size != args.batch_size:
        train_dataset = train_loader.dataset
        test_dataset = test_loader.dataset
        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            pin_memory=True,
            drop_last=False
        )
        test_loader = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=True,
            drop_last=False
        )
    elif args.num_workers and train_loader.num_workers != args.num_workers:
        # Only update num_workers
        train_dataset = train_loader.dataset
        test_dataset = test_loader.dataset
        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=train_loader.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            pin_memory=True,
            drop_last=False
        )
        test_loader = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=test_loader.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=True,
            drop_last=False
        )
    
    # Use DistributedSampler for distributed training
    train_sampler = None
    if is_distributed:
        train_sampler = DistributedSampler(train_loader.dataset, 
                                         num_replicas=world_size, 
                                         rank=rank,
                                         shuffle=True)
        train_loader = torch.utils.data.DataLoader(
            train_loader.dataset,
            batch_size=train_loader.batch_size,
            sampler=train_sampler,
            num_workers=args.num_workers,
            pin_memory=True,
            drop_last=True
        )
    
    if (not is_distributed) or (is_distributed and rank == 0):
        logger.info(f"Training samples: {len(train_loader.dataset)}")
        logger.info(f"Validation samples: {len(test_loader.dataset)}")
        logger.info(f"Batches per epoch: {len(train_loader)}")
    
    # Create model
    if (not is_distributed) or (is_distributed and rank == 0):
        logger.info("🤖 Creating ResNet50 model...")
    
    num_classes = dataset_config.get('classes', 1000)
    model = get_model(model_name='resnet50', dataset_name=args.dataset, num_classes=num_classes)
    model = model.to(device)
    
    # Quick fix labels mode: Override settings for fast FC-only retraining
    if args.quick_fix_labels:
        if (not is_distributed) or (is_distributed and rank == 0):
            logger.info("🔧 Quick Fix Labels Mode: Retraining FC layer only")
            logger.info("   This will fix label mapping issues quickly")
        # Force freeze backbone and set epochs
        args.freeze_backbone = True
        Config.NUM_EPOCHS = args.fix_epochs
        if args.epochs:
            Config.NUM_EPOCHS = args.epochs
        if (not is_distributed) or (is_distributed and rank == 0):
            logger.info(f"   Epochs: {Config.NUM_EPOCHS}")
            logger.info(f"   Backbone: Frozen (only FC layer will train)")
        # Use higher LR for FC-only training
        if args.lr is None:
            max_lr = dataset_config.get('lr', Config.LEARNING_RATE) * 10  # Higher LR for FC-only
            if (not is_distributed) or (is_distributed and rank == 0):
                logger.info(f"   Learning rate: {max_lr:.6f} (10x default for FC-only training)")
    
    # Load tiny_imagenet weights if not resuming
    start_epoch = 0
    if args.resume is None and os.path.exists(args.tiny_imagenet_weights):
        if (not is_distributed) or (is_distributed and rank == 0):
            load_tiny_imagenet_weights(model, args.tiny_imagenet_weights, logger, device)
    elif args.resume is None:
        if (not is_distributed) or (is_distributed and rank == 0):
            logger.warning(f"⚠️  Tiny ImageNet weights not found: {args.tiny_imagenet_weights}")
            logger.warning("   Training from scratch")
    
    # Initialize max_lr variable (if not set by quick-fix)
    if 'max_lr' not in locals():
        max_lr = args.lr if args.lr else dataset_config.get('lr', Config.LEARNING_RATE)
    
    # Run LR finder if requested
    optimal_lr = None
    if args.find_lr and ((not is_distributed) or (is_distributed and rank == 0)):
        logger.info("🔍 Running Learning Rate Finder...")
        try:
            # Get base model (unwrap DataParallel/DistributedDataParallel if needed)
            base_model = model.module if isinstance(model, (DataParallel, DistributedDataParallel)) else model
            
            optimal_lr = find_optimal_lr(
                base_model,
                train_loader,
                device,
                dataset_name=args.dataset,
                weight_decay=dataset_config.get('weight_decay', 1e-3)
            )
            
            logger.info(f"✅ LR Finder suggests max_lr = {optimal_lr:.6f}")
            logger.info(f"💡 Update your training command with: --lr {optimal_lr:.6f}")
            
            # Save optimal LR
            lr_file = os.path.join(args.checkpoint_dir, 'optimal_lr.txt')
            with open(lr_file, 'w') as f:
                f.write(f"{optimal_lr:.6f}\n")
            logger.info(f"   Saved to: {lr_file}")
            
            # Use optimal LR if not explicitly set
            if args.lr is None:
                max_lr = optimal_lr
                logger.info(f"   Using suggested LR: {max_lr:.6f} for training")
            
        except Exception as e:
            logger.error(f"❌ LR Finder failed: {e}")
            import traceback
            logger.error(traceback.format_exc())
            logger.info("   Continuing with default/configured LR...")
            max_lr = args.lr if args.lr else dataset_config.get('lr', Config.LEARNING_RATE)
    
    # Setup multi-GPU or distributed training
    if is_distributed:
        model = DistributedDataParallel(model, device_ids=[local_rank], 
                                       output_device=local_rank,
                                       find_unused_parameters=False)
        if rank == 0:
            logger.info("✅ Using DistributedDataParallel")
    elif use_cuda and torch.cuda.device_count() > 1:
        model = DataParallel(model)
        if rank == 0:
            logger.info(f"✅ Using DataParallel with {torch.cuda.device_count()} GPUs")
    
    # Setup optimizer, scheduler, criterion
    optimizer = get_optimizer(
        model.module if isinstance(model, (DataParallel, DistributedDataParallel)) else model,
        optimizer_name=dataset_config.get('optimizer', 'sgd'),
        lr=max_lr,
        weight_decay=Config.WEIGHT_DECAY
    )
    
    steps_per_epoch = len(train_loader)
    scheduler_name = dataset_config.get('scheduler', 'one_cycle')
    
    if scheduler_name.lower() == 'one_cycle':
        scheduler = get_scheduler(
            optimizer,
            scheduler_name=scheduler_name,
            max_lr=max_lr,
            epochs=Config.NUM_EPOCHS,
            steps_per_epoch=steps_per_epoch
        )
    else:
        scheduler = get_scheduler(
            optimizer,
            scheduler_name=scheduler_name,
            **{k: v for k, v in dataset_config.items() if k not in ['dataset', 'classes', 'image_size']}
        )
    
    criterion = get_criterion(criterion_name='cross_entropy', 
                             label_smoothing=dataset_config.get('label_smoothing', 0.1))
    
    # Mixed precision training
    scaler = GradScaler()
    
    # Load checkpoint if resuming
    best_test_loss = float('inf')
    best_test_accuracy = 0.0
    if args.resume:
        checkpoint, loaded_epoch = load_checkpoint(
            model, optimizer, scheduler, args.resume, device, logger
        )
        if checkpoint:
            best_test_loss = checkpoint.get('loss', float('inf'))
            best_test_accuracy = checkpoint.get('accuracy', 0.0)
            # For quick-fix mode, reset start_epoch to 0 to retrain FC layer from scratch
            # Otherwise, resume from the loaded epoch
            if args.quick_fix_labels:
                start_epoch = 0
                if (not is_distributed) or (is_distributed and rank == 0):
                    logger.info(f"   Quick-fix mode: Resetting epoch to 0 (will retrain FC for {Config.NUM_EPOCHS} epochs)")
            else:
                start_epoch = loaded_epoch
    
    # Freeze backbone layers if quick-fix mode (after checkpoint load to preserve weights)
    if args.quick_fix_labels:
        # Get base model (unwrap DataParallel/DistributedDataParallel if needed)
        base_model = model.module if isinstance(model, (DataParallel, DistributedDataParallel)) else model
        # Freeze all layers except FC
        for name, param in base_model.named_parameters():
            if 'fc' not in name:  # Keep FC layer trainable
                param.requires_grad = False
        if (not is_distributed) or (is_distributed and rank == 0):
            frozen_params = sum(1 for p in base_model.parameters() if not p.requires_grad)
            trainable_params = sum(1 for p in base_model.parameters() if p.requires_grad)
            logger.info(f"   Frozen {frozen_params} parameters, {trainable_params} trainable (FC layer only)")
    
    if (not is_distributed) or (is_distributed and rank == 0):
        logger.info("🚀 Starting training...")
        logger.info(f"   Start epoch: {start_epoch}, Total epochs: {Config.NUM_EPOCHS}")
    
    # Training loop
    for epoch in range(start_epoch, Config.NUM_EPOCHS):
        if is_distributed and train_sampler is not None:
            train_sampler.set_epoch(epoch)  # Shuffle each epoch
        
        # Train
        train_loss, train_acc, train_top5 = train_epoch_distributed(
            model, train_loader, optimizer, criterion, epoch, 
            scaler, scheduler, logger, device, is_distributed
        )
        
        # Validate
        test_loss, test_acc, test_top5 = validate(
            model, test_loader, criterion, logger, device, is_distributed
        )
        
        # Step scheduler (if not OneCycleLR which steps per batch)
        if scheduler is not None and not isinstance(scheduler, optim.lr_scheduler.OneCycleLR):
            if isinstance(scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(test_loss)
            else:
                scheduler.step()
        
        # Update best model
        is_best = test_loss < best_test_loss
        if is_best:
            best_test_loss = test_loss
            best_test_accuracy = test_acc
        
        # Save checkpoint (only rank 0)
        if (not is_distributed) or (is_distributed and rank == 0):
            save_checkpoint(
                model, optimizer, scheduler, epoch, test_loss, test_acc,
                args.checkpoint_dir, is_distributed, logger, is_best=is_best,
                id_to_class=id_to_class
            )
        
        # Log epoch summary
        if (not is_distributed) or (is_distributed and rank == 0):
            logger.info(f"Epoch {epoch+1}/{Config.NUM_EPOCHS} Summary:")
            logger.info(f"  Train: Loss={train_loss:.4f}, Top-1={train_acc:.2f}%, Top-5={train_top5:.2f}%")
            logger.info(f"  Val:   Loss={test_loss:.4f}, Top-1={test_acc:.2f}%, Top-5={test_top5:.2f}%")
            logger.info(f"  Best:  Loss={best_test_loss:.4f}, Top-1={best_test_accuracy:.2f}%")
            logger.info("-"*80)
    
    # Cleanup
    cleanup_distributed()
    
    if (not is_distributed) or (is_distributed and rank == 0):
        if args.quick_fix_labels:
            logger.info("✅ Quick Fix Labels completed!")
            logger.info(f"   Model FC layer retrained with correct labels")
            logger.info(f"   Best validation accuracy: {best_test_accuracy:.2f}%")
            logger.info(f"   Checkpoint saved with correct class mapping: {args.checkpoint_dir}/checkpoint_best.pth")
        else:
            logger.info("✅ Training completed!")
            logger.info(f"Best validation loss: {best_test_loss:.4f}")
            logger.info(f"Best validation accuracy: {best_test_accuracy:.2f}%")


if __name__ == '__main__':
    main()

