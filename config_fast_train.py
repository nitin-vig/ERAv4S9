"""
Fast Training Configuration for <1 Hour Training on A100
Use this for rapid experimentation or when time is critical.
"""

# Copy the imagenet1k config and modify for speed
FAST_TRAIN_CONFIG = {
    "dataset": "imagenet1k",
    "classes": 1000,
    "image_size": 224,
    "epochs": 15,  # CRITICAL: Reduced from 100 to 15
    "batch_size": 2048,  # HUGE batch (requires 4x A100s or gradient accumulation)
    "lr": 0.8,  # Scaled: 0.1 * (2048/256) = 0.8
    "optimizer": "sgd",
    "scheduler": "one_cycle",
    "weight_decay": 1e-4,
    "label_smoothing": 0.1,
    "gradient_accumulation_steps": 4,  # For single GPU
    "description": "Fast 15-epoch training targeting ~70% in <1 hour"
}

# For single A100 40GB
FAST_TRAIN_SINGLE_GPU = {
    "dataset": "imagenet1k",
    "classes": 1000,
    "image_size": 224,
    "epochs": 10,
    "batch_size": 512,
    "gradient_accumulation_steps": 4,  # Effective batch = 2048
    "lr": 0.8,  # Scaled for effective batch 2048
    "optimizer": "sgd",
    "scheduler": "one_cycle",
    "weight_decay": 1e-4,
    "label_smoothing": 0.1,
    "description": "Fast training on single A100 with gradient accumulation"
}

# For 4x A100 40GB (Multi-GPU)
FAST_TRAIN_MULTI_GPU = {
    "dataset": "imagenet1k",
    "classes": 1000,
    "image_size": 224,
    "epochs": 12,
    "batch_size": 512,  # Per GPU (512 * 4 = 2048 total)
    "lr": 0.8,  # Scaled for total batch 2048
    "optimizer": "sgd",
    "scheduler": "one_cycle",
    "weight_decay": 1e-4,
    "label_smoothing": 0.1,
    "description": "Fast training on 4x A100 with distributed training"
}

# Optimized data loading
FAST_DATA_LOADING = {
    "num_workers": 16,  # Increased from 4
    "pin_memory": True,
    "persistent_workers": True,
    "prefetch_factor": 4,  # Increased from 2
    "drop_last": False
}

