"""
Configuration file for Progressive ImageNet Training Strategy
Easy customization of training parameters and stages
"""

import os

class ProgressiveConfig:
    """Configuration class for progressive training strategy"""
    
    # Base configuration
    DEVICE = "cuda" if os.environ.get("CUDA_VISIBLE_DEVICES") else "cpu"
    DATA_ROOT = "/content/data"
    SAVE_DIR = "./progressive_models"
    
    # Additional attributes for notebook compatibility
    SAVE_MODEL_PATH = "./models"
    MODEL_NAME = "resnet50"
    MOUNT_DRIVE = False
    DRIVE_MODEL_PATH = "/content/drive/MyDrive/models"

    
    # Training stages configuration
    STAGES = {
        "imagenette": {
            "dataset": "imagenette",
            "classes": 10,
            "image_size": 224,
            "epochs": 20,
            "batch_size": 256,
            "lr": 0.001,
            "optimizer": "adamw",
            "scheduler": "cosine",
            "weight_decay": 1e-4,
            "label_smoothing": 0.1,
            "description": "Quick warmup and architecture validation",
            "enabled": True,
            "priority": 1
        },
        "tiny_imagenet": {
            "dataset": "tiny_imagenet",
            "classes": 200,
            "image_size": 64,
            "epochs": 30,
            "batch_size": 128,
            "lr": 0.0005,
            "optimizer": "adamw",
            "scheduler": "cosine",
            "weight_decay": 1e-4,
            "label_smoothing": 0.1,
            "description": "Medium complexity training",
            "enabled": True,
            "priority": 2
        },
        "imagenet_mini": {
            "dataset": "imagenet_mini",
            "classes": 1000,
            "image_size": 224,
            "epochs": 40,
            "batch_size": 96,
            "lr": 0.0003,
            "optimizer": "sgd",
            "scheduler": "step",
            "weight_decay": 1e-4,
            "label_smoothing": 0.1,
            "description": "Full ImageNet complexity with subset data",
            "enabled": True,
            "priority": 3
        },
        "imagenet": {
            "dataset": "imagenet",
            "classes": 1000,
            "image_size": 224,
            "epochs": 60,
            "batch_size": 128,
            "lr": 0.1,
            "optimizer": "sgd",
            "scheduler": "step",
            "weight_decay": 1e-4,
            "label_smoothing": 0.1,
            "description": "Final full-scale training",
            "enabled": True,
            "priority": 4
        }
    }
    
    # Data loading configuration
    DATA_LOADING = {
        "num_workers": 4,
        "pin_memory": True,
        "persistent_workers": True,
        "prefetch_factor": 2
    }
    
    # Training configuration
    TRAINING = {
        "mixed_precision": True,
        "gradient_clipping": 1.0,
        "early_stopping_patience": 10,
        "save_best_only": True,
        "monitor_metric": "val_acc",
        "monitor_mode": "max"
    }
    
    # Logging configuration
    LOGGING = {
        "log_interval": 50,
        "save_interval": 5,
        "plot_interval": 1,
        "verbose": True,
        "log_level": "INFO"
    }
    
    # Model configuration
    MODEL = {
        "architecture": "resnet50",
        "pretrained": False,
        "freeze_backbone": False,
        "dropout_rate": 0.0,
        "batch_norm_momentum": 0.1
    }
    
    # Augmentation configuration
    AUGMENTATION = {
        "train": {
            "resize": 256,
            "crop": 224,
            "horizontal_flip": True,
            "vertical_flip": False,
            "rotation": 0,
            "color_jitter": {
                "brightness": 0.2,
                "contrast": 0.2,
                "saturation": 0.2,
                "hue": 0.1
            },
            "normalize": {
                "mean": [0.485, 0.456, 0.406],
                "std": [0.229, 0.224, 0.225]
            }
        },
        "val": {
            "resize": 256,
            "crop": 224,
            "horizontal_flip": False,
            "normalize": {
                "mean": [0.485, 0.456, 0.406],
                "std": [0.229, 0.224, 0.225]
            }
        }
    }
    
    @classmethod
    def get_enabled_stages(cls):
        """Get list of enabled stages in priority order"""
        enabled_stages = {k: v for k, v in cls.STAGES.items() if v.get("enabled", True)}
        return sorted(enabled_stages.items(), key=lambda x: x[1]["priority"])
    
    @classmethod
    def get_stage_config(cls, stage_name):
        """Get configuration for specific stage"""
        if stage_name in cls.STAGES:
            return cls.STAGES[stage_name]
        else:
            raise ValueError(f"Unknown stage: {stage_name}")
    
    @classmethod
    def get_dataset_config(cls, dataset_name=None):
        """Get dataset configuration (alias for get_stage_config for compatibility)"""
        if dataset_name is None:
            # Return first enabled stage as default
            enabled_stages = cls.get_enabled_stages()
            if enabled_stages:
                return enabled_stages[0][1]
            else:
                return cls.STAGES["imagenette"]  # fallback
        return cls.get_stage_config(dataset_name)
    
    @classmethod
    def enable_stage(cls, stage_name):
        """Enable a specific stage"""
        if stage_name in cls.STAGES:
            cls.STAGES[stage_name]["enabled"] = True
        else:
            raise ValueError(f"Unknown stage: {stage_name}")
    
    @classmethod
    def disable_stage(cls, stage_name):
        """Disable a specific stage"""
        if stage_name in cls.STAGES:
            cls.STAGES[stage_name]["enabled"] = False
        else:
            raise ValueError(f"Unknown stage: {stage_name}")
    
    @classmethod
    def add_custom_stage(cls, stage_name, config):
        """Add a custom training stage"""
        cls.STAGES[stage_name] = config
        print(f"Custom stage '{stage_name}' added to stages")
    
    @classmethod
    def modify_stage_config(cls, stage_name, **kwargs):
        """Modify configuration for specific stage"""
        if stage_name in cls.STAGES:
            cls.STAGES[stage_name].update(kwargs)
        else:
            raise ValueError(f"Unknown stage: {stage_name}")
    
    @classmethod
    def update_for_dataset(cls, dataset_name):
        """Update configuration for a specific dataset (for backward compatibility)"""
        if dataset_name not in cls.STAGES:
            print(f"⚠️  Unknown dataset: {dataset_name}, using default configuration")
            raise TypeError(f"⚠️  Unknown dataset: {dataset_name}, using default configuration") 
        
        stage_config = cls.get_stage_config(dataset_name)
        
        # Update class attributes for backward compatibility with old code
        cls.DATASET_NAME = dataset_name
        cls.IMAGE_SIZE = stage_config.get("image_size")
        cls.NUM_CLASSES = stage_config.get("classes")
        cls.BATCH_SIZE = stage_config.get("batch_size")
        cls.NUM_EPOCHS = stage_config.get("epochs")
        cls.LEARNING_RATE = stage_config.get("lr")
        cls.WEIGHT_DECAY = stage_config.get("weight_decay")
        cls.SEED = 42
        
        print(f"Configuration updated for {dataset_name}")
        print(f"Image size: {cls.IMAGE_SIZE}")
        print(f"Number of classes: {cls.NUM_CLASSES}")
        print(f"Batch size: {cls.BATCH_SIZE}")
        print(f"Epochs: {cls.NUM_EPOCHS}")
        print(f"Learning rate: {cls.LEARNING_RATE}")
    
    @classmethod
    def print_config(cls):
        """Print current configuration"""
        print("Progressive Training Configuration")
        print("=" * 40)
        
        print(f"Device: {cls.DEVICE}")
        print(f"Data Root: {cls.DATA_ROOT}")
        print(f"Save Directory: {cls.SAVE_DIR}")
        
        print("\nEnabled Stages:")
        enabled_stages = cls.get_enabled_stages()
        for stage_name, config in enabled_stages:
            print(f"  {stage_name}: {config['description']}")
            print(f"    Classes: {config['classes']}, Epochs: {config['epochs']}, LR: {config['lr']}")
        
        print(f"\nModel: {cls.MODEL['architecture']}")
        print(f"Mixed Precision: {cls.TRAINING['mixed_precision']}")
        print(f"Early Stopping Patience: {cls.TRAINING['early_stopping_patience']}")

# Preset configurations for different use cases
class PresetConfigs:
    """Preset configurations for different training scenarios"""
    
    @staticmethod
    def quick_experiment():
        """Configuration for quick experiments"""
        config = ProgressiveConfig()
        
        # Reduce epochs for quick testing
        config.modify_stage_config("imagenette", epochs=5)
        config.modify_stage_config("tiny_imagenet", epochs=10)
        config.modify_stage_config("imagenet_mini", epochs=15)
        config.modify_stage_config("imagenet", epochs=20)
        
        # Reduce batch sizes for memory efficiency
        config.modify_stage_config("imagenette", batch_size=32)
        config.modify_stage_config("tiny_imagenet", batch_size=64)
        config.modify_stage_config("imagenet_mini", batch_size=48)
        config.modify_stage_config("imagenet", batch_size=64)
        
        return config
    
    @staticmethod
    def high_accuracy():
        """Configuration optimized for maximum accuracy"""
        config = ProgressiveConfig()
        
        # Increase epochs for better convergence
        config.modify_stage_config("imagenette", epochs=30)
        config.modify_stage_config("tiny_imagenet", epochs=50)
        config.modify_stage_config("imagenet_mini", epochs=60)
        config.modify_stage_config("imagenet", epochs=90)
        
        # Use larger batch sizes
        config.modify_stage_config("imagenette", batch_size=128)
        config.modify_stage_config("tiny_imagenet", batch_size=256)
        config.modify_stage_config("imagenet_mini", batch_size=192)
        config.modify_stage_config("imagenet", batch_size=256)
        
        # Enable additional regularization
        config.modify_stage_config("imagenette", weight_decay=1e-3)
        config.modify_stage_config("tiny_imagenet", weight_decay=1e-3)
        config.modify_stage_config("imagenet_mini", weight_decay=1e-3)
        config.modify_stage_config("imagenet", weight_decay=1e-3)
        
        return config
    
    @staticmethod
    def memory_efficient():
        """Configuration for memory-constrained environments"""
        config = ProgressiveConfig()
        
        # Small batch sizes
        config.modify_stage_config("imagenette", batch_size=16)
        config.modify_stage_config("tiny_imagenet", batch_size=32)
        config.modify_stage_config("imagenet_mini", batch_size=24)
        config.modify_stage_config("imagenet", batch_size=32)
        
        # Reduce data loading workers
        config.DATA_LOADING["num_workers"] = 2
        config.DATA_LOADING["prefetch_factor"] = 1
        
        # Enable gradient accumulation
        config.TRAINING["gradient_accumulation_steps"] = 4
        
        return config
    
    @staticmethod
    def fast_training():
        """Configuration optimized for speed"""
        config = ProgressiveConfig()
        
        # Reduce epochs
        config.modify_stage_config("imagenette", epochs=10)
        config.modify_stage_config("tiny_imagenet", epochs=20)
        config.modify_stage_config("imagenet_mini", epochs=25)
        config.modify_stage_config("imagenet", epochs=40)
        
        # Larger batch sizes for faster training
        config.modify_stage_config("imagenette", batch_size=128)
        config.modify_stage_config("tiny_imagenet", batch_size=256)
        config.modify_stage_config("imagenet_mini", batch_size=192)
        config.modify_stage_config("imagenet", batch_size=256)
        
        # Higher learning rates
        config.modify_stage_config("imagenette", lr=0.002)
        config.modify_stage_config("tiny_imagenet", lr=0.001)
        config.modify_stage_config("imagenet_mini", lr=0.0006)
        config.modify_stage_config("imagenet", lr=0.2)
        
        return config

# Example usage
if __name__ == "__main__":
    # Print default configuration
    config = ProgressiveConfig()
    config.print_config()
    
    print("\n" + "="*50)
    print("Preset Configurations Available:")
    print("1. ProgressiveConfig() - Default configuration")
    print("2. PresetConfigs.quick_experiment() - Quick testing")
    print("3. PresetConfigs.high_accuracy() - Maximum accuracy")
    print("4. PresetConfigs.memory_efficient() - Memory constrained")
    print("5. PresetConfigs.fast_training() - Speed optimized")
    
    # Example of using presets
    print("\nExample: Using quick experiment preset")
    quick_config = PresetConfigs.quick_experiment()
    quick_config.print_config()

# Alias for backwards compatibility
Config = ProgressiveConfig