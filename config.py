"""
Configuration file for ImageNet training with ResNet50
"""

import os

class Config:
    """Configuration class for multiple ImageNet variants training"""
    
    # Dataset Configuration
    DATASET_NAME = "imagenette"  # Options: "imagenet", "imagenet_mini", "tiny_imagenet", "imagenette"
    DATA_ROOT = "./data"
    
    # Dataset paths
    IMAGENET_PATH = os.path.join(DATA_ROOT, "imagenet")
    IMAGENET_MINI_PATH = os.path.join(DATA_ROOT, "imagenet-mini")
    TINY_IMAGENET_PATH = os.path.join(DATA_ROOT, "tiny-imagenet-200")
    IMAGENETTE_PATH = os.path.join(DATA_ROOT, "imagenette2")
    
    # Dataset URLs
    IMAGENET_MINI_URL = "https://www.kaggle.com/datasets/ifigotin/imagenetmini-1000/download?datasetVersionNumber=1"
    TINY_IMAGENET_URL = "http://cs231n.stanford.edu/tiny-imagenet-200.zip"
    IMAGENETTE_URL = "https://s3.amazonaws.com/fast-ai-imageclas/imagenette2.tgz"
    
    # Default image dimensions (will be overridden by dataset config)
    IMAGE_SIZE = 224
    NUM_CLASSES = 1000
    
    # Data augmentation settings
    MEAN = (0.485, 0.456, 0.406)  # ImageNet normalization
    STD = (0.229, 0.224, 0.225)
    
    # Training Configuration (will be adjusted per dataset)
    BATCH_SIZE = 128  # Default batch size
    NUM_EPOCHS = 50  # Default epochs
    LEARNING_RATE = 0.001  # Default learning rate
    WEIGHT_DECAY = 1e-4
    
    # Model Configuration
    MODEL_NAME = "resnet50"
    PRETRAINED = False  # Use custom implementation
    
    # Training settings
    NUM_WORKERS = 4  # Default workers
    PIN_MEMORY = True
    
    # Scheduler settings
    SCHEDULER_PATIENCE = 3
    SCHEDULER_FACTOR = 0.5
    SCHEDULER_MIN_LR = 1e-4
    
    # Device settings
    DEVICE = "cuda" if os.environ.get("CUDA_VISIBLE_DEVICES") else "cpu"
    
    # Logging and saving
    SAVE_MODEL_PATH = "./models"
    LOG_INTERVAL = 50
    
    # Colab specific settings
    COLAB_MODE = False  # Set to True for Colab environment
    MOUNT_DRIVE = True
    DRIVE_MODEL_PATH = "/content/drive/MyDrive/imagenet_models"
    
    @classmethod
    def get_dataset_config(cls):
        """Get dataset specific configuration"""
        if cls.DATASET_NAME == "imagenet":
            return {
                "image_size": 224,
                "num_classes": 1000,
                "mean": cls.MEAN,
                "std": cls.STD,
                "batch_size": 256,
                "epochs": 90,
                "lr": 0.1,
                "optimizer": "sgd",
                "scheduler": "step"
            }
        elif cls.DATASET_NAME == "imagenet_mini":
            return {
                "image_size": 224,
                "num_classes": 1000,
                "mean": cls.MEAN,
                "std": cls.STD,
                "batch_size": 128,
                "epochs": 50,
                "lr": 0.001,
                "optimizer": "adamw",
                "scheduler": "reduce_lr"
            }
        elif cls.DATASET_NAME == "tiny_imagenet":
            return {
                "image_size": 64,
                "num_classes": 200,
                "mean": cls.MEAN,
                "std": cls.STD,
                "batch_size": 128,
                "epochs": 50,
                "lr": 0.001,
                "optimizer": "adamw",
                "scheduler": "reduce_lr"
            }
        elif cls.DATASET_NAME == "imagenette":
            return {
                "image_size": 224,
                "num_classes": 10,
                "mean": cls.MEAN,
                "std": cls.STD,
                "batch_size": 64,
                "epochs": 30,
                "lr": 0.001,
                "optimizer": "adamw",
                "scheduler": "reduce_lr"
            }
        else:
            raise ValueError(f"Unknown dataset: {cls.DATASET_NAME}")
    
    @classmethod
    def update_for_dataset(cls, dataset_name):
        """Update configuration for specific dataset"""
        cls.DATASET_NAME = dataset_name
        dataset_config = cls.get_dataset_config()
        
        # Update training parameters
        cls.BATCH_SIZE = dataset_config["batch_size"]
        cls.NUM_EPOCHS = dataset_config["epochs"]
        cls.LEARNING_RATE = dataset_config["lr"]
        
        # Update image size and classes
        cls.IMAGE_SIZE = dataset_config["image_size"]
        cls.NUM_CLASSES = dataset_config["num_classes"]
        
        print(f"Configuration updated for {dataset_name}")
        print(f"Image size: {cls.IMAGE_SIZE}")
        print(f"Number of classes: {cls.NUM_CLASSES}")
        print(f"Batch size: {cls.BATCH_SIZE}")
        print(f"Epochs: {cls.NUM_EPOCHS}")
        print(f"Learning rate: {cls.LEARNING_RATE}")
    
    @classmethod
    def update_for_colab(cls):
        """Update configuration for Colab environment"""
        cls.COLAB_MODE = True
        cls.BATCH_SIZE = min(cls.BATCH_SIZE, 32)  # Reduce batch size for Colab
        cls.NUM_WORKERS = 2
        cls.DATA_ROOT = "/content/data"
        cls.SAVE_MODEL_PATH = "/content/models"
        
        # Create directories if they don't exist
        os.makedirs(cls.DATA_ROOT, exist_ok=True)
        os.makedirs(cls.SAVE_MODEL_PATH, exist_ok=True)
