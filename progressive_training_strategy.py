"""
Progressive ImageNet Training Strategy
Multi-stage training from smaller datasets to full ImageNet for optimal accuracy and speed
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torchvision import datasets
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import json
import time

class ProgressiveTrainingStrategy:
    """
    Progressive training strategy that scales from smaller datasets to full ImageNet
    """
    
    def __init__(self, base_model, device, save_dir="./progressive_models"):
        self.base_model = base_model
        self.device = device
        self.save_dir = save_dir
        self.training_history = {}
        
        # Create save directory
        os.makedirs(save_dir, exist_ok=True)
        
        # Define training stages
        self.stages = {
            "imagenette": {
                "dataset": "imagenette",
                "classes": 10,
                "image_size": 224,
                "epochs": 20,
                "batch_size": 64,
                "lr": 0.001,
                "optimizer": "adamw",
                "scheduler": "cosine",
                "description": "Quick warmup and architecture validation"
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
                "description": "Medium complexity training"
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
                "description": "Full ImageNet complexity with subset data"
            },
            "imagenet_full": {
                "dataset": "imagenet",
                "classes": 1000,
                "image_size": 224,
                "epochs": 60,
                "batch_size": 128,
                "lr": 0.1,
                "optimizer": "sgd",
                "scheduler": "step",
                "description": "Final full-scale training"
            }
        }
    
    def get_dataset_config(self, stage_name):
        """Get configuration for specific training stage"""
        return self.stages[stage_name]
    
    def create_model_for_stage(self, stage_name, pretrained_weights=None):
        """Create model for specific stage with appropriate final layer"""
        config = self.get_dataset_config(stage_name)
        
        # Clone the base model
        model = type(self.base_model)(**self.base_model.__dict__)
        
        # Modify final layer for number of classes
        if hasattr(model, 'fc'):
            model.fc = nn.Linear(model.fc.in_features, config["classes"])
        elif hasattr(model, 'classifier'):
            model.classifier = nn.Linear(model.classifier.in_features, config["classes"])
        
        # Load pretrained weights if provided
        if pretrained_weights:
            model.load_state_dict(pretrained_weights, strict=False)
            print(f"Loaded pretrained weights for {stage_name}")
        
        return model.to(self.device)
    
    def get_optimizer(self, model, stage_name):
        """Get optimizer for specific stage"""
        config = self.get_dataset_config(stage_name)
        
        if config["optimizer"] == "adamw":
            return optim.AdamW(
                model.parameters(),
                lr=config["lr"],
                weight_decay=1e-4
            )
        elif config["optimizer"] == "sgd":
            return optim.SGD(
                model.parameters(),
                lr=config["lr"],
                weight_decay=1e-4,
                momentum=0.9
            )
        else:
            raise ValueError(f"Unknown optimizer: {config['optimizer']}")
    
    def get_scheduler(self, optimizer, stage_name):
        """Get learning rate scheduler for specific stage"""
        config = self.get_dataset_config(stage_name)
        
        if config["scheduler"] == "cosine":
            return optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=config["epochs"],
                eta_min=config["lr"] * 0.01
            )
        elif config["scheduler"] == "step":
            return optim.lr_scheduler.StepLR(
                optimizer,
                step_size=config["epochs"] // 3,
                gamma=0.1
            )
        else:
            raise ValueError(f"Unknown scheduler: {config['scheduler']}")
    
    def train_stage(self, stage_name, train_loader, val_loader, pretrained_weights=None):
        """Train model for a specific stage"""
        print(f"\n{'='*60}")
        print(f"Training Stage: {stage_name.upper()}")
        print(f"Description: {self.stages[stage_name]['description']}")
        print(f"{'='*60}")
        
        # Create model for this stage
        model = self.create_model_for_stage(stage_name, pretrained_weights)
        
        # Get optimizer and scheduler
        optimizer = self.get_optimizer(model, stage_name)
        scheduler = self.get_scheduler(optimizer, stage_name)
        criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
        
        # Training metrics
        train_losses = []
        val_losses = []
        train_accs = []
        val_accs = []
        
        config = self.get_dataset_config(stage_name)
        best_val_acc = 0.0
        
        print(f"Training for {config['epochs']} epochs...")
        print(f"Batch size: {config['batch_size']}")
        print(f"Learning rate: {config['lr']}")
        print(f"Optimizer: {config['optimizer']}")
        print(f"Scheduler: {config['scheduler']}")
        
        start_time = time.time()
        
        for epoch in range(config["epochs"]):
            # Training phase
            model.train()
            train_loss = 0.0
            train_correct = 0
            train_total = 0
            
            train_pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{config["epochs"]}')
            for batch_idx, (data, target) in enumerate(train_pbar):
                data, target = data.to(self.device), target.to(self.device)
                
                optimizer.zero_grad()
                output = model(data)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
                _, predicted = output.max(1)
                train_total += target.size(0)
                train_correct += predicted.eq(target).sum().item()
                
                train_pbar.set_postfix({
                    'Loss': f'{loss.item():.4f}',
                    'Acc': f'{100.*train_correct/train_total:.2f}%'
                })
            
            # Validation phase
            model.eval()
            val_loss = 0.0
            val_correct = 0
            val_total = 0
            
            with torch.no_grad():
                for data, target in val_loader:
                    data, target = data.to(self.device), target.to(self.device)
                    output = model(data)
                    loss = criterion(output, target)
                    
                    val_loss += loss.item()
                    _, predicted = output.max(1)
                    val_total += target.size(0)
                    val_correct += predicted.eq(target).sum().item()
            
            # Calculate metrics
            train_loss /= len(train_loader)
            val_loss /= len(val_loader)
            train_acc = 100. * train_correct / train_total
            val_acc = 100. * val_correct / val_total
            
            train_losses.append(train_loss)
            val_losses.append(val_loss)
            train_accs.append(train_acc)
            val_accs.append(val_acc)
            
            # Update scheduler
            scheduler.step()
            
            # Save best model
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save(model.state_dict(), 
                          os.path.join(self.save_dir, f'best_{stage_name}.pth'))
            
            print(f'Epoch {epoch+1}: Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, '
                  f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%, '
                  f'LR: {optimizer.param_groups[0]["lr"]:.6f}')
        
        training_time = time.time() - start_time
        
        # Save final model
        torch.save(model.state_dict(), 
                  os.path.join(self.save_dir, f'final_{stage_name}.pth'))
        
        # Store training history
        self.training_history[stage_name] = {
            'train_losses': train_losses,
            'val_losses': val_losses,
            'train_accs': train_accs,
            'val_accs': val_accs,
            'best_val_acc': best_val_acc,
            'training_time': training_time,
            'config': config
        }
        
        print(f"\nStage {stage_name} completed!")
        print(f"Best validation accuracy: {best_val_acc:.2f}%")
        print(f"Training time: {training_time/60:.2f} minutes")
        
        return model.state_dict()
    
    def progressive_train(self, dataset_loaders):
        """Execute progressive training across all stages"""
        print("Starting Progressive ImageNet Training Strategy")
        print("=" * 60)
        
        pretrained_weights = None
        
        for stage_name in self.stages.keys():
            if stage_name in dataset_loaders:
                train_loader, val_loader = dataset_loaders[stage_name]
                
                # Train this stage
                stage_weights = self.train_stage(
                    stage_name, 
                    train_loader, 
                    val_loader, 
                    pretrained_weights
                )
                
                # Use weights from this stage for next stage
                pretrained_weights = stage_weights
                
                # Save intermediate results
                self.save_training_history()
            else:
                print(f"Warning: Dataset {stage_name} not found in dataset_loaders")
        
        print("\n" + "=" * 60)
        print("Progressive Training Complete!")
        print("=" * 60)
        
        return self.training_history
    
    def save_training_history(self):
        """Save training history to JSON file"""
        history_file = os.path.join(self.save_dir, 'training_history.json')
        
        # Convert numpy arrays to lists for JSON serialization
        serializable_history = {}
        for stage, data in self.training_history.items():
            serializable_history[stage] = {
                'train_losses': data['train_losses'],
                'val_losses': data['val_losses'],
                'train_accs': data['train_accs'],
                'val_accs': data['val_accs'],
                'best_val_acc': data['best_val_acc'],
                'training_time': data['training_time'],
                'config': data['config']
            }
        
        with open(history_file, 'w') as f:
            json.dump(serializable_history, f, indent=2)
    
    def plot_training_progress(self):
        """Plot training progress across all stages"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Progressive Training Progress', fontsize=16)
        
        # Plot 1: Training Loss
        ax1 = axes[0, 0]
        for stage, data in self.training_history.items():
            ax1.plot(data['train_losses'], label=f'{stage} (train)')
            ax1.plot(data['val_losses'], '--', label=f'{stage} (val)')
        ax1.set_title('Training and Validation Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()
        ax1.grid(True)
        
        # Plot 2: Training Accuracy
        ax2 = axes[0, 1]
        for stage, data in self.training_history.items():
            ax2.plot(data['train_accs'], label=f'{stage} (train)')
            ax2.plot(data['val_accs'], '--', label=f'{stage} (val)')
        ax2.set_title('Training and Validation Accuracy')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy (%)')
        ax2.legend()
        ax2.grid(True)
        
        # Plot 3: Best Validation Accuracy per Stage
        ax3 = axes[1, 0]
        stages = list(self.training_history.keys())
        best_accs = [self.training_history[stage]['best_val_acc'] for stage in stages]
        ax3.bar(stages, best_accs, color=['skyblue', 'lightgreen', 'orange', 'red'])
        ax3.set_title('Best Validation Accuracy per Stage')
        ax3.set_ylabel('Best Accuracy (%)')
        ax3.set_ylim(0, 100)
        
        # Add value labels on bars
        for i, v in enumerate(best_accs):
            ax3.text(i, v + 1, f'{v:.1f}%', ha='center', va='bottom')
        
        # Plot 4: Training Time per Stage
        ax4 = axes[1, 1]
        training_times = [self.training_history[stage]['training_time']/60 for stage in stages]
        ax4.bar(stages, training_times, color=['skyblue', 'lightgreen', 'orange', 'red'])
        ax4.set_title('Training Time per Stage')
        ax4.set_ylabel('Time (minutes)')
        
        # Add value labels on bars
        for i, v in enumerate(training_times):
            ax4.text(i, v + 0.5, f'{v:.1f}m', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.save_dir, 'training_progress.png'), dpi=300, bbox_inches='tight')
        plt.show()
    
    def generate_training_report(self):
        """Generate comprehensive training report"""
        report_file = os.path.join(self.save_dir, 'training_report.txt')
        
        with open(report_file, 'w') as f:
            f.write("Progressive ImageNet Training Report\n")
            f.write("=" * 50 + "\n\n")
            
            total_time = sum(data['training_time'] for data in self.training_history.values())
            f.write(f"Total Training Time: {total_time/3600:.2f} hours\n\n")
            
            for stage, data in self.training_history.items():
                f.write(f"Stage: {stage.upper()}\n")
                f.write(f"Description: {data['config']['description']}\n")
                f.write(f"Classes: {data['config']['classes']}\n")
                f.write(f"Image Size: {data['config']['image_size']}\n")
                f.write(f"Epochs: {data['config']['epochs']}\n")
                f.write(f"Batch Size: {data['config']['batch_size']}\n")
                f.write(f"Learning Rate: {data['config']['lr']}\n")
                f.write(f"Optimizer: {data['config']['optimizer']}\n")
                f.write(f"Scheduler: {data['config']['scheduler']}\n")
                f.write(f"Best Validation Accuracy: {data['best_val_acc']:.2f}%\n")
                f.write(f"Training Time: {data['training_time']/60:.2f} minutes\n")
                f.write("-" * 30 + "\n\n")
        
        print(f"Training report saved to: {report_file}")


class DatasetManager:
    """Manages dataset loading for progressive training"""
    
    def __init__(self, data_root="./data"):
        self.data_root = data_root
        self.dataset_loaders = {}
    
    def load_imagenette(self, batch_size=64):
        """Load ImageNette dataset"""
        print("Loading ImageNette dataset...")
        
        transform_train = transforms.Compose([
            transforms.Resize(256),
            transforms.RandomCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ])
        
        transform_val = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ])
        
        train_dataset = datasets.ImageFolder(
            os.path.join(self.data_root, 'imagenette2', 'train'),
            transform=transform_train
        )
        
        val_dataset = datasets.ImageFolder(
            os.path.join(self.data_root, 'imagenette2', 'val'),
            transform=transform_val
        )
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
        
        self.dataset_loaders['imagenette'] = (train_loader, val_loader)
        print(f"ImageNette loaded: {len(train_dataset)} train, {len(val_dataset)} val samples")
        
        return train_loader, val_loader
    
    def load_tiny_imagenet(self, batch_size=128):
        """Load Tiny ImageNet dataset"""
        print("Loading Tiny ImageNet dataset...")
        
        transform_train = transforms.Compose([
            transforms.Resize(64),
            transforms.RandomCrop(64),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ])
        
        transform_val = transforms.Compose([
            transforms.Resize(64),
            transforms.CenterCrop(64),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ])
        
        train_dataset = datasets.ImageFolder(
            os.path.join(self.data_root, 'tiny-imagenet-200', 'train'),
            transform=transform_train
        )
        
        val_dataset = datasets.ImageFolder(
            os.path.join(self.data_root, 'tiny-imagenet-200', 'val'),
            transform=transform_val
        )
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
        
        self.dataset_loaders['tiny_imagenet'] = (train_loader, val_loader)
        print(f"Tiny ImageNet loaded: {len(train_dataset)} train, {len(val_dataset)} val samples")
        
        return train_loader, val_loader
    
    def load_imagenet_mini(self, batch_size=96):
        """Load ImageNet Mini dataset"""
        print("Loading ImageNet Mini dataset...")
        
        transform_train = transforms.Compose([
            transforms.Resize(256),
            transforms.RandomCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ])
        
        transform_val = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ])
        
        train_dataset = datasets.ImageFolder(
            os.path.join(self.data_root, 'imagenet-mini', 'train'),
            transform=transform_train
        )
        
        val_dataset = datasets.ImageFolder(
            os.path.join(self.data_root, 'imagenet-mini', 'val'),
            transform=transform_val
        )
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
        
        self.dataset_loaders['imagenet_mini'] = (train_loader, val_loader)
        print(f"ImageNet Mini loaded: {len(train_dataset)} train, {len(val_dataset)} val samples")
        
        return train_loader, val_loader
    
    def load_imagenet_full(self, batch_size=128):
        """Load full ImageNet dataset"""
        print("Loading Full ImageNet dataset...")
        
        transform_train = transforms.Compose([
            transforms.Resize(256),
            transforms.RandomCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ])
        
        transform_val = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ])
        
        train_dataset = datasets.ImageFolder(
            os.path.join(self.data_root, 'imagenet', 'train'),
            transform=transform_train
        )
        
        val_dataset = datasets.ImageFolder(
            os.path.join(self.data_root, 'imagenet', 'val'),
            transform=transform_val
        )
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=8)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=8)
        
        self.dataset_loaders['imagenet'] = (train_loader, val_loader)
        print(f"Full ImageNet loaded: {len(train_dataset)} train, {len(val_dataset)} val samples")
        
        return train_loader, val_loader
    
    def load_all_datasets(self):
        """Load all datasets for progressive training"""
        print("Loading all datasets for progressive training...")
        
        try:
            self.load_imagenette()
        except Exception as e:
            print(f"Could not load ImageNette: {e}")
        
        try:
            self.load_tiny_imagenet()
        except Exception as e:
            print(f"Could not load Tiny ImageNet: {e}")
        
        try:
            self.load_imagenet_mini()
        except Exception as e:
            print(f"Could not load ImageNet Mini: {e}")
        
        try:
            self.load_imagenet_full()
        except Exception as e:
            print(f"Could not load Full ImageNet: {e}")
        
        print(f"Successfully loaded {len(self.dataset_loaders)} datasets")
        return self.dataset_loaders


# Example usage and main execution
def main():
    """Main execution function"""
    print("Progressive ImageNet Training Strategy")
    print("=" * 50)
    
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Define base model (ResNet50)
    class ResNet50(nn.Module):
        def __init__(self, num_classes=1000):
            super(ResNet50, self).__init__()
            # Simplified ResNet50 architecture
            self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
            self.bn1 = nn.BatchNorm2d(64)
            self.relu = nn.ReLU(inplace=True)
            self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
            
            # ResNet layers (simplified)
            self.layer1 = self._make_layer(64, 64, 3, stride=1)
            self.layer2 = self._make_layer(64, 128, 4, stride=2)
            self.layer3 = self._make_layer(128, 256, 6, stride=2)
            self.layer4 = self._make_layer(256, 512, 3, stride=2)
            
            self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
            self.fc = nn.Linear(512, num_classes)
            
        def _make_layer(self, in_channels, out_channels, blocks, stride):
            layers = []
            layers.append(nn.Conv2d(in_channels, out_channels, 3, stride, 1, bias=False))
            layers.append(nn.BatchNorm2d(out_channels))
            layers.append(nn.ReLU(inplace=True))
            
            for _ in range(1, blocks):
                layers.append(nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False))
                layers.append(nn.BatchNorm2d(out_channels))
                layers.append(nn.ReLU(inplace=True))
            
            return nn.Sequential(*layers)
        
        def forward(self, x):
            x = self.conv1(x)
            x = self.bn1(x)
            x = self.relu(x)
            x = self.maxpool(x)
            
            x = self.layer1(x)
            x = self.layer2(x)
            x = self.layer3(x)
            x = self.layer4(x)
            
            x = self.avgpool(x)
            x = torch.flatten(x, 1)
            x = self.fc(x)
            return x
    
    # Initialize base model
    base_model = ResNet50()
    
    # Initialize progressive training strategy
    strategy = ProgressiveTrainingStrategy(base_model, device)
    
    # Initialize dataset manager
    dataset_manager = DatasetManager()
    
    # Load all available datasets
    dataset_loaders = dataset_manager.load_all_datasets()
    
    if not dataset_loaders:
        print("No datasets loaded. Please ensure datasets are available.")
        return
    
    # Execute progressive training
    training_history = strategy.progressive_train(dataset_loaders)
    
    # Generate reports and visualizations
    strategy.plot_training_progress()
    strategy.generate_training_report()
    
    print("\nProgressive training completed successfully!")
    print("Check the './progressive_models' directory for saved models and reports.")


if __name__ == "__main__":
    main()
