"""
Enhanced Progressive Training Strategy with Advanced Optimizers and Schedulers
Integrates cutting-edge optimization techniques for maximum efficiency
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import time
import json
import os
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np

from advanced_optimizer_scheduler import AdvancedTrainingStrategy
from progressive_training_strategy import ProgressiveTrainingStrategy, DatasetManager

class EnhancedProgressiveTrainingStrategy(ProgressiveTrainingStrategy):
    """Enhanced progressive training with advanced optimizers and schedulers"""
    
    def __init__(self, base_model, device, save_dir="./enhanced_models"):
        super().__init__(base_model, device, save_dir)
        
        # Initialize advanced strategy
        self.advanced_strategy = AdvancedTrainingStrategy()
        
        # Enhanced training configurations
        self.enhanced_stages = {
            "imagenette": {
                "dataset": "imagenette",
                "classes": 10,
                "image_size": 224,
                "epochs": 20,
                "batch_size": 64,
                "lr": 0.002,
                "optimizer": "adamw",
                "scheduler": "one_cycle",
                "weight_decay": 1e-5,
                "label_smoothing": 0.1,
                "description": "Super-convergence with One Cycle LR",
                "enabled": True,
                "priority": 1,
                "mixed_precision": True,
                "gradient_clipping": 1.0
            },
            "tiny_imagenet": {
                "dataset": "tiny_imagenet",
                "classes": 200,
                "image_size": 64,
                "epochs": 30,
                "batch_size": 128,
                "lr": 0.001,
                "optimizer": "adamw",
                "scheduler": "cosine_warmup",
                "weight_decay": 1e-4,
                "label_smoothing": 0.1,
                "description": "Balanced approach with Cosine Warmup",
                "enabled": True,
                "priority": 2,
                "mixed_precision": True,
                "gradient_clipping": 1.0
            },
            "imagenet_mini": {
                "dataset": "imagenet_mini",
                "classes": 1000,
                "image_size": 224,
                "epochs": 40,
                "batch_size": 96,
                "lr": 0.0005,
                "optimizer": "sgd_momentum",
                "scheduler": "polynomial",
                "weight_decay": 1e-4,
                "label_smoothing": 0.1,
                "description": "Conservative SGD with Polynomial Decay",
                "enabled": True,
                "priority": 3,
                "mixed_precision": True,
                "gradient_clipping": 1.0
            },
            "imagenet": {
                "dataset": "imagenet",
                "classes": 1000,
                "image_size": 224,
                "epochs": 60,
                "batch_size": 128,
                "lr": 0.1,
                "optimizer": "sgd_momentum",
                "scheduler": "exponential_warmup",
                "weight_decay": 1e-4,
                "label_smoothing": 0.1,
                "description": "Aggressive SGD with Exponential Warmup",
                "enabled": True,
                "priority": 4,
                "mixed_precision": True,
                "gradient_clipping": 1.0
            }
        }
        
        # Update stages with enhanced configurations
        self.stages = self.enhanced_stages
    
    def get_dataset_config(self, stage_name):
        """Get enhanced dataset configuration"""
        return self.enhanced_stages[stage_name]
    
    def create_model_for_stage(self, stage_name, pretrained_weights=None):
        """Create enhanced model for specific stage"""
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
    
    def get_optimizer_and_scheduler(self, model, stage_name):
        """Get advanced optimizer and scheduler"""
        return self.advanced_strategy.get_optimizer_and_scheduler(model, stage_name)
    
    def train_stage(self, stage_name, train_loader, val_loader, pretrained_weights=None):
        """Enhanced training for a specific stage"""
        print(f"\n{'='*60}")
        print(f"Enhanced Training Stage: {stage_name.upper()}")
        print(f"Description: {self.enhanced_stages[stage_name]['description']}")
        print(f"{'='*60}")
        
        # Create model for this stage
        model = self.create_model_for_stage(stage_name, pretrained_weights)
        
        # Get advanced optimizer and scheduler
        optimizer, scheduler = self.get_optimizer_and_scheduler(model, stage_name)
        
        # Enhanced criterion with label smoothing
        config = self.get_dataset_config(stage_name)
        criterion = nn.CrossEntropyLoss(label_smoothing=config["label_smoothing"])
        
        # Mixed precision scaler
        scaler = torch.cuda.amp.GradScaler() if config.get("mixed_precision", False) else None
        
        # Training metrics
        train_losses = []
        val_losses = []
        train_accs = []
        val_accs = []
        
        best_val_acc = 0.0
        best_val_loss = float('inf')
        
        print(f"Training for {config['epochs']} epochs...")
        print(f"Batch size: {config['batch_size']}")
        print(f"Learning rate: {config['lr']}")
        print(f"Optimizer: {config['optimizer']}")
        print(f"Scheduler: {config['scheduler']}")
        print(f"Mixed Precision: {config.get('mixed_precision', False)}")
        print(f"Gradient Clipping: {config.get('gradient_clipping', 0)}")
        
        start_time = time.time()
        
        for epoch in range(config["epochs"]):
            # Training phase with mixed precision
            model.train()
            train_loss = 0.0
            train_correct = 0
            train_total = 0
            
            train_pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{config["epochs"]}')
            for batch_idx, (data, target) in enumerate(train_pbar):
                data, target = data.to(self.device), target.to(self.device)
                
                optimizer.zero_grad()
                
                if scaler is not None:
                    # Mixed precision training
                    with torch.cuda.amp.autocast():
                        output = model(data)
                        loss = criterion(output, target)
                    
                    scaler.scale(loss).backward()
                    
                    # Gradient clipping
                    if config.get("gradient_clipping", 0) > 0:
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(model.parameters(), config["gradient_clipping"])
                    
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    # Standard training
                    output = model(data)
                    loss = criterion(output, target)
                    loss.backward()
                    
                    # Gradient clipping
                    if config.get("gradient_clipping", 0) > 0:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), config["gradient_clipping"])
                    
                    optimizer.step()
                
                train_loss += loss.item()
                _, predicted = output.max(1)
                train_total += target.size(0)
                train_correct += predicted.eq(target).sum().item()
                
                train_pbar.set_postfix({
                    'Loss': f'{loss.item():.4f}',
                    'Acc': f'{100.*train_correct/train_total:.2f}%',
                    'LR': f'{optimizer.param_groups[0]["lr"]:.6f}'
                })
            
            # Validation phase
            model.eval()
            val_loss = 0.0
            val_correct = 0
            val_total = 0
            
            with torch.no_grad():
                for data, target in val_loader:
                    data, target = data.to(self.device), target.to(self.device)
                    
                    if scaler is not None:
                        with torch.cuda.amp.autocast():
                            output = model(data)
                            loss = criterion(output, target)
                    else:
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
                best_val_loss = val_loss
                torch.save(model.state_dict(), 
                          os.path.join(self.save_dir, f'best_{stage_name}.pth'))
            
            print(f'Epoch {epoch+1}: Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, '
                  f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%, '
                  f'LR: {optimizer.param_groups[0]["lr"]:.6f}')
        
        training_time = time.time() - start_time
        
        # Save final model
        torch.save(model.state_dict(), 
                  os.path.join(self.save_dir, f'final_{stage_name}.pth'))
        
        # Store enhanced training history
        self.training_history[stage_name] = {
            'train_losses': train_losses,
            'val_losses': val_losses,
            'train_accs': train_accs,
            'val_accs': val_accs,
            'best_val_acc': best_val_acc,
            'best_val_loss': best_val_loss,
            'training_time': training_time,
            'config': config,
            'optimizer': config['optimizer'],
            'scheduler': config['scheduler']
        }
        
        print(f"\nEnhanced stage {stage_name} completed!")
        print(f"Best validation accuracy: {best_val_acc:.2f}%")
        print(f"Best validation loss: {best_val_loss:.4f}")
        print(f"Training time: {training_time/60:.2f} minutes")
        
        return model.state_dict()
    
    def generate_enhanced_report(self):
        """Generate enhanced training report with optimizer/scheduler analysis"""
        report_file = os.path.join(self.save_dir, 'enhanced_training_report.txt')
        
        with open(report_file, 'w') as f:
            f.write("Enhanced Progressive ImageNet Training Report\n")
            f.write("=" * 50 + "\n\n")
            
            total_time = sum(data['training_time'] for data in self.training_history.values())
            f.write(f"Total Training Time: {total_time/3600:.2f} hours\n\n")
            
            f.write("Advanced Optimizer and Scheduler Analysis:\n")
            f.write("-" * 40 + "\n")
            
            for stage, data in self.training_history.items():
                f.write(f"\nStage: {stage.upper()}\n")
                f.write(f"Description: {data['config']['description']}\n")
                f.write(f"Optimizer: {data['optimizer']}\n")
                f.write(f"Scheduler: {data['scheduler']}\n")
                f.write(f"Classes: {data['config']['classes']}\n")
                f.write(f"Image Size: {data['config']['image_size']}\n")
                f.write(f"Epochs: {data['config']['epochs']}\n")
                f.write(f"Batch Size: {data['config']['batch_size']}\n")
                f.write(f"Learning Rate: {data['config']['lr']}\n")
                f.write(f"Mixed Precision: {data['config'].get('mixed_precision', False)}\n")
                f.write(f"Gradient Clipping: {data['config'].get('gradient_clipping', 0)}\n")
                f.write(f"Best Validation Accuracy: {data['best_val_acc']:.2f}%\n")
                f.write(f"Best Validation Loss: {data['best_val_loss']:.4f}\n")
                f.write(f"Training Time: {data['training_time']/60:.2f} minutes\n")
                f.write("-" * 30 + "\n\n")
            
            f.write("Strategy Benefits:\n")
            f.write("✓ One Cycle LR: Super-convergence for small datasets\n")
            f.write("✓ Cosine Warmup: Smooth convergence with restarts\n")
            f.write("✓ Polynomial Decay: Gradual learning rate reduction\n")
            f.write("✓ Exponential Warmup: Conservative approach for large datasets\n")
            f.write("✓ Mixed Precision: Faster training with lower memory usage\n")
            f.write("✓ Gradient Clipping: Stable training with large learning rates\n")
            f.write("✓ Label Smoothing: Better generalization and robustness\n")
        
        print(f"Enhanced training report saved to: {report_file}")
    
    def plot_enhanced_metrics(self):
        """Plot enhanced training metrics with optimizer/scheduler info"""
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Enhanced Progressive Training Metrics', fontsize=16)
        
        # Plot 1: Training Loss
        ax1 = axes[0, 0]
        for stage, data in self.training_history.items():
            ax1.plot(data['train_losses'], label=f'{stage} (train)')
            ax1.plot(data['val_losses'], '--', label=f'{stage} (val)')
        ax1.set_title("Training and Validation Loss")
        ax1.set_xlabel("Epoch")
        ax1.set_ylabel("Loss")
        ax1.legend()
        ax1.grid(True)
        
        # Plot 2: Training Accuracy
        ax2 = axes[0, 1]
        for stage, data in self.training_history.items():
            ax2.plot(data['train_accs'], label=f'{stage} (train)')
            ax2.plot(data['val_accs'], '--', label=f'{stage} (val)')
        ax2.set_title("Training and Validation Accuracy")
        ax2.set_xlabel("Epoch")
        ax2.set_ylabel("Accuracy (%)")
        ax2.legend()
        ax2.grid(True)
        
        # Plot 3: Best Validation Accuracy per Stage
        ax3 = axes[0, 2]
        stages = list(self.training_history.keys())
        best_accs = [self.training_history[stage]['best_val_acc'] for stage in stages]
        colors = ['skyblue', 'lightgreen', 'orange', 'red']
        bars = ax3.bar(stages, best_accs, color=colors)
        ax3.set_title("Best Validation Accuracy per Stage")
        ax3.set_ylabel("Best Accuracy (%)")
        ax3.set_ylim(0, 100)
        
        # Add value labels and optimizer/scheduler info
        for i, (bar, stage) in enumerate(zip(bars, stages)):
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2., height + 1,
                    f'{height:.1f}%\n{self.training_history[stage]["optimizer"]}\n{self.training_history[stage]["scheduler"]}',
                    ha='center', va='bottom', fontsize=8)
        
        # Plot 4: Training Time per Stage
        ax4 = axes[1, 0]
        training_times = [self.training_history[stage]['training_time']/60 for stage in stages]
        bars = ax4.bar(stages, training_times, color=colors)
        ax4.set_title("Training Time per Stage")
        ax4.set_ylabel("Time (minutes)")
        
        for i, (bar, stage) in enumerate(zip(bars, stages)):
            height = bar.get_height()
            ax4.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                    f'{height:.1f}m\n{self.training_history[stage]["optimizer"]}',
                    ha='center', va='bottom', fontsize=8)
        
        # Plot 5: Learning Rate Schedules
        ax5 = axes[1, 1]
        for stage, data in self.training_history.items():
            config = data['config']
            epochs = config['epochs']
            base_lr = config['lr']
            
            # Create dummy optimizer and scheduler for visualization
            dummy_optimizer = optim.AdamW([torch.tensor(1.0)], lr=base_lr)
            scheduler = self.advanced_strategy.scheduler_strategy.get_scheduler(
                dummy_optimizer, stage, config
            )
            
            lrs = []
            for epoch in range(epochs):
                lrs.append(scheduler.get_last_lr()[0])
                scheduler.step()
            
            ax5.plot(range(epochs), lrs, label=f'{stage} ({config["scheduler"]})')
        
        ax5.set_title("Learning Rate Schedules")
        ax5.set_xlabel("Epoch")
        ax5.set_ylabel("Learning Rate")
        ax5.set_yscale('log')
        ax5.legend()
        ax5.grid(True)
        
        # Plot 6: Convergence Speed Comparison
        ax6 = axes[1, 2]
        for stage, data in self.training_history.items():
            # Calculate convergence speed (epochs to reach 90% of best accuracy)
            best_acc = data['best_val_acc']
            target_acc = best_acc * 0.9
            
            convergence_epoch = None
            for epoch, acc in enumerate(data['val_accs']):
                if acc >= target_acc:
                    convergence_epoch = epoch + 1
                    break
            
            if convergence_epoch:
                ax6.bar(stage, convergence_epoch, color=colors[stages.index(stage)])
                ax6.text(stages.index(stage), convergence_epoch + 0.5,
                        f'{convergence_epoch} epochs\n{data["scheduler"]}',
                        ha='center', va='bottom', fontsize=8)
        
        ax6.set_title("Convergence Speed (90% of Best Acc)")
        ax6.set_ylabel("Epochs to Convergence")
        ax6.set_ylim(0, max([len(data['val_accs']) for data in self.training_history.values()]))
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.save_dir, 'enhanced_training_metrics.png'), 
                   dpi=300, bbox_inches='tight')
        plt.show()

# Example usage
def run_enhanced_progressive_training():
    """Run enhanced progressive training with advanced optimizers and schedulers"""
    print("Enhanced Progressive ImageNet Training")
    print("=" * 50)
    
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create base model (simplified ResNet50)
    class ResNet50(nn.Module):
        def __init__(self, num_classes=1000):
            super(ResNet50, self).__init__()
            self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
            self.bn1 = nn.BatchNorm2d(64)
            self.relu = nn.ReLU(inplace=True)
            self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
            
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
    
    # Initialize enhanced strategy
    base_model = ResNet50()
    strategy = EnhancedProgressiveTrainingStrategy(base_model, device)
    
    # Initialize dataset manager
    dataset_manager = DatasetManager()
    
    # Load available datasets
    print("\nLoading datasets...")
    dataset_loaders = dataset_manager.load_all_datasets()
    
    if not dataset_loaders:
        print("No datasets found. Please ensure datasets are available.")
        return
    
    # Execute enhanced progressive training
    print(f"\nStarting enhanced progressive training with {len(dataset_loaders)} datasets...")
    training_history = strategy.progressive_train(dataset_loaders)
    
    # Generate enhanced reports and visualizations
    print("\nGenerating enhanced reports and visualizations...")
    strategy.plot_enhanced_metrics()
    strategy.generate_enhanced_report()
    
    print("\nEnhanced progressive training completed!")
    print("Check the './enhanced_models' directory for results.")

if __name__ == "__main__":
    run_enhanced_progressive_training()
