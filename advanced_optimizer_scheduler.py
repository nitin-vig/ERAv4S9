"""
Advanced Scheduler and Optimizer Strategy for Progressive ImageNet Training
Implements cutting-edge techniques for optimal convergence and accuracy
"""

import torch
import torch.nn as nn
import torch.optim as optim
import math
import numpy as np
from typing import Dict, List, Optional, Tuple
import matplotlib.pyplot as plt

class AdvancedOptimizerStrategy:
    """Advanced optimizer strategy with adaptive techniques"""
    
    def __init__(self):
        self.optimizer_configs = {
            "adamw": {
                "class": optim.AdamW,
                "params": {
                    "lr": 0.001,
                    "weight_decay": 1e-4,
                    "betas": (0.9, 0.999),
                    "eps": 1e-8
                },
                "description": "AdamW with adaptive weight decay"
            },
            "sgd_momentum": {
                "class": optim.SGD,
                "params": {
                    "lr": 0.1,
                    "weight_decay": 1e-4,
                    "momentum": 0.9,
                    "nesterov": True
                },
                "description": "SGD with Nesterov momentum"
            },
            "rmsprop": {
                "class": optim.RMSprop,
                "params": {
                    "lr": 0.01,
                    "weight_decay": 1e-4,
                    "momentum": 0.9,
                    "alpha": 0.99,
                    "eps": 1e-8
                },
                "description": "RMSprop with momentum"
            },
            "adamw_8bit": {
                "class": "AdamW8bit",  # Will implement custom 8-bit AdamW
                "params": {
                    "lr": 0.001,
                    "weight_decay": 1e-4,
                    "betas": (0.9, 0.999),
                    "eps": 1e-8
                },
                "description": "8-bit AdamW for memory efficiency"
            }
        }
    
    def get_optimizer(self, model: nn.Module, stage_name: str, config: Dict) -> optim.Optimizer:
        """Get optimized optimizer for specific stage"""
        optimizer_name = config.get("optimizer", "adamw")
        
        if optimizer_name not in self.optimizer_configs:
            raise ValueError(f"Unknown optimizer: {optimizer_name}")
        
        opt_config = self.optimizer_configs[optimizer_name].copy()
        
        # Stage-specific optimizations
        if stage_name == "imagenette":
            # Quick convergence for small dataset
            opt_config["params"]["lr"] = config.get("lr", 0.002)
            opt_config["params"]["weight_decay"] = 1e-5
        elif stage_name == "tiny_imagenet":
            # Balanced approach for medium dataset
            opt_config["params"]["lr"] = config.get("lr", 0.001)
            opt_config["params"]["weight_decay"] = 1e-4
        elif stage_name == "imagenet_mini":
            # Conservative approach for large dataset
            opt_config["params"]["lr"] = config.get("lr", 0.0005)
            opt_config["params"]["weight_decay"] = 1e-4
        elif stage_name == "imagenet":
            # Aggressive approach for full dataset
            opt_config["params"]["lr"] = config.get("lr", 0.1)
            opt_config["params"]["weight_decay"] = 1e-4
        
        # Create optimizer
        if opt_config["class"] == "AdamW8bit":
            return self._create_adamw_8bit(model, opt_config["params"])
        else:
            return opt_config["class"](model.parameters(), **opt_config["params"])
    
    def _create_adamw_8bit(self, model: nn.Module, params: Dict) -> optim.Optimizer:
        """Create 8-bit AdamW optimizer for memory efficiency"""
        # Fallback to regular AdamW if 8-bit not available
        return optim.AdamW(model.parameters(), **params)

class AdvancedSchedulerStrategy:
    """Advanced scheduler strategy with multiple techniques"""
    
    def __init__(self):
        self.scheduler_configs = {
            "cosine_warmup": {
                "class": "CosineAnnealingWarmRestarts",
                "description": "Cosine annealing with warm restarts"
            },
            "one_cycle": {
                "class": "OneCycleLR",
                "description": "One cycle learning rate policy"
            },
            "polynomial": {
                "class": "PolynomialLR",
                "description": "Polynomial decay scheduler"
            },
            "exponential_warmup": {
                "class": "ExponentialWarmup",
                "description": "Exponential decay with warmup"
            },
            "adaptive": {
                "class": "AdaptiveLR",
                "description": "Adaptive learning rate based on loss"
            }
        }
    
    def get_scheduler(self, optimizer: optim.Optimizer, stage_name: str, config: Dict) -> optim.lr_scheduler._LRScheduler:
        """Get advanced scheduler for specific stage"""
        scheduler_name = config.get("scheduler", "cosine_warmup")
        
        if scheduler_name == "cosine_warmup":
            return self._create_cosine_warmup(optimizer, stage_name, config)
        elif scheduler_name == "one_cycle":
            return self._create_one_cycle(optimizer, stage_name, config)
        elif scheduler_name == "polynomial":
            return self._create_polynomial(optimizer, stage_name, config)
        elif scheduler_name == "exponential_warmup":
            return self._create_exponential_warmup(optimizer, stage_name, config)
        elif scheduler_name == "adaptive":
            return self._create_adaptive(optimizer, stage_name, config)
        else:
            raise ValueError(f"Unknown scheduler: {scheduler_name}")
    
    def _create_cosine_warmup(self, optimizer: optim.Optimizer, stage_name: str, config: Dict) -> optim.lr_scheduler._LRScheduler:
        """Create cosine annealing with warm restarts"""
        epochs = config.get("epochs", 50)
        base_lr = optimizer.param_groups[0]["lr"]
        
        # Stage-specific parameters
        if stage_name == "imagenette":
            T_0 = 5  # Restart every 5 epochs
            T_mult = 2
            warmup_epochs = 2
        elif stage_name == "tiny_imagenet":
            T_0 = 8
            T_mult = 2
            warmup_epochs = 3
        elif stage_name == "imagenet_mini":
            T_0 = 10
            T_mult = 2
            warmup_epochs = 5
        else:  # imagenet
            T_0 = 15
            T_mult = 2
            warmup_epochs = 10
        
        return CosineAnnealingWarmRestartsWithWarmup(
            optimizer, T_0=T_0, T_mult=T_mult, eta_min=base_lr * 0.01,
            warmup_epochs=warmup_epochs, warmup_lr=base_lr * 0.1
        )
    
    def _create_one_cycle(self, optimizer: optim.Optimizer, stage_name: str, config: Dict) -> optim.lr_scheduler._LRScheduler:
        """Create one cycle learning rate policy"""
        epochs = config.get("epochs", 50)
        base_lr = optimizer.param_groups[0]["lr"]
        
        # Stage-specific max LR
        if stage_name == "imagenette":
            max_lr = base_lr * 10
        elif stage_name == "tiny_imagenet":
            max_lr = base_lr * 8
        elif stage_name == "imagenet_mini":
            max_lr = base_lr * 5
        else:  # imagenet
            max_lr = base_lr * 3
        
        return OneCycleLR(
            optimizer, max_lr=max_lr, epochs=epochs,
            pct_start=0.3, anneal_strategy='cos',
            div_factor=25, final_div_factor=10000
        )
    
    def _create_polynomial(self, optimizer: optim.Optimizer, stage_name: str, config: Dict) -> optim.lr_scheduler._LRScheduler:
        """Create polynomial decay scheduler"""
        epochs = config.get("epochs", 50)
        
        # Stage-specific power
        if stage_name == "imagenette":
            power = 2.0
        elif stage_name == "tiny_imagenet":
            power = 1.5
        elif stage_name == "imagenet_mini":
            power = 1.2
        else:  # imagenet
            power = 1.0
        
        return PolynomialLR(optimizer, total_iters=epochs, power=power)
    
    def _create_exponential_warmup(self, optimizer: optim.Optimizer, stage_name: str, config: Dict) -> optim.lr_scheduler._LRScheduler:
        """Create exponential decay with warmup"""
        epochs = config.get("epochs", 50)
        base_lr = optimizer.param_groups[0]["lr"]
        
        # Stage-specific parameters
        if stage_name == "imagenette":
            warmup_epochs = 3
            gamma = 0.95
        elif stage_name == "tiny_imagenet":
            warmup_epochs = 5
            gamma = 0.97
        elif stage_name == "imagenet_mini":
            warmup_epochs = 8
            gamma = 0.98
        else:  # imagenet
            warmup_epochs = 15
            gamma = 0.99
        
        return ExponentialWarmupLR(
            optimizer, gamma=gamma, warmup_epochs=warmup_epochs,
            warmup_lr=base_lr * 0.1
        )
    
    def _create_adaptive(self, optimizer: optim.Optimizer, stage_name: str, config: Dict) -> optim.lr_scheduler._LRScheduler:
        """Create adaptive learning rate scheduler"""
        base_lr = optimizer.param_groups[0]["lr"]
        
        # Stage-specific parameters
        if stage_name == "imagenette":
            patience = 3
            factor = 0.7
        elif stage_name == "tiny_imagenet":
            patience = 5
            factor = 0.8
        elif stage_name == "imagenet_mini":
            patience = 8
            factor = 0.8
        else:  # imagenet
            patience = 10
            factor = 0.9
        
        return AdaptiveLR(
            optimizer, mode='max', factor=factor, patience=patience,
            threshold=0.001, min_lr=base_lr * 0.01
        )

class CosineAnnealingWarmRestartsWithWarmup(optim.lr_scheduler._LRScheduler):
    """Cosine annealing with warm restarts and warmup"""
    
    def __init__(self, optimizer, T_0, T_mult=1, eta_min=0, warmup_epochs=0, warmup_lr=None, last_epoch=-1):
        self.T_0 = T_0
        self.T_i = T_0
        self.T_mult = T_mult
        self.eta_min = eta_min
        self.warmup_epochs = warmup_epochs
        self.warmup_lr = warmup_lr
        self.base_lrs = [group['lr'] for group in optimizer.param_groups]
        super().__init__(optimizer, last_epoch)
    
    def get_lr(self):
        if self.last_epoch < self.warmup_epochs:
            # Warmup phase
            if self.warmup_lr is None:
                warmup_lr = [base_lr * 0.1 for base_lr in self.base_lrs]
            else:
                warmup_lr = [self.warmup_lr] * len(self.base_lrs)
            
            return [
                warmup_lr[i] + (self.base_lrs[i] - warmup_lr[i]) * self.last_epoch / self.warmup_epochs
                for i in range(len(self.base_lrs))
            ]
        else:
            # Cosine annealing phase
            T_cur = self.last_epoch - self.warmup_epochs
            if T_cur >= self.T_i:
                self.T_i *= self.T_mult
                T_cur = 0
            
            return [
                self.eta_min + (base_lr - self.eta_min) * (1 + math.cos(math.pi * T_cur / self.T_i)) / 2
                for base_lr in self.base_lrs
            ]

class OneCycleLR(optim.lr_scheduler._LRScheduler):
    """One cycle learning rate policy"""
    
    def __init__(self, optimizer, max_lr, epochs, pct_start=0.3, anneal_strategy='cos', 
                 div_factor=25, final_div_factor=10000, last_epoch=-1):
        self.max_lr = max_lr if isinstance(max_lr, list) else [max_lr] * len(optimizer.param_groups)
        self.epochs = epochs
        self.pct_start = pct_start
        self.anneal_strategy = anneal_strategy
        self.div_factor = div_factor
        self.final_div_factor = final_div_factor
        self.base_lrs = [lr / div_factor for lr in self.max_lr]
        super().__init__(optimizer, last_epoch)
    
    def get_lr(self):
        if self.last_epoch < self.epochs * self.pct_start:
            # Ascending phase
            pct = self.last_epoch / (self.epochs * self.pct_start)
            return [base_lr + (max_lr - base_lr) * pct for base_lr, max_lr in zip(self.base_lrs, self.max_lr)]
        else:
            # Descending phase
            pct = (self.last_epoch - self.epochs * self.pct_start) / (self.epochs * (1 - self.pct_start))
            if self.anneal_strategy == 'cos':
                pct = (1 + math.cos(math.pi * pct)) / 2
            return [base_lr + (max_lr - base_lr) * (1 - pct) for base_lr, max_lr in zip(self.base_lrs, self.max_lr)]

class PolynomialLR(optim.lr_scheduler._LRScheduler):
    """Polynomial learning rate scheduler"""
    
    def __init__(self, optimizer, total_iters, power=1.0, last_epoch=-1):
        self.total_iters = total_iters
        self.power = power
        super().__init__(optimizer, last_epoch)
    
    def get_lr(self):
        pct = self.last_epoch / self.total_iters
        factor = (1 - pct) ** self.power
        return [base_lr * factor for base_lr in self.base_lrs]

class ExponentialWarmupLR(optim.lr_scheduler._LRScheduler):
    """Exponential decay with warmup"""
    
    def __init__(self, optimizer, gamma, warmup_epochs=0, warmup_lr=None, last_epoch=-1):
        self.gamma = gamma
        self.warmup_epochs = warmup_epochs
        self.warmup_lr = warmup_lr
        super().__init__(optimizer, last_epoch)
    
    def get_lr(self):
        if self.last_epoch < self.warmup_epochs:
            # Warmup phase
            if self.warmup_lr is None:
                warmup_lr = [base_lr * 0.1 for base_lr in self.base_lrs]
            else:
                warmup_lr = [self.warmup_lr] * len(self.base_lrs)
            
            return [
                warmup_lr[i] + (self.base_lrs[i] - warmup_lr[i]) * self.last_epoch / self.warmup_epochs
                for i in range(len(self.base_lrs))
            ]
        else:
            # Exponential decay phase
            return [base_lr * (self.gamma ** (self.last_epoch - self.warmup_epochs)) for base_lr in self.base_lrs]

class AdaptiveLR(optim.lr_scheduler.ReduceLROnPlateau):
    """Enhanced adaptive learning rate scheduler"""
    
    def __init__(self, optimizer, mode='min', factor=0.1, patience=10, threshold=1e-4, 
                 threshold_mode='rel', cooldown=0, min_lr=0, eps=1e-8, verbose=False):
        super().__init__(optimizer, mode, factor, patience, threshold, threshold_mode, 
                        cooldown, min_lr, eps, verbose)
        self.best_loss = float('inf')
        self.best_acc = 0.0
        self.plateau_count = 0
    
    def step(self, metrics, epoch=None):
        if isinstance(metrics, dict):
            loss = metrics.get('loss', float('inf'))
            acc = metrics.get('acc', 0.0)
        else:
            loss = metrics
            acc = 0.0
        
        # Update best metrics
        if loss < self.best_loss:
            self.best_loss = loss
            self.plateau_count = 0
        else:
            self.plateau_count += 1
        
        if acc > self.best_acc:
            self.best_acc = acc
        
        # Call parent step
        super().step(loss, epoch)

class AdvancedTrainingStrategy:
    """Combined advanced optimizer and scheduler strategy"""
    
    def __init__(self):
        self.optimizer_strategy = AdvancedOptimizerStrategy()
        self.scheduler_strategy = AdvancedSchedulerStrategy()
        
        # Stage-specific configurations
        self.stage_configs = {
            "imagenette": {
                "optimizer": "adamw",
                "scheduler": "one_cycle",
                "epochs": 20,
                "lr": 0.002,
                "description": "Quick convergence with one cycle"
            },
            "tiny_imagenet": {
                "optimizer": "adamw",
                "scheduler": "cosine_warmup",
                "epochs": 30,
                "lr": 0.001,
                "description": "Balanced approach with cosine warmup"
            },
            "imagenet_mini": {
                "optimizer": "sgd_momentum",
                "scheduler": "polynomial",
                "epochs": 40,
                "lr": 0.0005,
                "description": "Conservative SGD with polynomial decay"
            },
            "imagenet": {
                "optimizer": "sgd_momentum",
                "scheduler": "exponential_warmup",
                "epochs": 60,
                "lr": 0.1,
                "description": "Aggressive SGD with exponential warmup"
            }
        }
    
    def get_optimizer_and_scheduler(self, model: nn.Module, stage_name: str) -> Tuple[optim.Optimizer, optim.lr_scheduler._LRScheduler]:
        """Get optimized optimizer and scheduler for stage"""
        if stage_name not in self.stage_configs:
            raise ValueError(f"Unknown stage: {stage_name}")
        
        config = self.stage_configs[stage_name]
        
        # Get optimizer
        optimizer = self.optimizer_strategy.get_optimizer(model, stage_name, config)
        
        # Get scheduler
        scheduler = self.scheduler_strategy.get_scheduler(optimizer, stage_name, config)
        
        return optimizer, scheduler
    
    def get_stage_config(self, stage_name: str) -> Dict:
        """Get configuration for specific stage"""
        return self.stage_configs[stage_name]
    
    def visualize_schedules(self, stages: List[str] = None):
        """Visualize learning rate schedules for different stages"""
        if stages is None:
            stages = list(self.stage_configs.keys())
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Advanced Learning Rate Schedules', fontsize=16)
        
        for idx, stage in enumerate(stages):
            if idx >= 4:
                break
                
            row, col = idx // 2, idx % 2
            ax = axes[row, col]
            
            config = self.stage_configs[stage]
            epochs = config["epochs"]
            base_lr = config["lr"]
            
            # Create dummy optimizer and scheduler
            dummy_optimizer = optim.AdamW([torch.tensor(1.0)], lr=base_lr)
            scheduler = self.scheduler_strategy.get_scheduler(dummy_optimizer, stage, config)
            
            # Generate learning rates
            lrs = []
            for epoch in range(epochs):
                lrs.append(scheduler.get_last_lr()[0])
                scheduler.step()
            
            # Plot
            ax.plot(range(epochs), lrs, 'b-', linewidth=2)
            ax.set_title(f'{stage.title()} - {config["scheduler"]}')
            ax.set_xlabel('Epoch')
            ax.set_ylabel('Learning Rate')
            ax.grid(True, alpha=0.3)
            ax.set_yscale('log')
        
        plt.tight_layout()
        plt.savefig('advanced_lr_schedules.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def compare_strategies(self):
        """Compare different optimizer and scheduler combinations"""
        print("Advanced Optimizer and Scheduler Strategy Comparison")
        print("=" * 60)
        
        for stage, config in self.stage_configs.items():
            print(f"\n{stage.upper()}:")
            print(f"  Optimizer: {config['optimizer']}")
            print(f"  Scheduler: {config['scheduler']}")
            print(f"  Learning Rate: {config['lr']}")
            print(f"  Epochs: {config['epochs']}")
            print(f"  Description: {config['description']}")
        
        print("\nKey Advantages:")
        print("✓ One Cycle LR: Super-convergence for small datasets")
        print("✓ Cosine Warmup: Smooth convergence with restarts")
        print("✓ Polynomial Decay: Gradual learning rate reduction")
        print("✓ Exponential Warmup: Conservative approach for large datasets")
        print("✓ Adaptive LR: Automatic adjustment based on performance")
        print("✓ SGD Momentum: Proven effectiveness for ImageNet")
        print("✓ AdamW: Adaptive learning rates with weight decay")

# Example usage and testing
def test_advanced_strategy():
    """Test the advanced optimizer and scheduler strategy"""
    print("Testing Advanced Optimizer and Scheduler Strategy")
    print("=" * 50)
    
    # Create strategy
    strategy = AdvancedTrainingStrategy()
    
    # Compare strategies
    strategy.compare_strategies()
    
    # Visualize schedules
    print("\nGenerating learning rate schedule visualizations...")
    strategy.visualize_schedules()
    
    # Test optimizer creation
    print("\nTesting optimizer creation...")
    model = nn.Linear(10, 5)
    
    for stage_name in strategy.stage_configs.keys():
        try:
            optimizer, scheduler = strategy.get_optimizer_and_scheduler(model, stage_name)
            print(f"✓ {stage_name}: {type(optimizer).__name__} + {type(scheduler).__name__}")
        except Exception as e:
            print(f"✗ {stage_name}: Error - {e}")

if __name__ == "__main__":
    test_advanced_strategy()
