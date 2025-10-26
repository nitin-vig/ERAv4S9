"""
Training utilities module for model training and evaluation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
from config import Config

class MetricsTracker:
    """Class to track training metrics"""
    
    def __init__(self):
        self.train_losses = []
        self.test_losses = []
        self.train_acc = []
        self.test_acc = []
        self.test_top5_acc = []
        self.learning_rates = []
    
    def add_train_metrics(self, loss, accuracy):
        """Add training metrics"""
        self.train_losses.append(loss)
        self.train_acc.append(accuracy)
    
    def add_test_metrics(self, loss, accuracy, top5_accuracy=None):
        """Add test metrics"""
        self.test_losses.append(loss)
        self.test_acc.append(accuracy)
        if top5_accuracy is not None:
            self.test_top5_acc.append(top5_accuracy)
    
    def add_lr(self, lr):
        """Add learning rate"""
        self.learning_rates.append(lr)
    
    def plot_metrics(self, save_path=None):
        """Plot training metrics"""
        fig, axs = plt.subplots(2, 2, figsize=(15, 10))
        
        # Training Loss
        axs[0, 0].plot(self.train_losses)
        axs[0, 0].set_title("Training Loss")
        axs[0, 0].set_xlabel("Batch")
        axs[0, 0].set_ylabel("Loss")
        
        # Training Accuracy
        axs[1, 0].plot(self.train_acc)
        axs[1, 0].set_title("Training Accuracy")
        axs[1, 0].set_xlabel("Batch")
        axs[1, 0].set_ylabel("Accuracy (%)")
        
        # Test Loss
        axs[0, 1].plot(self.test_losses)
        axs[0, 1].set_title("Test Loss")
        axs[0, 1].set_xlabel("Epoch")
        axs[0, 1].set_ylabel("Loss")
        
        # Test Accuracy
        axs[1, 1].plot(self.test_acc)
        axs[1, 1].set_title("Test Accuracy")
        axs[1, 1].set_xlabel("Epoch")
        axs[1, 1].set_ylabel("Accuracy (%)")
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
            print(f"Metrics plot saved to {save_path}")
        
        plt.show()

def topk_accuracy(output, target, k=5):
    """
    Calculates the top-k accuracy for a given output and target.

    Args:
        output (torch.Tensor): The model's raw output logits, typically from the last layer.
        target (torch.Tensor): The ground-truth labels.
        k (int): The number of top predictions to consider.

    Returns:
        float: The top-k accuracy.
    """
    with torch.no_grad():
        # Get the top k predictions
        _, topk_preds = output.topk(k, dim=1, largest=True, sorted=True)

        # Reshape the target tensor for comparison
        target_reshaped = target.view(1, -1).expand_as(topk_preds)

        # Check if any of the top k predictions match the true label
        correct = (topk_preds == target_reshaped)

        # Calculate the number of correct predictions
        correct_count = correct.any(dim=0).sum().item()

        # Return the accuracy as a percentage
        return (correct_count / target.size(0))

def train_epoch(model, device, train_loader, optimizer, criterion, epoch, metrics_tracker):
    """Train the model for one epoch"""
    model.train()
    pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}')
    correct = 0
    processed = 0
    
    for batch_idx, (data, target) in enumerate(pbar):
        # Move data to device
        data, target = data.to(device), target.to(device)

        # Zero gradients
        optimizer.zero_grad()

        # Forward pass
        output = model(data)
        loss = criterion(output, target)

        # Backward pass
        loss.backward()
        optimizer.step()

        # Calculate accuracy
        pred = output.argmax(dim=1, keepdim=True)
        correct += pred.eq(target.view_as(pred)).sum().item()
        processed += len(data)

        # Update metrics
        accuracy = 100. * correct / processed
        metrics_tracker.add_train_metrics(loss.item(), accuracy)

        # Update progress bar
        pbar.set_description(
            desc=f'Epoch {epoch+1} - Loss={loss.item():.4f} - Accuracy={accuracy:.2f}%'
        )

def test_epoch(model, device, test_loader, criterion, metrics_tracker):
    """Test the model"""
    model.eval()
    test_loss = 0
    correct = 0
    correct_top5 = 0
    
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            
            # Calculate loss (criterion returns mean per batch, multiply by batch size to get sum)
            batch_size = target.size(0)
            test_loss += criterion(output, target).item() * batch_size

            # Top-1 accuracy
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            
            # Top-5 accuracy
            correct_top5 += topk_accuracy(output, target, k=5)

    # Calculate average metrics
    test_loss /= len(test_loader.dataset)
    accuracy = 100. * correct / len(test_loader.dataset)
    top5_accuracy = 100. * correct_top5 / len(test_loader.dataset)

    # Update metrics
    metrics_tracker.add_test_metrics(test_loss, accuracy, top5_accuracy)

    print(f'Test set: Average loss: {test_loss:.4f}, '
          f'Top-1 Accuracy: {correct}/{len(test_loader.dataset)} ({accuracy:.2f}%), '
          f'Top-5 Accuracy: {correct_top5}/{len(test_loader.dataset)} ({top5_accuracy:.2f}%)')

    return test_loss

def get_optimizer(model, optimizer_name="sgd", lr=0.1, weight_decay=1e-4):
    """Get optimizer - SGD is standard for ImageNet"""
    if optimizer_name.lower() == "sgd":
        return optim.SGD(model.parameters(), lr=lr, weight_decay=weight_decay, momentum=0.9)
    elif optimizer_name.lower() == "adamw":
        return optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif optimizer_name.lower() == "adam":
        return optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    else:
        raise ValueError(f"Unknown optimizer: {optimizer_name}")

def get_scheduler(optimizer, scheduler_name="step", **kwargs):
    """Get learning rate scheduler - StepLR is standard for ImageNet"""
    if scheduler_name.lower() == "step":
        return optim.lr_scheduler.StepLR(
            optimizer,
            step_size=kwargs.get('step_size', 30),
            gamma=kwargs.get('gamma', 0.1)
        )
    elif scheduler_name.lower() == "reduce_lr":
        return optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=kwargs.get('factor', 0.5),
            patience=kwargs.get('patience', 3),
            threshold=kwargs.get('threshold', 1e-3),
            min_lr=kwargs.get('min_lr', 1e-4)
        )
    elif scheduler_name.lower() == "cosine":
        return optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=kwargs.get('T_max', 50),
            eta_min=kwargs.get('eta_min', 1e-6)
        )
    else:
        raise ValueError(f"Unknown scheduler: {scheduler_name}")

def get_criterion(criterion_name="cross_entropy", **kwargs):
    """Get loss criterion"""
    if criterion_name.lower() == "cross_entropy":
        return nn.CrossEntropyLoss(label_smoothing=kwargs.get('label_smoothing', 0.1))
    elif criterion_name.lower() == "focal":
        # You would need to implement FocalLoss
        raise NotImplementedError("FocalLoss not implemented")
    else:
        raise ValueError(f"Unknown criterion: {criterion_name}")

def train_model(model, train_loader, test_loader, device, config=None):
    """Complete training pipeline with dataset-specific configuration"""
    
    if config is None:
        config = Config()
    
    # Initialize metrics tracker
    metrics_tracker = MetricsTracker()
    
    # Get dataset-specific configuration
    dataset_config = config.get_dataset_config()
    
    # Get optimizer, scheduler, and criterion based on dataset
    optimizer = get_optimizer(
        model, 
        optimizer_name=dataset_config.get("optimizer", "adamw"), 
        lr=dataset_config.get("lr", config.LEARNING_RATE), 
        weight_decay=config.WEIGHT_DECAY
    )
    
    scheduler = get_scheduler(
        optimizer, 
        scheduler_name=dataset_config.get("scheduler", "reduce_lr"),
        step_size=30,
        gamma=0.1,
        patience=10,
        factor=0.5,
        min_lr=1e-6
    )
    
    criterion = get_criterion(criterion_name="cross_entropy", label_smoothing=0.1)
    
    # Training loop
    best_test_loss = float('inf')
    
    for epoch in range(config.NUM_EPOCHS):
        print(f"\nEpoch {epoch+1}/{config.NUM_EPOCHS}")
        
        # Train
        train_epoch(model, device, train_loader, optimizer, criterion, epoch, metrics_tracker)
        
        # Test
        test_loss = test_epoch(model, device, test_loader, criterion, metrics_tracker)
        
        # Update scheduler based on type
        if dataset_config.get("scheduler", "reduce_lr") == "step":
            scheduler.step()  # StepLR updates every epoch
        else:
            scheduler.step(test_loss)  # ReduceLROnPlateau updates based on loss
        
        # Track learning rate
        current_lr = optimizer.param_groups[0]['lr']
        metrics_tracker.add_lr(current_lr)
        
        # Save best model
        if test_loss < best_test_loss:
            best_test_loss = test_loss
            save_path = f"{config.SAVE_MODEL_PATH}/best_model_{config.DATASET_NAME}.pth"
            torch.save(model.state_dict(), save_path)
            print(f"New best model saved with test loss: {test_loss:.4f}")
        
        print(f"Current learning rate: {current_lr:.6f}")
    
    return metrics_tracker

def evaluate_model(model, test_loader, device, criterion=None):
    """Evaluate the model on test set"""
    if criterion is None:
        criterion = nn.CrossEntropyLoss()
    
    model.eval()
    test_loss = 0
    correct = 0
    correct_top5 = 0
    
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            
            # Calculate loss (criterion returns mean per batch, multiply by batch size to get sum)
            batch_size = target.size(0)
            test_loss += criterion(output, target).item() * batch_size
            
            # Top-1 accuracy
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            
            # Top-5 accuracy
            correct_top5 += topk_accuracy(output, target, k=5)

    test_loss /= len(test_loader.dataset)
    accuracy = 100. * correct / len(test_loader.dataset)
    top5_accuracy = 100. * correct_top5 / len(test_loader.dataset)

    print(f'Final Test Results:')
    print(f'Average loss: {test_loss:.4f}')
    print(f'Top-1 Accuracy: {correct}/{len(test_loader.dataset)} ({accuracy:.2f}%)')
    print(f'Top-5 Accuracy: {correct_top5}/{len(test_loader.dataset)} ({top5_accuracy:.2f}%)')
    
    return test_loss, accuracy, top5_accuracy
