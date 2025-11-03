"""
Performance Projection Tool
Analyzes recent training logs and projects final performance
"""
import numpy as np
import matplotlib.pyplot as plt

def project_performance():
    """Project final performance based on recent trend"""
    
    # Recent performance data
    epochs = [78, 79, 80]
    top1_acc = [65.82, 66.50, 67.41]
    
    # Calculate improvement rates
    improvements = [top1_acc[i] - top1_acc[i-1] for i in range(1, len(top1_acc))]
    avg_improvement = np.mean(improvements)
    
    print("="*70)
    print("PERFORMANCE PROJECTION ANALYSIS")
    print("="*70)
    print(f"\n📊 Recent Performance:")
    print(f"   Epoch 78: {top1_acc[0]:.2f}%")
    print(f"   Epoch 79: {top1_acc[1]:.2f}% (+{improvements[0]:.2f}%)")
    print(f"   Epoch 80: {top1_acc[1]:.2f}% (+{improvements[1]:.2f}%)")
    print(f"\n📈 Average improvement (last 2 epochs): {avg_improvement:.2f}%")
    
    # One Cycle LR schedule info
    max_lr = 0.16
    total_epochs = 100
    warmup_epochs = 20
    anneal_epochs = 80
    
    print(f"\n🔧 Training Configuration:")
    print(f"   Max LR: {max_lr}")
    print(f"   Total epochs: {total_epochs}")
    print(f"   Warmup: {warmup_epochs} epochs (peak LR at epoch 20)")
    print(f"   Anneal phase: {anneal_epochs} epochs (epochs 21-100)")
    
    # LR at different epochs (approximate)
    def lr_at_epoch(epoch, max_lr=0.16, warmup_epochs=20, total_epochs=100):
        """Estimate LR at given epoch for One Cycle LR"""
        if epoch <= warmup_epochs:
            # Warmup: linear from max_lr/25 to max_lr
            progress = epoch / warmup_epochs
            return (max_lr / 25) + progress * (max_lr - max_lr / 25)
        else:
            # Anneal: cosine from max_lr to max_lr/250000
            anneal_progress = (epoch - warmup_epochs) / (total_epochs - warmup_epochs)
            # Cosine schedule
            lr_range = max_lr - (max_lr / 250000)
            return max_lr - lr_range * (1 - np.cos(anneal_progress * np.pi / 2))
    
    current_lr = lr_at_epoch(80, max_lr, warmup_epochs, total_epochs)
    final_lr = lr_at_epoch(100, max_lr, warmup_epochs, total_epochs)
    
    print(f"\n⚡ Learning Rate Status:")
    print(f"   Current LR (epoch 80): ~{current_lr:.6f}")
    print(f"   Final LR (epoch 100): ~{final_lr:.9f}")
    print(f"   LR reduction factor: {current_lr/final_lr:.1f}x lower by end")
    
    # Projections with different scenarios
    epochs_remaining = 20
    current_acc = top1_acc[-1]
    
    # Scenario 1: Linear decay of improvement rate
    # Improvement rate will decrease as LR decreases
    lr_ratio = final_lr / current_lr  # How much LR will decrease
    
    # Conservative: improvement rate halves
    conservative_improvement = avg_improvement * 0.5
    conservative_projection = current_acc + (conservative_improvement * epochs_remaining * 0.5)
    
    # Moderate: improvement rate scales with LR reduction
    moderate_improvement = avg_improvement * (1 + lr_ratio) / 2
    moderate_projection = current_acc + (moderate_improvement * epochs_remaining * 0.4)
    
    # Optimistic: maintains current improvement rate
    optimistic_projection = current_acc + (avg_improvement * epochs_remaining * 0.3)
    
    print(f"\n🎯 PROJECTIONS FOR EPOCH 100:")
    print("="*70)
    print(f"   Conservative Estimate: {conservative_projection:.2f}% Top-1")
    print(f"     (Assumes slower gains as LR decreases significantly)")
    print()
    print(f"   Moderate Estimate:    {moderate_projection:.2f}% Top-1")
    print(f"     (Balanced view: some gains but diminishing returns)")
    print()
    print(f"   Optimistic Estimate:   {optimistic_projection:.2f}% Top-1")
    print(f"     (Best case if current trend continues)")
    print()
    print(f"   📊 Most Likely Range:  {moderate_projection-0.5:.2f}% - {moderate_projection+0.5:.2f}%")
    print("="*70)
    
    # Extrapolate future epochs
    future_epochs = list(range(81, 101))
    future_projections = []
    
    for epoch in future_epochs:
        # Use diminishing improvement rate
        progress = (epoch - 80) / epochs_remaining
        # Improvement rate decreases over remaining epochs
        effective_improvement_rate = avg_improvement * (1 - progress * 0.7)  # 70% reduction by end
        if epoch == 81:
            projected = current_acc + effective_improvement_rate
        else:
            projected = future_projections[-1] + effective_improvement_rate
        future_projections.append(min(projected, 100))  # Cap at 100%
    
    # Create visualization
    all_epochs = epochs + future_epochs
    all_acc = top1_acc + future_projections
    
    plt.figure(figsize=(12, 7))
    
    # Plot actual data
    plt.plot(epochs, top1_acc, 'bo-', linewidth=2, markersize=8, label='Actual Performance', zorder=3)
    
    # Plot projected data
    plt.plot(future_epochs, future_projections, 'r--', linewidth=2, alpha=0.7, label='Projected Performance', zorder=2)
    
    # Add projection range
    conservative_line = [conservative_projection] * len(future_epochs)
    optimistic_line = [optimistic_projection] * len(future_epochs)
    plt.fill_between(future_epochs, conservative_line, optimistic_line, alpha=0.2, color='green', 
                     label='Projection Range', zorder=1)
    
    # Mark key points
    plt.axvline(x=80, color='gray', linestyle=':', alpha=0.5, label='Current Epoch')
    plt.axvline(x=100, color='red', linestyle=':', alpha=0.5, label='Final Epoch')
    plt.axhline(y=moderate_projection, color='orange', linestyle='--', alpha=0.5, label='Moderate Projection')
    
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Top-1 Accuracy (%)', fontsize=12)
    plt.title(f'Performance Projection: Epoch 80 → 100\nMax LR: {max_lr}, One Cycle LR Schedule', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.legend(loc='best')
    plt.xlim(75, 105)
    plt.ylim(65, max(max(all_acc), moderate_projection + 2))
    
    # Add text annotations
    plt.text(82, current_acc + 0.5, f'Current: {current_acc:.2f}%', fontsize=10, 
             bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.7))
    plt.text(92, moderate_projection, f'Projected: {moderate_projection:.2f}%', fontsize=10,
             bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.7))
    
    plt.tight_layout()
    plt.savefig('performance_projection.png', dpi=150, bbox_inches='tight')
    print(f"\n📊 Visualization saved to: performance_projection.png")
    plt.show()
    
    return {
        'current': current_acc,
        'conservative': conservative_projection,
        'moderate': moderate_projection,
        'optimistic': optimistic_projection
    }

if __name__ == "__main__":
    results = project_performance()

