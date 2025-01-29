import numpy as np
import torch
import time
from typing import Dict, Any
import matplotlib.pyplot as plt
from fvcore.nn import FlopCountAnalysis, flop_count_table

def poly_lr_scheduler(optimizer: torch.optim.Optimizer, init_lr: float, 
                     iter: int, max_iter: int, power: float = 0.9) -> float:
    """Polynomial learning rate decay
    Args:
        optimizer: Optimizer to update learning rate
        init_lr: Initial learning rate
        iter: Current iteration
        max_iter: Maximum iterations
        power: Polynomial power
    Returns:
        Current learning rate
    """
    lr = init_lr * (1 - iter/max_iter)**power
    optimizer.param_groups[0]['lr'] = lr
    return lr

def fast_hist(label: np.ndarray, pred: np.ndarray, n: int) -> np.ndarray:
    """Compute confusion matrix
    Args:
        label: Ground truth labels
        pred: Predicted labels
        n: Number of classes
    Returns:
        Confusion matrix
    """
    k = (label >= 0) & (label < n)
    return np.bincount(
        n * label[k].astype(int) + pred[k], 
        minlength=n**2
    ).reshape(n, n)

def per_class_iou(hist: np.ndarray) -> np.ndarray:
    """Calculate per-class IoU
    Args:
        hist: Confusion matrix
    Returns:
        Per-class IoU scores
    """
    epsilon = 1e-5
    intersection = np.diag(hist)
    union = hist.sum(1) + hist.sum(0) - intersection
    return intersection / (union + epsilon)

def test_fps_latency(model: torch.nn.Module, device: torch.device, 
                     height: int = 512, width: int = 1024) -> Dict[str, Any]:
    """Test model latency, FPS and FLOPs
    Args:
        model: Model to test
        device: Device to run on
        height: Input height
        width: Input width
    Returns:
        Dictionary containing latency, FPS and FLOPs statistics
    """
    model.eval()
    dummy_input = torch.randn(1, 3, height, width).to(device)
    
    # Calculate FLOPs
    flops = FlopCountAnalysis(model.cpu(), torch.zeros((1, 3, height, width)))
    flops_table = flop_count_table(flops)
    total_flops = flops.total()
    model = model.to(device)
    
    # Warmup
    with torch.no_grad():
        for _ in range(10):
            _ = model(dummy_input)
    
    # Test
    times = []
    with torch.no_grad():
        for _ in range(100):
            start = time.time()
            _ = model(dummy_input)
            torch.cuda.synchronize()
            times.append(time.time() - start)
    
    times = np.array(times) * 1000  # Convert to ms
    return {
        'mean_latency': np.mean(times),
        'std_latency': np.std(times),
        'min_latency': np.min(times),
        'max_latency': np.max(times),
        'fps': 1000 / np.mean(times),
        'total_flops': total_flops,
        'flops_table': flops_table,
        'gflops': total_flops / 1e9
    }

def count_parameters(model: torch.nn.Module) -> Dict[str, float]:
    """Count model parameters and FLOPs
    Args:
        model: Model to analyze
    Returns:
        Dictionary containing parameter statistics
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    return {
        'total_params': total_params,
        'trainable_params': trainable_params,
        'params_M': total_params / 1e6
    }

def plot_training_curves(metrics: Dict[str, list], save_path: str) -> None:
    """Plot training curves including FLOPs analysis
    Args:
        metrics: Dictionary containing training metrics
        save_path: Path to save plot
    """
    plt.figure(figsize=(20, 5))
    
    plt.subplot(141)
    plt.plot(metrics['train_loss'], label='Train')
    plt.plot(metrics['val_loss'], label='Val')
    plt.title('Loss')
    plt.xlabel('Epoch')
    plt.legend()
    
    plt.subplot(142)
    plt.plot(metrics['train_miou'], label='Train')
    plt.plot(metrics['val_miou'], label='Val')
    plt.title('mIoU')
    plt.xlabel('Epoch')
    plt.legend()
    
    if 'lr' in metrics:
        plt.subplot(143)
        plt.plot(metrics['lr'])
        plt.title('Learning Rate')
        plt.xlabel('Epoch')
    
    if 'gflops' in metrics:
        plt.subplot(144)
        plt.bar(['GFLOPs'], [metrics['gflops']])
        plt.title('Computational Cost')
        plt.ylabel('GFLOPs')
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def analyze_model_complexity(model: torch.nn.Module, height: int = 512, width: int = 1024) -> Dict[str, Any]:
    """Comprehensive model analysis including parameters and FLOPs
    Args:
        model: Model to analyze
        height: Input height
        width: Input width
    Returns:
        Dictionary containing all model statistics
    """
    # Store original device
    device = next(model.parameters()).device
    
    # Move model to CPU for FLOPs analysis
    model = model.cpu()
    params_info = count_parameters(model)
    flops = FlopCountAnalysis(model, torch.zeros((1, 3, height, width)))
    
    # Move model back to original device
    model = model.to(device)
    
    return {
        **params_info,
        'total_flops': flops.total(),
        'gflops': flops.total() / 1e9,
        'flops_table': flop_count_table(flops)
    }