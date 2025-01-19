import numpy as np
import torch
import time
import matplotlib.pyplot as plt
from fvcore.nn import FlopCountAnalysis, flop_count_table
from pathlib import Path

class MetricTracker:
    """Enhanced metric tracking class"""
    def __init__(self, num_classes):
        self.num_classes = num_classes
        self.hist = np.zeros((num_classes, num_classes))
        self.class_metrics = {}
        self.reset()

    def reset(self):
        self.hist *= 0
        self.class_metrics = {
            'iou': np.zeros(self.num_classes),
            'precision': np.zeros(self.num_classes),
            'recall': np.zeros(self.num_classes)
        }

    def update(self, label, pred):
        """Update metrics with new predictions"""
        mask = (label >= 0) & (label < self.num_classes)
        self.hist += np.bincount(
            self.num_classes * label[mask].astype(int) + pred[mask],
            minlength=self.num_classes ** 2
        ).reshape(self.num_classes, self.num_classes)

    def compute_metrics(self):
        """Compute all metrics at once"""
        eps = 1e-7
        diag = np.diag(self.hist)
        
        # IoU = TP / (TP + FP + FN)
        union = (self.hist.sum(1) + self.hist.sum(0) - diag)
        iou = diag / (union + eps)
        
        # Precision = TP / (TP + FP)
        precision = diag / (self.hist.sum(0) + eps)
        
        # Recall = TP / (TP + FN)
        recall = diag / (self.hist.sum(1) + eps)
        
        # F1 Score = 2 * (Precision * Recall) / (Precision + Recall)
        f1 = 2 * precision * recall / (precision + recall + eps)
        
        return {
            'iou': iou,
            'miou': np.nanmean(iou),
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'pixel_acc': diag.sum() / self.hist.sum()
        }

class PerformanceMonitor:
    """Enhanced performance monitoring"""
    def __init__(self, model, device, batch_size=1, warmup=50, test_iters=1000):
        self.model = model
        self.device = device
        self.batch_size = batch_size
        self.warmup = warmup
        self.test_iters = test_iters
        
    def measure_latency(self, input_shape):
        """Measure model latency with warmup"""
        x = torch.randn(self.batch_size, *input_shape).to(self.device)
        
        # Warmup
        for _ in range(self.warmup):
            with torch.no_grad():
                _ = self.model(x)
        
        # Actual measurement
        torch.cuda.synchronize()
        latencies = []
        
        with torch.no_grad():
            for _ in range(self.test_iters):
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                
                start.record()
                _ = self.model(x)
                end.record()
                
                torch.cuda.synchronize()
                latencies.append(start.elapsed_time(end))  # in milliseconds
        
        latencies = np.array(latencies)
        return {
            'mean_latency': np.mean(latencies),
            'std_latency': np.std(latencies),
            'min_latency': np.min(latencies),
            'max_latency': np.max(latencies),
            'fps': 1000 / np.mean(latencies),
            'percentile_90': np.percentile(latencies, 90),
            'percentile_95': np.percentile(latencies, 95),
            'percentile_99': np.percentile(latencies, 99)
        }
    
    def measure_complexity(self, input_shape):
        """Measure model complexity metrics"""
        x = torch.randn(1, *input_shape).to(self.device)
        flops = FlopCountAnalysis(self.model, x)
        
        # Count parameters
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() 
                             if p.requires_grad)
        
        return {
            'flops': flops.total(),
            'gflops': flops.total() / 1e9,
            'params': total_params,
            'trainable_params': trainable_params,
            'flops_table': flop_count_table(flops)
        }

class Visualizer:
    """Enhanced visualization tools"""
    def __init__(self, save_dir='plots'):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(exist_ok=True)
        
    def plot_metrics(self, metrics_dict, title, filename):
        """Plot multiple metrics"""
        plt.figure(figsize=(12, 6))
        
        for name, values in metrics_dict.items():
            plt.plot(values, label=name)
            
        plt.title(title)
        plt.xlabel('Epoch')
        plt.ylabel('Value')
        plt.legend()
        plt.grid(True)
        plt.savefig(self.save_dir / filename)
        plt.close()
    
    def plot_learning_curves(self, train_metrics, val_metrics, best_epoch, model_name):
        """Plot comprehensive learning curves"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle(f'{model_name} Training Curves')
        
        # Loss curve
        if 'loss' in train_metrics:
            axes[0, 0].plot(train_metrics['loss'], label='Train')
            axes[0, 0].plot(val_metrics['loss'], label='Validation')
            axes[0, 0].set_title('Loss')
            axes[0, 0].legend()
            axes[0, 0].grid(True)
        
        # mIoU curve
        axes[0, 1].plot(train_metrics['miou'], label='Train')
        axes[0, 1].plot(val_metrics['miou'], label='Validation')
        axes[0, 1].axvline(x=best_epoch, color='r', linestyle='--', 
                          label=f'Best Epoch ({best_epoch+1})')
        axes[0, 1].set_title('Mean IoU')
        axes[0, 1].legend()
        axes[0, 1].grid(True)
        
        # Accuracy curve
        if 'acc' in train_metrics:
            axes[1, 0].plot(train_metrics['acc'], label='Train')
            axes[1, 0].plot(val_metrics['acc'], label='Validation')
            axes[1, 0].set_title('Accuracy')
            axes[1, 0].legend()
            axes[1, 0].grid(True)
        
        # Learning rate curve
        if 'lr' in train_metrics:
            axes[1, 1].plot(train_metrics['lr'])
            axes[1, 1].set_title('Learning Rate')
            axes[1, 1].grid(True)
        
        plt.tight_layout()
        plt.savefig(self.save_dir / f'{model_name}_learning_curves.png')
        plt.close()

class LRSchedulerWithWarmup:
    """Enhanced learning rate scheduler with warmup"""
    def __init__(self, optimizer, init_lr, warmup_epochs, max_epochs, 
                 warmup_method='linear', decay_method='poly', power=0.9):
        self.optimizer = optimizer
        self.init_lr = init_lr
        self.warmup_epochs = warmup_epochs
        self.max_epochs = max_epochs
        self.warmup_method = warmup_method
        self.decay_method = decay_method
        self.power = power
        
    def step(self, epoch, iteration, len_epoch):
        """Update learning rate"""
        current_iter = epoch * len_epoch + iteration
        max_iter = self.max_epochs * len_epoch
        
        if epoch < self.warmup_epochs:
            # Warmup phase
            if self.warmup_method == 'linear':
                lr = self.init_lr * current_iter / (self.warmup_epochs * len_epoch)
            else:  # log
                lr = self.init_lr * np.log(1 + current_iter) / \
                     np.log(1 + self.warmup_epochs * len_epoch)
        else:
            # Decay phase
            if self.decay_method == 'poly':
                lr = self.init_lr * (1 - current_iter/max_iter) ** self.power
            elif self.decay_method == 'cosine':
                lr = self.init_lr * \
                     (1 + np.cos(np.pi * current_iter / max_iter)) / 2
            else:  # step
                lr = self.init_lr * (0.1 ** (current_iter // (max_iter//3)))
        
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
        
        return lr