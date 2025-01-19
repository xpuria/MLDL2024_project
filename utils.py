import numpy as np
import torch
from fvcore.nn import FlopCountAnalysis, flop_count_table
import time
from pathlib import Path

def poly_lr_scheduler(optimizer, init_lr, iter, lr_decay_iter=1,
                     max_iter=300, power=0.9):
    """Polynomial decay of learning rate
            :param init_lr is base learning rate
            :param iter is a current iteration
            :param lr_decay_iter how frequently decay occurs, default is 1
            :param max_iter is number of maximum iterations
            :param power is a polymomial power
    """
    lr = init_lr*(1 - iter/max_iter)**power
    optimizer.param_groups[0]['lr'] = lr
    return lr

def fast_hist(a, b, n):
    '''
    a and b are label and prediction respectively
    n is the number of classes
    '''
    k = (a >= 0) & (a < n)
    return np.bincount(n * a[k].astype(int) + b[k], minlength=n ** 2).reshape(n, n)

def per_class_iou(hist):
    epsilon = 1e-5
    return (np.diag(hist)) / (hist.sum(1) + hist.sum(0) - np.diag(hist) + epsilon)

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.initialized = False
        self.val = None
        self.avg = None
        self.sum = None
        self.count = None

    def initialize(self, val, weight):
        self.val = val
        self.avg = val
        self.sum = val * weight
        self.count = weight
        self.initialized = True

    def update(self, val, weight=1):
        if not self.initialized:
            self.initialize(val, weight)
        else:
            self.add(val, weight)

    def add(self, val, weight):
        self.val = val
        self.sum += val * weight
        self.count += weight
        self.avg = self.sum / self.count

    def value(self):
        return self.val

    def average(self):
        return self.avg

def calculate_model_stats(model, height=1024, width=512):
    """Calculate model complexity and inference time"""
    device = next(model.parameters()).device
    dummy_input = torch.zeros((1, 3, height, width)).to(device)
    
    # Calculate FLOPs using fvcore
    flops = FlopCountAnalysis(model, dummy_input)
    flops_str = flop_count_table(flops)
    gflops = flops.total() / 1e9
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    # Measure inference time
    model.eval()
    with torch.no_grad():
        # Warmup
        for _ in range(10):
            _ = model(dummy_input)
        
        # Timing
        times = []
        for _ in range(100):
            start_time = time.time()
            _ = model(dummy_input)
            torch.cuda.synchronize()
            times.append(time.time() - start_time)
    
    avg_time = np.mean(times) * 1000  # Convert to ms
    fps = 1000 / avg_time  # Calculate FPS
    
    return {
        'GFLOPs': gflops,
        'Parameters(M)': total_params / 1e6,
        'Trainable Parameters(M)': trainable_params / 1e6,
        'Detailed FLOPs': flops_str,
        'Inference Time(ms)': avg_time,
        'FPS': fps
    }

def color_map(N=256, normalized=False):
    """Generate color map for visualization"""
    def bitget(byteval, idx):
        return (byteval & (1 << idx)) != 0

    dtype = 'float32' if normalized else 'uint8'
    cmap = np.zeros((N, 3), dtype=dtype)
    for i in range(N):
        r = g = b = 0
        c = i
        for j in range(8):
            r = r | (bitget(c, 0) << 7-j)
            g = g | (bitget(c, 1) << 7-j)
            b = b | (bitget(c, 2) << 7-j)
            c = c >> 3

        cmap[i] = np.array([r, g, b])

    cmap = cmap/255 if normalized else cmap
    return cmap

class Evaluator(object):
    def __init__(self, num_class):
        self.num_class = num_class
        self.confusion_matrix = np.zeros((self.num_class,)*2)

    def reset(self):
        self.confusion_matrix = np.zeros((self.num_class,)*2)

    def _generate_matrix(self, gt_image, pre_image):
        mask = (gt_image >= 0) & (gt_image < self.num_class)
        label = self.num_class * gt_image[mask].astype('int') + pre_image[mask]
        count = np.bincount(label, minlength=self.num_class**2)
        confusion_matrix = count.reshape(self.num_class, self.num_class)
        return confusion_matrix

    def add_batch(self, gt_image, pre_image):
        assert gt_image.shape == pre_image.shape
        self.confusion_matrix += self._generate_matrix(gt_image, pre_image)

    def get_scores(self):
        hist = self.confusion_matrix
        acc = np.diag(hist).sum() / hist.sum()
        acc_cls = np.diag(hist) / hist.sum(axis=1)
        acc_cls = np.nanmean(acc_cls)
        iu = np.diag(hist) / (hist.sum(axis=1) + hist.sum(axis=0) - np.diag(hist))
        mean_iu = np.nanmean(iu)
        freq = hist.sum(axis=1) / hist.sum()
        fwavacc = (freq[freq > 0] * iu[freq > 0]).sum()
        cls_iu = dict(zip(range(self.num_class), iu))

        return {
            "Pixel Accuracy": acc,
            "Mean Accuracy": acc_cls,
            "Frequency Weighted IoU": fwavacc,
            "Mean IoU": mean_iu,
            "Class IoU": cls_iu,
        }

def save_checkpoint(state, is_best, path, filename='checkpoint.pth'):
    """Save checkpoint during training"""
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    torch.save(state, path / filename)
    if is_best:
        torch.save(state, path / 'model_best.pth')

class Timer:
    """A simple timer class"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.start_time = time.time()
        self.last_time = self.start_time

    def get_time_from_start(self):
        return time.time() - self.start_time

    def get_time_from_last(self):
        current_time = time.time()
        duration = current_time - self.last_time
        self.last_time = current_time
        return duration