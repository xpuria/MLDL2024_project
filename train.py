import torch
import torch.nn as nn
from tqdm import tqdm
from pathlib import Path
import numpy as np
from MLDL2024_project.utils import fast_hist, per_class_iou, poly_lr_scheduler, plot_training_curves

def train_model(model: nn.Module,
               criterion: nn.Module,
               optimizer: torch.optim.Optimizer,
               train_loader: torch.utils.data.DataLoader,
               val_loader: torch.utils.data.DataLoader,
               device: torch.device,
               num_epochs: int,
               save_dir: str = 'checkpoints',
               scheduler_mode: bool = True) -> tuple:
    """Training function with validation"""
    save_dir = Path(save_dir)
    save_dir.mkdir(exist_ok=True)

    history = {
        'train_loss': [],
        'train_miou': [],
        'val_loss': [],
        'val_miou': [],
        'lr': []
    }
    
    best_miou = 0
    best_epoch = 0
    
    def print_iou_per_class(hist, phase='train'):
        """Print IoU for each class"""
        class_names = train_loader.dataset.get_class_names()
        iou_per_class = per_class_iou(hist) * 100
        print(f"\n{phase.capitalize()} IoU per class:")
        for class_idx, iou in enumerate(iou_per_class):
            print(f"{class_names[class_idx]:20s}: {iou:.2f}%")
    
    def process_labels(labels, dataset):
        """Process labels based on their shape and convert to class indices"""
        batch_class_labels = []
        for label in labels:
            # If label has 4 channels, use only the first 3
            if len(label.shape) == 3 and label.shape[2] == 4:  # [H, W, 4]
                label = label[:, :, :3]  # Use only RGB channels
            class_label = dataset.map_color_to_class(label)
            batch_class_labels.append(class_label)
        return torch.stack(batch_class_labels)
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0
        train_hist = np.zeros((19, 19))
        
        train_loop = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs} (Train)')
        for i, (images, labels) in enumerate(train_loop):
            # Process labels
            labels = process_labels(labels, train_loader.dataset)
                
            images = images.to(device)
            labels = labels.to(device)
            
            optimizer.zero_grad()
            outputs, _, _ = model(images)
            
            main_output = outputs
            loss = criterion(main_output, labels.long())
            
            loss.backward()
            
            if scheduler_mode:
                current_iter = epoch * len(train_loader) + i
                lr = poly_lr_scheduler(
                    optimizer, 
                    init_lr=0.01, 
                    iter=current_iter,
                    max_iter=num_epochs * len(train_loader)
                )
                history['lr'].append(lr)
            
            optimizer.step()
            
            train_loss += loss.item()
            predictions = main_output.argmax(1)
            train_hist += fast_hist(
                labels.cpu().numpy(),
                predictions.cpu().numpy(),
                19
            )
            
            # Update progress bar with current metrics
            miou = np.mean(per_class_iou(train_hist)) * 100
            train_loop.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'mIoU': f'{miou:.2f}%'
            })
        
        train_miou = np.mean(per_class_iou(train_hist)) * 100
        train_loss = train_loss / len(train_loader)
        
        # Validation phase
        model.eval()
        val_loss = 0
        val_hist = np.zeros((19, 19))
        
        with torch.no_grad():
            val_loop = tqdm(val_loader, desc=f'Epoch {epoch+1}/{num_epochs} (Val)')
            for images, labels in val_loop:
                # Process labels
                labels = process_labels(labels, val_loader.dataset)
                
                images = images.to(device)
                labels = labels.to(device)
                
                outputs = model(images)
                if isinstance(outputs, tuple):
                    outputs = outputs[0]
                
                loss = criterion(outputs, labels.long())
                val_loss += loss.item()
                
                predictions = outputs.argmax(1)
                val_hist += fast_hist(
                    labels.cpu().numpy(),
                    predictions.cpu().numpy(),
                    19
                )
                
                val_loop.set_postfix({'Loss': f'{loss.item():.4f}'})
        
        val_miou = np.mean(per_class_iou(val_hist)) * 100
        val_loss = val_loss / len(val_loader)
        
        # Update history
        history['train_loss'].append(train_loss)
        history['train_miou'].append(train_miou)
        history['val_loss'].append(val_loss)
        history['val_miou'].append(val_miou)
        
        # Save best model
        if val_miou > best_miou:
            best_miou = val_miou
            best_epoch = epoch
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_miou': best_miou,
            }, save_dir / 'best_model.pth')
        
        # Print epoch summary with per-class IoU
        print(f'\nEpoch {epoch+1}/{num_epochs}:')
        print(f'Train Loss: {train_loss:.4f}, mIoU: {train_miou:.2f}%')
        print_iou_per_class(train_hist, 'train')
        print(f'\nVal Loss: {val_loss:.4f}, mIoU: {val_miou:.2f}%')
        print_iou_per_class(val_hist, 'val')
        
        # Plot training curves
        plot_training_curves(history, save_dir / 'training_curves.png')
    
    return history['train_miou'], history['val_miou'], best_epoch