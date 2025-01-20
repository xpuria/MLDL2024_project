import torch
import torch.nn as nn
from tqdm import tqdm
from pathlib import Path
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
    """Training function with validation
    Args:
        model: Model to train
        criterion: Loss function
        optimizer: Optimizer
        train_loader: Training data loader
        val_loader: Validation data loader
        device: Device to train on
        num_epochs: Number of epochs
        save_dir: Directory to save checkpoints
        scheduler_mode: Whether to use learning rate scheduler
    Returns:
        Tuple containing training history and best epoch
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(exist_ok=True)

    # Training history
    history = {
        'train_loss': [],
        'train_miou': [],
        'val_loss': [],
        'val_miou': [],
        'lr': []
    }
    
    # Best metric tracking
    best_miou = 0
    best_epoch = 0
    
    # Main training loop
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0
        hist = np.zeros((19, 19))  # 19 classes
        
        train_loop = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs} (Train)')
        for i, (images, labels) in enumerate(train_loop):
            images = images.to(device)
            labels = labels.to(device)
            
            # Forward pass
            optimizer.zero_grad()
            outputs = model(images)
            
            # Handle auxiliary outputs (BiSeNet)
            if isinstance(outputs, tuple):
                loss = criterion(outputs[0], labels)
                if model.training:
                    aux_loss2 = criterion(outputs[1], labels)
                    aux_loss3 = criterion(outputs[2], labels)
                    loss = loss + 0.4 * aux_loss2 + 0.4 * aux_loss3
                outputs = outputs[0]
            else:
                loss = criterion(outputs, labels)
            
            # Backward pass
            loss.backward()
            
            # Update learning rate
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
            
            # Update metrics
            train_loss += loss.item()
            predictions = outputs.argmax(1)
            hist += fast_hist(
                labels.cpu().numpy(),
                predictions.cpu().numpy(),
                19
            )
            
            # Update progress bar
            miou = np.mean(per_class_iou(hist)) * 100
            train_loop.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'mIoU': f'{miou:.2f}%'
            })
        
        # Compute epoch metrics
        train_miou = np.mean(per_class_iou(hist)) * 100
        train_loss = train_loss / len(train_loader)
        
        # Validation phase
        model.eval()
        val_loss = 0
        hist = np.zeros((19, 19))
        
        with torch.no_grad():
            val_loop = tqdm(val_loader, desc=f'Epoch {epoch+1}/{num_epochs} (Val)')
            for images, labels in val_loop:
                images = images.to(device)
                labels = labels.to(device)
                
                outputs = model(images)
                if isinstance(outputs, tuple):
                    outputs = outputs[0]
                
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                
                predictions = outputs.argmax(1)
                hist += fast_hist(
                    labels.cpu().numpy(),
                    predictions.cpu().numpy(),
                    19
                )
                
                val_loop.set_postfix({'Loss': f'{loss.item():.4f}'})
        
        # Compute validation metrics
        val_miou = np.mean(per_class_iou(hist)) * 100
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
        
        # Print epoch summary
        print(f'\nEpoch {epoch+1}/{num_epochs}:')
        print(f'Train Loss: {train_loss:.4f}, mIoU: {train_miou:.2f}%')
        print(f'Val Loss: {val_loss:.4f}, mIoU: {val_miou:.2f}%')
        
        # Plot current progress
        plot_training_curves(history, save_dir / 'training_curves.png')
    
    return history['train_miou'], history['val_miou'], best_epoch