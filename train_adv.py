import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
from itertools import cycle
from torch.nn import functional as F
from pathlib import Path
from typing import Tuple, Dict, Any
from MLDL2024_project.utils import fast_hist, per_class_iou, poly_lr_scheduler, plot_training_curves


class CustomDiscriminator(nn.Module):
    """
    Discriminator network for adversarial training.
    Takes segmentation map as input and predicts if it's from source or target domain.
    """
    def __init__(self, num_classes: int, ndf: int = 32) -> None:
        super(CustomDiscriminator, self).__init__()
        self.conv1 = nn.Conv2d(num_classes, ndf, kernel_size=3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(ndf, ndf*2, kernel_size=3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(ndf*2, ndf*4, kernel_size=3, stride=2, padding=1)
        self.conv4 = nn.Conv2d(ndf*4, ndf*8, kernel_size=3, stride=2, padding=1)
        self.classifier = nn.Conv2d(ndf*8, 1, kernel_size=3, stride=2, padding=1)
        self.leaky_relu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.leaky_relu(self.conv1(x))
        x = self.leaky_relu(self.conv2(x))
        x = self.leaky_relu(self.conv3(x))
        x = self.leaky_relu(self.conv4(x))
        x = self.classifier(x)
        return x

def train_adversarial(
    generator: nn.Module,
    discriminator: nn.Module,
    gen_criterion: nn.Module,
    disc_criterion: nn.Module,
    gen_optimizer: torch.optim.Optimizer,
    disc_optimizer: torch.optim.Optimizer,
    source_loader: torch.utils.data.DataLoader,
    target_loader: torch.utils.data.DataLoader,
    source_interp: Any,
    target_interp: Any,
    num_classes: int,
    device: torch.device,
    epochs: int,
    save_dir: str = 'checkpoints',
    lambda_adv: float = 0.002,
    scheduler_mode: bool = True
) -> Tuple[list, list, int]:
    """
    Adversarial training function for domain adaptation.
    
    Args:
        generator: Generator network (segmentation model)
        discriminator: Discriminator network
        gen_criterion: Loss function for generator
        disc_criterion: Loss function for discriminator
        gen_optimizer: Optimizer for generator
        disc_optimizer: Optimizer for discriminator
        source_loader: DataLoader for source domain
        target_loader: DataLoader for target domain
        source_interp: Interpolation for source domain
        target_interp: Interpolation for target domain
        num_classes: Number of classes
        device: Device to run on
        epochs: Number of epochs
        save_dir: Directory to save checkpoints
        lambda_adv: Weight for adversarial loss
        scheduler_mode: Whether to use learning rate scheduler
    
    Returns:
        train_mious: List of training mIoUs
        val_mious: List of validation mIoUs
        best_epoch: Best epoch number
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(exist_ok=True)
    
    best_miou = 0.0
    best_epoch = 0
    train_mious = []
    val_mious = []
    
    def print_iou_per_class(hist: np.ndarray, phase: str = 'train') -> None:
        class_names = source_loader.dataset.get_class_names()
        iou_per_class = per_class_iou(hist) * 100
        print(f"\n{phase.capitalize()} IoU per class:")
        for class_idx, iou in enumerate(iou_per_class):
            print(f"{class_names[class_idx]:20s}: {iou:.2f}%")
    
    for epoch in range(epochs):
        generator.train()
        discriminator.train()
        
        train_loss = 0
        train_hist = np.zeros((num_classes, num_classes))
        
        target_iter = cycle(target_loader)
        train_loop = tqdm(enumerate(source_loader), total=len(source_loader),
                         desc=f'Epoch {epoch+1}/{epochs} (Train)')
        
        for batch_idx, (source_imgs, source_labels) in train_loop:
            target_imgs, _ = next(target_iter)
            
            source_imgs = source_imgs.to(device)
            source_labels = source_labels.to(device)
            target_imgs = target_imgs.to(device)

            # Train Generator
            gen_optimizer.zero_grad()
            disc_optimizer.zero_grad()
            
            for param in discriminator.parameters():
                param.requires_grad = False
                
            # Source domain
            pred_source = generator(source_imgs)
            if isinstance(pred_source, tuple):
                pred_source = pred_source[0]
            pred_source = source_interp(pred_source)
            loss_seg = gen_criterion(pred_source, source_labels)
            
            # Target domain
            pred_target = generator(target_imgs)
            if isinstance(pred_target, tuple):
                pred_target = pred_target[0]
            pred_target = target_interp(pred_target)
            
            disc_pred = discriminator(F.softmax(pred_target, dim=1))
            loss_adv = disc_criterion(disc_pred, 
                                    torch.ones_like(disc_pred).to(device))
            
            gen_loss = loss_seg + lambda_adv * loss_adv
            gen_loss.backward()
            
            if scheduler_mode:
                current_iter = epoch * len(source_loader) + batch_idx
                lr = poly_lr_scheduler(
                    gen_optimizer,
                    init_lr=0.01,
                    iter=current_iter,
                    max_iter=epochs * len(source_loader)
                )
            
            gen_optimizer.step()

            # Train Discriminator
            for param in discriminator.parameters():
                param.requires_grad = True
                
            disc_optimizer.zero_grad()
            
            # Real (source) samples
            disc_pred_source = discriminator(F.softmax(pred_source.detach(), dim=1))
            loss_disc_source = disc_criterion(disc_pred_source,
                                           torch.ones_like(disc_pred_source))
            
            # Fake (target) samples
            disc_pred_target = discriminator(F.softmax(pred_target.detach(), dim=1))
            loss_disc_target = disc_criterion(disc_pred_target,
                                           torch.zeros_like(disc_pred_target))
            
            disc_loss = (loss_disc_source + loss_disc_target) * 0.5
            disc_loss.backward()
            disc_optimizer.step()

            train_loss += gen_loss.item()
            predictions = torch.argmax(pred_source, dim=1)
            train_hist += fast_hist(
                source_labels.cpu().numpy(),
                predictions.cpu().numpy(),
                num_classes
            )
            
            miou = np.mean(per_class_iou(train_hist)) * 100
            train_loop.set_postfix({
                'Loss': f'{gen_loss.item():.4f}',
                'mIoU': f'{miou:.2f}%'
            })

        train_miou = np.mean(per_class_iou(train_hist)) * 100
        train_loss = train_loss / len(source_loader)
        
        # Validation
        generator.eval()
        val_loss = 0
        val_hist = np.zeros((num_classes, num_classes))
        
        with torch.no_grad():
            val_loop = tqdm(target_loader, desc=f'Epoch {epoch+1}/{epochs} (Val)')
            for target_imgs, target_labels in val_loop:
                target_imgs = target_imgs.to(device)
                target_labels = target_labels.to(device)
                
                outputs = generator(target_imgs)
                if isinstance(outputs, tuple):
                    outputs = outputs[0]
                
                loss = gen_criterion(outputs, target_labels)
                val_loss += loss.item()
                
                predictions = torch.argmax(outputs, dim=1)
                val_hist += fast_hist(
                    target_labels.cpu().numpy(),
                    predictions.cpu().numpy(),
                    num_classes
                )
                
                val_loop.set_postfix({'Loss': f'{loss.item():.4f}'})
        
        val_miou = np.mean(per_class_iou(val_hist)) * 100
        val_loss = val_loss / len(target_loader)
        
        train_mious.append(train_miou)
        val_mious.append(val_miou)
        
        # Save best model
        if val_miou > best_miou:
            best_miou = val_miou
            best_epoch = epoch
            torch.save({
                'epoch': epoch,
                'generator_state_dict': generator.state_dict(),
                'discriminator_state_dict': discriminator.state_dict(),
                'gen_optimizer_state_dict': gen_optimizer.state_dict(),
                'disc_optimizer_state_dict': disc_optimizer.state_dict(),
                'best_miou': best_miou,
            }, save_dir / 'best_model.pth')
        
        # Print epoch summary
        print(f'\nEpoch {epoch+1}/{epochs}:')
        print(f'Train Loss: {train_loss:.4f}, mIoU: {train_miou:.2f}%')
        print_iou_per_class(train_hist, 'train')
        print(f'\nVal Loss: {val_loss:.4f}, mIoU: {val_miou:.2f}%')
        print_iou_per_class(val_hist, 'val')
        
        # Plot training curves
        plot_training_curves({
            'train_miou': train_mious,
            'val_miou': val_mious
        }, save_dir / 'training_curves.png')
    
    return train_mious, val_mious, best_epoch

# Usage example:
"""
# Initialize models and optimizers
generator = YourGenerator(num_classes=19).to(device)
discriminator = CustomDiscriminator(num_classes=19).to(device)

# Loss functions
gen_criterion = nn.CrossEntropyLoss(ignore_index=255)
disc_criterion = nn.BCEWithLogitsLoss()

# Optimizers
gen_optimizer = torch.optim.SGD(generator.parameters(), lr=0.01, momentum=0.9)
disc_optimizer = torch.optim.Adam(discriminator.parameters(), lr=1e-4, betas=(0.9, 0.99))

# Train
train_mious, val_mious, best_epoch = train_adversarial(
    generator=generator,
    discriminator=discriminator,
    gen_criterion=gen_criterion,
    disc_criterion=disc_criterion,
    gen_optimizer=gen_optimizer,
    disc_optimizer=disc_optimizer,
    source_loader=source_loader,
    target_loader=target_loader,
    source_interp=source_interp,
    target_interp=target_interp,
    num_classes=19,
    device=device,
    epochs=50
)
"""