import torch
import time
from pathlib import Path
from tqdm import tqdm
from MLDL2024_project.utils import MetricTracker, PerformanceMonitor, Visualizer, LRSchedulerWithWarmup

class Trainer:
    def __init__(self, model, criterion, optimizer, device, num_classes, 
                 save_dir='checkpoints', exp_name='experiment'):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device
        self.num_classes = num_classes
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(exist_ok=True)
        self.exp_name = exp_name
        
        # Initialize helpers
        self.metric_tracker = MetricTracker(num_classes)
        self.visualizer = Visualizer()
        self.performance_monitor = PerformanceMonitor(model, device)
        
        # Metric history
        self.train_history = {'loss': [], 'miou': [], 'acc': [], 'lr': []}
        self.val_history = {'loss': [], 'miou': [], 'acc': []}
        
        # Best metrics
        self.best_miou = 0
        self.best_epoch = 0
        self.best_metrics = None

    def save_checkpoint(self, filename, epoch, metrics):
        """Save training checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'metrics': metrics,
            'best_miou': self.best_miou,
            'train_history': self.train_history,
            'val_history': self.val_history
        }
        torch.save(checkpoint, self.save_dir / filename)

    def train_one_epoch(self, train_loader, epoch, num_epochs):
        """Training for one epoch"""
        self.model.train()
        self.metric_tracker.reset()
        epoch_loss = 0

        progress = tqdm(enumerate(train_loader), total=len(train_loader))
        for iter, (images, labels) in progress:
            # Move data to device
            images = images.to(self.device)
            labels = labels.to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(images)
            
            # Handle auxiliary outputs (BiSeNet)
            if isinstance(outputs, tuple):
                main_out, aux2, aux3 = outputs
                loss = self.criterion(main_out, labels)
                if self.model.training:
                    aux_loss2 = self.criterion(aux2, labels)
                    aux_loss3 = self.criterion(aux3, labels)
                    loss = loss + 0.4 * aux_loss2 + 0.4 * aux_loss3
                outputs = main_out
            else:
                loss = self.criterion(outputs, labels)

            # Backward pass
            loss.backward()
            self.optimizer.step()

            # Update metrics
            preds = outputs.argmax(1)
            self.metric_tracker.update(labels.cpu().numpy(), preds.cpu().numpy())
            epoch_loss += loss.item()

            # Update progress bar
            metrics = self.metric_tracker.compute_metrics()
            progress.set_description(
                f"Epoch [{epoch+1}/{num_epochs}] "
                f"Loss: {loss.item():.4f} "
                f"mIoU: {metrics['miou']:.4f}"
            )

        # Compute epoch metrics
        final_metrics = self.metric_tracker.compute_metrics()
        final_metrics['loss'] = epoch_loss / len(train_loader)
        
        return final_metrics

    def validate(self, val_loader):
        """Validation loop"""
        self.model.eval()
        self.metric_tracker.reset()
        val_loss = 0

        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(self.device)
                labels = labels.to(self.device)

                outputs = self.model(images)
                if isinstance(outputs, tuple):
                    outputs = outputs[0]

                loss = self.criterion(outputs, labels)
                preds = outputs.argmax(1)

                self.metric_tracker.update(labels.cpu().numpy(), preds.cpu().numpy())
                val_loss += loss.item()

        final_metrics = self.metric_tracker.compute_metrics()
        final_metrics['loss'] = val_loss / len(val_loader)
        
        return final_metrics

    def train(self, train_loader, val_loader, num_epochs, scheduler=None, 
              early_stopping=10):
        """Complete training loop"""
        print(f"\nStarting training for {num_epochs} epochs...")
        start_time = time.time()
        epochs_without_improvement = 0

        for epoch in range(num_epochs):
            epoch_start = time.time()

            # Training phase
            train_metrics = self.train_one_epoch(train_loader, epoch, num_epochs)

            # Update history
            for k in self.train_history:
                if k in train_metrics:
                    self.train_history[k].append(train_metrics[k])

            # Validation phase
            val_metrics = self.validate(val_loader)
            for k in self.val_history:
                if k in val_metrics:
                    self.val_history[k].append(val_metrics[k])

            # Learning rate scheduling
            if scheduler is not None:
                if isinstance(scheduler, LRSchedulerWithWarmup):
                    lr = scheduler.step(epoch, 0, len(train_loader))
                else:
                    scheduler.step()
                    lr = scheduler.get_last_lr()[0]
                self.train_history['lr'].append(lr)

            # Save best model
            if val_metrics['miou'] > self.best_miou:
                self.best_miou = val_metrics['miou']
                self.best_epoch = epoch
                self.best_metrics = val_metrics
                self.save_checkpoint(f'{self.exp_name}_best.pth', epoch, val_metrics)
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1

            # Regular checkpoint
            if (epoch + 1) % 10 == 0:
                self.save_checkpoint(
                    f'{self.exp_name}_epoch_{epoch+1}.pth', 
                    epoch, 
                    val_metrics
                )

            # Print epoch summary
            epoch_time = time.time() - epoch_start
            print(f"\nEpoch {epoch+1}/{num_epochs} Summary:")
            print(f"Time: {epoch_time//60:.0f}m {epoch_time%60:.0f}s")
            print(f"Train Loss: {train_metrics['loss']:.4f}, mIoU: {train_metrics['miou']:.4f}")
            print(f"Val Loss: {val_metrics['loss']:.4f}, mIoU: {val_metrics['miou']:.4f}")

            # Early stopping check
            if epochs_without_improvement >= early_stopping:
                print(f"\nEarly stopping triggered after {early_stopping} epochs without improvement")
                break

        # Training complete
        total_time = time.time() - start_time
        print(f"\nTraining completed in {total_time//3600:.0f}h {(total_time%3600)//60:.0f}m")
        print(f"Best mIoU: {self.best_miou:.4f} at epoch {self.best_epoch+1}")

        # Plot training curves
        self.visualizer.plot_learning_curves(
            self.train_history,
            self.val_history,
            self.best_epoch,
            self.exp_name
        )

        return self.best_metrics

def train_model(model, criterion, optimizer, train_loader, val_loader, 
               class_names, device, num_epochs, model_name='model'):
    """Convenience function for training"""
    trainer = Trainer(
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        device=device,
        num_classes=len(class_names),
        exp_name=model_name
    )
    
    # Create learning rate scheduler with warmup
    scheduler = LRSchedulerWithWarmup(
        optimizer=optimizer,
        init_lr=0.01,
        warmup_epochs=5,
        max_epochs=num_epochs,
        warmup_method='linear',
        decay_method='poly'
    )
    
    # Train the model
    best_metrics = trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=num_epochs,
        scheduler=scheduler
    )
    
    return (
        trainer.train_history['miou'],
        trainer.val_history['miou'],
        trainer.best_epoch
    )