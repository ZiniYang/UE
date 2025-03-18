import time
import torch
import util

# Set device for computation
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class Trainer():
    """
    Handles model training process and metrics tracking.
    """
    def __init__(self, criterion, data_loader, logger, config, global_step=0,
                 target='train_dataset'):
        """
        Initialize the trainer.
        
        Args:
            criterion: Loss function for training
            data_loader: Dictionary of data loaders
            logger: Logger object for logging
            config: Configuration object
            global_step: Initial global step
            target: Key for the target dataset in data_loader
        """
        self.criterion = criterion
        self.data_loader = data_loader
        self.logger = logger
        self.config = config
        self.log_frequency = config.log_frequency if hasattr(config, 'log_frequency') else 100
        self.loss_meters = util.AverageMeter()
        self.acc_meters = util.AverageMeter()
        self.acc5_meters = util.AverageMeter()
        self.global_step = global_step
        self.target = target
        print(f"Training on: {self.target}")

    def _reset_stats(self):
        """Reset all tracking metrics."""
        self.loss_meters = util.AverageMeter()
        self.acc_meters = util.AverageMeter()
        self.acc5_meters = util.AverageMeter()

    def train(self, epoch, model, criterion, optimizer, random_noise=None):
        """
        Trains the model for one epoch.
        
        Args:
            epoch: Current epoch number
            model: Model to train
            criterion: Loss function
            optimizer: Optimizer for weight updates
            random_noise: Optional noise to add to samples (for unlearnable examples)
            
        Returns:
            Updated global step
        """
        model.train()
        
        # Iterate through batches
        for i, (images, labels) in enumerate(self.data_loader[self.target]):
            # Move data to device
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            
            # Apply noise if provided (for unlearnable examples)
            if random_noise is not None:
                random_noise = random_noise.detach().to(device)
                for i in range(len(labels)):
                    class_index = labels[i].item()
                    images[i] += random_noise[class_index].clone()
                    images[i] = torch.clamp(images[i], 0, 1)
            
            # Time the training step
            start = time.time()
            log_payload = self.train_batch(images, labels, model, optimizer)
            end = time.time()
            time_used = end - start
            
            # Log progress at specified frequency
            if self.global_step % self.log_frequency == 0:
                display = util.log_display(epoch=epoch,
                                         global_step=self.global_step,
                                         time_elapse=time_used,
                                         **log_payload)
                self.logger.info(display)
                
            self.global_step += 1
            
        return self.global_step

    def train_batch(self, images, labels, model, optimizer):
        """
        Trains the model on a single batch.
        
        Args:
            images: Input images
            labels: Target labels
            model: Model to train
            optimizer: Optimizer for weight updates
            
        Returns:
            Dictionary of training metrics
        """
        # Reset gradients
        model.zero_grad()
        optimizer.zero_grad()
        
        # Forward pass based on criterion type
        if isinstance(self.criterion, torch.nn.CrossEntropyLoss) or hasattr(self.criterion, '__name__') and self.criterion.__name__ == 'CutMixCrossEntropyLoss':
            logits = model(images)
            loss = self.criterion(logits, labels)
        else:
            # For custom loss functions that handle the forward pass internally
            logits, loss = self.criterion(model, images, labels, optimizer)
            
        # Handle one-hot encoded labels for CutMix
        if hasattr(self.criterion, '__name__') and self.criterion.__name__ == 'CutMixCrossEntropyLoss':
            _, labels = torch.max(labels.data, 1)
            
        # Backward pass and optimization
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), self.config.grad_clip)
        optimizer.step()
        
        # Calculate accuracy
        if logits.shape[1] >= 5:
            acc, acc5 = util.accuracy(logits, labels, topk=(1, 5))
            acc, acc5 = acc.item(), acc5.item()
        else:
            acc, = util.accuracy(logits, labels, topk=(1,))
            acc, acc5 = acc.item(), 1
            
        # Update metrics
        self.loss_meters.update(loss.item(), labels.shape[0])
        self.acc_meters.update(acc, labels.shape[0])
        self.acc5_meters.update(acc5, labels.shape[0])
        
        # Prepare log payload
        payload = {
            "acc": acc,
            "acc_avg": self.acc_meters.avg,
            "loss": loss.item(),
            "loss_avg": self.loss_meters.avg,
            "lr": optimizer.param_groups[0]['lr'],
            "|gn|": grad_norm
        }
        
        return payload