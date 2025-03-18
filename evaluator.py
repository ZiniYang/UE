import time
import torch
import torch.optim as optim
import util

# Set device for computation
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class Evaluator():
    """
    Handles model evaluation and metrics tracking.
    """
    def __init__(self, data_loader, logger, config):
        """
        Initialize the evaluator.
        
        Args:
            data_loader: Dictionary of data loaders
            logger: Logger object for logging
            config: Configuration object
        """
        self.loss_meters = util.AverageMeter()
        self.acc_meters = util.AverageMeter()
        self.acc5_meters = util.AverageMeter()
        self.criterion = torch.nn.CrossEntropyLoss()
        self.data_loader = data_loader
        self.logger = logger
        self.log_frequency = config.log_frequency if hasattr(config, 'log_frequency') else 100
        self.config = config
        self.current_acc = 0
        self.current_acc_top5 = 0
        self.confusion_matrix = torch.zeros(config.num_classes, config.num_classes)
        
    def _reset_stats(self):
        """Reset all tracking metrics."""
        self.loss_meters = util.AverageMeter()
        self.acc_meters = util.AverageMeter()
        self.acc5_meters = util.AverageMeter()
        self.confusion_matrix = torch.zeros(self.config.num_classes, self.config.num_classes)

    def eval(self, epoch, model):
        """
        Evaluates the model on the test dataset.
        
        Args:
            epoch: Current epoch number
            model: Model to evaluate
        """
        model.eval()
        
        # Evaluate all batches
        for i, (images, labels) in enumerate(self.data_loader["test_dataset"]):
            start = time.time()
            log_payload = self.eval_batch(images=images, labels=labels, model=model)
            end = time.time()
            time_used = end - start
            
        # Log final results
        display = util.log_display(epoch=epoch,
                                  global_step=i,
                                  time_elapse=time_used,
                                  **log_payload)
        if self.logger is not None:
            self.logger.info(display)
        # Calculate per class
        per_class_acc = {}
        for i in range(self.config.num_classes):
            if self.confusion_matrix[i].sum() > 0:
                per_class_acc[i] = (self.confusion_matrix[i, i] / self.confusion_matrix[i].sum()).item() * 100
            else:
                per_class_acc[i] = 0.0

        # Log per-class accuracy
        self.logger.info("Per-class accuracy:")
        for class_idx, acc in per_class_acc.items():
            self.logger.info(f"  Class {class_idx}: {acc:.2f}%")

        # Log accuracy for poisoned classes if specified
        if hasattr(self.config, 'poisoned_classes') and self.config.poisoned_classes:
            self.logger.info("Poisoned classes accuracy:")
            poisoned_acc = [per_class_acc[i] for i in self.config.poisoned_classes]
            self.logger.info(f"  Average: {sum(poisoned_acc)/len(poisoned_acc):.2f}%")
            for class_idx in self.config.poisoned_classes:
                self.logger.info(f"  Class {class_idx}: {per_class_acc[class_idx]:.2f}%")

    def eval_batch(self, images, labels, model):
        """
        Evaluates the model on a single batch.
        
        Args:
            images: Input images
            labels: Target labels
            model: Model to evaluate
            
        Returns:
            Dictionary of evaluation metrics
        """
        # Move data to device
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        
        # Forward pass without gradients
        with torch.no_grad():
            pred = model(images)
            loss = self.criterion(pred, labels)
            acc, acc5 = util.accuracy(pred, labels, topk=(1, 5))
            
            # Update confusion matrix
            _, preds = torch.max(pred, 1)
            for t, p in zip(labels.view(-1), preds.view(-1)):
                self.confusion_matrix[t.long(), p.long()] += 1

        # Update metrics
        self.loss_meters.update(loss.item(), n=images.size(0))
        self.acc_meters.update(acc.item(), n=images.size(0))
        self.acc5_meters.update(acc5.item(), n=images.size(0))
        
        # Prepare log payload
        payload = {
            "acc": acc.item(),
            "acc_avg": self.acc_meters.avg,
            "acc5": acc5.item(),
            "acc5_avg": self.acc5_meters.avg,
            "loss": loss.item(),
            "loss_avg": self.loss_meters.avg
        }
        
        return payload

    def _pgd_whitebox(self, model, X, y, random_start=True, epsilon=0.031, num_steps=20, step_size=0.003):
        """
        Evaluates model robustness using PGD attack.
        
        Args:
            model: Model to evaluate
            X: Input images
            y: Target labels
            random_start: Whether to use random initialization
            epsilon: Maximum perturbation magnitude
            num_steps: Number of PGD steps
            step_size: Size of each PGD step
            
        Returns:
            Tuple of (clean accuracy, robust accuracy)
        """
        model.eval()
        
        # Evaluate on clean data
        out = model(X)
        acc = (out.data.max(1)[1] == y.data).float().sum()
        
        # Initialize adversarial examples
        X_pgd = X.clone().requires_grad_(True)
        if random_start:
            random_noise = torch.FloatTensor(*X_pgd.shape).uniform_(-epsilon, epsilon).to(device)
            X_pgd = X_pgd.data + random_noise
            X_pgd = X_pgd.requires_grad_(True)

        # PGD attack
        for _ in range(num_steps):
            opt = optim.SGD([X_pgd], lr=1e-3)
            opt.zero_grad()

            with torch.enable_grad():
                loss = torch.nn.CrossEntropyLoss()(model(X_pgd), y)
            loss.backward()
            
            # Update adversarial examples
            eta = step_size * X_pgd.grad.data.sign()
            X_pgd = X_pgd.data + eta
            
            # Project back to epsilon ball and valid image range
            eta = torch.clamp(X_pgd.data - X.data, -epsilon, epsilon)
            X_pgd = X.data + eta
            X_pgd = torch.clamp(X_pgd, 0, 1.0).requires_grad_(True)
            
        # Evaluate on adversarial examples
        acc_pgd = (model(X_pgd).data.max(1)[1] == y.data).float().sum()
        
        return acc.item(), acc_pgd.item()