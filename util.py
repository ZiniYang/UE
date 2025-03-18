import logging
import os
import numpy as np
import torch
import matplotlib as plt

# Set device for computation
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Configure CUDA if available
if torch.cuda.is_available():
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = True


def _patch_noise_extend_to_img(noise, image_size=[3, 32, 32], patch_location='center'):
    """
    Extends a noise patch to the full image size by placing it at a specified location.
    
    Args:
        noise: The noise patch to extend
        image_size: Target image dimensions [channels, height, width]
        patch_location: Where to place the patch ('center' or 'random')
        
    Returns:
        Full-sized noise mask with the patch applied
    """
    c, h, w = image_size[0], image_size[1], image_size[2]
    mask = np.zeros((c, h, w), np.float32)
    x_len, y_len = noise.shape[1], noise.shape[2]

    # Determine patch placement location
    if patch_location == 'center' or (h == w == x_len == y_len):
        x = h // 2
        y = w // 2
    elif patch_location == 'random':
        x = np.random.randint(x_len // 2, w - x_len // 2)
        y = np.random.randint(y_len // 2, h - y_len // 2)
    else:
        raise ValueError('Invalid patch location')

    # Apply patch to mask
    x1 = np.clip(x - x_len // 2, 0, h)
    x2 = np.clip(x + x_len // 2, 0, h)
    y1 = np.clip(y - y_len // 2, 0, w)
    y2 = np.clip(y + y_len // 2, 0, w)
    mask[:, x1: x2, y1: y2] = noise
    
    return mask


def setup_logger(name, log_file, level=logging.INFO):
    """
    Sets up a logger with console and file handlers.
    
    Args:
        name: Logger name
        log_file: Path to log file
        level: Logging level
        
    Returns:
        Configured logger object
    """
    formatter = logging.Formatter('%(asctime)s %(message)s')
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    
    # File handler
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(formatter)
    
    # Configure logger
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger


def log_display(epoch, global_step, time_elapse, **kwargs):
    """
    Creates a formatted log display string.
    
    Args:
        epoch: Current epoch
        global_step: Global training step
        time_elapse: Time elapsed for the current step
        **kwargs: Additional metrics to display
        
    Returns:
        Formatted log string
    """
    display = f'epoch={epoch}\tglobal_step={global_step}'
    
    # Add additional metrics
    for key, value in kwargs.items():
        if isinstance(value, str):
            display = f'\t{key}={value}'
        else:
            display += f'\t{key}=%.4f' % value
            
    # Add throughput information
    display += f'\ttime=%.2fit/s' % (1. / time_elapse)
    
    return display


def accuracy(output, target, topk=(1,)):
    """
    Computes the accuracy over the k top predictions.
    
    Args:
        output: Model output logits
        target: Ground truth labels
        topk: Tuple of k values to compute accuracy for
        
    Returns:
        List of accuracies for each k
    """
    maxk = max(topk)
    batch_size = target.size(0)

    # Get top-k predictions
    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    # Calculate accuracies for each k
    res = []
    for k in topk:
        correct_k = correct[:k].float().sum()
        res.append(correct_k.mul_(1/batch_size))
        
    return res


def save_model(filename, epoch, model, optimizer, scheduler, save_best=False, **kwargs):
    """
    Saves model checkpoint.
    
    Args:
        filename: Base filename for saving
        epoch: Current epoch
        model: Model to save
        optimizer: Optimizer state to save
        scheduler: Scheduler state to save
        save_best: Whether to also save as best model
        **kwargs: Additional items to save
    """
    # Prepare state dictionary
    state = {
        'epoch': epoch + 1,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict() if scheduler is not None else None
    }
    
    # Add additional items
    for key, value in kwargs.items():
        state[key] = value
        
    # Save checkpoint
    torch.save(state, filename + '.pth')
    
    # Save best model if specified
    if save_best:
        best_filename = filename + '_best.pth'
        torch.save(state, best_filename)


def load_model(filename, model, optimizer, scheduler, **kwargs):
    """
    Loads model from checkpoint.
    
    Args:
        filename: Path to checkpoint file
        model: Model to load weights into
        optimizer: Optimizer to load state into
        scheduler: Scheduler to load state into
        
    Returns:
        Loaded checkpoint dictionary
    """
    # Load checkpoint
    filename = filename + '.pth'
    checkpoints = torch.load(filename, map_location=device)
    
    # Restore model and optimizer states
    model.load_state_dict(checkpoints['model_state_dict'])
    
    if optimizer is not None and checkpoints['optimizer_state_dict'] is not None:
        optimizer.load_state_dict(checkpoints['optimizer_state_dict'])
        
    if scheduler is not None and checkpoints['scheduler_state_dict'] is not None:
        scheduler.load_state_dict(checkpoints['scheduler_state_dict'])
        
    return checkpoints


def count_parameters_in_MB(model):
    """
    Counts the number of trainable parameters in a model.
    
    Args:
        model: PyTorch model
        
    Returns:
        Number of parameters in millions
    """
    return sum(np.prod(v.size()) for name, v in model.named_parameters() if "auxiliary_head" not in name) / 1e6


def build_dirs(path):
    """
    Creates directory if it doesn't exist.
    
    Args:
        path: Directory path to create
    """
    if not os.path.exists(path):
        os.makedirs(path)


class AverageMeter(object):
    """
    Computes and stores the average and current value.
    """
    def __init__(self):
        self.reset()

    def reset(self):
        """Resets all metrics."""
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
        self.max = 0

    def update(self, val, n=1):
        """
        Updates meter with new value.
        
        Args:
            val: New value
            n: Number of instances (for weighted average)
        """
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
        self.max = max(self.max, val)


def onehot(size, target):
    """
    Creates a one-hot encoded vector.
    
    Args:
        size: Size of the vector
        target: Index to set to 1
        
    Returns:
        One-hot encoded vector
    """
    vec = torch.zeros(size, dtype=torch.float32)
    vec[target] = 1.
    return vec


def rand_bbox(size, lam):
    """
    Generates a random bounding box for CutMix.
    
    Args:
        size: Image size
        lam: Lambda parameter controlling box size
        
    Returns:
        Bounding box coordinates (x1, y1, x2, y2)
    """
    if len(size) == 4:
        W = size[2]
        H = size[3]
    elif len(size) == 3:
        W = size[1]
        H = size[2]
    else:
        raise ValueError("Unexpected size format")

    # Calculate cut size
    cut_rat = np.sqrt(1. - lam)
    cut_w = np.int(W * cut_rat)
    cut_h = np.int(H * cut_rat)

    # Get random center point
    cx = np.random.randint(W)
    cy = np.random.randint(H)

    # Get bounding box coordinates
    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    return bbx1, bby1, bbx2, bby2

# Add these functions to your util.py file

def plot_training_results(checkpoint_path, output_dir=None, smooth=True, window_size=5):
    """
    Plot and save training results (accuracy curves and confusion matrix).
    Args:
        checkpoint_path: Path to model checkpoint file
        output_dir: Directory to save plots (uses parent of checkpoints directory if None)
        smooth: Apply curve smoothing
        window_size: Window size for smoothing
    """
    try:
        import os
        import numpy as np
        import matplotlib
        matplotlib.use('Agg')  # Use non-interactive backend
        import matplotlib.pyplot as plt
        
        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))
        
        # Determine output directory - save one level up from checkpoints
        if output_dir is None:
            checkpoint_dir = os.path.dirname(checkpoint_path)
            # Go one level up from "checkpoints" directory
            output_dir = os.path.dirname(checkpoint_dir)
        os.makedirs(output_dir, exist_ok=True)
        
        # Extract data
        train_acc = np.array(checkpoint['ENV']['train_history'])
        test_acc = np.array(checkpoint['ENV']['eval_history'])
        
        # Apply smoothing
        if smooth and len(train_acc) > window_size:
            smoothed_train = np.convolve(train_acc, np.ones(window_size)/window_size, mode='valid')
            padding = np.ones(window_size-1) * train_acc[0]
            smoothed_train = np.concatenate([padding, smoothed_train])
        else:
            smoothed_train = train_acc
        
        # Create accuracy plot
        plt.figure(figsize=(10, 6))
        epochs = np.arange(1, len(train_acc) + 1)
        
        plt.plot(epochs, smoothed_train, 'b-', label=f'Train ({train_acc[-1]:.2f}%)')
        plt.plot(epochs, test_acc, 'r-', label=f'Test ({test_acc[-1]:.2f}%)')
        
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy (%)')
        plt.title('Training and Test Accuracy')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.7)
        
        # Save plot
        acc_path = os.path.join(output_dir, 'accuracy.png')
        plt.savefig(acc_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        # Create confusion matrix plot if available
        if 'cm_history' in checkpoint['ENV'] and checkpoint['ENV']['cm_history']:
            cm = np.array(checkpoint['ENV']['cm_history'][-1])  # Get last epoch CM
            
            # Set class names
            if cm.shape[0] == 10:  # CIFAR-10
                class_names = ['airplane', 'auto', 'bird', 'cat', 'deer', 
                              'dog', 'frog', 'horse', 'ship', 'truck']
            else:
                class_names = [str(i) for i in range(cm.shape[0])]
            
            # Normalize
            cm = cm.astype('float') / (cm.sum(axis=1)[:, np.newaxis] + 1e-10)
            
            # Plot
            plt.figure(figsize=(10, 8))
            plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
            plt.title('Confusion Matrix')
            plt.colorbar()
            
            tick_marks = np.arange(len(class_names))
            plt.xticks(tick_marks, class_names, rotation=45)
            plt.yticks(tick_marks, class_names)
            
            plt.ylabel('True')
            plt.xlabel('Predicted')
            plt.tight_layout()
            
            # Save plot
            cm_path = os.path.join(output_dir, 'confusion_matrix.png')
            plt.savefig(cm_path, dpi=300, bbox_inches='tight')
            plt.close()
        
        print(f"Plots saved to {output_dir}")
        return True
    except Exception as e:
        print(f"Error generating plots: {str(e)}")
        return False


def compare_models(checkpoint_paths, labels, output_path, smooth=True, window_size=5):
    """
    Compare multiple model checkpoints.
    Args:
        checkpoint_paths: List of checkpoint paths
        labels: List of model labels
        output_path: Output image path
        smooth: Apply curve smoothing
        window_size: Window size for smoothing
    """
    try:
        import os
        import numpy as np
        import matplotlib
        matplotlib.use('Agg')  # Use non-interactive backend
        import matplotlib.pyplot as plt
        
        plt.figure(figsize=(12, 7))
        colors = ['b', 'r', 'g', 'c', 'm', 'y', 'k', 'orange']
        
        for i, (path, label) in enumerate(zip(checkpoint_paths, labels)):
            # Skip if file doesn't exist
            if not os.path.exists(path):
                print(f"Warning: Checkpoint {path} does not exist. Skipping.")
                continue
                
            # Load checkpoint
            try:
                checkpoint = torch.load(path, map_location=torch.device('cpu'))
            except Exception as e:
                print(f"Error loading checkpoint {path}: {str(e)}. Skipping.")
                continue
            
            # Extract test accuracy
            if 'ENV' not in checkpoint or 'eval_history' not in checkpoint['ENV']:
                print(f"Warning: Checkpoint {path} doesn't contain evaluation history. Skipping.")
                continue
                
            test_acc = np.array(checkpoint['ENV']['eval_history'])
            
            # Apply smoothing
            if smooth and len(test_acc) > window_size:
                smoothed_test = np.convolve(test_acc, np.ones(window_size)/window_size, mode='valid')
                padding = np.ones(window_size-1) * test_acc[0]
                smoothed_test = np.concatenate([padding, smoothed_test])
            else:
                smoothed_test = test_acc
            
            # Plot curve
            color = colors[i % len(colors)]
            epochs = np.arange(1, len(test_acc) + 1)
            plt.plot(epochs, smoothed_test, f'{color}-', label=f'{label} ({test_acc[-1]:.2f}%)')
        
        # Add chart elements
        plt.xlabel('Epoch')
        plt.ylabel('Test Accuracy (%)')
        plt.title('Model Comparison')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.7)
        
        # Create output directory if it doesn't exist
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Save figure
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Comparison plot saved to {output_path}")
        return True
    except Exception as e:
        print(f"Error comparing models: {str(e)}")
        return False