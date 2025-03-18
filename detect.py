import argparse
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import dataset
import mlconfig
import util
import time
from tqdm import tqdm
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.data import DataLoader, Subset

# Set device for computation
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Parse command-line arguments
parser = argparse.ArgumentParser(description='Detect Poisoned Samples')
# General Options
parser.add_argument('--seed', type=int, default=0, help='Random seed')
parser.add_argument('--detect_dir', type=str, default="detect_results", help='Directory for detection results')
parser.add_argument('--detection_method', type=str, default='simple-2NN', 
                   choices=['bias-0.5', 'bias+0.5', 'simple-linear', 'simple-2NN'],
                   help='Detection method')
parser.add_argument('--poison_method', type=str, default='classwise', help='Poisoning method used')
parser.add_argument('--no_cuda', action='store_true', default=False, help='Disable CUDA')
parser.add_argument('--gpu_id', type=str, default='0', help='GPU ID to use')
# Dataset Options
parser.add_argument('--batch_size', type=int, default=128, help='Batch size for detection')
parser.add_argument('--data', type=str, default='../datasets', help='Path to dataset')
parser.add_argument('--dataset', type=str, default='CIFAR-10', help='Dataset name')
parser.add_argument('--data_augmentation', action='store_true', default=False, help='Use data augmentation')
# Training Options
parser.add_argument('--victim_model', type=str, default='resnet18', help='Victim model architecture')
parser.add_argument('--epochs', type=int, default=100, help='Number of training epochs')
parser.add_argument('--detect_lr', type=float, default=0.01, help='Learning rate for detection model')
parser.add_argument('--momentum', type=float, default=0.9, help='SGD momentum')
parser.add_argument('--weight_decay', type=float, default=5e-4, help='Weight decay')
# Poison Options
parser.add_argument('--perturb_tensor_filepath', type=str, required=True, help='Path to perturbation tensor file')
parser.add_argument('--perturb_type', default='classwise', type=str, 
                   choices=['classwise', 'samplewise'], help='Perturbation type')
parser.add_argument('--patch_location', default='center', type=str, 
                   choices=['center', 'random'], help='Location of the noise patch')
parser.add_argument('--poison_rate', default=1.0, type=float, help='Portion of dataset to poison')
parser.add_argument('--poison_classwise', action='store_true', default=False, help='Poison specific classes')
parser.add_argument('--poison_classwise_idx', nargs='+', type=int, default=None, help='Indices of classes to poison')
parser.add_argument('--poison_class_percentage', default=1.0, type=float, help='Percentage of class to poison')
args = parser.parse_args()

class Linear(nn.Module):
    """Simple linear model for detection."""
    def __init__(self, input_dims=3072, num_classes=10):
        super(Linear, self).__init__()
        self.linear = nn.Linear(input_dims, num_classes)
        
    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.linear(x)
        return x

class TwoLayerNN(nn.Module):
    """Two-layer neural network for detection."""
    def __init__(self, input_dims=3072, hidden_dims=512, num_classes=10):
        super(TwoLayerNN, self).__init__()
        self.fc1 = nn.Linear(input_dims, hidden_dims)
        self.fc2 = nn.Linear(hidden_dims, num_classes)
        
    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

def train(model, device, train_loader, optimizer, epoch, attack='None'):
    """Train the detection model."""
    model.train()
    for batch_idx, batch_data in enumerate(train_loader):
        # Handle different data formats
        if isinstance(batch_data, (list, tuple)) and len(batch_data) == 3:
            data, target, poison_idx = batch_data
            
            # Handle case where data is a list of tensors (from bias methods)
            if isinstance(data, list):
                data = torch.stack(data)
        else:
            # Default unpacking
            try:
                data, target, poison_idx = batch_data
            except Exception as e:
                print(f"Warning: Unexpected batch format: {type(batch_data)}, error: {e}")
                continue
                
        # Move data to device
        data, target = data.to(device), target.to(device)
        
        # Forward pass and optimization
        optimizer.zero_grad()
        output = model(data)
        loss = F.cross_entropy(output, target)
        loss.backward()
        optimizer.step()
        
        if batch_idx % 100 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))

def eval_train(model, device, train_loader):
    """Evaluate model on training set."""
    model.eval()
    train_loss = 0
    correct = 0
    with torch.no_grad():
        for batch_data in train_loader:
            # Handle different data formats
            if isinstance(batch_data, (list, tuple)) and len(batch_data) == 3:
                data, target, poison_idx = batch_data
                
                # Handle case where data is a list of tensors
                if isinstance(data, list):
                    data = torch.stack(data)
            else:
                try:
                    data, target, poison_idx = batch_data
                except:
                    continue
                
            data, target = data.to(device), target.to(device)
            output = model(data)
            train_loss += F.cross_entropy(output, target, reduction='sum').item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
    
    train_loss /= len(train_loader.dataset)
    train_accuracy = 100. * correct / len(train_loader.dataset)
    
    return train_loss, train_accuracy

def eval_test(model, device, test_loader):
    """Evaluate model on validation/test set."""
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for batch_data in test_loader:
            # Handle different data formats
            if isinstance(batch_data, (list, tuple)) and len(batch_data) == 3:
                data, target, poison_idx = batch_data
                
                # Handle case where data is a list of tensors
                if isinstance(data, list):
                    data = torch.stack(data)
            else:
                try:
                    data, target, poison_idx = batch_data
                except:
                    continue
                
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.cross_entropy(output, target, reduction='sum').item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
    
    test_loss /= len(test_loader.dataset)
    test_accuracy = 100. * correct / len(test_loader.dataset)
    
    return test_loss, test_accuracy

def analyze_detector(model, device, data_loader):
    """Analyze poisoned samples using the trained detector."""
    model.eval()
    predictions = []
    targets = []
    is_poisoned = []
    original_data = []
    
    with torch.no_grad():
        for batch_data in data_loader:
            # Handle different data formats
            if isinstance(batch_data, (list, tuple)) and len(batch_data) == 3:
                data, target, poison_idx = batch_data
                
                # Handle case where data is a list of tensors
                if isinstance(data, list):
                    data = torch.stack(data)
            else:
                try:
                    data, target, poison_idx = batch_data
                except:
                    continue
                
            data, target = data.to(device), target.to(device)
            output = model(data)
            pred = output.argmax(dim=1)
            
            predictions.extend(pred.cpu().numpy())
            targets.extend(target.cpu().numpy())
            is_poisoned.extend(poison_idx.cpu().numpy())
            original_data.extend(data.cpu().numpy())
    
    # Calculate metrics
    predictions = np.array(predictions)
    targets = np.array(targets)
    is_poisoned = np.array(is_poisoned)
    
    # Accuracy on clean vs poisoned samples
    clean_idx = (is_poisoned == 0)
    poison_idx = (is_poisoned == 1)
    
    clean_acc = 100 * np.mean(predictions[clean_idx] == targets[clean_idx]) if np.any(clean_idx) else 0
    poison_acc = 100 * np.mean(predictions[poison_idx] == targets[poison_idx]) if np.any(poison_idx) else 0
    
    print(f"Clean samples accuracy: {clean_acc:.2f}%")
    print(f"Poisoned samples accuracy: {poison_acc:.2f}%")
    
    # Calculate loss differences between clean and poisoned samples
    losses = []
    for i in range(len(predictions)):
        data = torch.tensor(original_data[i]).unsqueeze(0).to(device)
        target = torch.tensor([targets[i]]).to(device)
        
        output = model(data)
        loss = F.cross_entropy(output, target).item()
        losses.append(loss)
    
    losses = np.array(losses)
    clean_loss = np.mean(losses[clean_idx]) if np.any(clean_idx) else 0
    poison_loss = np.mean(losses[poison_idx]) if np.any(poison_idx) else 0
    
    print(f"Clean samples loss: {clean_loss:.4f}")
    print(f"Poisoned samples loss: {poison_loss:.4f}")
    
    # Calculate confidence differences
    confidences = []
    with torch.no_grad():
        for batch_data in data_loader:
            # Handle different data formats
            if isinstance(batch_data, (list, tuple)) and len(batch_data) == 3:
                data, target, _ = batch_data
                
                # Handle case where data is a list of tensors
                if isinstance(data, list):
                    data = torch.stack(data)
            else:
                try:
                    data, target, _ = batch_data
                except:
                    continue
                
            data, target = data.to(device), target.to(device)
            output = model(data)
            sm_output = F.softmax(output, dim=1)
            confidence, _ = torch.max(sm_output, dim=1)
            confidences.extend(confidence.cpu().numpy())
    
    confidences = np.array(confidences)
    clean_conf = np.mean(confidences[clean_idx]) if np.any(clean_idx) else 0
    poison_conf = np.mean(confidences[poison_idx]) if np.any(poison_idx) else 0
    
    print(f"Clean samples confidence: {clean_conf:.4f}")
    print(f"Poisoned samples confidence: {poison_conf:.4f}")
    
    return {
        'clean_acc': clean_acc,
        'poison_acc': poison_acc,
        'clean_loss': clean_loss,
        'poison_loss': poison_loss,
        'clean_conf': clean_conf,
        'poison_conf': poison_conf,
        'predictions': predictions,
        'targets': targets,
        'is_poisoned': is_poisoned,
        'confidences': confidences
    }

class CustomDataset(torch.utils.data.Dataset):
    """Dataset wrapper that adds poisoning information."""
    def __init__(self, dataset, poisoned_indices):
        self.dataset = dataset
        self.poisoned_indices = set(poisoned_indices)
        
    def __getitem__(self, index):
        data, target = self.dataset[index]
        is_poisoned = 1 if index in self.poisoned_indices else 0
        return data, target, torch.tensor(is_poisoned)
    
    def __len__(self):
        return len(self.dataset)

def main():
    """Main function for poison detection."""
    # Set up experiment directory
    dir = args.detect_dir
    if not os.path.exists(dir):
        os.makedirs(dir)
    
    # Set up logging
    log_file_path = os.path.join(dir, f'detection-{args.detection_method}-{args.poison_method}.log')
    logger = util.setup_logger(name='detection', log_file=log_file_path)
    
    # Set GPU
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
    device = torch.device("cuda" if not args.no_cuda else "cpu")
    
    # Log parameters
    logger.info(f'Running poison detection with {args.detection_method} for {args.poison_method}')
    logger.info(f'Victim model: {args.victim_model}, learning rate: {args.detect_lr}')
    for arg in vars(args):
        logger.info("%s: %s" % (arg, getattr(args, arg)))
    
    # Load dataset
    logger.info("Loading dataset...")
    try:
        datasets_generator = dataset.DatasetGenerator(
            train_batch_size=args.batch_size,
            eval_batch_size=args.batch_size,
            train_data_type='PoisonCIFAR10',  # We're using CIFAR10
            train_data_path=args.data,
            test_data_type='CIFAR10',
            test_data_path=args.data,
            num_of_workers=2,  # Reduced number of workers for stability
            poison_rate=args.poison_rate,
            perturb_type=args.perturb_type,
            patch_location=args.patch_location,
            perturb_tensor_filepath=args.perturb_tensor_filepath,
            poison_classwise=args.poison_classwise,
            poison_classwise_idx=args.poison_classwise_idx,
            poison_class_percentage=args.poison_class_percentage,
            seed=args.seed
        )
    except Exception as e:
        logger.error(f"Error loading dataset: {e}")
        return
    
    # Get original dataset and create custom dataset with poison information
    poison_data = datasets_generator.datasets['train_dataset']
    poisoned_indices = poison_data.poison_samples_idx
    
    # Create custom dataset with poison flag
    custom_dataset = CustomDataset(poison_data, poisoned_indices)
    
    # Split into train/validation
    data_size = len(custom_dataset)
    train_size = int(0.8 * data_size)
    val_size = data_size - train_size
    
    torch.manual_seed(args.seed)
    indices = torch.randperm(data_size).tolist()
    train_indices = indices[:train_size]
    val_indices = indices[train_size:]
    
    train_set = Subset(custom_dataset, train_indices)
    val_set = Subset(custom_dataset, val_indices)
    
    # Prepare data loaders with reduced number of workers
    if args.detection_method == 'bias-0.5':
        # Add bias of -0.5 to the inputs with torch.stack
        train_loader = torch.utils.data.DataLoader(
            train_set, batch_size=args.batch_size, shuffle=True,
            num_workers=2, pin_memory=True,
            collate_fn=lambda batch: (torch.stack([x[0] - 0.5 for x in batch]), 
                                      torch.tensor([x[1] for x in batch]), 
                                      torch.tensor([x[2] for x in batch]))
        )
        val_loader = torch.utils.data.DataLoader(
            val_set, batch_size=args.batch_size, shuffle=False,
            num_workers=2, pin_memory=True,
            collate_fn=lambda batch: (torch.stack([x[0] - 0.5 for x in batch]), 
                                      torch.tensor([x[1] for x in batch]), 
                                      torch.tensor([x[2] for x in batch]))
        )
    elif args.detection_method == 'bias+0.5':
        # Add bias of +0.5 to the inputs with torch.stack
        train_loader = torch.utils.data.DataLoader(
            train_set, batch_size=args.batch_size, shuffle=True,
            num_workers=2, pin_memory=True,
            collate_fn=lambda batch: (torch.stack([x[0] + 0.5 for x in batch]), 
                                      torch.tensor([x[1] for x in batch]), 
                                      torch.tensor([x[2] for x in batch]))
        )
        val_loader = torch.utils.data.DataLoader(
            val_set, batch_size=args.batch_size, shuffle=False,
            num_workers=2, pin_memory=True,
            collate_fn=lambda batch: (torch.stack([x[0] + 0.5 for x in batch]), 
                                      torch.tensor([x[1] for x in batch]), 
                                      torch.tensor([x[2] for x in batch]))
        )
    else:
        # Regular data loading
        train_loader = torch.utils.data.DataLoader(
            train_set, batch_size=args.batch_size, shuffle=True,
            num_workers=2, pin_memory=True
        )
        val_loader = torch.utils.data.DataLoader(
            val_set, batch_size=args.batch_size, shuffle=False,
            num_workers=2, pin_memory=True
        )
    
    # Initialize model based on detection method
    input_dims = 32 * 32 * 3  # CIFAR image size
    
    if args.detection_method == 'bias-0.5' or args.detection_method == 'bias+0.5':
        from models.ResNet import ResNet18
        model = ResNet18(num_classes=10).to(device)
    elif args.detection_method == 'simple-linear':
        model = Linear(input_dims=input_dims).to(device)
    elif args.detection_method == 'simple-2NN':
        model = TwoLayerNN(input_dims=input_dims).to(device)
    else:
        raise ValueError(f"Unknown detection method: {args.detection_method}")
    
    model = torch.nn.DataParallel(model).to(device)
    
    # Initialize optimizer and scheduler
    optimizer = optim.SGD(model.parameters(), lr=args.detect_lr,
                          momentum=args.momentum, weight_decay=args.weight_decay)
    scheduler = MultiStepLR(optimizer=optimizer, milestones=[75, 90], gamma=0.1)
    
    # Training loop
    logger.info("Starting training...")
    logger.info('Epoch \t Train Time \t Test Time \t \t Train Loss \t Train Acc \t Test Loss \t Test Acc')
    
    best_val_acc = 0
    best_model_path = os.path.join(dir, f'best_model_{args.detection_method}.pth')
    
    for epoch in range(args.epochs):
        start_time = time.time()
        
        try:
            train(model, device, train_loader, optimizer, epoch)
            train_time = time.time()
            
            train_loss, train_accuracy = eval_train(model, device, train_loader)
            val_loss, val_accuracy = eval_test(model, device, val_loader)
            test_time = time.time()
        except Exception as e:
            logger.error(f"Error during training epoch {epoch}: {e}")
            continue
        
        scheduler.step()
        
        logger.info('%d \t %.1f \t \t %.1f \t %.4f \t %.4f \t %.4f \t %.4f',
                    epoch+1, train_time - start_time, test_time - train_time,
                    train_loss, train_accuracy, val_loss, val_accuracy)
        
        # Save best model
        if val_accuracy > best_val_acc:
            best_val_acc = val_accuracy
            torch.save(model.state_dict(), best_model_path)
            logger.info(f"Saved best model with validation accuracy: {best_val_acc:.2f}%")
    
    # Load best model for final evaluation
    if os.path.exists(best_model_path):
        model.load_state_dict(torch.load(best_model_path))
        logger.info(f"Loaded best model with validation accuracy: {best_val_acc:.2f}%")
    
    # Analyze detection performance
    logger.info("Analyzing detection performance...")
    results = analyze_detector(model, device, val_loader)
    
    # Log detection results
    logger.info(f"Clean samples accuracy: {results['clean_acc']:.2f}%")
    logger.info(f"Poisoned samples accuracy: {results['poison_acc']:.2f}%")
    logger.info(f"Clean samples loss: {results['clean_loss']:.4f}")
    logger.info(f"Poisoned samples loss: {results['poison_loss']:.4f}")
    logger.info(f"Clean samples confidence: {results['clean_conf']:.4f}")
    logger.info(f"Poisoned samples confidence: {results['poison_conf']:.4f}")
    
    # Plot losses and confidences distribution
    plt.figure(figsize=(12, 5))
    
    # Plot confidence distributions
    plt.subplot(1, 2, 1)
    clean_idx = results['is_poisoned'] == 0
    poisoned_idx = results['is_poisoned'] == 1
    
    plt.hist(results['confidences'][clean_idx], alpha=0.5, bins=20, label='Clean')
    plt.hist(results['confidences'][poisoned_idx], alpha=0.5, bins=20, label='Poisoned')
    plt.xlabel('Confidence')
    plt.ylabel('Count')
    plt.title('Confidence Distribution')
    plt.legend()
    
    # Calculate ROC curve for confidence-based detection
    confidences = results['confidences']
    is_poisoned = results['is_poisoned']
    # Invert confidence so higher values indicate higher likelihood of being poisoned
    detection_scores = 1 - confidences
    
    fpr, tpr, _ = roc_curve(is_poisoned, detection_scores)
    roc_auc = auc(fpr, tpr)
    
    # Plot ROC curve
    plt.subplot(1, 2, 2)
    plt.plot(fpr, tpr, label=f'ROC curve (AUC = {roc_auc:.3f})')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve for Poison Detection')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(dir, f'detection_results_{args.detection_method}.png'))
    
    # Save numerical results
    np.savez(os.path.join(dir, f'detection_results_{args.detection_method}.npz'),
             clean_acc=results['clean_acc'],
             poison_acc=results['poison_acc'],
             clean_loss=results['clean_loss'],
             poison_loss=results['poison_loss'],
             clean_conf=results['clean_conf'],
             poison_conf=results['poison_conf'],
             predictions=results['predictions'],
             targets=results['targets'],
             is_poisoned=results['is_poisoned'],
             confidences=results['confidences'],
             detection_roc_auc=roc_auc)
    
    logger.info(f"Results saved to {dir}")
    
    return results

if __name__ == "__main__":
    main()