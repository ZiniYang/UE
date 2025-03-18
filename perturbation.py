import argparse
import collections
import datetime
import os
import shutil
import time
import dataset
import mlconfig
import toolbox
import torch
import util
import madrys
import numpy as np
from evaluator import Evaluator
from tqdm import tqdm
from trainer import Trainer
mlconfig.register(madrys.MadrysLoss)

# Parse command-line arguments
parser = argparse.ArgumentParser(description='Generate Unlearnable Examples')
# General Options
parser.add_argument('--seed', type=int, default=0, help='Random seed')
parser.add_argument('--version', type=str, default="resnet18", help='Model version/name')
parser.add_argument('--exp_name', type=str, default="experiments", help='Experiment name')
parser.add_argument('--config_path', type=str, default='configs/cifar10', help='Path to config files')
parser.add_argument('--load_model', action='store_true', default=False, help='Load existing model')
parser.add_argument('--data_parallel', action='store_true', default=False, help='Use data parallelism')
# Dataset Options
parser.add_argument('--train_batch_size', default=512, type=int, help='Training batch size')
parser.add_argument('--eval_batch_size', default=512, type=int, help='Evaluation batch size')
parser.add_argument('--num_of_workers', default=4, type=int, help='Number of data loader workers')
parser.add_argument('--train_data_type', type=str, default='CIFAR10', help='Training dataset type')
parser.add_argument('--train_data_path', type=str, default='../datasets', help='Path to training data')
parser.add_argument('--test_data_type', type=str, default='CIFAR10', help='Test dataset type')
parser.add_argument('--test_data_path', type=str, default='../datasets', help='Path to test data')
# Perturbation Options
parser.add_argument('--universal_train_portion', default=0.2, type=float, help='Portion of data for universal perturbation')
parser.add_argument('--universal_stop_error', default=0.5, type=float, help='Target error rate for perturbation')
parser.add_argument('--universal_train_target', default='train_dataset', type=str, help='Target dataset for universal perturbation')
parser.add_argument('--train_step', default=10, type=int, help='Number of training steps')
parser.add_argument('--use_subset', action='store_true', default=False, help='Use data subset for faster iteration')
parser.add_argument('--attack_type', default='min-min', type=str, choices=['min-min', 'min-max', 'random'], help='Attack type')
parser.add_argument('--perturb_type', default='classwise', type=str, choices=['classwise', 'samplewise'], help='Perturbation type')
parser.add_argument('--patch_location', default='center', type=str, choices=['center', 'random'], help='Location of the noise patch')
parser.add_argument('--noise_shape', default=[10, 3, 32, 32], nargs='+', type=int, help='Noise tensor shape')
parser.add_argument('--epsilon', default=8, type=float, help='Perturbation magnitude (0-255 scale)')
parser.add_argument('--num_steps', default=1, type=int, help='Number of perturbation steps')
parser.add_argument('--step_size', default=0.8, type=float, help='Step size for perturbation (0-255 scale)')
parser.add_argument('--random_start', action='store_true', default=False, help='Use random initialization')
args = parser.parse_args()

args.epsilon = args.epsilon / 255
args.step_size = args.step_size / 255

if args.exp_name == '':
    args.exp_name = 'exp_' + datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

exp_path = os.path.join(args.exp_name, args.version)
log_file_path = os.path.join(exp_path, args.version)
checkpoint_path = os.path.join(exp_path, 'checkpoints')
checkpoint_path_file = os.path.join(checkpoint_path, args.version)
util.build_dirs(exp_path)
util.build_dirs(checkpoint_path)
logger = util.setup_logger(name=args.version, log_file=log_file_path + ".log")

logger.info("PyTorch Version: %s" % (torch.__version__))
if torch.cuda.is_available():
    torch.cuda.manual_seed(args.seed)
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    device = torch.device('cuda')
    device_list = [torch.cuda.get_device_name(i) for i in range(0, torch.cuda.device_count())]
    logger.info("GPU List: %s" % (device_list))
else:
    device = torch.device('cpu')

config_file = os.path.join(args.config_path, args.version)+'.yaml'
config = mlconfig.load(config_file)
for key in config:
    logger.info("%s: %s" % (key, config[key]))
shutil.copyfile(config_file, os.path.join(exp_path, args.version+'.yaml'))

def universal_perturbation_eval(noise_generator, random_noise, data_loader, model, eval_target='train_dataset'):
    if eval_target not in data_loader:
        logger.warning(f"Eval target '{eval_target}' not found in data_loader. Using 'train_dataset' instead.")
        eval_target = 'train_dataset'
        
    loss_meter = util.AverageMeter()
    err_meter = util.AverageMeter()
    random_noise = random_noise.to(device)
    model = model.to(device)
    
    for i, (images, labels) in enumerate(data_loader[eval_target]):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        
        if random_noise is not None:
            for i in range(len(labels)):
                class_index = labels[i].item()
                noise = random_noise[class_index]
                mask_cord, class_noise = noise_generator._patch_noise_extend_to_img(noise, image_size=images[i].shape, patch_location=args.patch_location)
                images[i] += class_noise
                
        pred = model(images)
        err = (pred.data.max(1)[1] != labels.data).float().sum()
        loss = torch.nn.CrossEntropyLoss()(pred, labels)
        
        loss_meter.update(loss.item(), len(labels))
        err_meter.update(err / len(labels))
        
    return loss_meter.avg, err_meter.avg


def universal_perturbation(noise_generator, trainer, evaluator, model, criterion, optimizer, scheduler, random_noise, ENV):
    """Generate universal (class-wise) perturbation."""
    datasets_generator = dataset.DatasetGenerator(train_batch_size=args.train_batch_size,
                                              eval_batch_size=args.eval_batch_size,
                                              train_data_type=args.train_data_type,
                                              train_data_path=args.train_data_path,
                                              test_data_type=args.test_data_type,
                                              test_data_path=args.test_data_path,
                                              num_of_workers=args.num_of_workers,
                                              seed=args.seed, no_train_augments=True)

    if args.use_subset:
        data_loader = datasets_generator._split_validation_set(train_portion=args.universal_train_portion,
                                                           train_shuffle=True, train_drop_last=True)
    else:
        data_loader = datasets_generator.getDataLoader(train_shuffle=True, train_drop_last=True)
    
    if args.universal_train_target not in data_loader:
        logger.warning(f"Target '{args.universal_train_target}' not found in data_loader. Using 'train_dataset' instead.")
        train_target = 'train_dataset'
    else:
        train_target = args.universal_train_target
        
    condition = True
    data_iter = iter(data_loader['train_dataset'])
    logger.info('=' * 20 + 'Searching Universal Perturbation' + '=' * 20)
    
    if hasattr(model, 'classify'):
        model.classify = True
    
    iter_count = 0
    while condition:
        iter_count += 1
        if args.attack_type == 'min-min' and not args.load_model:
            # Train model on perturbed data for min-min attack
            for j in range(0, args.train_step):
                try:
                    (images, labels) = next(data_iter)
                except:
                    data_iter = iter(data_loader['train_dataset'])
                    (images, labels) = next(data_iter)

                images = images.to(device)
                labels = labels.to(device)
                
                # Add Class-wise Noise to each sample
                train_imgs = []
                for i, (image, label) in enumerate(zip(images, labels)):
                    noise = random_noise[label.item()]
                    mask_cord, class_noise = noise_generator._patch_noise_extend_to_img(noise, image_size=image.shape, patch_location=args.patch_location)
                    train_imgs.append(images[i]+class_noise)
                    
                # Train model on perturbed images
                model.train()
                for param in model.parameters():
                    param.requires_grad = True
                trainer.train_batch(torch.stack(train_imgs).to(device), labels, model, optimizer)

        # Update perturbation based on all data
        for i, (images, labels) in tqdm(enumerate(data_loader[train_target]), total=len(data_loader[train_target])):
            images = images.to(device)
            labels = labels.to(device)
            
            # Add Class-wise Noise to each sample
            batch_noise, mask_cord_list = [], []
            for i, (image, label) in enumerate(zip(images, labels)):
                noise = random_noise[label.item()]
                mask_cord, class_noise = noise_generator._patch_noise_extend_to_img(noise, image_size=image.shape, patch_location=args.patch_location)
                batch_noise.append(class_noise)
                mask_cord_list.append(mask_cord)

            # Update universal perturbation using appropriate attack
            model.eval()
            for param in model.parameters():
                param.requires_grad = False

            batch_noise = torch.stack(batch_noise).to(device)
            if args.attack_type == 'min-min':
                perturb_img, eta = noise_generator.min_min_attack(images, labels, model, optimizer, criterion, random_noise=batch_noise)
            elif args.attack_type == 'min-max':
                perturb_img, eta = noise_generator.min_max_attack(images, labels, model, optimizer, criterion, random_noise=batch_noise)
            else:
                raise ValueError('Invalid attack type')

            # Update class-wise noise
            class_noise_eta = collections.defaultdict(list)
            for i in range(len(eta)):
                x1, x2, y1, y2 = mask_cord_list[i]
                delta = eta[i][:, x1: x2, y1: y2]
                class_noise_eta[labels[i].item()].append(delta.detach().cpu())

            # Average updates for each class
            for key in class_noise_eta:
                delta = torch.stack(class_noise_eta[key]).mean(dim=0) - random_noise[key]
                class_noise = random_noise[key]
                class_noise += delta
                random_noise[key] = torch.clamp(class_noise, -args.epsilon, args.epsilon)

        # Evaluate termination conditions
        loss_avg, error_rate = universal_perturbation_eval(noise_generator, random_noise, data_loader, model, eval_target=train_target)
        logger.info(f"Iteration {iter_count}: Loss={loss_avg:.4f}, Acc={100-error_rate*100:.2f}%, Target={100-args.universal_stop_error*100:.2f}%")
        random_noise = random_noise.detach()
        ENV['random_noise'] = random_noise
        
        # Check if perturbation is effective enough
        if args.attack_type == 'min-min':
            condition = error_rate > args.universal_stop_error
        elif args.attack_type == 'min-max':
            condition = error_rate < args.universal_stop_error
            
    logger.info(f"Final perturbation: min={random_noise.min():.4f}, max={random_noise.max():.4f}, mean={random_noise.mean():.4f}")
    return random_noise


def samplewise_perturbation_eval(random_noise, data_loader, model, eval_target='train_dataset', mask_cord_list=[]):
    """Evaluate the effectiveness of sample-wise perturbation."""
    loss_meter = util.AverageMeter()
    err_meter = util.AverageMeter()
    model = model.to(device)
    idx = 0
    
    for i, (images, labels) in enumerate(data_loader[eval_target]):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        
        if random_noise is not None:
            for i, (image, label) in enumerate(zip(images, labels)):
                if not torch.is_tensor(random_noise):
                    sample_noise = torch.tensor(random_noise[idx]).to(device)
                else:
                    sample_noise = random_noise[idx].to(device)
                    
                c, h, w = image.shape[0], image.shape[1], image.shape[2]
                mask = np.zeros((c, h, w), np.float32)
                x1, x2, y1, y2 = mask_cord_list[idx]
                mask[:, x1: x2, y1: y2] = sample_noise.cpu().numpy()
                sample_noise = torch.from_numpy(mask).to(device)
                
                images[i] = images[i] + sample_noise
                idx += 1
                
        pred = model(images)
        err = (pred.data.max(1)[1] != labels.data).float().sum()
        loss = torch.nn.CrossEntropyLoss()(pred, labels)
        
        loss_meter.update(loss.item(), len(labels))
        err_meter.update(err / len(labels))
        
    return loss_meter.avg, err_meter.avg


def sample_wise_perturbation(noise_generator, trainer, evaluator, model, criterion, optimizer, scheduler, random_noise, ENV):
    """Generate sample-wise perturbation."""
    datasets_generator = dataset.DatasetGenerator(train_batch_size=args.train_batch_size,
                                               eval_batch_size=args.eval_batch_size,
                                               train_data_type=args.train_data_type,
                                               train_data_path=args.train_data_path,
                                               test_data_type=args.test_data_type,
                                               test_data_path=args.test_data_path,
                                               num_of_workers=args.num_of_workers,
                                               seed=args.seed, no_train_augments=True)

    data_loader = datasets_generator.getDataLoader(train_shuffle=False, train_drop_last=False)
    
    # Calculate total number of samples in the dataset
    total_samples = len(data_loader['train_dataset'].dataset)
    logger.info(f"Total samples in dataset: {total_samples}")
    
    # For samplewise perturbation, we need noise for each sample
    # Check if random_noise matches the dataset size
    if random_noise.size(0) != total_samples:
        logger.info(f"Initializing sample-wise noise: {random_noise.shape} -> [{total_samples}, 3, 32, 32]")
        # Create a new noise tensor with the correct size
        sample_noise = []
        for i in range(total_samples):
            if args.random_start:
                # Generate random noise for each sample
                noise = torch.FloatTensor(3, 32, 32).uniform_(-args.epsilon, args.epsilon)
            else:
                # Use class-based noise or zeros
                class_idx = i % random_noise.size(0)  # Use class noise in a round-robin fashion
                noise = random_noise[class_idx].clone()
            sample_noise.append(noise)
        random_noise = torch.stack(sample_noise)
        logger.info(f"New sample-wise noise shape: {random_noise.shape}")
    
    # Generate mask coordinates for each sample
    mask_cord_list = []
    idx = 0
    for images, labels in data_loader['train_dataset']:
        for i, (image, label) in enumerate(zip(images, labels)):
            noise = random_noise[idx]
            mask_cord, _ = noise_generator._patch_noise_extend_to_img(noise, image_size=image.shape, patch_location=args.patch_location)
            mask_cord_list.append(mask_cord)
            idx += 1

    condition = True
    train_idx = 0
    data_iter = iter(data_loader['train_dataset'])
    logger.info('=' * 20 + 'Searching Samplewise Perturbation' + '=' * 20)
    
    iter_count = 0
    while condition:
        iter_count += 1
        if args.attack_type == 'min-min' and not args.load_model:
            # Train model on perturbed data for min-min attack
            for j in tqdm(range(0, args.train_step), total=args.train_step):
                try:
                    (images, labels) = next(data_iter)
                except:
                    train_idx = 0
                    data_iter = iter(data_loader['train_dataset'])
                    (images, labels) = next(data_iter)

                images = images.to(device)
                labels = labels.to(device)
                
                # Add Sample-wise Noise to each sample
                for i, (image, label) in enumerate(zip(images, labels)):
                    if train_idx >= len(random_noise):
                        logger.warning(f"train_idx {train_idx} exceeds random_noise size {len(random_noise)}, resetting to 0")
                        train_idx = 0
                        
                    sample_noise = random_noise[train_idx]
                    
                    c, h, w = image.shape[0], image.shape[1], image.shape[2]
                    mask = np.zeros((c, h, w), np.float32)
                    x1, x2, y1, y2 = mask_cord_list[train_idx]
                    
                    if isinstance(sample_noise, np.ndarray):
                        mask[:, x1: x2, y1: y2] = sample_noise
                    else:
                        mask[:, x1: x2, y1: y2] = sample_noise.cpu().numpy()
                        
                    sample_noise = torch.from_numpy(mask).to(device)
                    images[i] = images[i] + sample_noise
                    train_idx += 1

                # Train model on perturbed images
                model.train()
                for param in model.parameters():
                    param.requires_grad = True
                trainer.train_batch(images, labels, model, optimizer)

        # Update perturbation based on all data
        idx = 0
        for i, (images, labels) in tqdm(enumerate(data_loader['train_dataset']), total=len(data_loader['train_dataset'])):
            images = images.to(device)
            labels = labels.to(device)
            model = model.to(device)

            # Add Sample-wise Noise to each sample
            batch_noise, batch_start_idx = [], idx
            for i, (image, label) in enumerate(zip(images, labels)):
                if idx >= len(random_noise):
                    logger.warning(f"idx {idx} exceeds random_noise size {len(random_noise)}, breaking loop")
                    break
                    
                sample_noise = random_noise[idx]
                
                c, h, w = image.shape[0], image.shape[1], image.shape[2]
                mask = np.zeros((c, h, w), np.float32)
                x1, x2, y1, y2 = mask_cord_list[idx]
                
                if isinstance(sample_noise, np.ndarray):
                    mask[:, x1: x2, y1: y2] = sample_noise
                else:
                    mask[:, x1: x2, y1: y2] = sample_noise.cpu().numpy()
                    
                sample_noise = torch.from_numpy(mask).to(device)
                batch_noise.append(sample_noise)
                idx += 1
            
            if len(batch_noise) == 0:
                continue

            # Update sample-wise perturbation using appropriate attack
            model.eval()
            for param in model.parameters():
                param.requires_grad = False
                
            batch_noise = torch.stack(batch_noise).to(device)
            if args.attack_type == 'min-min':
                perturb_img, eta = noise_generator.min_min_attack(images[:len(batch_noise)], labels[:len(batch_noise)], model, optimizer, criterion, random_noise=batch_noise)
            elif args.attack_type == 'min-max':
                perturb_img, eta = noise_generator.min_max_attack(images[:len(batch_noise)], labels[:len(batch_noise)], model, optimizer, criterion, random_noise=batch_noise)
            else:
                raise ValueError('Invalid attack type')

            # Update each sample's noise
            for i, delta in enumerate(eta):
                x1, x2, y1, y2 = mask_cord_list[batch_start_idx+i]
                delta = delta[:, x1: x2, y1: y2]
                
                if torch.is_tensor(random_noise):
                    random_noise[batch_start_idx+i] = delta.detach().cpu().clone()
                else:
                    random_noise[batch_start_idx+i] = delta.detach().cpu().numpy()

        # Evaluate termination conditions
        loss_avg, error_rate = samplewise_perturbation_eval(random_noise, data_loader, model, 
                                                          eval_target='train_dataset',
                                                          mask_cord_list=mask_cord_list)
        logger.info(f"Iteration {iter_count}: Loss={loss_avg:.4f}, Acc={100-error_rate*100:.2f}%, Target={100-args.universal_stop_error*100:.2f}%")

        if torch.is_tensor(random_noise):
            random_noise = random_noise.detach()
            ENV['random_noise'] = random_noise
            
        if args.attack_type == 'min-min':
            condition = error_rate > args.universal_stop_error
        elif args.attack_type == 'min-max':
            condition = error_rate < args.universal_stop_error

    # Ensure consistent format of perturbation
    if torch.is_tensor(random_noise):
        new_random_noise = []
        for idx in range(len(random_noise)):
            sample_noise = random_noise[idx]
            c, h, w = 3, 32, 32  # Default for CIFAR
            mask = np.zeros((c, h, w), np.float32)
            x1, x2, y1, y2 = mask_cord_list[idx]
            mask[:, x1: x2, y1: y2] = sample_noise.cpu().numpy()
            new_random_noise.append(torch.from_numpy(mask))
        new_random_noise = torch.stack(new_random_noise)
        return new_random_noise
    else:
        return random_noise

def main():
    """Main function to generate unlearnable examples."""
    # Setup dataset
    datasets_generator = dataset.DatasetGenerator(train_batch_size=args.train_batch_size,
                                               eval_batch_size=args.eval_batch_size,
                                               train_data_type=args.train_data_type,
                                               train_data_path=args.train_data_path,
                                               test_data_type=args.test_data_type,
                                               test_data_path=args.test_data_path,
                                               num_of_workers=args.num_of_workers,
                                               seed=args.seed)
    data_loader = datasets_generator.getDataLoader()
    
    # Initialize model
    if config.model.name == "ResNet18":
        from models.ResNet import ResNet18
        model = ResNet18(num_classes=config.model.num_classes).to(device)
    elif config.model.name == "ResNet34":
        from models.ResNet import ResNet34
        model = ResNet34(num_classes=config.model.num_classes).to(device)
    elif config.model.name == "ResNet50":
        from models.ResNet import ResNet50
        model = ResNet50(num_classes=config.model.num_classes).to(device)
    else:
        logger.info(f"Unknown model: {config.model.name}")
        raise ValueError(f"Unsupported model type: {config.model.name}")

    # Initialize optimizer
    import torch.optim as optim
    if config.optimizer.name.lower() == "sgd":
        optimizer = optim.SGD(
            model.parameters(),
            lr=config.optimizer.lr,
            momentum=config.optimizer.momentum if hasattr(config.optimizer, 'momentum') else 0.9,
            weight_decay=config.optimizer.weight_decay if hasattr(config.optimizer, 'weight_decay') else 5e-4
        )
    elif config.optimizer.name.lower() == "adam":
        optimizer = optim.Adam(
            model.parameters(),
            lr=config.optimizer.lr,
            weight_decay=config.optimizer.weight_decay if hasattr(config.optimizer, 'weight_decay') else 0
        )
    else:
        logger.info(f"Unknown optimizer: {config.optimizer.name}")
        raise ValueError(f"Unsupported optimizer type: {config.optimizer.name}")

    # Initialize scheduler
    import torch.optim.lr_scheduler as lr_scheduler
    if config.scheduler.name.lower() == "cosineannealinglr":
        # Simple version
        scheduler = lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=config.epochs,
            eta_min=0
        )
    elif config.scheduler.name.lower() == "steplr":
        scheduler = lr_scheduler.StepLR(
            optimizer,
            step_size=30,
            gamma=0.1
        )
    elif config.scheduler.name.lower() == "multisteplr":
        scheduler = lr_scheduler.MultiStepLR(
            optimizer,
            milestones=[150, 225],
            gamma=0.1
        )
    else:
        logger.info(f"Unknown scheduler: {config.scheduler.name}")
        raise ValueError(f"Unsupported scheduler type: {config.scheduler.name}")

    # Initialize loss function
    import torch.nn as nn
    if config.criterion.name.lower() == "crossentropyloss":
        criterion = nn.CrossEntropyLoss()
    elif config.criterion.name.lower() == "madryloss":
        from madrys import MadrysLoss
        criterion = MadrysLoss(
            step_size=config.criterion.step_size if hasattr(config.criterion, 'step_size') else 0.007,
            epsilon=config.criterion.epsilon if hasattr(config.criterion, 'epsilon') else 0.031,
            perturb_steps=config.criterion.perturb_steps if hasattr(config.criterion, 'perturb_steps') else 10
        )
    else:
        logger.info(f"Unknown loss function: {config.criterion.name}")
        raise ValueError(f"Unsupported loss function type: {config.criterion.name}")

    # Setup training target and data loaders
    if args.perturb_type == 'samplewise':
        train_target = 'train_dataset'
    else:
        if args.use_subset:
            data_loader = datasets_generator._split_validation_set(train_portion=args.universal_train_portion,
                                                                train_shuffle=True, train_drop_last=True)
            train_target = 'train_subset'
        else:
            data_loader = datasets_generator.getDataLoader(train_shuffle=True, train_drop_last=True)
            train_target = 'train_dataset'

    # Initialize trainer and evaluator
    trainer = Trainer(criterion, data_loader, logger, config, target=train_target)
    evaluator = Evaluator(data_loader, logger, config)
    
    # Setup environment dictionary
    ENV = {
        'global_step': 0,
        'best_acc': 0.0,
        'curren_acc': 0.0,
        'best_pgd_acc': 0.0,
        'train_history': [],
        'eval_history': [],
        'pgd_eval_history': [],
        'genotype_list': []
    }

    # Apply data parallelism if requested
    if args.data_parallel:
        model = torch.nn.DataParallel(model)

    # Load existing model if requested
    if args.load_model:
        checkpoint = util.load_model(filename=checkpoint_path_file,
                                  model=model,
                                  optimizer=optimizer,
                                  scheduler=scheduler)
        ENV = checkpoint['ENV']
        trainer.global_step = ENV['global_step']
        logger.info("File %s loaded!" % (checkpoint_path_file))

    # Initialize perturbation generator
    noise_generator = toolbox.PerturbationTool(epsilon=args.epsilon,
                                            num_steps=args.num_steps,
                                            step_size=args.step_size)

    # Generate perturbation based on attack type
    if args.attack_type == 'random':
        # Generate random perturbation
        noise = noise_generator.random_noise(noise_shape=args.noise_shape)
        torch.save(noise, os.path.join(args.exp_name, 'perturbation.pt'))
        logger.info(f"Random noise saved at {os.path.join(args.exp_name, 'perturbation.pt')}")
    elif args.attack_type == 'min-min' or args.attack_type == 'min-max':
        # For min-max attack, train model first to converge
        if args.attack_type == 'min-max':
            logger.info("Training model before generating min-max perturbation...")
            def train_for_minmax(starting_epoch, model, optimizer, scheduler, criterion, trainer, evaluator, ENV, data_loader):
                logger.info("Pre-training model for min-max attack...")
                for epoch in range(starting_epoch, 5):  
                    logger.info(f"Pre-training Epoch {epoch}")
                    trainer._reset_stats()
                    ENV['global_step'] = trainer.train(epoch, model, criterion, optimizer)
                    evaluator._reset_stats()
                    evaluator.eval(epoch, model)
                    logger.info(f"Eval acc: {evaluator.acc_meters.avg*100:.2f}%")

                    scheduler.step()
            train_for_minmax(0, model, optimizer, scheduler, criterion, trainer, evaluator, ENV, data_loader)


            
        # Initialize perturbation
        if args.random_start:
            random_noise = noise_generator.random_noise(noise_shape=args.noise_shape)
        else:
            random_noise = torch.zeros(*args.noise_shape)
            
        # Generate perturbation based on type
        if args.perturb_type == 'samplewise':
            noise = sample_wise_perturbation(noise_generator, trainer, evaluator, model, criterion, optimizer, scheduler, random_noise, ENV)
        elif args.perturb_type == 'classwise':
            noise = universal_perturbation(noise_generator, trainer, evaluator, model, criterion, optimizer, scheduler, random_noise, ENV)
            
        # Save perturbation
        torch.save(noise, os.path.join(args.exp_name, 'perturbation.pt'))
        logger.info(f"Perturbation saved at {os.path.join(args.exp_name, 'perturbation.pt')}")
    else:
        raise ValueError('Attack type not implemented')
    return


if __name__ == '__main__':
    # Log arguments
    for arg in vars(args):
        logger.info("%s: %s" % (arg, getattr(args, arg)))
        
    # Run main function and time it
    start = time.time()
    main()
    end = time.time()
    cost = (end - start) / 86400
    payload = "Running Cost %.2f Days \n" % cost
    logger.info(payload)