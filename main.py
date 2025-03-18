import argparse
import datetime
import os
import shutil
import time
import numpy as np
import dataset
import mlconfig
import torch
import util
import madrys
import matplotlib as plt
from evaluator import Evaluator
from trainer import Trainer
mlconfig.register(madrys.MadrysLoss)

# Parse command-line arguments
parser = argparse.ArgumentParser(description='Train with Unlearnable Examples')

# General Options
parser.add_argument('--seed', type=int, default=0, help='Random seed')
parser.add_argument('--version', type=str, default="resnet18", help='Model version/name')
parser.add_argument('--exp_name', type=str, default="train_unlearnable", help='Experiment name')
parser.add_argument('--config_path', type=str, default='configs/cifar10', help='Path to config files')
parser.add_argument('--load_model', action='store_true', default=False, help='Load existing model')
parser.add_argument('--data_parallel', action='store_true', default=False, help='Use data parallelism')
parser.add_argument('--train', action='store_true', default=False, help='Whether to train the model')
parser.add_argument('--save_frequency', default=-1, type=int, help='Model save frequency (epochs)')

# Dataset Options
parser.add_argument('--train_portion', default=1.0, type=float, help='Portion of data to use for training')
parser.add_argument('--train_batch_size', default=128, type=int, help='Training batch size')
parser.add_argument('--eval_batch_size', default=256, type=int, help='Evaluation batch size')
parser.add_argument('--num_of_workers', default=8, type=int, help='Number of data loader workers')
parser.add_argument('--train_data_type', type=str, default='CIFAR10', help='Training dataset type')
parser.add_argument('--test_data_type', type=str, default='CIFAR10', help='Test dataset type')
parser.add_argument('--train_data_path', type=str, default='../datasets', help='Path to training data')
parser.add_argument('--test_data_path', type=str, default='../datasets', help='Path to test data')
parser.add_argument('--plot', action='store_true', default=True, 
                   help='Generate accuracy plots after training')
# Poison Classwise
parser.add_argument('--poison_classwise', action='store_true', default=False, 
                   help='Poison specific classes only')
parser.add_argument('--poison_classwise_idx', nargs='+', type=int, default=None, 
                   help='Indices of classes to poison')
parser.add_argument('--poison_class_percentage', default=1.0, type=float, 
                   help='Percentage of target class to poison')


# Perturbation/Poisoning Options
parser.add_argument('--perturb_type', default='classwise', type=str, 
                    choices=['classwise', 'samplewise'], help='Perturbation type')
parser.add_argument('--patch_location', default='center', type=str, 
                    choices=['center', 'random'], help='Location of the noise patch')
parser.add_argument('--poison_rate', default=1.0, type=float, help='Portion of dataset to poison')
parser.add_argument('--perturb_tensor_filepath', default=None, type=str, help='Path to perturbation tensor file')

args = parser.parse_args()

# Set up experiment directories
if args.exp_name == '':
    args.exp_name = 'exp_' + datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

exp_path = os.path.join(args.exp_name, args.version)
log_file_path = os.path.join(exp_path, args.version)
checkpoint_path = os.path.join(exp_path, 'checkpoints')
checkpoint_path_file = os.path.join(checkpoint_path, args.version)
util.build_dirs(exp_path)
util.build_dirs(checkpoint_path)
logger = util.setup_logger(name=args.version, log_file=log_file_path + ".log")

# Set up CUDA and random seed
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

# Load experiment configuration
config_file = os.path.join(args.config_path, args.version)+'.yaml'
config = mlconfig.load(config_file)
for key in config:
    logger.info("%s: %s" % (key, config[key]))
shutil.copyfile(config_file, os.path.join(exp_path, args.version+'.yaml'))


def train(starting_epoch, model, optimizer, scheduler, criterion, trainer, evaluator, ENV, data_loader):
    """
    Train the model.
    """
    for epoch in range(starting_epoch, config.epochs):
        logger.info("")
        logger.info("="*20 + "Training Epoch %d" % (epoch) + "="*20)

        # Train
        ENV['global_step'] = trainer.train(epoch, model, criterion, optimizer)
        ENV['train_history'].append(trainer.acc_meters.avg*100)
        scheduler.step()

        # Evaluate
        logger.info("="*20 + "Eval Epoch %d" % (epoch) + "="*20)
        is_best = False
        evaluator.eval(epoch, model)
        payload = ('Eval Loss:%.4f\tEval acc: %.2f' % (evaluator.loss_meters.avg, evaluator.acc_meters.avg*100))
        logger.info(payload)
        ENV['eval_history'].append(evaluator.acc_meters.avg*100)
        ENV['curren_acc'] = evaluator.acc_meters.avg*100
        ENV['cm_history'].append(evaluator.confusion_matrix.cpu().numpy().tolist())
        
        # Reset stats
        trainer._reset_stats()
        evaluator._reset_stats()

        # Save model
        target_model = model.module if args.data_parallel else model
        util.save_model(filename=checkpoint_path_file,
                      epoch=epoch,
                      model=target_model,
                      optimizer=optimizer,
                      scheduler=scheduler,
                      save_best=is_best,
                      ENV=ENV)
        logger.info('Model Saved at %s', checkpoint_path_file)

        # Save model at specified frequency
        if args.save_frequency > 0 and epoch % args.save_frequency == 0:
            filename = checkpoint_path_file + '_epoch%d' % (epoch)
            util.save_model(filename=filename,
                          epoch=epoch,
                          model=target_model,
                          optimizer=optimizer,
                          scheduler=scheduler,
                          ENV=ENV)
            logger.info('Model Saved at %s', filename)
    
    # Generate plot at the end of training
    try:
        logger.info('Generating accuracy plot...')
        plot_dir = os.path.join(os.path.dirname(checkpoint_path_file), 'plots')
        if not os.path.exists(plot_dir):
            os.makedirs(plot_dir)
            
        # Plot the accuracy curves using the saved checkpoint
        plot_file = os.path.join(plot_dir, 'accuracy.png')
        util.plot_training_results(checkpoint_path_file + '.pth', output_dir=plot_dir)
        logger.info('Accuracy plot saved to %s', plot_dir)
    except Exception as e:
        logger.error('Error generating accuracy plot: %s', str(e))
    
    return

def main():
    """Main function for training with unlearnable examples."""
    # Initialize model based on configuration
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
    
    # Generate datasets based on configuration
    datasets_generator = dataset.DatasetGenerator(
        train_batch_size=args.train_batch_size,
        eval_batch_size=args.eval_batch_size,
        train_data_type=args.train_data_type,
        train_data_path=args.train_data_path,
        test_data_type=args.test_data_type,
        test_data_path=args.test_data_path,
        num_of_workers=args.num_of_workers,
        poison_rate=args.poison_rate,
        perturb_type=args.perturb_type,
        patch_location=args.patch_location,
        perturb_tensor_filepath=args.perturb_tensor_filepath,
        poison_classwise=args.poison_classwise,
        poison_classwise_idx=args.poison_classwise_idx,
        seed=args.seed
    )
    if args.poison_classwise and args.poison_classwise_idx:
        config.poisoned_classes = args.poison_classwise_idx
    
    # Log dataset information
    logger.info('Training Dataset: %s' % str(datasets_generator.datasets['train_dataset']))
    logger.info('Test Dataset: %s' % str(datasets_generator.datasets['test_dataset']))
    
    # Handle poisoned datasets
    if 'Poison' in args.train_data_type:
        with open(os.path.join(exp_path, 'poison_targets.npy'), 'wb') as f:
            is_mixup = hasattr(datasets_generator.datasets['train_dataset'], '__name__') and (
                datasets_generator.datasets['train_dataset'].__name__ == 'MixUp' or 
                datasets_generator.datasets['train_dataset'].__name__ == 'CutMix'
            )
            
            if not is_mixup:
                poison_targets = np.array(datasets_generator.datasets['train_dataset'].poison_samples_idx)
                np.save(f, poison_targets)
                logger.info(poison_targets)
                logger.info('Poisoned: %d/%d' % (len(poison_targets), len(datasets_generator.datasets['train_dataset'])))
                logger.info('Poisoned samples idx saved at %s' % (os.path.join(exp_path, 'poison_targets')))
                logger.info('Poisoned Class %s' % (str(datasets_generator.datasets['train_dataset'].poison_class)))

    # Create data loaders
    if args.train_portion == 1.0:
        data_loader = datasets_generator.getDataLoader()
        train_target = 'train_dataset'
    else:
        train_target = 'train_subset'
        data_loader = datasets_generator._split_validation_set(args.train_portion,
                                                            train_shuffle=True,
                                                            train_drop_last=True)

    # Initialize training components
    logger.info("Model parameter size = %fMB", util.count_parameters_in_MB(model))
    
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
        try:
            if hasattr(config.scheduler, 'T_max'):
                t_max_str = str(config.scheduler.T_max)
                if t_max_str == '$epochs':
                    t_max = int(config.epochs)
                else:
                    t_max = int(t_max_str)
            else:
                t_max = int(config.epochs)
        except (ValueError, TypeError):
            t_max = int(config.epochs)  

        try:
            eta_min = float(config.scheduler.eta_min) if hasattr(config.scheduler, 'eta_min') else 0
        except (ValueError, TypeError):
            eta_min = 0

        scheduler = lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=t_max,
            eta_min=eta_min
        )
    elif config.scheduler.name.lower() == "steplr":
        try:
            if hasattr(config.scheduler, 'step_size'):
                step_size_str = str(config.scheduler.step_size)
                if step_size_str == '$epochs_div_3':  # Example of possible string value
                    step_size = int(config.epochs) // 3
                else:
                    step_size = int(step_size_str)
            else:
                step_size = 30
        except (ValueError, TypeError):
            step_size = 30

        try:
            gamma = float(config.scheduler.gamma) if hasattr(config.scheduler, 'gamma') else 0.1
        except (ValueError, TypeError):
            gamma = 0.1

        scheduler = lr_scheduler.StepLR(
            optimizer,
            step_size=step_size,
            gamma=gamma
        )
    elif config.scheduler.name.lower() == "multisteplr":
        try:
            if hasattr(config.scheduler, 'milestones'):
                milestones_str = str(config.scheduler.milestones)
                # Handle potential string notation like '$epochs_div_2,$epochs_mul_0.75'
                if '$' in milestones_str:
                    milestones = []
                    for item in milestones_str.split(','):
                        if '$epochs_div' in item:
                            divisor = float(item.split('_')[-1])
                            milestones.append(int(config.epochs // divisor))
                        elif '$epochs_mul' in item:
                            multiplier = float(item.split('_')[-1])
                            milestones.append(int(config.epochs * multiplier))
                        else:
                            milestones.append(int(item))
                else:
                    # Try to interpret as a list
                    try:
                        milestones = eval(milestones_str)
                    except:
                        milestones = [int(x) for x in milestones_str.replace('[', '').replace(']', '').split(',')]
            else:
                milestones = [150, 225]
        except (ValueError, TypeError):
            milestones = [150, 225]

        try:
            gamma = float(config.scheduler.gamma) if hasattr(config.scheduler, 'gamma') else 0.1
        except (ValueError, TypeError):
            gamma = 0.1

        scheduler = lr_scheduler.MultiStepLR(
            optimizer,
            milestones=milestones,
            gamma=gamma
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
    
    # Initialize trainer and evaluator
    trainer = Trainer(criterion, data_loader, logger, config, target=train_target)
    evaluator = Evaluator(data_loader, logger, config)

    # Setup environment dictionary
    starting_epoch = 0
    ENV = {
        'global_step': 0,
        'best_acc': 0.0,
        'curren_acc': 0.0,
        'best_pgd_acc': 0.0,
        'train_history': [],
        'eval_history': [],
        'pgd_eval_history': [],
        'genotype_list': [],
        'cm_history': []
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
        starting_epoch = checkpoint['epoch']
        ENV = checkpoint['ENV']
        trainer.global_step = ENV['global_step']
        logger.info("File %s loaded!" % (checkpoint_path_file))

    # Train model if requested
    if args.train:
        train(starting_epoch, model, optimizer, scheduler, criterion, trainer, evaluator, ENV, data_loader)


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