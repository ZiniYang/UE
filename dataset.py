import copy
import os
import collections
import numpy as np
import torch
import util
import random
import mlconfig
from util import onehot, rand_bbox
from torch.utils.data.dataset import Dataset
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# Set device for computation
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Define standard transformations for CIFAR datasets
transform_options = {
    "CIFAR10": {
        "train_transform": [transforms.RandomCrop(32, padding=4),
                            transforms.RandomHorizontalFlip(),
                            transforms.ToTensor()],
        "test_transform": [transforms.ToTensor()]},
    "CIFAR100": {
         "train_transform": [transforms.RandomCrop(32, padding=4),
                             transforms.RandomHorizontalFlip(),
                             transforms.RandomRotation(20),
                             transforms.ToTensor()],
         "test_transform": [transforms.ToTensor()]},
}
# Add poisoned dataset transformations
transform_options['PoisonCIFAR10'] = transform_options['CIFAR10']
transform_options['PoisonCIFAR100'] = transform_options['CIFAR100']


@mlconfig.register
class DatasetGenerator():
    """
    Factory class for generating and managing datasets.
    """
    def __init__(self, train_batch_size=128, eval_batch_size=256, num_of_workers=4,
                 train_data_path='../datasets/', train_data_type='CIFAR10', seed=0,
                 test_data_path='../datasets/', test_data_type='CIFAR10', fa=False,
                 no_train_augments=False, poison_rate=1.0, perturb_type='classwise',
                 perturb_tensor_filepath=None, patch_location='center', img_denoise=False,
                 add_uniform_noise=False, poison_classwise=False, poison_classwise_idx=None,
                 use_cutout=None, use_cutmix=False, use_mixup=False, 
                 poison_class_percentage=1.0):  # New parameter for partial class poisoning
        """
        Initialize the dataset generator.
        
        Args:
            train_batch_size: Batch size for training
            eval_batch_size: Batch size for evaluation
            num_of_workers: Number of worker threads for data loading
            train_data_path: Path to training data
            train_data_type: Type of training dataset
            seed: Random seed for reproducibility
            test_data_path: Path to test data
            test_data_type: Type of test dataset
            fa: Whether to use Fast AutoAugment
            no_train_augments: Whether to skip training augmentations
            poison_rate: Portion of dataset to poison
            perturb_type: Type of perturbation ('classwise' or 'samplewise')
            perturb_tensor_filepath: Path to perturbation tensor file
            patch_location: Location to apply perturbation patch
            img_denoise: Whether to denoise images
            add_uniform_noise: Whether to add uniform noise
            poison_classwise: Whether to poison specific classes
            poison_classwise_idx: Indices of classes to poison
            use_cutout: Whether to use Cutout augmentation
            use_cutmix: Whether to use CutMix augmentation
            use_mixup: Whether to use MixUp augmentation
            poison_class_percentage: Percentage of target class samples to poison
        """
        np.random.seed(seed)
        self.train_batch_size = train_batch_size
        self.eval_batch_size = eval_batch_size
        self.num_of_workers = num_of_workers
        self.seed = seed
        self.train_data_type = train_data_type
        self.test_data_type = test_data_type
        self.train_data_path = train_data_path
        self.test_data_path = test_data_path

        # Prepare transforms
        train_transform = transform_options[train_data_type]['train_transform']
        test_transform = transform_options[test_data_type]['test_transform']
        train_transform = transforms.Compose(train_transform)
        test_transform = transforms.Compose(test_transform)
        
        # Use test transforms for training if no augmentation
        if no_train_augments:
            train_transform = test_transform

        # Add Cutout if specified
        if use_cutout is not None:
            print('Using Cutout')
            train_transform.transforms.append(Cutout(16))

        # Create training dataset based on type
        if train_data_type == 'CIFAR10':
            num_of_classes = 10
            train_dataset = datasets.CIFAR10(root=train_data_path, train=True,
                                           download=True, transform=train_transform)
        elif train_data_type == 'PoisonCIFAR10':
            num_of_classes = 10
            train_dataset = PoisonCIFAR10(root=train_data_path, transform=train_transform,
                                        poison_rate=poison_rate, perturb_type=perturb_type,
                                        patch_location=patch_location, seed=seed, img_denoise=img_denoise,
                                        perturb_tensor_filepath=perturb_tensor_filepath,
                                        add_uniform_noise=add_uniform_noise,
                                        poison_classwise=poison_classwise,
                                        poison_classwise_idx=poison_classwise_idx,
                                        poison_class_percentage=poison_class_percentage)  # Pass the new parameter
        elif train_data_type == 'CIFAR100':
            num_of_classes = 100
            train_dataset = datasets.CIFAR100(root=train_data_path, train=True,
                                            download=True, transform=train_transform)
        elif train_data_type == 'PoisonCIFAR100':
            num_of_classes = 100
            train_dataset = PoisonCIFAR100(root=train_data_path, transform=train_transform,
                                         poison_rate=poison_rate, perturb_type=perturb_type,
                                         patch_location=patch_location, seed=seed, img_denoise=img_denoise,
                                         perturb_tensor_filepath=perturb_tensor_filepath,
                                         add_uniform_noise=add_uniform_noise,
                                         poison_classwise=poison_classwise,
                                         poison_class_percentage=poison_class_percentage)  # Pass the new parameter
        else:
            raise ValueError(f'Training dataset type {train_data_type} not implemented')

        # Create test dataset based on type
        if test_data_type == 'CIFAR10':
            test_dataset = datasets.CIFAR10(root=test_data_path, train=False,
                                          download=True, transform=test_transform)
        elif test_data_type == 'PoisonCIFAR10':
            test_dataset = PoisonCIFAR10(root=test_data_path, train=False, transform=test_transform,
                                       poison_rate=poison_rate, perturb_type=perturb_type,
                                       patch_location=patch_location, seed=seed, img_denoise=img_denoise,
                                       perturb_tensor_filepath=perturb_tensor_filepath,
                                       add_uniform_noise=add_uniform_noise,
                                       poison_classwise=poison_classwise,
                                       poison_classwise_idx=poison_classwise_idx,
                                       poison_class_percentage=poison_class_percentage)  # Pass the new parameter
        elif test_data_type == 'CIFAR100':
            test_dataset = datasets.CIFAR100(root=test_data_path, train=False,
                                           download=True, transform=test_transform)
        elif test_data_type == 'PoisonCIFAR100':
            test_dataset = PoisonCIFAR100(root=test_data_path, train=False, transform=test_transform,
                                        poison_rate=poison_rate, perturb_type=perturb_type,
                                        patch_location=patch_location, seed=seed, img_denoise=img_denoise,
                                        perturb_tensor_filepath=perturb_tensor_filepath,
                                        add_uniform_noise=add_uniform_noise,
                                        poison_classwise=poison_classwise,
                                        poison_class_percentage=poison_class_percentage)  # Pass the new parameter
        else:
            raise ValueError(f'Test dataset type {test_data_type} not implemented')

        # Apply data augmentation methods if specified
        if use_cutmix:
            train_dataset = CutMix(dataset=train_dataset, num_class=num_of_classes)
        elif use_mixup:
            train_dataset = MixUp(dataset=train_dataset, num_class=num_of_classes)

        # Store datasets
        self.datasets = {
            'train_dataset': train_dataset,
            'test_dataset': test_dataset,
        }
        return

    def getDataLoader(self, train_shuffle=True, train_drop_last=True):
        """
        Creates DataLoader objects for the datasets.
        
        Args:
            train_shuffle: Whether to shuffle training data
            train_drop_last: Whether to drop the last incomplete batch
            
        Returns:
            Dictionary of DataLoader objects
        """
        data_loaders = {}

        # Training data loader
        data_loaders['train_dataset'] = DataLoader(dataset=self.datasets['train_dataset'],
                                                 batch_size=self.train_batch_size,
                                                 shuffle=train_shuffle, pin_memory=True,
                                                 drop_last=train_drop_last, num_workers=self.num_of_workers)

        # Test data loader
        data_loaders['test_dataset'] = DataLoader(dataset=self.datasets['test_dataset'],
                                                batch_size=self.eval_batch_size,
                                                shuffle=False, pin_memory=True,
                                                drop_last=False, num_workers=self.num_of_workers)

        return data_loaders

    def _split_validation_set(self, train_portion, train_shuffle=True, train_drop_last=True):
        """
        Splits the training dataset into training and validation subsets.
        
        Args:
            train_portion: Portion of data to use for training
            train_shuffle: Whether to shuffle training data
            train_drop_last: Whether to drop the last incomplete batch
            
        Returns:
            Dictionary of DataLoader objects including validation set
        """
        np.random.seed(self.seed)
        train_subset = copy.deepcopy(self.datasets['train_dataset'])
        valid_subset = copy.deepcopy(self.datasets['train_dataset'])

        # Split data and labels
        datasplit = train_test_split(self.datasets['train_dataset'].data,
                                    self.datasets['train_dataset'].targets,
                                    test_size=1-train_portion, train_size=train_portion,
                                    shuffle=True, stratify=self.datasets['train_dataset'].targets)
        train_D, valid_D, train_L, valid_L = datasplit
        
        print('Train Labels: ', np.array(train_L))
        print('Valid Labels: ', np.array(valid_L))
        
        # Update subsets with split data
        train_subset.data = np.array(train_D)
        valid_subset.data = np.array(valid_D)
        train_subset.targets = train_L
        valid_subset.targets = valid_L

        self.datasets['train_subset'] = train_subset
        self.datasets['valid_subset'] = valid_subset
        print(self.datasets)

        # Create data loaders
        data_loaders = {}

        data_loaders['train_dataset'] = DataLoader(dataset=self.datasets['train_dataset'],
                                                 batch_size=self.train_batch_size,
                                                 shuffle=train_shuffle, pin_memory=True,
                                                 drop_last=train_drop_last, num_workers=self.num_of_workers)

        data_loaders['test_dataset'] = DataLoader(dataset=self.datasets['test_dataset'],
                                                batch_size=self.eval_batch_size,
                                                shuffle=False, pin_memory=True,
                                                drop_last=False, num_workers=self.num_of_workers)

        data_loaders['train_subset'] = DataLoader(dataset=self.datasets['train_subset'],
                                                batch_size=self.train_batch_size,
                                                shuffle=train_shuffle, pin_memory=True,
                                                drop_last=train_drop_last, num_workers=self.num_of_workers)

        data_loaders['valid_subset'] = DataLoader(dataset=self.datasets['valid_subset'],
                                                batch_size=self.eval_batch_size,
                                                shuffle=False, pin_memory=True,
                                                drop_last=False, num_workers=self.num_of_workers)
        return data_loaders


def patch_noise_extend_to_img(noise, image_size=[32, 32, 3], patch_location='center'):
    """
    Extends a noise patch to the full image by placing it at the specified location.
    
    Args:
        noise: Noise patch to extend
        image_size: Target image dimensions [height, width, channels]
        patch_location: Where to place the patch ('center' or 'random')
        
    Returns:
        Full-sized noise mask with the patch applied
    """
    h, w, c = image_size[0], image_size[1], image_size[2]
    mask = np.zeros((h, w, c), np.float32)
    x_len, y_len = noise.shape[0], noise.shape[1]

    # Determine patch placement location
    if patch_location == 'center' or (h == w == x_len == y_len):
        x = h // 2
        y = w // 2
    elif patch_location == 'random':
        x = np.random.randint(x_len // 2, w - x_len // 2)
        y = np.random.randint(y_len // 2, h - y_len // 2)
    else:
        raise ValueError('Invalid patch location')

    # Calculate patch coordinates
    x1 = np.clip(x - x_len // 2, 0, h)
    x2 = np.clip(x + x_len // 2, 0, h)
    y1 = np.clip(y - y_len // 2, 0, w)
    y2 = np.clip(y + y_len // 2, 0, w)
    
    # Apply patch to mask
    mask[x1: x2, y1: y2, :] = noise
    
    return mask


class PoisonCIFAR10(datasets.CIFAR10):
    """
    CIFAR10 dataset with perturbation-based poisoning.
    """
    def __init__(self, root, train=True, transform=None, target_transform=None,
                 download=False, poison_rate=1.0, perturb_tensor_filepath=None,
                 seed=0, perturb_type='classwise', patch_location='center', img_denoise=False,
                 add_uniform_noise=False, poison_classwise=False, poison_classwise_idx=None,
                 poison_class_percentage=1.0):  # New parameter for partial class poisoning
        """
        Initialize poisoned CIFAR10 dataset.
        
        Args:
            root: Dataset root directory
            train: Whether to use training set
            transform: Image transformations
            target_transform: Target transformations
            download: Whether to download dataset
            poison_rate: Portion of dataset to poison
            perturb_tensor_filepath: Path to perturbation tensor file
            seed: Random seed
            perturb_type: Type of perturbation ('classwise' or 'samplewise')
            patch_location: Location to apply perturbation patch
            img_denoise: Whether to denoise images
            add_uniform_noise: Whether to add uniform noise
            poison_classwise: Whether to poison specific classes
            poison_classwise_idx: Indices of classes to poison
            poison_class_percentage: Percentage of target class samples to poison
        """
        super(PoisonCIFAR10, self).__init__(root=root, train=train, download=download, 
                                          transform=transform, target_transform=target_transform)
        
        # Load perturbation tensor
        self.perturb_tensor = torch.load(perturb_tensor_filepath, map_location=device)
        print(self.perturb_tensor)
        
        # Convert tensor format
        if len(self.perturb_tensor.shape) == 4:
            self.perturb_tensor = self.perturb_tensor.mul(255).clamp_(0, 255).permute(0, 2, 3, 1).to('cpu').numpy()
        else:
            self.perturb_tensor = self.perturb_tensor.mul(255).clamp_(0, 255).permute(0, 1, 3, 4, 2).to('cpu').numpy()
            
        self.patch_location = patch_location
        self.img_denoise = img_denoise
        self.data = self.data.astype(np.float32)
        
        # Check shape compatibility
        target_dim = self.perturb_tensor.shape[0] if len(self.perturb_tensor.shape) == 4 else self.perturb_tensor.shape[1]
        if perturb_type == 'samplewise' and target_dim != len(self):
            raise ValueError('Poison Perturb Tensor size not match for samplewise')
        elif perturb_type == 'classwise' and target_dim != 10:
            raise ValueError('Poison Perturb Tensor size not match for classwise')

        # Select samples to poison
        self.poison_samples = collections.defaultdict(lambda: False)
        self.poison_class = []
        
        if poison_classwise:
            # Poison specific classes with percentage control
            targets = list(range(0, 10))
            if poison_classwise_idx is None:
                self.poison_class = sorted(np.random.choice(targets, int(len(targets) * poison_rate), replace=False).tolist())
            else:
                self.poison_class = poison_classwise_idx if isinstance(poison_classwise_idx, list) else [poison_classwise_idx]
                
            self.poison_samples_idx = []
            
            # Group samples by class
            class_indices = collections.defaultdict(list)
            for i, label in enumerate(self.targets):
                if label in self.poison_class:
                    class_indices[label].append(i)
            
            # For each class, select the percentage of samples to poison
            for class_idx, indices in class_indices.items():
                num_to_poison = int(len(indices) * poison_class_percentage)
                # Randomly select samples to poison
                poisoned_indices = sorted(np.random.choice(indices, num_to_poison, replace=False).tolist())
                self.poison_samples_idx.extend(poisoned_indices)
        else:
            # Poison random samples
            targets = list(range(0, len(self)))
            self.poison_samples_idx = sorted(np.random.choice(targets, int(len(targets) * poison_rate), replace=False).tolist())

        # Apply perturbations
        for idx in self.poison_samples_idx:
            self.poison_samples[idx] = True
            
            # Select perturbation
            if len(self.perturb_tensor.shape) == 5:
                perturb_id = random.choice(range(self.perturb_tensor.shape[0]))
                perturb_tensor = self.perturb_tensor[perturb_id]
            else:
                perturb_tensor = self.perturb_tensor
                
            # Apply appropriate perturbation type
            if perturb_type == 'samplewise':
                # Sample Wise poison
                noise = perturb_tensor[idx]
                noise = patch_noise_extend_to_img(noise, [32, 32, 3], patch_location=self.patch_location)
            elif perturb_type == 'classwise':
                # Class Wise Poison
                noise = perturb_tensor[self.targets[idx]]
                noise = patch_noise_extend_to_img(noise, [32, 32, 3], patch_location=self.patch_location)
                
            # Add optional uniform noise
            if add_uniform_noise:
                noise += np.random.uniform(0, 8, (32, 32, 3))

            # Apply noise to image
            self.data[idx] = self.data[idx] + noise
            self.data[idx] = np.clip(self.data[idx], a_min=0, a_max=255)
            
        # Convert back to uint8
        self.data = self.data.astype(np.uint8)
        print('add_uniform_noise: ', add_uniform_noise)
        print(self.perturb_tensor.shape)
        print(f'Poison samples: {len(self.poison_samples)}/{len(self)}')


class PoisonCIFAR100(datasets.CIFAR100):
    """
    CIFAR100 dataset with perturbation-based poisoning.
    """
    def __init__(self, root, train=True, transform=None, target_transform=None,
                 download=False, poison_rate=1.0, perturb_tensor_filepath=None,
                 seed=0, perturb_type='classwise', patch_location='center', img_denoise=False,
                 add_uniform_noise=False, poison_classwise=False, poison_class_percentage=1.0):
        """
        Initialize poisoned CIFAR100 dataset.
        
        Args:
            root: Dataset root directory
            train: Whether to use training set
            transform: Image transformations
            target_transform: Target transformations
            download: Whether to download dataset
            poison_rate: Portion of dataset to poison
            perturb_tensor_filepath: Path to perturbation tensor file
            seed: Random seed
            perturb_type: Type of perturbation ('classwise' or 'samplewise')
            patch_location: Location to apply perturbation patch
            img_denoise: Whether to denoise images
            add_uniform_noise: Whether to add uniform noise
            poison_classwise: Whether to poison specific classes
            poison_class_percentage: Percentage of target class samples to poison
        """
        super(PoisonCIFAR100, self).__init__(root=root, train=train, download=download, 
                                           transform=transform, target_transform=target_transform)
        
        # Load perturbation tensor
        self.perturb_tensor = torch.load(perturb_tensor_filepath, map_location=device)
        self.perturb_tensor = self.perturb_tensor.mul(255).clamp_(0, 255).permute(0, 2, 3, 1).to('cpu').numpy()
        
        self.patch_location = patch_location
        self.img_denoise = img_denoise
        self.data = self.data.astype(np.float32)

        # Check shape compatibility
        if perturb_type == 'samplewise' and self.perturb_tensor.shape[0] != len(self):
            raise ValueError('Poison Perturb Tensor size not match for samplewise')
        elif perturb_type == 'classwise' and self.perturb_tensor.shape[0] != 100:
            raise ValueError('Poison Perturb Tensor size not match for classwise')

        # Select samples to poison
        self.poison_samples = collections.defaultdict(lambda: False)
        self.poison_class = []
        
        if poison_classwise:
            # Poison specific classes with percentage control
            targets = list(range(0, 100))
            self.poison_class = sorted(np.random.choice(targets, int(len(targets) * poison_rate), replace=False).tolist())
            
            self.poison_samples_idx = []
            
            # Group samples by class
            class_indices = collections.defaultdict(list)
            for i, label in enumerate(self.targets):
                if label in self.poison_class:
                    class_indices[label].append(i)
            
            # For each class, select the percentage of samples to poison
            for class_idx, indices in class_indices.items():
                num_to_poison = int(len(indices) * poison_class_percentage)
                # Randomly select samples to poison
                poisoned_indices = sorted(np.random.choice(indices, num_to_poison, replace=False).tolist())
                self.poison_samples_idx.extend(poisoned_indices)
        else:
            # Poison random samples
            targets = list(range(0, len(self)))
            self.poison_samples_idx = sorted(np.random.choice(targets, int(len(targets) * poison_rate), replace=False).tolist())

        # Apply perturbations
        for idx in self.poison_samples_idx:
            self.poison_samples[idx] = True
            
            # Apply appropriate perturbation type
            if perturb_type == 'samplewise':
                # Sample Wise poison
                noise = self.perturb_tensor[idx]
                noise = patch_noise_extend_to_img(noise, [32, 32, 3], patch_location=self.patch_location)
            elif perturb_type == 'classwise':
                # Class Wise Poison
                noise = self.perturb_tensor[self.targets[idx]]
                noise = patch_noise_extend_to_img(noise, [32, 32, 3], patch_location=self.patch_location)

            # Add optional uniform noise
            if add_uniform_noise:
                noise = np.random.uniform(0, 8, (32, 32, 3))

            # Apply noise to image
            self.data[idx] += noise
            self.data[idx] = np.clip(self.data[idx], 0, 255)

        # Convert back to uint8
        self.data = self.data.astype(np.uint8)
        print('add_uniform_noise: ', add_uniform_noise)
        print(self.perturb_tensor.shape)
        print(f'Poison samples: {len(self.poison_samples)}/{len(self)}')


class Cutout(object):
    """
    Cutout data augmentation.
    
    Randomly masks out a square region in the image.
    """
    def __init__(self, length):
        """
        Initialize Cutout.
        
        Args:
            length: Side length of the cutout square
        """
        self.length = length

    def __call__(self, img):
        """
        Apply Cutout to an image.
        
        Args:
            img: Input image tensor
            
        Returns:
            Image with cutout applied
        """
        h, w = img.size(1), img.size(2)
        mask = np.ones((h, w), np.float32)
        
        # Get random cutout center
        y = np.random.randint(h)
        x = np.random.randint(w)

        # Calculate cutout coordinates
        y1 = np.clip(y - self.length // 2, 0, h)
        y2 = np.clip(y + self.length // 2, 0, h)
        x1 = np.clip(x - self.length // 2, 0, w)
        x2 = np.clip(x + self.length // 2, 0, w)

        # Apply cutout
        mask[y1: y2, x1: x2] = 0.
        mask = torch.from_numpy(mask)
        mask = mask.expand_as(img)
        img *= mask
        
        return img


class CutMix(Dataset):
    """
    CutMix data augmentation.
    
    Combines patches from one image with another, adjusting labels proportionally.
    """
    def __init__(self, dataset, num_class, num_mix=2, beta=1.0, prob=0.5):
        """
        Initialize CutMix.
        
        Args:
            dataset: Base dataset
            num_class: Number of classes
            num_mix: Number of images to mix
            beta: Beta distribution parameter
            prob: Probability of applying CutMix
        """
        self.dataset = dataset
        self.num_class = num_class
        self.num_mix = num_mix
        self.beta = beta
        self.prob = prob

    def __getitem__(self, index):
        """
        Get item with CutMix applied.
        
        Args:
            index: Item index
            
        Returns:
            Tuple of (mixed image, mixed label)
        """
        img, lb = self.dataset[index]
        lb_onehot = onehot(self.num_class, lb)

        # Apply multiple mixes
        for _ in range(self.num_mix):
            r = np.random.rand(1)
            if self.beta <= 0 or r > self.prob:
                continue

            # Generate mixed sample
            lam = np.random.beta(self.beta, self.beta)
            rand_index = random.choice(range(len(self)))

            img2, lb2 = self.dataset[rand_index]
            lb2_onehot = onehot(self.num_class, lb2)

            # Apply CutMix
            bbx1, bby1, bbx2, bby2 = rand_bbox(img.size(), lam)
            img[:, bbx1:bbx2, bby1:bby2] = img2[:, bbx1:bbx2, bby1:bby2]
            
            # Adjust lambda based on actual area ratio
            lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (img.size()[-1] * img.size()[-2]))
            
            # Mix labels
            lb_onehot = lb_onehot * lam + lb2_onehot * (1. - lam)

        return img, lb_onehot

    def __len__(self):
        """Get dataset length."""
        return len(self.dataset)


class MixUp(Dataset):
    """
    MixUp data augmentation.
    
    Linearly combines pairs of images and their labels.
    """
    def __init__(self, dataset, num_class, num_mix=2, beta=1.0, prob=0.5):
        """
        Initialize MixUp.
        
        Args:
            dataset: Base dataset
            num_class: Number of classes
            num_mix: Number of images to mix
            beta: Beta distribution parameter
            prob: Probability of applying MixUp
        """
        self.dataset = dataset
        self.num_class = num_class
        self.num_mix = num_mix
        self.beta = beta
        self.prob = prob

    def __getitem__(self, index):
        """
        Get item with MixUp applied.
        
        Args:
            index: Item index
            
        Returns:
            Tuple of (mixed image, mixed label)
        """
        img, lb = self.dataset[index]
        lb_onehot = onehot(self.num_class, lb)

        # Apply multiple mixes
        for _ in range(self.num_mix):
            r = np.random.rand(1)
            if self.beta <= 0 or r > self.prob:
                continue

            # Generate mixed sample
            lam = np.random.beta(self.beta, self.beta)
            rand_index = random.choice(range(len(self)))

            img2, lb2 = self.dataset[rand_index]
            lb2_onehot = onehot(self.num_class, lb2)

            # Mix images and labels
            img = img * lam + img2 * (1-lam)
            lb_onehot = lb_onehot * lam + lb2_onehot * (1. - lam)

        return img, lb_onehot

    def __len__(self):
        """Get dataset length."""
        return len(self.dataset)