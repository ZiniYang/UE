import numpy as np
import torch
import torch.optim as optim

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class PerturbationTool():
    """Tool for generating adversarial perturbations."""
    
    def __init__(self, seed=0, epsilon=0.03137254901, num_steps=20, step_size=0.00784313725):
        """Initialize with perturbation paraxmeters."""
        self.epsilon = epsilon
        self.num_steps = num_steps
        self.step_size = step_size
        self.seed = seed
        np.random.seed(seed)

    def random_noise(self, noise_shape=[10, 3, 32, 32]):
        """Generate random uniform noise within epsilon bounds."""
        return torch.FloatTensor(*noise_shape).uniform_(-self.epsilon, self.epsilon).to(device)

    def min_min_attack(self, images, labels, model, optimizer, criterion, random_noise=None, sample_wise=False):
        """Generate min-min perturbations (minimize loss for unlearnable examples)."""
        if random_noise is None:
            random_noise = torch.FloatTensor(*images.shape).uniform_(-self.epsilon, self.epsilon).to(device)

        perturb_img = images.data + random_noise
        perturb_img = torch.clamp(perturb_img, 0, 1).requires_grad_(True)
        eta = random_noise
        
        for _ in range(self.num_steps):
            opt = optim.SGD([perturb_img], lr=1e-3)
            opt.zero_grad()
            model.zero_grad()
            
            if isinstance(criterion, torch.nn.CrossEntropyLoss):
                if hasattr(model, 'classify'):
                    model.classify = True
                logits = model(perturb_img)
                loss = criterion(logits, labels)
            else:
                logits, loss = criterion(model, perturb_img, labels, optimizer)
                
            perturb_img.retain_grad()
            loss.backward()
            
            eta = self.step_size * perturb_img.grad.data.sign() * (-1)
            perturb_img = perturb_img.data + eta
            
            eta = torch.clamp(perturb_img.data - images.data, -self.epsilon, self.epsilon)
            perturb_img = images.data + eta
            perturb_img = torch.clamp(perturb_img, 0, 1).requires_grad_(True)

        return perturb_img, eta

    def min_max_attack(self, images, labels, model, optimizer, criterion, random_noise=None, sample_wise=False):
        """Generate min-max perturbations (maximize loss for adversarial examples)."""
        if random_noise is None:
            random_noise = torch.FloatTensor(*images.shape).uniform_(-self.epsilon, self.epsilon).to(device)

        perturb_img = images.data + random_noise
        perturb_img = torch.clamp(perturb_img, 0, 1).requires_grad_(True)
        eta = random_noise
        
        for _ in range(self.num_steps):
            opt = optim.SGD([perturb_img], lr=1e-3)
            opt.zero_grad()
            model.zero_grad()
            
            if isinstance(criterion, torch.nn.CrossEntropyLoss):
                logits = model(perturb_img)
                loss = criterion(logits, labels)
            else:
                logits, loss = criterion(model, perturb_img, labels, optimizer)
                
            loss.backward()
            
            eta = self.step_size * perturb_img.grad.data.sign()
            perturb_img = perturb_img.data + eta
            
            eta = torch.clamp(perturb_img.data - images.data, -self.epsilon, self.epsilon)
            perturb_img = images.data + eta
            perturb_img = torch.clamp(perturb_img, 0, 1).requires_grad_(True)

        return perturb_img, eta

    def _patch_noise_extend_to_img(self, noise, image_size=[3, 32, 32], patch_location='center'):
        """Extend patch noise to full image by placing at specified location."""
        c, h, w = image_size[0], image_size[1], image_size[2]
        mask = np.zeros((c, h, w), np.float32)
        x_len, y_len = noise.shape[1], noise.shape[1]

        if patch_location == 'center' or (h == w == x_len == y_len):
            x, y = h // 2, w // 2
        elif patch_location == 'random':
            x = np.random.randint(x_len // 2, w - x_len // 2)
            y = np.random.randint(y_len // 2, h - y_len // 2)
        else:
            raise ValueError('Invalid patch location')

        x1 = np.clip(x - x_len // 2, 0, h)
        x2 = np.clip(x + x_len // 2, 0, h)
        y1 = np.clip(y - y_len // 2, 0, w)
        y2 = np.clip(y + y_len // 2, 0, w)
        
        if isinstance(noise, np.ndarray):
            mask[:, x1: x2, y1: y2] = noise
        else:
            mask[:, x1: x2, y1: y2] = noise.cpu().numpy()
            
        return ((x1, x2, y1, y2), torch.from_numpy(mask).to(device))