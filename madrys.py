import torch
import torch.nn as nn
import torch.nn.functional as F

# Set device for computation
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class MadrysLoss(nn.Module):
    """
    Implementation of the adversarial training loss from Madry et al.
    "Towards Deep Learning Models Resistant to Adversarial Attacks"
    
    This loss creates adversarial examples on-the-fly during training and
    trains the model on these examples.
    """
    def __init__(self, step_size=0.007, epsilon=0.031, perturb_steps=10, distance='l_inf', cutmix=False):
        """
        Initialize Madry's adversarial training loss.
        
        Args:
            step_size: Step size for generating adversarial examples
            epsilon: Maximum perturbation magnitude
            perturb_steps: Number of steps for generating adversarial examples
            distance: Distance metric ('l_inf' supported)
            cutmix: Whether to use CutMix loss
        """
        super(MadrysLoss, self).__init__()
        self.step_size = step_size
        self.epsilon = epsilon
        self.perturb_steps = perturb_steps
        self.distance = distance
        # Set up the appropriate cross entropy loss
        if cutmix:
            from models import CutMixCrossEntropyLoss
            self.cross_entropy = CutMixCrossEntropyLoss()
        else:
            self.cross_entropy = torch.nn.CrossEntropyLoss()

    def forward(self, model, x_natural, y, optimizer):
        """
        Forward pass creating adversarial examples and computing loss.
        
        Args:
            model: Model being trained
            x_natural: Clean input images
            y: Target labels
            optimizer: Optimizer for model parameters
            
        Returns:
            Tuple of (logits on adversarial examples, adversarial loss)
        """
        # Set model to eval mode for generating adversarial examples
        model.eval()
        for param in model.parameters():
            param.requires_grad = False

        # Generate adversarial examples
        x_adv = x_natural.clone() + self.step_size * torch.randn(x_natural.shape).to(device)
        
        if self.distance == 'l_inf':
            for _ in range(self.perturb_steps):
                # Setup for gradient calculation
                x_adv.requires_grad_()
                
                # Forward pass and loss calculation
                loss_ce = self.cross_entropy(model(x_adv), y)
                
                # Gradient calculation
                grad = torch.autograd.grad(loss_ce, [x_adv])[0]
                
                # PGD step
                x_adv = x_adv.detach() + self.step_size * torch.sign(grad.detach())
                
                # Project back to epsilon ball and valid image range
                x_adv = torch.min(torch.max(x_adv, x_natural - self.epsilon), x_natural + self.epsilon)
                x_adv = torch.clamp(x_adv, 0.0, 1.0)
        else:
            # For other norms, just clamp to valid range
            x_adv = torch.clamp(x_adv, 0.0, 1.0)

        # Set model back to train mode for training update
        for param in model.parameters():
            param.requires_grad = True
            
        model.train()
        
        # Zero gradients before computing loss on adversarial examples
        optimizer.zero_grad()
        
        # Compute logits and loss on adversarial examples
        logits = model(x_adv)
        loss = self.cross_entropy(logits, y)

        return logits, loss