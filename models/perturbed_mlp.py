import torch
import torch.nn as nn

class FeatureNoiseInjector(nn.Module):
    """
    A simple module that injects Gaussian noise into intermediate feature maps 
    during the forward pass. This forces the downstream layers to learn 
    representations that are robust to internal activations becoming corrupted.
    """
    def __init__(self, noise_std=0.1, active_during_eval=False):
        """
        Args:
            noise_std (float): Standard deviation of the Gaussian noise.
            active_during_eval (bool): Whether to keep perturbing features during 
                                       model.eval() mode (like testing robustness).
        """
        super(FeatureNoiseInjector, self).__init__()
        self.noise_std = noise_std
        self.active_during_eval = active_during_eval

    def forward(self, x):
        # Only inject noise if we are actively training OR if explicitly requested during eval
        if self.training or self.active_during_eval:
            # Generate random gaussian noise matching the exact tensor schema/device of 'x'
            noise = torch.randn_like(x) * self.noise_std
            return x + noise
        return x

class PerturbedMLP(nn.Module):
    """
    Phase 5.1: Interior Feature Perturbation
    A wrapper built around the standard MLP from Phase 3. 
    It allows dynamic injection of statistical noise into specific network stages.
    
    Locations Map:
    - 'early':  Injects noise after the first hidden layer block.
    - 'middle': Injects noise after the middle hidden layer block.
    - 'late':   Injects noise after the final hidden layer block (penultimate features).
    - 'all':    Injects noise at every stage simultaneously.
    """
    def __init__(self, original_mlp, perturbation_location='middle', noise_std=0.2):
        super(PerturbedMLP, self).__init__()
        
        # 1. Inherit the core components
        self.flatten = original_mlp.flatten
        # The underlying network is: [Linear, BN, ReLU, Dropout,  Linear, BN, ReLU, Dropout,  Linear, BN, ReLU, Dropout,  Linear(Head)]
        # We need to extract the layers to manually walk through them and inject noise
        self.layers = list(original_mlp.network.children())
        self.location = perturbation_location.lower()
        self.injector = FeatureNoiseInjector(noise_std=noise_std)
        
        # 2. Map blocks (Since each fundamental block uses exactly 4 PyTorch components: Linear->BN->ReLU->Dropout)
        # Block 1 = layers 0 to 3
        # Block 2 = layers 4 to 7
        # Block 3 = layers 8 to 11
        # Head    = layer 12
        self.block1 = nn.Sequential(*self.layers[0:4])
        self.block2 = nn.Sequential(*self.layers[4:8])
        self.block3 = nn.Sequential(*self.layers[8:12])
        self.head   = self.layers[12]

    def forward(self, x):
        x = self.flatten(x)
        
        # --- Early Stage ---
        x = self.block1(x)
        if self.location in ['early', 'all']:
            x = self.injector(x)
            
        # --- Middle Stage ---
        x = self.block2(x)
        if self.location in ['middle', 'all']:
            x = self.injector(x)
            
        # --- Late Stage (Penultimate Features) ---
        x = self.block3(x)
        if self.location in ['late', 'all']:
            x = self.injector(x)
            
        # --- Final Classification Head ---
        logits = self.head(x)
        
        return logits

# Example Usage
# from models.mlp import get_mlp_model
# base_model = get_mlp_model('cifar10')
# early_perturb_model = PerturbedMLP(base_model, perturbation_location='early', noise_std=0.5)
