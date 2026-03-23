import torch
import torch.nn as nn

class MLP(nn.Module):
    """
    Phase 3.1: MLP (Multi-Layer Perceptron) Architecture
    A configurable feed-forward neural network for baseline and corruption experiments.
    """
    def __init__(self, input_dim, hidden_layers=None, num_classes=10, dropout_rate=0.2):
        """
        Args:
            input_dim (int): Total number of input features (H * W * C).
            hidden_layers (list): List of integers representing hidden layer sizes.
            num_classes (int): Number of output classes.
            dropout_rate (float): Dropout probability for regularization.
        """
        super(MLP, self).__init__()
        
        if hidden_layers is None:
            hidden_layers = [512, 256, 128]
        
        self.input_dim = input_dim
        self.flatten = nn.Flatten()
        
        layers = []
        in_features = input_dim
        
        # Build hidden layers dynamically
        for hidden_dim in hidden_layers:
            layers.append(nn.Linear(in_features, hidden_dim))
            layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.ReLU(inplace=True))
            layers.append(nn.Dropout(dropout_rate))
            in_features = hidden_dim
            
        # Final classification head
        layers.append(nn.Linear(in_features, num_classes))
        
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        # Flatten image inputs (e.g., B x 3 x 32 x 32 -> B x 3072)
        x = self.flatten(x)
        return self.network(x)

def get_mlp_model(dataset_name, hidden_layers=None, dropout_rate=0.2):
    """
    Helper function to instantiate the MLP to match the datasets.
    """
    if hidden_layers is None:
        hidden_layers = [512, 256, 128]

    if dataset_name == 'cifar10':
        # All datasets are upscaled to 224x224 in data_loader to prevent ViT/ConvNext crashes
        input_dim = 3 * 224 * 224
        num_classes = 10
    elif dataset_name == 'fmnist':
        # All datasets are upscaled to 224x224 in data_loader
        input_dim = 1 * 224 * 224
        num_classes = 10
    elif dataset_name == 'imagenet100':
        # Assuming we resized/cropped to 224x224 in data_loader
        input_dim = 3 * 224 * 224
        num_classes = 100
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
        
    return MLP(input_dim=input_dim, hidden_layers=hidden_layers, 
                 num_classes=num_classes, dropout_rate=dropout_rate)

# Example Usage:
# model = get_mlp_model('cifar10')
