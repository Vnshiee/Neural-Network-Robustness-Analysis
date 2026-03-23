import torch
import torch.nn as nn
import torchvision.models as models

def get_baseline_model(model_name, num_classes=10, in_channels=3):
    """
    Phase 2.1: Model Definitions
    Returns a standard baseline architecture modified for the correct number 
    of input channels and output classes.
    
    CRITICAL: This assumes your Phase 1 DataLoader resizes all images 
    (CIFAR and F-MNIST) to 224x224.
    """
    model_name = model_name.lower()
    
    if model_name == 'vgg':
        model = models.vgg16_bn(weights=None)
        
        if in_channels != 3:
            # Safely replace the first conv layer for 1-channel (F-MNIST)
            original_conv = model.features[0]
            model.features[0] = nn.Conv2d(in_channels, original_conv.out_channels, 
                                          kernel_size=original_conv.kernel_size, 
                                          stride=original_conv.stride, 
                                          padding=original_conv.padding)
            
        in_features = model.classifier[6].in_features
        model.classifier[6] = nn.Linear(in_features, num_classes)
        
    elif model_name == 'resnet':
        model = models.resnet18(weights=None)
        
        if in_channels != 3:
            # Safely replace the stem for 1-channel
            model.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
            
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, num_classes)
        
    elif model_name == 'convnext':
        model = models.convnext_tiny(weights=None)
        
        if in_channels != 3:
            # Safely replace the ConvNeXT stem
            model.features[0][0] = nn.Conv2d(in_channels, 96, kernel_size=4, stride=4)
            
        in_features = model.classifier[2].in_features
        model.classifier[2] = nn.Linear(in_features, num_classes)
        
    elif model_name == 'vit':
        # vit_b_16 strictly requires 224x224 inputs by default.
        model = models.vit_b_16(weights=None)
        
        if in_channels != 3:
            # Safely replace the patch embedding layer for 1-channel
            model.conv_proj = nn.Conv2d(in_channels, 768, kernel_size=16, stride=16)
            
        in_features = model.heads.head.in_features
        model.heads.head = nn.Linear(in_features, num_classes)
        
    else:
        raise ValueError(f"Unknown architecture: {model_name}. Allowed: 'vgg', 'resnet', 'convnext', 'vit'.")
        
    return model

# Example Usage:
# resnet_cifar10 = get_baseline_model('resnet', num_classes=10, in_channels=3)
# vgg_fmnist = get_baseline_model('vgg', num_classes=10, in_channels=1)
# vit_imagenet100 = get_baseline_model('vit', num_classes=100, in_channels=3)
