import torch
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader, random_split, Dataset
import os
import numpy as np
from PIL import Image

try:
    from imagecorruptions import corrupt
except ImportError as e:
    print(f"Warning: 'imagecorruptions' failed to import. Error: {e}")
    print("Perturbations will fail until resolved. If this is a libGL error on a server, run: pip install opencv-python-headless")
    def corrupt(image, corruption_name, severity):
        return image

class ApplyCorruption:
    """
    Phase 1.3: Apply corruptions dynamically using the imagecorruptions library.
    Acts as a torchvision transform. Checks if corruption_name is a list (randomly picks one).
    """
    def __init__(self, corruption_name, severity=2):
        self.corruption_name = corruption_name
        self.severity = severity

    def __call__(self, img):
        # Convert PIL Image to numpy array
        img_np = np.array(img)
        
        # imagecorruptions expects 3-channel (H, W, 3) arrays.
        # Handle grayscale images like Fashion-MNIST
        is_grayscale = False
        if len(img_np.shape) == 2:
            img_np = np.stack((img_np,) * 3, axis=-1)
            is_grayscale = True
            
        # Phase 3C: If corruption_name is a list, choose one randomly per image
        current_corruption = self.corruption_name
        if isinstance(self.corruption_name, list):
            import random
            current_corruption = random.choice(self.corruption_name)
            
        corrupted_img = corrupt(img_np, corruption_name=current_corruption, severity=self.severity)
        
        # Convert back to grayscale safely
        if is_grayscale:
            # Convert numpy array to PIL RGB, then to Grayscale ('L')
            corrupted_pil = Image.fromarray(corrupted_img, 'RGB').convert('L')
            return corrupted_pil
            
        return Image.fromarray(corrupted_img)

class DatasetWrapper(Dataset):
    """
    Wraps a PyTorch dataset/subset to apply specific transforms.
    Required because random_split shares the parent dataset's transform.
    """
    def __init__(self, subset, transform=None):
        self.subset = subset
        self.transform = transform
        
    def __getitem__(self, index):
        img, label = self.subset[index]
        if self.transform:
            img = self.transform(img)
        return img, label
        
    def __len__(self):
        return len(self.subset)

    def __getattr__(self, attr):
        # Prevent forwarding dunder methods (like __getitems__) to avoid 
        # bypassing this wrapper's __getitem__ logic during DataLoader batching.
        if attr.startswith('__') and attr.endswith('__'):
            raise AttributeError(f"Attribute '{attr}' not found in DatasetWrapper")
            
        # Forward attribute access to the underlying subset/dataset
        if hasattr(self.subset, attr):
            return getattr(self.subset, attr)
        # If it's a PyTorch Subset, the original dataset attributes are in .dataset
        if hasattr(self.subset, 'dataset') and hasattr(self.subset.dataset, attr):
            return getattr(self.subset.dataset, attr)
        raise AttributeError(f"Attribute '{attr}' not found in DatasetWrapper, nor its subset.")

def get_transforms(dataset_name, corruption_name=None, severity=2):
    """
    Returns the appropriate transforms for the given dataset.
    Phase 1.3: Optionally prepends an image corruption transform before standard ToTensor.
    """
    transforms_list = []

    if dataset_name == 'cifar10':
        transforms_list.extend([
            transforms.Resize(224), # Fix: Upscale to Standard Model Dimensions
        ])
        if corruption_name:
            transforms_list.append(ApplyCorruption(corruption_name, severity))
        # Standard normalization for CIFAR-10
        transforms_list.extend([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
    elif dataset_name == 'fmnist':
        transforms_list.extend([
            transforms.Resize(224), # Fix: Upscale to Standard Model Dimensions
        ])
        if corruption_name:
            transforms_list.append(ApplyCorruption(corruption_name, severity))
        # Standard normalization for Fashion-MNIST (Grayscale)
        transforms_list.extend([
            transforms.ToTensor(),
            transforms.Normalize((0.2860,), (0.3530,)),
        ])
    elif dataset_name == 'imagenet100':
        # Standard ResNet-style ImageNet transforms (resize to 224x224)
        transforms_list.extend([
            transforms.Resize(256),
            transforms.CenterCrop(224),
        ])
        
        # Phase 1.3: Inject corruption AFTER resize/crop, before ToTensor
        if corruption_name:
            transforms_list.append(ApplyCorruption(corruption_name, severity))
            
        transforms_list.extend([
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    
    return transforms.Compose(transforms_list)

def get_train_val_split(full_train_dataset, val_split=0.2, seed=42):
    """
    Phase 1.2: Data Splitting Strategy.
    Splits exactly `val_split` portion of the training data as a validation set.
    Uses a fixed random seed to guarantee the EXACT same 80/20 train/val split subsets 
    are used consistently across all subsequent experiments, preserving consistency.
    """
    total_train_size = len(full_train_dataset)
    val_size = int(total_train_size * val_split) # Example: 20% validation
    train_size = total_train_size - val_size     # Example: 80% training
    
    # Isolate the random seed specifically for the split operation
    generator = torch.Generator().manual_seed(seed)
    
    train_dataset, val_dataset = random_split(
        full_train_dataset, 
        [train_size, val_size], 
        generator=generator
    )
    return train_dataset, val_dataset

def get_dataloaders(dataset_name, data_dir='./data', batch_size=128, val_split=0.2, seed=42, 
                    corrupt_val=False, corrupt_test=False, corruption_name='gaussian_noise', corruption_severity=2, num_workers=0):
    """
    Loads the requested dataset, performs an exact 80/20 train/val split, 
    and optionally corrupts the validation and/or test set (Phase 1.3/4).
    
    Args:
        ... existing args ...
        corrupt_val (bool): Whether to explicitly corrupt the Validation set.
        corrupt_test (bool): Whether to explicitly corrupt the Test set.
        corruption_name (str): The name of the typical corruption (e.g. 'gaussian_noise').
        corruption_severity (int): Severity index (1 to 5).
        num_workers (int): Number of subprocesses to use for data loading.
        
    Returns:
        tuple: (train_loader, val_loader, test_loader)
    """
    # 1. Base Transforms (No corruption for training by default)
    train_transform = get_transforms(dataset_name)
    
    if corrupt_test:
        test_transform = get_transforms(dataset_name, corruption_name=corruption_name, severity=corruption_severity)
    else:
        test_transform = get_transforms(dataset_name)
    
    # Val Transform varies based on the Phase 1.3 switch
    if corrupt_val:
        val_transform = get_transforms(dataset_name, corruption_name=corruption_name, severity=corruption_severity)
    else:
        val_transform = get_transforms(dataset_name)
    
    # 2. Load the corresponding dataset WITHOUT transforms initially 
    # to avoid the split sharing the same transform across train and validation subsets.
    if dataset_name == 'cifar10':
        full_train_dataset = torchvision.datasets.CIFAR10(root=data_dir, train=True, download=True, transform=None)
        test_dataset = torchvision.datasets.CIFAR10(root=data_dir, train=False, download=True, transform=test_transform)
        
    elif dataset_name == 'fmnist':
        full_train_dataset = torchvision.datasets.FashionMNIST(root=data_dir, train=True, download=True, transform=None)
        test_dataset = torchvision.datasets.FashionMNIST(root=data_dir, train=False, download=True, transform=test_transform)
        
    elif dataset_name == 'imagenet100':
        train_dir = os.path.join(data_dir, 'imagenet100', 'train')
        test_dir = os.path.join(data_dir, 'imagenet100', 'val')
        
        if not os.path.exists(train_dir) or not os.path.exists(test_dir):
            raise RuntimeError(f"ImageNet-100 data missing at {data_dir}/imagenet100. Please run `python download_imagenet.py` first.")
            
        full_train_dataset = torchvision.datasets.ImageFolder(root=train_dir, transform=None)
        test_dataset = torchvision.datasets.ImageFolder(root=test_dir, transform=test_transform)
        
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    # 3. Perform the exact split (Phase 1.2 implementation)
    train_subset, val_subset = get_train_val_split(full_train_dataset, val_split, seed)

    # 4. Wrap subsets to apply individual transforms explicitly (Phase 1.3 capability)
    train_dataset = DatasetWrapper(train_subset, transform=train_transform)
    val_dataset = DatasetWrapper(val_subset, transform=val_transform)

    # 5. Create DataLoaders
    # FIX: Increased num_workers from 0 to 8 for massive A100 server IO throughput.
    # On Windows, num_workers=0 is required, but on Linux Server it causes massive CPU bottlenecking.
    server_num_workers = 8 
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=server_num_workers, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=server_num_workers, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=server_num_workers, pin_memory=True)

    print(f"[{dataset_name.upper()}] Data Loaded:")
    print(f" - Train samples: {len(train_dataset)}")
    print(f" - Val samples:   {len(val_dataset)}")
    print(f" - Test samples:  {len(test_dataset)}")

    return train_loader, val_loader, test_loader

# Example Usage:
# train_dl, val_dl, test_dl = get_dataloaders('cifar10', val_split=0.2)