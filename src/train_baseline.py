import torch
import torch.nn as nn
import torch.optim as optim
import os
import argparse
from tqdm import tqdm

from data_loader import get_dataloaders
from models.baseline_models import get_baseline_model

def train_epoch(model, dataloader, criterion, optimizer, device):
    """
    Phase 2.2: Standard training loop for a single epoch.
    """
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for inputs, labels in tqdm(dataloader, desc="Training", leave=False):
        inputs, labels = inputs.to(device), labels.to(device)
        
        optimizer.zero_grad()
        
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item() * inputs.size(0)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        
    epoch_loss = running_loss / total
    epoch_acc = 100. * correct / total
    return epoch_loss, epoch_acc

def validate_epoch(model, dataloader, criterion, device):
    """
    Phase 2.2: Standard validation loop.
    """
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for inputs, labels in tqdm(dataloader, desc="Validating", leave=False):
            inputs, labels = inputs.to(device), labels.to(device)
            
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
    epoch_loss = running_loss / total
    epoch_acc = 100. * correct / total
    return epoch_loss, epoch_acc

def train_baseline(model_name, dataset_name, epochs=10, batch_size=128, lr=0.001):
    """
    Phase 2.2: Master function to train standard architectures on clean datasets.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Training {model_name} on {dataset_name} using computing device: {device}")
    
    # 1. Dataset Configuration
    # Fashion-MNIST is single-channel grayscale, CIFAR-10 and ImageNet-100 are 3-channel
    in_channels = 1 if dataset_name == 'fmnist' else 3
    num_classes = 100 if dataset_name == 'imagenet100' else 10
    
    # Get Clean DataLoaders
    # By default, data_loader.py gets Clean Training and Clean Validation sets
    train_dl, val_dl, test_dl = get_dataloaders(
        dataset_name=dataset_name, 
        batch_size=batch_size, 
        corrupt_val=False # explicitly state we want clean validation for baseline
    )
    
    # 2. Model Initialization
    model = get_baseline_model(model_name, num_classes=num_classes, in_channels=in_channels)
    model = model.to(device)
    
    # 3. Optimization Setup
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    # 4. Training Loop
    best_val_acc = 0.0
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
    
    for epoch in range(epochs):
        print(f"\nEpoch {epoch+1}/{epochs}")
        
        train_loss, train_acc = train_epoch(model, train_dl, criterion, optimizer, device)
        val_loss, val_acc = validate_epoch(model, val_dl, criterion, device)
        
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        
        print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
        print(f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%")
        
        # 5. Checkpointing logic
        if val_acc > best_val_acc:
            print(f"-> Validation accuracy improved ({best_val_acc:.2f}% -> {val_acc:.2f}%). Saving model...")
            best_val_acc = val_acc
            
            os.makedirs('checkpoints', exist_ok=True)
            checkpoint_path = f"checkpoints/{model_name}_{dataset_name}_clean_best.pth"
            
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_val_acc': best_val_acc
            }, checkpoint_path)

    # 6. Plotting and saving history
    import matplotlib.pyplot as plt
    os.makedirs('results/plots', exist_ok=True)
    
    epochs_range = range(1, epochs + 1)
    plt.figure(figsize=(12, 5))
    
    # Loss plot
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, history['train_loss'], label='Train Loss', marker='o')
    plt.plot(epochs_range, history['val_loss'], label='Val Loss', marker='o')
    plt.title(f'{model_name.upper()} on {dataset_name} - Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    # Accuracy plot
    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, history['train_acc'], label='Train Acc', marker='o')
    plt.plot(epochs_range, history['val_acc'], label='Val Acc', marker='o')
    plt.title(f'{model_name.upper()} on {dataset_name} - Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(f"results/plots/{model_name}_{dataset_name}_training_curves.png", dpi=300)
    plt.close()
    
    # Export raw history to CSV for report tables
    import csv
    with open(f"results/{model_name}_{dataset_name}_training_log.csv", mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Epoch', 'Train_Loss', 'Train_Acc', 'Val_Loss', 'Val_Acc'])
        for i in range(epochs):
            writer.writerow([i+1, history['train_loss'][i], history['train_acc'][i], history['val_loss'][i], history['val_acc'][i]])

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Phase 2.2 Baseline Training")
    parser.add_argument('--model', type=str, required=True, choices=['vgg', 'resnet', 'convnext', 'vit'], help='Target architecture')
    parser.add_argument('--dataset', type=str, required=True, choices=['cifar10', 'fmnist', 'imagenet100'], help='Target dataset')
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    
    args = parser.parse_args()
    train_baseline(args.model, args.dataset, epochs=args.epochs, batch_size=args.batch_size, lr=args.lr)
