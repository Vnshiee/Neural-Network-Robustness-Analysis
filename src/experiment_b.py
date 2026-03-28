import torch
import torch.nn as nn
import torch.optim as optim
import os
import argparse
from tqdm import tqdm

from data_loader import get_dataloaders
from models.mlp import get_mlp_model
from train_baseline import train_epoch, validate_epoch

def run_experiment_b(dataset_name, corruption_name='gaussian_noise', corruption_severity=2, epochs=15, batch_size=128, lr=0.001, save_checkpoint=True, seed=42):
    """
    Phase 3.3: Experiment B (Corrupted Validation)
    Trains the exact same MLP architecture, but this time apply corruptions 
    exclusively to the validation set (the training set must remain clean).
    """
    # Fix: Set random seed to completely match Experiment A's initialization
    torch.manual_seed(seed)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"--- Running Experiment B (Corrupted Validation) on {dataset_name} ---")
    print(f"Corruption: {corruption_name} | Severity: {corruption_severity}")
    print(f"Using device: {device}")
    
    # 1. Get Mixed DataLoaders (Clean Train, Corrupted Val)
    train_dl, val_dl, _ = get_dataloaders(
        dataset_name=dataset_name, 
        batch_size=batch_size, 
        corrupt_val=True,              # Crucial for Experiment B
        corruption_name=corruption_name,
        corruption_severity=corruption_severity
    )
    
    # 2. Model Initialization (Exactly the same architecture as Exp A)
    model = get_mlp_model(dataset_name)
    model = model.to(device)
    
    # 3. Optimization Setup
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    # 4. Training Loop
    best_val_acc = 0.0
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
    
    for epoch in range(epochs):
        print(f"\nEpoch {epoch+1}/{epochs}")
        
        # Train on Clean Data
        train_loss, train_acc = train_epoch(model, train_dl, criterion, optimizer, device)
        
        # Evaluate on Corrupted Data
        val_loss, val_acc = validate_epoch(model, val_dl, criterion, device)
        
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        
        print(f"Train Loss (Clean): {train_loss:.4f} | Train Acc (Clean): {train_acc:.2f}%")
        print(f"Val Loss ({corruption_name}): {val_loss:.4f} | Val Acc ({corruption_name}): {val_acc:.2f}%")
        
        # 5. Checkpointing logic driven purely by Corrupted Validation Accuracy
        if val_acc > best_val_acc:
            print(f"-> Robust Validation accuracy improved ({best_val_acc:.2f}% -> {val_acc:.2f}%).")
            best_val_acc = val_acc
            
            if save_checkpoint:
                print("Saving model...")
                os.makedirs('checkpoints', exist_ok=True)
                checkpoint_path = f"checkpoints/mlp_{dataset_name}_exp_b_corrupt_{corruption_name}_sev{corruption_severity}.pth"
                
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'best_val_acc': best_val_acc,
                    'corruption_type': corruption_name,
                    'corruption_severity': corruption_severity
                }, checkpoint_path)

    # 6. Plotting and Tracking
    import matplotlib.pyplot as plt
    import csv
    os.makedirs('results/plots', exist_ok=True)
    
    plt.figure(figsize=(12, 5))
    epochs_range = range(1, epochs + 1)
    
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, history['train_loss'], label='Train Loss (Clean)', marker='o')
    plt.plot(epochs_range, history['val_loss'], label=f'Val Loss ({corruption_name})', marker='o')
    plt.title(f'MLP Exp B ({dataset_name}) - Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, history['train_acc'], label='Train Acc (Clean)', marker='o')
    plt.plot(epochs_range, history['val_acc'], label=f'Val Acc ({corruption_name})', marker='o')
    plt.title(f'MLP Exp B ({dataset_name}) - Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(f"results/plots/mlp_exp_b_{dataset_name}_{corruption_name}_curves.png", dpi=300)
    plt.close()
    
    with open(f"results/mlp_exp_b_{dataset_name}_{corruption_name}_training_log.csv", mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Epoch', 'Train_Loss', 'Train_Acc', 'Val_Loss', 'Val_Acc'])
        for i in range(epochs):
            writer.writerow([i+1, history['train_loss'][i], history['train_acc'][i], history['val_loss'][i], history['val_acc'][i]])
            
    print(f"Experiment B Completed. Best Corrupt Val Acc: {best_val_acc:.2f}%")
    return best_val_acc

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Phase 3.3: MLP Experiment B (Corrupted Validation)")
    parser.add_argument('--dataset', type=str, required=True, choices=['cifar10', 'fmnist', 'imagenet100'], help='Target dataset')
    parser.add_argument('--corruption_name', type=str, default='gaussian_noise', help='Type of corruption to apply to the validation set')
    parser.add_argument('--corruption_severity', type=int, default=2, choices=[1,2,3,4,5], help='Severity 1-5')
    parser.add_argument('--epochs', type=int, default=15, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    
    args = parser.parse_args()
    run_experiment_b(
        dataset_name=args.dataset,
        corruption_name=args.corruption_name,
        corruption_severity=args.corruption_severity,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr
    )
