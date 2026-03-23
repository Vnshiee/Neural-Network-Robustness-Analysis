import torch
import torch.nn as nn
import torch.optim as optim
import os
import argparse
import random
from imagecorruptions import get_corruption_names

from data_loader import get_dataloaders
from models.mlp import get_mlp_model
from train_baseline import train_epoch, validate_epoch

def run_optimization_experiment(dataset_name, num_corruptions, epochs=15, batch_size=128, lr=0.001, seed=42):
    """
    Phase 3C: Optimize the amount of corruption needed to corrupt the validation set.
    """
    torch.manual_seed(seed)
    random.seed(seed)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n=========================================================")
    print(f" Phase 3C: Optimizing Validation Corruptions ({num_corruptions} corruptions)")
    print(f" Dataset: {dataset_name}")
    print(f"=========================================================")
    
    # Select N corruptions randomly from the library 
    all_corruptions = get_corruption_names()
    
    if num_corruptions > len(all_corruptions):
        num_corruptions = len(all_corruptions)
        
    selected_corruptions = random.sample(all_corruptions, k=num_corruptions)
    print(f"Using exactly {num_corruptions} corruptions in Validation Set:")
    print(f"{selected_corruptions}")
    
    # 1. Get Mixed DataLoaders (Clean Train, Multi-Corrupted Val)
    train_dl, val_dl, _ = get_dataloaders(
        dataset_name=dataset_name, 
        batch_size=batch_size, 
        corrupt_val=True,              
        corruption_name=selected_corruptions, # Passes a list: applies 1 randomly per image
        corruption_severity=2                 # Fixed severity to 2 per assignment instructions
    )
    
    # 2. Model Initialization
    model = get_mlp_model(dataset_name).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    best_val_acc = 0.0
    
    # 3. Training Loop
    for epoch in range(epochs):
        train_loss, train_acc = train_epoch(model, train_dl, criterion, optimizer, device)
        val_loss, val_acc = validate_epoch(model, val_dl, criterion, device)
        
        print(f"Epoch [{epoch+1}/{epochs}] Clean Train Acc: {train_acc:.2f}% | Multi-Corrupt Val Acc: {val_acc:.2f}%")
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            # Save the checkpoint tracking the amount of corruptions
            os.makedirs('checkpoints', exist_ok=True)
            checkpoint_path = f"checkpoints/mlp_{dataset_name}_exp_c_optimal_corrupt_{num_corruptions}.pth"
            torch.save({
                'model_state_dict': model.state_dict(),
                'best_val_acc': best_val_acc,
                'num_corruptions': num_corruptions,
                'corruptions_used': selected_corruptions
            }, checkpoint_path)
            
    print(f"-> Best Acc for N={num_corruptions} Validation Corruptions: {best_val_acc:.2f}%")
    return best_val_acc

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Phase 3.4: Optimize amount of corruption")
    parser.add_argument('--dataset', type=str, required=True, choices=['cifar10', 'fmnist', 'imagenet100'])
    parser.add_argument('--epochs', type=int, default=15)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--lr', type=float, default=0.001)
    args = parser.parse_args()
    
    # We will test using 1, 5, 10, and ALL 15 corruptions to see which yields optimal mapping.
    amounts_to_test = [1, 5, 10, 15]
    results = {}
    
    for n in amounts_to_test:
        best_acc = run_optimization_experiment(
            dataset_name=args.dataset,
            num_corruptions=n,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr
        )
        results[n] = best_acc
        
    print("\n=========================================================")
    print("           OPTIMIZATION SWEEP RESULTS                      ")
    print("=========================================================")
    print("Number of Corruptions | Peak Robust Validation Accuracy")
    for n, acc in results.items():
        print(f"       {n:2d} corruptions  |    {acc:.2f}%")
    
    best_n = max(results, key=results.get)
    print(f"Conclusion: The optimal amount of validation corruption is {best_n} corruptions.")
    print("=========================================================")
    
    # Generate the optimization plot
    import matplotlib.pyplot as plt
    try:
        os.makedirs('results/plots', exist_ok=True)
        plt.figure(figsize=(8, 5))
        plt.plot(list(results.keys()), list(results.values()), marker='o', linestyle='-', color='purple', linewidth=2)
        plt.title(f'Phase 3C: Validation Corruption Optimization ({args.dataset})')
        plt.xlabel('Number of Unique Corruptions in Validation Set')
        plt.ylabel('Peak Robust Validation Accuracy (%)')
        plt.grid(True)
        # Mark the optimal point
        plt.scatter([best_n], [results[best_n]], color='gold', s=150, zorder=5, edgecolors='black', label=f'Optimal: {best_n}')
        plt.legend()
        
        plt.tight_layout()
        plt.savefig(f"results/plots/optimize_corruptions_{args.dataset}.png", dpi=300)
        plt.close()
    except Exception as e:
        pass
