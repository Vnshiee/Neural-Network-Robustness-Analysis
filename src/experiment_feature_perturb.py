import torch
import torch.nn as nn
import torch.optim as optim
import argparse
import os
import csv
from tqdm import tqdm

from data_loader import get_dataloaders
from models.mlp import get_mlp_model
from models.perturbed_mlp import PerturbedMLP
from evaluate_models import evaluate_model
from train_baseline import train_epoch

def run_feature_perturbation_experiment(dataset_name, noise_std=0.2, epochs=15, batch_size=128, lr=0.001):
    """
    Phase 5.2: Layer-wise feature map perturbation experiments.
    Trains parallel networks with internal noise injected at various distinct 
    topological depths (early, middle, late, all) to see which layer's integrity 
    is most vital for overall model robustness.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"=== Starting Bonus Phase 5: Feature Perturbation on {dataset_name} ===")
    print(f"Noise Deviation: {noise_std} | Device: {device}")
    
    # 1. Setup Dataloaders
    # We will train on clean data, but the `PerturbedMLP` wrapper natively 
    # synthesizes and injects noise internally into the geometric space during the forward passes.
    train_dl, clean_val_dl, clean_test_dl = get_dataloaders(
        dataset_name=dataset_name, 
        batch_size=batch_size, 
        corrupt_val=False,
        corrupt_test=False 
    )
    
    # Also evaluate robustness against one standard external physical image corruption 
    print("Loading external validation/test metrics...")
    _, corrupt_val_dl, corrupt_test_dl = get_dataloaders(
        dataset_name=dataset_name, 
        batch_size=batch_size, 
        corrupt_val=True, 
        corrupt_test=True,
        corruption_name='gaussian_noise',
        corruption_severity=3
    )

    LOCATIONS = ['baseline (no noise)', 'early', 'middle', 'late', 'all']
    results = []

    # 2. Iteratively test each topological location
    for location in LOCATIONS:
        print(f"\n---> Training Network Geometry: [{location.upper()}] <---")
        
        # Instantiate fresh Base MLP
        base_model = get_mlp_model(dataset_name)
        
        if location == 'baseline (no noise)':
            model = base_model
        else:
            # Wrap Base MLP into the physical internal noise injector
            model = PerturbedMLP(base_model, perturbation_location=location, noise_std=noise_std)
            
        model = model.to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=lr)
        
        # 3. Training execution
        for epoch in range(epochs):
            train_loss, train_acc = train_epoch(model, train_dl, criterion, optimizer, device)
            print(f"   Epoch {epoch+1}/{epochs} | Train Loss: {train_loss:.3f} | Train Acc: {train_acc:.2f}%")
            
        # 4. Deep Metric Validation Mapping
        # We test both physical world evaluation tests.
        clean_test_acc = evaluate_model(model, clean_test_dl, device)
        corrupt_test_acc = evaluate_model(model, corrupt_test_dl, device)
        
        print(f" ✓ Final Clean Acc:   {clean_test_acc:.2f}%")
        print(f" ✓ Final Corrupt Acc: {corrupt_test_acc:.2f}%")
        
        results.append({
            'Perturbation_Location': location,
            'Noise_Dev': noise_std,
            'Clean_Test_Accuracy': f"{clean_test_acc:.2f}",
            'Corrupted_Test_Accuracy': f"{corrupt_test_acc:.2f}"
        })
        
        # Output checkpoints uniquely for future studies
        os.makedirs('checkpoints', exist_ok=True)
        torch.save({
            'model_state_dict': model.state_dict(),
            'layer_location': location
        }, f"checkpoints/perturbed_mlp_{dataset_name}_{location.replace(' ', '_')}.pth")
        
    # 5. Output logging trace
    os.makedirs('results', exist_ok=True)
    report_file = f"results/feature_perturbation_report_{dataset_name}.csv"
    
    with open(report_file, mode='w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['Perturbation_Location', 'Noise_Dev', 'Clean_Test_Accuracy', 'Corrupted_Test_Accuracy'])
        writer.writeheader()
        writer.writerows(results)
        
    # 6. Plot the Bar Chart for Bonus Analysis
    import matplotlib.pyplot as plt
    try:
        locations_plot = [r['Perturbation_Location'] for r in results]
        clean_accs = [float(r['Clean_Test_Accuracy']) for r in results]
        corr_accs = [float(r['Corrupted_Test_Accuracy']) for r in results]
        
        x = range(len(locations_plot))
        width = 0.35
        
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.bar([i - width/2 for i in x], clean_accs, width, label='Clean Test Acc', color='royalblue')
        ax.bar([i + width/2 for i in x], corr_accs, width, label='Corrupt Test Acc', color='indianred')
        
        ax.set_ylabel('Accuracy (%)')
        ax.set_title(f'Bonus Phase 5: Feature Perturbation Effect ({dataset_name})')
        ax.set_xticks(x)
        ax.set_xticklabels(locations_plot, rotation=45, ha='right')
        ax.legend()
        plt.tight_layout()
        plt.savefig(f"results/plots/feature_perturbation_{dataset_name}_bar.png", dpi=300)
        plt.close()
    except Exception as e:
        print(f"Plotting failed: {e}")
        
    print(f"\n✓ Bonus Phase 5 Complete! Feature Perturbation metrics logged to {report_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Phase 5.2: Layer-wise Application Analysis")
    parser.add_argument('--dataset', type=str, required=True, choices=['cifar10', 'fmnist', 'imagenet100'])
    parser.add_argument('--noise_std', type=float, default=0.2, help='Amount of internal noise injected')
    parser.add_argument('--epochs', type=int, default=15, help='Number of epochs to train')
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size for dataloaders')
    
    args = parser.parse_args()
    run_feature_perturbation_experiment(args.dataset, noise_std=args.noise_std, epochs=args.epochs, batch_size=args.batch_size)
