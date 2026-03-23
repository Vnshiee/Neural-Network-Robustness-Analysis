import torch
import torch.nn as nn
import os
import csv
import argparse
import glob
from tqdm import tqdm

from data_loader import get_dataloaders
from models.baseline_models import get_baseline_model
from models.mlp import get_mlp_model

def evaluate_model(model, dataloader, device):
    """
    Evaluates a model's accuracy on a given dataloader.
    """
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for inputs, labels in tqdm(dataloader, desc="Evaluating", leave=False):
            inputs, labels = inputs.to(device), labels.to(device)
            
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
    accuracy = 100. * correct / total
    return accuracy

def load_checkpoint(model, checkpoint_path):
    """
    Safely loads model weights from a `.pth` file.
    """
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}. Please train the model first.")
        
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
    model.load_state_dict(checkpoint['model_state_dict'])
    return model

def run_evaluation(dataset_name, batch_size=128):
    """
    Phase 4.1 Evaluation Script: 
    Iterates over all trained architectures for the specified dataset,
    tests them on Clean Test data AND Corrupted Test data, and logs the results.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"=== Starting Mandatory Evaluation (Phase 4.1) on {dataset_name} ===")
    
    in_channels = 1 if dataset_name == 'fmnist' else 3
    num_classes = 100 if dataset_name == 'imagenet100' else 10
    
    # Define models to evaluate (matching Phase 2 & 3 outputs)
    architectures = ['vgg', 'resnet', 'convnext', 'vit']
    mlp_experiments = ['mlp_exp_a_clean'] # Standard MLP trained/validated naturally
    
    # Prepare Dataloaders (We need Clean Test, and we'll pick ONE standard corruption for Robusness Test)
    # E.g. We test robustness against Severity 3 Gaussian Noise universally
    standard_corruption = 'gaussian_noise'
    standard_severity = 3
    
    print("\nLoading Clean Test Data...")
    _, _, clean_test_loader = get_dataloaders(
        dataset_name=dataset_name, 
        batch_size=batch_size, 
        corrupt_val=False, 
        corrupt_test=False # Ensure Clean
    )
    
    print(f"Loading Corrupted Test Data ({standard_corruption} - Sev {standard_severity})...")
    _, _, corrupted_test_loader = get_dataloaders(
        dataset_name=dataset_name, 
        batch_size=batch_size, 
        corrupt_val=False, 
        corrupt_test=True, # Ensure Corrupted
        corruption_name=standard_corruption,
        corruption_severity=standard_severity
    )
    
    results = []
    
    # -------------------------------------------------------------------------
    # 1. Evaluate Baseline CNN / ViT Models
    # -------------------------------------------------------------------------
    for arch in architectures:
        print(f"\n--- Evaluating Baseline: {arch.upper()} ---")
        model = get_baseline_model(arch, num_classes=num_classes, in_channels=in_channels)
        checkpoint_path = f"checkpoints/{arch}_{dataset_name}_clean_best.pth"
        
        try:
            model = load_checkpoint(model, checkpoint_path)
            model = model.to(device)
            
            clean_acc = evaluate_model(model, clean_test_loader, device)
            corrupt_acc = evaluate_model(model, corrupted_test_loader, device)
            
            print(f"> {arch.upper()} Clean Acc:   {clean_acc:.2f}%")
            print(f"> {arch.upper()} Corrupt Acc: {corrupt_acc:.2f}%")
            
            results.append({
                'Model': arch.upper(),
                'Training_Regime': 'Clean',
                'Clean_Test_Acc': f"{clean_acc:.2f}",
                'Corrupted_Test_Acc': f"{corrupt_acc:.2f}"
            })
            
        except FileNotFoundError as e:
            print(f"Skipping {arch.upper()}: {e}")

    # -------------------------------------------------------------------------
    # 2. Evaluate MLP Experiments
    # -------------------------------------------------------------------------
    print(f"\n--- Evaluating MLP (Exp A: Clean Validation) ---")
    mlp_model = get_mlp_model(dataset_name)
    mlp_clean_cp = f"checkpoints/mlp_{dataset_name}_exp_a_clean.pth"
    
    try:
        mlp_model = load_checkpoint(mlp_model, mlp_clean_cp)
        mlp_model = mlp_model.to(device)
        
        clean_acc = evaluate_model(mlp_model, clean_test_loader, device)
        corrupt_acc = evaluate_model(mlp_model, corrupted_test_loader, device)
        
        print(f"> MLP (Exp A) Clean Acc:   {clean_acc:.2f}%")
        print(f"> MLP (Exp A) Corrupt Acc: {corrupt_acc:.2f}%")
        
        results.append({
            'Model': 'MLP',
            'Training_Regime': 'Clean Val (Exp A)',
            'Clean_Test_Acc': f"{clean_acc:.2f}",
            'Corrupted_Test_Acc': f"{corrupt_acc:.2f}"
        })
    except FileNotFoundError as e:
        print(f"Skipping MLP Exp A: {e}")

    # -------------------------------------------------------------------------
    # 2.5 Evaluate Robust MLP (Exp B: Corrupt Validation)
    # -------------------------------------------------------------------------
    print(f"\n--- Evaluating MLP (Exp B: Corrupted Validation) ---")
    # Finding the experiment B checkpoint dynamically since it contains variable severity/names
    exp_b_files = glob.glob(f"checkpoints/mlp_{dataset_name}_exp_b_corrupt_*.pth")
    
    if exp_b_files:
        # Assuming we just test the most recently found one, or ideally the user's best
        mlp_corrupt_cp = exp_b_files[0]
        print(f"Found robust checkpoint: {os.path.basename(mlp_corrupt_cp)}")
        
        mlp_robust_model = get_mlp_model(dataset_name)
        
        try:
            mlp_robust_model = load_checkpoint(mlp_robust_model, mlp_corrupt_cp)
            mlp_robust_model = mlp_robust_model.to(device)
            
            clean_acc = evaluate_model(mlp_robust_model, clean_test_loader, device)
            corrupt_acc = evaluate_model(mlp_robust_model, corrupted_test_loader, device)
            
            print(f"> MLP (Exp B) Clean Acc:   {clean_acc:.2f}%")
            print(f"> MLP (Exp B) Corrupt Acc: {corrupt_acc:.2f}%")
            
            results.append({
                'Model': 'MLP',
                'Training_Regime': f'Corrupt Val (Exp B: {os.path.basename(mlp_corrupt_cp)})',
                'Clean_Test_Acc': f"{clean_acc:.2f}",
                'Corrupted_Test_Acc': f"{corrupt_acc:.2f}"
            })
        except Exception as e:
            print(f"Failed to evaluate robust MLP: {e}")
    else:
        print(f"Skipping MLP Exp B: No checkpoints found matching `mlp_{dataset_name}_exp_b_corrupt_*.pth`.")
        
    # -------------------------------------------------------------------------
    # 2.6 Evaluate Optimized MLP (Exp C: Multi-Corrupt Validation)
    # -------------------------------------------------------------------------
    print(f"\n--- Evaluating MLP (Exp C: Optimized Multi-Corrupt Validation) ---")
    exp_c_files = glob.glob(f"checkpoints/mlp_{dataset_name}_exp_c_optimal_corrupt_*.pth")
    
    for exp_c_cp in exp_c_files:
        try:
            mlp_model = get_mlp_model(dataset_name)
            mlp_model = load_checkpoint(mlp_model, exp_c_cp)
            mlp_model = mlp_model.to(device)
            
            clean_acc = evaluate_model(mlp_model, clean_test_loader, device)
            corrupt_acc = evaluate_model(mlp_model, corrupted_test_loader, device)
            
            print(f"> MLP (Exp C: {os.path.basename(exp_c_cp)}) Clean Acc:   {clean_acc:.2f}%")
            print(f"> MLP (Exp C: {os.path.basename(exp_c_cp)}) Corrupt Acc: {corrupt_acc:.2f}%")
            
            results.append({
                'Model': 'MLP',
                'Training_Regime': f"Multi-Corrupt Val (Exp C: {os.path.basename(exp_c_cp)})",
                'Clean_Test_Acc': f"{clean_acc:.2f}",
                'Corrupted_Test_Acc': f"{corrupt_acc:.2f}"
            })
        except Exception as e:
            print(f"Error evaluating {exp_c_cp}: {e}")

    # -------------------------------------------------------------------------
    # 3. Export Final Report
    # -------------------------------------------------------------------------
    os.makedirs('results', exist_ok=True)
    report_file = f"results/evaluation_report_{dataset_name}.csv"
    
    if results:
        with open(report_file, mode='w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=['Model', 'Training_Regime', 'Clean_Test_Acc', 'Corrupted_Test_Acc'])
            writer.writeheader()
            writer.writerows(results)
            
        print(f"\n✓ Evaluation complete. Report generated at {report_file}")
    else:
        print("\n⚠ No models were successfully evaluated. Are the checkpoints trained?")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Phase 4.1: Mandatory Evaluation")
    parser.add_argument('--dataset', type=str, required=True, choices=['cifar10', 'fmnist', 'imagenet100'], help='Dataset to evaluate')
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size for evaluation')
    
    args = parser.parse_args()
    run_evaluation(args.dataset, args.batch_size)