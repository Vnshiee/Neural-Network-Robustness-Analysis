import argparse
import os
import csv
from experiment_b import run_experiment_b

# Standard 15 corruptions from imagecorruptions library
CORRUPTION_TYPES = [
    'gaussian_noise', 'shot_noise', 'impulse_noise', 'defocus_blur',
    'glass_blur', 'motion_blur', 'zoom_blur', 'snow', 'frost', 'fog',
    'brightness', 'contrast', 'elastic_transform', 'pixelate', 'jpeg_compression'
]

def optimize_corruption(dataset_name, fix_severity_2=False):
    """
    Phase 3.4: Experiment C (Optimization of Corruption)
    Runs a grid search to test if the validation set should be perturbed using 
    all available corruptions (e.g., 15 types) across different severities.
    """
    print(f"=== Starting Corruption Optimization on {dataset_name} ===")
    print(f"Fixed Severity 2 Mode: {fix_severity_2}")
    
    results = []
    
    # Define severity range based on the toggle flag
    severities = [2] if fix_severity_2 else [1, 2, 3, 4, 5]
    
    # Create an output directory for optimization logs
    os.makedirs('results', exist_ok=True)
    log_file = f"results/mlp_optimization_{dataset_name}_{'fixed_sev2' if fix_severity_2 else 'all_sev'}.csv"
    
    # Fast-fail configuration for grid-search (lower epochs to find general trends)
    search_epochs = 5 
    
    with open(log_file, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Corruption_Type', 'Severity', 'Best_Validation_Accuracy'])
        
        # Iterate over all 15 corruption types
        for corruption in CORRUPTION_TYPES:
            for severity in severities:
                print(f"\n---> Testing: {corruption} at Severity {severity} <---")
                
                # We reuse the core logic of Experiment B, but loop over all parameters
                # We reduce epochs for the optimization phase to save compute time
                best_acc = run_experiment_b(
                    dataset_name=dataset_name,
                    corruption_name=corruption,
                    corruption_severity=severity,
                    epochs=search_epochs, 
                    batch_size=128,
                    save_checkpoint=False  # Crucial to avoid 22GB+ disk memory explosion
                )
                
                results.append((corruption, severity, best_acc))
                writer.writerow([corruption, severity, f"{best_acc:.2f}"])
                # Flush to ensure data is saved immediately if interrupted
                f.flush()
                
    print("\n=== Optimization Complete ===")
    print(f"Results saved to {log_file}")
    
    # Find the corruption setting that yielded the highest robust validation accuracy
    best_config = max(results, key=lambda x: x[2])
    print(f"\n🏆 Best Validated Topology: {best_config[0]} at Severity {best_config[1]} (Acc: {best_config[2]:.2f}%)")
    
    return best_config

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Phase 3.4: Optimize Validation Corruption")
    parser.add_argument('--dataset', type=str, required=True, choices=['cifar10', 'fmnist', 'imagenet100'], help='Dataset to optimize on')
    parser.add_argument('--fix_severity_2', action='store_true', help='Toggle to strictly freeze severity at Level 2')
    
    args = parser.parse_args()
    optimize_corruption(args.dataset, fix_severity_2=args.fix_severity_2)
