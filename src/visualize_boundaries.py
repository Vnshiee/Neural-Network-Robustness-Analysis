import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import argparse
import os
from tqdm import tqdm

from data_loader import get_dataloaders
from models.mlp import get_mlp_model
from evaluate_models import load_checkpoint

def extract_features_and_predictions(model, dataloader, device, num_samples=1000):
    """
    Extracts raw inputs, model predictions, and true labels for a subset of the dataset.
    """
    model.eval()
    all_inputs = []
    all_preds = []
    all_labels = []
    
    samples_collected = 0
    
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            
            # Flatten inputs from (B, C, H, W) to (B, C*H*W) for PCA
            flat_inputs = inputs.view(inputs.size(0), -1)
            
            all_inputs.append(flat_inputs.cpu().numpy())
            all_preds.append(preds.cpu().numpy())
            all_labels.append(labels.cpu().numpy())
            
            samples_collected += inputs.size(0)
            if samples_collected >= num_samples:
                break
                
    return (
        np.concatenate(all_inputs)[:num_samples],
        np.concatenate(all_preds)[:num_samples],
        np.concatenate(all_labels)[:num_samples]
    )

def plot_decision_boundary(model_clean, model_corrupt, dataloader, dataset_name, device):
    """
    Phase 4.2: Decision Boundary Visualization
    Projects high-dimensional images down to 2D using PCA.
    Visualizes how the decision space shifts between a model trained on Clean data 
    vs one validated/selected on Corrupted data.
    """
    print(f"=== Generating Decision Boundary Visualizations for {dataset_name} ===")
    
    # 1. Extract a meaningful sample of data
    # We use the clean model to extract the raw input feature space
    print("Extracting features and generating predictions...")
    inputs_2d, preds_clean, true_labels = extract_features_and_predictions(model_clean, dataloader, device)
    
    # Generate predictions from the robust model on the EXACT same subset
    _, preds_corrupt, _ = extract_features_and_predictions(model_corrupt, dataloader, device)
    
    # 2. Dimensionality Reduction (PCA)
    # We cannot plot 150,000 dimensions. We use PCA to find the 2 most strictly variant spatial axes.
    print("Running PCA to reduce dimensionality to 2D...")
    pca = PCA(n_components=2)
    inputs_2d_pca = pca.fit_transform(inputs_2d)
    
    # Mathematical True Decision Boundary Generation via Meshgrid Inverse Transform
    print("Generating mathematical prediction meshgrid...")
    x_min, x_max = inputs_2d_pca[:, 0].min() - 2, inputs_2d_pca[:, 0].max() + 2
    y_min, y_max = inputs_2d_pca[:, 1].min() - 2, inputs_2d_pca[:, 1].max() + 2
    
    # FIX 2: Dynamically calculate step size to guarantee a safe 150x150 grid resolution
    grid_resolution = 150
    h_x = (x_max - x_min) / grid_resolution
    h_y = (y_max - y_min) / grid_resolution
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h_x), np.arange(y_min, y_max, h_y))
    
    # Flatten the 2D grid back into the PCA feature space (N_points, 2)
    mesh_pca_space = np.c_[xx.ravel(), yy.ravel()]
    
    def predict_mesh_in_batches(model_to_test, mesh_pca, pca_model, batch_size=500):
        """Processes the Meshgrid cleanly in batches to avoid 2TB+ RAM/VRAM crashes."""
        model_to_test.eval()
        all_mesh_preds = []
        with torch.no_grad():
            for i in tqdm(range(0, len(mesh_pca), batch_size), desc="Mapping Boundaries", leave=False):
                batch_pca = mesh_pca[i:i+batch_size]
                
                # FIX 1: Inverse project ONLY the small batch back to 150,528 dimensions
                batch_original = pca_model.inverse_transform(batch_pca)
                
                batch_tensor = torch.tensor(batch_original, dtype=torch.float32).to(device)
                
                # Direct to inner network bypassing flatten
                outputs = model_to_test.network(batch_tensor)  
                _, preds = torch.max(outputs, 1)
                all_mesh_preds.extend(preds.cpu().numpy())
        return np.array(all_mesh_preds)

    print("Mapping True Class Boundaries across planar geometry...")
    Z_clean = predict_mesh_in_batches(model_clean, mesh_pca_space, pca)
    Z_clean = Z_clean.reshape(xx.shape)
    
    Z_corrupt = predict_mesh_in_batches(model_corrupt, mesh_pca_space, pca)
    Z_corrupt = Z_corrupt.reshape(xx.shape)

    # 3. Setup the Matplotlib Plot
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    fig.suptitle(f"Mathematical Decision Boundary Shift (Inverse PCA Projection) - {dataset_name.upper()}", fontsize=16)
    
    # Common plotting logic
    def train_plot(ax, title, predictions, Z_mesh):
        # Plot the True Geographic Boundary
        contour = ax.contourf(xx, yy, Z_mesh, cmap='tab10', alpha=0.3)
        # Scatterplot true classes underlying the data
        scatter = ax.scatter(inputs_2d_pca[:, 0], inputs_2d_pca[:, 1], 
                             c=true_labels, cmap='tab10', alpha=0.9, s=25, edgecolors='k')

        ax.set_title(title)
        ax.set_xlabel("Principal Component 1")
        ax.set_ylabel("Principal Component 2")
        fig.colorbar(scatter, ax=ax, fraction=0.046, pad=0.04)

    # 4. Render Plots
    # Plot 1: Model trained mapped against purely clean validation bounds
    train_plot(axes[0], "Model A (Clean Validation Boundary)", preds_clean, Z_clean)
    
    # Plot 2: Model trained mapped against robust corrupted bounds
    train_plot(axes[1], "Model B (Corrupted Validation Boundary)", preds_corrupt, Z_corrupt)
    
    # Calculate difference
    diff_mask = preds_clean != preds_corrupt
    diff_percentage = (diff_mask.sum() / len(preds_clean)) * 100
    print(f"\nAnalysis: The robust model shifted its decision boundaries and changed predictions on {diff_percentage:.2f}% of the local manifold.")
    
    # 5. Save Artifacts
    os.makedirs('results/plots', exist_ok=True)
    save_path = f"results/plots/decision_boundary_{dataset_name}.png"
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    print(f"✓ Saved plot to {save_path}")

def execute_visualization(dataset_name, corrupt_model_file=None):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load dataset
    _, _, test_loader = get_dataloaders(dataset_name=dataset_name, batch_size=128, corrupt_test=False)
    
    # Load Models (We use the MLP architectures to examine Phase 3's hypothesis natively)
    model_clean = get_mlp_model(dataset_name)
    model_corrupt = get_mlp_model(dataset_name)
    
    try:
        model_clean = load_checkpoint(model_clean, f"checkpoints/mlp_{dataset_name}_exp_a_clean.pth").to(device)
        
        # Determine the best corrupted version from the optimization grid
        if corrupt_model_file is None:
            # Simple automatic fallback if no specific path provided (assumes gaussian noise sev 2 exists)
            corrupt_cp_path = f"checkpoints/mlp_{dataset_name}_exp_b_corrupt_gaussian_noise_sev2.pth" 
        else:
            corrupt_cp_path = corrupt_model_file
            
        model_corrupt = load_checkpoint(model_corrupt, corrupt_cp_path).to(device)
        
        plot_decision_boundary(model_clean, model_corrupt, test_loader, dataset_name, device)
        
    except FileNotFoundError as e:
        print(f"Error executing visualization. Ensure you have run Phase 3 scripts (experiment_a and experiment_b) first. \nMissing file: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Phase 4.2: Decision Boundary Visualization")
    parser.add_argument('--dataset', type=str, required=True, choices=['cifar10', 'fmnist', 'imagenet100'])
    parser.add_argument('--robust_checkpoint', type=str, default=None, help="Specific path to a robust checkpoint to diff against")
    args = parser.parse_args()
    
    execute_visualization(args.dataset, args.robust_checkpoint)
