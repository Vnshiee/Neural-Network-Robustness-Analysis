import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import argparse
import os
from tqdm import tqdm

from data_loader import get_dataloaders
from models.mlp import get_mlp_model
from evaluate_models import load_checkpoint

class FeatureExtractor(nn.Module):
    """
    Phase 4.3: Feature Extraction Module.
    Hooks into the penultimate layer of the MLP to extract intermediate feature maps 
    right before the final classification head determines the logits.
    """
    def __init__(self, original_mlp):
        super(FeatureExtractor, self).__init__()
        # Copy the original flattener and network
        self.flatten = original_mlp.flatten
        
        # Strip off the final classification nn.Linear layer.
        # Assuming the standard Phase 3 MLP structure: [Linear, BatchNorm, ReLU, Dropout, Linear(head)]
        # We index [:-1] to keep everything EXCEPT the final layer
        self.feature_layers = nn.Sequential(*list(original_mlp.network.children())[:-1])
        
    def forward(self, x):
        x = self.flatten(x)
        features = self.feature_layers(x)
        return features

def extract_penultimate_features(model, dataloader, device, num_samples=1000):
    """
    Pulls the high-dimensional internal representation space embeddings 
    for plotting class clustering behavior.
    """
    # Wrap model purely in feature extractor
    extractor = FeatureExtractor(model)
    extractor = extractor.to(device)
    extractor.eval()
    
    all_features = []
    all_labels = []
    
    samples_collected = 0
    
    with torch.no_grad():
        for inputs, labels in tqdm(dataloader, desc="Extracting Feature Maps", leave=False):
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            # Extract internal representation (e.g. 128-dimensional embedding)
            features = extractor(inputs)
            
            all_features.append(features.cpu().numpy())
            all_labels.append(labels.cpu().numpy())
            
            samples_collected += inputs.size(0)
            if samples_collected >= num_samples:
                break
                
    return (
        np.concatenate(all_features)[:num_samples],
        np.concatenate(all_labels)[:num_samples]
    )

def plot_tsne_clusters(model_clean, model_corrupt, dataloader, dataset_name, device):
    """
    Phase 4.3: t-SNE Plot Generation
    Runs exactly the same underlying data through two distinct model weights, 
    pulls their penultimate feature activations, and projects them to 2D via t-SNE 
    to visualize how strongly class separation was learned.
    """
    print(f"\n=== Generating t-SNE Representations for {dataset_name} ===")
    
    # 1. Extract internal representations (embeddings natively learned by the network)
    print("-> Siphoning features from Clean Model...")
    features_clean, labels_clean = extract_penultimate_features(model_clean, dataloader, device)
    
    print("-> Siphoning features from Robust Model...")
    features_corrupt, _ = extract_penultimate_features(model_corrupt, dataloader, device)

    # 2. Setup standard t-SNE projector (perplexity=30 is mathematically considered optimal for images)
    print("\n-> Running computationally heavy t-SNE decomposition (this may take a minute)...")
    tsne = TSNE(n_components=2, perplexity=30.0, random_state=42)
    
    # Fit the mathematical t-SNE embedding transforms for BOTH representations
    embeddings_clean = tsne.fit_transform(features_clean)
    print("      Clean t-SNE projection calculated.")
    
    embeddings_corrupt = tsne.fit_transform(features_corrupt)
    print("      Robust t-SNE projection calculated.")

    # 3. Setup Matplotlib Plot
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    fig.suptitle(f"t-SNE Feature Map Separation - Penultimate Layer ({dataset_name.upper()})", fontsize=16)

    def draw_tsne_axes(ax, title, embeddings, unique_labels):
        scatter = ax.scatter(embeddings[:, 0], embeddings[:, 1], 
                             c=unique_labels, cmap='tab10', alpha=0.7, s=20)
        ax.set_title(title)
        ax.set_xticks([]) # t-SNE axes are mathematically arbitrary, drop ticks for clean looks
        ax.set_yticks([])
        fig.colorbar(scatter, ax=ax, fraction=0.046, pad=0.04, label="Class Labels")

    # 4. Render
    draw_tsne_axes(axes[0], "Trained with Clean Validation", embeddings_clean, labels_clean)
    draw_tsne_axes(axes[1], "Trained with Corrupted Validation", embeddings_corrupt, labels_clean)

    # 5. Export
    os.makedirs('results/plots', exist_ok=True)
    save_path = f"results/plots/tsne_features_{dataset_name}.png"
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    print(f"\n✓ Saved t-SNE cluster map to {save_path}")


def execute_tsne(dataset_name, corrupt_model_file=None):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Get standard Test Data
    _, _, test_loader = get_dataloaders(dataset_name=dataset_name, batch_size=128, corrupt_test=False)
    
    # Instantiate raw architectures
    model_clean = get_mlp_model(dataset_name)
    model_corrupt = get_mlp_model(dataset_name)
    
    try:
        # Rehydrate weights 
        model_clean = load_checkpoint(model_clean, f"checkpoints/mlp_{dataset_name}_exp_a_clean.pth").to(device)
        
        if corrupt_model_file is None:
            corrupt_cp_path = f"checkpoints/mlp_{dataset_name}_exp_b_corrupt_gaussian_noise_sev2.pth" 
        else:
            corrupt_cp_path = corrupt_model_file
            
        model_corrupt = load_checkpoint(model_corrupt, corrupt_cp_path).to(device)
        
        # Fire visualization logic
        plot_tsne_clusters(model_clean, model_corrupt, test_loader, dataset_name, device)
        
    except FileNotFoundError as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Phase 4.3: t-SNE Feature Extractor")
    parser.add_argument('--dataset', type=str, required=True, choices=['cifar10', 'fmnist', 'imagenet100'])
    parser.add_argument('--robust_checkpoint', type=str, default=None, help="Specific path to a robust checkpoint")
    args = parser.parse_args()
    
    execute_tsne(args.dataset, args.robust_checkpoint)
