from train_baseline import train_baseline
from experiment_a import run_experiment_a
from experiment_b import run_experiment_b
from optimize_corruption import optimize_corruption
from evaluate_models import run_evaluation
from visualize_boundaries import execute_visualization
from visualize_tsne import execute_tsne
from experiment_feature_perturb import run_feature_perturbation_experiment
import gc
import torch

class AssignmentRunner:
    """
    A unified wrapper class designed for Jupyter/Colab environments.
    It encapsulates the environment state (dataset name, batch size, etc.) 
    and exposes the structural functions natively as object methods, preventing 
    the need to pass redundant arguments repeatedly in cells.
    """
    def __init__(self, dataset_name='cifar10', batch_size=128, base_epochs=15):
        self.dataset_name = dataset_name.lower()
        self.batch_size = batch_size
        self.base_epochs = base_epochs

    def _cleanup_memory(self):
        """Forces CUDA/CPU memory collection to prevent Notebook crash between cells."""
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()

    # --- Phase 2: Baselines ---
    def train_baseline(self, model_name='vgg', lr=0.001):
        print(f"\n[Running Phase 2 | {model_name.upper()} Baseline]")
        try:
            train_baseline(model_name, self.dataset_name, epochs=self.base_epochs, batch_size=self.batch_size, lr=lr)
        finally:
            self._cleanup_memory()

    # --- Phase 3: MLP Experiments ---
    def run_mlp_experiment_a(self, lr=0.001):
        print("\n[Running Phase 3.2 | MLP Clean Validation]")
        try:
            run_experiment_a(self.dataset_name, epochs=self.base_epochs, batch_size=self.batch_size, lr=lr)
        finally:
            self._cleanup_memory()

    def run_mlp_experiment_b(self, corruption='gaussian_noise', severity=2, lr=0.001):
        print(f"\n[Running Phase 3.3 | MLP Corrupted Validation -> {corruption} Sev {severity}]")
        try:
            run_experiment_b(
                self.dataset_name, 
                corruption_name=corruption, 
                corruption_severity=severity, 
                epochs=self.base_epochs, 
                batch_size=self.batch_size, 
                lr=lr
            )
        finally:
            self._cleanup_memory()

    def optimize_corruptions(self, fix_severity_2=True):
        print("\n[Running Phase 3.4 | Optimizing Robust MLP Topologies]")
        try:
            best_config = optimize_corruption(self.dataset_name, fix_severity_2=fix_severity_2)
            return best_config
        finally:
            self._cleanup_memory()

    # --- Phase 4: Evaluations & Visualizations ---
    def evaluate_all(self):
        print("\n[Running Phase 4.1 | Generating Master Evaluation Readout]")
        try:
            run_evaluation(self.dataset_name, batch_size=self.batch_size)
        finally:
            self._cleanup_memory()

    def map_decision_boundaries(self, robust_checkpoint_path=None):
        print("\n[Running Phase 4.2 | PCA Inverse Meshgrid Projections]")
        try:
            execute_visualization(self.dataset_name, robust_checkpoint_path)
        finally:
            self._cleanup_memory()

    def map_feature_tsne(self, robust_checkpoint_path=None):
        print("\n[Running Phase 4.3 | t-SNE Clustering]")
        try:
            execute_tsne(self.dataset_name, robust_checkpoint_path)
        finally:
            self._cleanup_memory()

    # --- Phase 5: Layer Geometry Perturbations ---
    def run_feature_perturbation(self, noise_std=0.2):
        print("\n[Running Phase 5 | Internal Layer Perturbation Sweeps]")
        try:
            run_feature_perturbation_experiment(
                self.dataset_name,
                noise_std=noise_std, 
                epochs=self.base_epochs, 
                batch_size=self.batch_size
            )
        finally:
            self._cleanup_memory()
