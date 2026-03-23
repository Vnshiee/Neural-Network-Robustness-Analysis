# Deep Learning Project Plan: Robustness and Corruption Analysis

This document outlines a systematic plan to implement the deep learning robustness and corruption analysis project. The project is divided into five distinct phases, covering data preparation, standard baseline training, MLP experimentation, evaluation, and advanced feature map perturbation.

## Phase 1: Data Preparation and Splitting

**Goal:** diverse datasets (CIFAR-10, F-MNIST, ImageNet-100) and prepare robust data loaders.

### 1.1 Dataset Implementation
-   **Datasets:**
    -   **CIFAR-10:** Use `torchvision.datasets.CIFAR10`.
    -   **Fashion-MNIST (F-MNIST):** Use `torchvision.datasets.FashionMNIST`.
    -   **ImageNet-100:** This is a subset of ImageNet. You will need a custom `Dataset` class or script to load this specific subset (assumes data is organized in folders).
-   **Action:** Create a `data_loader.py` module.

### 1.2 Data Splitting Strategy
-   **Validation Split:** Reserve exactly **20%** of the training data for validation.
-   **Consistency:** Use a fixed random seed (e.g., `torch.manual_seed(42)`) or save the indices to a file to ensure the *exact same* 80/20 split is used across all experiments (VGG, ResNet, MLP, etc.).
-   **Output:** `get_data_loaders(dataset_name, batch_size, split_ratio=0.2)` function returning `train_loader`, `val_loader`, and `test_loader`.

### 1.3 Corruption Integration
-   **Library:** Integrate the `imagecorruptions` or `robustness` library.
-   **Implementation:**
    -   Create a wrapper class or transform `CorruptTransform(severity, noise_type)` that applies perturbations.
    -   Ensure this can be dynamically turned on/off for the validation set.

---

## Phase 2: Standard Architecture Baselines

**Goal:** Establish performance benchmarks on standard architectures.

### 2.1 Model Definitions
-   **Architectures:**
    -   **VGG:** (e.g., VGG16 or VGG19 with BatchNorm)
    -   **ResNet:** (e.g., ResNet18 or ResNet50)
    -   **ConvNeXT:** (e.g., ConvNeXT-Tiny)
    -   **ViT:** (Vision Transformer, e.g., ViT-B/16)
-   **Implementation:** Use `torchvision.models` or `timm` (PyTorch Image Models) for robust, pre-defined architectures. Modify the head (final fully connected layer) to match the number of classes (10 for CIFAR/F-MNIST, 100 for ImageNet-100).

### 2.2 Training Baseline
-   **Script:** `train_baseline.py`
-   **Process:**
    -   Train each model on all 3 datasets.
    -   Use the **Clean** training set.
    -   Evaluate on the **Clean** validation set.
    -   Save model checkpoints (`state_dict`) for later analysis.

---

## Phase 3: The MLP Validation Experiments

**Goal:** Investigate how validation set corruption affects model selection and robustness.

### 3.1 MLP Architecture
-   **Model:** Design a standard Multi-Layer Perceptron (MLP) with configurable hidden layers and dropout.
-   **File:** `models/mlp.py`

### 3.2 Experiment A: Clean Validation
-   **Training:** Train MLP on Clean Train set.
-   **Validation:** Evaluate on **Clean Validation set**.
-   **Outcome:** Baseline accuracy for the MLP.

### 3.3 Experiment B: Corrupted Validation
-   **Training:** Train MLP on Clean Train set (same as Exp A).
-   **Validation:** Evaluate on **Corrupted Validation set**.
-   **Hypothesis:** Does validating on corrupted data lead to selecting a more robust model checkpoint?

### 3.4 Experiment C: Optimization of Corruption
-   **Script:** `optimize_corruption.py`
-   **Variables:**
    -   **Severity:** Test specifically fixed at **Level 2** vs. variable levels (1-5).
    -   **Corruption Type:** Iterate through all available corruptions (e.g., Gaussian Noise, Blur, Weather, etc. - typically 15 types).
-   **Toggle:** Implement a configuration flag (e.g., `--fix_severity_2`) to easily switch strategies.

---

## Phase 4: Mandatory Evaluation and Visualization

**Goal:** comprehensive reporting and visual analysis of the trained models.

### 4.1 Evaluation Script
-   **Script:** `evaluate_models.py`
-   **Metrics:**
    -   **Clean Test Accuracy:** Standard performance.
    -   **Corrupted Test Accuracy:** Robustness metric (average accuracy across different corruption types/severities).
-   **Report:** Generate a CSV or table comparing (Clean Val trained) vs (Corrupted Val trained) models across all architectures and datasets.

### 4.2 Decision Boundary Visualization
-   **Tool:** Dimensionality reduction (PCA/t-SNE) on inputs to 2D, or use low-dimensional toy datasets if applicable. Alternatively, fix all but 2 input dimensions (unlikely for images) or plot prediction landscape around data points.
-   **Action:** Plot how the decision boundary shifts between the model trained with Clean Validation vs. Corrupted Validation.

### 4.3 Feature Map Analysis (t-SNE)
-   **Extraction:** Hook into the penultimate layer (feature extractor output) of the models.
-   **Visualization:** Run t-SNE on these features for the Test Set.
-   **Analysis:** Visualize class separation. Compare clusters between clean and robust models.

---

## Phase 5: Bonus Objective (Feature Map Perturbation)

**Goal:** Analyze internal layer robustness by perturbing features instead of inputs.

### 5.1 Interior Feature Perturbation
-   **Method:** Implement "Feature Noise" injection.
-   **Wrapper:** Create a `PerturbedFeatureModel` class that wraps a base model.
-   **Mechanism:** During forward pass, add noise to the output of specific layers.

### 5.2 Layer-wise Experiments
-   **Locations:**
    -   **Early Layers:** (e.g., after first conv block).
    -   **Middle Layers:** (e.g., middle of ResNet blocks).
    -   **Late Layers:** (e.g., before the final classifier).
    -   **Combined:** Perturb multiple stages.
-   **Experiment:** Train or Fine-tune models with this internal noise.
-   **Logging:** Record accuracy drop-off or robustness gain for each perturbation location to identify vulnerable network sections.

---

## Implementation Roadmap

1.  **Environment Setup:** Install `torch`, `torchvision`, `timm`, `imagecorruptions`, `scikit-learn`, `matplotlib`.
2.  **Code Skeleton:** Create folder structure (`data/`, `models/`, `utils/`, `experiments/`).
3.  **Phase 1-2 Execution:** Run baseline training for standard models.
4.  **Phase 3 Execution:** Run MLP Clean vs Corrupted Val experiments.
5.  **Phase 4 Execution:** Run evaluation metrics and generate plots.
6.  **Phase 5 Execution:** Run feature perturbation study.
7.  **Final Report:** Compile all plots and metrics into the final submission.