# Analysis of Network Robustness, Validation Optimization, and Feature Perturbation

This project explores the robustness of various deep learning network architectures—including Convolutional Neural Networks (CNNs) and Vision Transformers (ViTs)—under explicit out-of-distribution (OOD) dataset shifts and image corruptions. 

It evaluates how models handle synthetic corruptions (like Gaussian Noise, Blur, etc.), tests how validation setup impacts real-world robustness, and analyzes feature perturbation directly inside network layers.

## 📊 Key Findings & Results
- **ResNet Memorization vs. Overfitting:** Residual networks reduce training loss very rapidly across CIFAR-10, F-MNIST, and ImageNet-100. However, they consistently exhibited sharp overfitting on the validation set compared to modernization approaches like ConvNeXT.
- **ViT Stability:** Vision Transformers (ViT) demonstrated remarkable training stability. Relying on global self-attention rather than local convolutions resulted in smoother learning curves without the sharp divergence symptomatic of standard CNN overfitting over early epochs.
- **Validation Optimization (MLP):** Sweeping over random corruptions showed that standard models typically peak early when exposed to isolated corruptions, heavily decaying when generalizing to compound or structural noise.

## 📂 Project Structure

```text
├── data_loader.py            # Custom Loaders for CIFAR10, F-MNIST, ImageNet-100 (80/20 structured split)
├── train_baseline.py         # Standard network training scripts (VGG, ResNet, ConvNeXT, ViT)
├── evaluate_models.py        # Validates model performance under OOD synthetic image corruptions
├── experiment_a.py           # MLP validated against a standard clean distribution
├── experiment_b.py           # MLP validated against focused Gaussian Noise distribution
├── experiment_feature_perturb.py # Feature mapping robustness via intra-layer noise injection
├── visualize_boundaries.py   # Decision boundary visualization using Inverse PCA
├── visualize_tsne.py         # Topological feature representation maps using t-SNE
├── main.ipynb                # End-to-end interactive exploration notebook
├── run_all.sh                # Automation script 
├── requirements.txt          # Python dependencies
└── project_plan.md           # Underlying architectural methodology and phase planning
```

## 🚀 Setup Instructions

1. **Clone the Repository:**
   ```bash
   git clone https://github.com/Vnshiee/Neural-Network-Robustness-Analysis.git
   cd Neural-Network-Robustness-Analysis
   ```

2. **Install Dependencies:**
   It is recommended to run this within a virtual environment.
   ```bash
   pip install -r requirements.txt
   ```

3. **Data Setup:**
   - Run the local setup or scripts available in `hf_local_downloader.py / download_imagenet.py`.
   - Datasets will be downloaded dynamically or placed directly in a `./data` directory (ignored by git to save space).

## 🏃‍♂️ How to Run

You can run individual experiment loops or the baseline training script:
```bash
python train_baseline.py
```

To run the entire suite (designed for GPU acceleration like A100):
```bash
chmod +x run_all.sh
./run_all.sh
```

For exploratory data analysis and visual results, open `main.ipynb` in Jupyter Notebook/Lab.

## 📈 Visualizations
- View model checkpoints inside the localized `checkpoints/` directory.
- Plots mapped against training iterations are output directly to `results/plots/`.

*(Note: Data, checkpoints, and plotting artifacts are excluded from this repository via `.gitignore` to maintain a lightweight footprint.)*
