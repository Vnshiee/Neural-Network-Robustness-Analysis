# Deep Learning (DSE316/616) Assignment-2: Analysis of Network Robustness

## 📖 Overview
Deep learning models typically perform well on clean datasets but often struggle with out-of-distribution (OOD) shifts or corrupted images. This project evaluates different neural network architectures on standard datasets under varied image corruptions, examining how introducing perturbations into the validation set or internal network layers impacts classification accuracy, decision boundaries, and topological feature representations.

## 🚀 Architectures Evaluated
- **VGG:** Standard deep convolutional configuration.
- **ResNet:** CNN utilizing residual skip connections.
- **ConvNeXT:** Modernized CNN incorporating design cues from vision transformers.
- **Vision Transformer (ViT):** Architecture based fundamentally on global self-attention.

## 📊 Datasets
- CIFAR-10
- Fashion-MNIST (F-MNIST)
- ImageNet-100

## 🔬 Key Experiments
1. **Architectural Baselines & OOD Resilience:** Evaluated baseline architectures against OOD synthetic data using structural corruptions. 
2. **Validation Protocol Optimization:** Assessed a Multi-Layer Perceptron (MLP) under different validation scenarios (Clean vs. Gaussian Noise) to see how synthetic regularization softens over-constrained geometries.
3. **Decision Boundary & Topographical Mapping:** Utilized Inverse PCA and t-SNE projections to visualize topological structures and latent class separation.
4. **Internal Feature Perturbation:** Injected Gaussian noise directly into early, middle, and late network layers during inference to observe structural resilience.

## 📈 Key Findings
- **ResNet Memorization vs Transformer Dominance:** While ResNet achieves rapid loss reduction, its local feature maps dissolve under OOD noise. Conversely, ViTs utilize multi-headed self-attention to calculate macro relationships, exhibiting massive geometric resistance.
- **Validation Regularization:** Models trained against corrupted validation sets synthetically relax sharp decision boundaries, operating as an organic buffer against OOD spatial deformations.
- **Layer Sensitivity:** Early-stage initial matrices are critical bottlenecks; early disruption destroys downstream capabilities. However, deep layer regularization (late-stage noise) behaves similarly to pseudo-dropout, marginally protecting generalization metrics.

## 📂 Project Structure
- `models/` - Contains the implementations for baselines and MLPs.
- `results/` - Visual assets including training curves, t-SNE maps, and decision boundaries. 
- *Includes automation scripts allowing for fully reproducible experiments.*

## 📜 Full Documentation
For an in-depth dive into the methodology and mathematical findings:
- [Academic Report (PDF)](report.pdf)
- [Visual Presentation Deck](Vanshika_Visual_Presentation_Final.pptx)
