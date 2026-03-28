#!/bin/bash
# Master execution script for DL Assignment 2

# removed 'set -e' so if one dataset fails (e.g. missing imagenet data), 
# the script will gracefully continue and finish the other datasets overnight!

# Default parameters (We can safely bump batch size back up on an A100 80GB)
BATCH_SIZE=512
EPOCHS=7
LR=0.001

# Loop over all three datasets required for the report!
DATASETS=("cifar10" "fmnist" "imagenet100")

for DATASET in "${DATASETS[@]}"; do

    echo "========================================================="
    echo "  Starting Deep Learning Pipeline for Dataset: ${DATASET}"
    echo "========================================================="

    # ---------------------------------------------------------
    # Phase 2: Baselines
    # ---------------------------------------------------------
    echo ">>> Phase 2: Training VGG Baseline (${DATASET})"
    CUDA_VISIBLE_DEVICES=1 python train_baseline.py --model vgg --dataset ${DATASET} --epochs ${EPOCHS} --batch_size ${BATCH_SIZE} --lr ${LR}

    echo ">>> Phase 2: Training ResNet Baseline (${DATASET})"
    CUDA_VISIBLE_DEVICES=1 python train_baseline.py --model resnet --dataset ${DATASET} --epochs ${EPOCHS} --batch_size ${BATCH_SIZE} --lr ${LR}

    echo ">>> Phase 2: Training ConvNeXT Baseline (${DATASET})"
    CUDA_VISIBLE_DEVICES=1 python train_baseline.py --model convnext --dataset ${DATASET} --epochs ${EPOCHS} --batch_size ${BATCH_SIZE} --lr ${LR}

    echo ">>> Phase 2: Training ViT Baseline (${DATASET})"
    CUDA_VISIBLE_DEVICES=1 python train_baseline.py --model vit --dataset ${DATASET} --epochs ${EPOCHS} --batch_size ${BATCH_SIZE} --lr ${LR}

    # ---------------------------------------------------------
    # Phase 3: MLP Experiments
    # ---------------------------------------------------------
    echo ">>> Phase 3A: Training MLP Clean Validation (${DATASET})"
    CUDA_VISIBLE_DEVICES=1 python experiment_a.py --dataset ${DATASET} --epochs ${EPOCHS} --batch_size ${BATCH_SIZE} --lr ${LR}

    echo ">>> Phase 3B: Training MLP Corrupt Validation (${DATASET})"
    CUDA_VISIBLE_DEVICES=1 python experiment_b.py --dataset ${DATASET} --corruption_name gaussian_noise --corruption_severity 2 --epochs ${EPOCHS} --batch_size ${BATCH_SIZE} --lr ${LR}

    echo ">>> Phase 3C: Optimizing Validation Corruptions (${DATASET})"
    CUDA_VISIBLE_DEVICES=1 python optimize_corruptions.py --dataset ${DATASET} --epochs ${EPOCHS} --batch_size ${BATCH_SIZE} --lr ${LR}

    # ---------------------------------------------------------
    # Phase 4: Evaluations & Visualization
    # ---------------------------------------------------------
    echo ">>> Phase 4.1: Generating Global Evaluation Sheet (${DATASET})"
    CUDA_VISIBLE_DEVICES=1 python evaluate_models.py --dataset ${DATASET} --batch_size ${BATCH_SIZE}

    echo ">>> Phase 4.2: Mapping PCA Inverse Visuals (${DATASET})"
    CUDA_VISIBLE_DEVICES=1 python visualize_boundaries.py --dataset ${DATASET}

    echo ">>> Phase 4.3: Mapping t-SNE Clustering (${DATASET})"
    CUDA_VISIBLE_DEVICES=1 python visualize_tsne.py --dataset ${DATASET}

    # ---------------------------------------------------------
    # Phase 5: Feature Perturbation
    # ---------------------------------------------------------
    echo ">>> Phase 5: Sweeping Geometrical Perturbations (${DATASET})"
    CUDA_VISIBLE_DEVICES=1 python experiment_feature_perturb.py --dataset ${DATASET} --noise_std 0.2 --epochs ${EPOCHS} --batch_size ${BATCH_SIZE}

done

echo "========================================================="
echo "        OVERNIGHT PIPELINE COMPLETELY FINISHED!          "
echo "========================================================="
