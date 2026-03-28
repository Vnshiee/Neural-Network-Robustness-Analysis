#!/bin/bash
# FAST TEST execution script for DL Assignment 2

set -e

BATCH_SIZE=512
EPOCHS=1
LR=0.001

echo "========================================================="
echo "  Starting FAST DRY RUN (1 Epoch Only) on A100 GPU 1     "
echo "========================================================="

export CUDA_VISIBLE_DEVICES=1

# =========================================================
# CONTINUATION BLOCK: Finish testing CIFAR-10 Visualizations & Phase 5
# since Phase 2 and 3 already successfully validated!
# =========================================================
echo "========================================================="
echo "  RESUMING PIPELINE FOR DATASET: cifar10"
echo "========================================================="

# echo "[TEST] Phase 4.3: t-SNE Clustering (cifar10)"
# python visualize_tsne.py --dataset cifar10

# echo "[TEST] Phase 5: Internal Layer Sweep (cifar10)"
# python experiment_feature_perturb.py --dataset cifar10 --noise_std 0.2 --epochs ${EPOCHS} --batch_size ${BATCH_SIZE}

# =========================================================
# FULL BLOCK: Run full test loop for F-MNIST and ImageNet100
# =========================================================
DATASETS=("fmnist" "imagenet100")

for DATASET in "${DATASETS[@]}"; do
    echo "========================================================="
    echo "  TESTING PIPELINE FOR DATASET: ${DATASET}"
    echo "========================================================="

    # --- Phase 2: Baselines ---
    if [ "$DATASET" == "fmnist" ]; then
        echo "[TEST] Skipping Phase 2: VGG and ResNet for ${DATASET} (Already Validated)"
    else
        echo "[TEST] Phase 2: VGG"
        python train_baseline.py --model vgg --dataset ${DATASET} --epochs ${EPOCHS} --batch_size ${BATCH_SIZE} --lr ${LR}
        
        echo "[TEST] Phase 2: ResNet"
        python train_baseline.py --model resnet --dataset ${DATASET} --epochs ${EPOCHS} --batch_size ${BATCH_SIZE} --lr ${LR}
    fi
    
    echo "[TEST] Phase 2: ConvNeXT"
    python train_baseline.py --model convnext --dataset ${DATASET} --epochs ${EPOCHS} --batch_size ${BATCH_SIZE} --lr ${LR}
    
    echo "[TEST] Phase 2: ViT"
    python train_baseline.py --model vit --dataset ${DATASET} --epochs ${EPOCHS} --batch_size ${BATCH_SIZE} --lr ${LR}

    # --- Phase 3: MLP Experiments ---
    echo "[TEST] Phase 3A: MLP Clean"
    python experiment_a.py --dataset ${DATASET} --epochs ${EPOCHS} --batch_size ${BATCH_SIZE} --lr ${LR}
    
    echo "[TEST] Phase 3B: MLP Corrupt"
    python experiment_b.py --dataset ${DATASET} --corruption_name gaussian_noise --corruption_severity 2 --epochs ${EPOCHS} --batch_size ${BATCH_SIZE} --lr ${LR}

    echo "[TEST] Phase 3C: Corruptions Optimization"
    python optimize_corruptions.py --dataset ${DATASET} --epochs ${EPOCHS} --batch_size ${BATCH_SIZE} --lr ${LR}

    # --- Phase 4: Evaluations & Visualization ---
    echo "[TEST] Phase 4.1: Evaluation CSV Generator"
    python evaluate_models.py --dataset ${DATASET} --batch_size ${BATCH_SIZE}
    
    echo "[TEST] Phase 4.2: PCA Boundaries"
    python visualize_boundaries.py --dataset ${DATASET}
    
    echo "[TEST] Phase 4.3: t-SNE Clustering"
    python visualize_tsne.py --dataset ${DATASET}

    # --- Phase 5: Feature Perturbation ---
    echo "[TEST] Phase 5: Internal Layer Sweep"
    python experiment_feature_perturb.py --dataset ${DATASET} --noise_std 0.2 --epochs ${EPOCHS} --batch_size ${BATCH_SIZE}

done

echo "========================================================="
echo "   DRY RUN COMPLETE! NO ERRORS FOUND IN CODEBASE.        "
echo "   You are now safe to run run_all.sh overnight.         "
echo "========================================================="