#!/bin/bash

# Experiment Runner for MNIST MLP Classification
# This script runs all the ablation studies mentioned in the homework



echo "================================================"
echo "MNIST MLP - Automated Experiment Runner"
echo "================================================"
echo ""
echo "This script will run multiple experiments to analyze:"
echo "1. Architecture impact (layers & width)"
echo "2. Activation functions (ReLU vs GELU)"
echo "3. Dropout impact"
echo "4. Batch normalization"
echo "5. Training framework (optimizers & schedulers)"
echo "6. Regularization (L1/L2)"
echo ""
echo "Total experiments: ~25-30"
echo "Estimated time: 2-4 hours (depending on hardware)"
echo ""
read -p "Press Enter to continue or Ctrl+C to cancel..."

# Create experiments directory
mkdir -p experiments

# 1. ARCHITECTURE EXPERIMENTS
echo ""
echo "========================================"
echo "1. Architecture Experiments"
echo "========================================"

echo "Running: 2-layer [256, 128]..."
python main.py --mode both --hidden_dims 256 128 \
    --save_dir experiments/arch_2layer_256_128 \
    --checkpoint_path checkpoints/arch_2layer_256_128.pth

echo "Running: 3-layer [512, 256, 128]..."
python main.py --mode both --hidden_dims 512 256 128 \
    --save_dir experiments/arch_3layer_512_256_128 \
    --checkpoint_path checkpoints/arch_3layer_512_256_128.pth

echo "Running: 4-layer [512, 256, 128, 64]..."
python main.py --mode both --hidden_dims 512 256 128 64 \
    --save_dir experiments/arch_4layer_512_256_128_64 \
    --checkpoint_path checkpoints/arch_4layer_512_256_128_64.pth

echo "Running: Wide [512, 256]..."
python main.py --mode both --hidden_dims 512 256 \
    --save_dir experiments/arch_wide_512_256 \
    --checkpoint_path checkpoints/arch_wide_512_256.pth

echo "Running: Narrow [128, 64]..."
python main.py --mode both --hidden_dims 128 64 \
    --save_dir experiments/arch_narrow_128_64 \
    --checkpoint_path checkpoints/arch_narrow_128_64.pth

# 2. ACTIVATION FUNCTION EXPERIMENTS
echo ""
echo "========================================"
echo "2. Activation Function Experiments"
echo "========================================"

echo "Running: ReLU activation..."
python main.py --mode both --activation relu \
    --save_dir experiments/activation_relu \
    --checkpoint_path checkpoints/activation_relu.pth

echo "Running: GELU activation..."
python main.py --mode both --activation gelu \
    --save_dir experiments/activation_gelu \
    --checkpoint_path checkpoints/activation_gelu.pth

# 3. DROPOUT EXPERIMENTS
echo ""
echo "========================================"
echo "3. Dropout Experiments"
echo "========================================"

echo "Running: No dropout (0.0)..."
python main.py --mode both --dropout 0.0 \
    --save_dir experiments/dropout_0.0 \
    --checkpoint_path checkpoints/dropout_0.0.pth

echo "Running: Dropout 0.1..."
python main.py --mode both --dropout 0.1 \
    --save_dir experiments/dropout_0.1 \
    --checkpoint_path checkpoints/dropout_0.1.pth

echo "Running: Dropout 0.3..."
python main.py --mode both --dropout 0.3 \
    --save_dir experiments/dropout_0.3 \
    --checkpoint_path checkpoints/dropout_0.3.pth

echo "Running: Dropout 0.5..."
python main.py --mode both --dropout 0.5 \
    --save_dir experiments/dropout_0.5 \
    --checkpoint_path checkpoints/dropout_0.5.pth

# 4. BATCH NORMALIZATION EXPERIMENTS
echo ""
echo "========================================"
echo "4. Batch Normalization Experiments"
echo "========================================"

echo "Running: Without BN..."
python main.py --mode both \
    --save_dir experiments/no_bn \
    --checkpoint_path checkpoints/no_bn.pth

echo "Running: With BN..."
python main.py --mode both --use_bn \
    --save_dir experiments/with_bn \
    --checkpoint_path checkpoints/with_bn.pth



# 5. OPTIMIZER EXPERIMENTS
echo ""
echo "========================================"
echo "5. Optimizer Experiments"
echo "========================================"

echo "Running: SGD optimizer..."
python main.py --mode both --optimizer sgd --lr 0.01 --momentum 0.9 \
    --save_dir experiments/optimizer_sgd \
    --checkpoint_path checkpoints/optimizer_sgd.pth

echo "Running: Adam optimizer..."
python main.py --mode both --optimizer adam --lr 0.001 \
    --save_dir experiments/optimizer_adam \
    --checkpoint_path checkpoints/optimizer_adam.pth

echo "Running: AdamW optimizer..."
python main.py --mode both --optimizer adamw --lr 0.001 --weight_decay 0.01 \
    --save_dir experiments/optimizer_adamw \
    --checkpoint_path checkpoints/optimizer_adamw.pth

# 6. LEARNING RATE SCHEDULER EXPERIMENTS
echo ""
echo "========================================"
echo "6. Learning Rate Scheduler Experiments"
echo "========================================"

echo "Running: No scheduler..."
python main.py --mode both \
    --save_dir experiments/scheduler_none \
    --checkpoint_path checkpoints/scheduler_none.pth

echo "Running: StepLR scheduler..."
python main.py --mode both --scheduler step --scheduler_step_size 10 --scheduler_gamma 0.1 \
    --save_dir experiments/scheduler_step \
    --checkpoint_path checkpoints/scheduler_step.pth

echo "Running: ReduceLROnPlateau scheduler..."
python main.py --mode both --scheduler plateau --scheduler_patience 5 \
    --save_dir experiments/scheduler_plateau \
    --checkpoint_path checkpoints/scheduler_plateau.pth

echo "Running: CosineAnnealingLR scheduler..."
python main.py --mode both --scheduler cosine \
    --save_dir experiments/scheduler_cosine \
    --checkpoint_path checkpoints/scheduler_cosine.pth

# 7. REGULARIZATION EXPERIMENTS
echo ""
echo "========================================"
echo "7. Regularization Experiments"
echo "========================================"

echo "Running: No regularization..."
python main.py --mode both \
    --save_dir experiments/reg_none \
    --checkpoint_path checkpoints/reg_none.pth

echo "Running: L2 regularization (1e-4)..."
python main.py --mode both --weight_decay 0.0001 \
    --save_dir experiments/reg_l2_1e4 \
    --checkpoint_path checkpoints/reg_l2_1e4.pth

echo "Running: L2 regularization (1e-3)..."
python main.py --mode both --weight_decay 0.001 \
    --save_dir experiments/reg_l2_1e3 \
    --checkpoint_path checkpoints/reg_l2_1e3.pth

echo "Running: L1 regularization (1e-4)..."
python main.py --mode both --l1_lambda 0.0001 \
    --save_dir experiments/reg_l1_1e4 \
    --checkpoint_path checkpoints/reg_l1_1e4.pth

echo "Running: L1 regularization (1e-3)..."
python main.py --mode both --l1_lambda 0.001 \
    --save_dir experiments/reg_l1_1e3 \
    --checkpoint_path checkpoints/reg_l1_1e3.pth

# 8. BEST MODEL (combining insights from experiments)
# Based on experimental results:
# - Best test accuracy: 98.34% (scheduler_plateau)
# - Best dropout: 0.1 (98.27%)
# - Best regularization: L2 1e-3 (98.09%)
# - Best batch norm: with_bn (98.21%)
# - Best architecture: 3-layer [512, 256, 128] (97.91%)
# - Best activation: GELU (97.90%)
echo ""
echo "========================================"
echo "8. Best Model Configuration"
echo "========================================"

echo "Running: Best configuration (combining top performers)..."
python main.py --mode both \
    --hidden_dims 256 128 \
    --activation gelu \
    --dropout 0.1 \
    --use_bn \
    --optimizer adam \
    --lr 0.001 \
    --scheduler plateau \
    --scheduler_patience 5 \
    --weight_decay 0.001 \
    --epochs 50 \
    --compute_tsne \
    --save_dir experiments/best_model \
    --checkpoint_path checkpoints/best_model.pth
