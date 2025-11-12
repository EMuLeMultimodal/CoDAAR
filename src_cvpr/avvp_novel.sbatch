#!/bin/bash

#SBATCH --job-name=AVVP_train
#SBATCH --partition=leinegpu
#SBATCH --nodelist=leinevmgpu006
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --mem-per-cpu=5G 
#SBATCH --time=1:00:00
#SBATCH --output=/avvp_%j.out
#SBATCH --error=/avvp_%j.err

# Set source directory
SRCDIR=

# Create output directories for logs
mkdir -p $SRCDIR/AVVP_Tests

# Activate conda environment
source ~/.bashrc
conda activate /envs

CHECKPOINT_PATH_AV_40k=" "

# Run AVVP training
python $SRCDIR/avvp_novel.py \
    --gpu 0 \
    --lr 0.0004 \
    --clip_gradient 0.5 \
    --snapshot_pref "$SRCDIR/AVVP_Tests/AV/" \
    --n_epoch 50 \
    --batch_size 64 \
    --test_batch_size 64 \
    --dataset_name "avvp_av" \
    --modality "AV" \
    --model_save_path "$SRCDIR/AVVP_Tests/AV/" \
    --checkpoint_path "$CHECKPOINT_PATH_AV_40k" \
    --print_freq 1 \
    --eval_freq 1

echo "AVVP training complete!"