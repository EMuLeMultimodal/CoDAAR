#!/bin/bash

#SBATCH --job-name=AV_pretrain
#SBATCH --partition=leinegpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:a100-80g:1
#SBATCH --cpus-per-task=32
#SBATCH --mem-per-cpu=5G 
#SBATCH --time=2:00:00
#SBATCH --output=/pretrain_%j.out
#SBATCH --error=/pretrain_%j.err

# Set source directory
SRCDIR=

# Create output directories for logs
mkdir -p $SRCDIR/AV_Novel
mkdir -p $SRCDIR/AV/90k/checkpoint
mkdir -p $SRCDIR/AV/40k/checkpoint


# Activate conda environment
source ~/.bashrc
conda activate /envs


# Run pretraining
python $SRCDIR/pretrain_novel_AV.py \
    --gpu 0 \
    --lr 0.0004 \
    --clip_gradient 0.5 \
    --snapshot_pref "$SRCDIR/AV/90k/" \
    --n_epoch 6 \
    --batch_size 80 \
    --test_batch_size 64 \
    --dataset_name "vggsound_AVT_90k" \
    --model_save_path "$SRCDIR/90k/checkpoint/" \
    --print_freq 1

echo "Pretraining complete!"