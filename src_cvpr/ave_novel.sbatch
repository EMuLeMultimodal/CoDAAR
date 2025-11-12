#!/bin/bash

#SBATCH --job-name=AVE_train
#SBATCH --partition=leinegpu
#SBATCH --nodelist=leinevmgpu006
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --mem-per-cpu=5G 
#SBATCH --time=1:00:00
#SBATCH --output=/ave_%j.out
#SBATCH --error=/ave_%j.err

# Set source directory
SRCDIR=

# Create output directories for logs
mkdir -p $SRCDIR/AVE_Tests

# Activate conda environment
source ~/.bashrc
conda activate /envs

CHECKPOINT_PATH_AV_40k=" "

# Run AVE training
python $SRCDIR/ave_novel.py \
    --gpu 0 \
    --lr 0.00025 \
    --clip_gradient 0.5 \
    --snapshot_pref "$SRCDIR/AVE_Tests/A2V" \
    --n_epoch 30 \
    --batch_size 256 \
    --test_batch_size 128 \
    --dataset_name "ave_av" \
    --modality "AV" \
    --model_save_path "$SRCDIR/AVE_Tests/A2V/" \
    --checkpoint_path "$CHECKPOINT_PATH_AV_40k" \
    --print_freq 1 \
    --eval_freq 1

# echo "AVE training complete !!"