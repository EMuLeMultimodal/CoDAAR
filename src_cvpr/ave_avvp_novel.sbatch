#!/bin/bash

#SBATCH --job-name=AVE_train
#SBATCH --partition=leinegpu
#SBATCH --nodelist=leinevmgpu006
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --mem-per-cpu=5G 
#SBATCH --time=1:00:00
#SBATCH --output=/ave_avvp%j.out
#SBATCH --error=/ave_avvp%j.err

# Set source directory
SRCDIR=

# Create output directories for logs
mkdir -p $SRCDIR/AVE2AVVP

# Activate conda environment
source ~/.bashrc
conda activate /envs

CHECKPOINT_PATH_AV_40k=" "

python $SRCDIR/ave_avvp_novel.py \
    --gpu 0 \
    --lr 0.0004 \
    --clip_gradient 0.5 \
    --snapshot_pref "$SRCDIR/AVE2AVVP/V2A/" \
    --n_epoch 30 \
    --batch_size 80 \
    --test_batch_size 64 \
    --dataset_name "ave_avvp_va" \
    --modality "AV" \
    --checkpoint_path "$CHECKPOINT_PATH_AV_40k" \
    --print_freq 1 \
    --eval_freq 1

echo "AVE to AVVP training complete!"