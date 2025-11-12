#!/bin/bash

#SBATCH --job-name=AVT_pretrain
#SBATCH --partition=leinegpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:a100-40g:1
#SBATCH --cpus-per-task=32
#SBATCH --mem-per-cpu=5G 
#SBATCH --time=10:00:00
#SBATCH --output=/pretrain_%j.out
#SBATCH --error=/pretrain_%j.err

# Set source directory
SRCDIR=

mkdir -p $SRCDIR/AVS_Results_Novel_train
mkdir -p $SRCDIR/40k/train/T2A
mkdir -p $SRCDIR/40k/train/A2T

CHECKPOINT_PATH_AVT_40k=" "

# Activate conda environment
source ~/.bashrc
conda activate /envs

setting='S4'
visual_backbone="pvt"

#Train T2A 40k

python $SRCDIR/AVSBench_dowmstream/avs_scripts/avs_s4/train_ta_novel.py \
        --session_name ${setting}_${visual_backbone} \
        --visual_backbone ${visual_backbone} \
        --train_batch_size 4 \
        --lr 0.0001 \
        --max_epoches 4 \
        --tpavi_stages 0 1 2 3 \
        --checkpoint_path "$CHECKPOINT_PATH_AVT_40k" \
        --log_dir "$SRCDIR/40k/train/T2A/" \
        --tpavi_va_flag 


#Train A2T 40k

python $SRCDIR/AVSBench_dowmstream/avs_scripts/avs_s4/train_at_novel.py \
        --session_name ${setting}_${visual_backbone} \
        --visual_backbone ${visual_backbone} \
        --train_batch_size 4 \
        --lr 0.0001 \
        --max_epoches 4 \
        --tpavi_stages 0 1 2 3 \
        --checkpoint_path "$CHECKPOINT_PATH_AVT_40k" \
        --log_dir "$SRCDIR/40k/train/A2T/" \
        --tpavi_va_flag 


