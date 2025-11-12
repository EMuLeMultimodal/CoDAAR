#!/bin/bash

#SBATCH --job-name=AVT_pretest
#SBATCH --partition=leinegpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:a100-40g:1
#SBATCH --cpus-per-task=32
#SBATCH --mem-per-cpu=5G 
#SBATCH --time=10:00:00
#SBATCH --output=/pretest_%j.out
#SBATCH --error=/pretest_%j.err

# Set source directory
SRCDIR=

mkdir -p $SRCDIR/AVS_Results_Novel_test
mkdir -p $SRCDIR/40k/test/T2A
mkdir -p $SRCDIR/40k/test/A2T


CHECKPOINT_PATH_AVT_40k=" "

# Activate conda environment

# Activate conda environment
source ~/.bashrc
conda activate /envs


setting='S4'
visual_backbone="pvt"

#test T2A 40k
weight_TA_40k=" "

python $SRCDIR/AVSBench_dowmstream/avs_scripts/avs_s4/test_ta_novel.py \
        --session_name ${setting}_${visual_backbone} \
        --visual_backbone ${visual_backbone} \
        --test_batch_size 4 \
        --lr 0.0001 \
        --tpavi_stages 0 1 2 3 \
        --checkpoint_path "$CHECKPOINT_PATH_AVT_40k" \
        --weights "$weight_TA_40k" \
        --log_dir "$SRCDIR/40k/test/T2A/" \
        --tpavi_va_flag \
        --save_pred_mask 


#test A2T 40k
weight_AT_40k=" "

python $SRCDIR/AVSBench_dowmstream/avs_scripts/avs_s4/test_at_novel.py \
        --session_name ${setting}_${visual_backbone} \
        --visual_backbone ${visual_backbone} \
        --test_batch_size 4 \
        --lr 0.0001 \
        --tpavi_stages 0 1 2 3 \
        --checkpoint_path "$CHECKPOINT_PATH_AVT_40k" \
        --weights "$weight_AT_40k" \
        --log_dir "$SRCDIR/40k/test/A2T/" \
        --tpavi_va_flag \
        --save_pred_mask 


