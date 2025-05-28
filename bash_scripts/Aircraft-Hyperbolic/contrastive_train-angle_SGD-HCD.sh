#!/bin/bash

#SBATCH --output="logs/GCD-Aircraft-Hyperbolic-Angle-SGD-HCD.log"
#SBATCH --job-name="GCD-Aircraft-Hyperbolic-Angle-SGD-HCD"
#SBATCH --time=12:00:00
#SBATCH --signal=B:SIGTERM@30
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-gpu=8
#SBATCH --mem=16G
# #SBATCH --nodelist=ailab-l4-07
#SBATCH --exclude=ailab-l4-02

#####################################################################################

# Script arguments
container_path="${HOME}/pytorch-24.08.sif"

# Dynamically set output and error filenames using job ID and iteration
outfile="logs/GCD-Aircraft-Hyperbolic-Angle-SGD-HCD.out"

# Print the filenames for debugging
echo "Output file: ${outfile}"
#echo "Error file: ${errfile}"
#echo "Restart num: ${restarts}"
echo "Using container: ${container_path}"

PYTHON='/ceph/home/student.aau.dk/mdalal20/P10-project/hyperbolic-generalized-category-discovery/venv/bin/python'

hostname
nvidia-smi

# export CUDA_VISIBLE_DEVICES=0

# Get unique log file,
#SAVE_DIR=/ceph/home/student.aau.dk/mdalal20/P10-project/hyperbolic-generalized-category-discovery/dev_outputs/

#EXP_NUM=$(ls ${SAVE_DIR} | wc -l)
#EXP_NUM=$((${EXP_NUM}+1))
#echo $EXP_NUM

srun --output="${outfile}" --error="${outfile}" singularity exec --nv ${container_path} ${PYTHON} -m methods.contrastive_training.contrastive_training \
            --dataset_name 'aircraft' \
            --batch_size 128 \
            --grad_from_block 11 \
            --epochs 200 \
            --epochs_warmup 20 \
            --base_model vit_dino \
            --num_workers 16 \
            --use_ssb_splits 'True' \
            --sup_con_weight 0.35 \
            --weight_decay 5e-5 \
            --contrast_unlabel_only 'False' \
            --transform 'imagenet' \
            --lr 0.1 \
            --eval_funcs 'v2' \
            --exp_id 'Aircraft-Hyperbolic-Angle-SGD-HCD-Train' \
            --hyperbolic 'True' \
            --kmeans 'True' \
            --kmeans_frequency 300 \
            --curvature 0.05 \
            --proj_alpha 1.0 \
            --freeze_curvature 'full' \
            --freeze_proj_alpha 'full' \
            --angle_loss 'True' \
            --max_angle_loss_weight 1.0 \
            --decay_angle_loss_weight 'True' \
            --euclidean_clipping 2.3 \
            --max_grad_norm 1.0 \
            --avg_grad_norm 0.25 \
            --mlp_out_dim 256 \
            #--use_adam 'True' \

#> ${SAVE_DIR}logfile_${EXP_NUM}.out