#!/bin/bash

#SBATCH --output="logs/GCD-CUB-Hyperbolic-HCD-angle-only.log"
#SBATCH --job-name="GCD-CUB-Hyperbolic-HCD-angle-only"
#SBATCH --time=12:00:00
#SBATCH --signal=B:SIGTERM@30
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-gpu=8
#SBATCH --mem=16G
# #SBATCH --nodelist=ailab-l4-07
# #SBATCH --exclude=ailab-l4-02

#####################################################################################

# Script arguments
container_path="${HOME}/pytorch-24.08.sif"

# Dynamically set output and error filenames using job ID and iteration
outfile="logs/GCD-CUB-Hyperbolic-HCD-angle-only.out"

exp_id="GCD-CUB-Hyperbolic-HCD-angle-only"

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
            --dataset_name 'cub' \
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
            --exp_id ${exp_id} \
            --transform 'imagenet' \
            --lr 0.1 \
            --eval_funcs 'v2' \
            --exp_id 'CUB-Hyperbolic-HCD-angle-only-Train' \
            --hyperbolic 'True' \
            --poincare 'True' \
            --kmeans 'True' \
            --kmeans_frequency 20 \
            --curvature '0.05' \
            --proj_alpha 1.0 \
            --freeze_curvature 'full' \
            --freeze_proj_alpha 'full' \
            --angle_loss 'True' \
            --max_angle_loss_weight 1.0 \
            --decay_angle_loss_weight 'False' \
            --euclidean_clipping 2.3 \
            --mlp_out_dim 256 \
            --checkpoint_path '/ceph/home/student.aau.dk/mdalal20/P10-project/hyperbolic-generalized-category-discovery/osr_novel_categories/metric_learn_gcd/log/CUB-Hyperbolic-HCD-angle-only-Train/checkpoints/model.pt' \
#> ${SAVE_DIR}logfile_${EXP_NUM}.out

#-m methods.contrastive_training.contrastive_training --dataset_name 'cub' --batch_size 128 --grad_from_block 11 --epochs 200 --base_model vit_dino --num_workers 16 --use_ssb_splits 'True' --sup_con_weight 0.35 --weight_decay 5e-5 --contrast_unlabel_only 'False' --exp_id test_exp --transform 'imagenet' --lr 0.1 --eval_funcs 'v1' 'v2'