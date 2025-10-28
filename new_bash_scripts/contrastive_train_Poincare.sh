#!/bin/bash

# #SBATCH --output="logs/GCD-${data}-Hyperbolic-max_grad-${max_grad}-avg_grad-${avg_grad}-c-${c}-HypCD-${HypCD_mode}-Seed${seed}.log"
# #SBATCH --job-name="GCD-${data}-Hyperbolic-max_grad-${max_grad}-avg_grad-${avg_grad}-c-${c}-HypCD-${HypCD_mode}-Seed${seed}"
#SBATCH --time=2-00:00:00
#SBATCH --signal=B:SIGTERM@30
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-gpu=8
#SBATCH --mem=16G
#SBATCH --nodelist=a768-l40s-03
# #SBATCH --exclude=ailab-l4-02

#####################################################################################

data=$1
max_grad=$2
avg_grad=$3
c=$4
cs=$5
HypCD_mode=$6
seed=$7

# Default arguments: sbatch new_bash_scripts/contrastive_train_Poincare.sh cub 1.0 0.25 2.3 1 false 0

# Script arguments
container_path="${HOME}/pytorch25-06.sif"

# Dynamically set output and error filenames using job ID and iteration
outfile="logs/GCD-${data}-Hyperbolic-Poincare-max_grad-${max_grad}-avg_grad-${avg_grad}-c-${c}-cs-${cs}-HypCD-${HypCD_mode}-Seed${seed}.out"

exp_id="GCD-${data}-Hyperbolic-Poincare-max_grad-${max_grad}-avg_grad-${avg_grad}-c-${c}-cs-${cs}-HypCD-${HypCD_mode}-Seed${seed}"

# Print the filenames for debugging
echo "Output file: ${outfile}"
#echo "Error file: ${errfile}"
#echo "Restart num: ${restarts}"
echo "Using container: ${container_path}"

PYTHON='venv/bin/python'

hostname
nvidia-smi

# export CUDA_VISIBLE_DEVICES=0

# Get unique log file,
#SAVE_DIR=/ceph/home/student.aau.dk/mdalal20/P10-project/hyperbolic-generalized-category-discovery/dev_outputs/

#EXP_NUM=$(ls ${SAVE_DIR} | wc -l)
#EXP_NUM=$((${EXP_NUM}+1))
#echo $EXP_NUM

srun --output="${outfile}" --error="${outfile}" singularity exec --nv ${container_path} ${PYTHON} -m methods.contrastive_training.contrastive_training \
            --dataset_name ${data} \
            --batch_size 128 \
            --grad_from_block 11 \
            --epochs 200 \
            --epochs_warmup 20 \
            --base_model vit_dino \
            --num_workers 8 \
            --use_ssb_splits 'True' \
            --sup_con_weight 0.35 \
            --weight_decay 5e-5 \
            --contrast_unlabel_only 'False' \
            --transform 'imagenet' \
            --lr 0.1 \
            --eval_funcs 'v2' \
            --exp_id "${data}-Hyperbolic-Poincare-max_grad-${max_grad}-avg_grad-${avg_grad}-c-${c}-cs-${cs}-HypCD-${HypCD_mode}-Seed${seed}-Train" \
            --hyperbolic 'True' \
            --poincare 'True' \
            --kmeans 'True' \
            --kmeans_frequency 30 \
            --curvature '0.05' \
            --proj_alpha 1.0 \
            --freeze_curvature 'full' \
            --freeze_proj_alpha 'full' \
            --angle_loss 'True' \
            --max_angle_loss_weight 1.0 \
            --decay_angle_loss_weight 'True' \
            --euclidean_clipping ${c} \
            --clip_gradients 'False' \
            --mlp_out_dim 256 \
            --use_dinov2 'True' \
            --seed ${seed} \
            --cluster_size ${cs} \
            --HypCD_mode ${HypCD_mode} \
            #--use_adam 'True' \
#> ${SAVE_DIR}logfile_${EXP_NUM}.out

#-m methods.contrastive_training.contrastive_training --dataset_name 'cub' --batch_size 128 --grad_from_block 11 --epochs 200 --base_model vit_dino --num_workers 16 --use_ssb_splits 'True' --sup_con_weight 0.35 --weight_decay 5e-5 --contrast_unlabel_only 'False' --exp_id test_exp --transform 'imagenet' --lr 0.1 --eval_funcs 'v1' 'v2'