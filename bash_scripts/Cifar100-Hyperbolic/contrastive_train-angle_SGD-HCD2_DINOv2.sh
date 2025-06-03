#!/bin/bash

#SBATCH --output="logs/GCD-CIFAR100-Hyperbolic-Angle-SGD-HCD2-DINOv2.log"
#SBATCH --job-name="GCD-CIFAR100-Hyperbolic-Angle-SGD-HCD2-DINOv2"
#SBATCH --time=12:00:00
#SBATCH --signal=B:SIGTERM@30
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-gpu=8
#SBATCH --mem=16G
# #SBATCH --nodelist=ailab-l4-07
#SBATCH --exclude=ailab-l4-02

#####################################################################################

# This means that the model will potentially run 4 times
max_restarts=6

# Fetch the current restarts value from the job context
scontext=$(scontrol show job ${SLURM_JOB_ID})
restarts=$(echo ${scontext} | grep -o 'Restarts=[0-9]*' | cut -d= -f2)

# If no restarts found, it's the first run, so set restarts to 0
iteration=${restarts:-0}

# Script arguments
container_path="${HOME}/pytorch-24.08.sif"

# Dynamically set output and error filenames using job ID and iteration
outfile="logs/GCD-CIFAR100-Hyperbolic-Angle-SGD-HCD2-DINOv2.out"

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

##  Define a term-handler function to be executed           ##
##  when the job gets the SIGTERM (before timeout)          ##

term_handler()
{
    echo "Executing term handler at $(date)"
    if [[ $restarts -lt $max_restarts ]]; then
        # Requeue the job, allowing it to restart with incremented iteration
        scontrol requeue ${SLURM_JOB_ID}
        exit 0
    else
        echo "Maximum restarts reached, exiting."
        exit 1
    fi
}

# Trap SIGTERM to execute the term_handler when the job gets terminated
trap 'term_handler' SIGTERM

#######################################################################################
if [ $restarts -gt  ]; then
    srun --output="${outfile}" --error="${outfile}" singularity exec --nv ${container_path} ${PYTHON} -m methods.contrastive_training.contrastive_training \
            --dataset_name 'cifar100' \
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
            --exp_id 'CIFAR100-Hyperbolic-Angle-SGD-HCD2-DINOv2-Train' \
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
            --max_grad_norm 1.0 \
            --avg_grad_norm 0.25 \
            --mlp_out_dim 256 \
            --use_dinov2 'True' \
            --checkpoint_path '/ceph/home/student.aau.dk/mdalal20/P10-project/hyperbolic-generalized-category-discovery/osr_novel_categories/metric_learn_gcd/log/CIFAR100-Hyperbolic-Angle-SGD-HCD2-DINOv2-Train/checkpoints/model.pt'
            #--use_adam 'True' \
else
    srun --output="${outfile}" --error="${outfile}" singularity exec --nv ${container_path} ${PYTHON} -m methods.contrastive_training.contrastive_training \
            --dataset_name 'cifar100' \
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
            --exp_id 'CIFAR100-Hyperbolic-Angle-SGD-HCD2-DINOv2-Train' \
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
            --max_grad_norm 1.0 \
            --avg_grad_norm 0.25 \
            --mlp_out_dim 256 \
            --use_dinov2 'True' \
            #--use_adam 'True' \
fi

#> ${SAVE_DIR}logfile_${EXP_NUM}.out