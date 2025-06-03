#!/bin/bash

#SBATCH --output="logs/GCD-Extract-CUB-Hyperbolic-Angle-SGD-HCD-DINOv2-65536.log"
#SBATCH --job-name="GCD-Extract-CUB-Hyperbolic-Angle-SGD-HCD-DINOv2-65536"
#SBATCH --time=2:00:00
#SBATCH --signal=B:SIGTERM@30
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-gpu=8
#SBATCH --mem=16G
# #SBATCH --nodelist=ailab-l4-07
# #SBATCH --exclude=nv-ai-03

#####################################################################################

# Script arguments
container_path="${HOME}/pytorch-24.08.sif"

# Dynamically set output and error filenames using job ID and iteration
outfile="logs/GCD-Extract-CUB-Hyperbolic-Angle-SGD-HCD-DINOv2-65536.out"

# Print the filenames for debugging
echo "Output file: ${outfile}"
#echo "Error file: ${errfile}"
#echo "Restart num: ${restarts}"
echo "Using container: ${container_path}"

PYTHON='/ceph/home/student.aau.dk/mdalal20/P10-project/hyperbolic-generalized-category-discovery/venv/bin/python'

hostname
nvidia-smi

#export CUDA_VISIBLE_DEVICES=0

srun --output="${outfile}" --error="${outfile}" singularity exec --nv ${container_path} ${PYTHON} -m methods.clustering.extract_features --dataset cub --use_best_model 'False' \
 --warmup_model_dir '/ceph/home/student.aau.dk/mdalal20/P10-project/hyperbolic-generalized-category-discovery/osr_novel_categories/metric_learn_gcd/log/CUB-Hyperbolic-Angle-SGD-HCD-DINOv2-65536-Train/checkpoints/model_best_loss.pt' \
 --exp_id '_Hyperbolic-Angle-SGD-HCD-DINOv2-65536_Euclidean' --hyperbolic 'True' --poincare 'False' --euclidean_clipping 2.3 --remove_dyno_head 'True' --mlp_out_dim 65536 --use_dinov2 'True'