#!/bin/bash

#SBATCH --output="logs/GCD-Extract-Aircraft-Hyperbolic-HCD-angle-only.log"
#SBATCH --job-name="GCD-Extract-Aircraft-Hyperbolic-HCD-angle-only"
#SBATCH --time=12:00:00
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
outfile="logs/GCD-Extract-Aircraft-Hyperbolic-HCD-angle-only.out"

# Print the filenames for debugging
echo "Output file: ${outfile}"
#echo "Error file: ${errfile}"
#echo "Restart num: ${restarts}"
echo "Using container: ${container_path}"

PYTHON='/ceph/home/student.aau.dk/mdalal20/P10-project/hyperbolic-generalized-category-discovery/venv/bin/python'

hostname
nvidia-smi

#export CUDA_VISIBLE_DEVICES=0

srun --output="${outfile}" --error="${outfile}" singularity exec --nv ${container_path} ${PYTHON} -m methods.clustering.extract_features --dataset aircraft --use_best_model 'False' \
 --warmup_model_dir '/ceph/home/student.aau.dk/mdalal20/P10-project/hyperbolic-generalized-category-discovery/osr_novel_categories/metric_learn_gcd/log/Aircraft-Hyperbolic-HCD-angle-only-Train/checkpoints/model_best_loss.pt' \
 --exp_id '_Hyperbolic-HCD-angle-only_Euclidean' --hyperbolic 'True' --poincare 'True' --euclidean_clipping 2.3 --remove_dyno_head 'True' --mlp_out_dim 256