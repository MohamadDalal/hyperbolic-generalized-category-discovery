#!/bin/bash

#SBATCH --output="logs/GCD-KMeans-CUB-Hyperbolic-Angle-SGD-HCD-DINOv2-32.log"
#SBATCH --job-name="GCD-KMeans-CUB-Hyperbolic-Angle-SGD-HCD-DINOv2-32"
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
outfile="logs/GCD-KMeans-CUB-Hyperbolic-Angle-SGD-HCD-DINOv2-32.out"

# Print the filenames for debugging
echo "Output file: ${outfile}"
#echo "Error file: ${errfile}"
#echo "Restart num: ${restarts}"
echo "Using container: ${container_path}"

PYTHON='/ceph/home/student.aau.dk/mdalal20/P10-project/hyperbolic-generalized-category-discovery/venv/bin/python'

hostname
nvidia-smi

#export CUDA_VISIBLE_DEVICES=0

# Get unique log file
#SAVE_DIR=/work/sagar/osr_novel_categories/dev_outputs/

#EXP_NUM=$(ls ${SAVE_DIR} | wc -l)
#EXP_NUM=$((${EXP_NUM}+1))
#echo $EXP_NUM

srun --output="${outfile}" --error="${outfile}" singularity exec --nv ${container_path} ${PYTHON} -m methods.clustering.k_means --dataset 'cub' --semi_sup 'True' --use_ssb_splits 'True' \
 --use_best_model 'False' --max_kmeans_iter 200 --k_means_init 100 --warmup_model_exp_id 'Hyperbolic-Angle-SGD-HCD-DINOv2-32_Euclidean' --hyperbolic 'False' --poincare 'False' #--K 50
 #> ${SAVE_DIR}logfile_${EXP_NUM}.out