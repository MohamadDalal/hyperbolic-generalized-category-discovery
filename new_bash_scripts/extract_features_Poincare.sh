#!/bin/bash

# #SBATCH --output="logs/GCD-Extract-Aircraft-Hyperbolic-Angle-SGD-HCD-DINOv2.log"
# #SBATCH --job-name="GCD-Extract-Aircraft-Hyperbolic-Angle-SGD-HCD-DINOv2"
#SBATCH --time=2:00:00
#SBATCH --signal=B:SIGTERM@30
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-gpu=8
#SBATCH --mem=16G
# #SBATCH --nodelist=ailab-l4-07
# #SBATCH --exclude=nv-ai-03

#####################################################################################

data=$1
c=$2
cs=$3
HypCD_mode=$4
seed=$5
Euclidean=$6

# Script arguments
container_path="${HOME}/pytorch25-06.sif"

# Dynamically set output and error filenames using job ID and iteration
outfile="logs/GCD-Extract-${data}-Hyperbolic-Poincare-c-${c}-cs-${cs}-HypCD-${HypCD_mode}-Seed${seed}-Euclidean${Euclidean}.out"

# Print the filenames for debugging
echo "Output file: ${outfile}"
#echo "Error file: ${errfile}"
#echo "Restart num: ${restarts}"
echo "Using container: ${container_path}"

PYTHON='venv/bin/python'

hostname
nvidia-smi

#export CUDA_VISIBLE_DEVICES=0
if [ "$Euclidean" = "True" ]; then
    echo "Extracting features in Euclidean space"
    srun --output="${outfile}" --error="${outfile}" singularity exec --nv ${container_path} ${PYTHON} -m methods.clustering.extract_features --dataset ${data} --use_best_model 'False' \
    --warmup_model_dir "/home/create.aau.dk/lt61yz/P10-project/hyperbolic-generalized-category-discovery/osr_novel_categories/metric_learn_gcd/log/${data}-Hyperbolic-max_grad-${max_grad}-avg_grad-${avg_grad}-c-${c}-HypCD-${HypCD_mode}-Seed${seed}-Train/checkpoints/model_best_loss.pt" \
    --exp_id "_Hyperbolic-Poincare-c-${c}-cs-${cs}-HypCD-${HypCD_mode}-Seed${seed}_Euclidean" --hyperbolic 'True' --poincare 'True' --euclidean_clipping 2.3 --remove_dyno_head 'True' --mlp_out_dim 256 --use_dinov2 'True' --HypCD_mode ${HypCD_mode}
else
    echo "Extracting features in Hyperbolic space"
    srun --output="${outfile}" --error="${outfile}" singularity exec --nv ${container_path} ${PYTHON} -m methods.clustering.extract_features --dataset ${data} --use_best_model 'False' \
    --warmup_model_dir "/home/create.aau.dk/lt61yz/P10-project/hyperbolic-generalized-category-discovery/osr_novel_categories/metric_learn_gcd/log/${data}-Hyperbolic-max_grad-${max_grad}-avg_grad-${avg_grad}-c-${c}-HypCD-${HypCD_mode}-Seed${seed}-Train/checkpoints/model_best_loss.pt" \
    --exp_id "_Hyperbolic-Poincare-c-${c}-cs-${cs}-HypCD-${HypCD_mode}-Seed${seed}" --hyperbolic 'True' --poincare 'True' --euclidean_clipping 2.3 --remove_dyno_head 'False' --mlp_out_dim 256 --use_dinov2 'True' --HypCD_mode ${HypCD_mode}
fi
