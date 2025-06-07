#!/bin/bash
#SBATCH --job-name=eval_artist
#SBATCH --output=logs/out-%x-%j.log       # Specify the output log file
#SBATCH --error=logs/err-%x-%j.log         # Specify the error log file
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:h100-sxm:1
#SBATCH --ntasks-per-node=1
#SBATCH --time=06:00:00
#SBATCH --qos=high
#SBATCH --mem=128G
#SBATCH --array=0-2


source ~/.bashrc
conda activate style_moe

cd path/to//research/projects/concept_prune || exit

method="uce"
concepts=("Van Gogh" "Monet" "Pablo Picasso")
concept=${concepts[$SLURM_ARRAY_TASK_ID]}

srun python artist_erasure.py \
--target \
"$concept" \
--baseline \
$method \
--base_config_path \
path/to/configs/styles/img/sd-2-1_coco.yaml \
--model_id \
"stabilityai/stable-diffusion-2-1"
