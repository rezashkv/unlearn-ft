#!/bin/bash
#SBATCH --job-name=eval_nudity
#SBATCH --output=logs/out-%x-%j.log       # Specify the output log file
#SBATCH --error=logs/err-%x-%j.log         # Specify the error log file
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:rtx6000ada:1
#SBATCH --ntasks-per-node=1
#SBATCH --time=72:00:00
#SBATCH --qos=high
#SBATCH --mem=64G
#SBATCH --array=0


source ~/.bashrc
conda activate unlearn-ft

cd path/to//projects/concept_prune || exit

methods=("esd")
method=${methods[$SLURM_ARRAY_TASK_ID]}
eval_dataset="i2p"

srun python nudity_eval.py \
--target \
"naked" \
--eval_dataset \
"$eval_dataset" \
--baseline \
"$method" \
--base_config_path \
path/to/unlearn-ft/configs/img/sd-2-1_coco.yaml \
--model_id \
"stabilityai/stable-diffusion-2-1" \

