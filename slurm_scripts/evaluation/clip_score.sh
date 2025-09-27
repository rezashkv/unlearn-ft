#!/bin/bash
#SBATCH --job-name=clip # Specify a name for your job
#SBATCH --output=logs/out-%x-%j.log       # Specify the output log file
#SBATCH --error=logs/err-%x-%j.log         # Specify the error log file
#SBATCH --cpus-per-task=4
#SBATCH --nodes=1
#SBATCH --gres=gpu:rtx6000ada:1   # Number of GPUs to request and specify the GPU type
#SBATCH --time=06:00:00            # Maximum execution time (HH:MM:SS)
#SBATCH --qos=high
#SBATCH --mem=64G                # Memory per node


source ~/.bashrc
conda activate unlearn-ft
cd path/to/unlearn-ft/scripts/metrics || exit

fid_images_dir=$1
coco_features_dir="path/to//coco/annotations/clip-captions/ViT-B-32_clip_features/"
result_dir="path/to/results/"


python3 clip_score.py --gen_images_dir $fid_images_dir --text_features_dir $coco_features_dir --dataset_name coco-val-2017 --result_dir $result_dir 2>&1
