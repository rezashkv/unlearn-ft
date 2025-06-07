#!/bin/bash
#SBATCH --job-name=run_concept_prune
#SBATCH --output=logs/out-%x-%j.log       # Specify the output log file
#SBATCH --error=logs/err-%x-%j.log         # Specify the error log file
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:h100-sxm:1
#SBATCH --ntasks-per-node=1
#SBATCH --time=06:00:00
#SBATCH --qos=high
#SBATCH --mem=128G

source ~/.bashrc
conda activate unlearn-ft

cd path/to/projects/concept_prune || exit
targets=("Pablo Picasso")

for target in "${targets[@]}"
do
    python wanda.py \
    --base_config_path \
    path/to/configs/styles/img/sd-2-1_coco.yaml \
    --ckpt_path \
    path/to/sd-2-1_coco_aptp_both_512/aptp_coco_82_both_sd/checkpoint-20000/ \
    --target \
    "$target"

    python remove_neurons.py \
     --base_config_path \
    path/to/projects/diffusion_pruning/configs/styles/img/sd-2-1_coco.yaml \
    --ckpt_path \
    path/to/scripts/baselines/sd-2-1_coco_aptp_both_512/aptp_coco_82_both_sd/checkpoint-20000/ \
    --target \
    "$target"

    python save_union_over_time.py \
     --base_config_path \
    path/to//projects/diffusion_pruning/configs/styles/img/sd-2-1_coco.yaml \
    --ckpt_path \
    path/to/sd-2-1_coco_aptp_both_512/aptp_coco_82_both_sd/checkpoint-20000/ \
    --target \
    "$target" \
    --select_ratio \
    0.0
done

