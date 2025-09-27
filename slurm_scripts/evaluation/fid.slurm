#!/bin/bash
#SBATCH --job-name=fid
#SBATCH --output=./logs/out-%x-%j.log
#SBATCH --error=./logs/err-%x-%j.log
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:rtx6000ada:1
#SBATCH --ntasks-per-node=1
#SBATCH --time=72:00:00
#SBATCH --qos=high
#SBATCH --mem=64G


set -x -e

source ~/.bashrc
conda activate unlearn-ft
cd path/to/diffusion_pruning/scripts/metrics || exit

#checkpoints=(2000 4000 6000 8000 10000)
#checkpoints=(20000)
#ckpt=${checkpoints[$SLURM_ARRAY_TASK_ID]}
gen_dir=$1
#gen_dir="${gen_dir}/checkpoint-${ckpt}/coco_fid_images"
result_dir="path/to/results/distillation"

srun bash -c "python3 fid.py --gen_dir $gen_dir --dataset coco-val2017 --mode legacy_pytorch --result_dir $result_dir" 2>&1
