#!/bin/bash
#SBATCH --job-name=aptp_coco_82_both_sd_bilevel_monet
#SBATCH --output=./logs/out-%x-%j.log
#SBATCH --error=./logs/err-%x-%j.log
#SBATCH --nodes=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:h100-sxm:4
#SBATCH --ntasks-per-node=1
#SBATCH --time=72:00:00
#SBATCH --qos=high
#SBATCH --mem=128G

set -x -e
# log the sbatch environment
echo "start time: $(date)"
echo "SLURM_JOBID="$SLURM_JOBID
echo "SLURM_JOB_NODELIST"=$SLURM_JOB_NODELIST
echo "SLURM_JOB_PARTITION"=$SLURM_JOB_PARTITION
echo "SLURM_NNODES"=$SLURM_NNODES
echo "SLURM_GPUS_ON_NODE"=$SLURM_GPUS_ON_NODE
echo "SLURM_SUBMIT_DIR"s=$SLURM_SUBMIT_DIR

export WANDB_CACHE_DIR=path/to/wandb

# Training setup
MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
MASTER_PORT=5562
NNODES=$SLURM_NNODES
NODE_RANK=$SLURM_PROCID
WORLD_SIZE=$(($SLURM_GPUS_ON_NODE * $NNODES))

echo "MASTER_ADDR"=$MASTER_ADDR
echo "NNODES"=$NNODES
echo "NODE_RANK"=$NODE_RANK

export NCCL_DEBUG=INFO
#export NCCL_IB_DISABLE=1

source ~/.bashrc
conda activate unlearn-ft
cd path/to/unlearn-ft/scripts/aptp || exit

CMD=" \
  bilevel_finetune.py \
  --base_config_path \
  path/to/unlearn-ft/configs/baselines/sd-2-1_coco_aptp_both_512_bilevel.yaml \
  --cache_dir \
  /path/to/huggingface/ \
  --wandb_run_name \
  aptp_coco_82_both_sd_bilevel_512_monet \
  --pruning_ckpt_dir \
  path/to/unlearn-ft/ckpts/82/ \
  --expert_id \
  5 \
"
LAUNCHER="accelerate launch \
    --multi_gpu \
    --num_machines $NNODES \
    --num_processes $WORLD_SIZE \
    --main_process_ip "$MASTER_ADDR" \
    --main_process_port $MASTER_PORT \
    --machine_rank \$SLURM_PROCID \
    --role $SLURMD_NODENAME: \
    --rdzv_conf rdzv_backend=c10d \
    --max_restarts 0 \
    --tee 3 \
"
SRUN_ARGS=" \
    --wait=60 \
    --kill-on-bad-exit=1 \
"
srun $SRUN_ARGS --jobid $SLURM_JOB_ID bash -c "$LAUNCHER $CMD" 2>&1
echo "END TIME: $(date)"
