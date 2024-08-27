#!/bin/bash

#SBATCH --time=48:00:00   # walltime
#SBATCH --ntasks=4   # number of processor cores (i.e. tasks)
#SBATCH --gpus=1

#set -e

num_tasks=$1
seed=$2
num_exemplars=$3
lamb_pod_spatial=$4
lamb_pod_flat=$5
nepochs_finetuning=$6
lr_finetuning_factor=$7
nb_proxy=$8
ic_config=$9

eval "$(conda shell.bash hook)"
conda activate FACIL

n_epochs=200
tag="cifar100x${num_tasks}"
approach='podnet'

python src/main_incremental.py \
    --gpu 0 \
    --num-workers 0 \
    --seed ${seed} \
    --network resnet32 \
    --ic-config ${ic_config} \
    --datasets cifar100_icarl \
    --num-workers 0 \
    --num-tasks ${num_tasks} --nc-first-task 50 \
    --num-exemplars ${num_exemplars} \
    --use-test-as-val \
    --nepochs ${n_epochs} \
    --batch-size 128 \
    --lr 0.1 \
    --approach ${approach} \
    --lamb-pod-spatial ${lamb_pod_spatial} \
    --lamb-pod-flat ${lamb_pod_flat} \
    --nepochs-finetuning ${nepochs_finetuning} \
    --lr-finetuning-factor ${lr_finetuning_factor} \
    --nb-proxy ${nb_proxy} \
    --log disk wandb \
    --results-path ./results/CIFAR100x${num_tasks}/${approach}_ex${num_exemplars}_lamb_spatial_${lamb_pod_spatial}_lamb_flat_${lamb_pod_flat}_ft_epochs_${nepochs_finetuning}_lr_${lr_finetuning_factor}_nb_proxy_${nb_proxy}_${ic_config}/seed${seed} \
    --tags ${tag}
