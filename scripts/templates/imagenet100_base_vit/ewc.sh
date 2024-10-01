#!/bin/bash

#SBATCH --time=48:00:00   # walltime
#SBATCH --ntasks=4   # number of processor cores (i.e. tasks)
#SBATCH --gpus=1

#set -e

num_tasks=$1
seed=$2
lamb=$3
alpha=$4

eval "$(conda shell.bash hook)"
conda activate FACIL

n_epochs=100
tag="imagenet100x${num_tasks}"
approach='ewc'


python src/main_incremental.py \
    --gpu 0 \
    --seed ${seed} \
    --network vit_b_16 \
    --scheduler-name cosine \
    --datasets imagenet_subset_kaggle \
    --num-tasks ${num_tasks} \
    --num-exemplars 0 \
    --use-test-as-val \
    --nepochs ${n_epochs} \
    --batch-size 64 \
    --lr 0.01 \
    --approach ${approach} \
    --lamb ${lamb} \
    --alpha ${alpha} \
    --log disk \
    --results-path /data/SHARE/fszatkowski/results/ImageNet100x${num_tasks}_vit/${approach}_lamb_${lamb}_alpha_${alpha}/seed${seed} \
    --tags ${tag}
