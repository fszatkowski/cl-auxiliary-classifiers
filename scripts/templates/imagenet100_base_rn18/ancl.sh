#!/bin/bash

#SBATCH --time=48:00:00   # walltime
#SBATCH --ntasks=8   # number of processor cores (i.e. tasks)
#SBATCH --gpus=1

#set -e

num_tasks=$1
seed=$2
num_exemplars=$3
lamb=$4
lamb_a=$5

eval "$(conda shell.bash hook)"
conda activate FACIL

n_epochs=100
tag="imagenet100x${num_tasks}"
approach='ancl'

python src/main_incremental.py \
    --gpu 0 \
    --seed ${seed} \
    --network resnet18 \
    --datasets imagenet_subset_kaggle \
    --num-tasks ${num_tasks} \
    --num-exemplars ${num_exemplars} \
    --use-test-as-val \
    --scheduler-name cosine \
    --nepochs ${n_epochs} \
    --batch-size 128 \
    --lr 0.1 \
    --approach ${approach} \
    --taskwise-kd \
    --lamb ${lamb} \
    --lamb-a ${lamb_a} \
    --results-path /data/SHARE/fszatkowski/results/ImageNet100x${num_tasks}_rn18/${approach}_tw_ex_${num_exemplars}_lamb_${lamb}_lamb_a_${lamb_a}/seed${seed} \
    --log disk wandb \
    --tags ${tag}
