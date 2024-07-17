#!/bin/bash

#SBATCH --time=48:00:00   # walltime
#SBATCH --ntasks=4   # number of processor cores (i.e. tasks)
#SBATCH --gpus=1

#set -e

num_tasks=$1
seed=$2
lamb=$3
ic_config=$4

eval "$(conda shell.bash hook)"
conda activate FACIL

n_epochs=200
tag="imagenet100x${num_tasks}"
approach='lwf'

python src/main_incremental.py \
    --gpu 0 \
    --seed ${seed} \
    --network resnet18 \
    --ic-config ${ic_config} \
    --datasets imagenet_subset_kaggle \
    --num-tasks ${num_tasks} \
    --num-exemplars 0 \
    --use-test-as-val \
    --nepochs ${n_epochs} \
    --batch-size 128 \
    --lr 0.1 \
    --approach ${approach} \
    --taskwise-kd \
    --lamb ${lamb} \
    --results-path ./results/ImageNet100x${num_tasks}/${approach}_rn18_tw_lamb_${lamb}_${ic_config}/seed${seed} \
    --log disk wandb \
    --exp-name ${tag} \
    --save-models \
    --tags ${tag}

