#!/bin/bash

#SBATCH --time=48:00:00   # walltime
#SBATCH --ntasks=4   # number of processor cores (i.e. tasks)
#SBATCH --gpus=1

#set -e

num_tasks=$1
seed=$2
num_exemplars=$3
ic_config=$4

eval "$(conda shell.bash hook)"
conda activate FACIL

n_epochs=200
tag="cifar100x${num_tasks}"
approach='gdumb'

python src/main_incremental.py \
    --gpu 0 \
    --seed ${seed} \
    --network resnet32 \
    --ic-config ${ic_config} \
    --datasets cifar100_icarl \
    --num-tasks ${num_tasks} \
    --num-exemplars ${num_exemplars} \
    --use-test-as-val \
    --nepochs ${n_epochs} \
    --batch-size 128 \
    --lr 0.1 \
    --approach ${approach} \
    --log disk \
    --results-path ./results/CIFAR100x${num_tasks}/${approach}_ex${num_exemplars}_${ic_config}/seed${seed} \
    --tags ${tag}
