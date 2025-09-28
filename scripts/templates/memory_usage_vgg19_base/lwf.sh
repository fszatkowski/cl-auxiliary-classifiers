#!/bin/bash

#SBATCH --time=48:00:00   # walltime
#SBATCH --ntasks=4   # number of processor cores (i.e. tasks)
#SBATCH --gpus=1

#set -e

num_tasks=$1
seed=$2
lamb=$3

eval "$(conda shell.bash hook)"
conda activate FACIL

n_epochs=10
tag="cifar100x${num_tasks}"
approach='lwf'

python src/main_incremental.py \
    --gpu 0 \
    --seed ${seed} \
    --network vgg19_bn_cifar \
    --datasets cifar100_icarl \
    --num-tasks ${num_tasks} \
    --num-exemplars 0 \
    --use-test-as-val \
    --nepochs ${n_epochs} \
    --batch-size 128 \
    --lr 0.1 \
    --approach ${approach} \
    --taskwise-kd \
    --lamb ${lamb} \
    --results-path ./results_memory_vgg19/CIFAR100x${num_tasks}/${approach}_tw_lamb_${lamb}/seed${seed} \
    --log disk \
    --tags ${tag}

