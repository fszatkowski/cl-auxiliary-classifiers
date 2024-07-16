#!/bin/bash

#SBATCH --time=48:00:00   # walltime
#SBATCH --ntasks=4   # number of processor cores (i.e. tasks)
#SBATCH --gpus=1

#set -e

num_tasks=$1
seed=$2
num_exemplars=$3
lamb=$4
lamb_a=$5
ic_config=$6

eval "$(conda shell.bash hook)"
conda activate FACIL

n_epochs=200
tag="cifar100x${num_tasks}"
approach='ancl'

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
    --taskwise-kd \
    --lamb ${lamb} \
    --lamb-a ${lamb_a} \
    --results-path ./results/CIFAR100x${num_tasks}/${approach}_tw_ex_${num_exemplars}_lamb_${lamb}_lamb_a_${lamb_a}_${ic_config}/seed${seed} \
    --log disk wandb \
    --exp-name ${tag} \
    --save-models \
    --tags ${tag}

