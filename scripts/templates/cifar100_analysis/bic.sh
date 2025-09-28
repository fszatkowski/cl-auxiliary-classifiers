#!/bin/bash

#SBATCH --time=48:00:00   # walltime
#SBATCH --ntasks=4   # number of processor cores (i.e. tasks)
#SBATCH --gpus=1

#set -e

num_tasks=$1
seed=$2
num_exemplars=$3
lamb=$4

eval "$(conda shell.bash hook)"
conda activate FACIL

n_epochs=200
tag="ablation_cifar100x${num_tasks}"
approach='bic'

python src/main_incremental.py \
    --gpu 0 \
    --seed ${seed} \
    --network resnet32 \
    --datasets cifar100_icarl \
    --num-tasks ${num_tasks} \
    --num-exemplars ${num_exemplars} \
    --use-test-as-val \
    --nepochs ${n_epochs} \
    --batch-size 128 \
    --lr 0.1 \
    --approach ${approach} \
    --lamb ${lamb} \
    --log disk wandb \
    --results-path ./results_analysis/CIFAR100x${num_tasks}/${approach}_num_exemplars_${num_exemplars}_lamb_${lamb}/seed${seed} \
    --save-test-features \
    --save-test-logits \
    --tags ${tag}
