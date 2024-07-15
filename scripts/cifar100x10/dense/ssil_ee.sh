#!/bin/bash

#SBATCH --time=48:00:00   # walltime
#SBATCH --ntasks=4   # number of processor cores (i.e. tasks)
#SBATCH --gpus=1

#set -e

eval "$(conda shell.bash hook)"
conda activate FACIL

num_tasks=5
n_epochs=200
tag="cifar100x${num_tasks}"
approach='ssil'
ic_config='cifar100_resnet32_dense'

lamb=0.25
num_exemplars=2000

for seed in 0 1 2; do
  python src/main_incremental.py \
    --gpu 0 \
    --num-workers 0 \
    --seed ${seed} \
    --network resnet32 \
    --ic-config 'cifar100_resnet32_sdn' \
    --datasets cifar100_icarl \
    --num-workers 0 \
    --num-tasks ${num_tasks} \
    --num-exemplars ${num_exemplars} \
    --use-test-as-val \
    --nepochs ${n_epochs} \
    --batch-size 128 \
    --lr 0.1 \
    --approach ${approach} \
    --lamb ${lamb} \
    --log disk wandb \
    --results-path ./results/CIFAR100x${num_tasks}/${approach}_ex${num_exemplars}_lamb_${lamb}_${ic_config}/seed${seed} \
    --exp-name ${tag}_${ic_config} \
    --save-models \
    --tags ${tag}
done
