#!/bin/bash

#SBATCH --time=48:00:00   # walltime
#SBATCH --ntasks=4   # number of processor cores (i.e. tasks)
#SBATCH --gpus=1

#set -e

num_tasks=$1
seed=$2
num_exemplars=$3
alpha=${4:-0.5}
beta=${5:-0.5}

eval "$(conda shell.bash hook)"
conda activate FACIL

n_epochs=200
tag="cifar100x${num_tasks}"
approach='der++'

results_path=./results_wideresnet/CIFAR100x${num_tasks}/${approach}_ex${num_exemplars}_alpha_${alpha}_beta_${beta}/seed${seed}
# Exit script if results path exists
if [ -d "${results_path}" ]; then
    echo "Results path ${results_path} already exists. Skipping the computation."
    exit 1
fi

python src/main_incremental.py \
    --gpu 0 \
    --num-workers 0 \
    --seed ${seed} \
    --network wide_resnet16_2 \
    --datasets cifar100_icarl \
    --num-tasks ${num_tasks} \
    --num-exemplars ${num_exemplars} \
    --use-test-as-val \
    --nepochs ${n_epochs} \
    --batch-size 128 \
    --lr 0.1 \
    --approach ${approach} \
    --alpha ${alpha} \
    --beta ${beta} \
    --log disk \
    --results-path ${results_path} \
    --tags ${tag}
