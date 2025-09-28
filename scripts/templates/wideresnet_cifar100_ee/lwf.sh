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
tag="cifar100x${num_tasks}"
approach='lwf'

results_path=./results_wideresnet/CIFAR100x${num_tasks}/${approach}_tw_lamb_${lamb}_${ic_config}/seed${seed}
# Exit script if results path exists
if [ -d "${results_path}" ]; then
    echo "Results path ${results_path} already exists. Skipping the computation."
    exit 1
fi

python src/main_incremental.py \
    --gpu 0 \
    --seed ${seed} \
    --network wide_resnet16_2 \
    --ic-config ${ic_config} \
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
    --results-path ${results_path} \
    --log disk \
    --tags ${tag}

