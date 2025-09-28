#!/bin/bash

#SBATCH --time=48:00:00   # walltime
#SBATCH --ntasks=8   # number of processor cores (i.e. tasks)
#SBATCH --gpus=1

#set -e
num_tasks=$1
seed=$2
num_exemplars=$3
lamb=$4

eval "$(conda shell.bash hook)"
conda activate FACIL

n_epochs=100
tag="imagenet100x${num_tasks}"
approach='bic'

python src/main_incremental.py \
    --gpu 0 \
    --seed ${seed} \
    --network resnet18 \
    --datasets imagenet_subset_kaggle \
    --num-tasks ${num_tasks} \
    --num-exemplars ${num_exemplars} \
    --use-test-as-val \
    --nepochs ${n_epochs} \
    --scheduler-name cosine \
    --num-bias-epochs ${n_epochs} \
    --batch-size 128 \
    --lr 0.1 \
    --approach ${approach} \
    --lamb ${lamb} \
    --log disk \
    --results-path /data/SHARE/fszatkowski/results_recomputed_in/ImageNet100x${num_tasks}_rn18/${approach}_ex${num_exemplars}_lamb_${lamb}/seed${seed} \
    --tags ${tag}
