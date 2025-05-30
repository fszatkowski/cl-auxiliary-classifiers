#!/bin/bash

#SBATCH --time=48:00:00   # walltime
#SBATCH --ntasks=4   # number of processor cores (i.e. tasks)
#SBATCH --gpus=1

#set -e

num_tasks=$1
seed=$2
num_exemplars=$3
const=$4
ro=$5
ic_config=$6

eval "$(conda shell.bash hook)"
conda activate FACIL

n_epochs=100
tag="imagenet100x${num_tasks}"
approach='lode'

python src/main_incremental.py \
    --gpu 0 \
    --seed ${seed} \
    --network vit_b_16 \
    --scheduler-name cosine \
    --ic-config ${ic_config} \
    --datasets imagenet_subset_kaggle \
    --num-tasks ${num_tasks} \
    --num-exemplars ${num_exemplars} \
    --use-test-as-val \
    --nepochs ${n_epochs} \
    --batch-size 64 \
    --lr 0.01 \
    --approach ${approach} \
    --const ${const} \
    --ro ${ro} \
    --results-path /data/SHARE/fszatkowski/results/ImageNet100x${num_tasks}_vit/${approach}_ex${num_exemplars}_c_${const}_ro${ro}_${ic_config}/seed${seed} \
    --log disk \
    --tags ${tag}

