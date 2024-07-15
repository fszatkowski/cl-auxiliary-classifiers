#!/bin/bash

#SBATCH --time=48:00:00   # walltime
#SBATCH --ntasks=4   # number of processor cores (i.e. tasks)
#SBATCH --gpus=1

#set -e

eval "$(conda shell.bash hook)"
conda activate FACIL

num_tasks=5
n_epochs=200
tag="imagenet100x5"
approach='finetuning'
num_exemplars=2000

for seed in 0; do
  python src/main_incremental.py \
    --gpu 0 \
    --seed ${seed} \
    --network resnet18 \
    --ic-layers layer1.0.relu layer1.0 layer1.1.relu layer1.1 layer2.0.relu layer2.0 layer2.1.relu layer2.1 layer3.0.relu layer3.0 layer3.1.relu layer3.1 layer4.0.relu layer4.0 layer4.1.relu \
    --ic-type standard_conv standard_conv standard_conv standard_conv standard_conv standard_conv standard_conv standard_conv standard_conv standard_conv standard_conv standard_conv adaptive_standard_conv adaptive_standard_conv adaptive_standard_conv \
    --ic-weighting proportional \
    --input-size 3 224 224 \
    --datasets imagenet_subset_kaggle \
    --num-tasks ${num_tasks} \
    --num-exemplars ${num_exemplars} \
    --use-test-as-val \
    --nepochs ${n_epochs} \
    --batch-size 128 \
    --lr 0.1 \
    --approach ${approach} \
    --log disk wandb \
    --results-path ./results/ImageNet100x${num_tasks}/${approach}_ex${num_exemplars}_ee_dense/seed${seed} \
    --exp-name ee_${tag}_dense \
    --save-models \
    --tags ${tag}
done
