#!/bin/bash

set -e

for seed in 0 1 2; do
    for ic_config in cifar100_resnet32_sdn cifar100_resnet32_uniform_1_only cifar100_resnet32_uniform_2_only cifar100_resnet32_uniform_3_only cifar100_resnet32_uniform_4_only cifar100_resnet32_uniform_5_only cifar100_resnet32_uniform_6_only; do
        for num_tasks in 5 10; do
            # BiC
            lamb=2
            sbatch -A plgimprmoe-gpu-a100 -p plgrid-gpu-a100 scripts/templates/cifar100_single_ic/bic.sh ${num_tasks} ${seed} ${num_exemplars} ${lamb} ${ic_config}

            # FT
            num_exemplars=0
            sbatch -A plgimprmoe-gpu-a100 -p plgrid-gpu-a100 scripts/templates/cifar100_single_ic/ft.sh ${num_tasks} ${seed} ${num_exemplars} ${ic_config}

            # FT+Ex
            num_exemplars=2000
            sbatch -A plgimprmoe-gpu-a100 -p plgrid-gpu-a100 scripts/templates/cifar100_single_ic/ft.sh ${num_tasks} ${seed} ${num_exemplars} ${ic_config}
        done
        # LWF
        lamb=0.5
        sbatch -A plgimprmoe-gpu-a100 -p plgrid-gpu-a100 scripts/templates/cifar100_single_ic/lwf.sh 10 ${seed} ${lamb} ${ic_config}
        lamb=1.0
        sbatch -A plgimprmoe-gpu-a100 -p plgrid-gpu-a100 scripts/templates/cifar100_single_ic/lwf.sh 5 ${seed} ${lamb} ${ic_config}

    done
done