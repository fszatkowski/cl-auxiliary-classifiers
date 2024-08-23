#!/bin/bash

set -e

for seed in 0 1 2; do
    for num_tasks in 5 10; do
        # FT
        num_exemplars=0
        sbatch -A plgdynamic2-gpu-a100 -p plgrid-gpu-a100 scripts/templates/cifar100_analysis/ft.sh ${num_tasks} ${seed} ${num_exemplars}

        # FT+Ex
        num_exemplars=2000
        sbatch -A plgdynamic2-gpu-a100 -p plgrid-gpu-a100 scripts/templates/cifar100_analysis/ft.sh ${num_tasks} ${seed} ${num_exemplars}

        ic_config=cifar100_resnet32_sdn
        # FT
        num_exemplars=0
        sbatch -A plgdynamic2-gpu-a100 -p plgrid-gpu-a100 scripts/templates/cifar100_analysis/ft_ee.sh ${num_tasks} ${seed} ${num_exemplars} ${ic_config}

        # FT+Ex
        num_exemplars=2000
        sbatch -A plgdynamic2-gpu-a100 -p plgrid-gpu-a100 scripts/templates/cifar100_analysis/ft_ee.sh ${num_tasks} ${seed} ${num_exemplars} ${ic_config}

        # Joint
        sbatch -A plgdynamic2-gpu-a100 -p plgrid-gpu-a100 scripts/templates/cifar100_analysis/joint_ee.sh ${num_tasks} ${seed} ${ic_config}

        ic_config=cifar100_resnet32_sdn_detach
        # FT
        num_exemplars=0
        sbatch -A plgdynamic2-gpu-a100 -p plgrid-gpu-a100 scripts/templates/cifar100_analysis/ft_ee.sh ${num_tasks} ${seed} ${num_exemplars} ${ic_config}

        # FT+Ex
        num_exemplars=2000
        sbatch -A plgdynamic2-gpu-a100 -p plgrid-gpu-a100 scripts/templates/cifar100_analysis/ft_ee.sh ${num_tasks} ${seed} ${num_exemplars} ${ic_config}

        # Joint
        sbatch -A plgdynamic2-gpu-a100 -p plgrid-gpu-a100 scripts/templates/cifar100_analysis/joint_ee.sh ${num_tasks} ${seed} ${ic_config}
    done
done