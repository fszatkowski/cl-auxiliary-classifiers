#!/bin/bash

set -e

for seed in 0 1 2; do
    #     for num_tasks in 5 10; do
    for num_tasks in 1; do
        ic_config=cifar100_resnet32_sdn
        # FT
        num_exemplars=0
        sbatch -A plgdynamic2-gpu-a100 -p plgrid-gpu-a100 scripts/templates/cifar100_analysis/ft_ee.sh ${num_tasks} ${seed} ${num_exemplars} ${ic_config}

        ic_config=cifar100_resnet32_sdn_detach
        # FT
        num_exemplars=0
        sbatch -A plgdynamic2-gpu-a100 -p plgrid-gpu-a100 scripts/templates/cifar100_analysis/ft_ee.sh ${num_tasks} ${seed} ${num_exemplars} ${ic_config}

        #         # FT
        #         num_exemplars=0
        #         sbatch -A plgdynamic2-gpu-a100 -p plgrid-gpu-a100 scripts/templates/cifar100_analysis/ft.sh ${num_tasks} ${seed} ${num_exemplars}
        #
        #         # FT+Ex
        #         num_exemplars=2000
        #         sbatch -A plgdynamic2-gpu-a100 -p plgrid-gpu-a100 scripts/templates/cifar100_analysis/ft.sh ${num_tasks} ${seed} ${num_exemplars}
        #
        #         ic_config=cifar100_resnet32_sdn
        #         # FT
        #         num_exemplars=0
        #         sbatch -A plgdynamic2-gpu-a100 -p plgrid-gpu-a100 scripts/templates/cifar100_analysis/ft_ee.sh ${num_tasks} ${seed} ${num_exemplars} ${ic_config}
        #
        #         # FT+Ex
        #         num_exemplars=2000
        #         sbatch -A plgdynamic2-gpu-a100 -p plgrid-gpu-a100 scripts/templates/cifar100_analysis/ft_ee.sh ${num_tasks} ${seed} ${num_exemplars} ${ic_config}
        #
        #         # Joint
        #         sbatch -A plgdynamic2-gpu-a100 -p plgrid-gpu-a100 scripts/templates/cifar100_analysis/joint_ee.sh ${num_tasks} ${seed} ${ic_config}
        #
        #         ic_config=cifar100_resnet32_sdn_detach
        #         # FT
        #         num_exemplars=0
        #         sbatch -A plgdynamic2-gpu-a100 -p plgrid-gpu-a100 scripts/templates/cifar100_analysis/ft_ee.sh ${num_tasks} ${seed} ${num_exemplars} ${ic_config}
        #
        #         # FT+Ex
        #         num_exemplars=2000
        #         sbatch -A plgdynamic2-gpu-a100 -p plgrid-gpu-a100 scripts/templates/cifar100_analysis/ft_ee.sh ${num_tasks} ${seed} ${num_exemplars} ${ic_config}
        #
        #         # Joint
        #         sbatch -A plgdynamic2-gpu-a100 -p plgrid-gpu-a100 scripts/templates/cifar100_analysis/joint_ee.sh ${num_tasks} ${seed} ${ic_config}
        #         # LwF
        #         num_exemplars=0
        #         lamb=0.5
        #         sbatch -A plgdynamic2-gpu-a100 -p plgrid-gpu-a100 scripts/templates/cifar100_analysis/lwf.sh ${num_tasks} ${seed} ${lamb}
        #         num_exemplars=0
        #         lamb=1.0
        #         sbatch -A plgdynamic2-gpu-a100 -p plgrid-gpu-a100 scripts/templates/cifar100_analysis/lwf.sh ${num_tasks} ${seed} ${lamb}
        #
        #         # BiC
        #         num_exemplars=2000
        #         lamb=2
        #         sbatch -A plgdynamic2-gpu-a100 -p plgrid-gpu-a100 scripts/templates/cifar100_analysis/bic.sh ${num_tasks} ${seed} ${num_exemplars} ${lamb}
        #         num_exemplars=2000
        #         lamb=3
        #         sbatch -A plgdynamic2-gpu-a100 -p plgrid-gpu-a100 scripts/templates/cifar100_analysis/bic.sh ${num_tasks} ${seed} ${num_exemplars} ${lamb}
        #
        #         ic_config=cifar100_resnet32_sdn
        #         # LwF
        #         num_exemplars=0
        #         lamb=0.5
        #         sbatch -A plgdynamic2-gpu-a100 -p plgrid-gpu-a100 scripts/templates/cifar100_analysis/lwf_ee.sh ${num_tasks} ${seed} ${lamb} ${ic_config}
        #         num_exemplars=0
        #         lamb=1.0
        #         sbatch -A plgdynamic2-gpu-a100 -p plgrid-gpu-a100 scripts/templates/cifar100_analysis/lwf_ee.sh ${num_tasks} ${seed} ${lamb} ${ic_config}
        #
        #         # BiC
        #         num_exemplars=2000
        #         lamb=2
        #         sbatch -A plgdynamic2-gpu-a100 -p plgrid-gpu-a100 scripts/templates/cifar100_analysis/bic_ee.sh ${num_tasks} ${seed} ${num_exemplars} ${lamb} ${ic_config}
        #         num_exemplars=2000
        #         lamb=3
        #         sbatch -A plgdynamic2-gpu-a100 -p plgrid-gpu-a100 scripts/templates/cifar100_analysis/bic_ee.sh ${num_tasks} ${seed} ${num_exemplars} ${lamb} ${ic_config}
        #
        #
        #         ic_config=cifar100_resnet32_sdn_detach
        #         # LwF
        #         num_exemplars=0
        #         lamb=0.5
        #         sbatch -A plgdynamic2-gpu-a100 -p plgrid-gpu-a100 scripts/templates/cifar100_analysis/lwf_ee.sh ${num_tasks} ${seed} ${lamb} ${ic_config}
        #         num_exemplars=0
        #         lamb=1.0
        #         sbatch -A plgdynamic2-gpu-a100 -p plgrid-gpu-a100 scripts/templates/cifar100_analysis/lwf_ee.sh ${num_tasks} ${seed} ${lamb} ${ic_config}
        #
        #         # BiC
        #         num_exemplars=2000
        #         lamb=2
        #         sbatch -A plgdynamic2-gpu-a100 -p plgrid-gpu-a100 scripts/templates/cifar100_analysis/bic_ee.sh ${num_tasks} ${seed} ${num_exemplars} ${lamb} ${ic_config}
        #         num_exemplars=2000
        #         lamb=3
        #         sbatch -A plgdynamic2-gpu-a100 -p plgrid-gpu-a100 scripts/templates/cifar100_analysis/bic_ee.sh ${num_tasks} ${seed} ${num_exemplars} ${lamb} ${ic_config}
    done
done