#!/bin/bash

set -e

for seed in 0; do
    for num_tasks in 5 10; do
        # ANCL
        num_exemplars=0
        lamb=1.0
        lamb_a=1.0
        sbatch -A plgimprmoe-gpu-a100 -p plgrid-gpu-a100 scripts/templates/cifar100_base/ancl.sh ${num_tasks} ${seed} ${num_exemplars} ${lamb} ${lamb_a}
        lamb=2.0
        lamb_a=2.0
        sbatch -A plgimprmoe-gpu-a100 -p plgrid-gpu-a100 scripts/templates/cifar100_base/ancl.sh ${num_tasks} ${seed} ${num_exemplars} ${lamb} ${lamb_a}
        lamb=0.5
        lamb_a=0.5
        sbatch -A plgimprmoe-gpu-a100 -p plgrid-gpu-a100 scripts/templates/cifar100_base/ancl.sh ${num_tasks} ${seed} ${num_exemplars} ${lamb} ${lamb_a}
        lamb=1.0
        lamb_a=2.0
        sbatch -A plgimprmoe-gpu-a100 -p plgrid-gpu-a100 scripts/templates/cifar100_base/ancl.sh ${num_tasks} ${seed} ${num_exemplars} ${lamb} ${lamb_a}
        lamb=2.0
        lamb_a=1.0
        sbatch -A plgimprmoe-gpu-a100 -p plgrid-gpu-a100 scripts/templates/cifar100_base/ancl.sh ${num_tasks} ${seed} ${num_exemplars} ${lamb} ${lamb_a}
        lamb=1.0
        lamb_a=0.5
        sbatch -A plgimprmoe-gpu-a100 -p plgrid-gpu-a100 scripts/templates/cifar100_base/ancl.sh ${num_tasks} ${seed} ${num_exemplars} ${lamb} ${lamb_a}
        lamb=0.5
        lamb_a=1.0
        sbatch -A plgimprmoe-gpu-a100 -p plgrid-gpu-a100 scripts/templates/cifar100_base/ancl.sh ${num_tasks} ${seed} ${num_exemplars} ${lamb} ${lamb_a}

        # ANCL+Ex
        num_exemplars=2000
        lamb=1.0
        lamb_a=1.0
        sbatch -A plgimprmoe-gpu-a100 -p plgrid-gpu-a100 scripts/templates/cifar100_base/ancl.sh ${num_tasks} ${seed} ${num_exemplars} ${lamb} ${lamb_a}
        lamb=2.0
        lamb_a=2.0
        sbatch -A plgimprmoe-gpu-a100 -p plgrid-gpu-a100 scripts/templates/cifar100_base/ancl.sh ${num_tasks} ${seed} ${num_exemplars} ${lamb} ${lamb_a}
        lamb=0.5
        lamb_a=0.5
        sbatch -A plgimprmoe-gpu-a100 -p plgrid-gpu-a100 scripts/templates/cifar100_base/ancl.sh ${num_tasks} ${seed} ${num_exemplars} ${lamb} ${lamb_a}
        lamb=1.0
        lamb_a=2.0
        sbatch -A plgimprmoe-gpu-a100 -p plgrid-gpu-a100 scripts/templates/cifar100_base/ancl.sh ${num_tasks} ${seed} ${num_exemplars} ${lamb} ${lamb_a}
        lamb=2.0
        lamb_a=1.0
        sbatch -A plgimprmoe-gpu-a100 -p plgrid-gpu-a100 scripts/templates/cifar100_base/ancl.sh ${num_tasks} ${seed} ${num_exemplars} ${lamb} ${lamb_a}
        lamb=1.0
        lamb_a=0.5
        sbatch -A plgimprmoe-gpu-a100 -p plgrid-gpu-a100 scripts/templates/cifar100_base/ancl.sh ${num_tasks} ${seed} ${num_exemplars} ${lamb} ${lamb_a}
        lamb=0.5
        lamb_a=1.0
        sbatch -A plgimprmoe-gpu-a100 -p plgrid-gpu-a100 scripts/templates/cifar100_base/ancl.sh ${num_tasks} ${seed} ${num_exemplars} ${lamb} ${lamb_a}

        # BiC
        num_exemplars=2000
        lamb=-1
        sbatch -A plgimprmoe-gpu-a100 -p plgrid-gpu-a100 scripts/templates/cifar100_base/bic.sh ${num_tasks} ${seed} ${num_exemplars} ${lamb}
        lamb=0.5
        sbatch -A plgimprmoe-gpu-a100 -p plgrid-gpu-a100 scripts/templates/cifar100_base/bic.sh ${num_tasks} ${seed} ${num_exemplars} ${lamb}
        lamb=1
        sbatch -A plgimprmoe-gpu-a100 -p plgrid-gpu-a100 scripts/templates/cifar100_base/bic.sh ${num_tasks} ${seed} ${num_exemplars} ${lamb}
        lamb=2
        sbatch -A plgimprmoe-gpu-a100 -p plgrid-gpu-a100 scripts/templates/cifar100_base/bic.sh ${num_tasks} ${seed} ${num_exemplars} ${lamb}
        lamb=3
        sbatch -A plgimprmoe-gpu-a100 -p plgrid-gpu-a100 scripts/templates/cifar100_base/bic.sh ${num_tasks} ${seed} ${num_exemplars} ${lamb}
        lamb=4
        sbatch -A plgimprmoe-gpu-a100 -p plgrid-gpu-a100 scripts/templates/cifar100_base/bic.sh ${num_tasks} ${seed} ${num_exemplars} ${lamb}
        lamb=5
        sbatch -A plgimprmoe-gpu-a100 -p plgrid-gpu-a100 scripts/templates/cifar100_base/bic.sh ${num_tasks} ${seed} ${num_exemplars} ${lamb}

        # EWC
        lamb=10000
        alpha=0.5
        sbatch -A plgimprmoe-gpu-a100 -p plgrid-gpu-a100 scripts/templates/cifar100_base/ewc.sh ${num_tasks} ${seed} ${lamb} ${alpha}

        # FT
        num_exemplars=0
        sbatch -A plgimprmoe-gpu-a100 -p plgrid-gpu-a100 scripts/templates/cifar100_base/ft.sh ${num_tasks} ${seed} ${num_exemplars}

        # FT+Ex
        num_exemplars=2000
        sbatch -A plgimprmoe-gpu-a100 -p plgrid-gpu-a100 scripts/templates/cifar100_base/ft.sh ${num_tasks} ${seed} ${num_exemplars}

        # GDumb
        num_exemplars=2000
        sbatch -A plgimprmoe-gpu-a100 -p plgrid-gpu-a100 scripts/templates/cifar100_base/gdumb.sh ${num_tasks} ${seed} ${num_exemplars}

        # iCaRL TODO

        # LWF
        lamb=0.25
        sbatch -A plgimprmoe-gpu-a100 -p plgrid-gpu-a100 scripts/templates/cifar100_base/lwf.sh ${num_tasks} ${seed} ${lamb}
        lamb=0.5
        sbatch -A plgimprmoe-gpu-a100 -p plgrid-gpu-a100 scripts/templates/cifar100_base/lwf.sh ${num_tasks} ${seed} ${lamb}
        lamb=0.75
        sbatch -A plgimprmoe-gpu-a100 -p plgrid-gpu-a100 scripts/templates/cifar100_base/lwf.sh ${num_tasks} ${seed} ${lamb}
        lamb=1.0
        sbatch -A plgimprmoe-gpu-a100 -p plgrid-gpu-a100 scripts/templates/cifar100_base/lwf.sh ${num_tasks} ${seed} ${lamb}
        lamb=1.5
        sbatch -A plgimprmoe-gpu-a100 -p plgrid-gpu-a100 scripts/templates/cifar100_base/lwf.sh ${num_tasks} ${seed} ${lamb}
        lamb=2.0
        sbatch -A plgimprmoe-gpu-a100 -p plgrid-gpu-a100 scripts/templates/cifar100_base/lwf.sh ${num_tasks} ${seed} ${lamb}

        # Joint
        sbatch -A plgimprmoe-gpu-a100 -p plgrid-gpu-a100 scripts/templates/cifar100_base/joint.sh ${num_tasks} ${seed}

        # SSIL
        num_exemplars=2000
        lamb=0.25
        sbatch -A plgimprmoe-gpu-a100 -p plgrid-gpu-a100 scripts/templates/cifar100_base/lwf.sh ${num_tasks} ${seed} ${num_exemplars} ${lamb}
        lamb=0.5
        sbatch -A plgimprmoe-gpu-a100 -p plgrid-gpu-a100 scripts/templates/cifar100_base/lwf.sh ${num_tasks} ${seed} ${num_exemplars} ${lamb}
        lamb=0.75
        sbatch -A plgimprmoe-gpu-a100 -p plgrid-gpu-a100 scripts/templates/cifar100_base/lwf.sh ${num_tasks} ${seed} ${num_exemplars} ${lamb}
        lamb=1.0
        sbatch -A plgimprmoe-gpu-a100 -p plgrid-gpu-a100 scripts/templates/cifar100_base/lwf.sh ${num_tasks} ${seed} ${num_exemplars} ${lamb}
        lamb=1.5
        sbatch -A plgimprmoe-gpu-a100 -p plgrid-gpu-a100 scripts/templates/cifar100_base/lwf.sh ${num_tasks} ${seed} ${num_exemplars} ${lamb}
        lamb=2.0
        sbatch -A plgimprmoe-gpu-a100 -p plgrid-gpu-a100 scripts/templates/cifar100_base/lwf.sh ${num_tasks} ${seed} ${num_exemplars} ${lamb}

    done
done