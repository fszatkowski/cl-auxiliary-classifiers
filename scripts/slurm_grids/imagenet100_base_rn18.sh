#!/bin/bash

set -e

for seed in 0 1 2; do
    for num_tasks in 5 10; do
        # ANCL
        num_exemplars=0
        lamb=1.0
        lamb_a=1.0
        sbatch <account_name> <partition> scripts/templates/imagenet100_base_rn18/ancl.sh ${num_tasks} ${seed} ${num_exemplars} ${lamb} ${lamb_a}
        lamb=1.0
        lamb_a=2.0
        sbatch <account_name> <partition> scripts/templates/imagenet100_base_rn18/ancl.sh ${num_tasks} ${seed} ${num_exemplars} ${lamb} ${lamb_a}

        # BiC
        num_exemplars=2000
        lamb=2
        sbatch <account_name> <partition> scripts/templates/imagenet100_base_rn18/bic.sh ${num_tasks} ${seed} ${num_exemplars} ${lamb}
        lamb=3
        sbatch <account_name> <partition> scripts/templates/imagenet100_base_rn18/bic.sh ${num_tasks} ${seed} ${num_exemplars} ${lamb}

        # EWC
        lamb=10000
        alpha=0.5
        sbatch <account_name> <partition> scripts/templates/imagenet100_base_rn18/ewc.sh ${num_tasks} ${seed} ${lamb} ${alpha}

        # ER
        num_exemplars=2000
        sbatch <account_name> <partition> scripts/templates/imagenet100_base_rn18/er.sh ${num_tasks} ${seed} ${num_exemplars}

        # FT
        num_exemplars=0
        sbatch <account_name> <partition> scripts/templates/imagenet100_base_rn18/ft.sh ${num_tasks} ${seed} ${num_exemplars}

        # FT+Ex
        num_exemplars=2000
        sbatch <account_name> <partition> scripts/templates/imagenet100_base_rn18/ft.sh ${num_tasks} ${seed} ${num_exemplars}

        # GDumb
        num_exemplars=2000
        sbatch <account_name> <partition> scripts/templates/imagenet100_base_rn18/gdumb.sh ${num_tasks} ${seed} ${num_exemplars}

        # LODE
        num_exemplars=2000
        const=1
        ro=0.1
        sbatch <account_name> <partition> scripts/templates/imagenet100_base_rn18/lode.sh ${num_tasks} ${seed} ${num_exemplars} ${const} ${ro}
        const=3
        ro=0.5
        sbatch <account_name> <partition> scripts/templates/imagenet100_base_rn18/lode.sh ${num_tasks} ${seed} ${num_exemplars} ${const} ${ro}

        # LWF
        lamb=0.5
        sbatch <account_name> <partition> scripts/templates/imagenet100_base_rn18/lwf.sh ${num_tasks} ${seed} ${lamb}
        lamb=1.0
        sbatch <account_name> <partition> scripts/templates/imagenet100_base_rn18/lwf.sh ${num_tasks} ${seed} ${lamb}

        # SSIL
        num_exemplars=2000
        lamb=0.25
        sbatch <account_name> <partition> scripts/templates/imagenet100_base_rn18/ssil.sh ${num_tasks} ${seed} ${num_exemplars} ${lamb}
    done
done