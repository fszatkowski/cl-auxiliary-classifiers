#!/bin/bash

set -e

for seed in 0; do
    for num_tasks in 5 10; do
        for ic_config in imagenet100_vit_base imagenet100_vit_ln; do
            # ANCL
            num_exemplars=0
            lamb=1.0
            lamb_a=1.0
            sbatch <account_name> <partition> scripts/templates/imagenet100_ee_vit/ancl.sh ${num_tasks} ${seed} ${num_exemplars} ${lamb} ${lamb_a} ${ic_config}
            lamb=1.0
            lamb_a=2.0
            sbatch <account_name> <partition> scripts/templates/imagenet100_ee_vit/ancl.sh ${num_tasks} ${seed} ${num_exemplars} ${lamb} ${lamb_a} ${ic_config}

            # BiC
            num_exemplars=2000
            lamb=2
            sbatch <account_name> <partition> scripts/templates/imagenet100_ee_vit/bic.sh ${num_tasks} ${seed} ${num_exemplars} ${lamb} ${ic_config}
            lamb=3
            sbatch <account_name> <partition> scripts/templates/imagenet100_ee_vit/bic.sh ${num_tasks} ${seed} ${num_exemplars} ${lamb} ${ic_config}

            # EWC
            lamb=10000
            alpha=0.5
            sbatch <account_name> <partition> scripts/templates/imagenet100_ee_vit/ewc.sh ${num_tasks} ${seed} ${lamb} ${alpha} ${ic_config}

            # ER
            num_exemplars=2000
            sbatch <account_name> <partition> scripts/templates/imagenet100_ee_vit/er.sh ${num_tasks} ${seed} ${num_exemplars} ${ic_config}

            # FT
            num_exemplars=0
            sbatch <account_name> <partition> scripts/templates/imagenet100_ee_vit/ft.sh ${num_tasks} ${seed} ${num_exemplars} ${ic_config}

            # FT+Ex
            num_exemplars=2000
            sbatch <account_name> <partition> scripts/templates/imagenet100_ee_vit/ft.sh ${num_tasks} ${seed} ${num_exemplars} ${ic_config}

            # GDumb
            num_exemplars=2000
            sbatch <account_name> <partition> scripts/templates/imagenet100_ee_vit/gdumb.sh ${num_tasks} ${seed} ${num_exemplars} ${ic_config}

            # LODE
            num_exemplars=2000
            const=3.0
            ro=0.5
            sbatch <account_name> <partition> scripts/templates/imagenet100_ee_vit/lode.sh ${num_tasks} ${seed} ${num_exemplars} ${const} ${ro} ${ic_config}
            const=1.0
            ro=0.1
            sbatch <account_name> <partition> scripts/templates/imagenet100_ee_vit/lode.sh ${num_tasks} ${seed} ${num_exemplars} ${const} ${ro} ${ic_config}

            # LWF
            lamb=0.5
            sbatch <account_name> <partition> scripts/templates/imagenet100_ee_vit/lwf.sh ${num_tasks} ${seed} ${lamb} ${ic_config}
            lamb=1.0
            sbatch <account_name> <partition> scripts/templates/imagenet100_ee_vit/lwf.sh ${num_tasks} ${seed} ${lamb} ${ic_config}


            # SSIL
            num_exemplars=2000
            lamb=0.25
            sbatch <account_name> <partition> scripts/templates/imagenet100_ee_vit/ssil.sh ${num_tasks} ${seed} ${num_exemplars} ${lamb} ${ic_config}

        done
    done
done