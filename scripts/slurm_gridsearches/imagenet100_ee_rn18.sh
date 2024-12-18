#!/bin/bash

set -e

for seed in 0 1 2; do
    for num_tasks in 5 10; do
        for ic_config in imagenet100_resnet18_sdn; do
            # ANCL
            num_exemplars=0
            lamb=1.0
            lamb_a=1.0
            sbatch -A plgdynamic2-gpu-a100 -p plgrid-gpu-a100 scripts/templates/imagenet100_ee_rn18/ancl.sh ${num_tasks} ${seed} ${num_exemplars} ${lamb} ${lamb_a} ${ic_config}
            lamb=1.0
            lamb_a=2.0
            sbatch -A plgdynamic2-gpu-a100 -p plgrid-gpu-a100 scripts/templates/imagenet100_ee_rn18/ancl.sh ${num_tasks} ${seed} ${num_exemplars} ${lamb} ${lamb_a} ${ic_config}

            # BiC
            num_exemplars=2000
            lamb=2
            sbatch -A plgdynamic2-gpu-a100 -p plgrid-gpu-a100 scripts/templates/imagenet100_ee_rn18/bic.sh ${num_tasks} ${seed} ${num_exemplars} ${lamb} ${ic_config}
            lamb=3
            sbatch -A plgdynamic2-gpu-a100 -p plgrid-gpu-a100 scripts/templates/imagenet100_ee_rn18/bic.sh ${num_tasks} ${seed} ${num_exemplars} ${lamb} ${ic_config}

            # EWC
            lamb=10000
            alpha=0.5
            sbatch -A plgdynamic2-gpu-a100 -p plgrid-gpu-a100 scripts/templates/imagenet100_ee_rn18/ewc.sh ${num_tasks} ${seed} ${lamb} ${alpha} ${ic_config}

            # ER
            num_exemplars=2000
            sbatch -A plgdynamic2-gpu-a100 -p plgrid-gpu-a100 scripts/templates/imagenet100_ee_rn18/er.sh ${num_tasks} ${seed} ${num_exemplars} ${ic_config}

            # FT
            num_exemplars=0
            sbatch -A plgdynamic2-gpu-a100 -p plgrid-gpu-a100 scripts/templates/imagenet100_ee_rn18/ft.sh ${num_tasks} ${seed} ${num_exemplars} ${ic_config}

            # FT+Ex
            num_exemplars=2000
            sbatch -A plgdynamic2-gpu-a100 -p plgrid-gpu-a100 scripts/templates/imagenet100_ee_rn18/ft.sh ${num_tasks} ${seed} ${num_exemplars} ${ic_config}

            # GDumb
            num_exemplars=2000
            sbatch -A plgdynamic2-gpu-a100 -p plgrid-gpu-a100 scripts/templates/imagenet100_ee_rn18/gdumb.sh ${num_tasks} ${seed} ${num_exemplars} ${ic_config}


            # LODE
            num_exemplars=2000
            const=3.0
            ro=0.5
            const=1.0
            ro=0.1
            sbatch -A plgdynamic2-gpu-a100 -p plgrid-gpu-a100 scripts/templates/imagenet100_ee_rn18/lode.sh ${num_tasks} ${seed} ${num_exemplars} ${const} ${ro} ${ic_config}

            # LWF
            lamb=0.5
            sbatch -A plgdynamic2-gpu-a100 -p plgrid-gpu-a100 scripts/templates/imagenet100_ee_rn18/lwf.sh ${num_tasks} ${seed} ${lamb} ${ic_config}
            lamb=1.0
            sbatch -A plgdynamic2-gpu-a100 -p plgrid-gpu-a100 scripts/templates/imagenet100_ee_rn18/lwf.sh ${num_tasks} ${seed} ${lamb} ${ic_config}

            # SSIL
            num_exemplars=2000
            lamb=0.25
            sbatch -A plgdynamic2-gpu-a100 -p plgrid-gpu-a100 scripts/templates/imagenet100_ee_rn18/ssil.sh ${num_tasks} ${seed} ${num_exemplars} ${lamb} ${ic_config}

        done
    done
done