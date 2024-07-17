#!/bin/bash

set -e

for seed in 0; do
    for num_tasks in 5 10; do
        for ic_config in cifar100_resnet32_sdn cifar100_resnet32_dense; do
            # ANCL
            num_exemplars=0
            #             lamb=1.0
            #             lamb_a=1.0
            #             sbatch -A plgimprmoe-gpu-a100 -p plgrid-gpu-a100 scripts/templates/cifar100_ee/ancl.sh ${num_tasks} ${seed} ${num_exemplars} ${lamb} ${ic_config} ${lamb_a}
            #             lamb=2.0
            #             lamb_a=2.0
            #             sbatch -A plgimprmoe-gpu-a100 -p plgrid-gpu-a100 scripts/templates/cifar100_ee/ancl.sh ${num_tasks} ${seed} ${num_exemplars} ${lamb} ${ic_config} ${lamb_a}
            #             lamb=0.5
            #             lamb_a=0.5
            #             sbatch -A plgimprmoe-gpu-a100 -p plgrid-gpu-a100 scripts/templates/cifar100_ee/ancl.sh ${num_tasks} ${seed} ${num_exemplars} ${lamb} ${ic_config} ${lamb_a}
            lamb=1.0
            lamb_a=2.0
            sbatch -A plgimprmoe-gpu-a100 -p plgrid-gpu-a100 scripts/templates/cifar100_ee/ancl.sh ${num_tasks} ${seed} ${num_exemplars} ${lamb} ${ic_config} ${lamb_a}
            #             lamb=2.0
            #             lamb_a=1.0
            #             sbatch -A plgimprmoe-gpu-a100 -p plgrid-gpu-a100 scripts/templates/cifar100_ee/ancl.sh ${num_tasks} ${seed} ${num_exemplars} ${lamb} ${ic_config} ${lamb_a}
            #             lamb=1.0
            #             lamb_a=0.5
            #             sbatch -A plgimprmoe-gpu-a100 -p plgrid-gpu-a100 scripts/templates/cifar100_ee/ancl.sh ${num_tasks} ${seed} ${num_exemplars} ${lamb} ${ic_config} ${lamb_a}
            #             lamb=0.5
            #             lamb_a=1.0
            #             sbatch -A plgimprmoe-gpu-a100 -p plgrid-gpu-a100 scripts/templates/cifar100_ee/ancl.sh ${num_tasks} ${seed} ${num_exemplars} ${lamb} ${ic_config} ${lamb_a}

            # ANCL+Ex
            num_exemplars=2000
            #             lamb=1.0
            #             lamb_a=1.0
            #             sbatch -A plgimprmoe-gpu-a100 -p plgrid-gpu-a100 scripts/templates/cifar100_ee/ancl.sh ${num_tasks} ${seed} ${num_exemplars} ${lamb} ${ic_config} ${lamb_a}
            #             lamb=2.0
            #             lamb_a=2.0
            #             sbatch -A plgimprmoe-gpu-a100 -p plgrid-gpu-a100 scripts/templates/cifar100_ee/ancl.sh ${num_tasks} ${seed} ${num_exemplars} ${lamb} ${ic_config} ${lamb_a}
            #             lamb=0.5
            #             lamb_a=0.5
            #             sbatch -A plgimprmoe-gpu-a100 -p plgrid-gpu-a100 scripts/templates/cifar100_ee/ancl.sh ${num_tasks} ${seed} ${num_exemplars} ${lamb} ${ic_config} ${lamb_a}
            #             lamb=1.0
            #             lamb_a=2.0
            #             sbatch -A plgimprmoe-gpu-a100 -p plgrid-gpu-a100 scripts/templates/cifar100_ee/ancl.sh ${num_tasks} ${seed} ${num_exemplars} ${lamb} ${ic_config} ${lamb_a}
            #             lamb=2.0
            #             lamb_a=1.0
            #             sbatch -A plgimprmoe-gpu-a100 -p plgrid-gpu-a100 scripts/templates/cifar100_ee/ancl.sh ${num_tasks} ${seed} ${num_exemplars} ${lamb} ${ic_config} ${lamb_a}
            #             lamb=1.0
            #             lamb_a=0.5
            #             sbatch -A plgimprmoe-gpu-a100 -p plgrid-gpu-a100 scripts/templates/cifar100_ee/ancl.sh ${num_tasks} ${seed} ${num_exemplars} ${lamb} ${ic_config} ${lamb_a}
            lamb=0.5
            lamb_a=1.0
            sbatch -A plgimprmoe-gpu-a100 -p plgrid-gpu-a100 scripts/templates/cifar100_ee/ancl.sh ${num_tasks} ${seed} ${num_exemplars} ${lamb} ${ic_config} ${lamb_a}

            # BiC
            num_exemplars=2000
            #             lamb=-1
            #             sbatch -A plgimprmoe-gpu-a100 -p plgrid-gpu-a100 scripts/templates/cifar100_ee/bic.sh ${num_tasks} ${seed} ${num_exemplars} ${lamb} ${ic_config}
            #             lamb=0.5
            #             sbatch -A plgimprmoe-gpu-a100 -p plgrid-gpu-a100 scripts/templates/cifar100_ee/bic.sh ${num_tasks} ${seed} ${num_exemplars} ${lamb} ${ic_config}
            #             lamb=1
            #             sbatch -A plgimprmoe-gpu-a100 -p plgrid-gpu-a100 scripts/templates/cifar100_ee/bic.sh ${num_tasks} ${seed} ${num_exemplars} ${lamb} ${ic_config}
            lamb=2
            sbatch -A plgimprmoe-gpu-a100 -p plgrid-gpu-a100 scripts/templates/cifar100_ee/bic.sh ${num_tasks} ${seed} ${num_exemplars} ${lamb} ${ic_config}
            lamb=3
            sbatch -A plgimprmoe-gpu-a100 -p plgrid-gpu-a100 scripts/templates/cifar100_ee/bic.sh ${num_tasks} ${seed} ${num_exemplars} ${lamb} ${ic_config}
            #             lamb=4
            #             sbatch -A plgimprmoe-gpu-a100 -p plgrid-gpu-a100 scripts/templates/cifar100_ee/bic.sh ${num_tasks} ${seed} ${num_exemplars} ${lamb} ${ic_config}
            #             lamb=5
            #             sbatch -A plgimprmoe-gpu-a100 -p plgrid-gpu-a100 scripts/templates/cifar100_ee/bic.sh ${num_tasks} ${seed} ${num_exemplars} ${lamb} ${ic_config}

            # EWC
            lamb=10000
            alpha=0.5
            sbatch -A plgimprmoe-gpu-a100 -p plgrid-gpu-a100 scripts/templates/cifar100_ee/ewc.sh ${num_tasks} ${seed} ${lamb} ${alpha} ${ic_config}

            # ER
            num_exemplars=2000
            sbatch -A plgimprmoe-gpu-a100 -p plgrid-gpu-a100 scripts/templates/cifar100_ee/er.sh ${num_tasks} ${seed} ${num_exemplars}

            # FT
            num_exemplars=0
            sbatch -A plgimprmoe-gpu-a100 -p plgrid-gpu-a100 scripts/templates/cifar100_ee/ft.sh ${num_tasks} ${seed} ${num_exemplars} ${ic_config}

            # FT+Ex
            num_exemplars=2000
            sbatch -A plgimprmoe-gpu-a100 -p plgrid-gpu-a100 scripts/templates/cifar100_ee/ft.sh ${num_tasks} ${seed} ${num_exemplars} ${ic_config}

            # GDumb
            num_exemplars=2000
            sbatch -A plgimprmoe-gpu-a100 -p plgrid-gpu-a100 scripts/templates/cifar100_ee/gdumb.sh ${num_tasks} ${seed} ${num_exemplars} ${ic_config}

            # iCaRL TODO

            # LODE
            num_exemplars=2000
            const=1.0
            ro=0.1
            sbatch -A plgimprmoe-gpu-a100 -p plgrid-gpu-a100 scripts/templates/cifar100_ee/lode.sh ${num_tasks} ${seed} ${num_exemplars} ${const} ${ro} ${ic_config}
            const=1.0
            ro=0.05
            sbatch -A plgimprmoe-gpu-a100 -p plgrid-gpu-a100 scripts/templates/cifar100_ee/lode.sh ${num_tasks} ${seed} ${num_exemplars} ${const} ${ro} ${ic_config}
            const=1.0
            ro=0.01
            sbatch -A plgimprmoe-gpu-a100 -p plgrid-gpu-a100 scripts/templates/cifar100_ee/lode.sh ${num_tasks} ${seed} ${num_exemplars} ${const} ${ro} ${ic_config}

            # LWF
            lamb=0.25
            sbatch -A plgimprmoe-gpu-a100 -p plgrid-gpu-a100 scripts/templates/cifar100_ee/lwf.sh ${num_tasks} ${seed} ${lamb} ${ic_config}
            lamb=0.5
            sbatch -A plgimprmoe-gpu-a100 -p plgrid-gpu-a100 scripts/templates/cifar100_ee/lwf.sh ${num_tasks} ${seed} ${lamb} ${ic_config}
            lamb=0.75
            sbatch -A plgimprmoe-gpu-a100 -p plgrid-gpu-a100 scripts/templates/cifar100_ee/lwf.sh ${num_tasks} ${seed} ${lamb} ${ic_config}
            lamb=1.0
            sbatch -A plgimprmoe-gpu-a100 -p plgrid-gpu-a100 scripts/templates/cifar100_ee/lwf.sh ${num_tasks} ${seed} ${lamb} ${ic_config}
            lamb=1.5
            sbatch -A plgimprmoe-gpu-a100 -p plgrid-gpu-a100 scripts/templates/cifar100_ee/lwf.sh ${num_tasks} ${seed} ${lamb} ${ic_config}
            lamb=2.0
            sbatch -A plgimprmoe-gpu-a100 -p plgrid-gpu-a100 scripts/templates/cifar100_ee/lwf.sh ${num_tasks} ${seed} ${lamb} ${ic_config}

            # Joint
            #             sbatch -A plgimprmoe-gpu-a100 -p plgrid-gpu-a100 scripts/templates/cifar100_ee/joint.sh ${num_tasks} ${seed} ${ic_config}

            # SSIL
            num_exemplars=2000
            lamb=0.25
            sbatch -A plgimprmoe-gpu-a100 -p plgrid-gpu-a100 scripts/templates/cifar100_ee/ssil.sh ${num_tasks} ${seed} ${num_exemplars} ${lamb} ${ic_config}
            lamb=0.5
            #             sbatch -A plgimprmoe-gpu-a100 -p plgrid-gpu-a100 scripts/templates/cifar100_ee/ssil.sh ${num_tasks} ${seed} ${num_exemplars} ${lamb} ${ic_config}
            #             lamb=0.75
            #             sbatch -A plgimprmoe-gpu-a100 -p plgrid-gpu-a100 scripts/templates/cifar100_ee/ssil.sh ${num_tasks} ${seed} ${num_exemplars} ${lamb} ${ic_config}
            #             lamb=1.0
            #             sbatch -A plgimprmoe-gpu-a100 -p plgrid-gpu-a100 scripts/templates/cifar100_ee/ssil.sh ${num_tasks} ${seed} ${num_exemplars} ${lamb} ${ic_config}
            #             lamb=1.5
            #             sbatch -A plgimprmoe-gpu-a100 -p plgrid-gpu-a100 scripts/templates/cifar100_ee/ssil.sh ${num_tasks} ${seed} ${num_exemplars} ${lamb} ${ic_config}
            #             lamb=2.0
            #             sbatch -A plgimprmoe-gpu-a100 -p plgrid-gpu-a100 scripts/templates/cifar100_ee/ssil.sh ${num_tasks} ${seed} ${num_exemplars} ${lamb} ${ic_config}

        done
    done
done