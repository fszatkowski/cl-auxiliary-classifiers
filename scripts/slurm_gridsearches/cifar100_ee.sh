#!/bin/bash

set -e

for seed in 0 1 2; do
    for ic_config in cifar100_resnet32_sdn; do
        for num_tasks in 5 10 20; do
            # ANCL
            num_exemplars=0
            #             lamb=1.0
            #             lamb_a=1.0
            #             sbatch -A plgdynamic2-gpu-a100 -p plgrid-gpu-a100 scripts/templates/cifar100_ee/ancl.sh ${num_tasks} ${seed} ${num_exemplars} ${lamb} ${lamb_a} ${ic_config}
            #             lamb=2.0
            #             lamb_a=2.0
            #             sbatch -A plgdynamic2-gpu-a100 -p plgrid-gpu-a100 scripts/templates/cifar100_ee/ancl.sh ${num_tasks} ${seed} ${num_exemplars} ${lamb} ${lamb_a} ${ic_config}
            #             lamb=0.5
            #             lamb_a=0.5
            #             sbatch -A plgdynamic2-gpu-a100 -p plgrid-gpu-a100 scripts/templates/cifar100_ee/ancl.sh ${num_tasks} ${seed} ${num_exemplars} ${lamb} ${lamb_a} ${ic_config}
            #             lamb=0.75
            #             lamb_a=0.75
            #             sbatch -A plgdynamic2-gpu-a100 -p plgrid-gpu-a100 scripts/templates/cifar100_ee/ancl.sh ${num_tasks} ${seed} ${num_exemplars} ${lamb} ${lamb_a} ${ic_config}
            lamb=1.0
            lamb_a=0.5
            sbatch -A plgdynamic2-gpu-a100 -p plgrid-gpu-a100 scripts/templates/cifar100_ee/ancl.sh ${num_tasks} ${seed} ${num_exemplars} ${lamb} ${lamb_a} ${ic_config}
            #             lamb=0.5
            #             lamb_a=1.0
            #             sbatch -A plgdynamic2-gpu-a100 -p plgrid-gpu-a100 scripts/templates/cifar100_ee/ancl.sh ${num_tasks} ${seed} ${num_exemplars} ${lamb} ${lamb_a} ${ic_config}
            #             lamb=2.0
            #             lamb_a=1.0
            #             sbatch -A plgdynamic2-gpu-a100 -p plgrid-gpu-a100 scripts/templates/cifar100_ee/ancl.sh ${num_tasks} ${seed} ${num_exemplars} ${lamb} ${lamb_a} ${ic_config}
            #             lamb=1.0
            #             lamb_a=0.5
            #             sbatch -A plgdynamic2-gpu-a100 -p plgrid-gpu-a100 scripts/templates/cifar100_ee/ancl.sh ${num_tasks} ${seed} ${num_exemplars} ${lamb} ${lamb_a} ${ic_config}
            #             lamb=0.5
            #             lamb_a=1.0
            #             sbatch -A plgdynamic2-gpu-a100 -p plgrid-gpu-a100 scripts/templates/cifar100_ee/ancl.sh ${num_tasks} ${seed} ${num_exemplars} ${lamb} ${lamb_a} ${ic_config}

            #             # ANCL+Ex
            #             num_exemplars=2000
            #             lamb=0.1
            #             lamb_a=0.1
            #             sbatch -A plgdynamic2-gpu-a100 -p plgrid-gpu-a100 scripts/templates/cifar100_ee/ancl.sh ${num_tasks} ${seed} ${num_exemplars} ${lamb} ${lamb_a} ${ic_config}
            #             #             lamb=2.0
            #             #             lamb_a=2.0
            #             #             sbatch -A plgdynamic2-gpu-a100 -p plgrid-gpu-a100 scripts/templates/cifar100_ee/ancl.sh ${num_tasks} ${seed} ${num_exemplars} ${lamb} ${lamb_a} ${ic_config}
            #             lamb=0.25
            #             lamb_a=0.25
            #             sbatch -A plgdynamic2-gpu-a100 -p plgrid-gpu-a100 scripts/templates/cifar100_ee/ancl.sh ${num_tasks} ${seed} ${num_exemplars} ${lamb} ${lamb_a} ${ic_config}
            #             lamb=0.5
            #             lamb_a=0.5
            #             sbatch -A plgdynamic2-gpu-a100 -p plgrid-gpu-a100 scripts/templates/cifar100_ee/ancl.sh ${num_tasks} ${seed} ${num_exemplars} ${lamb} ${lamb_a} ${ic_config}
            #             lamb=0.5
            #             lamb_a=0.25
            #             sbatch -A plgdynamic2-gpu-a100 -p plgrid-gpu-a100 scripts/templates/cifar100_ee/ancl.sh ${num_tasks} ${seed} ${num_exemplars} ${lamb} ${lamb_a} ${ic_config}
            #             lamb=0.25
            #             lamb_a=0.5
            #             sbatch -A plgdynamic2-gpu-a100 -p plgrid-gpu-a100 scripts/templates/cifar100_ee/ancl.sh ${num_tasks} ${seed} ${num_exemplars} ${lamb} ${lamb_a} ${ic_config}
            #             #             lamb=2.0
            #             #             lamb_a=1.0
            #             #             sbatch -A plgdynamic2-gpu-a100 -p plgrid-gpu-a100 scripts/templates/cifar100_ee/ancl.sh ${num_tasks} ${seed} ${num_exemplars} ${lamb} ${lamb_a} ${ic_config}
            #             #             lamb=1.0
            #             #             lamb_a=0.5
            #             #             sbatch -A plgdynamic2-gpu-a100 -p plgrid-gpu-a100 scripts/templates/cifar100_ee/ancl.sh ${num_tasks} ${seed} ${num_exemplars} ${lamb} ${lamb_a} ${ic_config}
            #             lamb=0.5
            #             lamb_a=1.0
            #             sbatch -A plgdynamic2-gpu-a100 -p plgrid-gpu-a100 scripts/templates/cifar100_ee/ancl.sh ${num_tasks} ${seed} ${num_exemplars} ${lamb} ${lamb_a} ${ic_config}

            # BiC
            num_exemplars=2000
            #             lamb=-1
            #             sbatch -A plgdynamic2-gpu-a100 -p plgrid-gpu-a100 scripts/templates/cifar100_ee/bic.sh ${num_tasks} ${seed} ${num_exemplars} ${lamb} ${ic_config}
            #             lamb=0.5
            #             sbatch -A plgdynamic2-gpu-a100 -p plgrid-gpu-a100 scripts/templates/cifar100_ee/bic.sh ${num_tasks} ${seed} ${num_exemplars} ${lamb} ${ic_config}
            #             lamb=1
            #             sbatch -A plgdynamic2-gpu-a100 -p plgrid-gpu-a100 scripts/templates/cifar100_ee/bic.sh ${num_tasks} ${seed} ${num_exemplars} ${lamb} ${ic_config}
            lamb=2
            sbatch -A plgdynamic2-gpu-a100 -p plgrid-gpu-a100 scripts/templates/cifar100_ee/bic.sh ${num_tasks} ${seed} ${num_exemplars} ${lamb} ${ic_config}
            #             lamb=3
            #             sbatch -A plgdynamic2-gpu-a100 -p plgrid-gpu-a100 scripts/templates/cifar100_ee/bic.sh ${num_tasks} ${seed} ${num_exemplars} ${lamb} ${ic_config}
            #                         lamb=4
            #             sbatch -A plgdynamic2-gpu-a100 -p plgrid-gpu-a100 scripts/templates/cifar100_ee/bic.sh ${num_tasks} ${seed} ${num_exemplars} ${lamb} ${ic_config}
            #             lamb=5
            #             sbatch -A plgdynamic2-gpu-a100 -p plgrid-gpu-a100 scripts/templates/cifar100_ee/bic.sh ${num_tasks} ${seed} ${num_exemplars} ${lamb} ${ic_config}

            # DER++
            num_exemplars=2000
            alpha=0.5
            beta=0.5
            sbatch -A plgimprmoe-gpu-a100 -p plgrid-gpu-a100 scripts/templates/cifar100_ee/der++.sh ${num_tasks} ${seed} ${num_exemplars} ${ic_config} ${alpha} ${beta}

            alpha=0.5
            beta=0.25
            sbatch -A plgimprmoe-gpu-a100 -p plgrid-gpu-a100 scripts/templates/cifar100_ee/der++.sh ${num_tasks} ${seed} ${num_exemplars} ${ic_config} ${alpha} ${beta}

            alpha=0.25
            beta=0.5
            sbatch -A plgimprmoe-gpu-a100 -p plgrid-gpu-a100 scripts/templates/cifar100_ee/der++.sh ${num_tasks} ${seed} ${num_exemplars} ${ic_config} ${alpha} ${beta}

            alpha=0.5
            beta=1.0
            sbatch -A plgimprmoe-gpu-a100 -p plgrid-gpu-a100 scripts/templates/cifar100_ee/der++.sh ${num_tasks} ${seed} ${num_exemplars} ${ic_config} ${alpha} ${beta}

            alpha=1.0
            beta=0.5
            sbatch -A plgimprmoe-gpu-a100 -p plgrid-gpu-a100 scripts/templates/cifar100_ee/der++.sh ${num_tasks} ${seed} ${num_exemplars} ${ic_config} ${alpha} ${beta}

            # EWC
            lamb=10000
            alpha=0.5
            sbatch -A plgdynamic2-gpu-a100 -p plgrid-gpu-a100 scripts/templates/cifar100_ee/ewc.sh ${num_tasks} ${seed} ${lamb} ${alpha} ${ic_config}

            # ER
            num_exemplars=2000
            sbatch -A plgdynamic2-gpu-a100 -p plgrid-gpu-a100 scripts/templates/cifar100_ee/er.sh ${num_tasks} ${seed} ${num_exemplars} ${ic_config}

            # FT
            num_exemplars=0
            sbatch -A plgdynamic2-gpu-a100 -p plgrid-gpu-a100 scripts/templates/cifar100_ee/ft.sh ${num_tasks} ${seed} ${num_exemplars} ${ic_config}

            # FT+Ex
            num_exemplars=2000
            sbatch -A plgdynamic2-gpu-a100 -p plgrid-gpu-a100 scripts/templates/cifar100_ee/ft.sh ${num_tasks} ${seed} ${num_exemplars} ${ic_config}

            # GDumb
            num_exemplars=2000
            sbatch -A plgdynamic2-gpu-a100 -p plgrid-gpu-a100 scripts/templates/cifar100_ee/gdumb.sh ${num_tasks} ${seed} ${num_exemplars} ${ic_config}

            # LODE
            num_exemplars=2000
            #             const=3.0
            #             ro=1.0
            #             sbatch -A plgdynamic2-gpu-a100 -p plgrid-gpu-a100 scripts/templates/cifar100_ee/lode.sh ${num_tasks} ${seed} ${num_exemplars} ${const} ${ro} ${ic_config}
            #             const=4.0
            #             ro=0.5
            #             sbatch -A plgdynamic2-gpu-a100 -p plgrid-gpu-a100 scripts/templates/cifar100_ee/lode.sh ${num_tasks} ${seed} ${num_exemplars} ${const} ${ro} ${ic_config}
            const=3.0
            ro=0.5
            sbatch -A plgdynamic2-gpu-a100 -p plgrid-gpu-a100 scripts/templates/cifar100_ee/lode.sh ${num_tasks} ${seed} ${num_exemplars} ${const} ${ro} ${ic_config}
            #             const=2.0
            #             ro=0.5
            #             sbatch -A plgdynamic2-gpu-a100 -p plgrid-gpu-a100 scripts/templates/cifar100_ee/lode.sh ${num_tasks} ${seed} ${num_exemplars} ${const} ${ro} ${ic_config}
            #             const=2.0
            #             ro=1
            #             sbatch -A plgdynamic2-gpu-a100 -p plgrid-gpu-a100 scripts/templates/cifar100_ee/lode.sh ${num_tasks} ${seed} ${num_exemplars} ${const} ${ro} ${ic_config}
            #             const=1.5
            #             ro=0.5
            #             sbatch -A plgdynamic2-gpu-a100 -p plgrid-gpu-a100 scripts/templates/cifar100_ee/lode.sh ${num_tasks} ${seed} ${num_exemplars} ${const} ${ro} ${ic_config}
            #             const=1.0
            #             ro=0.05
            #             sbatch -A plgdynamic2-gpu-a100 -p plgrid-gpu-a100 scripts/templates/cifar100_ee/lode.sh ${num_tasks} ${seed} ${num_exemplars} ${const} ${ro} ${ic_config}
            #             const=1.0
            #             ro=0.01
            #             sbatch -A plgdynamic2-gpu-a100 -p plgrid-gpu-a100 scripts/templates/cifar100_ee/lode.sh ${num_tasks} ${seed} ${num_exemplars} ${const} ${ro} ${ic_config}

            # LWF
            #             lamb=0.25
            #             sbatch -A plgdynamic2-gpu-a100 -p plgrid-gpu-a100 scripts/templates/cifar100_ee/lwf.sh ${num_tasks} ${seed} ${lamb} ${ic_config}
            lamb=0.5
            sbatch -A plgdynamic2-gpu-a100 -p plgrid-gpu-a100 scripts/templates/cifar100_ee/lwf.sh ${num_tasks} ${seed} ${lamb} ${ic_config}
            lamb=0.75
            sbatch -A plgdynamic2-gpu-a100 -p plgrid-gpu-a100 scripts/templates/cifar100_ee/lwf.sh ${num_tasks} ${seed} ${lamb} ${ic_config}
            lamb=1.0
            sbatch -A plgdynamic2-gpu-a100 -p plgrid-gpu-a100 scripts/templates/cifar100_ee/lwf.sh ${num_tasks} ${seed} ${lamb} ${ic_config}
            #             lamb=1.5
            #             sbatch -A plgdynamic2-gpu-a100 -p plgrid-gpu-a100 scripts/templates/cifar100_ee/lwf.sh ${num_tasks} ${seed} ${lamb} ${ic_config}
            #             lamb=2.0
            #             sbatch -A plgdynamic2-gpu-a100 -p plgrid-gpu-a100 scripts/templates/cifar100_ee/lwf.sh ${num_tasks} ${seed} ${lamb} ${ic_config}

            # Joint
            #             sbatch -A plgdynamic2-gpu-a100 -p plgrid-gpu-a100 scripts/templates/cifar100_ee/joint.sh ${num_tasks} ${seed} ${ic_config}

            # SSIL
            num_exemplars=2000
            lamb=0.25
            sbatch -A plgdynamic2-gpu-a100 -p plgrid-gpu-a100 scripts/templates/cifar100_ee/ssil.sh ${num_tasks} ${seed} ${num_exemplars} ${lamb} ${ic_config}
            #             lamb=0.5
            #             sbatch -A plgdynamic2-gpu-a100 -p plgrid-gpu-a100 scripts/templates/cifar100_ee/ssil.sh ${num_tasks} ${seed} ${num_exemplars} ${lamb} ${ic_config}
            #             lamb=0.5
            #             sbatch -A plgdynamic2-gpu-a100 -p plgrid-gpu-a100 scripts/templates/cifar100_ee/ssil.sh ${num_tasks} ${seed} ${num_exemplars} ${lamb} ${ic_config}
            #             lamb=0.75
            #             sbatch -A plgdynamic2-gpu-a100 -p plgrid-gpu-a100 scripts/templates/cifar100_ee/ssil.sh ${num_tasks} ${seed} ${num_exemplars} ${lamb} ${ic_config}
            #             lamb=1.0
            #             sbatch -A plgdynamic2-gpu-a100 -p plgrid-gpu-a100 scripts/templates/cifar100_ee/ssil.sh ${num_tasks} ${seed} ${num_exemplars} ${lamb} ${ic_config}
            #             lamb=1.5
            #             sbatch -A plgdynamic2-gpu-a100 -p plgrid-gpu-a100 scripts/templates/cifar100_ee/ssil.sh ${num_tasks} ${seed} ${num_exemplars} ${lamb} ${ic_config}
            #             lamb=2.0
            #             sbatch -A plgdynamic2-gpu-a100 -p plgrid-gpu-a100 scripts/templates/cifar100_ee/ssil.sh ${num_tasks} ${seed} ${num_exemplars} ${lamb} ${ic_config}

        done
    done
done