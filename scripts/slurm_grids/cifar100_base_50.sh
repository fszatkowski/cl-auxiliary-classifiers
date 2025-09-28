#!/bin/bash

set -e

for seed in 0 1 2; do
    for num_tasks in 50; do
        for num_exemplars in 20; do
            # BiC
            #         lamb=-1
            #         sbatch <account_name> <partition> scripts/templates/cifar100_base_growing_mem/bic.sh ${num_tasks} ${seed} ${num_exemplars} ${lamb}
            #         lamb=0.5
            #         sbatch <account_name> <partition> scripts/templates/cifar100_base_growing_mem/bic.sh ${num_tasks} ${seed} ${num_exemplars} ${lamb}
            #         lamb=1
            #         sbatch <account_name> <partition> scripts/templates/cifar100_base_growing_mem/bic.sh ${num_tasks} ${seed} ${num_exemplars} ${lamb}
            #         lamb=-1
            #         sbatch <account_name> <partition> scripts/templates/cifar100_base_growing_mem/bic.sh ${num_tasks} ${seed} ${num_exemplars} ${lamb}
            #         lamb=1
            #         sbatch <account_name> <partition> scripts/templates/cifar100_base_growing_mem/bic.sh ${num_tasks} ${seed} ${num_exemplars} ${lamb}
            #         lamb=2
            #         sbatch <account_name> <partition> scripts/templates/cifar100_base_growing_mem/bic.sh ${num_tasks} ${seed} ${num_exemplars} ${lamb}
            #         lamb=3
            #         sbatch <account_name> <partition> scripts/templates/cifar100_base_growing_mem/bic.sh ${num_tasks} ${seed} ${num_exemplars} ${lamb}
            #         lamb=4
            #         sbatch <account_name> <partition> scripts/templates/cifar100_base_growing_mem/bic.sh ${num_tasks} ${seed} ${num_exemplars} ${lamb}
            #         lamb=3
            #         sbatch <account_name> <partition> scripts/templates/cifar100_base_growing_mem/bic.sh ${num_tasks} ${seed} ${num_exemplars} ${lamb}
            #         lamb=4
            #         sbatch <account_name> <partition> scripts/templates/cifar100_base_growing_mem/bic.sh ${num_tasks} ${seed} ${num_exemplars} ${lamb}
            #         lamb=5
            #         sbatch <account_name> <partition> scripts/templates/cifar100_base_growing_mem/bic.sh ${num_tasks} ${seed} ${num_exemplars} ${lamb}

            # DER++
            num_tasks=50
            num_exemplars=20
            alpha=0.5
            beta=0.5
            sbatch <account_name> <partition> scripts/templates/cifar100_base_growing_mem/der++.sh ${num_tasks} ${seed} ${num_exemplars} ${alpha} ${beta}

            # ER
            sbatch <account_name> <partition> scripts/templates/cifar100_base_growing_mem/er.sh ${num_tasks} ${seed} ${num_exemplars}

            #
            #         # FT+Ex
            #         sbatch <account_name> <partition> scripts/templates/cifar100_base_growing_mem/ft.sh ${num_tasks} ${seed} ${num_exemplars}
            #
            #         # GDumb
            #         sbatch <account_name> <partition> scripts/templates/cifar100_base_growing_mem/gdumb.sh ${num_tasks} ${seed} ${num_exemplars}

            # LODE
            #         const=1
            #         ro=0.1
            #         sbatch <account_name> <partition> scripts/templates/cifar100_base_growing_mem/lode.sh ${num_tasks} ${seed} ${num_exemplars} ${const} ${ro}
            #         const=2
            #         ro=0.1
            #         sbatch <account_name> <partition> scripts/templates/cifar100_base_growing_mem/lode.sh ${num_tasks} ${seed} ${num_exemplars} ${const} ${ro}
            #         const=3
            #         ro=0.1
            #         sbatch <account_name> <partition> scripts/templates/cifar100_base_growing_mem/lode.sh ${num_tasks} ${seed} ${num_exemplars} ${const} ${ro}
            #         const=4
            #         ro=0.1
            #         sbatch <account_name> <partition> scripts/templates/cifar100_base_growing_mem/lode.sh ${num_tasks} ${seed} ${num_exemplars} ${const} ${ro}
            #         const=1
            #         ro=0.5
            #         sbatch <account_name> <partition> scripts/templates/cifar100_base_growing_mem/lode.sh ${num_tasks} ${seed} ${num_exemplars} ${const} ${ro}
            const=2
            ro=0.5
            sbatch <account_name> <partition> scripts/templates/cifar100_base_growing_mem/lode.sh ${num_tasks} ${seed} ${num_exemplars} ${const} ${ro}
            const=3
            ro=0.5
            sbatch <account_name> <partition> scripts/templates/cifar100_base_growing_mem/lode.sh ${num_tasks} ${seed} ${num_exemplars} ${const} ${ro}
            const=4
            ro=0.5
            sbatch <account_name> <partition> scripts/templates/cifar100_base_growing_mem/lode.sh ${num_tasks} ${seed} ${num_exemplars} ${const} ${ro}
            #         const=1.0
            #         ro=0.05
            #         sbatch <account_name> <partition> scripts/templates/cifar100_ee/lode.sh ${num_tasks} ${seed} ${num_exemplars} ${const} ${ro}
            #         const=1.0
            #         ro=0.01
            #         sbatch <account_name> <partition> scripts/templates/cifar100_ee/lode.sh ${num_tasks} ${seed} ${num_exemplars} ${const} ${ro}


            # SSIL
            #         lamb=0.25
            #         sbatch <account_name> <partition> scripts/templates/cifar100_base_growing_mem/ssil.sh ${num_tasks} ${seed} ${num_exemplars} ${lamb}
            #         lamb=0.1
            #         sbatch <account_name> <partition> scripts/templates/cifar100_base_growing_mem/ssil.sh ${num_tasks} ${seed} ${num_exemplars} ${lamb}
            #         lamb=0.5
            #         sbatch <account_name> <partition> scripts/templates/cifar100_base_growing_mem/ssil.sh ${num_tasks} ${seed} ${num_exemplars} ${lamb}
            #         lamb=0.75
            #         sbatch <account_name> <partition> scripts/templates/cifar100_base_growing_mem/ssil.sh ${num_tasks} ${seed} ${num_exemplars} ${lamb}
            #         lamb=1.0
            #         sbatch <account_name> <partition> scripts/templates/cifar100_base_growing_mem/ssil.sh ${num_tasks} ${seed} ${num_exemplars} ${lamb}
            #         lamb=0.5
            #         sbatch <account_name> <partition> scripts/templates/cifar100_base_growing_mem/ssil.sh ${num_tasks} ${seed} ${num_exemplars} ${lamb}
            #                  lamb=0.75
            #         sbatch <account_name> <partition> scripts/templates/cifar100_base_growing_mem/ssil.sh ${num_tasks} ${seed} ${num_exemplars} ${lamb}
            #         lamb=1.0
            #         sbatch <account_name> <partition> scripts/templates/cifar100_base_growing_mem/ssil.sh ${num_tasks} ${seed} ${num_exemplars} ${lamb}
            #         lamb=1.5
            #         sbatch <account_name> <partition> scripts/templates/cifar100_base_growing_mem/ssil.sh ${num_tasks} ${seed} ${num_exemplars} ${lamb}
            #         lamb=2.0
            #         sbatch <account_name> <partition> scripts/templates/cifar100_base_growing_mem/ssil.sh ${num_tasks} ${seed} ${num_exemplars} ${lamb}
        done
    done
done