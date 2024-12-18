#!/bin/bash

set -e

for seed in 0 1 2; do
    for num_tasks in 5 10; do
        #         # ANCL
        #         num_exemplars=0
        #         lamb=1.0
        #         lamb_a=1.0
        #         sbatch -Aplgimprmoe-gpu-a100 -p plgrid-gpu-a100 scripts/templates/vgg19_cifar100_base/ancl.sh ${num_tasks} ${seed} ${num_exemplars} ${lamb} ${lamb_a}

        # DER++
        num_exemplars=2000
        alpha=0.5
        beta=0.5
        sbatch -Aplgimprmoe-gpu-a100 -p plgrid-gpu-a100 scripts/templates/vgg19_cifar100_base/der++.sh ${num_tasks} ${seed} ${num_exemplars} ${alpha} ${beta}

        #         # ER
        #         num_exemplars=2000
        #         sbatch -Aplgimprmoe-gpu-a100 -p plgrid-gpu-a100 scripts/templates/vgg19_cifar100_base/er.sh ${num_tasks} ${seed} ${num_exemplars}
        #
        #         # GDumb
        #         num_exemplars=2000
        #         sbatch -Aplgimprmoe-gpu-a100 -p plgrid-gpu-a100 scripts/templates/vgg19_cifar100_base/gdumb.sh ${num_tasks} ${seed} ${num_exemplars}
        #
        #         # EWC
        #         lamb=10000
        #         alpha=0.5
        #         sbatch -Aplgimprmoe-gpu-a100 -p plgrid-gpu-a100 scripts/templates/vgg19_cifar100_base/ewc.sh ${num_tasks} ${seed} ${lamb} ${alpha}
        #
        #         # LODE
        #         num_exemplars=2000
        #         const=3.0
        #         ro=0.5
        #         sbatch -Aplgimprmoe-gpu-a100 -p plgrid-gpu-a100 scripts/templates/vgg19_cifar100_base/lode.sh ${num_tasks} ${seed} ${num_exemplars} ${const} ${ro}
        #         const=4.0
        #         ro=0.5
        #         sbatch -Aplgimprmoe-gpu-a100 -p plgrid-gpu-a100 scripts/templates/vgg19_cifar100_base/lode.sh ${num_tasks} ${seed} ${num_exemplars} ${const} ${ro}
        #         const=1.0
        #         ro=0.1
        #         sbatch -Aplgimprmoe-gpu-a100 -p plgrid-gpu-a100 scripts/templates/vgg19_cifar100_base/lode.sh ${num_tasks} ${seed} ${num_exemplars} ${const} ${ro}
        #         const=2.0
        #         ro=0.1
        #         sbatch -Aplgimprmoe-gpu-a100 -p plgrid-gpu-a100 scripts/templates/vgg19_cifar100_base/lode.sh ${num_tasks} ${seed} ${num_exemplars} ${const} ${ro}
        #         const=3.0
        #         ro=0.1
        #         sbatch -Aplgimprmoe-gpu-a100 -p plgrid-gpu-a100 scripts/templates/vgg19_cifar100_base/lode.sh ${num_tasks} ${seed} ${num_exemplars} ${const} ${ro}
        #         const=4.0
        #         ro=0.1
        #         sbatch -Aplgimprmoe-gpu-a100 -p plgrid-gpu-a100 scripts/templates/vgg19_cifar100_base/lode.sh ${num_tasks} ${seed} ${num_exemplars} ${const} ${ro}
        #
        #
        #         # SSIL
        #         num_exemplars=2000
        #         lamb=0.25
        #         sbatch -A plgimprmoe-gpu-a100 -p plgrid-gpu-a100 scripts/templates/vgg19_cifar100_base/ssil.sh ${num_tasks} ${seed} ${num_exemplars} ${lamb}
        #         lamb=0.1
        #         sbatch -A plgimprmoe-gpu-a100 -p plgrid-gpu-a100 scripts/templates/vgg19_cifar100_base/ssil.sh ${num_tasks} ${seed} ${num_exemplars} ${lamb}
        #         lamb=0.5
        #         sbatch -A plgimprmoe-gpu-a100 -p plgrid-gpu-a100 scripts/templates/vgg19_cifar100_base/ssil.sh ${num_tasks} ${seed} ${num_exemplars} ${lamb}
        #         lamb=0.75
        #         sbatch -A plgimprmoe-gpu-a100 -p plgrid-gpu-a100 scripts/templates/vgg19_cifar100_base/ssil.sh ${num_tasks} ${seed} ${num_exemplars} ${lamb}
        #         lamb=1.0
        #         sbatch -A plgimprmoe-gpu-a100 -p plgrid-gpu-a100 scripts/templates/vgg19_cifar100_base/ssil.sh ${num_tasks} ${seed} ${num_exemplars} ${lamb}
        #
        #
        #         #         # BiC
        #         #         num_exemplars=2000
        #         #         #         lamb=-1
        #         #         #         sbatch -A plgimprmoe-gpu-a100 -p plgrid-gpu-a100 scripts/templates/vgg19_cifar100_base/bic.sh ${num_tasks} ${seed} ${num_exemplars} ${lamb}
        #         #         #         lamb=0.5
        #         #         #         sbatch -A plgimprmoe-gpu-a100 -p plgrid-gpu-a100 scripts/templates/vgg19_cifar100_base/bic.sh ${num_tasks} ${seed} ${num_exemplars} ${lamb}
        #         #         #         lamb=1
        #         #         #         sbatch -A plgimprmoe-gpu-a100 -p plgrid-gpu-a100 scripts/templates/vgg19_cifar100_base/bic.sh ${num_tasks} ${seed} ${num_exemplars} ${lamb}
        #         #         lamb=-1
        #         #         sbatch -A plgimprmoe-gpu-a100 -p plgrid-gpu-a100 scripts/templates/vgg19_cifar100_base/bic.sh ${num_tasks} ${seed} ${num_exemplars} ${lamb}
        #         #         lamb=1
        #         #         sbatch -A plgimprmoe-gpu-a100 -p plgrid-gpu-a100 scripts/templates/vgg19_cifar100_base/bic.sh ${num_tasks} ${seed} ${num_exemplars} ${lamb}
        #         #         lamb=2
        #         #         sbatch -A plgimprmoe-gpu-a100 -p plgrid-gpu-a100 scripts/templates/vgg19_cifar100_base/bic.sh ${num_tasks} ${seed} ${num_exemplars} ${lamb}
        #         #         lamb=3
        #         #         sbatch -A plgimprmoe-gpu-a100 -p plgrid-gpu-a100 scripts/templates/vgg19_cifar100_base/bic.sh ${num_tasks} ${seed} ${num_exemplars} ${lamb}
        #         #         lamb=4
        #         #         sbatch -A plgimprmoe-gpu-a100 -p plgrid-gpu-a100 scripts/templates/vgg19_cifar100_base/bic.sh ${num_tasks} ${seed} ${num_exemplars} ${lamb}
        #         #         #         lamb=3
        #         #         #         sbatch -A plgimprmoe-gpu-a100 -p plgrid-gpu-a100 scripts/templates/vgg19_cifar100_base/bic.sh ${num_tasks} ${seed} ${num_exemplars} ${lamb}
        #         #         #         lamb=4
        #         #         #         sbatch -A plgimprmoe-gpu-a100 -p plgrid-gpu-a100 scripts/templates/vgg19_cifar100_base/bic.sh ${num_tasks} ${seed} ${num_exemplars} ${lamb}
        #         #         #         lamb=5
        #         #         #         sbatch -A plgimprmoe-gpu-a100 -p plgrid-gpu-a100 scripts/templates/vgg19_cifar100_base/bic.sh ${num_tasks} ${seed} ${num_exemplars} ${lamb}
        #         #
        #         #
        #         #         # FT
        #         #         num_exemplars=0
        #         #         sbatch -A plgimprmoe-gpu-a100 -p plgrid-gpu-a100 scripts/templates/vgg19_cifar100_base/ft.sh ${num_tasks} ${seed} ${num_exemplars}
        #         #
        #         #         # FT+Ex
        #         #         num_exemplars=2000
        #         #         sbatch -A plgimprmoe-gpu-a100 -p plgrid-gpu-a100 scripts/templates/vgg19_cifar100_base/ft.sh ${num_tasks} ${seed} ${num_exemplars}
        #         #
        #         #
        #         #         # LWF
        #         #         #         lamb=0.25
        #         #         #         sbatch -A plgimprmoe-gpu-a100 -p plgrid-gpu-a100 scripts/templates/vgg19_cifar100_base/lwf.sh ${num_tasks} ${seed} ${lamb}
        #         #         #         lamb=0.5
        #         #         #         sbatch -A plgimprmoe-gpu-a100 -p plgrid-gpu-a100 scripts/templates/vgg19_cifar100_base/lwf.sh ${num_tasks} ${seed} ${lamb}
        #         #         #         lamb=0.75
        #         #         #         sbatch -A plgimprmoe-gpu-a100 -p plgrid-gpu-a100 scripts/templates/vgg19_cifar100_base/lwf.sh ${num_tasks} ${seed} ${lamb}
        #         #         lamb=0.25
        #         #         sbatch -A plgimprmoe-gpu-a100 -p plgrid-gpu-a100 scripts/templates/vgg19_cifar100_base/lwf.sh ${num_tasks} ${seed} ${lamb}
        #         #         lamb=0.5
        #         #         sbatch -A plgimprmoe-gpu-a100 -p plgrid-gpu-a100 scripts/templates/vgg19_cifar100_base/lwf.sh ${num_tasks} ${seed} ${lamb}
        #         #         lamb=1.0
        #         #         sbatch -A plgimprmoe-gpu-a100 -p plgrid-gpu-a100 scripts/templates/vgg19_cifar100_base/lwf.sh ${num_tasks} ${seed} ${lamb}
        #         #         lamb=1.5
        #         #         sbatch -A plgimprmoe-gpu-a100 -p plgrid-gpu-a100 scripts/templates/vgg19_cifar100_base/lwf.sh ${num_tasks} ${seed} ${lamb}
        #         #         lamb=2.0
        #         #         sbatch -A plgimprmoe-gpu-a100 -p plgrid-gpu-a100 scripts/templates/vgg19_cifar100_base/lwf.sh ${num_tasks} ${seed} ${lamb}
        #         #         #         lamb=1.5
        #         #         #         sbatch -A plgimprmoe-gpu-a100 -p plgrid-gpu-a100 scripts/templates/vgg19_cifar100_base/lwf.sh ${num_tasks} ${seed} ${lamb}
        #         #         #         lamb=2.0
        #         #         #         sbatch -A plgimprmoe-gpu-a100 -p plgrid-gpu-a100 scripts/templates/vgg19_cifar100_base/lwf.sh ${num_tasks} ${seed} ${lamb}


    done
done