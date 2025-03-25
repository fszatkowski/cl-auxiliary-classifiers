#!/bin/bash

set -e

# This slurm script allows to run VGG19 experiments on equal split of CIFAR100


# <--- --- --- CIFAR100x5 --- --- --- --->
num_tasks=5

# <--- ANCL --->
num_exemplars=0
lamb=1.0
lamb_a=1.0
sbatch -A plgdynamic2-gpu-a100 -p plgrid-gpu-a100 scripts/templates/vgg19_cifar100_base/ancl.sh ${num_tasks} 0 ${num_exemplars} ${lamb} ${lamb_a}
sbatch -A plgdynamic2-gpu-a100 -p plgrid-gpu-a100 scripts/templates/vgg19_cifar100_base/ancl.sh ${num_tasks} 1 ${num_exemplars} ${lamb} ${lamb_a}
sbatch -A plgdynamic2-gpu-a100 -p plgrid-gpu-a100 scripts/templates/vgg19_cifar100_base/ancl.sh ${num_tasks} 2 ${num_exemplars} ${lamb} ${lamb_a}

lamb=1.0
lamb_a=1.0
sbatch -A plgdynamic2-gpu-a100 -p plgrid-gpu-a100 scripts/templates/vgg19_cifar100_ee/ancl.sh ${num_tasks} 0 ${num_exemplars} ${lamb} ${lamb_a} cifar100_vgg19_small
sbatch -A plgdynamic2-gpu-a100 -p plgrid-gpu-a100 scripts/templates/vgg19_cifar100_ee/ancl.sh ${num_tasks} 1 ${num_exemplars} ${lamb} ${lamb_a} cifar100_vgg19_small
sbatch -A plgdynamic2-gpu-a100 -p plgrid-gpu-a100 scripts/templates/vgg19_cifar100_ee/ancl.sh ${num_tasks} 2 ${num_exemplars} ${lamb} ${lamb_a} cifar100_vgg19_small

lamb=1.0
lamb_a=1.0
sbatch -A plgdynamic2-gpu-a100 -p plgrid-gpu-a100 scripts/templates/vgg19_cifar100_ee/ancl.sh ${num_tasks} 0 ${num_exemplars} ${lamb} ${lamb_a} cifar100_vgg19_medium
sbatch -A plgdynamic2-gpu-a100 -p plgrid-gpu-a100 scripts/templates/vgg19_cifar100_ee/ancl.sh ${num_tasks} 1 ${num_exemplars} ${lamb} ${lamb_a} cifar100_vgg19_medium
sbatch -A plgdynamic2-gpu-a100 -p plgrid-gpu-a100 scripts/templates/vgg19_cifar100_ee/ancl.sh ${num_tasks} 2 ${num_exemplars} ${lamb} ${lamb_a} cifar100_vgg19_medium

lamb=1.0
lamb_a=1.0
sbatch -A plgdynamic2-gpu-a100 -p plgrid-gpu-a100 scripts/templates/vgg19_cifar100_ee/ancl.sh ${num_tasks} 0 ${num_exemplars} ${lamb} ${lamb_a} cifar100_vgg19_dense
sbatch -A plgdynamic2-gpu-a100 -p plgrid-gpu-a100 scripts/templates/vgg19_cifar100_ee/ancl.sh ${num_tasks} 1 ${num_exemplars} ${lamb} ${lamb_a} cifar100_vgg19_dense
sbatch -A plgdynamic2-gpu-a100 -p plgrid-gpu-a100 scripts/templates/vgg19_cifar100_ee/ancl.sh ${num_tasks} 2 ${num_exemplars} ${lamb} ${lamb_a} cifar100_vgg19_dense

# <--- BiC --->
num_exemplars=2000
lamb=3
sbatch -A plgdynamic2-gpu-a100 -p plgrid-gpu-a100 scripts/templates/vgg19_cifar100_base/bic.sh ${num_tasks} 0 ${num_exemplars} ${lamb}
sbatch -A plgdynamic2-gpu-a100 -p plgrid-gpu-a100 scripts/templates/vgg19_cifar100_base/bic.sh ${num_tasks} 1 ${num_exemplars} ${lamb}
sbatch -A plgdynamic2-gpu-a100 -p plgrid-gpu-a100 scripts/templates/vgg19_cifar100_base/bic.sh ${num_tasks} 2 ${num_exemplars} ${lamb}

lamb=3
sbatch -A plgdynamic2-gpu-a100 -p plgrid-gpu-a100 scripts/templates/vgg19_cifar100_ee/bic.sh ${num_tasks} 0 ${num_exemplars} ${lamb} cifar100_vgg19_small
sbatch -A plgdynamic2-gpu-a100 -p plgrid-gpu-a100 scripts/templates/vgg19_cifar100_ee/bic.sh ${num_tasks} 1 ${num_exemplars} ${lamb} cifar100_vgg19_small
sbatch -A plgdynamic2-gpu-a100 -p plgrid-gpu-a100 scripts/templates/vgg19_cifar100_ee/bic.sh ${num_tasks} 2 ${num_exemplars} ${lamb} cifar100_vgg19_small

lamb=3
sbatch -A plgdynamic2-gpu-a100 -p plgrid-gpu-a100 scripts/templates/vgg19_cifar100_ee/bic.sh ${num_tasks} 0 ${num_exemplars} ${lamb} cifar100_vgg19_medium
sbatch -A plgdynamic2-gpu-a100 -p plgrid-gpu-a100 scripts/templates/vgg19_cifar100_ee/bic.sh ${num_tasks} 1 ${num_exemplars} ${lamb} cifar100_vgg19_medium
sbatch -A plgdynamic2-gpu-a100 -p plgrid-gpu-a100 scripts/templates/vgg19_cifar100_ee/bic.sh ${num_tasks} 2 ${num_exemplars} ${lamb} cifar100_vgg19_medium

lamb=3
sbatch -A plgdynamic2-gpu-a100 -p plgrid-gpu-a100 scripts/templates/vgg19_cifar100_ee/bic.sh ${num_tasks} 0 ${num_exemplars} ${lamb} cifar100_vgg19_dense
sbatch -A plgdynamic2-gpu-a100 -p plgrid-gpu-a100 scripts/templates/vgg19_cifar100_ee/bic.sh ${num_tasks} 1 ${num_exemplars} ${lamb} cifar100_vgg19_dense
sbatch -A plgdynamic2-gpu-a100 -p plgrid-gpu-a100 scripts/templates/vgg19_cifar100_ee/bic.sh ${num_tasks} 2 ${num_exemplars} ${lamb} cifar100_vgg19_dense

# <--- DER++ --->
num_exemplars=2000
alpha=0.5
beta=0.5
sbatch -A plgimprmoe-gpu-a100 -p plgrid-gpu-a100 scripts/templates/vgg19_cifar100_base/der++.sh ${num_tasks} 0 ${num_exemplars} ${alpha} ${beta}
sbatch -A plgimprmoe-gpu-a100 -p plgrid-gpu-a100 scripts/templates/vgg19_cifar100_base/der++.sh ${num_tasks} 1 ${num_exemplars} ${alpha} ${beta}
sbatch -A plgimprmoe-gpu-a100 -p plgrid-gpu-a100 scripts/templates/vgg19_cifar100_base/der++.sh ${num_tasks} 2 ${num_exemplars} ${alpha} ${beta}

alpha=0.5
beta=1.0
sbatch -A plgimprmoe-gpu-a100 -p plgrid-gpu-a100 scripts/templates/vgg19_cifar100_ee/der++.sh ${num_tasks} 0 ${num_exemplars} ${alpha} ${beta} cifar100_vgg19_small
sbatch -A plgimprmoe-gpu-a100 -p plgrid-gpu-a100 scripts/templates/vgg19_cifar100_ee/der++.sh ${num_tasks} 1 ${num_exemplars} ${alpha} ${beta} cifar100_vgg19_small
sbatch -A plgimprmoe-gpu-a100 -p plgrid-gpu-a100 scripts/templates/vgg19_cifar100_ee/der++.sh ${num_tasks} 2 ${num_exemplars} ${alpha} ${beta} cifar100_vgg19_small

alpha=0.5
beta=1.0
sbatch -A plgimprmoe-gpu-a100 -p plgrid-gpu-a100 scripts/templates/vgg19_cifar100_ee/der++.sh ${num_tasks} 0 ${num_exemplars} ${alpha} ${beta} cifar100_vgg19_medium
sbatch -A plgimprmoe-gpu-a100 -p plgrid-gpu-a100 scripts/templates/vgg19_cifar100_ee/der++.sh ${num_tasks} 1 ${num_exemplars} ${alpha} ${beta} cifar100_vgg19_medium
sbatch -A plgimprmoe-gpu-a100 -p plgrid-gpu-a100 scripts/templates/vgg19_cifar100_ee/der++.sh ${num_tasks} 2 ${num_exemplars} ${alpha} ${beta} cifar100_vgg19_medium

alpha=0.5
beta=1.0
sbatch -A plgimprmoe-gpu-a100 -p plgrid-gpu-a100 scripts/templates/vgg19_cifar100_ee/der++.sh ${num_tasks} 0 ${num_exemplars} ${alpha} ${beta} cifar100_vgg19_dense
sbatch -A plgimprmoe-gpu-a100 -p plgrid-gpu-a100 scripts/templates/vgg19_cifar100_ee/der++.sh ${num_tasks} 1 ${num_exemplars} ${alpha} ${beta} cifar100_vgg19_dense
sbatch -A plgimprmoe-gpu-a100 -p plgrid-gpu-a100 scripts/templates/vgg19_cifar100_ee/der++.sh ${num_tasks} 2 ${num_exemplars} ${alpha} ${beta} cifar100_vgg19_dense

# <--- EWC --->
lamb=10000
alpha=0.5
sbatch -A plgdynamic2-gpu-a100 -p plgrid-gpu-a100 scripts/templates/vgg19_cifar100_base/ewc.sh ${num_tasks} 0 ${lamb} ${alpha}
sbatch -A plgdynamic2-gpu-a100 -p plgrid-gpu-a100 scripts/templates/vgg19_cifar100_base/ewc.sh ${num_tasks} 1 ${lamb} ${alpha}
sbatch -A plgdynamic2-gpu-a100 -p plgrid-gpu-a100 scripts/templates/vgg19_cifar100_base/ewc.sh ${num_tasks} 2 ${lamb} ${alpha}

lamb=10000
alpha=0.5
sbatch -A plgdynamic2-gpu-a100 -p plgrid-gpu-a100 scripts/templates/vgg19_cifar100_ee/ewc.sh ${num_tasks} 0 ${lamb} ${alpha} cifar100_vgg19_small
sbatch -A plgdynamic2-gpu-a100 -p plgrid-gpu-a100 scripts/templates/vgg19_cifar100_ee/ewc.sh ${num_tasks} 1 ${lamb} ${alpha} cifar100_vgg19_small
sbatch -A plgdynamic2-gpu-a100 -p plgrid-gpu-a100 scripts/templates/vgg19_cifar100_ee/ewc.sh ${num_tasks} 2 ${lamb} ${alpha} cifar100_vgg19_small

lamb=10000
alpha=0.5
sbatch -A plgdynamic2-gpu-a100 -p plgrid-gpu-a100 scripts/templates/vgg19_cifar100_ee/ewc.sh ${num_tasks} 0 ${lamb} ${alpha} cifar100_vgg19_medium
sbatch -A plgdynamic2-gpu-a100 -p plgrid-gpu-a100 scripts/templates/vgg19_cifar100_ee/ewc.sh ${num_tasks} 1 ${lamb} ${alpha} cifar100_vgg19_medium
sbatch -A plgdynamic2-gpu-a100 -p plgrid-gpu-a100 scripts/templates/vgg19_cifar100_ee/ewc.sh ${num_tasks} 2 ${lamb} ${alpha} cifar100_vgg19_medium

lamb=10000
alpha=0.5
sbatch -A plgdynamic2-gpu-a100 -p plgrid-gpu-a100 scripts/templates/vgg19_cifar100_ee/ewc.sh ${num_tasks} 0 ${lamb} ${alpha} cifar100_vgg19_dense
sbatch -A plgdynamic2-gpu-a100 -p plgrid-gpu-a100 scripts/templates/vgg19_cifar100_ee/ewc.sh ${num_tasks} 1 ${lamb} ${alpha} cifar100_vgg19_dense
sbatch -A plgdynamic2-gpu-a100 -p plgrid-gpu-a100 scripts/templates/vgg19_cifar100_ee/ewc.sh ${num_tasks} 2 ${lamb} ${alpha} cifar100_vgg19_dense

# <--- ER --->
num_exemplars=2000
sbatch -A plgdynamic2-gpu-a100 -p plgrid-gpu-a100 scripts/templates/vgg19_cifar100_base/er.sh ${num_tasks} 0 ${num_exemplars}
sbatch -A plgdynamic2-gpu-a100 -p plgrid-gpu-a100 scripts/templates/vgg19_cifar100_base/er.sh ${num_tasks} 1 ${num_exemplars}
sbatch -A plgdynamic2-gpu-a100 -p plgrid-gpu-a100 scripts/templates/vgg19_cifar100_base/er.sh ${num_tasks} 2 ${num_exemplars}

sbatch -A plgdynamic2-gpu-a100 -p plgrid-gpu-a100 scripts/templates/vgg19_cifar100_ee/er.sh ${num_tasks} 0 ${num_exemplars} cifar100_vgg19_small
sbatch -A plgdynamic2-gpu-a100 -p plgrid-gpu-a100 scripts/templates/vgg19_cifar100_ee/er.sh ${num_tasks} 1 ${num_exemplars} cifar100_vgg19_small
sbatch -A plgdynamic2-gpu-a100 -p plgrid-gpu-a100 scripts/templates/vgg19_cifar100_ee/er.sh ${num_tasks} 2 ${num_exemplars} cifar100_vgg19_small

sbatch -A plgdynamic2-gpu-a100 -p plgrid-gpu-a100 scripts/templates/vgg19_cifar100_ee/er.sh ${num_tasks} 0 ${num_exemplars} cifar100_vgg19_medium
sbatch -A plgdynamic2-gpu-a100 -p plgrid-gpu-a100 scripts/templates/vgg19_cifar100_ee/er.sh ${num_tasks} 1 ${num_exemplars} cifar100_vgg19_medium
sbatch -A plgdynamic2-gpu-a100 -p plgrid-gpu-a100 scripts/templates/vgg19_cifar100_ee/er.sh ${num_tasks} 2 ${num_exemplars} cifar100_vgg19_medium

sbatch -A plgdynamic2-gpu-a100 -p plgrid-gpu-a100 scripts/templates/vgg19_cifar100_ee/er.sh ${num_tasks} 0 ${num_exemplars} cifar100_vgg19_dense
sbatch -A plgdynamic2-gpu-a100 -p plgrid-gpu-a100 scripts/templates/vgg19_cifar100_ee/er.sh ${num_tasks} 1 ${num_exemplars} cifar100_vgg19_dense
sbatch -A plgdynamic2-gpu-a100 -p plgrid-gpu-a100 scripts/templates/vgg19_cifar100_ee/er.sh ${num_tasks} 2 ${num_exemplars} cifar100_vgg19_dense

# <--- FT --->
num_exemplars=0
sbatch -A plgdynamic2-gpu-a100 -p plgrid-gpu-a100 scripts/templates/vgg19_cifar100_base/ft.sh ${num_tasks} 0 ${num_exemplars}
sbatch -A plgdynamic2-gpu-a100 -p plgrid-gpu-a100 scripts/templates/vgg19_cifar100_base/ft.sh ${num_tasks} 1 ${num_exemplars}
sbatch -A plgdynamic2-gpu-a100 -p plgrid-gpu-a100 scripts/templates/vgg19_cifar100_base/ft.sh ${num_tasks} 2 ${num_exemplars}

sbatch -A plgdynamic2-gpu-a100 -p plgrid-gpu-a100 scripts/templates/vgg19_cifar100_ee/ft.sh ${num_tasks} 0 ${num_exemplars} cifar100_vgg19_small
sbatch -A plgdynamic2-gpu-a100 -p plgrid-gpu-a100 scripts/templates/vgg19_cifar100_ee/ft.sh ${num_tasks} 1 ${num_exemplars} cifar100_vgg19_small
sbatch -A plgdynamic2-gpu-a100 -p plgrid-gpu-a100 scripts/templates/vgg19_cifar100_ee/ft.sh ${num_tasks} 2 ${num_exemplars} cifar100_vgg19_small

sbatch -A plgdynamic2-gpu-a100 -p plgrid-gpu-a100 scripts/templates/vgg19_cifar100_ee/ft.sh ${num_tasks} 0 ${num_exemplars} cifar100_vgg19_medium
sbatch -A plgdynamic2-gpu-a100 -p plgrid-gpu-a100 scripts/templates/vgg19_cifar100_ee/ft.sh ${num_tasks} 1 ${num_exemplars} cifar100_vgg19_medium
sbatch -A plgdynamic2-gpu-a100 -p plgrid-gpu-a100 scripts/templates/vgg19_cifar100_ee/ft.sh ${num_tasks} 2 ${num_exemplars} cifar100_vgg19_medium

sbatch -A plgdynamic2-gpu-a100 -p plgrid-gpu-a100 scripts/templates/vgg19_cifar100_ee/ft.sh ${num_tasks} 0 ${num_exemplars} cifar100_vgg19_dense
sbatch -A plgdynamic2-gpu-a100 -p plgrid-gpu-a100 scripts/templates/vgg19_cifar100_ee/ft.sh ${num_tasks} 1 ${num_exemplars} cifar100_vgg19_dense
sbatch -A plgdynamic2-gpu-a100 -p plgrid-gpu-a100 scripts/templates/vgg19_cifar100_ee/ft.sh ${num_tasks} 2 ${num_exemplars} cifar100_vgg19_dense

# <--- FT+Ex --->
num_exemplars=2000
sbatch -A plgdynamic2-gpu-a100 -p plgrid-gpu-a100 scripts/templates/vgg19_cifar100_base/ft.sh ${num_tasks} 0 ${num_exemplars}
sbatch -A plgdynamic2-gpu-a100 -p plgrid-gpu-a100 scripts/templates/vgg19_cifar100_base/ft.sh ${num_tasks} 1 ${num_exemplars}
sbatch -A plgdynamic2-gpu-a100 -p plgrid-gpu-a100 scripts/templates/vgg19_cifar100_base/ft.sh ${num_tasks} 2 ${num_exemplars}

sbatch -A plgdynamic2-gpu-a100 -p plgrid-gpu-a100 scripts/templates/vgg19_cifar100_ee/ft.sh ${num_tasks} 0 ${num_exemplars} cifar100_vgg19_small
sbatch -A plgdynamic2-gpu-a100 -p plgrid-gpu-a100 scripts/templates/vgg19_cifar100_ee/ft.sh ${num_tasks} 1 ${num_exemplars} cifar100_vgg19_small
sbatch -A plgdynamic2-gpu-a100 -p plgrid-gpu-a100 scripts/templates/vgg19_cifar100_ee/ft.sh ${num_tasks} 2 ${num_exemplars} cifar100_vgg19_small

sbatch -A plgdynamic2-gpu-a100 -p plgrid-gpu-a100 scripts/templates/vgg19_cifar100_ee/ft.sh ${num_tasks} 0 ${num_exemplars} cifar100_vgg19_medium
sbatch -A plgdynamic2-gpu-a100 -p plgrid-gpu-a100 scripts/templates/vgg19_cifar100_ee/ft.sh ${num_tasks} 1 ${num_exemplars} cifar100_vgg19_medium
sbatch -A plgdynamic2-gpu-a100 -p plgrid-gpu-a100 scripts/templates/vgg19_cifar100_ee/ft.sh ${num_tasks} 2 ${num_exemplars} cifar100_vgg19_medium

sbatch -A plgdynamic2-gpu-a100 -p plgrid-gpu-a100 scripts/templates/vgg19_cifar100_ee/ft.sh ${num_tasks} 0 ${num_exemplars} cifar100_vgg19_dense
sbatch -A plgdynamic2-gpu-a100 -p plgrid-gpu-a100 scripts/templates/vgg19_cifar100_ee/ft.sh ${num_tasks} 1 ${num_exemplars} cifar100_vgg19_dense
sbatch -A plgdynamic2-gpu-a100 -p plgrid-gpu-a100 scripts/templates/vgg19_cifar100_ee/ft.sh ${num_tasks} 2 ${num_exemplars} cifar100_vgg19_dense

# <--- GDumb --->
num_exemplars=2000
sbatch -A plgdynamic2-gpu-a100 -p plgrid-gpu-a100 scripts/templates/vgg19_cifar100_base/gdumb.sh ${num_tasks} 0 ${num_exemplars}
sbatch -A plgdynamic2-gpu-a100 -p plgrid-gpu-a100 scripts/templates/vgg19_cifar100_base/gdumb.sh ${num_tasks} 1 ${num_exemplars}
sbatch -A plgdynamic2-gpu-a100 -p plgrid-gpu-a100 scripts/templates/vgg19_cifar100_base/gdumb.sh ${num_tasks} 2 ${num_exemplars}

sbatch -A plgdynamic2-gpu-a100 -p plgrid-gpu-a100 scripts/templates/vgg19_cifar100_ee/gdumb.sh ${num_tasks} 0 ${num_exemplars} cifar100_vgg19_small
sbatch -A plgdynamic2-gpu-a100 -p plgrid-gpu-a100 scripts/templates/vgg19_cifar100_ee/gdumb.sh ${num_tasks} 1 ${num_exemplars} cifar100_vgg19_small
sbatch -A plgdynamic2-gpu-a100 -p plgrid-gpu-a100 scripts/templates/vgg19_cifar100_ee/gdumb.sh ${num_tasks} 2 ${num_exemplars} cifar100_vgg19_small

sbatch -A plgdynamic2-gpu-a100 -p plgrid-gpu-a100 scripts/templates/vgg19_cifar100_ee/gdumb.sh ${num_tasks} 0 ${num_exemplars} cifar100_vgg19_medium
sbatch -A plgdynamic2-gpu-a100 -p plgrid-gpu-a100 scripts/templates/vgg19_cifar100_ee/gdumb.sh ${num_tasks} 1 ${num_exemplars} cifar100_vgg19_medium
sbatch -A plgdynamic2-gpu-a100 -p plgrid-gpu-a100 scripts/templates/vgg19_cifar100_ee/gdumb.sh ${num_tasks} 2 ${num_exemplars} cifar100_vgg19_medium

sbatch -A plgdynamic2-gpu-a100 -p plgrid-gpu-a100 scripts/templates/vgg19_cifar100_ee/gdumb.sh ${num_tasks} 0 ${num_exemplars} cifar100_vgg19_dense
sbatch -A plgdynamic2-gpu-a100 -p plgrid-gpu-a100 scripts/templates/vgg19_cifar100_ee/gdumb.sh ${num_tasks} 1 ${num_exemplars} cifar100_vgg19_dense
sbatch -A plgdynamic2-gpu-a100 -p plgrid-gpu-a100 scripts/templates/vgg19_cifar100_ee/gdumb.sh ${num_tasks} 2 ${num_exemplars} cifar100_vgg19_dense

# <--- LODE --->
num_exemplars=2000
const=4.0
ro=0.1
sbatch -A plgdynamic2-gpu-a100 -p plgrid-gpu-a100 scripts/templates/vgg19_cifar100_base/lode.sh ${num_tasks} 0 ${num_exemplars} ${const} ${ro}
sbatch -A plgdynamic2-gpu-a100 -p plgrid-gpu-a100 scripts/templates/vgg19_cifar100_base/lode.sh ${num_tasks} 1 ${num_exemplars} ${const} ${ro}
sbatch -A plgdynamic2-gpu-a100 -p plgrid-gpu-a100 scripts/templates/vgg19_cifar100_base/lode.sh ${num_tasks} 2 ${num_exemplars} ${const} ${ro}

const=4.0
ro=0.1
sbatch -A plgdynamic2-gpu-a100 -p plgrid-gpu-a100 scripts/templates/vgg19_cifar100_ee/lode.sh ${num_tasks} 0 ${num_exemplars} ${const} ${ro} cifar100_vgg19_small
sbatch -A plgdynamic2-gpu-a100 -p plgrid-gpu-a100 scripts/templates/vgg19_cifar100_ee/lode.sh ${num_tasks} 1 ${num_exemplars} ${const} ${ro} cifar100_vgg19_small
sbatch -A plgdynamic2-gpu-a100 -p plgrid-gpu-a100 scripts/templates/vgg19_cifar100_ee/lode.sh ${num_tasks} 2 ${num_exemplars} ${const} ${ro} cifar100_vgg19_small

const=3.0
ro=0.5
sbatch -A plgdynamic2-gpu-a100 -p plgrid-gpu-a100 scripts/templates/vgg19_cifar100_ee/lode.sh ${num_tasks} 0 ${num_exemplars} ${const} ${ro} cifar100_vgg19_medium
sbatch -A plgdynamic2-gpu-a100 -p plgrid-gpu-a100 scripts/templates/vgg19_cifar100_ee/lode.sh ${num_tasks} 1 ${num_exemplars} ${const} ${ro} cifar100_vgg19_medium
sbatch -A plgdynamic2-gpu-a100 -p plgrid-gpu-a100 scripts/templates/vgg19_cifar100_ee/lode.sh ${num_tasks} 2 ${num_exemplars} ${const} ${ro} cifar100_vgg19_medium

const=3.0
ro=0.5
sbatch -A plgdynamic2-gpu-a100 -p plgrid-gpu-a100 scripts/templates/vgg19_cifar100_ee/lode.sh ${num_tasks} 0 ${num_exemplars} ${const} ${ro} cifar100_vgg19_dense
sbatch -A plgdynamic2-gpu-a100 -p plgrid-gpu-a100 scripts/templates/vgg19_cifar100_ee/lode.sh ${num_tasks} 1 ${num_exemplars} ${const} ${ro} cifar100_vgg19_dense
sbatch -A plgdynamic2-gpu-a100 -p plgrid-gpu-a100 scripts/templates/vgg19_cifar100_ee/lode.sh ${num_tasks} 2 ${num_exemplars} ${const} ${ro} cifar100_vgg19_dense

# <--- LwF --->
lamb=1.0
sbatch -A plgdynamic2-gpu-a100 -p plgrid-gpu-a100 scripts/templates/vgg19_cifar100_base/lwf.sh ${num_tasks} 0 ${lamb}
sbatch -A plgdynamic2-gpu-a100 -p plgrid-gpu-a100 scripts/templates/vgg19_cifar100_base/lwf.sh ${num_tasks} 1 ${lamb}
sbatch -A plgdynamic2-gpu-a100 -p plgrid-gpu-a100 scripts/templates/vgg19_cifar100_base/lwf.sh ${num_tasks} 2 ${lamb}

lamb=1.0
sbatch -A plgdynamic2-gpu-a100 -p plgrid-gpu-a100 scripts/templates/vgg19_cifar100_ee/lwf.sh ${num_tasks} 0 ${lamb} cifar100_vgg19_small
sbatch -A plgdynamic2-gpu-a100 -p plgrid-gpu-a100 scripts/templates/vgg19_cifar100_ee/lwf.sh ${num_tasks} 1 ${lamb} cifar100_vgg19_small
sbatch -A plgdynamic2-gpu-a100 -p plgrid-gpu-a100 scripts/templates/vgg19_cifar100_ee/lwf.sh ${num_tasks} 2 ${lamb} cifar100_vgg19_small

lamb=1.0
sbatch -A plgdynamic2-gpu-a100 -p plgrid-gpu-a100 scripts/templates/vgg19_cifar100_ee/lwf.sh ${num_tasks} 0 ${lamb} cifar100_vgg19_medium
sbatch -A plgdynamic2-gpu-a100 -p plgrid-gpu-a100 scripts/templates/vgg19_cifar100_ee/lwf.sh ${num_tasks} 1 ${lamb} cifar100_vgg19_medium
sbatch -A plgdynamic2-gpu-a100 -p plgrid-gpu-a100 scripts/templates/vgg19_cifar100_ee/lwf.sh ${num_tasks} 2 ${lamb} cifar100_vgg19_medium

lamb=1.0
sbatch -A plgdynamic2-gpu-a100 -p plgrid-gpu-a100 scripts/templates/vgg19_cifar100_ee/lwf.sh ${num_tasks} 0 ${lamb} cifar100_vgg19_dense
sbatch -A plgdynamic2-gpu-a100 -p plgrid-gpu-a100 scripts/templates/vgg19_cifar100_ee/lwf.sh ${num_tasks} 1 ${lamb} cifar100_vgg19_dense
sbatch -A plgdynamic2-gpu-a100 -p plgrid-gpu-a100 scripts/templates/vgg19_cifar100_ee/lwf.sh ${num_tasks} 2 ${lamb} cifar100_vgg19_dense

# <--- Joint --->
sbatch -A plgdynamic2-gpu-a100 -p plgrid-gpu-a100 scripts/templates/vgg19_cifar100_base/joint.sh ${num_tasks} 0
sbatch -A plgdynamic2-gpu-a100 -p plgrid-gpu-a100 scripts/templates/vgg19_cifar100_base/joint.sh ${num_tasks} 1
sbatch -A plgdynamic2-gpu-a100 -p plgrid-gpu-a100 scripts/templates/vgg19_cifar100_base/joint.sh ${num_tasks} 2

sbatch -A plgdynamic2-gpu-a100 -p plgrid-gpu-a100 scripts/templates/vgg19_cifar100_ee/joint.sh ${num_tasks} 0 cifar100_vgg19_small
sbatch -A plgdynamic2-gpu-a100 -p plgrid-gpu-a100 scripts/templates/vgg19_cifar100_ee/joint.sh ${num_tasks} 1 cifar100_vgg19_small
sbatch -A plgdynamic2-gpu-a100 -p plgrid-gpu-a100 scripts/templates/vgg19_cifar100_ee/joint.sh ${num_tasks} 2 cifar100_vgg19_small

sbatch -A plgdynamic2-gpu-a100 -p plgrid-gpu-a100 scripts/templates/vgg19_cifar100_ee/joint.sh ${num_tasks} 0 cifar100_vgg19_medium
sbatch -A plgdynamic2-gpu-a100 -p plgrid-gpu-a100 scripts/templates/vgg19_cifar100_ee/joint.sh ${num_tasks} 1 cifar100_vgg19_medium
sbatch -A plgdynamic2-gpu-a100 -p plgrid-gpu-a100 scripts/templates/vgg19_cifar100_ee/joint.sh ${num_tasks} 2 cifar100_vgg19_medium

sbatch -A plgdynamic2-gpu-a100 -p plgrid-gpu-a100 scripts/templates/vgg19_cifar100_ee/joint.sh ${num_tasks} 0 cifar100_vgg19_dense
sbatch -A plgdynamic2-gpu-a100 -p plgrid-gpu-a100 scripts/templates/vgg19_cifar100_ee/joint.sh ${num_tasks} 1 cifar100_vgg19_dense
sbatch -A plgdynamic2-gpu-a100 -p plgrid-gpu-a100 scripts/templates/vgg19_cifar100_ee/joint.sh ${num_tasks} 2 cifar100_vgg19_dense

# <--- SSIL --->
num_exemplars=2000
lamb=0.25
sbatch -A plgdynamic2-gpu-a100 -p plgrid-gpu-a100 scripts/templates/vgg19_cifar100_base/ssil.sh ${num_tasks} 0 ${num_exemplars} ${lamb}
sbatch -A plgdynamic2-gpu-a100 -p plgrid-gpu-a100 scripts/templates/vgg19_cifar100_base/ssil.sh ${num_tasks} 1 ${num_exemplars} ${lamb}
sbatch -A plgdynamic2-gpu-a100 -p plgrid-gpu-a100 scripts/templates/vgg19_cifar100_base/ssil.sh ${num_tasks} 2 ${num_exemplars} ${lamb}

lamb=0.5
sbatch -A plgdynamic2-gpu-a100 -p plgrid-gpu-a100 scripts/templates/vgg19_cifar100_ee/ssil.sh ${num_tasks} 0 ${num_exemplars} ${lamb} cifar100_vgg19_small
sbatch -A plgdynamic2-gpu-a100 -p plgrid-gpu-a100 scripts/templates/vgg19_cifar100_ee/ssil.sh ${num_tasks} 1 ${num_exemplars} ${lamb} cifar100_vgg19_small
sbatch -A plgdynamic2-gpu-a100 -p plgrid-gpu-a100 scripts/templates/vgg19_cifar100_ee/ssil.sh ${num_tasks} 2 ${num_exemplars} ${lamb} cifar100_vgg19_small

lamb=0.5
sbatch -A plgdynamic2-gpu-a100 -p plgrid-gpu-a100 scripts/templates/vgg19_cifar100_ee/ssil.sh ${num_tasks} 0 ${num_exemplars} ${lamb} cifar100_vgg19_medium
sbatch -A plgdynamic2-gpu-a100 -p plgrid-gpu-a100 scripts/templates/vgg19_cifar100_ee/ssil.sh ${num_tasks} 1 ${num_exemplars} ${lamb} cifar100_vgg19_medium
sbatch -A plgdynamic2-gpu-a100 -p plgrid-gpu-a100 scripts/templates/vgg19_cifar100_ee/ssil.sh ${num_tasks} 2 ${num_exemplars} ${lamb} cifar100_vgg19_medium

lamb=0.5
sbatch -A plgdynamic2-gpu-a100 -p plgrid-gpu-a100 scripts/templates/vgg19_cifar100_ee/ssil.sh ${num_tasks} 0 ${num_exemplars} ${lamb} cifar100_vgg19_dense
sbatch -A plgdynamic2-gpu-a100 -p plgrid-gpu-a100 scripts/templates/vgg19_cifar100_ee/ssil.sh ${num_tasks} 1 ${num_exemplars} ${lamb} cifar100_vgg19_dense
sbatch -A plgdynamic2-gpu-a100 -p plgrid-gpu-a100 scripts/templates/vgg19_cifar100_ee/ssil.sh ${num_tasks} 2 ${num_exemplars} ${lamb} cifar100_vgg19_dense





# <--- --- --- CIFAR100x10 --- --- --- --->
num_tasks=10

# <--- ANCL --->
num_exemplars=0
lamb=1.0
lamb_a=1.0
sbatch -A plgdynamic2-gpu-a100 -p plgrid-gpu-a100 scripts/templates/vgg19_cifar100_base/ancl.sh ${num_tasks} 0 ${num_exemplars} ${lamb} ${lamb_a}
sbatch -A plgdynamic2-gpu-a100 -p plgrid-gpu-a100 scripts/templates/vgg19_cifar100_base/ancl.sh ${num_tasks} 1 ${num_exemplars} ${lamb} ${lamb_a}
sbatch -A plgdynamic2-gpu-a100 -p plgrid-gpu-a100 scripts/templates/vgg19_cifar100_base/ancl.sh ${num_tasks} 2 ${num_exemplars} ${lamb} ${lamb_a}

lamb=1.0
lamb_a=1.0
sbatch -A plgdynamic2-gpu-a100 -p plgrid-gpu-a100 scripts/templates/vgg19_cifar100_ee/ancl.sh ${num_tasks} 0 ${num_exemplars} ${lamb} ${lamb_a} cifar100_vgg19_small
sbatch -A plgdynamic2-gpu-a100 -p plgrid-gpu-a100 scripts/templates/vgg19_cifar100_ee/ancl.sh ${num_tasks} 1 ${num_exemplars} ${lamb} ${lamb_a} cifar100_vgg19_small
sbatch -A plgdynamic2-gpu-a100 -p plgrid-gpu-a100 scripts/templates/vgg19_cifar100_ee/ancl.sh ${num_tasks} 2 ${num_exemplars} ${lamb} ${lamb_a} cifar100_vgg19_small

lamb=1.0
lamb_a=1.0
sbatch -A plgdynamic2-gpu-a100 -p plgrid-gpu-a100 scripts/templates/vgg19_cifar100_ee/ancl.sh ${num_tasks} 0 ${num_exemplars} ${lamb} ${lamb_a} cifar100_vgg19_medium
sbatch -A plgdynamic2-gpu-a100 -p plgrid-gpu-a100 scripts/templates/vgg19_cifar100_ee/ancl.sh ${num_tasks} 1 ${num_exemplars} ${lamb} ${lamb_a} cifar100_vgg19_medium
sbatch -A plgdynamic2-gpu-a100 -p plgrid-gpu-a100 scripts/templates/vgg19_cifar100_ee/ancl.sh ${num_tasks} 2 ${num_exemplars} ${lamb} ${lamb_a} cifar100_vgg19_medium

lamb=1.0
lamb_a=1.0
sbatch -A plgdynamic2-gpu-a100 -p plgrid-gpu-a100 scripts/templates/vgg19_cifar100_ee/ancl.sh ${num_tasks} 0 ${num_exemplars} ${lamb} ${lamb_a} cifar100_vgg19_dense
sbatch -A plgdynamic2-gpu-a100 -p plgrid-gpu-a100 scripts/templates/vgg19_cifar100_ee/ancl.sh ${num_tasks} 1 ${num_exemplars} ${lamb} ${lamb_a} cifar100_vgg19_dense
sbatch -A plgdynamic2-gpu-a100 -p plgrid-gpu-a100 scripts/templates/vgg19_cifar100_ee/ancl.sh ${num_tasks} 2 ${num_exemplars} ${lamb} ${lamb_a} cifar100_vgg19_dense

# <--- BiC --->
num_exemplars=2000
lamb=4
sbatch -A plgdynamic2-gpu-a100 -p plgrid-gpu-a100 scripts/templates/vgg19_cifar100_base/bic.sh ${num_tasks} 0 ${num_exemplars} ${lamb}
sbatch -A plgdynamic2-gpu-a100 -p plgrid-gpu-a100 scripts/templates/vgg19_cifar100_base/bic.sh ${num_tasks} 1 ${num_exemplars} ${lamb}
sbatch -A plgdynamic2-gpu-a100 -p plgrid-gpu-a100 scripts/templates/vgg19_cifar100_base/bic.sh ${num_tasks} 2 ${num_exemplars} ${lamb}

lamb=3
sbatch -A plgdynamic2-gpu-a100 -p plgrid-gpu-a100 scripts/templates/vgg19_cifar100_ee/bic.sh ${num_tasks} 0 ${num_exemplars} ${lamb} cifar100_vgg19_small
sbatch -A plgdynamic2-gpu-a100 -p plgrid-gpu-a100 scripts/templates/vgg19_cifar100_ee/bic.sh ${num_tasks} 1 ${num_exemplars} ${lamb} cifar100_vgg19_small
sbatch -A plgdynamic2-gpu-a100 -p plgrid-gpu-a100 scripts/templates/vgg19_cifar100_ee/bic.sh ${num_tasks} 2 ${num_exemplars} ${lamb} cifar100_vgg19_small

lamb=4
sbatch -A plgdynamic2-gpu-a100 -p plgrid-gpu-a100 scripts/templates/vgg19_cifar100_ee/bic.sh ${num_tasks} 0 ${num_exemplars} ${lamb} cifar100_vgg19_medium
sbatch -A plgdynamic2-gpu-a100 -p plgrid-gpu-a100 scripts/templates/vgg19_cifar100_ee/bic.sh ${num_tasks} 1 ${num_exemplars} ${lamb} cifar100_vgg19_medium
sbatch -A plgdynamic2-gpu-a100 -p plgrid-gpu-a100 scripts/templates/vgg19_cifar100_ee/bic.sh ${num_tasks} 2 ${num_exemplars} ${lamb} cifar100_vgg19_medium

lamb=3
sbatch -A plgdynamic2-gpu-a100 -p plgrid-gpu-a100 scripts/templates/vgg19_cifar100_ee/bic.sh ${num_tasks} 0 ${num_exemplars} ${lamb} cifar100_vgg19_dense
sbatch -A plgdynamic2-gpu-a100 -p plgrid-gpu-a100 scripts/templates/vgg19_cifar100_ee/bic.sh ${num_tasks} 1 ${num_exemplars} ${lamb} cifar100_vgg19_dense
sbatch -A plgdynamic2-gpu-a100 -p plgrid-gpu-a100 scripts/templates/vgg19_cifar100_ee/bic.sh ${num_tasks} 2 ${num_exemplars} ${lamb} cifar100_vgg19_dense

# <--- DER++ --->
num_exemplars=2000
alpha=0.5
beta=0.5
sbatch -A plgimprmoe-gpu-a100 -p plgrid-gpu-a100 scripts/templates/vgg19_cifar100_base/der++.sh ${num_tasks} 0 ${num_exemplars} ${alpha} ${beta}
sbatch -A plgimprmoe-gpu-a100 -p plgrid-gpu-a100 scripts/templates/vgg19_cifar100_base/der++.sh ${num_tasks} 1 ${num_exemplars} ${alpha} ${beta}
sbatch -A plgimprmoe-gpu-a100 -p plgrid-gpu-a100 scripts/templates/vgg19_cifar100_base/der++.sh ${num_tasks} 2 ${num_exemplars} ${alpha} ${beta}

alpha=0.5
beta=1.0
sbatch -A plgimprmoe-gpu-a100 -p plgrid-gpu-a100 scripts/templates/vgg19_cifar100_ee/der++.sh ${num_tasks} 0 ${num_exemplars} ${alpha} ${beta} cifar100_vgg19_small
sbatch -A plgimprmoe-gpu-a100 -p plgrid-gpu-a100 scripts/templates/vgg19_cifar100_ee/der++.sh ${num_tasks} 1 ${num_exemplars} ${alpha} ${beta} cifar100_vgg19_small
sbatch -A plgimprmoe-gpu-a100 -p plgrid-gpu-a100 scripts/templates/vgg19_cifar100_ee/der++.sh ${num_tasks} 2 ${num_exemplars} ${alpha} ${beta} cifar100_vgg19_small

alpha=0.5
beta=1.0
sbatch -A plgimprmoe-gpu-a100 -p plgrid-gpu-a100 scripts/templates/vgg19_cifar100_ee/der++.sh ${num_tasks} 0 ${num_exemplars} ${alpha} ${beta} cifar100_vgg19_medium
sbatch -A plgimprmoe-gpu-a100 -p plgrid-gpu-a100 scripts/templates/vgg19_cifar100_ee/der++.sh ${num_tasks} 1 ${num_exemplars} ${alpha} ${beta} cifar100_vgg19_medium
sbatch -A plgimprmoe-gpu-a100 -p plgrid-gpu-a100 scripts/templates/vgg19_cifar100_ee/der++.sh ${num_tasks} 2 ${num_exemplars} ${alpha} ${beta} cifar100_vgg19_medium

alpha=0.5
beta=1.0
sbatch -A plgimprmoe-gpu-a100 -p plgrid-gpu-a100 scripts/templates/vgg19_cifar100_ee/der++.sh ${num_tasks} 0 ${num_exemplars} ${alpha} ${beta} cifar100_vgg19_dense
sbatch -A plgimprmoe-gpu-a100 -p plgrid-gpu-a100 scripts/templates/vgg19_cifar100_ee/der++.sh ${num_tasks} 1 ${num_exemplars} ${alpha} ${beta} cifar100_vgg19_dense
sbatch -A plgimprmoe-gpu-a100 -p plgrid-gpu-a100 scripts/templates/vgg19_cifar100_ee/der++.sh ${num_tasks} 2 ${num_exemplars} ${alpha} ${beta} cifar100_vgg19_dense

# <--- EWC --->
lamb=10000
alpha=0.5
sbatch -A plgdynamic2-gpu-a100 -p plgrid-gpu-a100 scripts/templates/vgg19_cifar100_base/ewc.sh ${num_tasks} 0 ${lamb} ${alpha}
sbatch -A plgdynamic2-gpu-a100 -p plgrid-gpu-a100 scripts/templates/vgg19_cifar100_base/ewc.sh ${num_tasks} 1 ${lamb} ${alpha}
sbatch -A plgdynamic2-gpu-a100 -p plgrid-gpu-a100 scripts/templates/vgg19_cifar100_base/ewc.sh ${num_tasks} 2 ${lamb} ${alpha}

lamb=10000
alpha=0.5
sbatch -A plgdynamic2-gpu-a100 -p plgrid-gpu-a100 scripts/templates/vgg19_cifar100_ee/ewc.sh ${num_tasks} 0 ${lamb} ${alpha} cifar100_vgg19_small
sbatch -A plgdynamic2-gpu-a100 -p plgrid-gpu-a100 scripts/templates/vgg19_cifar100_ee/ewc.sh ${num_tasks} 1 ${lamb} ${alpha} cifar100_vgg19_small
sbatch -A plgdynamic2-gpu-a100 -p plgrid-gpu-a100 scripts/templates/vgg19_cifar100_ee/ewc.sh ${num_tasks} 2 ${lamb} ${alpha} cifar100_vgg19_small

lamb=10000
alpha=0.5
sbatch -A plgdynamic2-gpu-a100 -p plgrid-gpu-a100 scripts/templates/vgg19_cifar100_ee/ewc.sh ${num_tasks} 0 ${lamb} ${alpha} cifar100_vgg19_medium
sbatch -A plgdynamic2-gpu-a100 -p plgrid-gpu-a100 scripts/templates/vgg19_cifar100_ee/ewc.sh ${num_tasks} 1 ${lamb} ${alpha} cifar100_vgg19_medium
sbatch -A plgdynamic2-gpu-a100 -p plgrid-gpu-a100 scripts/templates/vgg19_cifar100_ee/ewc.sh ${num_tasks} 2 ${lamb} ${alpha} cifar100_vgg19_medium

lamb=10000
alpha=0.5
sbatch -A plgdynamic2-gpu-a100 -p plgrid-gpu-a100 scripts/templates/vgg19_cifar100_ee/ewc.sh ${num_tasks} 0 ${lamb} ${alpha} cifar100_vgg19_dense
sbatch -A plgdynamic2-gpu-a100 -p plgrid-gpu-a100 scripts/templates/vgg19_cifar100_ee/ewc.sh ${num_tasks} 1 ${lamb} ${alpha} cifar100_vgg19_dense
sbatch -A plgdynamic2-gpu-a100 -p plgrid-gpu-a100 scripts/templates/vgg19_cifar100_ee/ewc.sh ${num_tasks} 2 ${lamb} ${alpha} cifar100_vgg19_dense

# <--- ER --->
num_exemplars=2000
sbatch -A plgdynamic2-gpu-a100 -p plgrid-gpu-a100 scripts/templates/vgg19_cifar100_base/er.sh ${num_tasks} 0 ${num_exemplars}
sbatch -A plgdynamic2-gpu-a100 -p plgrid-gpu-a100 scripts/templates/vgg19_cifar100_base/er.sh ${num_tasks} 1 ${num_exemplars}
sbatch -A plgdynamic2-gpu-a100 -p plgrid-gpu-a100 scripts/templates/vgg19_cifar100_base/er.sh ${num_tasks} 2 ${num_exemplars}

sbatch -A plgdynamic2-gpu-a100 -p plgrid-gpu-a100 scripts/templates/vgg19_cifar100_ee/er.sh ${num_tasks} 0 ${num_exemplars} cifar100_vgg19_small
sbatch -A plgdynamic2-gpu-a100 -p plgrid-gpu-a100 scripts/templates/vgg19_cifar100_ee/er.sh ${num_tasks} 1 ${num_exemplars} cifar100_vgg19_small
sbatch -A plgdynamic2-gpu-a100 -p plgrid-gpu-a100 scripts/templates/vgg19_cifar100_ee/er.sh ${num_tasks} 2 ${num_exemplars} cifar100_vgg19_small

sbatch -A plgdynamic2-gpu-a100 -p plgrid-gpu-a100 scripts/templates/vgg19_cifar100_ee/er.sh ${num_tasks} 0 ${num_exemplars} cifar100_vgg19_medium
sbatch -A plgdynamic2-gpu-a100 -p plgrid-gpu-a100 scripts/templates/vgg19_cifar100_ee/er.sh ${num_tasks} 1 ${num_exemplars} cifar100_vgg19_medium
sbatch -A plgdynamic2-gpu-a100 -p plgrid-gpu-a100 scripts/templates/vgg19_cifar100_ee/er.sh ${num_tasks} 2 ${num_exemplars} cifar100_vgg19_medium

sbatch -A plgdynamic2-gpu-a100 -p plgrid-gpu-a100 scripts/templates/vgg19_cifar100_ee/er.sh ${num_tasks} 0 ${num_exemplars} cifar100_vgg19_dense
sbatch -A plgdynamic2-gpu-a100 -p plgrid-gpu-a100 scripts/templates/vgg19_cifar100_ee/er.sh ${num_tasks} 1 ${num_exemplars} cifar100_vgg19_dense
sbatch -A plgdynamic2-gpu-a100 -p plgrid-gpu-a100 scripts/templates/vgg19_cifar100_ee/er.sh ${num_tasks} 2 ${num_exemplars} cifar100_vgg19_dense

# <--- FT --->
num_exemplars=0
sbatch -A plgdynamic2-gpu-a100 -p plgrid-gpu-a100 scripts/templates/vgg19_cifar100_base/ft.sh ${num_tasks} 0 ${num_exemplars}
sbatch -A plgdynamic2-gpu-a100 -p plgrid-gpu-a100 scripts/templates/vgg19_cifar100_base/ft.sh ${num_tasks} 1 ${num_exemplars}
sbatch -A plgdynamic2-gpu-a100 -p plgrid-gpu-a100 scripts/templates/vgg19_cifar100_base/ft.sh ${num_tasks} 2 ${num_exemplars}

sbatch -A plgdynamic2-gpu-a100 -p plgrid-gpu-a100 scripts/templates/vgg19_cifar100_ee/ft.sh ${num_tasks} 0 ${num_exemplars} cifar100_vgg19_small
sbatch -A plgdynamic2-gpu-a100 -p plgrid-gpu-a100 scripts/templates/vgg19_cifar100_ee/ft.sh ${num_tasks} 1 ${num_exemplars} cifar100_vgg19_small
sbatch -A plgdynamic2-gpu-a100 -p plgrid-gpu-a100 scripts/templates/vgg19_cifar100_ee/ft.sh ${num_tasks} 2 ${num_exemplars} cifar100_vgg19_small

sbatch -A plgdynamic2-gpu-a100 -p plgrid-gpu-a100 scripts/templates/vgg19_cifar100_ee/ft.sh ${num_tasks} 0 ${num_exemplars} cifar100_vgg19_medium
sbatch -A plgdynamic2-gpu-a100 -p plgrid-gpu-a100 scripts/templates/vgg19_cifar100_ee/ft.sh ${num_tasks} 1 ${num_exemplars} cifar100_vgg19_medium
sbatch -A plgdynamic2-gpu-a100 -p plgrid-gpu-a100 scripts/templates/vgg19_cifar100_ee/ft.sh ${num_tasks} 2 ${num_exemplars} cifar100_vgg19_medium

sbatch -A plgdynamic2-gpu-a100 -p plgrid-gpu-a100 scripts/templates/vgg19_cifar100_ee/ft.sh ${num_tasks} 0 ${num_exemplars} cifar100_vgg19_dense
sbatch -A plgdynamic2-gpu-a100 -p plgrid-gpu-a100 scripts/templates/vgg19_cifar100_ee/ft.sh ${num_tasks} 1 ${num_exemplars} cifar100_vgg19_dense
sbatch -A plgdynamic2-gpu-a100 -p plgrid-gpu-a100 scripts/templates/vgg19_cifar100_ee/ft.sh ${num_tasks} 2 ${num_exemplars} cifar100_vgg19_dense

# <--- FT+Ex --->
num_exemplars=2000
sbatch -A plgdynamic2-gpu-a100 -p plgrid-gpu-a100 scripts/templates/vgg19_cifar100_base/ft.sh ${num_tasks} 0 ${num_exemplars}
sbatch -A plgdynamic2-gpu-a100 -p plgrid-gpu-a100 scripts/templates/vgg19_cifar100_base/ft.sh ${num_tasks} 1 ${num_exemplars}
sbatch -A plgdynamic2-gpu-a100 -p plgrid-gpu-a100 scripts/templates/vgg19_cifar100_base/ft.sh ${num_tasks} 2 ${num_exemplars}

sbatch -A plgdynamic2-gpu-a100 -p plgrid-gpu-a100 scripts/templates/vgg19_cifar100_ee/ft.sh ${num_tasks} 0 ${num_exemplars} cifar100_vgg19_small
sbatch -A plgdynamic2-gpu-a100 -p plgrid-gpu-a100 scripts/templates/vgg19_cifar100_ee/ft.sh ${num_tasks} 1 ${num_exemplars} cifar100_vgg19_small
sbatch -A plgdynamic2-gpu-a100 -p plgrid-gpu-a100 scripts/templates/vgg19_cifar100_ee/ft.sh ${num_tasks} 2 ${num_exemplars} cifar100_vgg19_small

sbatch -A plgdynamic2-gpu-a100 -p plgrid-gpu-a100 scripts/templates/vgg19_cifar100_ee/ft.sh ${num_tasks} 0 ${num_exemplars} cifar100_vgg19_medium
sbatch -A plgdynamic2-gpu-a100 -p plgrid-gpu-a100 scripts/templates/vgg19_cifar100_ee/ft.sh ${num_tasks} 1 ${num_exemplars} cifar100_vgg19_medium
sbatch -A plgdynamic2-gpu-a100 -p plgrid-gpu-a100 scripts/templates/vgg19_cifar100_ee/ft.sh ${num_tasks} 2 ${num_exemplars} cifar100_vgg19_medium

sbatch -A plgdynamic2-gpu-a100 -p plgrid-gpu-a100 scripts/templates/vgg19_cifar100_ee/ft.sh ${num_tasks} 0 ${num_exemplars} cifar100_vgg19_dense
sbatch -A plgdynamic2-gpu-a100 -p plgrid-gpu-a100 scripts/templates/vgg19_cifar100_ee/ft.sh ${num_tasks} 1 ${num_exemplars} cifar100_vgg19_dense
sbatch -A plgdynamic2-gpu-a100 -p plgrid-gpu-a100 scripts/templates/vgg19_cifar100_ee/ft.sh ${num_tasks} 2 ${num_exemplars} cifar100_vgg19_dense

# <--- GDumb --->
num_exemplars=2000
sbatch -A plgdynamic2-gpu-a100 -p plgrid-gpu-a100 scripts/templates/vgg19_cifar100_base/gdumb.sh ${num_tasks} 0 ${num_exemplars}
sbatch -A plgdynamic2-gpu-a100 -p plgrid-gpu-a100 scripts/templates/vgg19_cifar100_base/gdumb.sh ${num_tasks} 1 ${num_exemplars}
sbatch -A plgdynamic2-gpu-a100 -p plgrid-gpu-a100 scripts/templates/vgg19_cifar100_base/gdumb.sh ${num_tasks} 2 ${num_exemplars}

sbatch -A plgdynamic2-gpu-a100 -p plgrid-gpu-a100 scripts/templates/vgg19_cifar100_ee/gdumb.sh ${num_tasks} 0 ${num_exemplars} cifar100_vgg19_small
sbatch -A plgdynamic2-gpu-a100 -p plgrid-gpu-a100 scripts/templates/vgg19_cifar100_ee/gdumb.sh ${num_tasks} 1 ${num_exemplars} cifar100_vgg19_small
sbatch -A plgdynamic2-gpu-a100 -p plgrid-gpu-a100 scripts/templates/vgg19_cifar100_ee/gdumb.sh ${num_tasks} 2 ${num_exemplars} cifar100_vgg19_small

sbatch -A plgdynamic2-gpu-a100 -p plgrid-gpu-a100 scripts/templates/vgg19_cifar100_ee/gdumb.sh ${num_tasks} 0 ${num_exemplars} cifar100_vgg19_medium
sbatch -A plgdynamic2-gpu-a100 -p plgrid-gpu-a100 scripts/templates/vgg19_cifar100_ee/gdumb.sh ${num_tasks} 1 ${num_exemplars} cifar100_vgg19_medium
sbatch -A plgdynamic2-gpu-a100 -p plgrid-gpu-a100 scripts/templates/vgg19_cifar100_ee/gdumb.sh ${num_tasks} 2 ${num_exemplars} cifar100_vgg19_medium

sbatch -A plgdynamic2-gpu-a100 -p plgrid-gpu-a100 scripts/templates/vgg19_cifar100_ee/gdumb.sh ${num_tasks} 0 ${num_exemplars} cifar100_vgg19_dense
sbatch -A plgdynamic2-gpu-a100 -p plgrid-gpu-a100 scripts/templates/vgg19_cifar100_ee/gdumb.sh ${num_tasks} 1 ${num_exemplars} cifar100_vgg19_dense
sbatch -A plgdynamic2-gpu-a100 -p plgrid-gpu-a100 scripts/templates/vgg19_cifar100_ee/gdumb.sh ${num_tasks} 2 ${num_exemplars} cifar100_vgg19_dense

# <--- LODE --->
num_exemplars=2000
const=4.0
ro=0.1
sbatch -A plgdynamic2-gpu-a100 -p plgrid-gpu-a100 scripts/templates/vgg19_cifar100_base/lode.sh ${num_tasks} 0 ${num_exemplars} ${const} ${ro}
sbatch -A plgdynamic2-gpu-a100 -p plgrid-gpu-a100 scripts/templates/vgg19_cifar100_base/lode.sh ${num_tasks} 1 ${num_exemplars} ${const} ${ro}
sbatch -A plgdynamic2-gpu-a100 -p plgrid-gpu-a100 scripts/templates/vgg19_cifar100_base/lode.sh ${num_tasks} 2 ${num_exemplars} ${const} ${ro}

const=4.0
ro=0.1
sbatch -A plgdynamic2-gpu-a100 -p plgrid-gpu-a100 scripts/templates/vgg19_cifar100_ee/lode.sh ${num_tasks} 0 ${num_exemplars} ${const} ${ro} cifar100_vgg19_small
sbatch -A plgdynamic2-gpu-a100 -p plgrid-gpu-a100 scripts/templates/vgg19_cifar100_ee/lode.sh ${num_tasks} 1 ${num_exemplars} ${const} ${ro} cifar100_vgg19_small
sbatch -A plgdynamic2-gpu-a100 -p plgrid-gpu-a100 scripts/templates/vgg19_cifar100_ee/lode.sh ${num_tasks} 2 ${num_exemplars} ${const} ${ro} cifar100_vgg19_small

const=3.0
ro=0.5
sbatch -A plgdynamic2-gpu-a100 -p plgrid-gpu-a100 scripts/templates/vgg19_cifar100_ee/lode.sh ${num_tasks} 0 ${num_exemplars} ${const} ${ro} cifar100_vgg19_medium
sbatch -A plgdynamic2-gpu-a100 -p plgrid-gpu-a100 scripts/templates/vgg19_cifar100_ee/lode.sh ${num_tasks} 1 ${num_exemplars} ${const} ${ro} cifar100_vgg19_medium
sbatch -A plgdynamic2-gpu-a100 -p plgrid-gpu-a100 scripts/templates/vgg19_cifar100_ee/lode.sh ${num_tasks} 2 ${num_exemplars} ${const} ${ro} cifar100_vgg19_medium

const=3.0
ro=0.5
sbatch -A plgdynamic2-gpu-a100 -p plgrid-gpu-a100 scripts/templates/vgg19_cifar100_ee/lode.sh ${num_tasks} 0 ${num_exemplars} ${const} ${ro} cifar100_vgg19_dense
sbatch -A plgdynamic2-gpu-a100 -p plgrid-gpu-a100 scripts/templates/vgg19_cifar100_ee/lode.sh ${num_tasks} 1 ${num_exemplars} ${const} ${ro} cifar100_vgg19_dense
sbatch -A plgdynamic2-gpu-a100 -p plgrid-gpu-a100 scripts/templates/vgg19_cifar100_ee/lode.sh ${num_tasks} 2 ${num_exemplars} ${const} ${ro} cifar100_vgg19_dense

# <--- LwF --->
lamb=1.0
sbatch -A plgdynamic2-gpu-a100 -p plgrid-gpu-a100 scripts/templates/vgg19_cifar100_base/lwf.sh ${num_tasks} 0 ${lamb}
sbatch -A plgdynamic2-gpu-a100 -p plgrid-gpu-a100 scripts/templates/vgg19_cifar100_base/lwf.sh ${num_tasks} 1 ${lamb}
sbatch -A plgdynamic2-gpu-a100 -p plgrid-gpu-a100 scripts/templates/vgg19_cifar100_base/lwf.sh ${num_tasks} 2 ${lamb}

lamb=1.0
sbatch -A plgdynamic2-gpu-a100 -p plgrid-gpu-a100 scripts/templates/vgg19_cifar100_ee/lwf.sh ${num_tasks} 0 ${lamb} cifar100_vgg19_small
sbatch -A plgdynamic2-gpu-a100 -p plgrid-gpu-a100 scripts/templates/vgg19_cifar100_ee/lwf.sh ${num_tasks} 1 ${lamb} cifar100_vgg19_small
sbatch -A plgdynamic2-gpu-a100 -p plgrid-gpu-a100 scripts/templates/vgg19_cifar100_ee/lwf.sh ${num_tasks} 2 ${lamb} cifar100_vgg19_small

lamb=1.0
sbatch -A plgdynamic2-gpu-a100 -p plgrid-gpu-a100 scripts/templates/vgg19_cifar100_ee/lwf.sh ${num_tasks} 0 ${lamb} cifar100_vgg19_medium
sbatch -A plgdynamic2-gpu-a100 -p plgrid-gpu-a100 scripts/templates/vgg19_cifar100_ee/lwf.sh ${num_tasks} 1 ${lamb} cifar100_vgg19_medium
sbatch -A plgdynamic2-gpu-a100 -p plgrid-gpu-a100 scripts/templates/vgg19_cifar100_ee/lwf.sh ${num_tasks} 2 ${lamb} cifar100_vgg19_medium

lamb=1.0
sbatch -A plgdynamic2-gpu-a100 -p plgrid-gpu-a100 scripts/templates/vgg19_cifar100_ee/lwf.sh ${num_tasks} 0 ${lamb} cifar100_vgg19_dense
sbatch -A plgdynamic2-gpu-a100 -p plgrid-gpu-a100 scripts/templates/vgg19_cifar100_ee/lwf.sh ${num_tasks} 1 ${lamb} cifar100_vgg19_dense
sbatch -A plgdynamic2-gpu-a100 -p plgrid-gpu-a100 scripts/templates/vgg19_cifar100_ee/lwf.sh ${num_tasks} 2 ${lamb} cifar100_vgg19_dense

# <--- Joint --->
sbatch -A plgdynamic2-gpu-a100 -p plgrid-gpu-a100 scripts/templates/vgg19_cifar100_base/joint.sh ${num_tasks} 0
sbatch -A plgdynamic2-gpu-a100 -p plgrid-gpu-a100 scripts/templates/vgg19_cifar100_base/joint.sh ${num_tasks} 1
sbatch -A plgdynamic2-gpu-a100 -p plgrid-gpu-a100 scripts/templates/vgg19_cifar100_base/joint.sh ${num_tasks} 2

sbatch -A plgdynamic2-gpu-a100 -p plgrid-gpu-a100 scripts/templates/vgg19_cifar100_ee/joint.sh ${num_tasks} 0 cifar100_vgg19_small
sbatch -A plgdynamic2-gpu-a100 -p plgrid-gpu-a100 scripts/templates/vgg19_cifar100_ee/joint.sh ${num_tasks} 1 cifar100_vgg19_small
sbatch -A plgdynamic2-gpu-a100 -p plgrid-gpu-a100 scripts/templates/vgg19_cifar100_ee/joint.sh ${num_tasks} 2 cifar100_vgg19_small

sbatch -A plgdynamic2-gpu-a100 -p plgrid-gpu-a100 scripts/templates/vgg19_cifar100_ee/joint.sh ${num_tasks} 0 cifar100_vgg19_medium
sbatch -A plgdynamic2-gpu-a100 -p plgrid-gpu-a100 scripts/templates/vgg19_cifar100_ee/joint.sh ${num_tasks} 1 cifar100_vgg19_medium
sbatch -A plgdynamic2-gpu-a100 -p plgrid-gpu-a100 scripts/templates/vgg19_cifar100_ee/joint.sh ${num_tasks} 2 cifar100_vgg19_medium

sbatch -A plgdynamic2-gpu-a100 -p plgrid-gpu-a100 scripts/templates/vgg19_cifar100_ee/joint.sh ${num_tasks} 0 cifar100_vgg19_dense
sbatch -A plgdynamic2-gpu-a100 -p plgrid-gpu-a100 scripts/templates/vgg19_cifar100_ee/joint.sh ${num_tasks} 1 cifar100_vgg19_dense
sbatch -A plgdynamic2-gpu-a100 -p plgrid-gpu-a100 scripts/templates/vgg19_cifar100_ee/joint.sh ${num_tasks} 2 cifar100_vgg19_dense

# <--- SSIL --->
num_exemplars=2000
lamb=0.25
sbatch -A plgdynamic2-gpu-a100 -p plgrid-gpu-a100 scripts/templates/vgg19_cifar100_base/ssil.sh ${num_tasks} 0 ${num_exemplars} ${lamb}
sbatch -A plgdynamic2-gpu-a100 -p plgrid-gpu-a100 scripts/templates/vgg19_cifar100_base/ssil.sh ${num_tasks} 1 ${num_exemplars} ${lamb}
sbatch -A plgdynamic2-gpu-a100 -p plgrid-gpu-a100 scripts/templates/vgg19_cifar100_base/ssil.sh ${num_tasks} 2 ${num_exemplars} ${lamb}

lamb=0.5
sbatch -A plgdynamic2-gpu-a100 -p plgrid-gpu-a100 scripts/templates/vgg19_cifar100_ee/ssil.sh ${num_tasks} 0 ${num_exemplars} ${lamb} cifar100_vgg19_small
sbatch -A plgdynamic2-gpu-a100 -p plgrid-gpu-a100 scripts/templates/vgg19_cifar100_ee/ssil.sh ${num_tasks} 1 ${num_exemplars} ${lamb} cifar100_vgg19_small
sbatch -A plgdynamic2-gpu-a100 -p plgrid-gpu-a100 scripts/templates/vgg19_cifar100_ee/ssil.sh ${num_tasks} 2 ${num_exemplars} ${lamb} cifar100_vgg19_small

lamb=0.25
sbatch -A plgdynamic2-gpu-a100 -p plgrid-gpu-a100 scripts/templates/vgg19_cifar100_ee/ssil.sh ${num_tasks} 0 ${num_exemplars} ${lamb} cifar100_vgg19_medium
sbatch -A plgdynamic2-gpu-a100 -p plgrid-gpu-a100 scripts/templates/vgg19_cifar100_ee/ssil.sh ${num_tasks} 1 ${num_exemplars} ${lamb} cifar100_vgg19_medium
sbatch -A plgdynamic2-gpu-a100 -p plgrid-gpu-a100 scripts/templates/vgg19_cifar100_ee/ssil.sh ${num_tasks} 2 ${num_exemplars} ${lamb} cifar100_vgg19_medium

lamb=0.25
sbatch -A plgdynamic2-gpu-a100 -p plgrid-gpu-a100 scripts/templates/vgg19_cifar100_ee/ssil.sh ${num_tasks} 0 ${num_exemplars} ${lamb} cifar100_vgg19_dense
sbatch -A plgdynamic2-gpu-a100 -p plgrid-gpu-a100 scripts/templates/vgg19_cifar100_ee/ssil.sh ${num_tasks} 1 ${num_exemplars} ${lamb} cifar100_vgg19_dense
sbatch -A plgdynamic2-gpu-a100 -p plgrid-gpu-a100 scripts/templates/vgg19_cifar100_ee/ssil.sh ${num_tasks} 2 ${num_exemplars} ${lamb} cifar100_vgg19_dense
