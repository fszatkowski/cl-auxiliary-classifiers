#!/bin/bash

set -e

# This slurm script allows to run all main experiments on warm start setting for CIFAR100

ic_config=cifar100_resnet32_sdn
# <--- --- --- CIFAR100x6 --- --- --- --->
num_tasks=6

# <--- ANCL --->
num_exemplars=0
lamb=1.0
lamb_a=2.0
sbatch -A plgdynamic2-gpu-a100 -p plgrid-gpu-a100 scripts/templates/pretrain_cifar100_base/ancl.sh ${num_tasks} 0 ${num_exemplars} ${lamb} ${lamb_a}
sbatch -A plgdynamic2-gpu-a100 -p plgrid-gpu-a100 scripts/templates/pretrain_cifar100_base/ancl.sh ${num_tasks} 1 ${num_exemplars} ${lamb} ${lamb_a}
sbatch -A plgdynamic2-gpu-a100 -p plgrid-gpu-a100 scripts/templates/pretrain_cifar100_base/ancl.sh ${num_tasks} 2 ${num_exemplars} ${lamb} ${lamb_a}

lamb=2.0
lamb_a=0.1
sbatch -A plgdynamic2-gpu-a100 -p plgrid-gpu-a100 scripts/templates/pretrain_cifar100_ee/ancl.sh ${num_tasks} 0 ${num_exemplars} ${lamb} ${lamb_a} ${ic_config}
sbatch -A plgdynamic2-gpu-a100 -p plgrid-gpu-a100 scripts/templates/pretrain_cifar100_ee/ancl.sh ${num_tasks} 1 ${num_exemplars} ${lamb} ${lamb_a} ${ic_config}
sbatch -A plgdynamic2-gpu-a100 -p plgrid-gpu-a100 scripts/templates/pretrain_cifar100_ee/ancl.sh ${num_tasks} 2 ${num_exemplars} ${lamb} ${lamb_a} ${ic_config}

# <--- BiC --->
num_exemplars=2000
lamb=3
sbatch -A plgdynamic2-gpu-a100 -p plgrid-gpu-a100 scripts/templates/pretrain_cifar100_base/bic.sh ${num_tasks} 3 ${num_exemplars} ${lamb}
sbatch -A plgdynamic2-gpu-a100 -p plgrid-gpu-a100 scripts/templates/pretrain_cifar100_base/bic.sh ${num_tasks} 4 ${num_exemplars} ${lamb}
sbatch -A plgdynamic2-gpu-a100 -p plgrid-gpu-a100 scripts/templates/pretrain_cifar100_base/bic.sh ${num_tasks} 5 ${num_exemplars} ${lamb}

lamb=3
sbatch -A plgdynamic2-gpu-a100 -p plgrid-gpu-a100 scripts/templates/pretrain_cifar100_ee/bic.sh ${num_tasks} 3 ${num_exemplars} ${lamb} ${ic_config}
sbatch -A plgdynamic2-gpu-a100 -p plgrid-gpu-a100 scripts/templates/pretrain_cifar100_ee/bic.sh ${num_tasks} 4 ${num_exemplars} ${lamb} ${ic_config}
sbatch -A plgdynamic2-gpu-a100 -p plgrid-gpu-a100 scripts/templates/pretrain_cifar100_ee/bic.sh ${num_tasks} 5 ${num_exemplars} ${lamb} ${ic_config}

# <--- DER++ --->
num_exemplars=2000
alpha=0.5
beta=0.5
sbatch -A plgimprmoe-gpu-a100 -p plgrid-gpu-a100 scripts/templates/pretrain_cifar100_base/der++.sh ${num_tasks} 0 ${num_exemplars} ${alpha} ${beta}
sbatch -A plgimprmoe-gpu-a100 -p plgrid-gpu-a100 scripts/templates/pretrain_cifar100_base/der++.sh ${num_tasks} 1 ${num_exemplars} ${alpha} ${beta}
sbatch -A plgimprmoe-gpu-a100 -p plgrid-gpu-a100 scripts/templates/pretrain_cifar100_base/der++.sh ${num_tasks} 2 ${num_exemplars} ${alpha} ${beta}

alpha=0.5
beta=1.0
sbatch -A plgimprmoe-gpu-a100 -p plgrid-gpu-a100 scripts/templates/pretrain_cifar100_ee/der++.sh ${num_tasks} 0 ${num_exemplars} ${alpha} ${beta} ${ic_config}
sbatch -A plgimprmoe-gpu-a100 -p plgrid-gpu-a100 scripts/templates/pretrain_cifar100_ee/der++.sh ${num_tasks} 1 ${num_exemplars} ${alpha} ${beta} ${ic_config}
sbatch -A plgimprmoe-gpu-a100 -p plgrid-gpu-a100 scripts/templates/pretrain_cifar100_ee/der++.sh ${num_tasks} 2 ${num_exemplars} ${alpha} ${beta} ${ic_config}

# <--- EWC --->
lamb=10000
alpha=0.5
sbatch -A plgdynamic2-gpu-a100 -p plgrid-gpu-a100 scripts/templates/pretrain_cifar100_base/ewc.sh ${num_tasks} 0 ${lamb} ${alpha}
sbatch -A plgdynamic2-gpu-a100 -p plgrid-gpu-a100 scripts/templates/pretrain_cifar100_base/ewc.sh ${num_tasks} 1 ${lamb} ${alpha}
sbatch -A plgdynamic2-gpu-a100 -p plgrid-gpu-a100 scripts/templates/pretrain_cifar100_base/ewc.sh ${num_tasks} 2 ${lamb} ${alpha}

lamb=10000
alpha=0.5
sbatch -A plgdynamic2-gpu-a100 -p plgrid-gpu-a100 scripts/templates/pretrain_cifar100_ee/ewc.sh ${num_tasks} 0 ${lamb} ${alpha} ${ic_config}
sbatch -A plgdynamic2-gpu-a100 -p plgrid-gpu-a100 scripts/templates/pretrain_cifar100_ee/ewc.sh ${num_tasks} 1 ${lamb} ${alpha} ${ic_config}
sbatch -A plgdynamic2-gpu-a100 -p plgrid-gpu-a100 scripts/templates/pretrain_cifar100_ee/ewc.sh ${num_tasks} 2 ${lamb} ${alpha} ${ic_config}

# <--- ER --->
num_exemplars=2000
sbatch -A plgdynamic2-gpu-a100 -p plgrid-gpu-a100 scripts/templates/pretrain_cifar100_base/er.sh ${num_tasks} 0 ${num_exemplars}
sbatch -A plgdynamic2-gpu-a100 -p plgrid-gpu-a100 scripts/templates/pretrain_cifar100_base/er.sh ${num_tasks} 1 ${num_exemplars}
sbatch -A plgdynamic2-gpu-a100 -p plgrid-gpu-a100 scripts/templates/pretrain_cifar100_base/er.sh ${num_tasks} 2 ${num_exemplars}

sbatch -A plgdynamic2-gpu-a100 -p plgrid-gpu-a100 scripts/templates/pretrain_cifar100_ee/er.sh ${num_tasks} 0 ${num_exemplars} ${ic_config}
sbatch -A plgdynamic2-gpu-a100 -p plgrid-gpu-a100 scripts/templates/pretrain_cifar100_ee/er.sh ${num_tasks} 1 ${num_exemplars} ${ic_config}
sbatch -A plgdynamic2-gpu-a100 -p plgrid-gpu-a100 scripts/templates/pretrain_cifar100_ee/er.sh ${num_tasks} 2 ${num_exemplars} ${ic_config}

# <--- FT --->
num_exemplars=0
sbatch -A plgdynamic2-gpu-a100 -p plgrid-gpu-a100 scripts/templates/pretrain_cifar100_base/ft.sh ${num_tasks} 0 ${num_exemplars}
sbatch -A plgdynamic2-gpu-a100 -p plgrid-gpu-a100 scripts/templates/pretrain_cifar100_base/ft.sh ${num_tasks} 1 ${num_exemplars}
sbatch -A plgdynamic2-gpu-a100 -p plgrid-gpu-a100 scripts/templates/pretrain_cifar100_base/ft.sh ${num_tasks} 2 ${num_exemplars}

sbatch -A plgdynamic2-gpu-a100 -p plgrid-gpu-a100 scripts/templates/pretrain_cifar100_ee/ft.sh ${num_tasks} 0 ${num_exemplars} ${ic_config}
sbatch -A plgdynamic2-gpu-a100 -p plgrid-gpu-a100 scripts/templates/pretrain_cifar100_ee/ft.sh ${num_tasks} 1 ${num_exemplars} ${ic_config}
sbatch -A plgdynamic2-gpu-a100 -p plgrid-gpu-a100 scripts/templates/pretrain_cifar100_ee/ft.sh ${num_tasks} 2 ${num_exemplars} ${ic_config}

# <--- FT+Ex --->
num_exemplars=2000
sbatch -A plgdynamic2-gpu-a100 -p plgrid-gpu-a100 scripts/templates/pretrain_cifar100_base/ft.sh ${num_tasks} 0 ${num_exemplars}
sbatch -A plgdynamic2-gpu-a100 -p plgrid-gpu-a100 scripts/templates/pretrain_cifar100_base/ft.sh ${num_tasks} 1 ${num_exemplars}
sbatch -A plgdynamic2-gpu-a100 -p plgrid-gpu-a100 scripts/templates/pretrain_cifar100_base/ft.sh ${num_tasks} 2 ${num_exemplars}

sbatch -A plgdynamic2-gpu-a100 -p plgrid-gpu-a100 scripts/templates/pretrain_cifar100_ee/ft.sh ${num_tasks} 0 ${num_exemplars} ${ic_config}
sbatch -A plgdynamic2-gpu-a100 -p plgrid-gpu-a100 scripts/templates/pretrain_cifar100_ee/ft.sh ${num_tasks} 1 ${num_exemplars} ${ic_config}
sbatch -A plgdynamic2-gpu-a100 -p plgrid-gpu-a100 scripts/templates/pretrain_cifar100_ee/ft.sh ${num_tasks} 2 ${num_exemplars} ${ic_config}

# <--- GDumb --->
num_exemplars=2000
sbatch -A plgdynamic2-gpu-a100 -p plgrid-gpu-a100 scripts/templates/pretrain_cifar100_base/gdumb.sh ${num_tasks} 0 ${num_exemplars}
sbatch -A plgdynamic2-gpu-a100 -p plgrid-gpu-a100 scripts/templates/pretrain_cifar100_base/gdumb.sh ${num_tasks} 1 ${num_exemplars}
sbatch -A plgdynamic2-gpu-a100 -p plgrid-gpu-a100 scripts/templates/pretrain_cifar100_base/gdumb.sh ${num_tasks} 2 ${num_exemplars}

sbatch -A plgdynamic2-gpu-a100 -p plgrid-gpu-a100 scripts/templates/pretrain_cifar100_ee/gdumb.sh ${num_tasks} 0 ${num_exemplars} ${ic_config}
sbatch -A plgdynamic2-gpu-a100 -p plgrid-gpu-a100 scripts/templates/pretrain_cifar100_ee/gdumb.sh ${num_tasks} 1 ${num_exemplars} ${ic_config}
sbatch -A plgdynamic2-gpu-a100 -p plgrid-gpu-a100 scripts/templates/pretrain_cifar100_ee/gdumb.sh ${num_tasks} 2 ${num_exemplars} ${ic_config}

# <--- LODE --->
num_exemplars=2000
const=1.0
ro=0.5
sbatch -A plgdynamic2-gpu-a100 -p plgrid-gpu-a100 scripts/templates/pretrain_cifar100_base/lode.sh ${num_tasks} 0 ${num_exemplars} ${const} ${ro}
sbatch -A plgdynamic2-gpu-a100 -p plgrid-gpu-a100 scripts/templates/pretrain_cifar100_base/lode.sh ${num_tasks} 1 ${num_exemplars} ${const} ${ro}
sbatch -A plgdynamic2-gpu-a100 -p plgrid-gpu-a100 scripts/templates/pretrain_cifar100_base/lode.sh ${num_tasks} 2 ${num_exemplars} ${const} ${ro}

const=4.0
ro=0.5
sbatch -A plgdynamic2-gpu-a100 -p plgrid-gpu-a100 scripts/templates/pretrain_cifar100_ee/lode.sh ${num_tasks} 0 ${num_exemplars} ${const} ${ro} ${ic_config}
sbatch -A plgdynamic2-gpu-a100 -p plgrid-gpu-a100 scripts/templates/pretrain_cifar100_ee/lode.sh ${num_tasks} 1 ${num_exemplars} ${const} ${ro} ${ic_config}
sbatch -A plgdynamic2-gpu-a100 -p plgrid-gpu-a100 scripts/templates/pretrain_cifar100_ee/lode.sh ${num_tasks} 2 ${num_exemplars} ${const} ${ro} ${ic_config}

# <--- LwF --->
lamb=0.5
sbatch -A plgdynamic2-gpu-a100 -p plgrid-gpu-a100 scripts/templates/pretrain_cifar100_base/lwf.sh ${num_tasks} 0 ${lamb}
sbatch -A plgdynamic2-gpu-a100 -p plgrid-gpu-a100 scripts/templates/pretrain_cifar100_base/lwf.sh ${num_tasks} 1 ${lamb}
sbatch -A plgdynamic2-gpu-a100 -p plgrid-gpu-a100 scripts/templates/pretrain_cifar100_base/lwf.sh ${num_tasks} 2 ${lamb}

lamb=1.0
sbatch -A plgdynamic2-gpu-a100 -p plgrid-gpu-a100 scripts/templates/pretrain_cifar100_ee/lwf.sh ${num_tasks} 0 ${lamb} ${ic_config}
sbatch -A plgdynamic2-gpu-a100 -p plgrid-gpu-a100 scripts/templates/pretrain_cifar100_ee/lwf.sh ${num_tasks} 1 ${lamb} ${ic_config}
sbatch -A plgdynamic2-gpu-a100 -p plgrid-gpu-a100 scripts/templates/pretrain_cifar100_ee/lwf.sh ${num_tasks} 2 ${lamb} ${ic_config}

# <--- Joint --->
sbatch -A plgdynamic2-gpu-a100 -p plgrid-gpu-a100 scripts/templates/pretrain_cifar100_base/joint.sh ${num_tasks} 0
sbatch -A plgdynamic2-gpu-a100 -p plgrid-gpu-a100 scripts/templates/pretrain_cifar100_base/joint.sh ${num_tasks} 1
sbatch -A plgdynamic2-gpu-a100 -p plgrid-gpu-a100 scripts/templates/pretrain_cifar100_base/joint.sh ${num_tasks} 2

sbatch -A plgdynamic2-gpu-a100 -p plgrid-gpu-a100 scripts/templates/pretrain_cifar100_ee/joint.sh ${num_tasks} 0 ${ic_config}
sbatch -A plgdynamic2-gpu-a100 -p plgrid-gpu-a100 scripts/templates/pretrain_cifar100_ee/joint.sh ${num_tasks} 1 ${ic_config}
sbatch -A plgdynamic2-gpu-a100 -p plgrid-gpu-a100 scripts/templates/pretrain_cifar100_ee/joint.sh ${num_tasks} 2 ${ic_config}

# <--- SSIL --->
num_exemplars=2000
lamb=0.25
sbatch -A plgdynamic2-gpu-a100 -p plgrid-gpu-a100 scripts/templates/pretrain_cifar100_base/ssil.sh ${num_tasks} 0 ${num_exemplars} ${lamb}
sbatch -A plgdynamic2-gpu-a100 -p plgrid-gpu-a100 scripts/templates/pretrain_cifar100_base/ssil.sh ${num_tasks} 1 ${num_exemplars} ${lamb}
sbatch -A plgdynamic2-gpu-a100 -p plgrid-gpu-a100 scripts/templates/pretrain_cifar100_base/ssil.sh ${num_tasks} 2 ${num_exemplars} ${lamb}

lamb=0.25
sbatch -A plgdynamic2-gpu-a100 -p plgrid-gpu-a100 scripts/templates/pretrain_cifar100_ee/ssil.sh ${num_tasks} 0 ${num_exemplars} ${lamb} ${ic_config}
sbatch -A plgdynamic2-gpu-a100 -p plgrid-gpu-a100 scripts/templates/pretrain_cifar100_ee/ssil.sh ${num_tasks} 1 ${num_exemplars} ${lamb} ${ic_config}
sbatch -A plgdynamic2-gpu-a100 -p plgrid-gpu-a100 scripts/templates/pretrain_cifar100_ee/ssil.sh ${num_tasks} 2 ${num_exemplars} ${lamb} ${ic_config}






# <--- --- --- CIFAR100x11 --- --- --- --->
num_tasks=11

# <--- ANCL --->
num_exemplars=0
lamb=1.0
lamb_a=1.0
sbatch -A plgdynamic2-gpu-a100 -p plgrid-gpu-a100 scripts/templates/pretrain_cifar100_base/ancl.sh ${num_tasks} 0 ${num_exemplars} ${lamb} ${lamb_a}
sbatch -A plgdynamic2-gpu-a100 -p plgrid-gpu-a100 scripts/templates/pretrain_cifar100_base/ancl.sh ${num_tasks} 1 ${num_exemplars} ${lamb} ${lamb_a}
sbatch -A plgdynamic2-gpu-a100 -p plgrid-gpu-a100 scripts/templates/pretrain_cifar100_base/ancl.sh ${num_tasks} 2 ${num_exemplars} ${lamb} ${lamb_a}

lamb=2.0
lamb_a=0.1
sbatch -A plgdynamic2-gpu-a100 -p plgrid-gpu-a100 scripts/templates/pretrain_cifar100_ee/ancl.sh ${num_tasks} 0 ${num_exemplars} ${lamb} ${lamb_a} ${ic_config}
sbatch -A plgdynamic2-gpu-a100 -p plgrid-gpu-a100 scripts/templates/pretrain_cifar100_ee/ancl.sh ${num_tasks} 1 ${num_exemplars} ${lamb} ${lamb_a} ${ic_config}
sbatch -A plgdynamic2-gpu-a100 -p plgrid-gpu-a100 scripts/templates/pretrain_cifar100_ee/ancl.sh ${num_tasks} 2 ${num_exemplars} ${lamb} ${lamb_a} ${ic_config}

# <--- BiC --->
num_exemplars=2000
lamb=3
sbatch -A plgdynamic2-gpu-a100 -p plgrid-gpu-a100 scripts/templates/pretrain_cifar100_base/bic.sh ${num_tasks} 0 ${num_exemplars} ${lamb}
sbatch -A plgdynamic2-gpu-a100 -p plgrid-gpu-a100 scripts/templates/pretrain_cifar100_base/bic.sh ${num_tasks} 1 ${num_exemplars} ${lamb}
sbatch -A plgdynamic2-gpu-a100 -p plgrid-gpu-a100 scripts/templates/pretrain_cifar100_base/bic.sh ${num_tasks} 2 ${num_exemplars} ${lamb}

lamb=3
sbatch -A plgdynamic2-gpu-a100 -p plgrid-gpu-a100 scripts/templates/pretrain_cifar100_ee/bic.sh ${num_tasks} 0 ${num_exemplars} ${lamb} ${ic_config}
sbatch -A plgdynamic2-gpu-a100 -p plgrid-gpu-a100 scripts/templates/pretrain_cifar100_ee/bic.sh ${num_tasks} 1 ${num_exemplars} ${lamb} ${ic_config}
sbatch -A plgdynamic2-gpu-a100 -p plgrid-gpu-a100 scripts/templates/pretrain_cifar100_ee/bic.sh ${num_tasks} 2 ${num_exemplars} ${lamb} ${ic_config}

# <--- DER++ --->
num_exemplars=2000
alpha=0.5
beta=0.5
sbatch -A plgimprmoe-gpu-a100 -p plgrid-gpu-a100 scripts/templates/pretrain_cifar100_base/der++.sh ${num_tasks} 0 ${num_exemplars} ${alpha} ${beta}
sbatch -A plgimprmoe-gpu-a100 -p plgrid-gpu-a100 scripts/templates/pretrain_cifar100_base/der++.sh ${num_tasks} 1 ${num_exemplars} ${alpha} ${beta}
sbatch -A plgimprmoe-gpu-a100 -p plgrid-gpu-a100 scripts/templates/pretrain_cifar100_base/der++.sh ${num_tasks} 2 ${num_exemplars} ${alpha} ${beta}

alpha=0.5
beta=1.0
sbatch -A plgimprmoe-gpu-a100 -p plgrid-gpu-a100 scripts/templates/pretrain_cifar100_ee/der++.sh ${num_tasks} 0 ${num_exemplars} ${alpha} ${beta} ${ic_config}
sbatch -A plgimprmoe-gpu-a100 -p plgrid-gpu-a100 scripts/templates/pretrain_cifar100_ee/der++.sh ${num_tasks} 1 ${num_exemplars} ${alpha} ${beta} ${ic_config}
sbatch -A plgimprmoe-gpu-a100 -p plgrid-gpu-a100 scripts/templates/pretrain_cifar100_ee/der++.sh ${num_tasks} 2 ${num_exemplars} ${alpha} ${beta} ${ic_config}

# <--- EWC --->
lamb=10000
alpha=0.5
sbatch -A plgdynamic2-gpu-a100 -p plgrid-gpu-a100 scripts/templates/pretrain_cifar100_base/ewc.sh ${num_tasks} 0 ${lamb} ${alpha}
sbatch -A plgdynamic2-gpu-a100 -p plgrid-gpu-a100 scripts/templates/pretrain_cifar100_base/ewc.sh ${num_tasks} 1 ${lamb} ${alpha}
sbatch -A plgdynamic2-gpu-a100 -p plgrid-gpu-a100 scripts/templates/pretrain_cifar100_base/ewc.sh ${num_tasks} 2 ${lamb} ${alpha}

lamb=10000
alpha=0.5
sbatch -A plgdynamic2-gpu-a100 -p plgrid-gpu-a100 scripts/templates/pretrain_cifar100_ee/ewc.sh ${num_tasks} 0 ${lamb} ${alpha} ${ic_config}
sbatch -A plgdynamic2-gpu-a100 -p plgrid-gpu-a100 scripts/templates/pretrain_cifar100_ee/ewc.sh ${num_tasks} 1 ${lamb} ${alpha} ${ic_config}
sbatch -A plgdynamic2-gpu-a100 -p plgrid-gpu-a100 scripts/templates/pretrain_cifar100_ee/ewc.sh ${num_tasks} 2 ${lamb} ${alpha} ${ic_config}

# <--- ER --->
num_exemplars=2000
sbatch -A plgdynamic2-gpu-a100 -p plgrid-gpu-a100 scripts/templates/pretrain_cifar100_base/er.sh ${num_tasks} 0 ${num_exemplars}
sbatch -A plgdynamic2-gpu-a100 -p plgrid-gpu-a100 scripts/templates/pretrain_cifar100_base/er.sh ${num_tasks} 1 ${num_exemplars}
sbatch -A plgdynamic2-gpu-a100 -p plgrid-gpu-a100 scripts/templates/pretrain_cifar100_base/er.sh ${num_tasks} 2 ${num_exemplars}

sbatch -A plgdynamic2-gpu-a100 -p plgrid-gpu-a100 scripts/templates/pretrain_cifar100_ee/er.sh ${num_tasks} 0 ${num_exemplars} ${ic_config}
sbatch -A plgdynamic2-gpu-a100 -p plgrid-gpu-a100 scripts/templates/pretrain_cifar100_ee/er.sh ${num_tasks} 1 ${num_exemplars} ${ic_config}
sbatch -A plgdynamic2-gpu-a100 -p plgrid-gpu-a100 scripts/templates/pretrain_cifar100_ee/er.sh ${num_tasks} 2 ${num_exemplars} ${ic_config}

# <--- FT --->
num_exemplars=0
sbatch -A plgdynamic2-gpu-a100 -p plgrid-gpu-a100 scripts/templates/pretrain_cifar100_base/ft.sh ${num_tasks} 0 ${num_exemplars}
sbatch -A plgdynamic2-gpu-a100 -p plgrid-gpu-a100 scripts/templates/pretrain_cifar100_base/ft.sh ${num_tasks} 1 ${num_exemplars}
sbatch -A plgdynamic2-gpu-a100 -p plgrid-gpu-a100 scripts/templates/pretrain_cifar100_base/ft.sh ${num_tasks} 2 ${num_exemplars}

sbatch -A plgdynamic2-gpu-a100 -p plgrid-gpu-a100 scripts/templates/pretrain_cifar100_ee/ft.sh ${num_tasks} 0 ${num_exemplars} ${ic_config}
sbatch -A plgdynamic2-gpu-a100 -p plgrid-gpu-a100 scripts/templates/pretrain_cifar100_ee/ft.sh ${num_tasks} 1 ${num_exemplars} ${ic_config}
sbatch -A plgdynamic2-gpu-a100 -p plgrid-gpu-a100 scripts/templates/pretrain_cifar100_ee/ft.sh ${num_tasks} 2 ${num_exemplars} ${ic_config}

# <--- FT+Ex --->
num_exemplars=2000
sbatch -A plgdynamic2-gpu-a100 -p plgrid-gpu-a100 scripts/templates/pretrain_cifar100_base/ft.sh ${num_tasks} 0 ${num_exemplars}
sbatch -A plgdynamic2-gpu-a100 -p plgrid-gpu-a100 scripts/templates/pretrain_cifar100_base/ft.sh ${num_tasks} 1 ${num_exemplars}
sbatch -A plgdynamic2-gpu-a100 -p plgrid-gpu-a100 scripts/templates/pretrain_cifar100_base/ft.sh ${num_tasks} 2 ${num_exemplars}

sbatch -A plgdynamic2-gpu-a100 -p plgrid-gpu-a100 scripts/templates/pretrain_cifar100_ee/ft.sh ${num_tasks} 0 ${num_exemplars} ${ic_config}
sbatch -A plgdynamic2-gpu-a100 -p plgrid-gpu-a100 scripts/templates/pretrain_cifar100_ee/ft.sh ${num_tasks} 1 ${num_exemplars} ${ic_config}
sbatch -A plgdynamic2-gpu-a100 -p plgrid-gpu-a100 scripts/templates/pretrain_cifar100_ee/ft.sh ${num_tasks} 2 ${num_exemplars} ${ic_config}

# <--- GDumb --->
num_exemplars=2000
sbatch -A plgdynamic2-gpu-a100 -p plgrid-gpu-a100 scripts/templates/pretrain_cifar100_base/gdumb.sh ${num_tasks} 0 ${num_exemplars}
sbatch -A plgdynamic2-gpu-a100 -p plgrid-gpu-a100 scripts/templates/pretrain_cifar100_base/gdumb.sh ${num_tasks} 1 ${num_exemplars}
sbatch -A plgdynamic2-gpu-a100 -p plgrid-gpu-a100 scripts/templates/pretrain_cifar100_base/gdumb.sh ${num_tasks} 2 ${num_exemplars}

sbatch -A plgdynamic2-gpu-a100 -p plgrid-gpu-a100 scripts/templates/pretrain_cifar100_ee/gdumb.sh ${num_tasks} 0 ${num_exemplars} ${ic_config}
sbatch -A plgdynamic2-gpu-a100 -p plgrid-gpu-a100 scripts/templates/pretrain_cifar100_ee/gdumb.sh ${num_tasks} 1 ${num_exemplars} ${ic_config}
sbatch -A plgdynamic2-gpu-a100 -p plgrid-gpu-a100 scripts/templates/pretrain_cifar100_ee/gdumb.sh ${num_tasks} 2 ${num_exemplars} ${ic_config}

# <--- LODE --->
num_exemplars=2000
const=1.0
ro=0.5
sbatch -A plgdynamic2-gpu-a100 -p plgrid-gpu-a100 scripts/templates/pretrain_cifar100_base/lode.sh ${num_tasks} 0 ${num_exemplars} ${const} ${ro}
sbatch -A plgdynamic2-gpu-a100 -p plgrid-gpu-a100 scripts/templates/pretrain_cifar100_base/lode.sh ${num_tasks} 1 ${num_exemplars} ${const} ${ro}
sbatch -A plgdynamic2-gpu-a100 -p plgrid-gpu-a100 scripts/templates/pretrain_cifar100_base/lode.sh ${num_tasks} 2 ${num_exemplars} ${const} ${ro}

const=4.0
ro=0.5
sbatch -A plgdynamic2-gpu-a100 -p plgrid-gpu-a100 scripts/templates/pretrain_cifar100_ee/lode.sh ${num_tasks} 0 ${num_exemplars} ${const} ${ro} ${ic_config}
sbatch -A plgdynamic2-gpu-a100 -p plgrid-gpu-a100 scripts/templates/pretrain_cifar100_ee/lode.sh ${num_tasks} 1 ${num_exemplars} ${const} ${ro} ${ic_config}
sbatch -A plgdynamic2-gpu-a100 -p plgrid-gpu-a100 scripts/templates/pretrain_cifar100_ee/lode.sh ${num_tasks} 2 ${num_exemplars} ${const} ${ro} ${ic_config}

# <--- LwF --->
lamb=0.5
sbatch -A plgdynamic2-gpu-a100 -p plgrid-gpu-a100 scripts/templates/pretrain_cifar100_base/lwf.sh ${num_tasks} 0 ${lamb}
sbatch -A plgdynamic2-gpu-a100 -p plgrid-gpu-a100 scripts/templates/pretrain_cifar100_base/lwf.sh ${num_tasks} 1 ${lamb}
sbatch -A plgdynamic2-gpu-a100 -p plgrid-gpu-a100 scripts/templates/pretrain_cifar100_base/lwf.sh ${num_tasks} 2 ${lamb}

lamb=1.0
sbatch -A plgdynamic2-gpu-a100 -p plgrid-gpu-a100 scripts/templates/pretrain_cifar100_ee/lwf.sh ${num_tasks} 0 ${lamb} ${ic_config}
sbatch -A plgdynamic2-gpu-a100 -p plgrid-gpu-a100 scripts/templates/pretrain_cifar100_ee/lwf.sh ${num_tasks} 1 ${lamb} ${ic_config}
sbatch -A plgdynamic2-gpu-a100 -p plgrid-gpu-a100 scripts/templates/pretrain_cifar100_ee/lwf.sh ${num_tasks} 2 ${lamb} ${ic_config}

# <--- Joint --->
sbatch -A plgdynamic2-gpu-a100 -p plgrid-gpu-a100 scripts/templates/pretrain_cifar100_base/joint.sh ${num_tasks} 0
sbatch -A plgdynamic2-gpu-a100 -p plgrid-gpu-a100 scripts/templates/pretrain_cifar100_base/joint.sh ${num_tasks} 1
sbatch -A plgdynamic2-gpu-a100 -p plgrid-gpu-a100 scripts/templates/pretrain_cifar100_base/joint.sh ${num_tasks} 2

sbatch -A plgdynamic2-gpu-a100 -p plgrid-gpu-a100 scripts/templates/pretrain_cifar100_ee/joint.sh ${num_tasks} 0 ${ic_config}
sbatch -A plgdynamic2-gpu-a100 -p plgrid-gpu-a100 scripts/templates/pretrain_cifar100_ee/joint.sh ${num_tasks} 1 ${ic_config}
sbatch -A plgdynamic2-gpu-a100 -p plgrid-gpu-a100 scripts/templates/pretrain_cifar100_ee/joint.sh ${num_tasks} 2 ${ic_config}

# <--- SSIL --->
num_exemplars=2000
lamb=0.25
sbatch -A plgdynamic2-gpu-a100 -p plgrid-gpu-a100 scripts/templates/pretrain_cifar100_base/ssil.sh ${num_tasks} 0 ${num_exemplars} ${lamb}
sbatch -A plgdynamic2-gpu-a100 -p plgrid-gpu-a100 scripts/templates/pretrain_cifar100_base/ssil.sh ${num_tasks} 1 ${num_exemplars} ${lamb}
sbatch -A plgdynamic2-gpu-a100 -p plgrid-gpu-a100 scripts/templates/pretrain_cifar100_base/ssil.sh ${num_tasks} 2 ${num_exemplars} ${lamb}

lamb=0.25
sbatch -A plgdynamic2-gpu-a100 -p plgrid-gpu-a100 scripts/templates/pretrain_cifar100_ee/ssil.sh ${num_tasks} 0 ${num_exemplars} ${lamb} ${ic_config}
sbatch -A plgdynamic2-gpu-a100 -p plgrid-gpu-a100 scripts/templates/pretrain_cifar100_ee/ssil.sh ${num_tasks} 1 ${num_exemplars} ${lamb} ${ic_config}
sbatch -A plgdynamic2-gpu-a100 -p plgrid-gpu-a100 scripts/templates/pretrain_cifar100_ee/ssil.sh ${num_tasks} 2 ${num_exemplars} ${lamb} ${ic_config}
