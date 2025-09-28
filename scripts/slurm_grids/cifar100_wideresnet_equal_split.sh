#!/bin/bash

set -e

# This slurm script allows to run all main experiments on equal split of CIFAR100

slurm_acc=...
slurm_partition=...

ic_config=cifar100_wideresnet16_sdn
# <--- --- --- CIFAR100x5 --- --- --- --->
num_tasks=5

# <--- ANCL --->
num_exemplars=0
lamb=1.0
lamb_a=2.0
sbatch -A ${slurm_acc} -p ${slurm_partition} scripts/templates/wideresnet_cifar100_base/ancl.sh ${num_tasks} 0 ${num_exemplars} ${lamb} ${lamb_a}
sbatch -A ${slurm_acc} -p ${slurm_partition} scripts/templates/wideresnet_cifar100_base/ancl.sh ${num_tasks} 1 ${num_exemplars} ${lamb} ${lamb_a}
sbatch -A ${slurm_acc} -p ${slurm_partition} scripts/templates/wideresnet_cifar100_base/ancl.sh ${num_tasks} 2 ${num_exemplars} ${lamb} ${lamb_a}

lamb=1.0
lamb_a=1.0
sbatch -A ${slurm_acc} -p ${slurm_partition} scripts/templates/wideresnet_cifar100_ee/ancl.sh ${num_tasks} 0 ${num_exemplars} ${lamb} ${lamb_a} ${ic_config}
sbatch -A ${slurm_acc} -p ${slurm_partition} scripts/templates/wideresnet_cifar100_ee/ancl.sh ${num_tasks} 1 ${num_exemplars} ${lamb} ${lamb_a} ${ic_config}
sbatch -A ${slurm_acc} -p ${slurm_partition} scripts/templates/wideresnet_cifar100_ee/ancl.sh ${num_tasks} 2 ${num_exemplars} ${lamb} ${lamb_a} ${ic_config}

# <--- BiC --->
num_exemplars=2000
lamb=2
sbatch -A ${slurm_acc} -p ${slurm_partition} scripts/templates/wideresnet_cifar100_base/bic.sh ${num_tasks} 0 ${num_exemplars} ${lamb}
sbatch -A ${slurm_acc} -p ${slurm_partition} scripts/templates/wideresnet_cifar100_base/bic.sh ${num_tasks} 1 ${num_exemplars} ${lamb}
sbatch -A ${slurm_acc} -p ${slurm_partition} scripts/templates/wideresnet_cifar100_base/bic.sh ${num_tasks} 2 ${num_exemplars} ${lamb}

lamb=2
sbatch -A ${slurm_acc} -p ${slurm_partition} scripts/templates/wideresnet_cifar100_ee/bic.sh ${num_tasks} 0 ${num_exemplars} ${lamb} ${ic_config}
sbatch -A ${slurm_acc} -p ${slurm_partition} scripts/templates/wideresnet_cifar100_ee/bic.sh ${num_tasks} 1 ${num_exemplars} ${lamb} ${ic_config}
sbatch -A ${slurm_acc} -p ${slurm_partition} scripts/templates/wideresnet_cifar100_ee/bic.sh ${num_tasks} 2 ${num_exemplars} ${lamb} ${ic_config}

# <--- DER++ --->
num_exemplars=2000
alpha=0.5
beta=0.5
sbatch -A ${slurm_acc} -p ${slurm_partition} scripts/templates/wideresnet_cifar100_base/der++.sh ${num_tasks} 1 ${num_exemplars} ${alpha} ${beta}
sbatch -A ${slurm_acc} -p ${slurm_partition} scripts/templates/wideresnet_cifar100_base/der++.sh ${num_tasks} 4 ${num_exemplars} ${alpha} ${beta}
sbatch -A ${slurm_acc} -p ${slurm_partition} scripts/templates/wideresnet_cifar100_base/der++.sh ${num_tasks} 5 ${num_exemplars} ${alpha} ${beta}

alpha=0.1
beta=0.5
sbatch -A ${slurm_acc} -p ${slurm_partition} scripts/templates/wideresnet_cifar100_ee/der++.sh ${num_tasks} 1 ${num_exemplars} ${alpha} ${beta} ${ic_config}
sbatch -A ${slurm_acc} -p ${slurm_partition} scripts/templates/wideresnet_cifar100_ee/der++.sh ${num_tasks} 4 ${num_exemplars} ${alpha} ${beta} ${ic_config}
sbatch -A ${slurm_acc} -p ${slurm_partition} scripts/templates/wideresnet_cifar100_ee/der++.sh ${num_tasks} 5 ${num_exemplars} ${alpha} ${beta} ${ic_config}

# <--- EWC --->
lamb=10000
alpha=0.5
sbatch -A ${slurm_acc} -p ${slurm_partition} scripts/templates/wideresnet_cifar100_base/ewc.sh ${num_tasks} 0 ${lamb} ${alpha}
sbatch -A ${slurm_acc} -p ${slurm_partition} scripts/templates/wideresnet_cifar100_base/ewc.sh ${num_tasks} 1 ${lamb} ${alpha}
sbatch -A ${slurm_acc} -p ${slurm_partition} scripts/templates/wideresnet_cifar100_base/ewc.sh ${num_tasks} 2 ${lamb} ${alpha}

lamb=10000
alpha=0.5
sbatch -A ${slurm_acc} -p ${slurm_partition} scripts/templates/wideresnet_cifar100_ee/ewc.sh ${num_tasks} 0 ${lamb} ${alpha} ${ic_config}
sbatch -A ${slurm_acc} -p ${slurm_partition} scripts/templates/wideresnet_cifar100_ee/ewc.sh ${num_tasks} 1 ${lamb} ${alpha} ${ic_config}
sbatch -A ${slurm_acc} -p ${slurm_partition} scripts/templates/wideresnet_cifar100_ee/ewc.sh ${num_tasks} 2 ${lamb} ${alpha} ${ic_config}

# <--- ER --->
num_exemplars=2000
sbatch -A ${slurm_acc} -p ${slurm_partition} scripts/templates/wideresnet_cifar100_base/er.sh ${num_tasks} 0 ${num_exemplars}
sbatch -A ${slurm_acc} -p ${slurm_partition} scripts/templates/wideresnet_cifar100_base/er.sh ${num_tasks} 1 ${num_exemplars}
sbatch -A ${slurm_acc} -p ${slurm_partition} scripts/templates/wideresnet_cifar100_base/er.sh ${num_tasks} 2 ${num_exemplars}

sbatch -A ${slurm_acc} -p ${slurm_partition} scripts/templates/wideresnet_cifar100_ee/er.sh ${num_tasks} 0 ${num_exemplars} ${ic_config}
sbatch -A ${slurm_acc} -p ${slurm_partition} scripts/templates/wideresnet_cifar100_ee/er.sh ${num_tasks} 1 ${num_exemplars} ${ic_config}
sbatch -A ${slurm_acc} -p ${slurm_partition} scripts/templates/wideresnet_cifar100_ee/er.sh ${num_tasks} 2 ${num_exemplars} ${ic_config}

# <--- FT --->
num_exemplars=0
sbatch -A ${slurm_acc} -p ${slurm_partition} scripts/templates/wideresnet_cifar100_base/ft.sh ${num_tasks} 0 ${num_exemplars}
sbatch -A ${slurm_acc} -p ${slurm_partition} scripts/templates/wideresnet_cifar100_base/ft.sh ${num_tasks} 1 ${num_exemplars}
sbatch -A ${slurm_acc} -p ${slurm_partition} scripts/templates/wideresnet_cifar100_base/ft.sh ${num_tasks} 2 ${num_exemplars}

sbatch -A ${slurm_acc} -p ${slurm_partition} scripts/templates/wideresnet_cifar100_ee/ft.sh ${num_tasks} 0 ${num_exemplars} ${ic_config}
sbatch -A ${slurm_acc} -p ${slurm_partition} scripts/templates/wideresnet_cifar100_ee/ft.sh ${num_tasks} 1 ${num_exemplars} ${ic_config}
sbatch -A ${slurm_acc} -p ${slurm_partition} scripts/templates/wideresnet_cifar100_ee/ft.sh ${num_tasks} 2 ${num_exemplars} ${ic_config}

# <--- FT+Ex --->
num_exemplars=2000
sbatch -A ${slurm_acc} -p ${slurm_partition} scripts/templates/wideresnet_cifar100_base/ft.sh ${num_tasks} 0 ${num_exemplars}
sbatch -A ${slurm_acc} -p ${slurm_partition} scripts/templates/wideresnet_cifar100_base/ft.sh ${num_tasks} 1 ${num_exemplars}
sbatch -A ${slurm_acc} -p ${slurm_partition} scripts/templates/wideresnet_cifar100_base/ft.sh ${num_tasks} 2 ${num_exemplars}

sbatch -A ${slurm_acc} -p ${slurm_partition} scripts/templates/wideresnet_cifar100_ee/ft.sh ${num_tasks} 0 ${num_exemplars} ${ic_config}
sbatch -A ${slurm_acc} -p ${slurm_partition} scripts/templates/wideresnet_cifar100_ee/ft.sh ${num_tasks} 1 ${num_exemplars} ${ic_config}
sbatch -A ${slurm_acc} -p ${slurm_partition} scripts/templates/wideresnet_cifar100_ee/ft.sh ${num_tasks} 2 ${num_exemplars} ${ic_config}

# <--- GDumb --->
num_exemplars=2000
sbatch -A ${slurm_acc} -p ${slurm_partition} scripts/templates/wideresnet_cifar100_base/gdumb.sh ${num_tasks} 0 ${num_exemplars}
sbatch -A ${slurm_acc} -p ${slurm_partition} scripts/templates/wideresnet_cifar100_base/gdumb.sh ${num_tasks} 1 ${num_exemplars}
sbatch -A ${slurm_acc} -p ${slurm_partition} scripts/templates/wideresnet_cifar100_base/gdumb.sh ${num_tasks} 2 ${num_exemplars}

sbatch -A ${slurm_acc} -p ${slurm_partition} scripts/templates/wideresnet_cifar100_ee/gdumb.sh ${num_tasks} 0 ${num_exemplars} ${ic_config}
sbatch -A ${slurm_acc} -p ${slurm_partition} scripts/templates/wideresnet_cifar100_ee/gdumb.sh ${num_tasks} 1 ${num_exemplars} ${ic_config}
sbatch -A ${slurm_acc} -p ${slurm_partition} scripts/templates/wideresnet_cifar100_ee/gdumb.sh ${num_tasks} 2 ${num_exemplars} ${ic_config}

# <--- LODE --->
num_exemplars=2000
const=1.0
ro=0.1
sbatch -A ${slurm_acc} -p ${slurm_partition} scripts/templates/wideresnet_cifar100_base/lode.sh ${num_tasks} 0 ${num_exemplars} ${const} ${ro}
sbatch -A ${slurm_acc} -p ${slurm_partition} scripts/templates/wideresnet_cifar100_base/lode.sh ${num_tasks} 1 ${num_exemplars} ${const} ${ro}
sbatch -A ${slurm_acc} -p ${slurm_partition} scripts/templates/wideresnet_cifar100_base/lode.sh ${num_tasks} 2 ${num_exemplars} ${const} ${ro}

const=4.0
ro=0.5
sbatch -A ${slurm_acc} -p ${slurm_partition} scripts/templates/wideresnet_cifar100_ee/lode.sh ${num_tasks} 0 ${num_exemplars} ${const} ${ro} ${ic_config}
sbatch -A ${slurm_acc} -p ${slurm_partition} scripts/templates/wideresnet_cifar100_ee/lode.sh ${num_tasks} 1 ${num_exemplars} ${const} ${ro} ${ic_config}
sbatch -A ${slurm_acc} -p ${slurm_partition} scripts/templates/wideresnet_cifar100_ee/lode.sh ${num_tasks} 2 ${num_exemplars} ${const} ${ro} ${ic_config}

# <--- LwF --->
lamb=1.0
sbatch -A ${slurm_acc} -p ${slurm_partition} scripts/templates/wideresnet_cifar100_base/lwf.sh ${num_tasks} 0 ${lamb}
sbatch -A ${slurm_acc} -p ${slurm_partition} scripts/templates/wideresnet_cifar100_base/lwf.sh ${num_tasks} 1 ${lamb}
sbatch -A ${slurm_acc} -p ${slurm_partition} scripts/templates/wideresnet_cifar100_base/lwf.sh ${num_tasks} 2 ${lamb}

lamb=1.0
sbatch -A ${slurm_acc} -p ${slurm_partition} scripts/templates/wideresnet_cifar100_ee/lwf.sh ${num_tasks} 0 ${lamb} ${ic_config}
sbatch -A ${slurm_acc} -p ${slurm_partition} scripts/templates/wideresnet_cifar100_ee/lwf.sh ${num_tasks} 1 ${lamb} ${ic_config}
sbatch -A ${slurm_acc} -p ${slurm_partition} scripts/templates/wideresnet_cifar100_ee/lwf.sh ${num_tasks} 2 ${lamb} ${ic_config}

# <--- Joint --->
# sbatch -A ${slurm_acc} -p ${slurm_partition} scripts/templates/wideresnet_cifar100_base/joint.sh ${num_tasks} 0
# sbatch -A ${slurm_acc} -p ${slurm_partition} scripts/templates/wideresnet_cifar100_base/joint.sh ${num_tasks} 1
# sbatch -A ${slurm_acc} -p ${slurm_partition} scripts/templates/wideresnet_cifar100_base/joint.sh ${num_tasks} 2
#
# sbatch -A ${slurm_acc} -p ${slurm_partition} scripts/templates/wideresnet_cifar100_ee/joint.sh ${num_tasks} 0 ${ic_config}
# sbatch -A ${slurm_acc} -p ${slurm_partition} scripts/templates/wideresnet_cifar100_ee/joint.sh ${num_tasks} 1 ${ic_config}
# sbatch -A ${slurm_acc} -p ${slurm_partition} scripts/templates/wideresnet_cifar100_ee/joint.sh ${num_tasks} 2 ${ic_config}

# <--- SSIL --->
num_exemplars=2000
lamb=0.25
sbatch -A ${slurm_acc} -p ${slurm_partition} scripts/templates/wideresnet_cifar100_base/ssil.sh ${num_tasks} 0 ${num_exemplars} ${lamb}
sbatch -A ${slurm_acc} -p ${slurm_partition} scripts/templates/wideresnet_cifar100_base/ssil.sh ${num_tasks} 1 ${num_exemplars} ${lamb}
sbatch -A ${slurm_acc} -p ${slurm_partition} scripts/templates/wideresnet_cifar100_base/ssil.sh ${num_tasks} 2 ${num_exemplars} ${lamb}

lamb=0.25
sbatch -A ${slurm_acc} -p ${slurm_partition} scripts/templates/wideresnet_cifar100_ee/ssil.sh ${num_tasks} 0 ${num_exemplars} ${lamb} ${ic_config}
sbatch -A ${slurm_acc} -p ${slurm_partition} scripts/templates/wideresnet_cifar100_ee/ssil.sh ${num_tasks} 1 ${num_exemplars} ${lamb} ${ic_config}
sbatch -A ${slurm_acc} -p ${slurm_partition} scripts/templates/wideresnet_cifar100_ee/ssil.sh ${num_tasks} 2 ${num_exemplars} ${lamb} ${ic_config}






# <--- --- --- CIFAR100x10 --- --- --- --->
num_tasks=10

# <--- ANCL --->
num_exemplars=0
lamb=1.0
lamb_a=2.0
sbatch -A ${slurm_acc} -p ${slurm_partition} scripts/templates/wideresnet_cifar100_base/ancl.sh ${num_tasks} 0 ${num_exemplars} ${lamb} ${lamb_a}
sbatch -A ${slurm_acc} -p ${slurm_partition} scripts/templates/wideresnet_cifar100_base/ancl.sh ${num_tasks} 1 ${num_exemplars} ${lamb} ${lamb_a}
sbatch -A ${slurm_acc} -p ${slurm_partition} scripts/templates/wideresnet_cifar100_base/ancl.sh ${num_tasks} 2 ${num_exemplars} ${lamb} ${lamb_a}

lamb=1.0
lamb_a=1.0
sbatch -A ${slurm_acc} -p ${slurm_partition} scripts/templates/wideresnet_cifar100_ee/ancl.sh ${num_tasks} 0 ${num_exemplars} ${lamb} ${lamb_a} ${ic_config}
sbatch -A ${slurm_acc} -p ${slurm_partition} scripts/templates/wideresnet_cifar100_ee/ancl.sh ${num_tasks} 1 ${num_exemplars} ${lamb} ${lamb_a} ${ic_config}
sbatch -A ${slurm_acc} -p ${slurm_partition} scripts/templates/wideresnet_cifar100_ee/ancl.sh ${num_tasks} 2 ${num_exemplars} ${lamb} ${lamb_a} ${ic_config}

# <--- BiC --->
num_exemplars=2000
lamb=2
sbatch -A ${slurm_acc} -p ${slurm_partition} scripts/templates/wideresnet_cifar100_base/bic.sh ${num_tasks} 0 ${num_exemplars} ${lamb}
sbatch -A ${slurm_acc} -p ${slurm_partition} scripts/templates/wideresnet_cifar100_base/bic.sh ${num_tasks} 1 ${num_exemplars} ${lamb}
sbatch -A ${slurm_acc} -p ${slurm_partition} scripts/templates/wideresnet_cifar100_base/bic.sh ${num_tasks} 2 ${num_exemplars} ${lamb}

lamb=2
sbatch -A ${slurm_acc} -p ${slurm_partition} scripts/templates/wideresnet_cifar100_ee/bic.sh ${num_tasks} 0 ${num_exemplars} ${lamb} ${ic_config}
sbatch -A ${slurm_acc} -p ${slurm_partition} scripts/templates/wideresnet_cifar100_ee/bic.sh ${num_tasks} 1 ${num_exemplars} ${lamb} ${ic_config}
sbatch -A ${slurm_acc} -p ${slurm_partition} scripts/templates/wideresnet_cifar100_ee/bic.sh ${num_tasks} 2 ${num_exemplars} ${lamb} ${ic_config}

# <--- DER++ --->
num_exemplars=2000
alpha=0.5
beta=0.5
sbatch -A ${slurm_acc} -p ${slurm_partition} scripts/templates/wideresnet_cifar100_base/der++.sh ${num_tasks} 0 ${num_exemplars} ${alpha} ${beta}
sbatch -A ${slurm_acc} -p ${slurm_partition} scripts/templates/wideresnet_cifar100_base/der++.sh ${num_tasks} 1 ${num_exemplars} ${alpha} ${beta}
sbatch -A ${slurm_acc} -p ${slurm_partition} scripts/templates/wideresnet_cifar100_base/der++.sh ${num_tasks} 2 ${num_exemplars} ${alpha} ${beta}

alpha=0.1
beta=0.5
sbatch -A ${slurm_acc} -p ${slurm_partition} scripts/templates/wideresnet_cifar100_ee/der++.sh ${num_tasks} 0 ${num_exemplars} ${alpha} ${beta} ${ic_config}
sbatch -A ${slurm_acc} -p ${slurm_partition} scripts/templates/wideresnet_cifar100_ee/der++.sh ${num_tasks} 1 ${num_exemplars} ${alpha} ${beta} ${ic_config}
sbatch -A ${slurm_acc} -p ${slurm_partition} scripts/templates/wideresnet_cifar100_ee/der++.sh ${num_tasks} 2 ${num_exemplars} ${alpha} ${beta} ${ic_config}

# <--- EWC --->
lamb=10000
alpha=0.5
sbatch -A ${slurm_acc} -p ${slurm_partition} scripts/templates/wideresnet_cifar100_base/ewc.sh ${num_tasks} 0 ${lamb} ${alpha}
sbatch -A ${slurm_acc} -p ${slurm_partition} scripts/templates/wideresnet_cifar100_base/ewc.sh ${num_tasks} 1 ${lamb} ${alpha}
sbatch -A ${slurm_acc} -p ${slurm_partition} scripts/templates/wideresnet_cifar100_base/ewc.sh ${num_tasks} 2 ${lamb} ${alpha}

lamb=10000
alpha=0.5
sbatch -A ${slurm_acc} -p ${slurm_partition} scripts/templates/wideresnet_cifar100_ee/ewc.sh ${num_tasks} 0 ${lamb} ${alpha} ${ic_config}
sbatch -A ${slurm_acc} -p ${slurm_partition} scripts/templates/wideresnet_cifar100_ee/ewc.sh ${num_tasks} 1 ${lamb} ${alpha} ${ic_config}
sbatch -A ${slurm_acc} -p ${slurm_partition} scripts/templates/wideresnet_cifar100_ee/ewc.sh ${num_tasks} 2 ${lamb} ${alpha} ${ic_config}

# <--- ER --->
num_exemplars=2000
sbatch -A ${slurm_acc} -p ${slurm_partition} scripts/templates/wideresnet_cifar100_base/er.sh ${num_tasks} 0 ${num_exemplars}
sbatch -A ${slurm_acc} -p ${slurm_partition} scripts/templates/wideresnet_cifar100_base/er.sh ${num_tasks} 1 ${num_exemplars}
sbatch -A ${slurm_acc} -p ${slurm_partition} scripts/templates/wideresnet_cifar100_base/er.sh ${num_tasks} 2 ${num_exemplars}

sbatch -A ${slurm_acc} -p ${slurm_partition} scripts/templates/wideresnet_cifar100_ee/er.sh ${num_tasks} 0 ${num_exemplars} ${ic_config}
sbatch -A ${slurm_acc} -p ${slurm_partition} scripts/templates/wideresnet_cifar100_ee/er.sh ${num_tasks} 1 ${num_exemplars} ${ic_config}
sbatch -A ${slurm_acc} -p ${slurm_partition} scripts/templates/wideresnet_cifar100_ee/er.sh ${num_tasks} 2 ${num_exemplars} ${ic_config}

# <--- FT --->
num_exemplars=0
sbatch -A ${slurm_acc} -p ${slurm_partition} scripts/templates/wideresnet_cifar100_base/ft.sh ${num_tasks} 0 ${num_exemplars}
sbatch -A ${slurm_acc} -p ${slurm_partition} scripts/templates/wideresnet_cifar100_base/ft.sh ${num_tasks} 1 ${num_exemplars}
sbatch -A ${slurm_acc} -p ${slurm_partition} scripts/templates/wideresnet_cifar100_base/ft.sh ${num_tasks} 2 ${num_exemplars}

sbatch -A ${slurm_acc} -p ${slurm_partition} scripts/templates/wideresnet_cifar100_ee/ft.sh ${num_tasks} 0 ${num_exemplars} ${ic_config}
sbatch -A ${slurm_acc} -p ${slurm_partition} scripts/templates/wideresnet_cifar100_ee/ft.sh ${num_tasks} 1 ${num_exemplars} ${ic_config}
sbatch -A ${slurm_acc} -p ${slurm_partition} scripts/templates/wideresnet_cifar100_ee/ft.sh ${num_tasks} 2 ${num_exemplars} ${ic_config}

# <--- FT+Ex --->
num_exemplars=2000
sbatch -A ${slurm_acc} -p ${slurm_partition} scripts/templates/wideresnet_cifar100_base/ft.sh ${num_tasks} 0 ${num_exemplars}
sbatch -A ${slurm_acc} -p ${slurm_partition} scripts/templates/wideresnet_cifar100_base/ft.sh ${num_tasks} 1 ${num_exemplars}
sbatch -A ${slurm_acc} -p ${slurm_partition} scripts/templates/wideresnet_cifar100_base/ft.sh ${num_tasks} 2 ${num_exemplars}

sbatch -A ${slurm_acc} -p ${slurm_partition} scripts/templates/wideresnet_cifar100_ee/ft.sh ${num_tasks} 0 ${num_exemplars} ${ic_config}
sbatch -A ${slurm_acc} -p ${slurm_partition} scripts/templates/wideresnet_cifar100_ee/ft.sh ${num_tasks} 1 ${num_exemplars} ${ic_config}
sbatch -A ${slurm_acc} -p ${slurm_partition} scripts/templates/wideresnet_cifar100_ee/ft.sh ${num_tasks} 2 ${num_exemplars} ${ic_config}

# <--- GDumb --->
num_exemplars=2000
sbatch -A ${slurm_acc} -p ${slurm_partition} scripts/templates/wideresnet_cifar100_base/gdumb.sh ${num_tasks} 0 ${num_exemplars}
sbatch -A ${slurm_acc} -p ${slurm_partition} scripts/templates/wideresnet_cifar100_base/gdumb.sh ${num_tasks} 1 ${num_exemplars}
sbatch -A ${slurm_acc} -p ${slurm_partition} scripts/templates/wideresnet_cifar100_base/gdumb.sh ${num_tasks} 2 ${num_exemplars}

sbatch -A ${slurm_acc} -p ${slurm_partition} scripts/templates/wideresnet_cifar100_ee/gdumb.sh ${num_tasks} 0 ${num_exemplars} ${ic_config}
sbatch -A ${slurm_acc} -p ${slurm_partition} scripts/templates/wideresnet_cifar100_ee/gdumb.sh ${num_tasks} 1 ${num_exemplars} ${ic_config}
sbatch -A ${slurm_acc} -p ${slurm_partition} scripts/templates/wideresnet_cifar100_ee/gdumb.sh ${num_tasks} 2 ${num_exemplars} ${ic_config}

# <--- LODE --->
num_exemplars=2000
const=1.0
ro=0.1
sbatch -A ${slurm_acc} -p ${slurm_partition} scripts/templates/wideresnet_cifar100_base/lode.sh ${num_tasks} 0 ${num_exemplars} ${const} ${ro}
sbatch -A ${slurm_acc} -p ${slurm_partition} scripts/templates/wideresnet_cifar100_base/lode.sh ${num_tasks} 1 ${num_exemplars} ${const} ${ro}
sbatch -A ${slurm_acc} -p ${slurm_partition} scripts/templates/wideresnet_cifar100_base/lode.sh ${num_tasks} 2 ${num_exemplars} ${const} ${ro}

const=3.0
ro=0.5
sbatch -A ${slurm_acc} -p ${slurm_partition} scripts/templates/wideresnet_cifar100_ee/lode.sh ${num_tasks} 0 ${num_exemplars} ${const} ${ro} ${ic_config}
sbatch -A ${slurm_acc} -p ${slurm_partition} scripts/templates/wideresnet_cifar100_ee/lode.sh ${num_tasks} 1 ${num_exemplars} ${const} ${ro} ${ic_config}
sbatch -A ${slurm_acc} -p ${slurm_partition} scripts/templates/wideresnet_cifar100_ee/lode.sh ${num_tasks} 2 ${num_exemplars} ${const} ${ro} ${ic_config}

# <--- LwF --->
lamb=1.0
sbatch -A ${slurm_acc} -p ${slurm_partition} scripts/templates/wideresnet_cifar100_base/lwf.sh ${num_tasks} 0 ${lamb}
sbatch -A ${slurm_acc} -p ${slurm_partition} scripts/templates/wideresnet_cifar100_base/lwf.sh ${num_tasks} 1 ${lamb}
sbatch -A ${slurm_acc} -p ${slurm_partition} scripts/templates/wideresnet_cifar100_base/lwf.sh ${num_tasks} 2 ${lamb}

lamb=0.5
sbatch -A ${slurm_acc} -p ${slurm_partition} scripts/templates/wideresnet_cifar100_ee/lwf.sh ${num_tasks} 0 ${lamb} ${ic_config}
sbatch -A ${slurm_acc} -p ${slurm_partition} scripts/templates/wideresnet_cifar100_ee/lwf.sh ${num_tasks} 1 ${lamb} ${ic_config}
sbatch -A ${slurm_acc} -p ${slurm_partition} scripts/templates/wideresnet_cifar100_ee/lwf.sh ${num_tasks} 2 ${lamb} ${ic_config}

# <--- Joint --->
# sbatch -A ${slurm_acc} -p ${slurm_partition} scripts/templates/wideresnet_cifar100_base/joint.sh ${num_tasks} 0
# sbatch -A ${slurm_acc} -p ${slurm_partition} scripts/templates/wideresnet_cifar100_base/joint.sh ${num_tasks} 1
# sbatch -A ${slurm_acc} -p ${slurm_partition} scripts/templates/wideresnet_cifar100_base/joint.sh ${num_tasks} 2
#
# sbatch -A ${slurm_acc} -p ${slurm_partition} scripts/templates/wideresnet_cifar100_ee/joint.sh ${num_tasks} 0 ${ic_config}
# sbatch -A ${slurm_acc} -p ${slurm_partition} scripts/templates/wideresnet_cifar100_ee/joint.sh ${num_tasks} 1 ${ic_config}
# sbatch -A ${slurm_acc} -p ${slurm_partition} scripts/templates/wideresnet_cifar100_ee/joint.sh ${num_tasks} 2 ${ic_config}

# <--- SSIL --->
num_exemplars=2000
lamb=0.25
sbatch -A ${slurm_acc} -p ${slurm_partition} scripts/templates/wideresnet_cifar100_base/ssil.sh ${num_tasks} 0 ${num_exemplars} ${lamb}
sbatch -A ${slurm_acc} -p ${slurm_partition} scripts/templates/wideresnet_cifar100_base/ssil.sh ${num_tasks} 1 ${num_exemplars} ${lamb}
sbatch -A ${slurm_acc} -p ${slurm_partition} scripts/templates/wideresnet_cifar100_base/ssil.sh ${num_tasks} 2 ${num_exemplars} ${lamb}

lamb=0.25
sbatch -A ${slurm_acc} -p ${slurm_partition} scripts/templates/wideresnet_cifar100_ee/ssil.sh ${num_tasks} 0 ${num_exemplars} ${lamb} ${ic_config}
sbatch -A ${slurm_acc} -p ${slurm_partition} scripts/templates/wideresnet_cifar100_ee/ssil.sh ${num_tasks} 1 ${num_exemplars} ${lamb} ${ic_config}
sbatch -A ${slurm_acc} -p ${slurm_partition} scripts/templates/wideresnet_cifar100_ee/ssil.sh ${num_tasks} 2 ${num_exemplars} ${lamb} ${ic_config}


#
#
#
#
# # <--- --- --- CIFAR100x20 --- --- --- --->
# num_tasks=20
#
# # <--- ANCL --->
# num_exemplars=0
# lamb=1.0
# lamb_a=1.0
# sbatch -A ${slurm_acc} -p ${slurm_partition} scripts/templates/wideresnet_cifar100_base/ancl.sh ${num_tasks} 0 ${num_exemplars} ${lamb} ${lamb_a}
# sbatch -A ${slurm_acc} -p ${slurm_partition} scripts/templates/wideresnet_cifar100_base/ancl.sh ${num_tasks} 1 ${num_exemplars} ${lamb} ${lamb_a}
# sbatch -A ${slurm_acc} -p ${slurm_partition} scripts/templates/wideresnet_cifar100_base/ancl.sh ${num_tasks} 2 ${num_exemplars} ${lamb} ${lamb_a}
#
# lamb=1.0
# lamb_a=1.0
# sbatch -A ${slurm_acc} -p ${slurm_partition} scripts/templates/wideresnet_cifar100_ee/ancl.sh ${num_tasks} 0 ${num_exemplars} ${lamb} ${lamb_a} ${ic_config}
# sbatch -A ${slurm_acc} -p ${slurm_partition} scripts/templates/wideresnet_cifar100_ee/ancl.sh ${num_tasks} 1 ${num_exemplars} ${lamb} ${lamb_a} ${ic_config}
# sbatch -A ${slurm_acc} -p ${slurm_partition} scripts/templates/wideresnet_cifar100_ee/ancl.sh ${num_tasks} 2 ${num_exemplars} ${lamb} ${lamb_a} ${ic_config}
#
# # <--- BiC --->
# num_exemplars=2000
# lamb=3
# sbatch -A ${slurm_acc} -p ${slurm_partition} scripts/templates/wideresnet_cifar100_base/bic.sh ${num_tasks} 0 ${num_exemplars} ${lamb}
# sbatch -A ${slurm_acc} -p ${slurm_partition} scripts/templates/wideresnet_cifar100_base/bic.sh ${num_tasks} 1 ${num_exemplars} ${lamb}
# sbatch -A ${slurm_acc} -p ${slurm_partition} scripts/templates/wideresnet_cifar100_base/bic.sh ${num_tasks} 2 ${num_exemplars} ${lamb}
#
# lamb=3
# sbatch -A ${slurm_acc} -p ${slurm_partition} scripts/templates/wideresnet_cifar100_ee/bic.sh ${num_tasks} 0 ${num_exemplars} ${lamb} ${ic_config}
# sbatch -A ${slurm_acc} -p ${slurm_partition} scripts/templates/wideresnet_cifar100_ee/bic.sh ${num_tasks} 1 ${num_exemplars} ${lamb} ${ic_config}
# sbatch -A ${slurm_acc} -p ${slurm_partition} scripts/templates/wideresnet_cifar100_ee/bic.sh ${num_tasks} 2 ${num_exemplars} ${lamb} ${ic_config}
#
# # <--- DER++ --->
# num_exemplars=2000
# alpha=0.5
# beta=0.5
# sbatch -A ${slurm_acc} -p ${slurm_partition} scripts/templates/wideresnet_cifar100_base/der++.sh ${num_tasks} 0 ${num_exemplars} ${alpha} ${beta}
# sbatch -A ${slurm_acc} -p ${slurm_partition} scripts/templates/wideresnet_cifar100_base/der++.sh ${num_tasks} 1 ${num_exemplars} ${alpha} ${beta}
# sbatch -A ${slurm_acc} -p ${slurm_partition} scripts/templates/wideresnet_cifar100_base/der++.sh ${num_tasks} 2 ${num_exemplars} ${alpha} ${beta}
#
# alpha=0.25
# beta=0.5
# sbatch -A ${slurm_acc} -p ${slurm_partition} scripts/templates/wideresnet_cifar100_ee/der++.sh ${num_tasks} 0 ${num_exemplars} ${alpha} ${beta} ${ic_config}
# sbatch -A ${slurm_acc} -p ${slurm_partition} scripts/templates/wideresnet_cifar100_ee/der++.sh ${num_tasks} 1 ${num_exemplars} ${alpha} ${beta} ${ic_config}
# sbatch -A ${slurm_acc} -p ${slurm_partition} scripts/templates/wideresnet_cifar100_ee/der++.sh ${num_tasks} 2 ${num_exemplars} ${alpha} ${beta} ${ic_config}
#
# # <--- EWC --->
# lamb=10000
# alpha=0.5
# sbatch -A ${slurm_acc} -p ${slurm_partition} scripts/templates/wideresnet_cifar100_base/ewc.sh ${num_tasks} 0 ${lamb} ${alpha}
# sbatch -A ${slurm_acc} -p ${slurm_partition} scripts/templates/wideresnet_cifar100_base/ewc.sh ${num_tasks} 1 ${lamb} ${alpha}
# sbatch -A ${slurm_acc} -p ${slurm_partition} scripts/templates/wideresnet_cifar100_base/ewc.sh ${num_tasks} 2 ${lamb} ${alpha}
#
# lamb=10000
# alpha=0.5
# sbatch -A ${slurm_acc} -p ${slurm_partition} scripts/templates/wideresnet_cifar100_ee/ewc.sh ${num_tasks} 0 ${lamb} ${alpha} ${ic_config}
# sbatch -A ${slurm_acc} -p ${slurm_partition} scripts/templates/wideresnet_cifar100_ee/ewc.sh ${num_tasks} 1 ${lamb} ${alpha} ${ic_config}
# sbatch -A ${slurm_acc} -p ${slurm_partition} scripts/templates/wideresnet_cifar100_ee/ewc.sh ${num_tasks} 2 ${lamb} ${alpha} ${ic_config}
#
# # <--- ER --->
# num_exemplars=2000
# sbatch -A ${slurm_acc} -p ${slurm_partition} scripts/templates/wideresnet_cifar100_base/er.sh ${num_tasks} 0 ${num_exemplars}
# sbatch -A ${slurm_acc} -p ${slurm_partition} scripts/templates/wideresnet_cifar100_base/er.sh ${num_tasks} 1 ${num_exemplars}
# sbatch -A ${slurm_acc} -p ${slurm_partition} scripts/templates/wideresnet_cifar100_base/er.sh ${num_tasks} 2 ${num_exemplars}
#
# sbatch -A ${slurm_acc} -p ${slurm_partition} scripts/templates/wideresnet_cifar100_ee/er.sh ${num_tasks} 0 ${num_exemplars} ${ic_config}
# sbatch -A ${slurm_acc} -p ${slurm_partition} scripts/templates/wideresnet_cifar100_ee/er.sh ${num_tasks} 1 ${num_exemplars} ${ic_config}
# sbatch -A ${slurm_acc} -p ${slurm_partition} scripts/templates/wideresnet_cifar100_ee/er.sh ${num_tasks} 2 ${num_exemplars} ${ic_config}
#
# # <--- FT --->
# num_exemplars=0
# sbatch -A ${slurm_acc} -p ${slurm_partition} scripts/templates/wideresnet_cifar100_base/ft.sh ${num_tasks} 0 ${num_exemplars}
# sbatch -A ${slurm_acc} -p ${slurm_partition} scripts/templates/wideresnet_cifar100_base/ft.sh ${num_tasks} 1 ${num_exemplars}
# sbatch -A ${slurm_acc} -p ${slurm_partition} scripts/templates/wideresnet_cifar100_base/ft.sh ${num_tasks} 2 ${num_exemplars}
#
# sbatch -A ${slurm_acc} -p ${slurm_partition} scripts/templates/wideresnet_cifar100_ee/ft.sh ${num_tasks} 0 ${num_exemplars} ${ic_config}
# sbatch -A ${slurm_acc} -p ${slurm_partition} scripts/templates/wideresnet_cifar100_ee/ft.sh ${num_tasks} 1 ${num_exemplars} ${ic_config}
# sbatch -A ${slurm_acc} -p ${slurm_partition} scripts/templates/wideresnet_cifar100_ee/ft.sh ${num_tasks} 2 ${num_exemplars} ${ic_config}
#
# # <--- FT+Ex --->
# num_exemplars=2000
# sbatch -A ${slurm_acc} -p ${slurm_partition} scripts/templates/wideresnet_cifar100_base/ft.sh ${num_tasks} 0 ${num_exemplars}
# sbatch -A ${slurm_acc} -p ${slurm_partition} scripts/templates/wideresnet_cifar100_base/ft.sh ${num_tasks} 1 ${num_exemplars}
# sbatch -A ${slurm_acc} -p ${slurm_partition} scripts/templates/wideresnet_cifar100_base/ft.sh ${num_tasks} 2 ${num_exemplars}
#
# sbatch -A ${slurm_acc} -p ${slurm_partition} scripts/templates/wideresnet_cifar100_ee/ft.sh ${num_tasks} 0 ${num_exemplars} ${ic_config}
# sbatch -A ${slurm_acc} -p ${slurm_partition} scripts/templates/wideresnet_cifar100_ee/ft.sh ${num_tasks} 1 ${num_exemplars} ${ic_config}
# sbatch -A ${slurm_acc} -p ${slurm_partition} scripts/templates/wideresnet_cifar100_ee/ft.sh ${num_tasks} 2 ${num_exemplars} ${ic_config}
#
# # <--- GDumb --->
# num_exemplars=2000
# sbatch -A ${slurm_acc} -p ${slurm_partition} scripts/templates/wideresnet_cifar100_base/gdumb.sh ${num_tasks} 0 ${num_exemplars}
# sbatch -A ${slurm_acc} -p ${slurm_partition} scripts/templates/wideresnet_cifar100_base/gdumb.sh ${num_tasks} 1 ${num_exemplars}
# sbatch -A ${slurm_acc} -p ${slurm_partition} scripts/templates/wideresnet_cifar100_base/gdumb.sh ${num_tasks} 2 ${num_exemplars}
#
# sbatch -A ${slurm_acc} -p ${slurm_partition} scripts/templates/wideresnet_cifar100_ee/gdumb.sh ${num_tasks} 0 ${num_exemplars} ${ic_config}
# sbatch -A ${slurm_acc} -p ${slurm_partition} scripts/templates/wideresnet_cifar100_ee/gdumb.sh ${num_tasks} 1 ${num_exemplars} ${ic_config}
# sbatch -A ${slurm_acc} -p ${slurm_partition} scripts/templates/wideresnet_cifar100_ee/gdumb.sh ${num_tasks} 2 ${num_exemplars} ${ic_config}
#
# # <--- LODE --->
# num_exemplars=2000
# const=4.0
# ro=0.1
# sbatch -A ${slurm_acc} -p ${slurm_partition} scripts/templates/wideresnet_cifar100_base/lode.sh ${num_tasks} 0 ${num_exemplars} ${const} ${ro}
# sbatch -A ${slurm_acc} -p ${slurm_partition} scripts/templates/wideresnet_cifar100_base/lode.sh ${num_tasks} 1 ${num_exemplars} ${const} ${ro}
# sbatch -A ${slurm_acc} -p ${slurm_partition} scripts/templates/wideresnet_cifar100_base/lode.sh ${num_tasks} 2 ${num_exemplars} ${const} ${ro}
#
# const=4.0
# ro=0.5
# sbatch -A ${slurm_acc} -p ${slurm_partition} scripts/templates/wideresnet_cifar100_ee/lode.sh ${num_tasks} 0 ${num_exemplars} ${const} ${ro} ${ic_config}
# sbatch -A ${slurm_acc} -p ${slurm_partition} scripts/templates/wideresnet_cifar100_ee/lode.sh ${num_tasks} 1 ${num_exemplars} ${const} ${ro} ${ic_config}
# sbatch -A ${slurm_acc} -p ${slurm_partition} scripts/templates/wideresnet_cifar100_ee/lode.sh ${num_tasks} 2 ${num_exemplars} ${const} ${ro} ${ic_config}
#
# # <--- LwF --->
# lamb=0.5
# sbatch -A ${slurm_acc} -p ${slurm_partition} scripts/templates/wideresnet_cifar100_base/lwf.sh ${num_tasks} 0 ${lamb}
# sbatch -A ${slurm_acc} -p ${slurm_partition} scripts/templates/wideresnet_cifar100_base/lwf.sh ${num_tasks} 1 ${lamb}
# sbatch -A ${slurm_acc} -p ${slurm_partition} scripts/templates/wideresnet_cifar100_base/lwf.sh ${num_tasks} 2 ${lamb}
#
# lamb=0.5
# sbatch -A ${slurm_acc} -p ${slurm_partition} scripts/templates/wideresnet_cifar100_ee/lwf.sh ${num_tasks} 0 ${lamb} ${ic_config}
# sbatch -A ${slurm_acc} -p ${slurm_partition} scripts/templates/wideresnet_cifar100_ee/lwf.sh ${num_tasks} 1 ${lamb} ${ic_config}
# sbatch -A ${slurm_acc} -p ${slurm_partition} scripts/templates/wideresnet_cifar100_ee/lwf.sh ${num_tasks} 2 ${lamb} ${ic_config}
#
# # <--- Joint --->
# sbatch -A ${slurm_acc} -p ${slurm_partition} scripts/templates/wideresnet_cifar100_base/joint.sh ${num_tasks} 0
# sbatch -A ${slurm_acc} -p ${slurm_partition} scripts/templates/wideresnet_cifar100_base/joint.sh ${num_tasks} 1
# sbatch -A ${slurm_acc} -p ${slurm_partition} scripts/templates/wideresnet_cifar100_base/joint.sh ${num_tasks} 2
#
# sbatch -A ${slurm_acc} -p ${slurm_partition} scripts/templates/wideresnet_cifar100_ee/joint.sh ${num_tasks} 0 ${ic_config}
# sbatch -A ${slurm_acc} -p ${slurm_partition} scripts/templates/wideresnet_cifar100_ee/joint.sh ${num_tasks} 1 ${ic_config}
# sbatch -A ${slurm_acc} -p ${slurm_partition} scripts/templates/wideresnet_cifar100_ee/joint.sh ${num_tasks} 2 ${ic_config}
#
# # <--- SSIL --->
# num_exemplars=2000
# lamb=0.25
# sbatch -A ${slurm_acc} -p ${slurm_partition} scripts/templates/wideresnet_cifar100_base/ssil.sh ${num_tasks} 0 ${num_exemplars} ${lamb}
# sbatch -A ${slurm_acc} -p ${slurm_partition} scripts/templates/wideresnet_cifar100_base/ssil.sh ${num_tasks} 1 ${num_exemplars} ${lamb}
# sbatch -A ${slurm_acc} -p ${slurm_partition} scripts/templates/wideresnet_cifar100_base/ssil.sh ${num_tasks} 2 ${num_exemplars} ${lamb}
#
# lamb=0.1
# sbatch -A ${slurm_acc} -p ${slurm_partition} scripts/templates/wideresnet_cifar100_ee/ssil.sh ${num_tasks} 0 ${num_exemplars} ${lamb} ${ic_config}
# sbatch -A ${slurm_acc} -p ${slurm_partition} scripts/templates/wideresnet_cifar100_ee/ssil.sh ${num_tasks} 1 ${num_exemplars} ${lamb} ${ic_config}
# sbatch -A ${slurm_acc} -p ${slurm_partition} scripts/templates/wideresnet_cifar100_ee/ssil.sh ${num_tasks} 2 ${num_exemplars} ${lamb} ${ic_config}
#
#
#
#
#
#
# # <--- --- --- CIFAR100x50 --- --- --- --->
# num_tasks=50
#
# # <--- ANCL --->
# num_exemplars=0
# lamb=1.0
# lamb_a=1.0
# sbatch -A ${slurm_acc} -p ${slurm_partition} scripts/templates/wideresnet_cifar100_base_growing_mem/ancl.sh ${num_tasks} 1 ${num_exemplars} ${lamb} ${lamb_a}
# sbatch -A ${slurm_acc} -p ${slurm_partition} scripts/templates/wideresnet_cifar100_base_growing_mem/ancl.sh ${num_tasks} 9 ${num_exemplars} ${lamb} ${lamb_a}
# sbatch -A ${slurm_acc} -p ${slurm_partition} scripts/templates/wideresnet_cifar100_base_growing_mem/ancl.sh ${num_tasks} 18 ${num_exemplars} ${lamb} ${lamb_a}
#
# lamb=1.0
# lamb_a=1.0
# sbatch -A ${slurm_acc} -p ${slurm_partition} scripts/templates/wideresnet_cifar100_ee_growing_mem/ancl.sh ${num_tasks} 1 ${num_exemplars} ${lamb} ${lamb_a} ${ic_config}
# sbatch -A ${slurm_acc} -p ${slurm_partition} scripts/templates/wideresnet_cifar100_ee_growing_mem/ancl.sh ${num_tasks} 9 ${num_exemplars} ${lamb} ${lamb_a} ${ic_config}
# sbatch -A ${slurm_acc} -p ${slurm_partition} scripts/templates/wideresnet_cifar100_ee_growing_mem/ancl.sh ${num_tasks} 18 ${num_exemplars} ${lamb} ${lamb_a} ${ic_config}
#
# # <--- BiC --->
# num_exemplars=20
# lamb=3
# sbatch -A ${slurm_acc} -p ${slurm_partition} scripts/templates/wideresnet_cifar100_base_growing_mem/bic.sh ${num_tasks} 16 ${num_exemplars} ${lamb}
# sbatch -A ${slurm_acc} -p ${slurm_partition} scripts/templates/wideresnet_cifar100_base_growing_mem/bic.sh ${num_tasks} 17 ${num_exemplars} ${lamb}
# sbatch -A ${slurm_acc} -p ${slurm_partition} scripts/templates/wideresnet_cifar100_base_growing_mem/bic.sh ${num_tasks} 18 ${num_exemplars} ${lamb}
#
# lamb=3
# sbatch -A ${slurm_acc} -p ${slurm_partition} scripts/templates/wideresnet_cifar100_ee_growing_mem/bic.sh ${num_tasks} 16 ${num_exemplars} ${lamb} ${ic_config}
# sbatch -A ${slurm_acc} -p ${slurm_partition} scripts/templates/wideresnet_cifar100_ee_growing_mem/bic.sh ${num_tasks} 17 ${num_exemplars} ${lamb} ${ic_config}
# sbatch -A ${slurm_acc} -p ${slurm_partition} scripts/templates/wideresnet_cifar100_ee_growing_mem/bic.sh ${num_tasks} 18 ${num_exemplars} ${lamb} ${ic_config}
#
# # <--- DER++ --->
# num_exemplars=20
# alpha=0.5
# beta=0.5
# sbatch -A ${slurm_acc} -p ${slurm_partition} scripts/templates/wideresnet_cifar100_base_growing_mem/der++.sh ${num_tasks} 0 ${num_exemplars} ${alpha} ${beta}
# sbatch -A ${slurm_acc} -p ${slurm_partition} scripts/templates/wideresnet_cifar100_base_growing_mem/der++.sh ${num_tasks} 1 ${num_exemplars} ${alpha} ${beta}
# sbatch -A ${slurm_acc} -p ${slurm_partition} scripts/templates/wideresnet_cifar100_base_growing_mem/der++.sh ${num_tasks} 2 ${num_exemplars} ${alpha} ${beta}
#
# alpha=0.25
# beta=0.5
# sbatch -A ${slurm_acc} -p ${slurm_partition} scripts/templates/wideresnet_cifar100_ee_growing_mem/der++.sh ${num_tasks} 0 ${num_exemplars} ${alpha} ${beta} ${ic_config}
# sbatch -A ${slurm_acc} -p ${slurm_partition} scripts/templates/wideresnet_cifar100_ee_growing_mem/der++.sh ${num_tasks} 1 ${num_exemplars} ${alpha} ${beta} ${ic_config}
# sbatch -A ${slurm_acc} -p ${slurm_partition} scripts/templates/wideresnet_cifar100_ee_growing_mem/der++.sh ${num_tasks} 2 ${num_exemplars} ${alpha} ${beta} ${ic_config}
#
# # <--- EWC --->
# lamb=10000
# alpha=0.5
# sbatch -A ${slurm_acc} -p ${slurm_partition} scripts/templates/wideresnet_cifar100_base_growing_mem/ewc.sh ${num_tasks} 0 ${lamb} ${alpha}
# sbatch -A ${slurm_acc} -p ${slurm_partition} scripts/templates/wideresnet_cifar100_base_growing_mem/ewc.sh ${num_tasks} 1 ${lamb} ${alpha}
# sbatch -A ${slurm_acc} -p ${slurm_partition} scripts/templates/wideresnet_cifar100_base_growing_mem/ewc.sh ${num_tasks} 2 ${lamb} ${alpha}
#
# lamb=10000
# alpha=0.5
# sbatch -A ${slurm_acc} -p ${slurm_partition} scripts/templates/wideresnet_cifar100_ee_growing_mem/ewc.sh ${num_tasks} 0 ${lamb} ${alpha} ${ic_config}
# sbatch -A ${slurm_acc} -p ${slurm_partition} scripts/templates/wideresnet_cifar100_ee_growing_mem/ewc.sh ${num_tasks} 1 ${lamb} ${alpha} ${ic_config}
# sbatch -A ${slurm_acc} -p ${slurm_partition} scripts/templates/wideresnet_cifar100_ee_growing_mem/ewc.sh ${num_tasks} 2 ${lamb} ${alpha} ${ic_config}
#
# # <--- ER --->
# num_exemplars=20
# sbatch -A ${slurm_acc} -p ${slurm_partition} scripts/templates/wideresnet_cifar100_base_growing_mem/er.sh ${num_tasks} 0 ${num_exemplars}
# sbatch -A ${slurm_acc} -p ${slurm_partition} scripts/templates/wideresnet_cifar100_base_growing_mem/er.sh ${num_tasks} 1 ${num_exemplars}
# sbatch -A ${slurm_acc} -p ${slurm_partition} scripts/templates/wideresnet_cifar100_base_growing_mem/er.sh ${num_tasks} 2 ${num_exemplars}
#
# sbatch -A ${slurm_acc} -p ${slurm_partition} scripts/templates/wideresnet_cifar100_ee_growing_mem/er.sh ${num_tasks} 0 ${num_exemplars} ${ic_config}
# sbatch -A ${slurm_acc} -p ${slurm_partition} scripts/templates/wideresnet_cifar100_ee_growing_mem/er.sh ${num_tasks} 1 ${num_exemplars} ${ic_config}
# sbatch -A ${slurm_acc} -p ${slurm_partition} scripts/templates/wideresnet_cifar100_ee_growing_mem/er.sh ${num_tasks} 2 ${num_exemplars} ${ic_config}
#
# # <--- FT --->
# num_exemplars=0
# sbatch -A ${slurm_acc} -p ${slurm_partition} scripts/templates/wideresnet_cifar100_base_growing_mem/ft.sh ${num_tasks} 0 ${num_exemplars}
# sbatch -A ${slurm_acc} -p ${slurm_partition} scripts/templates/wideresnet_cifar100_base_growing_mem/ft.sh ${num_tasks} 1 ${num_exemplars}
# sbatch -A ${slurm_acc} -p ${slurm_partition} scripts/templates/wideresnet_cifar100_base_growing_mem/ft.sh ${num_tasks} 2 ${num_exemplars}
#
# sbatch -A ${slurm_acc} -p ${slurm_partition} scripts/templates/wideresnet_cifar100_ee_growing_mem/ft.sh ${num_tasks} 0 ${num_exemplars} ${ic_config}
# sbatch -A ${slurm_acc} -p ${slurm_partition} scripts/templates/wideresnet_cifar100_ee_growing_mem/ft.sh ${num_tasks} 1 ${num_exemplars} ${ic_config}
# sbatch -A ${slurm_acc} -p ${slurm_partition} scripts/templates/wideresnet_cifar100_ee_growing_mem/ft.sh ${num_tasks} 2 ${num_exemplars} ${ic_config}
#
# # <--- FT+Ex --->
# num_exemplars=20
# sbatch -A ${slurm_acc} -p ${slurm_partition} scripts/templates/wideresnet_cifar100_base_growing_mem/ft.sh ${num_tasks} 0 ${num_exemplars}
# sbatch -A ${slurm_acc} -p ${slurm_partition} scripts/templates/wideresnet_cifar100_base_growing_mem/ft.sh ${num_tasks} 1 ${num_exemplars}
# sbatch -A ${slurm_acc} -p ${slurm_partition} scripts/templates/wideresnet_cifar100_base_growing_mem/ft.sh ${num_tasks} 2 ${num_exemplars}
#
# sbatch -A ${slurm_acc} -p ${slurm_partition} scripts/templates/wideresnet_cifar100_ee_growing_mem/ft.sh ${num_tasks} 0 ${num_exemplars} ${ic_config}
# sbatch -A ${slurm_acc} -p ${slurm_partition} scripts/templates/wideresnet_cifar100_ee_growing_mem/ft.sh ${num_tasks} 1 ${num_exemplars} ${ic_config}
# sbatch -A ${slurm_acc} -p ${slurm_partition} scripts/templates/wideresnet_cifar100_ee_growing_mem/ft.sh ${num_tasks} 2 ${num_exemplars} ${ic_config}
#
# # <--- GDumb --->
# num_exemplars=20
# sbatch -A ${slurm_acc} -p ${slurm_partition} scripts/templates/wideresnet_cifar100_base_growing_mem/gdumb.sh ${num_tasks} 0 ${num_exemplars}
# sbatch -A ${slurm_acc} -p ${slurm_partition} scripts/templates/wideresnet_cifar100_base_growing_mem/gdumb.sh ${num_tasks} 1 ${num_exemplars}
# sbatch -A ${slurm_acc} -p ${slurm_partition} scripts/templates/wideresnet_cifar100_base_growing_mem/gdumb.sh ${num_tasks} 2 ${num_exemplars}
#
# sbatch -A ${slurm_acc} -p ${slurm_partition} scripts/templates/wideresnet_cifar100_ee_growing_mem/gdumb.sh ${num_tasks} 0 ${num_exemplars} ${ic_config}
# sbatch -A ${slurm_acc} -p ${slurm_partition} scripts/templates/wideresnet_cifar100_ee_growing_mem/gdumb.sh ${num_tasks} 1 ${num_exemplars} ${ic_config}
# sbatch -A ${slurm_acc} -p ${slurm_partition} scripts/templates/wideresnet_cifar100_ee_growing_mem/gdumb.sh ${num_tasks} 2 ${num_exemplars} ${ic_config}
#
# # <--- LODE --->
# num_exemplars=20
# const=4.0
# ro=0.1
# sbatch -A ${slurm_acc} -p ${slurm_partition} scripts/templates/wideresnet_cifar100_base_growing_mem/lode.sh ${num_tasks} 0 ${num_exemplars} ${const} ${ro}
# sbatch -A ${slurm_acc} -p ${slurm_partition} scripts/templates/wideresnet_cifar100_base_growing_mem/lode.sh ${num_tasks} 1 ${num_exemplars} ${const} ${ro}
# sbatch -A ${slurm_acc} -p ${slurm_partition} scripts/templates/wideresnet_cifar100_base_growing_mem/lode.sh ${num_tasks} 2 ${num_exemplars} ${const} ${ro}
#
# const=4.0
# ro=0.5
# sbatch -A ${slurm_acc} -p ${slurm_partition} scripts/templates/wideresnet_cifar100_ee_growing_mem/lode.sh ${num_tasks} 0 ${num_exemplars} ${const} ${ro} ${ic_config}
# sbatch -A ${slurm_acc} -p ${slurm_partition} scripts/templates/wideresnet_cifar100_ee_growing_mem/lode.sh ${num_tasks} 1 ${num_exemplars} ${const} ${ro} ${ic_config}
# sbatch -A ${slurm_acc} -p ${slurm_partition} scripts/templates/wideresnet_cifar100_ee_growing_mem/lode.sh ${num_tasks} 2 ${num_exemplars} ${const} ${ro} ${ic_config}
#
# # <--- LwF --->
# lamb=0.5
# sbatch -A ${slurm_acc} -p ${slurm_partition} scripts/templates/wideresnet_cifar100_base_growing_mem/lwf.sh ${num_tasks} 0 ${lamb}
# sbatch -A ${slurm_acc} -p ${slurm_partition} scripts/templates/wideresnet_cifar100_base_growing_mem/lwf.sh ${num_tasks} 1 ${lamb}
# sbatch -A ${slurm_acc} -p ${slurm_partition} scripts/templates/wideresnet_cifar100_base_growing_mem/lwf.sh ${num_tasks} 2 ${lamb}
#
# lamb=0.5
# sbatch -A ${slurm_acc} -p ${slurm_partition} scripts/templates/wideresnet_cifar100_ee_growing_mem/lwf.sh ${num_tasks} 0 ${lamb} ${ic_config}
# sbatch -A ${slurm_acc} -p ${slurm_partition} scripts/templates/wideresnet_cifar100_ee_growing_mem/lwf.sh ${num_tasks} 1 ${lamb} ${ic_config}
# sbatch -A ${slurm_acc} -p ${slurm_partition} scripts/templates/wideresnet_cifar100_ee_growing_mem/lwf.sh ${num_tasks} 2 ${lamb} ${ic_config}
#
# # <--- Joint --->
# sbatch -A ${slurm_acc} -p ${slurm_partition} scripts/templates/wideresnet_cifar100_base_growing_mem/joint.sh ${num_tasks} 0
# sbatch -A ${slurm_acc} -p ${slurm_partition} scripts/templates/wideresnet_cifar100_base_growing_mem/joint.sh ${num_tasks} 1
# sbatch -A ${slurm_acc} -p ${slurm_partition} scripts/templates/wideresnet_cifar100_base_growing_mem/joint.sh ${num_tasks} 2
#
# sbatch -A ${slurm_acc} -p ${slurm_partition} scripts/templates/wideresnet_cifar100_ee_growing_mem/joint.sh ${num_tasks} 0 ${ic_config}
# sbatch -A ${slurm_acc} -p ${slurm_partition} scripts/templates/wideresnet_cifar100_ee_growing_mem/joint.sh ${num_tasks} 1 ${ic_config}
# sbatch -A ${slurm_acc} -p ${slurm_partition} scripts/templates/wideresnet_cifar100_ee_growing_mem/joint.sh ${num_tasks} 2 ${ic_config}
#
# # <--- SSIL --->
# num_exemplars=20
# lamb=0.25
# sbatch -A ${slurm_acc} -p ${slurm_partition} scripts/templates/wideresnet_cifar100_base_growing_mem/ssil.sh ${num_tasks} 0 ${num_exemplars} ${lamb}
# sbatch -A ${slurm_acc} -p ${slurm_partition} scripts/templates/wideresnet_cifar100_base_growing_mem/ssil.sh ${num_tasks} 1 ${num_exemplars} ${lamb}
# sbatch -A ${slurm_acc} -p ${slurm_partition} scripts/templates/wideresnet_cifar100_base_growing_mem/ssil.sh ${num_tasks} 2 ${num_exemplars} ${lamb}
#
# lamb=0.1
# sbatch -A ${slurm_acc} -p ${slurm_partition} scripts/templates/wideresnet_cifar100_ee_growing_mem/ssil.sh ${num_tasks} 0 ${num_exemplars} ${lamb} ${ic_config}
# sbatch -A ${slurm_acc} -p ${slurm_partition} scripts/templates/wideresnet_cifar100_ee_growing_mem/ssil.sh ${num_tasks} 1 ${num_exemplars} ${lamb} ${ic_config}
# sbatch -A ${slurm_acc} -p ${slurm_partition} scripts/templates/wideresnet_cifar100_ee_growing_mem/ssil.sh ${num_tasks} 2 ${num_exemplars} ${lamb} ${ic_config}
