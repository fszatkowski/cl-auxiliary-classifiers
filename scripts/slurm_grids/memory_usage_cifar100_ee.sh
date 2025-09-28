#!/bin/bash

set -e
ic_config=cifar100_resnet32_sdn
num_tasks=5

# <--- ANCL --->
num_exemplars=0
lamb=0.5
lamb_a=0.5
./scripts/templates/memory_usage_cifar100_ee/ancl.sh ${num_tasks} 0 ${num_exemplars} ${lamb} ${lamb_a} ${ic_config}

# <--- BiC --->
num_exemplars=2000
lamb=2
./scripts/templates/memory_usage_cifar100_ee/bic.sh ${num_tasks} 0 ${num_exemplars} ${lamb} ${ic_config}

# <--- DER++ --->
num_exemplars=2000
alpha=0.5
beta=0.5
./scripts/templates/memory_usage_cifar100_ee/der++.sh ${num_tasks} 1 ${num_exemplars} ${alpha} ${beta} ${ic_config}

# <--- EWC --->
lamb=10000
alpha=0.5
./scripts/templates/memory_usage_cifar100_ee/ewc.sh ${num_tasks} 0 ${lamb} ${alpha} ${ic_config}

# <--- ER --->
num_exemplars=2000
./scripts/templates/memory_usage_cifar100_ee/er.sh ${num_tasks} 0 ${num_exemplars} ${ic_config}

# <--- FT --->
num_exemplars=0
./scripts/templates/memory_usage_cifar100_ee/ft.sh ${num_tasks} 0 ${num_exemplars} ${ic_config}

# <--- FT+Ex --->
num_exemplars=2000
./scripts/templates/memory_usage_cifar100_ee/ft.sh ${num_tasks} 0 ${num_exemplars} ${ic_config}

# <--- GDumb --->
num_exemplars=2000
./scripts/templates/memory_usage_cifar100_ee/gdumb.sh ${num_tasks} 0 ${num_exemplars} ${ic_config}

# <--- LODE --->
num_exemplars=2000
const=1.0
ro=0.1
./scripts/templates/memory_usage_cifar100_ee/lode.sh ${num_tasks} 0 ${num_exemplars} ${const} ${ro} ${ic_config}

# <--- LwF --->
lamb=1.0
./scripts/templates/memory_usage_cifar100_ee/lwf.sh ${num_tasks} 0 ${lamb} ${ic_config}

# <--- SSIL --->
num_exemplars=2000
lamb=0.25
./scripts/templates/memory_usage_cifar100_ee/ssil.sh ${num_tasks} 0 ${num_exemplars} ${lamb} ${ic_config}
