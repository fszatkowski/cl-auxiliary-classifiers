#!/bin/bash

set -e
ic_config=cifar100_vgg19_medium
num_tasks=5

# <--- ANCL --->
num_exemplars=0
lamb=0
lamb_a=0
./scripts/templates/memory_usage_vgg19_base/ancl.sh ${num_tasks} 0 ${num_exemplars} ${lamb} ${lamb_a}

# <--- BiC --->
num_exemplars=2000
lamb=2
./scripts/templates/memory_usage_vgg19_base/bic.sh ${num_tasks} 0 ${num_exemplars} ${lamb}

# <--- DER++ --->
num_exemplars=2000
alpha=0.5
beta=0.5
./scripts/templates/memory_usage_vgg19_base/der++.sh ${num_tasks} 1 ${num_exemplars} ${alpha} ${beta}

# <--- EWC --->
lamb=10000
alpha=0.5
./scripts/templates/memory_usage_vgg19_base/ewc.sh ${num_tasks} 0 ${lamb} ${alpha}

# <--- ER --->
num_exemplars=2000
./scripts/templates/memory_usage_vgg19_base/er.sh ${num_tasks} 0 ${num_exemplars}

# <--- FT --->
num_exemplars=0
./scripts/templates/memory_usage_vgg19_base/ft.sh ${num_tasks} 0 ${num_exemplars}

# <--- FT+Ex --->
num_exemplars=2000
./scripts/templates/memory_usage_vgg19_base/ft.sh ${num_tasks} 0 ${num_exemplars}

# <--- GDumb --->
num_exemplars=2000
./scripts/templates/memory_usage_vgg19_base/gdumb.sh ${num_tasks} 0 ${num_exemplars}

# <--- LODE --->
num_exemplars=2000
const=1.0
ro=0.1
./scripts/templates/memory_usage_vgg19_base/lode.sh ${num_tasks} 0 ${num_exemplars} ${const} ${ro}

# <--- LwF --->
lamb=1.0
./scripts/templates/memory_usage_vgg19_base/lwf.sh ${num_tasks} 0 ${lamb}

# <--- SSIL --->
num_exemplars=2000
lamb=0.25
./scripts/templates/memory_usage_vgg19_base/ssil.sh ${num_tasks} 0 ${num_exemplars} ${lamb}
