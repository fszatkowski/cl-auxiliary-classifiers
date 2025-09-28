#!/bin/bash

set -e

# This slurm script allows to run all main experiments on equal split of CIFAR100

slurm_acc=...
slurm_partition=...

ic_config=cifar100_wideresnet16_sdn
# <--- --- --- CIFAR100x5 --- --- --- --->
num_tasks=5
# <--- DER++ --->
num_exemplars=2000
alpha=0.5
beta=0.5
sbatch -A ${slurm_acc} -p ${slurm_partition} scripts/templates/wideresnet_cifar100_ee/der++.sh ${num_tasks} 3 ${num_exemplars} ${alpha} ${beta} ${ic_config}
sbatch -A ${slurm_acc} -p ${slurm_partition} scripts/templates/wideresnet_cifar100_ee/der++.sh ${num_tasks} 4 ${num_exemplars} ${alpha} ${beta} ${ic_config}
sbatch -A ${slurm_acc} -p ${slurm_partition} scripts/templates/wideresnet_cifar100_ee/der++.sh ${num_tasks} 5 ${num_exemplars} ${alpha} ${beta} ${ic_config}
alpha=0.2
beta=1.0
sbatch -A ${slurm_acc} -p ${slurm_partition} scripts/templates/wideresnet_cifar100_ee/der++.sh ${num_tasks} 3 ${num_exemplars} ${alpha} ${beta} ${ic_config}
sbatch -A ${slurm_acc} -p ${slurm_partition} scripts/templates/wideresnet_cifar100_ee/der++.sh ${num_tasks} 4 ${num_exemplars} ${alpha} ${beta} ${ic_config}
sbatch -A ${slurm_acc} -p ${slurm_partition} scripts/templates/wideresnet_cifar100_ee/der++.sh ${num_tasks} 5 ${num_exemplars} ${alpha} ${beta} ${ic_config}
alpha=0.1
beta=1.0
sbatch -A ${slurm_acc} -p ${slurm_partition} scripts/templates/wideresnet_cifar100_ee/der++.sh ${num_tasks} 3 ${num_exemplars} ${alpha} ${beta} ${ic_config}
sbatch -A ${slurm_acc} -p ${slurm_partition} scripts/templates/wideresnet_cifar100_ee/der++.sh ${num_tasks} 4 ${num_exemplars} ${alpha} ${beta} ${ic_config}
sbatch -A ${slurm_acc} -p ${slurm_partition} scripts/templates/wideresnet_cifar100_ee/der++.sh ${num_tasks} 5 ${num_exemplars} ${alpha} ${beta} ${ic_config}
alpha=0.01
beta=1.0
sbatch -A ${slurm_acc} -p ${slurm_partition} scripts/templates/wideresnet_cifar100_ee/der++.sh ${num_tasks} 3 ${num_exemplars} ${alpha} ${beta} ${ic_config}
sbatch -A ${slurm_acc} -p ${slurm_partition} scripts/templates/wideresnet_cifar100_ee/der++.sh ${num_tasks} 4 ${num_exemplars} ${alpha} ${beta} ${ic_config}
sbatch -A ${slurm_acc} -p ${slurm_partition} scripts/templates/wideresnet_cifar100_ee/der++.sh ${num_tasks} 5 ${num_exemplars} ${alpha} ${beta} ${ic_config}
alpha=0.2
beta=0.5
sbatch -A ${slurm_acc} -p ${slurm_partition} scripts/templates/wideresnet_cifar100_ee/der++.sh ${num_tasks} 3 ${num_exemplars} ${alpha} ${beta} ${ic_config}
sbatch -A ${slurm_acc} -p ${slurm_partition} scripts/templates/wideresnet_cifar100_ee/der++.sh ${num_tasks} 4 ${num_exemplars} ${alpha} ${beta} ${ic_config}
sbatch -A ${slurm_acc} -p ${slurm_partition} scripts/templates/wideresnet_cifar100_ee/der++.sh ${num_tasks} 5 ${num_exemplars} ${alpha} ${beta} ${ic_config}
alpha=0.1
beta=0.5
sbatch -A ${slurm_acc} -p ${slurm_partition} scripts/templates/wideresnet_cifar100_ee/der++.sh ${num_tasks} 3 ${num_exemplars} ${alpha} ${beta} ${ic_config}
sbatch -A ${slurm_acc} -p ${slurm_partition} scripts/templates/wideresnet_cifar100_ee/der++.sh ${num_tasks} 4 ${num_exemplars} ${alpha} ${beta} ${ic_config}
sbatch -A ${slurm_acc} -p ${slurm_partition} scripts/templates/wideresnet_cifar100_ee/der++.sh ${num_tasks} 5 ${num_exemplars} ${alpha} ${beta} ${ic_config}
alpha=0.01
beta=0.5
sbatch -A ${slurm_acc} -p ${slurm_partition} scripts/templates/wideresnet_cifar100_ee/der++.sh ${num_tasks} 3 ${num_exemplars} ${alpha} ${beta} ${ic_config}
sbatch -A ${slurm_acc} -p ${slurm_partition} scripts/templates/wideresnet_cifar100_ee/der++.sh ${num_tasks} 4 ${num_exemplars} ${alpha} ${beta} ${ic_config}
sbatch -A ${slurm_acc} -p ${slurm_partition} scripts/templates/wideresnet_cifar100_ee/der++.sh ${num_tasks} 5 ${num_exemplars} ${alpha} ${beta} ${ic_config}

# # <--- --- --- CIFAR100x10 --- --- --- --->
num_tasks=10
# # <--- DER++ --->
num_exemplars=2000
alpha=0.5
beta=0.5
sbatch -A ${slurm_acc} -p ${slurm_partition} scripts/templates/wideresnet_cifar100_ee/der++.sh ${num_tasks} 3 ${num_exemplars} ${alpha} ${beta} ${ic_config}
sbatch -A ${slurm_acc} -p ${slurm_partition} scripts/templates/wideresnet_cifar100_ee/der++.sh ${num_tasks} 4 ${num_exemplars} ${alpha} ${beta} ${ic_config}
sbatch -A ${slurm_acc} -p ${slurm_partition} scripts/templates/wideresnet_cifar100_ee/der++.sh ${num_tasks} 5 ${num_exemplars} ${alpha} ${beta} ${ic_config}
alpha=0.2
beta=1.0
sbatch -A ${slurm_acc} -p ${slurm_partition} scripts/templates/wideresnet_cifar100_ee/der++.sh ${num_tasks} 3 ${num_exemplars} ${alpha} ${beta} ${ic_config}
sbatch -A ${slurm_acc} -p ${slurm_partition} scripts/templates/wideresnet_cifar100_ee/der++.sh ${num_tasks} 4 ${num_exemplars} ${alpha} ${beta} ${ic_config}
sbatch -A ${slurm_acc} -p ${slurm_partition} scripts/templates/wideresnet_cifar100_ee/der++.sh ${num_tasks} 5 ${num_exemplars} ${alpha} ${beta} ${ic_config}
alpha=0.1
beta=1.0
sbatch -A ${slurm_acc} -p ${slurm_partition} scripts/templates/wideresnet_cifar100_ee/der++.sh ${num_tasks} 3 ${num_exemplars} ${alpha} ${beta} ${ic_config}
sbatch -A ${slurm_acc} -p ${slurm_partition} scripts/templates/wideresnet_cifar100_ee/der++.sh ${num_tasks} 4 ${num_exemplars} ${alpha} ${beta} ${ic_config}
sbatch -A ${slurm_acc} -p ${slurm_partition} scripts/templates/wideresnet_cifar100_ee/der++.sh ${num_tasks} 5 ${num_exemplars} ${alpha} ${beta} ${ic_config}
alpha=0.01
beta=1.0
sbatch -A ${slurm_acc} -p ${slurm_partition} scripts/templates/wideresnet_cifar100_ee/der++.sh ${num_tasks} 3 ${num_exemplars} ${alpha} ${beta} ${ic_config}
sbatch -A ${slurm_acc} -p ${slurm_partition} scripts/templates/wideresnet_cifar100_ee/der++.sh ${num_tasks} 4 ${num_exemplars} ${alpha} ${beta} ${ic_config}
sbatch -A ${slurm_acc} -p ${slurm_partition} scripts/templates/wideresnet_cifar100_ee/der++.sh ${num_tasks} 5 ${num_exemplars} ${alpha} ${beta} ${ic_config}
alpha=0.2
beta=0.5
sbatch -A ${slurm_acc} -p ${slurm_partition} scripts/templates/wideresnet_cifar100_ee/der++.sh ${num_tasks} 3 ${num_exemplars} ${alpha} ${beta} ${ic_config}
sbatch -A ${slurm_acc} -p ${slurm_partition} scripts/templates/wideresnet_cifar100_ee/der++.sh ${num_tasks} 4 ${num_exemplars} ${alpha} ${beta} ${ic_config}
sbatch -A ${slurm_acc} -p ${slurm_partition} scripts/templates/wideresnet_cifar100_ee/der++.sh ${num_tasks} 5 ${num_exemplars} ${alpha} ${beta} ${ic_config}
alpha=0.1
beta=0.5
sbatch -A ${slurm_acc} -p ${slurm_partition} scripts/templates/wideresnet_cifar100_ee/der++.sh ${num_tasks} 3 ${num_exemplars} ${alpha} ${beta} ${ic_config}
sbatch -A ${slurm_acc} -p ${slurm_partition} scripts/templates/wideresnet_cifar100_ee/der++.sh ${num_tasks} 4 ${num_exemplars} ${alpha} ${beta} ${ic_config}
sbatch -A ${slurm_acc} -p ${slurm_partition} scripts/templates/wideresnet_cifar100_ee/der++.sh ${num_tasks} 5 ${num_exemplars} ${alpha} ${beta} ${ic_config}
alpha=0.01
beta=0.5
sbatch -A ${slurm_acc} -p ${slurm_partition} scripts/templates/wideresnet_cifar100_ee/der++.sh ${num_tasks} 3 ${num_exemplars} ${alpha} ${beta} ${ic_config}
sbatch -A ${slurm_acc} -p ${slurm_partition} scripts/templates/wideresnet_cifar100_ee/der++.sh ${num_tasks} 4 ${num_exemplars} ${alpha} ${beta} ${ic_config}
sbatch -A ${slurm_acc} -p ${slurm_partition} scripts/templates/wideresnet_cifar100_ee/der++.sh ${num_tasks} 5 ${num_exemplars} ${alpha} ${beta} ${ic_config}
