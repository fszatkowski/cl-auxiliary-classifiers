#!/bin/bash

set -e

# This slurm script allows to run all main experiments on equal split of CIFAR100

slurm_acc=...
slurm_partition=...

ic_config=cifar100_wideresnet16_sdn
num_exemplars=2000
num_tasks=10

const=3.0
ro=0.5
sbatch -A ${slurm_acc} -p ${slurm_partition} scripts/templates/wideresnet_cifar100_ee/lode.sh ${num_tasks} 0 ${num_exemplars} ${const} ${ro} ${ic_config}
sbatch -A ${slurm_acc} -p ${slurm_partition} scripts/templates/wideresnet_cifar100_ee/lode.sh ${num_tasks} 1 ${num_exemplars} ${const} ${ro} ${ic_config}
sbatch -A ${slurm_acc} -p ${slurm_partition} scripts/templates/wideresnet_cifar100_ee/lode.sh ${num_tasks} 2 ${num_exemplars} ${const} ${ro} ${ic_config}

const=2.0
ro=0.5
sbatch -A ${slurm_acc} -p ${slurm_partition} scripts/templates/wideresnet_cifar100_ee/lode.sh ${num_tasks} 0 ${num_exemplars} ${const} ${ro} ${ic_config}
sbatch -A ${slurm_acc} -p ${slurm_partition} scripts/templates/wideresnet_cifar100_ee/lode.sh ${num_tasks} 1 ${num_exemplars} ${const} ${ro} ${ic_config}
sbatch -A ${slurm_acc} -p ${slurm_partition} scripts/templates/wideresnet_cifar100_ee/lode.sh ${num_tasks} 2 ${num_exemplars} ${const} ${ro} ${ic_config}

const=1.0
ro=0.5
sbatch -A ${slurm_acc} -p ${slurm_partition} scripts/templates/wideresnet_cifar100_ee/lode.sh ${num_tasks} 0 ${num_exemplars} ${const} ${ro} ${ic_config}
sbatch -A ${slurm_acc} -p ${slurm_partition} scripts/templates/wideresnet_cifar100_ee/lode.sh ${num_tasks} 1 ${num_exemplars} ${const} ${ro} ${ic_config}
sbatch -A ${slurm_acc} -p ${slurm_partition} scripts/templates/wideresnet_cifar100_ee/lode.sh ${num_tasks} 2 ${num_exemplars} ${const} ${ro} ${ic_config}

const=4.0
ro=0.1
sbatch -A ${slurm_acc} -p ${slurm_partition} scripts/templates/wideresnet_cifar100_ee/lode.sh ${num_tasks} 0 ${num_exemplars} ${const} ${ro} ${ic_config}
sbatch -A ${slurm_acc} -p ${slurm_partition} scripts/templates/wideresnet_cifar100_ee/lode.sh ${num_tasks} 1 ${num_exemplars} ${const} ${ro} ${ic_config}
sbatch -A ${slurm_acc} -p ${slurm_partition} scripts/templates/wideresnet_cifar100_ee/lode.sh ${num_tasks} 2 ${num_exemplars} ${const} ${ro} ${ic_config}

const=3.0
ro=0.1
sbatch -A ${slurm_acc} -p ${slurm_partition} scripts/templates/wideresnet_cifar100_ee/lode.sh ${num_tasks} 0 ${num_exemplars} ${const} ${ro} ${ic_config}
sbatch -A ${slurm_acc} -p ${slurm_partition} scripts/templates/wideresnet_cifar100_ee/lode.sh ${num_tasks} 1 ${num_exemplars} ${const} ${ro} ${ic_config}
sbatch -A ${slurm_acc} -p ${slurm_partition} scripts/templates/wideresnet_cifar100_ee/lode.sh ${num_tasks} 2 ${num_exemplars} ${const} ${ro} ${ic_config}

const=2.0
ro=0.1
sbatch -A ${slurm_acc} -p ${slurm_partition} scripts/templates/wideresnet_cifar100_ee/lode.sh ${num_tasks} 0 ${num_exemplars} ${const} ${ro} ${ic_config}
sbatch -A ${slurm_acc} -p ${slurm_partition} scripts/templates/wideresnet_cifar100_ee/lode.sh ${num_tasks} 1 ${num_exemplars} ${const} ${ro} ${ic_config}
sbatch -A ${slurm_acc} -p ${slurm_partition} scripts/templates/wideresnet_cifar100_ee/lode.sh ${num_tasks} 2 ${num_exemplars} ${const} ${ro} ${ic_config}

const=1.0
ro=0.1
sbatch -A ${slurm_acc} -p ${slurm_partition} scripts/templates/wideresnet_cifar100_ee/lode.sh ${num_tasks} 0 ${num_exemplars} ${const} ${ro} ${ic_config}
sbatch -A ${slurm_acc} -p ${slurm_partition} scripts/templates/wideresnet_cifar100_ee/lode.sh ${num_tasks} 1 ${num_exemplars} ${const} ${ro} ${ic_config}
sbatch -A ${slurm_acc} -p ${slurm_partition} scripts/templates/wideresnet_cifar100_ee/lode.sh ${num_tasks} 2 ${num_exemplars} ${const} ${ro} ${ic_config}

