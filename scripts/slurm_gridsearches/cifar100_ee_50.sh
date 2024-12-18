#!/bin/bash

set -e

for seed in 0 1 2; do
    for ic_config in cifar100_resnet32_sdn; do
        for num_tasks in 50; do
            for num_exemplars in 20; do

                # BiC
                #             lamb=-1
                #             sbatch -A plgdynamic2-gpu-a100 -p plgrid-gpu-a100 scripts/templates/cifar100_ee_growing_mem/bic.sh ${num_tasks} ${seed} ${num_exemplars} ${lamb} ${ic_config}
                #             lamb=0.5
                #             sbatch -A plgdynamic2-gpu-a100 -p plgrid-gpu-a100 scripts/templates/cifar100_ee_growing_mem/bic.sh ${num_tasks} ${seed} ${num_exemplars} ${lamb} ${ic_config}
                #             lamb=1
                #             sbatch -A plgdynamic2-gpu-a100 -p plgrid-gpu-a100 scripts/templates/cifar100_ee_growing_mem/bic.sh ${num_tasks} ${seed} ${num_exemplars} ${lamb} ${ic_config}
                #             lamb=-1
                #             sbatch -A plgdynamic2-gpu-a100 -p plgrid-gpu-a100 scripts/templates/cifar100_ee_growing_mem/bic.sh ${num_tasks} ${seed} ${num_exemplars} ${lamb} ${ic_config}
                #             lamb=1
                #             sbatch -A plgdynamic2-gpu-a100 -p plgrid-gpu-a100 scripts/templates/cifar100_ee_growing_mem/bic.sh ${num_tasks} ${seed} ${num_exemplars} ${lamb} ${ic_config}
                #             lamb=2
                #             sbatch -A plgdynamic2-gpu-a100 -p plgrid-gpu-a100 scripts/templates/cifar100_ee_growing_mem/bic.sh ${num_tasks} ${seed} ${num_exemplars} ${lamb} ${ic_config}
                #             lamb=3
                #             sbatch -A plgdynamic2-gpu-a100 -p plgrid-gpu-a100 scripts/templates/cifar100_ee_growing_mem/bic.sh ${num_tasks} ${seed} ${num_exemplars} ${lamb} ${ic_config}
                #             lamb=5
                #             sbatch -A plgdynamic2-gpu-a100 -p plgrid-gpu-a100 scripts/templates/cifar100_ee_growing_mem/bic.sh ${num_tasks} ${seed} ${num_exemplars} ${lamb} ${ic_config}
                #             lamb=3
                #             sbatch -A plgdynamic2-gpu-a100 -p plgrid-gpu-a100 scripts/templates/cifar100_ee_growing_mem/bic.sh ${num_tasks} ${seed} ${num_exemplars} ${lamb} ${ic_config}
                #                         lamb=4
                #             sbatch -A plgdynamic2-gpu-a100 -p plgrid-gpu-a100 scripts/templates/cifar100_ee_growing_mem/bic.sh ${num_tasks} ${seed} ${num_exemplars} ${lamb} ${ic_config}
                #             lamb=5
                #             sbatch -A plgdynamic2-gpu-a100 -p plgrid-gpu-a100 scripts/templates/cifar100_ee_growing_mem/bic.sh ${num_tasks} ${seed} ${num_exemplars} ${lamb} ${ic_config}

                alpha=0.5
                beta=0.5
                sbatch -A plgimprmoe-gpu-a100 -p plgrid-gpu-a100 scripts/templates/cifar100_ee_growing_mem/der++.sh ${num_tasks} ${seed} ${num_exemplars} ${ic_config} ${alpha} ${beta}

                alpha=0.5
                beta=1.0
                sbatch -A plgimprmoe-gpu-a100 -p plgrid-gpu-a100 scripts/templates/cifar100_ee_growing_mem/der++.sh ${num_tasks} ${seed} ${num_exemplars} ${ic_config} ${alpha} ${beta}

                alpha=1.0
                beta=0.5
                sbatch -A plgimprmoe-gpu-a100 -p plgrid-gpu-a100 scripts/templates/cifar100_ee_growing_mem/der++.sh ${num_tasks} ${seed} ${num_exemplars} ${ic_config} ${alpha} ${beta}

                alpha=0.25
                beta=0.5
                sbatch -A plgimprmoe-gpu-a100 -p plgrid-gpu-a100 scripts/templates/cifar100_ee_growing_mem/der++.sh ${num_tasks} ${seed} ${num_exemplars} ${ic_config} ${alpha} ${beta}

                alpha=0.5
                beta=0.25
                sbatch -A plgimprmoe-gpu-a100 -p plgrid-gpu-a100 scripts/templates/cifar100_ee_growing_mem/der++.sh ${num_tasks} ${seed} ${num_exemplars} ${ic_config} ${alpha} ${beta}


                # ER
                sbatch -A plgdynamic2-gpu-a100 -p plgrid-gpu-a100 scripts/templates/cifar100_ee_growing_mem/er.sh ${num_tasks} ${seed} ${num_exemplars} ${ic_config}

                #             # FT+Ex
                #             sbatch -A plgdynamic2-gpu-a100 -p plgrid-gpu-a100 scripts/templates/cifar100_ee_growing_mem/ft.sh ${num_tasks} ${seed} ${num_exemplars} ${ic_config}
                #
                #             # GDumb
                #             sbatch -A plgdynamic2-gpu-a100 -p plgrid-gpu-a100 scripts/templates/cifar100_ee_growing_mem/gdumb.sh ${num_tasks} ${seed} ${num_exemplars} ${ic_config}

                # LODE
                #             const=3.0
                #             ro=1.0
                #             sbatch -A plgdynamic2-gpu-a100 -p plgrid-gpu-a100 scripts/templates/cifar100_ee_growing_mem/lode.sh ${num_tasks} ${seed} ${num_exemplars} ${const} ${ro} ${ic_config}
                #             const=4.0
                #             ro=0.5
                #             sbatch -A plgdynamic2-gpu-a100 -p plgrid-gpu-a100 scripts/templates/cifar100_ee_growing_mem/lode.sh ${num_tasks} ${seed} ${num_exemplars} ${const} ${ro} ${ic_config}
                #             const=1.0
                #             ro=0.5
                #             sbatch -A plgdynamic2-gpu-a100 -p plgrid-gpu-a100 scripts/templates/cifar100_ee_growing_mem/lode.sh ${num_tasks} ${seed} ${num_exemplars} ${const} ${ro} ${ic_config}
                const=2.0
                ro=0.5
                sbatch -A plgdynamic2-gpu-a100 -p plgrid-gpu-a100 scripts/templates/cifar100_ee_growing_mem/lode.sh ${num_tasks} ${seed} ${num_exemplars} ${const} ${ro} ${ic_config}
                const=3.0
                ro=0.5
                sbatch -A plgdynamic2-gpu-a100 -p plgrid-gpu-a100 scripts/templates/cifar100_ee_growing_mem/lode.sh ${num_tasks} ${seed} ${num_exemplars} ${const} ${ro} ${ic_config}
                const=4.0
                ro=0.5
                sbatch -A plgdynamic2-gpu-a100 -p plgrid-gpu-a100 scripts/templates/cifar100_ee_growing_mem/lode.sh ${num_tasks} ${seed} ${num_exemplars} ${const} ${ro} ${ic_config}
                #             const=1.0
                #             ro=0.1
                #             sbatch -A plgdynamic2-gpu-a100 -p plgrid-gpu-a100 scripts/templates/cifar100_ee_growing_mem/lode.sh ${num_tasks} ${seed} ${num_exemplars} ${const} ${ro} ${ic_config}
                #             const=2.0
                #             ro=0.1
                #             sbatch -A plgdynamic2-gpu-a100 -p plgrid-gpu-a100 scripts/templates/cifar100_ee_growing_mem/lode.sh ${num_tasks} ${seed} ${num_exemplars} ${const} ${ro} ${ic_config}
                #             const=3.0
                #             ro=0.1
                #             sbatch -A plgdynamic2-gpu-a100 -p plgrid-gpu-a100 scripts/templates/cifar100_ee_growing_mem/lode.sh ${num_tasks} ${seed} ${num_exemplars} ${const} ${ro} ${ic_config}
                #             const=4.0
                #             ro=0.1
                #             sbatch -A plgdynamic2-gpu-a100 -p plgrid-gpu-a100 scripts/templates/cifar100_ee_growing_mem/lode.sh ${num_tasks} ${seed} ${num_exemplars} ${const} ${ro} ${ic_config}
                #             const=2.0
                #             ro=0.5
                #             sbatch -A plgdynamic2-gpu-a100 -p plgrid-gpu-a100 scripts/templates/cifar100_ee_growing_mem/lode.sh ${num_tasks} ${seed} ${num_exemplars} ${const} ${ro} ${ic_config}
                #             const=2.0
                #             ro=1
                #             sbatch -A plgdynamic2-gpu-a100 -p plgrid-gpu-a100 scripts/templates/cifar100_ee_growing_mem/lode.sh ${num_tasks} ${seed} ${num_exemplars} ${const} ${ro} ${ic_config}
                #             const=1.5
                #             ro=0.5
                #             sbatch -A plgdynamic2-gpu-a100 -p plgrid-gpu-a100 scripts/templates/cifar100_ee_growing_mem/lode.sh ${num_tasks} ${seed} ${num_exemplars} ${const} ${ro} ${ic_config}
                #             const=1.0
                #             ro=0.05
                #             sbatch -A plgdynamic2-gpu-a100 -p plgrid-gpu-a100 scripts/templates/cifar100_ee_growing_mem/lode.sh ${num_tasks} ${seed} ${num_exemplars} ${const} ${ro} ${ic_config}
                #             const=1.0
                #             ro=0.01
                #             sbatch -A plgdynamic2-gpu-a100 -p plgrid-gpu-a100 scripts/templates/cifar100_ee_growing_mem/lode.sh ${num_tasks} ${seed} ${num_exemplars} ${const} ${ro} ${ic_config}


                # SSIL
                #             lamb=0.25
                #             sbatch -A plgdynamic2-gpu-a100 -p plgrid-gpu-a100 scripts/templates/cifar100_ee_growing_mem/ssil.sh ${num_tasks} ${seed} ${num_exemplars} ${lamb} ${ic_config}
                #             lamb=0.1
                #             sbatch -A plgdynamic2-gpu-a100 -p plgrid-gpu-a100 scripts/templates/cifar100_ee_growing_mem/ssil.sh ${num_tasks} ${seed} ${num_exemplars} ${lamb} ${ic_config}
                # #             lamb=0.5
                #             sbatch -A plgdynamic2-gpu-a100 -p plgrid-gpu-a100 scripts/templates/cifar100_ee_growing_mem/ssil.sh ${num_tasks} ${seed} ${num_exemplars} ${lamb} ${ic_config}
                #             lamb=0.75
                #             sbatch -A plgdynamic2-gpu-a100 -p plgrid-gpu-a100 scripts/templates/cifar100_ee_growing_mem/ssil.sh ${num_tasks} ${seed} ${num_exemplars} ${lamb} ${ic_config}
                #             lamb=1.0
                #             sbatch -A plgdynamic2-gpu-a100 -p plgrid-gpu-a100 scripts/templates/cifar100_ee_growing_mem/ssil.sh ${num_tasks} ${seed} ${num_exemplars} ${lamb} ${ic_config}
                #             lamb=1.5
                #             sbatch -A plgdynamic2-gpu-a100 -p plgrid-gpu-a100 scripts/templates/cifar100_ee_growing_mem/ssil.sh ${num_tasks} ${seed} ${num_exemplars} ${lamb} ${ic_config}
                #             lamb=2.0
                #             sbatch -A plgdynamic2-gpu-a100 -p plgrid-gpu-a100 scripts/templates/cifar100_ee_growing_mem/ssil.sh ${num_tasks} ${seed} ${num_exemplars} ${lamb} ${ic_config}
            done
        done
    done
done