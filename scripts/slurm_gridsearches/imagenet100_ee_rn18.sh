#!/bin/bash

set -e

for seed in 0; do
    for num_tasks in 5 10; do
        for ic_config in imagenet100_resnet18_sdn; do
            # ANCL
            num_exemplars=0
            lamb=1.0
            lamb_a=1.0
            sbatch -A plgdynamic2-gpu-a100 -p plgrid-gpu-a100 scripts/templates/imagenet100_ee_rn18/ancl.sh ${num_tasks} ${seed} ${num_exemplars} ${lamb} ${lamb_a} ${ic_config}
            #             lamb=2.0
            #             lamb_a=2.0
            #             sbatch -A plgdynamic2-gpu-a100 -p plgrid-gpu-a100 scripts/templates/imagenet100_ee_rn18/ancl.sh ${num_tasks} ${seed} ${num_exemplars} ${lamb} ${lamb_a} ${ic_config}
            lamb=0.5
            lamb_a=0.5
            sbatch -A plgdynamic2-gpu-a100 -p plgrid-gpu-a100 scripts/templates/imagenet100_ee_rn18/ancl.sh ${num_tasks} ${seed} ${num_exemplars} ${lamb} ${lamb_a} ${ic_config}
            lamb=0.75
            lamb_a=0.75
            sbatch -A plgdynamic2-gpu-a100 -p plgrid-gpu-a100 scripts/templates/imagenet100_ee_rn18/ancl.sh ${num_tasks} ${seed} ${num_exemplars} ${lamb} ${lamb_a} ${ic_config}
            lamb=1.0
            lamb_a=0.5
            sbatch -A plgdynamic2-gpu-a100 -p plgrid-gpu-a100 scripts/templates/imagenet100_ee_rn18/ancl.sh ${num_tasks} ${seed} ${num_exemplars} ${lamb} ${lamb_a} ${ic_config}
            lamb=0.5
            lamb_a=1.0
            sbatch -A plgdynamic2-gpu-a100 -p plgrid-gpu-a100 scripts/templates/imagenet100_ee_rn18/ancl.sh ${num_tasks} ${seed} ${num_exemplars} ${lamb} ${lamb_a} ${ic_config}
            #             lamb=2.0
            #             lamb_a=1.0
            #             sbatch -A plgdynamic2-gpu-a100 -p plgrid-gpu-a100 scripts/templates/imagenet100_ee_rn18/ancl.sh ${num_tasks} ${seed} ${num_exemplars} ${lamb} ${lamb_a} ${ic_config}
            #             lamb=1.0
            #             lamb_a=0.5
            #             sbatch -A plgdynamic2-gpu-a100 -p plgrid-gpu-a100 scripts/templates/imagenet100_ee_rn18/ancl.sh ${num_tasks} ${seed} ${num_exemplars} ${lamb} ${lamb_a} ${ic_config}
            #             lamb=0.5
            #             lamb_a=1.0
            #             sbatch -A plgdynamic2-gpu-a100 -p plgrid-gpu-a100 scripts/templates/imagenet100_ee_rn18/ancl.sh ${num_tasks} ${seed} ${num_exemplars} ${lamb} ${lamb_a} ${ic_config}

            # ANCL+Ex
            num_exemplars=2000
            lamb=0.1
            lamb_a=0.1
            sbatch -A plgdynamic2-gpu-a100 -p plgrid-gpu-a100 scripts/templates/imagenet100_ee_rn18/ancl.sh ${num_tasks} ${seed} ${num_exemplars} ${lamb} ${lamb_a} ${ic_config}
            #             lamb=2.0
            #             lamb_a=2.0
            #             sbatch -A plgdynamic2-gpu-a100 -p plgrid-gpu-a100 scripts/templates/imagenet100_ee_rn18/ancl.sh ${num_tasks} ${seed} ${num_exemplars} ${lamb} ${lamb_a} ${ic_config}
            lamb=0.25
            lamb_a=0.25
            sbatch -A plgdynamic2-gpu-a100 -p plgrid-gpu-a100 scripts/templates/imagenet100_ee_rn18/ancl.sh ${num_tasks} ${seed} ${num_exemplars} ${lamb} ${lamb_a} ${ic_config}
            lamb=0.5
            lamb_a=0.5
            sbatch -A plgdynamic2-gpu-a100 -p plgrid-gpu-a100 scripts/templates/imagenet100_ee_rn18/ancl.sh ${num_tasks} ${seed} ${num_exemplars} ${lamb} ${lamb_a} ${ic_config}
            lamb=0.5
            lamb_a=0.25
            sbatch -A plgdynamic2-gpu-a100 -p plgrid-gpu-a100 scripts/templates/imagenet100_ee_rn18/ancl.sh ${num_tasks} ${seed} ${num_exemplars} ${lamb} ${lamb_a} ${ic_config}
            lamb=0.25
            lamb_a=0.5
            sbatch -A plgdynamic2-gpu-a100 -p plgrid-gpu-a100 scripts/templates/imagenet100_ee_rn18/ancl.sh ${num_tasks} ${seed} ${num_exemplars} ${lamb} ${lamb_a} ${ic_config}
            #             lamb=2.0
            #             lamb_a=1.0
            #             sbatch -A plgdynamic2-gpu-a100 -p plgrid-gpu-a100 scripts/templates/imagenet100_ee_rn18/ancl.sh ${num_tasks} ${seed} ${num_exemplars} ${lamb} ${lamb_a} ${ic_config}
            #             lamb=1.0
            #             lamb_a=0.5
            #             sbatch -A plgdynamic2-gpu-a100 -p plgrid-gpu-a100 scripts/templates/imagenet100_ee_rn18/ancl.sh ${num_tasks} ${seed} ${num_exemplars} ${lamb} ${lamb_a} ${ic_config}
            lamb=0.5
            lamb_a=1.0
            sbatch -A plgdynamic2-gpu-a100 -p plgrid-gpu-a100 scripts/templates/imagenet100_ee_rn18/ancl.sh ${num_tasks} ${seed} ${num_exemplars} ${lamb} ${lamb_a} ${ic_config}

            # BiC
            num_exemplars=2000
            #             lamb=-1
            #             sbatch -A plgdynamic2-gpu-a100 -p plgrid-gpu-a100 scripts/templates/imagenet100_ee_rn18/bic.sh ${num_tasks} ${seed} ${num_exemplars} ${lamb} ${ic_config}
            #             lamb=0.5
            #             sbatch -A plgdynamic2-gpu-a100 -p plgrid-gpu-a100 scripts/templates/imagenet100_ee_rn18/bic.sh ${num_tasks} ${seed} ${num_exemplars} ${lamb} ${ic_config}
            #             lamb=1
            #             sbatch -A plgdynamic2-gpu-a100 -p plgrid-gpu-a100 scripts/templates/imagenet100_ee_rn18/bic.sh ${num_tasks} ${seed} ${num_exemplars} ${lamb} ${ic_config}
            lamb=2
            sbatch -A plgdynamic2-gpu-a100 -p plgrid-gpu-a100 scripts/templates/imagenet100_ee_rn18/bic.sh ${num_tasks} ${seed} ${num_exemplars} ${lamb} ${ic_config}
            lamb=3
            sbatch -A plgdynamic2-gpu-a100 -p plgrid-gpu-a100 scripts/templates/imagenet100_ee_rn18/bic.sh ${num_tasks} ${seed} ${num_exemplars} ${lamb} ${ic_config}
            #                         lamb=4
            #             sbatch -A plgdynamic2-gpu-a100 -p plgrid-gpu-a100 scripts/templates/imagenet100_ee_rn18/bic.sh ${num_tasks} ${seed} ${num_exemplars} ${lamb} ${ic_config}
            #             lamb=5
            #             sbatch -A plgdynamic2-gpu-a100 -p plgrid-gpu-a100 scripts/templates/imagenet100_ee_rn18/bic.sh ${num_tasks} ${seed} ${num_exemplars} ${lamb} ${ic_config}

            # EWC
            lamb=10000
            alpha=0.5
            sbatch -A plgdynamic2-gpu-a100 -p plgrid-gpu-a100 scripts/templates/imagenet100_ee_rn18/ewc.sh ${num_tasks} ${seed} ${lamb} ${alpha} ${ic_config}

            # ER
            num_exemplars=2000
            sbatch -A plgdynamic2-gpu-a100 -p plgrid-gpu-a100 scripts/templates/imagenet100_ee_rn18/er.sh ${num_tasks} ${seed} ${num_exemplars} ${ic_config}

            # FT
            num_exemplars=0
            sbatch -A plgdynamic2-gpu-a100 -p plgrid-gpu-a100 scripts/templates/imagenet100_ee_rn18/ft.sh ${num_tasks} ${seed} ${num_exemplars} ${ic_config}

            # FT+Ex
            num_exemplars=2000
            sbatch -A plgdynamic2-gpu-a100 -p plgrid-gpu-a100 scripts/templates/imagenet100_ee_rn18/ft.sh ${num_tasks} ${seed} ${num_exemplars} ${ic_config}

            # GDumb
            num_exemplars=2000
            sbatch -A plgdynamic2-gpu-a100 -p plgrid-gpu-a100 scripts/templates/imagenet100_ee_rn18/gdumb.sh ${num_tasks} ${seed} ${num_exemplars} ${ic_config}

            # iCaRL
            num_exemplars=2000
            lamb=0.25
            sbatch -A plgdynamic2-gpu-a100 -p plgrid-gpu-a100 scripts/templates/imagenet100_ee_rn18/icarl.sh ${num_tasks} ${seed} ${num_exemplars} ${lamb} ${ic_config}
            lamb=0.5
            sbatch -A plgdynamic2-gpu-a100 -p plgrid-gpu-a100 scripts/templates/imagenet100_ee_rn18/icarl.sh ${num_tasks} ${seed} ${num_exemplars} ${lamb} ${ic_config}
            lamb=0.75
            sbatch -A plgdynamic2-gpu-a100 -p plgrid-gpu-a100 scripts/templates/imagenet100_ee_rn18/icarl.sh ${num_tasks} ${seed} ${num_exemplars} ${lamb} ${ic_config}
            lamb=1
            sbatch -A plgdynamic2-gpu-a100 -p plgrid-gpu-a100 scripts/templates/imagenet100_ee_rn18/icarl.sh ${num_tasks} ${seed} ${num_exemplars} ${lamb} ${ic_config}
            #                         lamb=1.5
            #             sbatch -A plgdynamic2-gpu-a100 -p plgrid-gpu-a100 scripts/templates/imagenet100_ee_rn18/icarl.sh ${num_tasks} ${seed} ${num_exemplars} ${lamb} ${ic_config}
            #             lamb=2
            #             sbatch -A plgdynamic2-gpu-a100 -p plgrid-gpu-a100 scripts/templates/imagenet100_ee_rn18/icarl.sh ${num_tasks} ${seed} ${num_exemplars} ${lamb} ${ic_config}

            # LODE
            num_exemplars=2000
            const=3.0
            ro=1.0
            sbatch -A plgdynamic2-gpu-a100 -p plgrid-gpu-a100 scripts/templates/imagenet100_ee_rn18/lode.sh ${num_tasks} ${seed} ${num_exemplars} ${const} ${ro} ${ic_config}
            const=3.0
            ro=0.5
            sbatch -A plgdynamic2-gpu-a100 -p plgrid-gpu-a100 scripts/templates/imagenet100_ee_rn18/lode.sh ${num_tasks} ${seed} ${num_exemplars} ${const} ${ro} ${ic_config}
            const=2.0
            ro=0.5
            sbatch -A plgdynamic2-gpu-a100 -p plgrid-gpu-a100 scripts/templates/imagenet100_ee_rn18/lode.sh ${num_tasks} ${seed} ${num_exemplars} ${const} ${ro} ${ic_config}
            const=2.0
            ro=1
            sbatch -A plgdynamic2-gpu-a100 -p plgrid-gpu-a100 scripts/templates/imagenet100_ee_rn18/lode.sh ${num_tasks} ${seed} ${num_exemplars} ${const} ${ro} ${ic_config}
            const=1.5
            ro=0.5
            sbatch -A plgdynamic2-gpu-a100 -p plgrid-gpu-a100 scripts/templates/imagenet100_ee_rn18/lode.sh ${num_tasks} ${seed} ${num_exemplars} ${const} ${ro} ${ic_config}
            #             const=1.0
            #             ro=0.05
            #             sbatch -A plgdynamic2-gpu-a100 -p plgrid-gpu-a100 scripts/templates/imagenet100_ee_rn18/lode.sh ${num_tasks} ${seed} ${num_exemplars} ${const} ${ro} ${ic_config}
            #             const=1.0
            #             ro=0.01
            #             sbatch -A plgdynamic2-gpu-a100 -p plgrid-gpu-a100 scripts/templates/imagenet100_ee_rn18/lode.sh ${num_tasks} ${seed} ${num_exemplars} ${const} ${ro} ${ic_config}

            # LWF
            lamb=0.25
            sbatch -A plgdynamic2-gpu-a100 -p plgrid-gpu-a100 scripts/templates/imagenet100_ee_rn18/lwf.sh ${num_tasks} ${seed} ${lamb} ${ic_config}
            lamb=0.5
            sbatch -A plgdynamic2-gpu-a100 -p plgrid-gpu-a100 scripts/templates/imagenet100_ee_rn18/lwf.sh ${num_tasks} ${seed} ${lamb} ${ic_config}
            lamb=0.75
            sbatch -A plgdynamic2-gpu-a100 -p plgrid-gpu-a100 scripts/templates/imagenet100_ee_rn18/lwf.sh ${num_tasks} ${seed} ${lamb} ${ic_config}
            lamb=1.0
            sbatch -A plgdynamic2-gpu-a100 -p plgrid-gpu-a100 scripts/templates/imagenet100_ee_rn18/lwf.sh ${num_tasks} ${seed} ${lamb} ${ic_config}
            #             lamb=1.5
            #             sbatch -A plgdynamic2-gpu-a100 -p plgrid-gpu-a100 scripts/templates/imagenet100_ee_rn18/lwf.sh ${num_tasks} ${seed} ${lamb} ${ic_config}
            #             lamb=2.0
            #             sbatch -A plgdynamic2-gpu-a100 -p plgrid-gpu-a100 scripts/templates/imagenet100_ee_rn18/lwf.sh ${num_tasks} ${seed} ${lamb} ${ic_config}

            # Joint
            #             sbatch -A plgdynamic2-gpu-a100 -p plgrid-gpu-a100 scripts/templates/imagenet100_ee_rn18/joint.sh ${num_tasks} ${seed} ${ic_config}

            # SSIL
            num_exemplars=2000
            lamb=0.25
            sbatch -A plgdynamic2-gpu-a100 -p plgrid-gpu-a100 scripts/templates/imagenet100_ee_rn18/ssil.sh ${num_tasks} ${seed} ${num_exemplars} ${lamb} ${ic_config}
            lamb=0.5
            sbatch -A plgdynamic2-gpu-a100 -p plgrid-gpu-a100 scripts/templates/imagenet100_ee_rn18/ssil.sh ${num_tasks} ${seed} ${num_exemplars} ${lamb} ${ic_config}
            #             lamb=0.5
            #             sbatch -A plgdynamic2-gpu-a100 -p plgrid-gpu-a100 scripts/templates/imagenet100_ee_rn18/ssil.sh ${num_tasks} ${seed} ${num_exemplars} ${lamb} ${ic_config}
            #             lamb=0.75
            #             sbatch -A plgdynamic2-gpu-a100 -p plgrid-gpu-a100 scripts/templates/imagenet100_ee_rn18/ssil.sh ${num_tasks} ${seed} ${num_exemplars} ${lamb} ${ic_config}
            #             lamb=1.0
            #             sbatch -A plgdynamic2-gpu-a100 -p plgrid-gpu-a100 scripts/templates/imagenet100_ee_rn18/ssil.sh ${num_tasks} ${seed} ${num_exemplars} ${lamb} ${ic_config}
            #             lamb=1.5
            #             sbatch -A plgdynamic2-gpu-a100 -p plgrid-gpu-a100 scripts/templates/imagenet100_ee_rn18/ssil.sh ${num_tasks} ${seed} ${num_exemplars} ${lamb} ${ic_config}
            #             lamb=2.0
            #             sbatch -A plgdynamic2-gpu-a100 -p plgrid-gpu-a100 scripts/templates/imagenet100_ee_rn18/ssil.sh ${num_tasks} ${seed} ${num_exemplars} ${lamb} ${ic_config}

        done
    done
done