#!/bin/bash

set -e

# This slurm script allows to run all main experiments on equal split of ImageNet100

ic_config=imagenet100_resnet18_sdn
# <--- --- --- ImageNet100x5 --- --- --- --->
num_tasks=5

# <--- ANCL --->
num_exemplars=0
lamb=1.0
lamb_a=1.0
sbatch -A plgdynamic2-gpu-a100 -p plgrid-gpu-a100 scripts/templates/imagenet100_base_rn18/ancl.sh ${num_tasks} 0 ${num_exemplars} ${lamb} ${lamb_a}
sbatch -A plgdynamic2-gpu-a100 -p plgrid-gpu-a100 scripts/templates/imagenet100_base_rn18/ancl.sh ${num_tasks} 1 ${num_exemplars} ${lamb} ${lamb_a}
sbatch -A plgdynamic2-gpu-a100 -p plgrid-gpu-a100 scripts/templates/imagenet100_base_rn18/ancl.sh ${num_tasks} 2 ${num_exemplars} ${lamb} ${lamb_a}

lamb=1.0
lamb_a=2.0
sbatch -A plgdynamic2-gpu-a100 -p plgrid-gpu-a100 scripts/templates/imagenet100_ee_rn18/ancl.sh ${num_tasks} 0 ${num_exemplars} ${lamb} ${lamb_a} ${ic_config}
sbatch -A plgdynamic2-gpu-a100 -p plgrid-gpu-a100 scripts/templates/imagenet100_ee_rn18/ancl.sh ${num_tasks} 1 ${num_exemplars} ${lamb} ${lamb_a} ${ic_config}
sbatch -A plgdynamic2-gpu-a100 -p plgrid-gpu-a100 scripts/templates/imagenet100_ee_rn18/ancl.sh ${num_tasks} 2 ${num_exemplars} ${lamb} ${lamb_a} ${ic_config}

# <--- BiC --->
num_exemplars=2000
lamb=3
sbatch -A plgdynamic2-gpu-a100 -p plgrid-gpu-a100 scripts/templates/imagenet100_base_rn18/bic.sh ${num_tasks} 0 ${num_exemplars} ${lamb}
sbatch -A plgdynamic2-gpu-a100 -p plgrid-gpu-a100 scripts/templates/imagenet100_base_rn18/bic.sh ${num_tasks} 1 ${num_exemplars} ${lamb}
sbatch -A plgdynamic2-gpu-a100 -p plgrid-gpu-a100 scripts/templates/imagenet100_base_rn18/bic.sh ${num_tasks} 2 ${num_exemplars} ${lamb}

lamb=2
sbatch -A plgdynamic2-gpu-a100 -p plgrid-gpu-a100 scripts/templates/imagenet100_ee_rn18/bic.sh ${num_tasks} 0 ${num_exemplars} ${lamb} ${ic_config}
sbatch -A plgdynamic2-gpu-a100 -p plgrid-gpu-a100 scripts/templates/imagenet100_ee_rn18/bic.sh ${num_tasks} 1 ${num_exemplars} ${lamb} ${ic_config}
sbatch -A plgdynamic2-gpu-a100 -p plgrid-gpu-a100 scripts/templates/imagenet100_ee_rn18/bic.sh ${num_tasks} 2 ${num_exemplars} ${lamb} ${ic_config}

# <--- DER++ --->
num_exemplars=2000
alpha=0.5
beta=0.5
sbatch -A plgimprmoe-gpu-a100 -p plgrid-gpu-a100 scripts/templates/imagenet100_base_rn18/der++.sh ${num_tasks} 0 ${num_exemplars} ${alpha} ${beta}
sbatch -A plgimprmoe-gpu-a100 -p plgrid-gpu-a100 scripts/templates/imagenet100_base_rn18/der++.sh ${num_tasks} 1 ${num_exemplars} ${alpha} ${beta}
sbatch -A plgimprmoe-gpu-a100 -p plgrid-gpu-a100 scripts/templates/imagenet100_base_rn18/der++.sh ${num_tasks} 2 ${num_exemplars} ${alpha} ${beta}

alpha=0.5
beta=0.5
sbatch -A plgimprmoe-gpu-a100 -p plgrid-gpu-a100 scripts/templates/imagenet100_ee_rn18/der++.sh ${num_tasks} 0 ${num_exemplars} ${alpha} ${beta} ${ic_config}
sbatch -A plgimprmoe-gpu-a100 -p plgrid-gpu-a100 scripts/templates/imagenet100_ee_rn18/der++.sh ${num_tasks} 1 ${num_exemplars} ${alpha} ${beta} ${ic_config}
sbatch -A plgimprmoe-gpu-a100 -p plgrid-gpu-a100 scripts/templates/imagenet100_ee_rn18/der++.sh ${num_tasks} 2 ${num_exemplars} ${alpha} ${beta} ${ic_config}

# <--- EWC --->
lamb=10000
alpha=0.5
sbatch -A plgdynamic2-gpu-a100 -p plgrid-gpu-a100 scripts/templates/imagenet100_base_rn18/ewc.sh ${num_tasks} 0 ${lamb} ${alpha}
sbatch -A plgdynamic2-gpu-a100 -p plgrid-gpu-a100 scripts/templates/imagenet100_base_rn18/ewc.sh ${num_tasks} 1 ${lamb} ${alpha}
sbatch -A plgdynamic2-gpu-a100 -p plgrid-gpu-a100 scripts/templates/imagenet100_base_rn18/ewc.sh ${num_tasks} 2 ${lamb} ${alpha}

lamb=10000
alpha=0.5
sbatch -A plgdynamic2-gpu-a100 -p plgrid-gpu-a100 scripts/templates/imagenet100_ee_rn18/ewc.sh ${num_tasks} 0 ${lamb} ${alpha} ${ic_config}
sbatch -A plgdynamic2-gpu-a100 -p plgrid-gpu-a100 scripts/templates/imagenet100_ee_rn18/ewc.sh ${num_tasks} 1 ${lamb} ${alpha} ${ic_config}
sbatch -A plgdynamic2-gpu-a100 -p plgrid-gpu-a100 scripts/templates/imagenet100_ee_rn18/ewc.sh ${num_tasks} 2 ${lamb} ${alpha} ${ic_config}

# <--- ER --->
num_exemplars=2000
sbatch -A plgdynamic2-gpu-a100 -p plgrid-gpu-a100 scripts/templates/imagenet100_base_rn18/er.sh ${num_tasks} 0 ${num_exemplars}
sbatch -A plgdynamic2-gpu-a100 -p plgrid-gpu-a100 scripts/templates/imagenet100_base_rn18/er.sh ${num_tasks} 1 ${num_exemplars}
sbatch -A plgdynamic2-gpu-a100 -p plgrid-gpu-a100 scripts/templates/imagenet100_base_rn18/er.sh ${num_tasks} 2 ${num_exemplars}

sbatch -A plgdynamic2-gpu-a100 -p plgrid-gpu-a100 scripts/templates/imagenet100_ee_rn18/er.sh ${num_tasks} 0 ${num_exemplars} ${ic_config}
sbatch -A plgdynamic2-gpu-a100 -p plgrid-gpu-a100 scripts/templates/imagenet100_ee_rn18/er.sh ${num_tasks} 1 ${num_exemplars} ${ic_config}
sbatch -A plgdynamic2-gpu-a100 -p plgrid-gpu-a100 scripts/templates/imagenet100_ee_rn18/er.sh ${num_tasks} 2 ${num_exemplars} ${ic_config}

# <--- FT --->
num_exemplars=0
sbatch -A plgdynamic2-gpu-a100 -p plgrid-gpu-a100 scripts/templates/imagenet100_base_rn18/ft.sh ${num_tasks} 0 ${num_exemplars}
sbatch -A plgdynamic2-gpu-a100 -p plgrid-gpu-a100 scripts/templates/imagenet100_base_rn18/ft.sh ${num_tasks} 1 ${num_exemplars}
sbatch -A plgdynamic2-gpu-a100 -p plgrid-gpu-a100 scripts/templates/imagenet100_base_rn18/ft.sh ${num_tasks} 2 ${num_exemplars}

sbatch -A plgdynamic2-gpu-a100 -p plgrid-gpu-a100 scripts/templates/imagenet100_ee_rn18/ft.sh ${num_tasks} 0 ${num_exemplars} ${ic_config}
sbatch -A plgdynamic2-gpu-a100 -p plgrid-gpu-a100 scripts/templates/imagenet100_ee_rn18/ft.sh ${num_tasks} 1 ${num_exemplars} ${ic_config}
sbatch -A plgdynamic2-gpu-a100 -p plgrid-gpu-a100 scripts/templates/imagenet100_ee_rn18/ft.sh ${num_tasks} 2 ${num_exemplars} ${ic_config}

# <--- FT+Ex --->
num_exemplars=2000
sbatch -A plgdynamic2-gpu-a100 -p plgrid-gpu-a100 scripts/templates/imagenet100_base_rn18/ft.sh ${num_tasks} 0 ${num_exemplars}
sbatch -A plgdynamic2-gpu-a100 -p plgrid-gpu-a100 scripts/templates/imagenet100_base_rn18/ft.sh ${num_tasks} 1 ${num_exemplars}
sbatch -A plgdynamic2-gpu-a100 -p plgrid-gpu-a100 scripts/templates/imagenet100_base_rn18/ft.sh ${num_tasks} 2 ${num_exemplars}

sbatch -A plgdynamic2-gpu-a100 -p plgrid-gpu-a100 scripts/templates/imagenet100_ee_rn18/ft.sh ${num_tasks} 0 ${num_exemplars} ${ic_config}
sbatch -A plgdynamic2-gpu-a100 -p plgrid-gpu-a100 scripts/templates/imagenet100_ee_rn18/ft.sh ${num_tasks} 1 ${num_exemplars} ${ic_config}
sbatch -A plgdynamic2-gpu-a100 -p plgrid-gpu-a100 scripts/templates/imagenet100_ee_rn18/ft.sh ${num_tasks} 2 ${num_exemplars} ${ic_config}

# <--- GDumb --->
num_exemplars=2000
sbatch -A plgdynamic2-gpu-a100 -p plgrid-gpu-a100 scripts/templates/imagenet100_base_rn18/gdumb.sh ${num_tasks} 0 ${num_exemplars}
sbatch -A plgdynamic2-gpu-a100 -p plgrid-gpu-a100 scripts/templates/imagenet100_base_rn18/gdumb.sh ${num_tasks} 1 ${num_exemplars}
sbatch -A plgdynamic2-gpu-a100 -p plgrid-gpu-a100 scripts/templates/imagenet100_base_rn18/gdumb.sh ${num_tasks} 2 ${num_exemplars}

sbatch -A plgdynamic2-gpu-a100 -p plgrid-gpu-a100 scripts/templates/imagenet100_ee_rn18/gdumb.sh ${num_tasks} 0 ${num_exemplars} ${ic_config}
sbatch -A plgdynamic2-gpu-a100 -p plgrid-gpu-a100 scripts/templates/imagenet100_ee_rn18/gdumb.sh ${num_tasks} 1 ${num_exemplars} ${ic_config}
sbatch -A plgdynamic2-gpu-a100 -p plgrid-gpu-a100 scripts/templates/imagenet100_ee_rn18/gdumb.sh ${num_tasks} 2 ${num_exemplars} ${ic_config}

# <--- LODE --->
num_exemplars=2000
const=1.0
ro=0.1
sbatch -A plgdynamic2-gpu-a100 -p plgrid-gpu-a100 scripts/templates/imagenet100_base_rn18/lode.sh ${num_tasks} 0 ${num_exemplars} ${const} ${ro}
sbatch -A plgdynamic2-gpu-a100 -p plgrid-gpu-a100 scripts/templates/imagenet100_base_rn18/lode.sh ${num_tasks} 1 ${num_exemplars} ${const} ${ro}
sbatch -A plgdynamic2-gpu-a100 -p plgrid-gpu-a100 scripts/templates/imagenet100_base_rn18/lode.sh ${num_tasks} 2 ${num_exemplars} ${const} ${ro}

const=3.0
ro=0.5
sbatch -A plgdynamic2-gpu-a100 -p plgrid-gpu-a100 scripts/templates/imagenet100_ee_rn18/lode.sh ${num_tasks} 0 ${num_exemplars} ${const} ${ro} ${ic_config}
sbatch -A plgdynamic2-gpu-a100 -p plgrid-gpu-a100 scripts/templates/imagenet100_ee_rn18/lode.sh ${num_tasks} 1 ${num_exemplars} ${const} ${ro} ${ic_config}
sbatch -A plgdynamic2-gpu-a100 -p plgrid-gpu-a100 scripts/templates/imagenet100_ee_rn18/lode.sh ${num_tasks} 2 ${num_exemplars} ${const} ${ro} ${ic_config}

# <--- LwF --->
lamb=1.0
sbatch -A plgdynamic2-gpu-a100 -p plgrid-gpu-a100 scripts/templates/imagenet100_base_rn18/lwf.sh ${num_tasks} 0 ${lamb}
sbatch -A plgdynamic2-gpu-a100 -p plgrid-gpu-a100 scripts/templates/imagenet100_base_rn18/lwf.sh ${num_tasks} 1 ${lamb}
sbatch -A plgdynamic2-gpu-a100 -p plgrid-gpu-a100 scripts/templates/imagenet100_base_rn18/lwf.sh ${num_tasks} 2 ${lamb}

lamb=1.0
sbatch -A plgdynamic2-gpu-a100 -p plgrid-gpu-a100 scripts/templates/imagenet100_ee_rn18/lwf.sh ${num_tasks} 0 ${lamb} ${ic_config}
sbatch -A plgdynamic2-gpu-a100 -p plgrid-gpu-a100 scripts/templates/imagenet100_ee_rn18/lwf.sh ${num_tasks} 1 ${lamb} ${ic_config}
sbatch -A plgdynamic2-gpu-a100 -p plgrid-gpu-a100 scripts/templates/imagenet100_ee_rn18/lwf.sh ${num_tasks} 2 ${lamb} ${ic_config}

# <--- Joint --->
sbatch -A plgdynamic2-gpu-a100 -p plgrid-gpu-a100 scripts/templates/imagenet100_base_rn18/joint.sh ${num_tasks} 0
sbatch -A plgdynamic2-gpu-a100 -p plgrid-gpu-a100 scripts/templates/imagenet100_base_rn18/joint.sh ${num_tasks} 1
sbatch -A plgdynamic2-gpu-a100 -p plgrid-gpu-a100 scripts/templates/imagenet100_base_rn18/joint.sh ${num_tasks} 2

sbatch -A plgdynamic2-gpu-a100 -p plgrid-gpu-a100 scripts/templates/imagenet100_ee_rn18/joint.sh ${num_tasks} 0 ${ic_config}
sbatch -A plgdynamic2-gpu-a100 -p plgrid-gpu-a100 scripts/templates/imagenet100_ee_rn18/joint.sh ${num_tasks} 1 ${ic_config}
sbatch -A plgdynamic2-gpu-a100 -p plgrid-gpu-a100 scripts/templates/imagenet100_ee_rn18/joint.sh ${num_tasks} 2 ${ic_config}

# <--- SSIL --->
num_exemplars=2000
lamb=0.25
sbatch -A plgdynamic2-gpu-a100 -p plgrid-gpu-a100 scripts/templates/imagenet100_base_rn18/ssil.sh ${num_tasks} 0 ${num_exemplars} ${lamb}
sbatch -A plgdynamic2-gpu-a100 -p plgrid-gpu-a100 scripts/templates/imagenet100_base_rn18/ssil.sh ${num_tasks} 1 ${num_exemplars} ${lamb}
sbatch -A plgdynamic2-gpu-a100 -p plgrid-gpu-a100 scripts/templates/imagenet100_base_rn18/ssil.sh ${num_tasks} 2 ${num_exemplars} ${lamb}

lamb=0.25
sbatch -A plgdynamic2-gpu-a100 -p plgrid-gpu-a100 scripts/templates/imagenet100_ee_rn18/ssil.sh ${num_tasks} 0 ${num_exemplars} ${lamb} ${ic_config}
sbatch -A plgdynamic2-gpu-a100 -p plgrid-gpu-a100 scripts/templates/imagenet100_ee_rn18/ssil.sh ${num_tasks} 1 ${num_exemplars} ${lamb} ${ic_config}
sbatch -A plgdynamic2-gpu-a100 -p plgrid-gpu-a100 scripts/templates/imagenet100_ee_rn18/ssil.sh ${num_tasks} 2 ${num_exemplars} ${lamb} ${ic_config}






# <--- --- --- ImageNet100x10 --- --- --- --->
num_tasks=10

# <--- ANCL --->
num_exemplars=0
lamb=1.0
lamb_a=1.0
sbatch -A plgdynamic2-gpu-a100 -p plgrid-gpu-a100 scripts/templates/imagenet100_base_rn18/ancl.sh ${num_tasks} 0 ${num_exemplars} ${lamb} ${lamb_a}
sbatch -A plgdynamic2-gpu-a100 -p plgrid-gpu-a100 scripts/templates/imagenet100_base_rn18/ancl.sh ${num_tasks} 1 ${num_exemplars} ${lamb} ${lamb_a}
sbatch -A plgdynamic2-gpu-a100 -p plgrid-gpu-a100 scripts/templates/imagenet100_base_rn18/ancl.sh ${num_tasks} 2 ${num_exemplars} ${lamb} ${lamb_a}

lamb=1.0
lamb_a=2.0
sbatch -A plgdynamic2-gpu-a100 -p plgrid-gpu-a100 scripts/templates/imagenet100_ee_rn18/ancl.sh ${num_tasks} 0 ${num_exemplars} ${lamb} ${lamb_a} ${ic_config}
sbatch -A plgdynamic2-gpu-a100 -p plgrid-gpu-a100 scripts/templates/imagenet100_ee_rn18/ancl.sh ${num_tasks} 1 ${num_exemplars} ${lamb} ${lamb_a} ${ic_config}
sbatch -A plgdynamic2-gpu-a100 -p plgrid-gpu-a100 scripts/templates/imagenet100_ee_rn18/ancl.sh ${num_tasks} 2 ${num_exemplars} ${lamb} ${lamb_a} ${ic_config}

# <--- BiC --->
num_exemplars=2000
lamb=3
sbatch -A plgdynamic2-gpu-a100 -p plgrid-gpu-a100 scripts/templates/imagenet100_base_rn18/bic.sh ${num_tasks} 0 ${num_exemplars} ${lamb}
sbatch -A plgdynamic2-gpu-a100 -p plgrid-gpu-a100 scripts/templates/imagenet100_base_rn18/bic.sh ${num_tasks} 1 ${num_exemplars} ${lamb}
sbatch -A plgdynamic2-gpu-a100 -p plgrid-gpu-a100 scripts/templates/imagenet100_base_rn18/bic.sh ${num_tasks} 2 ${num_exemplars} ${lamb}

lamb=2
sbatch -A plgdynamic2-gpu-a100 -p plgrid-gpu-a100 scripts/templates/imagenet100_ee_rn18/bic.sh ${num_tasks} 0 ${num_exemplars} ${lamb} ${ic_config}
sbatch -A plgdynamic2-gpu-a100 -p plgrid-gpu-a100 scripts/templates/imagenet100_ee_rn18/bic.sh ${num_tasks} 1 ${num_exemplars} ${lamb} ${ic_config}
sbatch -A plgdynamic2-gpu-a100 -p plgrid-gpu-a100 scripts/templates/imagenet100_ee_rn18/bic.sh ${num_tasks} 2 ${num_exemplars} ${lamb} ${ic_config}

# <--- DER++ --->
num_exemplars=2000
alpha=0.5
beta=0.5
sbatch -A plgimprmoe-gpu-a100 -p plgrid-gpu-a100 scripts/templates/imagenet100_base_rn18/der++.sh ${num_tasks} 0 ${num_exemplars} ${alpha} ${beta}
sbatch -A plgimprmoe-gpu-a100 -p plgrid-gpu-a100 scripts/templates/imagenet100_base_rn18/der++.sh ${num_tasks} 1 ${num_exemplars} ${alpha} ${beta}
sbatch -A plgimprmoe-gpu-a100 -p plgrid-gpu-a100 scripts/templates/imagenet100_base_rn18/der++.sh ${num_tasks} 4 ${num_exemplars} ${alpha} ${beta}

alpha=0.5
beta=0.5
sbatch -A plgimprmoe-gpu-a100 -p plgrid-gpu-a100 scripts/templates/imagenet100_ee_rn18/der++.sh ${num_tasks} 0 ${num_exemplars} ${alpha} ${beta} ${ic_config}
sbatch -A plgimprmoe-gpu-a100 -p plgrid-gpu-a100 scripts/templates/imagenet100_ee_rn18/der++.sh ${num_tasks} 3 ${num_exemplars} ${alpha} ${beta} ${ic_config}
sbatch -A plgimprmoe-gpu-a100 -p plgrid-gpu-a100 scripts/templates/imagenet100_ee_rn18/der++.sh ${num_tasks} 4 ${num_exemplars} ${alpha} ${beta} ${ic_config}

# <--- EWC --->
lamb=10000
alpha=0.5
sbatch -A plgdynamic2-gpu-a100 -p plgrid-gpu-a100 scripts/templates/imagenet100_base_rn18/ewc.sh ${num_tasks} 0 ${lamb} ${alpha}
sbatch -A plgdynamic2-gpu-a100 -p plgrid-gpu-a100 scripts/templates/imagenet100_base_rn18/ewc.sh ${num_tasks} 1 ${lamb} ${alpha}
sbatch -A plgdynamic2-gpu-a100 -p plgrid-gpu-a100 scripts/templates/imagenet100_base_rn18/ewc.sh ${num_tasks} 2 ${lamb} ${alpha}

lamb=10000
alpha=0.5
sbatch -A plgdynamic2-gpu-a100 -p plgrid-gpu-a100 scripts/templates/imagenet100_ee_rn18/ewc.sh ${num_tasks} 0 ${lamb} ${alpha} ${ic_config}
sbatch -A plgdynamic2-gpu-a100 -p plgrid-gpu-a100 scripts/templates/imagenet100_ee_rn18/ewc.sh ${num_tasks} 1 ${lamb} ${alpha} ${ic_config}
sbatch -A plgdynamic2-gpu-a100 -p plgrid-gpu-a100 scripts/templates/imagenet100_ee_rn18/ewc.sh ${num_tasks} 2 ${lamb} ${alpha} ${ic_config}

# <--- ER --->
num_exemplars=2000
sbatch -A plgdynamic2-gpu-a100 -p plgrid-gpu-a100 scripts/templates/imagenet100_base_rn18/er.sh ${num_tasks} 0 ${num_exemplars}
sbatch -A plgdynamic2-gpu-a100 -p plgrid-gpu-a100 scripts/templates/imagenet100_base_rn18/er.sh ${num_tasks} 1 ${num_exemplars}
sbatch -A plgdynamic2-gpu-a100 -p plgrid-gpu-a100 scripts/templates/imagenet100_base_rn18/er.sh ${num_tasks} 2 ${num_exemplars}

sbatch -A plgdynamic2-gpu-a100 -p plgrid-gpu-a100 scripts/templates/imagenet100_ee_rn18/er.sh ${num_tasks} 0 ${num_exemplars} ${ic_config}
sbatch -A plgdynamic2-gpu-a100 -p plgrid-gpu-a100 scripts/templates/imagenet100_ee_rn18/er.sh ${num_tasks} 1 ${num_exemplars} ${ic_config}
sbatch -A plgdynamic2-gpu-a100 -p plgrid-gpu-a100 scripts/templates/imagenet100_ee_rn18/er.sh ${num_tasks} 2 ${num_exemplars} ${ic_config}

# <--- FT --->
num_exemplars=0
sbatch -A plgdynamic2-gpu-a100 -p plgrid-gpu-a100 scripts/templates/imagenet100_base_rn18/ft.sh ${num_tasks} 0 ${num_exemplars}
sbatch -A plgdynamic2-gpu-a100 -p plgrid-gpu-a100 scripts/templates/imagenet100_base_rn18/ft.sh ${num_tasks} 1 ${num_exemplars}
sbatch -A plgdynamic2-gpu-a100 -p plgrid-gpu-a100 scripts/templates/imagenet100_base_rn18/ft.sh ${num_tasks} 2 ${num_exemplars}

sbatch -A plgdynamic2-gpu-a100 -p plgrid-gpu-a100 scripts/templates/imagenet100_ee_rn18/ft.sh ${num_tasks} 0 ${num_exemplars} ${ic_config}
sbatch -A plgdynamic2-gpu-a100 -p plgrid-gpu-a100 scripts/templates/imagenet100_ee_rn18/ft.sh ${num_tasks} 1 ${num_exemplars} ${ic_config}
sbatch -A plgdynamic2-gpu-a100 -p plgrid-gpu-a100 scripts/templates/imagenet100_ee_rn18/ft.sh ${num_tasks} 2 ${num_exemplars} ${ic_config}

# <--- FT+Ex --->
num_exemplars=2000
sbatch -A plgdynamic2-gpu-a100 -p plgrid-gpu-a100 scripts/templates/imagenet100_base_rn18/ft.sh ${num_tasks} 0 ${num_exemplars}
sbatch -A plgdynamic2-gpu-a100 -p plgrid-gpu-a100 scripts/templates/imagenet100_base_rn18/ft.sh ${num_tasks} 1 ${num_exemplars}
sbatch -A plgdynamic2-gpu-a100 -p plgrid-gpu-a100 scripts/templates/imagenet100_base_rn18/ft.sh ${num_tasks} 2 ${num_exemplars}

sbatch -A plgdynamic2-gpu-a100 -p plgrid-gpu-a100 scripts/templates/imagenet100_ee_rn18/ft.sh ${num_tasks} 0 ${num_exemplars} ${ic_config}
sbatch -A plgdynamic2-gpu-a100 -p plgrid-gpu-a100 scripts/templates/imagenet100_ee_rn18/ft.sh ${num_tasks} 1 ${num_exemplars} ${ic_config}
sbatch -A plgdynamic2-gpu-a100 -p plgrid-gpu-a100 scripts/templates/imagenet100_ee_rn18/ft.sh ${num_tasks} 2 ${num_exemplars} ${ic_config}

# <--- GDumb --->
num_exemplars=2000
sbatch -A plgdynamic2-gpu-a100 -p plgrid-gpu-a100 scripts/templates/imagenet100_base_rn18/gdumb.sh ${num_tasks} 0 ${num_exemplars}
sbatch -A plgdynamic2-gpu-a100 -p plgrid-gpu-a100 scripts/templates/imagenet100_base_rn18/gdumb.sh ${num_tasks} 1 ${num_exemplars}
sbatch -A plgdynamic2-gpu-a100 -p plgrid-gpu-a100 scripts/templates/imagenet100_base_rn18/gdumb.sh ${num_tasks} 2 ${num_exemplars}

sbatch -A plgdynamic2-gpu-a100 -p plgrid-gpu-a100 scripts/templates/imagenet100_ee_rn18/gdumb.sh ${num_tasks} 0 ${num_exemplars} ${ic_config}
sbatch -A plgdynamic2-gpu-a100 -p plgrid-gpu-a100 scripts/templates/imagenet100_ee_rn18/gdumb.sh ${num_tasks} 1 ${num_exemplars} ${ic_config}
sbatch -A plgdynamic2-gpu-a100 -p plgrid-gpu-a100 scripts/templates/imagenet100_ee_rn18/gdumb.sh ${num_tasks} 2 ${num_exemplars} ${ic_config}

# <--- LODE --->
num_exemplars=2000
const=1.0
ro=0.1
sbatch -A plgdynamic2-gpu-a100 -p plgrid-gpu-a100 scripts/templates/imagenet100_base_rn18/lode.sh ${num_tasks} 0 ${num_exemplars} ${const} ${ro}
sbatch -A plgdynamic2-gpu-a100 -p plgrid-gpu-a100 scripts/templates/imagenet100_base_rn18/lode.sh ${num_tasks} 1 ${num_exemplars} ${const} ${ro}
sbatch -A plgdynamic2-gpu-a100 -p plgrid-gpu-a100 scripts/templates/imagenet100_base_rn18/lode.sh ${num_tasks} 2 ${num_exemplars} ${const} ${ro}

const=3.0
ro=0.5
sbatch -A plgdynamic2-gpu-a100 -p plgrid-gpu-a100 scripts/templates/imagenet100_ee_rn18/lode.sh ${num_tasks} 0 ${num_exemplars} ${const} ${ro} ${ic_config}
sbatch -A plgdynamic2-gpu-a100 -p plgrid-gpu-a100 scripts/templates/imagenet100_ee_rn18/lode.sh ${num_tasks} 1 ${num_exemplars} ${const} ${ro} ${ic_config}
sbatch -A plgdynamic2-gpu-a100 -p plgrid-gpu-a100 scripts/templates/imagenet100_ee_rn18/lode.sh ${num_tasks} 2 ${num_exemplars} ${const} ${ro} ${ic_config}

# <--- LwF --->
lamb=1.0
sbatch -A plgdynamic2-gpu-a100 -p plgrid-gpu-a100 scripts/templates/imagenet100_base_rn18/lwf.sh ${num_tasks} 0 ${lamb}
sbatch -A plgdynamic2-gpu-a100 -p plgrid-gpu-a100 scripts/templates/imagenet100_base_rn18/lwf.sh ${num_tasks} 1 ${lamb}
sbatch -A plgdynamic2-gpu-a100 -p plgrid-gpu-a100 scripts/templates/imagenet100_base_rn18/lwf.sh ${num_tasks} 2 ${lamb}

lamb=1.0
sbatch -A plgdynamic2-gpu-a100 -p plgrid-gpu-a100 scripts/templates/imagenet100_ee_rn18/lwf.sh ${num_tasks} 0 ${lamb} ${ic_config}
sbatch -A plgdynamic2-gpu-a100 -p plgrid-gpu-a100 scripts/templates/imagenet100_ee_rn18/lwf.sh ${num_tasks} 1 ${lamb} ${ic_config}
sbatch -A plgdynamic2-gpu-a100 -p plgrid-gpu-a100 scripts/templates/imagenet100_ee_rn18/lwf.sh ${num_tasks} 2 ${lamb} ${ic_config}

# <--- Joint --->
sbatch -A plgdynamic2-gpu-a100 -p plgrid-gpu-a100 scripts/templates/imagenet100_base_rn18/joint.sh ${num_tasks} 0
sbatch -A plgdynamic2-gpu-a100 -p plgrid-gpu-a100 scripts/templates/imagenet100_base_rn18/joint.sh ${num_tasks} 1
sbatch -A plgdynamic2-gpu-a100 -p plgrid-gpu-a100 scripts/templates/imagenet100_base_rn18/joint.sh ${num_tasks} 2

sbatch -A plgdynamic2-gpu-a100 -p plgrid-gpu-a100 scripts/templates/imagenet100_ee_rn18/joint.sh ${num_tasks} 0 ${ic_config}
sbatch -A plgdynamic2-gpu-a100 -p plgrid-gpu-a100 scripts/templates/imagenet100_ee_rn18/joint.sh ${num_tasks} 1 ${ic_config}
sbatch -A plgdynamic2-gpu-a100 -p plgrid-gpu-a100 scripts/templates/imagenet100_ee_rn18/joint.sh ${num_tasks} 2 ${ic_config}

# <--- SSIL --->
num_exemplars=2000
lamb=0.25
sbatch -A plgdynamic2-gpu-a100 -p plgrid-gpu-a100 scripts/templates/imagenet100_base_rn18/ssil.sh ${num_tasks} 0 ${num_exemplars} ${lamb}
sbatch -A plgdynamic2-gpu-a100 -p plgrid-gpu-a100 scripts/templates/imagenet100_base_rn18/ssil.sh ${num_tasks} 1 ${num_exemplars} ${lamb}
sbatch -A plgdynamic2-gpu-a100 -p plgrid-gpu-a100 scripts/templates/imagenet100_base_rn18/ssil.sh ${num_tasks} 2 ${num_exemplars} ${lamb}

lamb=0.25
sbatch -A plgdynamic2-gpu-a100 -p plgrid-gpu-a100 scripts/templates/imagenet100_ee_rn18/ssil.sh ${num_tasks} 0 ${num_exemplars} ${lamb} ${ic_config}
sbatch -A plgdynamic2-gpu-a100 -p plgrid-gpu-a100 scripts/templates/imagenet100_ee_rn18/ssil.sh ${num_tasks} 1 ${num_exemplars} ${lamb} ${ic_config}
sbatch -A plgdynamic2-gpu-a100 -p plgrid-gpu-a100 scripts/templates/imagenet100_ee_rn18/ssil.sh ${num_tasks} 2 ${num_exemplars} ${lamb} ${ic_config}
