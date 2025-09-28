#!/bin/bash

set -e

export PYTHONPATH=src

# Experiments
python src/plotting/main_results.py
python src/plotting/main_results_teaser.py
python src/plotting/longer_tasks.py
python src/plotting/warm_start.py
python src/plotting/vit.py
python src/plotting/vgg.py
python src/plotting/vit_vgg_joint.py

# Ablations
python src/plotting/ac_number_ablation.py
python src/plotting/ac_arch_ablation.py
python src/plotting/ac_lp_ablation.py

# Analyses
python src/plotting/acc_change_with_acs.py
python src/plotting/relative_overthinking.py
python src/plotting/unique_overthinking.py
python src/plotting/overthinking.py