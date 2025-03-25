#!/bin/bash

set -e

export PYTHONPATH=src

# Experiments
python src/plotting/icml/main_results.py
python src/plotting/icml/main_results_teaser.py
python src/plotting/icml/longer_tasks.py
python src/plotting/icml/warm_start.py
python src/plotting/icml/vit.py
python src/plotting/icml/vgg.py
python src/plotting/icml/vit_vgg_joint.py

# Ablations
python src/plotting/icml/ac_number_ablation.py
python src/plotting/icml/ac_arch_ablation.py
python src/plotting/icml/ac_lp_ablation.py

# Analyses
python src/plotting/icml/acc_change_with_acs.py
python src/plotting/icml/relative_overthinking.py
python src/plotting/icml/unique_overthinking.py
python src/plotting/icml/overthinking.py