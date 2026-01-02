#!/bin/bash



# Environment Variables
export PYENV_VERSION=mmxbrcp
export LD_LIBRARY_PATH="/autofs/space/crater_001/tools/usr/lib64:${LD_LIBRARY_PATH}"



python code/clinical-subtype-analysis/create_results_plots_v2.py \
 --config code/clinical-subtype-analysis/models_checkpoints.json \
 --output_path results/figures/clinical-subtype-analysis \
 --metrics AUC

python code/clinical-subtype-analysis/create_results_plots_v2.py \
 --config code/clinical-subtype-analysis/models_checkpoints.json \
 --output_path results/figures/clinical-subtype-analysis \
 --metrics PCC