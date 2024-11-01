#!/bin/bash



# Environment Variables
export PYENV_VERSION=mmxbrcp
export LD_LIBRARY_PATH="/autofs/space/crater_001/tools/usr/lib64:${LD_LIBRARY_PATH}"
export CUDA_VISIBLE_DEVICES=MIG-752466c4-c584-5e2c-9402-e840a3cf5e6f



echo 'Started AM-SB (Regression) Training on TCGA-BRCA Database.'



# Train
python code/models/clam/train_val_model_fp.py \
 --gpu_id 0 \
 --results_dir '/autofs/space/crater_001/projects/breast-cancer-pathology/results/CLAM/TCGA-BRCA/mmxbrcp/All/checkpoints' \
 --dataset 'TCGA-BRCA' \
 --base_data_path '/autofs/space/crater_001/datasets/public/TCGA-BRCA' \
 --experimental_strategy 'All' \
 --features_h5_dir '/autofs/space/crater_001/projects/breast-cancer-pathology/results/CONCH/TCGA-BRCA/mmxbrcp/DiagnosticSlide/SegmentationHistoQC/features' '/autofs/space/crater_001/projects/breast-cancer-pathology/results/CONCH/TCGA-BRCA/mmxbrcp/TissueSlide/SegmentationHistoQC/features' \
 --label 'hallmark_angiogenesis' \
 --config_json 'code/models/clam/config/regression/tcgabrca_conch_fts_am_sb_config.json'



 echo 'Finished AM-SB (Regression) Training on TCGA-BRCA Database.'