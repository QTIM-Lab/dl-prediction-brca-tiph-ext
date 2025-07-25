#!/bin/bash



# Environment Variables
export PYENV_VERSION=mmxbrcp
export LD_LIBRARY_PATH="/autofs/space/crater_001/tools/usr/lib64:${LD_LIBRARY_PATH}"
export CUDA_VISIBLE_DEVICES=GPU-4458e3b8-bad5-04a4-5bf0-6e691d36235a



echo 'Started AM-SB (Regression) Training on TCGA-BRCA Database.'



# These are Dago's Augmented Features Models
CHECKPOINT_DIRS=('/autofs/space/crater_001/projects/micropath/results/augmented/hallmark_angiogenesis/2025-02-10_13-23-05' \
 '/autofs/space/crater_001/projects/micropath/results/augmented/kegg_antigen_processing_and_presentation/2025-02-10_14-48-01' \
 '/autofs/space/crater_001/projects/micropath/results/augmented/gobp_b_cell_proliferation/2025-02-10_11-33-37' \
 '/autofs/space/crater_001/projects/micropath/results/augmented/kegg_cell_cycle/2025-02-10_14-52-14' \
 '/autofs/space/crater_001/projects/micropath/results/augmented/hallmark_epithelial_mesenchymal_transition/2025-02-10_13-58-25' \
 '/autofs/space/crater_001/projects/micropath/results/augmented/hallmark_fatty_acid_metabolism/2025-02-10_14-21-51' \
 '/autofs/space/crater_001/projects/micropath/results/augmented/hallmark_glycolysis/2025-02-10_14-30-02' \
 '/autofs/space/crater_001/projects/micropath/results/augmented/immunosuppression/2025-02-10_14-40-02' \
 '/autofs/space/crater_001/projects/micropath/results/augmented/hallmark_oxidative_phosphorylation/2025-02-10_14-34-38' \
 '/autofs/space/crater_001/projects/micropath/results/augmented/gobp_t_cell_mediated_cytotoxicity/2025-02-10_12-57-08')



for checkpoint_dir in "${CHECKPOINT_DIRS[@]}"
do
    echo "Checkpoint: $checkpoint_dir"

    python code/models/clam/test_model_fp.py \
    --gpu_id 0 \
    --checkpoint_dir $checkpoint_dir \
    --dataset 'TCGA-BRCA' \
    --base_data_path '/autofs/space/crater_001/datasets/public/TCGA-BRCA' \
    --experimental_strategy 'All' \
    --features_h5_dir '/autofs/space/crater_001/projects/breast-cancer-pathology/results/CONCH/TCGA-BRCA/mmxbrcp/DiagnosticSlide/SegmentationHistoQC/features' '/autofs/space/crater_001/projects/breast-cancer-pathology/results/CONCH/TCGA-BRCA/mmxbrcp/TissueSlide/SegmentationHistoQC/features'
done

echo 'Finished AM-SB (Regression) Training on TCGA-BRCA Database.'