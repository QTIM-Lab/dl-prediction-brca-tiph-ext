#!/bin/bash



# Environment Variables
export PYENV_VERSION=mmxbrcp
export LD_LIBRARY_PATH="/autofs/space/crater_001/tools/usr/lib64:${LD_LIBRARY_PATH}"
export CUDA_VISIBLE_DEVICES=MIG-be134b7f-cab5-570b-8041-0593378dbb63



echo 'Started Inference Info about Clinical Subtype TCGA-BRCA Database.'

# List of checkpoint directories for AM_SB (CLAM/ResNet50 Features)
# CHECKPOINT_DIRS=('/autofs/space/crater_001/projects/breast-cancer-pathology/results/CLAM/TCGA-BRCA/mmxbrcp/All/checkpoints/hallmark_angiogenesis/2024-11-02_13-45-43'\
#  '/autofs/space/crater_001/projects/breast-cancer-pathology/results/CLAM/TCGA-BRCA/mmxbrcp/All/checkpoints/hallmark_epithelial_mesenchymal_transition/2024-11-02_15-07-48'\
#  '/autofs/space/crater_001/projects/breast-cancer-pathology/results/CLAM/TCGA-BRCA/mmxbrcp/All/checkpoints/hallmark_fatty_acid_metabolism/2024-11-02_16-26-16'\
#  '/autofs/space/crater_001/projects/breast-cancer-pathology/results/CLAM/TCGA-BRCA/mmxbrcp/All/checkpoints/hallmark_oxidative_phosphorylation/2024-11-02_17-46-30'\
#  '/autofs/space/crater_001/projects/breast-cancer-pathology/results/CLAM/TCGA-BRCA/mmxbrcp/All/checkpoints/hallmark_glycolysis/2024-11-02_18-59-12'\
#  '/autofs/space/crater_001/projects/breast-cancer-pathology/results/CLAM/TCGA-BRCA/mmxbrcp/All/checkpoints/kegg_antigen_processing_and_presentation/2024-11-02_20-13-27'\
#  '/autofs/space/crater_001/projects/breast-cancer-pathology/results/CLAM/TCGA-BRCA/mmxbrcp/All/checkpoints/gobp_t_cell_mediated_cytotoxicity/2024-11-02_21-28-36'\
#  '/autofs/space/crater_001/projects/breast-cancer-pathology/results/CLAM/TCGA-BRCA/mmxbrcp/All/checkpoints/gobp_b_cell_proliferation/2024-11-02_22-42-27'\ 
#  '/autofs/space/crater_001/projects/breast-cancer-pathology/results/CLAM/TCGA-BRCA/mmxbrcp/All/checkpoints/kegg_cell_cycle/2024-11-03_00-01-46'\
#  '/autofs/space/crater_001/projects/breast-cancer-pathology/results/CLAM/TCGA-BRCA/mmxbrcp/All/checkpoints/immunosuppression/2024-11-03_01-16-59')

# Dago's Augmented Feature Based Models
CHECKPOINT_DIRS=('/autofs/space/crater_001/projects/micropath/results/augmented/gobp_b_cell_proliferation/2025-02-10_11-33-37' \
 '/autofs/space/crater_001/projects/micropath/results/augmented/gobp_t_cell_mediated_cytotoxicity/2025-02-10_12-57-08' \
 '/autofs/space/crater_001/projects/micropath/results/augmented/hallmark_angiogenesis/2025-02-10_13-23-05' \
 '/autofs/space/crater_001/projects/micropath/results/augmented/hallmark_epithelial_mesenchymal_transition/2025-02-10_13-58-25' \
 '/autofs/space/crater_001/projects/micropath/results/augmented/hallmark_fatty_acid_metabolism/2025-02-10_14-21-51' \
 '/autofs/space/crater_001/projects/micropath/results/augmented/hallmark_glycolysis/2025-02-10_14-30-02' \
 '/autofs/space/crater_001/projects/micropath/results/augmented/hallmark_oxidative_phosphorylation/2025-02-10_14-34-38' \
 '/autofs/space/crater_001/projects/micropath/results/augmented/immunosuppression/2025-02-10_14-40-02' \
 '/autofs/space/crater_001/projects/micropath/results/augmented/kegg_antigen_processing_and_presentation/2025-02-10_14-48-01' \
 '/autofs/space/crater_001/projects/micropath/results/augmented/kegg_cell_cycle/2025-02-10_14-52-14')

 for checkpoint_dir in "${CHECKPOINT_DIRS[@]}"
do
    echo "Checkpoint: $checkpoint_dir"

    python code/clinical-subtype-analysis/run_clinical_subtype_analysis.py \
    --gpu_id 0 \
    --checkpoint_dir $checkpoint_dir \
    --dataset 'TCGA-BRCA' \
    --base_data_path '/autofs/space/crater_001/datasets/public/TCGA-BRCA' \
    --experimental_strategy 'All' \
    --features_h5_dir '/autofs/space/crater_001/projects/breast-cancer-pathology/results/CONCH/TCGA-BRCA/mmxbrcp/DiagnosticSlide/SegmentationHistoQC/features' '/autofs/space/crater_001/projects/breast-cancer-pathology/results/CONCH/TCGA-BRCA/mmxbrcp/TissueSlide/SegmentationHistoQC/features'
done

echo 'Finished Inference Info about Clinical Subtype TCGA-BRCA Database.'