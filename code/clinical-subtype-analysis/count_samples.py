# Imports
import os
import pandas as pd
import numpy as np



# Paths
paths = [
    'results/hallmark_angiogenesis/test_inference_info_kf0.csv',
    'results/kegg_antigen_processing_and_presentation/test_inference_info_kf0.csv',
    'results/gobp_b_cell_proliferation/test_inference_info_kf0.csv',
    'results/kegg_cell_cycle/test_inference_info_kf0.csv',
    'results/hallmark_epithelial_mesenchymal_transition/test_inference_info_kf0.csv',
    'results/hallmark_fatty_acid_metabolism/test_inference_info_kf0.csv',
    'results/hallmark_glycolysis/test_inference_info_kf0.csv',
    'results/immunosuppression/test_inference_info_kf0.csv',
    'results/hallmark_oxidative_phosphorylation/test_inference_info_kf0.csv',
    'results/gobp_t_cell_mediated_cytotoxicity/test_inference_info_kf0.csv'
]


# Go through paths
for path_ in paths:

    # Open file
    inf_info = pd.read_csv(path_)
    print('HER2-/HR-', len(inf_info.copy()[inf_info['c_subtype'] == 'HER2-/HR-']))
    print('HER2-/HR+', len(inf_info.copy()[inf_info['c_subtype'] == 'HER2-/HR+']))
    print('HER2+/HR-', len(inf_info.copy()[inf_info['c_subtype'] == 'HER2+/HR-']))
    print('HER2+/HR+', len(inf_info.copy()[inf_info['c_subtype'] == 'HER2+/HR+']))
    
    # c_subtype = inf_info.copy()['c_subtype']
    # print(c_subtype)
    # print(c_subtype[c_subtype['c_subtype'] == 'HER2-/HR+'])

    # print(c_subtype.unique())
    # ['HER2-/HR+']
    # ['HER2-/HR-'] 
    # ['HER2+/HR-'] 
    # ['HER2+/HR+']