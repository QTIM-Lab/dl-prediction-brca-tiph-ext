# Imports
import pandas as pd



# Paths
splits = ["train", "val", "test"]
for split in splits:
    paths = [
        f'results/hallmark_angiogenesis/{split}_inference_info_kf0.csv',
        f'results/kegg_antigen_processing_and_presentation/{split}_inference_info_kf0.csv',
        f'results/gobp_b_cell_proliferation/{split}_inference_info_kf0.csv',
        f'results/kegg_cell_cycle/{split}_inference_info_kf0.csv',
        f'results/hallmark_epithelial_mesenchymal_transition/{split}_inference_info_kf0.csv',
        f'results/hallmark_fatty_acid_metabolism/{split}_inference_info_kf0.csv',
        f'results/hallmark_glycolysis/{split}_inference_info_kf0.csv',
        f'results/immunosuppression/{split}_inference_info_kf0.csv',
        f'results/hallmark_oxidative_phosphorylation/{split}_inference_info_kf0.csv',
        f'results/gobp_t_cell_mediated_cytotoxicity/{split}_inference_info_kf0.csv'
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