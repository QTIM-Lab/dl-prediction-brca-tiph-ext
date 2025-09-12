# Imports
import os
import pandas as pd



# List of paths to assess
# angiogenesis, epithelial-mesenchymal transition, cell cycling, immunosuppression, t-cell mediated cytotoxicity
# paths_to_assess = [
#     '/autofs/space/crater_001/projects/breast-cancer-pathology/results/CLAM/TCGA-BRCA/mmxbrcp/All/checkpoints/gobp_t_cell_mediated_cytotoxicity/2024-04-25_21-05-55/heatmaps',
#     '/autofs/space/crater_001/projects//breast-cancer-pathology/results/CLAM/TCGA-BRCA/mmxbrcp/All/checkpoints/hallmark_angiogenesis/2024-04-25_09-13-05/heatmaps',
#     '/autofs/space/crater_001/projects//breast-cancer-pathology/results/CLAM/TCGA-BRCA/mmxbrcp/All/checkpoints/hallmark_epithelial_mesenchymal_transition/2024-04-25_11-05-38/heatmaps',
#     '/autofs/space/crater_001/projects//breast-cancer-pathology/results/CLAM/TCGA-BRCA/mmxbrcp/All/checkpoints/kegg_cell_cycle/2024-04-26_01-28-02/heatmaps',
#     '/autofs/space/crater_001/projects//breast-cancer-pathology/results/CLAM/TCGA-BRCA/mmxbrcp/All/checkpoints/immunosuppression/2024-04-29_03-31-36/heatmaps'
# ]



POS = True
paths_to_assess_pos = [
    '/autofs/space/crater_001/projects/breast-cancer-pathology/analysis/heatmaps/pos/angiogenesis',
    '/autofs/space/crater_001/projects/breast-cancer-pathology/analysis/heatmaps/pos/cell_cycle',
    '/autofs/space/crater_001/projects/breast-cancer-pathology/analysis/heatmaps/pos/epithelial_mesenchymal_transition',
    '/autofs/space/crater_001/projects/breast-cancer-pathology/analysis/heatmaps/pos/immunosuppression',
    '/autofs/space/crater_001/projects/breast-cancer-pathology/analysis/heatmaps/pos/t_cell_mediated_cytotoxicity'
]
paths_to_assess_neg = [
    '/autofs/space/crater_001/projects/breast-cancer-pathology/analysis/heatmaps/neg/angiogenesis',
    '/autofs/space/crater_001/projects/breast-cancer-pathology/analysis/heatmaps/neg/cell_cycle',
    '/autofs/space/crater_001/projects/breast-cancer-pathology/analysis/heatmaps/neg/epithelial_mesenchymal_transition',
    '/autofs/space/crater_001/projects/breast-cancer-pathology/analysis/heatmaps/neg/immunosuppression',
    '/autofs/space/crater_001/projects/breast-cancer-pathology/analysis/heatmaps/neg/t_cell_mediated_cytotoxicity'
]
if POS:
    paths_to_assess = paths_to_assess_pos.copy()
else:
    paths_to_assess = paths_to_assess_neg.copy()




# Get the split
case_counts = dict()

# Iterate through the data in the dataframes
for path_ in paths_to_assess:

    # Get values
    cases = [c for c in os.listdir(path_) if not c.startswith('.')]
    # print(cases)

    # Count these values
    for case_ in cases:
        case__contents = [c for c in os.listdir(os.path.join(path_, case_)) if not c.startswith('.')]
        if len(case__contents) >= 1:
            # print(case_)
            if case_ not in case_counts.keys():
                case_counts[case_] = 1
            else:
                case_counts[case_] += 1

# Get image names that are common to the n CSVs that we loaded
image_counts_inv = dict()
for img_name, img_count in case_counts.items():
    # print(img_name, img_count)
    if img_count not in image_counts_inv.keys():
        image_counts_inv[img_count] = list()
        image_counts_inv[img_count].append(img_name)
    else:
        image_counts_inv[img_count].append(img_name)

# print(image_counts_inv)
img_count_values = [k for k in image_counts_inv.keys()]
# print(img_count_values)

for cnt in img_count_values:
    cnt_dict = {
        'wsi':list(),
        'has_angiogenesis':list(),
        'angiogenesis_path':list(),
        'has_cell_cycle':list(),
        'cell_cycle_path':list(),
        'has_epithelial_mesenchymal_transition':list(),
        'epithelial_mesenchymal_transition_path':list(),
        'has_immunosuppression':list(),
        'immunosuppression_path':list(),
        'has_t_cell_mediated_cytotoxicity':list(),
        't_cell_mediated_cytotoxicity_path':list(),
    }
    for i_cnt, i_name in image_counts_inv.items():
        if i_cnt == cnt:
            # print(i_name)
            for fpath in i_name:
                cnt_dict['wsi'].append(fpath)

                # Check where we have these heatmaps
                # Angiogenesis
                if os.path.exists(os.path.join(paths_to_assess[0], fpath)):
                    a = [c for c in os.listdir(os.path.join(paths_to_assess[0], fpath)) if not c.startswith('.')]
                else:
                    a = list()
                if len(a) >= 1:
                    cnt_dict['has_angiogenesis'].append(1)
                    cnt_dict['angiogenesis_path'].append(os.path.join(paths_to_assess[0], fpath))
                else:
                    cnt_dict['has_angiogenesis'].append(0)
                    cnt_dict['angiogenesis_path'].append('')
                
                # Cell Cyle
                if os.path.exists(os.path.join(paths_to_assess[1], fpath)):
                    cc = [c for c in os.listdir(os.path.join(paths_to_assess[1], fpath)) if not c.startswith('.')]
                else:
                    cc = list()
                if len(cc) >= 1:
                    cnt_dict['has_cell_cycle'].append(1)
                    cnt_dict['cell_cycle_path'].append(os.path.join(paths_to_assess[1], fpath))
                else:
                    cnt_dict['has_cell_cycle'].append(0)
                    cnt_dict['cell_cycle_path'].append('')
                
                # Epithelial Mesenchymal Transition
                if os.path.exists(os.path.join(paths_to_assess[2], fpath)):
                    emt = [c for c in os.listdir(os.path.join(paths_to_assess[2], fpath)) if not c.startswith('.')]
                else:
                    emt = list()
                if len(emt) >= 1:
                    cnt_dict['has_epithelial_mesenchymal_transition'].append(1)
                    cnt_dict['epithelial_mesenchymal_transition_path'].append(os.path.join(paths_to_assess[2], fpath))
                else:
                    cnt_dict['has_epithelial_mesenchymal_transition'].append(0)
                    cnt_dict['epithelial_mesenchymal_transition_path'].append('')
                
                # Immunosuppression
                if os.path.exists(os.path.join(paths_to_assess[3], fpath)):
                    imm = [c for c in os.listdir(os.path.join(paths_to_assess[3], fpath)) if not c.startswith('.')]
                else:
                    imm = list()
                if len(imm) >= 1:
                    cnt_dict['has_immunosuppression'].append(1)
                    cnt_dict['immunosuppression_path'].append(os.path.join(paths_to_assess[3], fpath))
                else:
                    cnt_dict['has_immunosuppression'].append(0)
                    cnt_dict['immunosuppression_path'].append('')
                
                # T Cell Mediated Cytotoxicity
                if os.path.exists(os.path.join(paths_to_assess[4], fpath)):
                    tcm = [c for c in os.listdir(os.path.join(paths_to_assess[4], fpath)) if not c.startswith('.')]
                else:
                    tcm = list()
                if len(tcm) >= 1:
                    cnt_dict['has_t_cell_mediated_cytotoxicity'].append(1)
                    cnt_dict['t_cell_mediated_cytotoxicity_path'].append(os.path.join(paths_to_assess[4], fpath))
                else:
                    cnt_dict['has_t_cell_mediated_cytotoxicity'].append(0)
                    cnt_dict['t_cell_mediated_cytotoxicity_path'].append('')



    cnt_dict_df = pd.DataFrame.from_dict(cnt_dict)
    # print(cnt_dict_df)
    os.makedirs('common_results', exist_ok=True)
    cnt_dict_df.to_csv(f"common_results/common_results_idx{cnt}.csv", index=False)