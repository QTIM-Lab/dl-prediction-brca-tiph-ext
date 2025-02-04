# Imports
from __future__ import print_function
import os
import argparse
import numpy as np
import pandas as pd
import random
import copy
import shutil
import json
import h5py
from scipy.stats import percentileofscore
import string
from PIL import Image

# PyTorch Imports
import torch
from torch.utils.data import DataLoader
from torchinfo import summary

# Project Imports
from data_utilities import TCGABRCA_MIL_Dataset
from model_utilities import AM_SB, AM_MB
from wsi_class import WholeSlideImage



# Function: See the seed for reproducibility purposes
def set_seed(seed=42):

    # Random Seed
    random.seed(seed)

    # Environment Variable Seed
    os.environ['PYTHONHASHSEED'] = str(seed)

    # NumPy Seed
    np.random.seed(seed)

    # PyTorch Seed(s)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    return



# Function: Load dataloader and model
def load_datataloader_and_model(dataset, config_json, device, checkpoint_dir, fold):

    # Load the parameters from the configuration JSON
    n_classes = config_json["data"]["n_classes"]
    dropout = config_json["hyperparameters"]["dropout"]
    dropout_prob = config_json["hyperparameters"]["dropout_prob"]
    model_size = config_json["hyperparameters"]["model_size"]
    model_type = config_json["hyperparameters"]["model_type"]
    verbose = config_json["verbose"]
    num_workers = config_json["data"]["num_workers"]
    pin_memory = config_json["data"]["pin_memory"]
    encoding_size = config_json['data']['encoding_size']
    features_ = config_json["features"] if 'features' in config_json.keys() else "clam"


    # Dictionary with model settings for the initialization of the model object
    model_dict = {
        "dropout":dropout,
        "dropout_prob":dropout_prob,
        'n_classes':n_classes,
        "encoding_size":encoding_size
    }
    
    assert model_size is not None, "Please specifiy a model size."

    model_dict.update({"size_arg": model_size})
    
    # Select model
    if model_type == 'am_sb':
        model = AM_SB(**model_dict)
    elif model_type == 'am_mb':
        model = AM_MB(**model_dict)


    # Move into model into device
    model.to(device=device)

    if verbose:
        print(f"Using features: {features_}")
        summary(model)


    # Create DataLoaders 
    # FIXME: The code is currently optimized for batch_size == 1)
    dataloader = DataLoader(
        dataset=dataset,
        batch_size=1,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory
    )

    # Load model checkpoint
    model.load_state_dict(torch.load(os.path.join(checkpoint_dir, f"best_model_kf{fold}.pt"), map_location=device))
    

    # Put model into evaluation 
    model.eval()

    return dataloader, model



# Function: Perform inference for single slide
def infer_single_slide(model, features, device, label, model_type, verbose=True):

    assert model_type in ('am_sb', 'am_mb')

    # Move features into device
    features = features.to(device)


    # Perform inference
    with torch.no_grad():

        # Get ouput dictionary
        ouput_dict = model(torch.unsqueeze(features, 0))
        y_proba, y_pred, A = ouput_dict["y_proba"], ouput_dict["y_pred"], ouput_dict["A_raw"]
        y_pred = y_pred.item()

        # Check if MB-based model
        if model_type == 'am_mb':
            # print("A.shape ", A.shape)
            A = A[0, y_pred]
            A = torch.unsqueeze(A, 0)

        # Reshape Attention Matrix
        # A = A.view(-1, 1).cpu().numpy()
        A = torch.reshape(A, (-1, 1)).cpu().numpy()

        if verbose:
            print('y_pred: {}, y_gt: {}, y_prob: {}'.format(y_pred, label, ["{:.4f}".format(p) for p in y_proba.cpu().flatten()]))

    return y_pred, y_proba.cpu().flatten().numpy(), label, A



# Function: Convert score to percentile
def score2percentile(score, ref):
    percentile = percentileofscore(ref, score)
    return percentile



# Function: Draw heatmaps
def drawHeatmap(scores, coords, slide_path=None, wsi_object=None, vis_level = -1, **kwargs):
    if wsi_object is None:
        wsi_object = WholeSlideImage(slide_path)
        print(wsi_object.name)
    
    wsi = wsi_object.getOpenSlide()
    if vis_level < 0:
        vis_level = wsi.get_best_level_for_downsample(32)
    
    heatmap = wsi_object.visHeatmap(scores=scores, coords=coords, vis_level=vis_level, **kwargs)
    return heatmap



# Function: Draw heatmaps patches
def drawHeatmapPatch(scores, indices, coords, slide_path=None, wsi_object=None, vis_level=-1, **kwargs):
    if wsi_object is None:
        wsi_object = WholeSlideImage(slide_path)
        print(wsi_object.name)
    
    wsi = wsi_object.getOpenSlide()
    if vis_level < 0:
        vis_level = wsi.get_best_level_for_downsample(32)
    
    heatmap = wsi_object.visHeatmapPatch(scores=scores, indices=indices, coords=coords, vis_level=vis_level, **kwargs)
    return heatmap



# Function: Calculate heatmaps from patches
def compute_from_patches(model_type, y_pred, model, features, coords, ref_scores, device):
    
    assert model_type in ('am_sb', 'am_mb')

    # Move features into device
    features = features.to(device)
    
    # Perform inference
    with torch.no_grad():
        
        # Get output dictionary
        ouput_dict = model(torch.unsqueeze(features, 0))
        A = ouput_dict["A_raw"]


       # Check if MB-based model
        if model_type == 'tinyclam_mb':
            # print("A.shape ", A.shape)
            A = A[y_pred]
            A = torch.unsqueeze(A, 0)

        # Reshape A for further processing
        # A = A.view(-1, 1).cpu().numpy()
        A = torch.reshape(A, (-1, 1)).cpu().numpy()
        
        if ref_scores is not None:
            for score_idx in range(len(A)):
                A[score_idx] = score2percentile(A[score_idx], ref_scores)


    # Build asset dictionary
    asset_dict = {
        'attention_scores':A,
        'coords':coords,
        'features':features.cpu().numpy()
    }

    return asset_dict



# Function: Generate a random string with a give length (to generate the directories and folders)
def generate_random_string(length=8):

    random_string = ''.join(random.choices(string.ascii_letters + string.digits, k=length))

    return random_string



# Dictionary to map names of the tasks
names_dict = {
    'gobp_b_cell_proliferation':'b_cell_proliferation',
    'gobp_t_cell_mediated_cytotoxicity':'t_cell_mediated_cytotoxicity',
    'hallmark_angiogenesis':'angiogenesis',
    'hallmark_epithelial_mesenchymal_transition':'epithelial_mesenchymal_transition',
    'hallmark_fatty_acid_metabolism':'fatty_acid_metabolism',
    'hallmark_glycolysis':'glycolysis',
    'hallmark_oxidative_phosphorylation':'oxidative_phosphorylation',
    'immunosuppression':'immunosuppression',
    'kegg_antigen_processing_and_presentation':'antigen_processing_and_presentation',
    'kegg_cell_cycle':'cell_cycle'
}



# Dictionary to map labels
labels_dict = {
    0:'neg',
    1:'pos'
}



if __name__ == '__main__':

    # CLI
    parser = argparse.ArgumentParser(description='CLAM: Heatmap Generation (for Clinical Study).')
    parser.add_argument('--clinicians_dir', type=str, required=True, help="The directory where we will save these studies for the clinicians.")
    parser.add_argument('--researchers_dir', type=str, required=True, help="The directory where we will save these studies for the researchers.")
    parser.add_argument('--gpu_id', type=int, default=0, help="The ID of the GPU we will use to run the program.")
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility (default: 42).')
    parser.add_argument('--checkpoint_dir', type=str, required=True, help='The path to the checkpoint directory.')
    parser.add_argument('--dataset', type=str, required=True, choices=['TCGA-BRCA'], help='The dataset for the experiments.')
    parser.add_argument('--base_data_path', type=str, required=True, help='Base data path for TCGA-BRCA dataset.')
    parser.add_argument('--experimental_strategy', type=str, choices=['All', 'DiagnosticSlide', 'TissueSlide'], required=True, help="The experimental strategy for the TCGA-BRCA dataset.")
    parser.add_argument('--features_h5_dir', nargs='+', type=str, required=True, help="The directory of the features in .pt format for the TCGA-BRCA dataset.")
    parser.add_argument('--patches_from', type=str, required=True, choices=["CLAM"], help="The framework used to pre-process the WSIs.")
    parser.add_argument('--generate_heatmaps_for', type=str, default='test', choices=["train", "validation", "test"], help="The data split to obtain heatmaps.")
    parser.add_argument('--heatmap_config_file', type=str, required=True, help="The configuration JSON file for the heatmap generation.")
    parser.add_argument('--use_histoqc_quality_file', type=str, default=None, help="Use the quality file generated by the HistoQC framework.")
    parser.add_argument('--use_histoqc_seg_masks', action="store_true", help="Use the segmentation masks generated by the HistoQC framework.")
    parser.add_argument('--verbose', action="store_true", help="Print execution information.")
    args = parser.parse_args()

    # Set seeds (for reproducibility reasons)
    set_seed(seed=args.seed)



    # Get SSGSEA task from checkpoint directory
    task_ = args.checkpoint_dir.split('/')[-2]
    task = names_dict[task_]

    # Create the directories where we will save the results for the clinicians
    clinicians_dir = os.path.join(args.clinicians_dir, task)
    os.makedirs(clinicians_dir, exist_ok=True)

    # Create the directories where we will save the results for the researchers
    researchers_dir = os.path.join(args.researchers_dir, task)
    os.makedirs(researchers_dir, exist_ok=True)


    # Load configuration JSON
    with open(os.path.join(args.checkpoint_dir, "config.json"), 'r') as j:
        config_json = json.load(j)


    # Load GPU/CPU device
    if args.gpu_id == -1:
        device = torch.device('cpu')
    else:
        device = torch.device(f'cuda:{args.gpu_id}') if torch.cuda.is_available() else torch.device('cpu')


    # Get the encoding size for the feature vectors
    encoding_size = config_json['data']['encoding_size']

    # Get verbose
    verbose = config_json['verbose']


    # Load data
    print('Loading dataset...')
    if args.dataset == 'TCGA-BRCA':
        dataset = TCGABRCA_MIL_Dataset(
            base_data_path=args.base_data_path,
            experimental_strategy=args.experimental_strategy,
            label=args.checkpoint_dir.split('/')[-2],
            features_h5_dir=args.features_h5_dir,
            n_folds=int(config_json["data"]["n_folds"]),
            seed=int(args.seed)
        )

        # Create the data splits from the original dataset
        train_set = copy.deepcopy(dataset)
        train_set.select_split(split='train')

        val_set = copy.deepcopy(dataset)
        val_set.select_split(split='validation')

        test_set = copy.deepcopy(dataset)
        test_set.select_split(split='test')


    # Iterate through folds
    n_folds = int(config_json["data"]["n_folds"])
    for fold in range(n_folds):

        # Set seed
        set_seed(seed=args.seed)

        if verbose:
            print(f"Current Fold {fold+1}/{n_folds}")
        

        # Select folds in the database
        train_set.select_fold(fold=fold)
        val_set.select_fold(fold=fold)
        test_set.select_fold(fold=fold)

        # Select dataset
        if args.generate_heatmaps_for == "train":
            dataset = train_set
        elif args.generate_heatmaps_for == "validation":
            dataset = val_set
        else:
            dataset = test_set


        # Load dataloader and model
        dataloder, model = load_datataloader_and_model(
            dataset=dataset,
            config_json=config_json,
            device=device,
            checkpoint_dir=args.checkpoint_dir,
            fold=fold
        )

        # Generate a CSV for the data split and folds
        split_fold_csv = f"{args.generate_heatmaps_for}_set_kf{fold}.csv"

        csv_data_dict = {
            'case_id':list(),
            'svs_path':list(),
            'features_h5':list(),
            'ssgea_id':list(),
            'ssgsea_scores':list()
        }

        # Go through the dataloader to generate a CSV with certain information
        with torch.no_grad():
            for _, input_data_dict in enumerate(dataloder):
                
                # Get data
                case_id = input_data_dict['case_id'][0]
                svs_path = input_data_dict['svs_path'][0]
                features_h5 = input_data_dict['features_h5'][0]
                ssgea_id = input_data_dict['ssgea_id'][0]
                ssgsea_scores = input_data_dict['ssgsea_scores'].item()

                # Append this to the CSV data dict
                csv_data_dict['case_id'].append(case_id)
                csv_data_dict['svs_path'].append(svs_path)
                csv_data_dict['features_h5'].append(features_h5)
                csv_data_dict['ssgea_id'].append(ssgea_id)
                csv_data_dict['ssgsea_scores'].append(ssgsea_scores)


        # Create CSV
        csv_data_df = pd.DataFrame.from_dict(csv_data_dict)
        csv_data_df.to_csv(os.path.join(args.checkpoint_dir, split_fold_csv))


        # HistoQC: Quality Assessment of WSI
        hqc_q = None
        if args.use_histoqc_quality_file:

            # Load and read the HistQC quality files
            hqc_q = pd.read_csv(args.use_histoqc_quality_file)

            # Process the dataframe to get the good quality cases
            hqc_q = hqc_q[hqc_q['is_good_quality'] == True]
            hqc_q = hqc_q[['wsi_folder_path', 'is_good_quality']]
            hqc_q = hqc_q['wsi_folder_path']
            hqc_q = list(hqc_q.values)

            # Get the WSI IDs
            hqc_slide_ids = [s.split('/')[-1] for s in hqc_q]

            # Get the dataset WSI IDs
            slides = csv_data_df['svs_path'].values
            slide_ids = [s.split('/')[-1] for s in slides]

            # Process the subset of WSI IDs after the results of HistoQC
            slide_ids_ = list()
            for s_id in slide_ids:
                if s_id in hqc_slide_ids:
                    slide_ids_.append(s_id)


            # Build a new list of slides
            slides_ = list()
            for s in slides:
                s_id = s.split('/')[-1]
                if s_id in slide_ids_:
                    slides_.append(s)
            slides = slides_.copy()


        # HistoQC: Use HistoQC Segmentation Masks
        histo_qc_map_masks = None
        if args.use_histoqc_seg_masks:
            assert hqc_q is not None, "Please provide the <use_histoqc_quality_file> as a parameter."

            # Build a dictionary that maps the path of the WSI to its HistoQC Segmentation Mask
            histo_qc_map_masks = dict()
            for wsi_path in slides:
                for wsi_m_path in hqc_q:
                    if wsi_path.split('/')[-1] == wsi_m_path.split('/')[-1]:
                        histo_qc_map_masks[wsi_path] = wsi_m_path
        # print(histo_qc_map_masks)


        # Copy configuration JSON to the experiment directory
        _ = shutil.copyfile(
            src=args.heatmap_config_file,
            dst=os.path.join(args.checkpoint_dir, 'heatmap_config_file.json')
        )

        # Open heatmap configuration JSON file
        with open(args.heatmap_config_file, 'r') as j:
            heatmap_config_file = json.load(j)
        
        # Get parameters from the heatmap configuration JSON file
        data_args = heatmap_config_file['data']
        heatmap_args = heatmap_config_file['heatmap_arguments']


        # Get final patch size and step sizes from JSON configuration file
        patch_size = tuple([data_args['patch_size'] for _ in range(2)])
        step_size = tuple([int(data_args['step_size'] * (1-data_args['overlap'])) for _ in range(2)])
        if verbose:
            print('Patch_size: {} x {}, with {:.2f} overlap, step size is {} x {}'.format(patch_size[0], patch_size[1], data_args['overlap'], step_size[0], step_size[1]))
        
        # Get more arguments for the WSI
        blocky_wsi_kwargs = {
            'top_left': None,
            'bot_right': None,
            'patch_size': patch_size,
            'step_size': patch_size, 
            'custom_downsample':data_args['custom_downsample'],
            'level':data_args['patch_level'],
            'use_center_shift':heatmap_args['use_center_shift']
        }


        # TODO: Go through provided dataset
        for row_idx, row in csv_data_df.iterrows():

            # Get slide filename
            slide_name = row['svs_path'].split('/')[-1]
            if verbose:
                print('Processing: ', slide_name)	

            # Get ssgsea scores
            ssgsea_scores = row['ssgsea_scores']
            if verbose:
                print('Label: ', ssgsea_scores)

            # Get slide id
            slide_id = slide_name.replace('.svs', '')
            if verbose:
                print("Slide ID: ", slide_id)

            
            # Initialise ROI
            top_left = None
            bot_right = None
            if verbose:
                print('top left: ', top_left, ' bot right: ', bot_right)


            # Get slide path
            slide_path = row['svs_path']
            if verbose:
                print("Slide path: ", slide_path)


            # FIXME: For now, this script only supports HistoQC masks
            if histo_qc_map_masks:
                mask_dir = histo_qc_map_masks[slide_path]
                mask_id = mask_dir.split('/')[-1]
                mask_fname = f"{mask_id}_mask_use.png"
                mask_file = os.path.join(mask_dir, mask_fname)
                mask_file = mask_file.replace('/autofs/cluster/qtim/projects/', '/autofs/space/crater_001/projects/')
            else:
                pass
            

            if verbose:
                print('Initializing WSI object')
            wsi_object = WholeSlideImage(path=slide_path)
            
            # FIXME: For now, this script only supports HistoQC masks
            if histo_qc_map_masks:
                wsi_object.initHistoQCSegmentation(mask_file)
            
            # Get the WSI reference downsample
            wsi_ref_downsample = wsi_object.level_downsamples[data_args['patch_level']]

            # The actual patch size for heatmap visualization should be the patch size * downsample factor * custom downsample factor
            vis_patch_size = tuple((np.array(patch_size) * np.array(wsi_ref_downsample) * data_args['custom_downsample']).astype(int))

            # Get the path to the features
            features_path = row['features_h5']
            # print(features_path)

            # Get the path to the patches coordinates
            base_path_idx = features_path.find('features')
            base_path = features_path[0:base_path_idx]
            # print("base_path ", base_path)
            base_path_split = base_path.split('/')
            next_ = False
            for s_idx, s_str in enumerate(base_path_split):
                if next_:
                    base_path_split[s_idx] = args.patches_from
                    break
                else:
                    if s_str == "results":
                        next_ = True
            base_path = "/".join(base_path_split)
            # print("base_path ", base_path)

            patches_path = os.path.join(base_path, 'patches', slide_id+'.h5')
            # print(patches_path)


            # Load features
            with h5py.File(features_path, "r") as f:
                features = f["features"][()]
            features = torch.from_numpy(features)

            # Load coordinates
            with h5py.File(patches_path, "r") as p:
                coords = p['coords'][:]

            
            # Get predictions, predictions probabilities and attention scores
            y_pred, y_probas, label, attention_scores = infer_single_slide(
                model=model, 
                features=features,
                device=device,
                label=row['ssgsea_scores'], 
                model_type=config_json["hyperparameters"]["model_type"],
                verbose=verbose
            )


            # Save only for the correctly predicted cases
            if y_pred == label:

                # Select slide results directory for clinicians and researchers
                slide_save_dir_clinicians = os.path.join(clinicians_dir,  labels_dict[label], slide_id)
                slide_save_dir_researchers = os.path.join(researchers_dir, labels_dict[label], slide_id)
                os.makedirs(slide_save_dir_clinicians, exist_ok=True)
                os.makedirs(slide_save_dir_researchers, exist_ok=True)

                # Compute high-, low- and random attention scores
                attention_scores = attention_scores.flatten()
                s_indices = np.argsort(attention_scores)
                # print(coords)
                # print(coords.shape)


                # High-attention and indices
                h_indices = s_indices[-5:]
                h_attention_scores = attention_scores[h_indices]
                h_coords = coords[h_indices]
                # print(h_attention_scores.shape)
                # print(h_indices)
                # print(h_coords)

                # Low-attention and indices
                l_indices = s_indices[0:5]
                l_attention_scores = attention_scores[l_indices]
                l_coords = coords[l_indices]
                # print(l_attention_scores.shape)
                # print(l_indices)
                # print(l_coords)

                # Random attention and indices
                indices = [i for i in range(attention_scores.shape[0])]
                r_indices = np.random.choice(indices, 5, False)
                r_attention_scores = attention_scores[r_indices]
                r_coords = coords[r_indices]
                # print(r_coords)



                # Create a dictionary with all data we need to build the studies 
                study_data_dict = {
                    'high':{'attention_scores':h_attention_scores, 'indices':h_indices, 'directory':generate_random_string(), 'coords':h_coords},
                    'low':{'attention_scores':l_attention_scores, 'indices':l_indices, 'directory':generate_random_string(), 'coords':l_coords},
                    'random':{'attention_scores':r_attention_scores, 'indices':r_indices, 'directory':generate_random_string(), 'coords':r_coords}
                }
                # print(study_data_dict)

                for patch_set, patch_set_metadata in study_data_dict.items():
                    # print(patch_set)
                    # print(patch_set_metadata)

                    patch_set_save_dir = os.path.join(slide_save_dir_clinicians, patch_set_metadata['directory'])
                    os.makedirs(patch_set_save_dir, exist_ok=True)

                    # Compute/draw heatmap using the simplest parameters and save it
                    heatmap_patches = drawHeatmapPatch(
                        scores=patch_set_metadata["attention_scores"],
                        indices=patch_set_metadata['indices'],
                        coords=patch_set_metadata["coords"], 
                        slide_path=slide_path, 
                        wsi_object=wsi_object,
                        vis_level=0, 
                        patch_size=vis_patch_size, 
                        convert_to_percentiles=True
                    )
                    # print(len(heatmap_patches))
                    
                    for patch_idx, patch_arr in enumerate(heatmap_patches):
                        patch_pil = Image.fromarray(patch_arr).convert('RGB')
                        patch_pil.save(os.path.join(patch_set_save_dir, f'{patch_idx}.png'))
                    print(f"Saved {patch_set}-attention patches heatmap image at: {patch_set_save_dir}")
                    del heatmap_patches



                # Save original slide (if needed)
                if heatmap_args['save_orig']:

                    # Get visualization level
                    assert heatmap_args['vis_level'] >= 0
                    vis_level = heatmap_args['vis_level']

                    # Create a new save name for this slide
                    wsi_img_save_name = f'{slide_id}_original.png'

                    # Save the original slide
                    wsi_img = wsi_object.visWSI(
                        vis_level=vis_level,
                        view_slide_only=True,
                        custom_downsample=heatmap_args['custom_downsample']
                    )
                    wsi_img.save(os.path.join(slide_save_dir_clinicians, f'original_wsi.png'))
                    print("Saved original WSI image at: ", os.path.join(slide_save_dir_clinicians, f'original_wsi.png'))
                    del wsi_img

                

                # Create a helper CSV for clinicians annotation
                annotation_dict = dict()
                annotation_dict["folders"] = list()
                annotation_dict["annotations"] = list()

                # Create a helper CSV for researchers cross-data
                gt_dict = dict()
                gt_dict["folders"] = list()
                gt_dict["annotations"] = list()

                for patch_set, patch_set_metadata in study_data_dict.items():
                    annotation_dict["folders"].append(patch_set_metadata['directory'])
                    annotation_dict["annotations"].append("")
                    gt_dict["folders"].append(patch_set_metadata['directory'])
                    gt_dict["annotations"].append(patch_set)
                random.shuffle(annotation_dict["folders"])

                # Convert dictionaries into DataFrames
                annotation_df = pd.DataFrame.from_dict(annotation_dict)
                gt_df = pd.DataFrame.from_dict(gt_dict)
                
                # Save CSV files to the right directories
                annotation_df.to_csv(os.path.join(slide_save_dir_clinicians, "annotation_file.csv"))
                print("Saved annotation file to: ", os.path.join(slide_save_dir_clinicians, "annotation_file.csv"))
                
                gt_df.to_csv(os.path.join(slide_save_dir_researchers, "gt_file.csv"))
                print("Saved gt file to: ", os.path.join(slide_save_dir_researchers, "gt_file.csv"))
