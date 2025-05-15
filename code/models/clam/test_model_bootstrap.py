# Imports
from __future__ import print_function
import os
import argparse
import pandas as pd
import numpy as np
import random
import json
import copy
import matplotlib.pyplot as plt
from tqdm import tqdm

# PyTorch Imports
import torch

# Project Imports
from train_val_test_utilities import test_pipeline
from data_utilities import TCGABRCA_MIL_Dataset, TCGABRCA_MIL_DatasetRegression, TCGABRCA_MIL_DatasetClinicalSubtype

# Function: Set the seed for reproducibility purposes
def set_seed(seed=42):
    """
    Set the random seed for reproducibility.

    Parameters
    ----------
    seed : int, optional
        Random seed value, by default 42

    Returns
    -------
    None
    """
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

# Function: Perform bootstrap analysis
def bootstrap_analysis(y_true, y_pred, metric_fn, metric_name="", bins=1000, confidence=0.95):
    """
    Perform bootstrap analysis to calculate confidence intervals for a given metric.

    Parameters
    ----------
    y_true : array-like
        Ground truth values
    y_pred : array-like
        Predicted values
    metric_fn : callable
        Function to calculate the metric
    metric_name : str, optional
        Name of the metric for progress display, by default ""
    bins : int, optional
        Number of bootstrap samples, by default 1000
    confidence : float, optional
        Confidence level for the interval, by default 0.95

    Returns
    -------
    dict
        Dictionary containing the metric value, lower bound, and upper bound
    """
    n = len(y_true)
    results = []

    # Use tqdm to show progress
    desc = f"Bootstrap analysis for {metric_name}" if metric_name else "Bootstrap analysis"
    for _ in tqdm(range(bins), desc=desc):
        # Generate bootstrap sample indices
        indices = np.random.choice(n, n, replace=True)

        # Calculate metric on bootstrap sample
        bootstrap_true = y_true[indices]
        bootstrap_pred = y_pred[indices]
        result = metric_fn(bootstrap_true, bootstrap_pred)
        results.append(result)

    # Calculate confidence interval
    alpha = (1 - confidence) / 2
    lower_bound = np.percentile(results, alpha * 100)
    upper_bound = np.percentile(results, (1 - alpha) * 100)

    return {
        'value': metric_fn(y_true, y_pred),
        'lower_bound': lower_bound,
        'upper_bound': upper_bound
    }

if __name__ == "__main__":

    # CLI
    parser = argparse.ArgumentParser(description='CLAM: Model Bootstrap Analysis.')
    parser.add_argument('--gpu_id', type=int, default=0, help="The ID of the GPU we will use to run the program.")
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility (default: 42).')
    parser.add_argument('--checkpoint_dir', type=str, required=True, help='The path to the checkpoint directory.')
    parser.add_argument('--dataset', type=str, required=True, choices=['TCGA-BRCA'], help='The dataset for the experiments.')
    parser.add_argument('--base_data_path', type=str, required=True, help='Base data path for TCGA-BRCA dataset.')
    parser.add_argument('--experimental_strategy', type=str, choices=['All', 'DiagnosticSlide', 'TissueSlide'], required=True, help="The experimental strategy for the TCGA-BRCA dataset.")
    parser.add_argument('--features_h5_dir', nargs='+', type=str, required=True, help="The directory of the features in .pt format for the TCGA-BRCA dataset.")
    parser.add_argument('--bins', type=int, default=1000, help='Number of bootstrap samples (default: 1000).')
    args = parser.parse_args()

    # Set seeds (for reproducibility reasons)
    set_seed(seed=args.seed)

    # Load configuration JSON
    with open(os.path.join(args.checkpoint_dir, "config.json"), 'r') as j:
        config_json = json.load(j)

    # Load GPU/CPU device
    device = torch.device(f'cuda:{args.gpu_id}') if torch.cuda.is_available() else torch.device('cpu')

    # Get the encoding size for the feature vectors
    encoding_size = config_json['data']['encoding_size']

    # Get verbose
    verbose = config_json['verbose']

    # Task type
    if "task_type" in config_json.keys():
        task_type = config_json["task_type"]
    else:
        task_type = "classification"
        config_json["task_type"] = "classification"

    # Load data
    print('Loading dataset...')
    if args.dataset == 'TCGA-BRCA':
        if task_type == "classification":
            dataset = TCGABRCA_MIL_Dataset(
                base_data_path=args.base_data_path,
                experimental_strategy=args.experimental_strategy,
                label=args.checkpoint_dir.split('/')[-2],
                features_h5_dir=args.features_h5_dir,
                n_folds=int(config_json["data"]["n_folds"]),
                seed=int(args.seed)
            )
        elif task_type == "clinical_subtype_classification":
            dataset = TCGABRCA_MIL_DatasetClinicalSubtype(
                base_data_path=args.base_data_path,
                experimental_strategy=args.experimental_strategy,
                features_h5_dir=args.features_h5_dir,
                n_folds=int(config_json["data"]["n_folds"]),
                seed=int(args.seed)
            )
        elif task_type == "regression":
            dataset = TCGABRCA_MIL_DatasetRegression(
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
        train_set.select_fold(fold=fold),
        val_set.select_fold(fold=fold),
        test_set.select_fold(fold=fold)

        # Test model
        test_metrics, test_y_c, test_y_pred_c, test_icc = test_pipeline(
            test_set=test_set,
            config_json=config_json,
            device=device,
            checkpoint_dir=args.checkpoint_dir,
            fold=fold
        )

        # Perform bootstrap analysis
        bootstrap_metrics = {}

        if task_type == "classification" or task_type == "clinical_subtype_classification":
            # Convert to numpy arrays
            test_y = np.array(test_set.get_labels())
            test_y_pred = np.array(test_y_pred_c)

            # Show progress bar for bootstrap analysis
            print("Performing bootstrap analysis for classification metrics...")

            # Define metric functions
            def calc_accuracy(y_true, y_pred):
                return np.mean(y_true == (y_pred > 0.5).astype(int))

            def calc_precision(y_true, y_pred):
                y_pred_binary = (y_pred > 0.5).astype(int)
                true_positives = np.sum((y_true == 1) & (y_pred_binary == 1))
                predicted_positives = np.sum(y_pred_binary == 1)
                return true_positives / predicted_positives if predicted_positives > 0 else 0

            def calc_recall(y_true, y_pred):
                y_pred_binary = (y_pred > 0.5).astype(int)
                true_positives = np.sum((y_true == 1) & (y_pred_binary == 1))
                actual_positives = np.sum(y_true == 1)
                return true_positives / actual_positives if actual_positives > 0 else 0

            # Calculate bootstrap metrics
            bootstrap_metrics['accuracy'] = bootstrap_analysis(test_y, test_y_pred, calc_accuracy, metric_name="accuracy", bins=args.bins)
            bootstrap_metrics['precision'] = bootstrap_analysis(test_y, test_y_pred, calc_precision, metric_name="precision", bins=args.bins)
            bootstrap_metrics['recall'] = bootstrap_analysis(test_y, test_y_pred, calc_recall, metric_name="recall", bins=args.bins)

        elif task_type == "regression":
            # Convert to numpy arrays if they're not already
            if test_y_c is not None and test_y_pred_c is not None:
                test_y_c = np.array(test_y_c)
                test_y_pred_c = np.array(test_y_pred_c)

                # Show progress bar for bootstrap analysis
                print("Performing bootstrap analysis for regression metrics...")

                # Define metric functions
                def calc_pcc(y_true, y_pred):
                    return np.corrcoef(y_true, y_pred)[0, 1]

                def calc_mse(y_true, y_pred):
                    return np.mean((y_true - y_pred) ** 2)

                # Calculate bootstrap metrics
                bootstrap_metrics['pcc'] = bootstrap_analysis(test_y_c, test_y_pred_c, calc_pcc, metric_name="PCC", bins=args.bins)
                bootstrap_metrics['mse'] = bootstrap_analysis(test_y_c, test_y_pred_c, calc_mse, metric_name="MSE", bins=args.bins)

                # For ICC, we'll use the original ICC calculation from test_pipeline
                # but format it to match our bootstrap results format
                if test_icc is not None:
                    icc_value = test_icc.loc[test_icc['Type'] == 'ICC3', 'ICC'].values[0]
                    icc_lower = test_icc.loc[test_icc['Type'] == 'ICC3', 'CI95%'].values[0].split('[')[1].split(',')[0]
                    icc_upper = test_icc.loc[test_icc['Type'] == 'ICC3', 'CI95%'].values[0].split(',')[1].split(']')[0]

                    bootstrap_metrics['icc'] = {
                        'value': float(icc_value),
                        'lower_bound': float(icc_lower),
                        'upper_bound': float(icc_upper)
                    }

            # Create regression plot
            plt.title("Regression " + args.checkpoint_dir.split('/')[-2])
            d_indices = [i for i in range(len(test_y_c))]
            plt.plot(test_y_c, d_indices, "ro", label="ground-truth")
            plt.plot(test_y_pred_c, d_indices, "bo", label="prediction")
            plt.legend()
            plt.savefig(
                fname=os.path.join(args.checkpoint_dir, f"regression_fig{fold}.png"),
                bbox_inches='tight'
            )
            plt.clf()
            plt.close()

            # Save ICC
            if test_icc is not None:
                test_icc.to_csv(os.path.join(args.checkpoint_dir, f"test_icc_kf{fold}.csv"))

        # Convert bootstrap metrics to DataFrame
        bootstrap_df = pd.DataFrame()

        for metric_name, metric_data in bootstrap_metrics.items():
            bootstrap_df.loc[0, f"{metric_name}"] = metric_data['value']
            bootstrap_df.loc[0, f"{metric_name}_lower"] = metric_data['lower_bound']
            bootstrap_df.loc[0, f"{metric_name}_upper"] = metric_data['upper_bound']

        # Save bootstrap metrics
        bootstrap_df.to_csv(os.path.join(args.checkpoint_dir, f"bootstrap_metrics_kf{fold}.csv"))

        # Also save original test metrics for comparison
        test_metrics_df = pd.DataFrame.from_dict(test_metrics)
        test_metrics_df.to_csv(os.path.join(args.checkpoint_dir, f"test_metrics_kf{fold}.csv"))

        if verbose:
            print("Bootstrap Metrics:")
            print(bootstrap_df)
