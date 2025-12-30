# Imports
import os
import numpy as np
from tqdm import tqdm
import pandas as pd

# PyTorch Imports
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchinfo import summary

# TorchMetrics Imports
from torchmetrics.functional.classification import (
    accuracy,
    f1_score,
    recall,
    precision,
    auroc
)
from torchmetrics.functional.regression import (
    mean_absolute_error,
    mean_squared_error,
    concordance_corrcoef,
    kendall_rank_corrcoef,
    pearson_corrcoef,
    r2_score,
    relative_squared_error,
    spearman_corrcoef
)

# pingouin-stats
import pingouin as pg

# Project Imports
from model_utilities import AM_SB, AM_MB, AM_SB_Regression



# Function: Get optimizer
def get_optim(model, optimizer, lr, weight_decay):
    if optimizer == "Adam":
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr, weight_decay=weight_decay)
    elif optimizer == 'SGD':
        optimizer = optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=lr, momentum=0.9, weight_decay=weight_decay)
    else:
        raise NotImplementedError
    return optimizer



# Function: Test Model Pipeline
def test_pipeline(test_set, config_json, device, checkpoint_dir, fold):

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
    features_ = config_json["features"]
    task_type = config_json["task_type"]


    # Dictionary with model settings for the initialization of the model object
    if task_type in ("classification", "clinical_subtype_classification"):
        model_dict = {
            "dropout":dropout,
            "dropout_prob":dropout_prob,
            'n_classes':n_classes,
            "encoding_size":encoding_size
        }
    elif task_type == "regression":
        model_dict = {
            "dropout":dropout,
            "dropout_prob":dropout_prob,
            "encoding_size":encoding_size
        }


    # Adapted from the CLAM framework
    assert model_size is not None, "Please define a model size."

    model_dict.update({"size_arg": model_size})

    # AM-SB
    if task_type in ("classification", "clinical_subtype_classification"):
        if model_type == 'am_sb':
            model = AM_SB(**model_dict)
        elif model_type == 'am_mb':
            model = AM_MB(**model_dict)
    elif task_type == "regression":
        if model_type == 'am_sb':
            model = AM_SB_Regression(**model_dict)

    if verbose:
        print(f"Using features: {features_}")
        summary(model)


    # Create DataLoaders
    # FIXME: The code is currently optimized for batch_size == 1)
    test_loader = DataLoader(
        dataset=test_set,
        batch_size=1,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory
    )

    # Load model checkpoint
    model.load_state_dict(torch.load(os.path.join(checkpoint_dir, f"best_model_kf{fold}.pt"), map_location=device))
    model.to(device)

    # Put model into evaluation
    model.eval()

    # Initialize variables to track values
    test_y_pred = list()
    test_y_pred_proba = list()
    test_y = list()

    # Create a dictionary for test metrics
    test_inference_info = dict()


    # Initialize variables
    test_y_pred = list()
    test_y_pred_proba = list()
    test_y_pred_c = list()
    test_y = list()
    test_y_c = list()
    test_y_cs = list()
    case_ids = list()
    svs_paths = list()

    # Get batch of data
    for _, input_data_dict in enumerate(test_loader):

        features, ssgsea_scores, ssgsea_scores_bin, c_subtype, case_id, svs_path, = input_data_dict["features"].to(device), input_data_dict["ssgsea_scores"].to(device), input_data_dict["ssgsea_scores_bin"].to(device), input_data_dict["c_subtype"], input_data_dict["case_id"], input_data_dict["svs_path"]
        output_dict = model(features)
        logits = output_dict['logits']
        y_pred = torch.where(logits > 0, 1.0, 0.0)
        y_pred_proba = F.sigmoid(logits)
        test_y_pred_c.extend(list(logits.squeeze(0).cpu().detach().numpy()))
        test_y_pred.extend(list(y_pred.squeeze(0).cpu().detach().numpy()))
        test_y.extend(list(ssgsea_scores_bin.cpu().detach().numpy()))
        test_y_c.extend(list(ssgsea_scores.cpu().detach().numpy()))
        test_y_pred_proba.extend(list(y_pred_proba.squeeze(0).cpu().detach().numpy()))
        test_y_cs.extend(c_subtype)
        case_ids.extend(case_id)
        svs_paths.extend(svs_path)


    # Test inference information
    test_inference_info["case_id"] = case_ids
    test_inference_info["svs_path"] = svs_paths
    test_inference_info["ssgsea_c"] = test_y_c
    test_inference_info["ssgsea_b"] = test_y
    test_inference_info["ssgsea_c_pred"] = test_y_pred_c
    test_inference_info["ssgsea_b_pred"] = test_y_pred
    test_inference_info["ssgsea_b_pred_proba"] = test_y_pred_proba
    test_inference_info["c_subtype"] = test_y_cs

    return test_inference_info



# Function: Perform bootstrap analysis
def bootstrap_analysis(y_true, y_pred, metric_value, task='binary', metric_name='accuracy', bins=1000, confidence=0.95):
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

        if metric_name == "accuracy":
            result = accuracy(preds=bootstrap_pred, target=bootstrap_true, task=task)
            result = result.item()
        elif metric_name == "f1_score":
            result = f1_score(preds=bootstrap_pred, target=bootstrap_true, task=task)
            result = result.item()
        elif metric_name == "precision":
            result = precision(preds=bootstrap_pred, target=bootstrap_true, task=task)
            result = result.item()
        elif metric_name == "recall":
            result = recall(preds=bootstrap_pred, target=bootstrap_true, task=task)
            result = result.item()
        elif metric_name == "auroc":
            result = auroc(preds=bootstrap_pred, target=bootstrap_true, task=task)
            result = result.item()
        elif metric_name == "auroc-clf-2c":
            result = auroc(preds=bootstrap_pred, target=bootstrap_true, num_classes=2, task=task)
            result = result.item()
        elif metric_name == "mean_squared_error":
            result = mean_squared_error(preds=bootstrap_pred, target=bootstrap_true)
            result = result.item()
        elif metric_name == "pearson_corrcoef":
            result = pearson_corrcoef(preds=bootstrap_pred, target=bootstrap_true)
            result = result.item()
        elif metric_name == "intraclass_corr":
            icc_samples = list()
            icc_judges = list()
            icc_scores = list()

            for i, s in enumerate(list(bootstrap_pred.numpy())):
                icc_samples.append(i)
                icc_judges.append('A')
                icc_scores.append(s)

            for j, c in enumerate(list(bootstrap_true.numpy())):
                icc_samples.append(j)
                icc_judges.append('B')
                icc_scores.append(c)

            icc_data = {
                'icc_samples':icc_samples,
                'icc_judges':icc_judges,
                'icc_scores':icc_scores
            }
            icc_data_df = pd.DataFrame.from_dict(icc_data)
            icc = pg.intraclass_corr(data=icc_data_df, targets='icc_samples', raters='icc_judges', ratings='icc_scores').round(3)
            result = icc.values[2,2]

        results.append(result)

    # Calculate confidence interval
    alpha = (1 - confidence) / 2
    lower_bound = np.percentile(results, alpha * 100)
    upper_bound = np.percentile(results, (1 - alpha) * 100)

    return {
        'value': metric_value,
        'lower_bound': lower_bound,
        'upper_bound': upper_bound,
        'results':results
    }



# Function: Get metrics per clinical subtype and split
def compute_metrics_per_clinical_subtype(checkpoint_dir, n_classes=2, fold=0, bts_nbins=1000):

    # Go through the possible evaluation names
    for eval_name in ('train', 'val', 'test'):
        csv_fpath = os.path.join(checkpoint_dir, 'clinical-subtype-analysis', 'results', f"{eval_name}_inference_info_kf{fold}.csv")
        info_df = pd.read_csv(csv_fpath)
        eval_metrics = dict()
        bootstrap_metrics = dict()

        # Go through the possible clinical subtypes
        for c_subtype, c_subtype_name in zip(("HER2+/HR+", "HER2+/HR-", "HER2-/HR+", "HER2-/HR-"),("her2pos_hrpos", "her2pos_hrneg", "her2neg_hrpos", "her2neg_hrneg")):

            # Load evaluation data for the clinical subtype
            info_df_ = info_df.copy()[info_df["c_subtype"]==c_subtype]

            # Get needed values
            y = torch.from_numpy(info_df_["ssgsea_b"].values)
            y_c = torch.from_numpy(info_df_["ssgsea_c"].values)
            y_pred = torch.from_numpy(info_df_["ssgsea_b_pred"].values)
            y_pred_proba = torch.from_numpy(info_df_["ssgsea_b_pred_proba"].values)
            y_pred_c = torch.from_numpy(info_df_["ssgsea_c_pred"].values)

            # Compute metrics
            if n_classes == 2:
                if len(y_pred.shape) == 2:
                    y_pred = y_pred.squeeze()

                acc = accuracy(
                    preds=y_pred,
                    target=y,
                    task='binary'
                )
                bootstrap_metrics['acc'] = bootstrap_analysis(y, y_pred, acc.item(), task='binary', metric_name="accuracy", bins=bts_nbins)


                f1 = f1_score(
                    preds=y_pred,
                    target=y,
                    task='binary'
                )
                bootstrap_metrics['f1'] = bootstrap_analysis(y, y_pred, f1.item(), task='binary', metric_name="f1_score", bins=bts_nbins)


                rec = recall(
                    preds=y_pred,
                    target=y,
                    task='binary'
                )
                bootstrap_metrics['rec'] = bootstrap_analysis(y, y_pred, rec.item(), task='binary', metric_name="recall", bins=bts_nbins)


                prec = precision(
                    preds=y_pred,
                    target=y,
                    task='binary'
                )
                bootstrap_metrics['prec'] = bootstrap_analysis(y, y_pred, prec.item(), task='binary', metric_name="precision", bins=bts_nbins)


                auc = auroc(
                    preds=y_pred_proba,
                    target=y,
                    task='binary'
                )
                bootstrap_metrics['auc'] = bootstrap_analysis(y, y_pred_proba, auc.item(), task='binary', metric_name="auroc", bins=bts_nbins)


            else:
                acc = accuracy(
                    preds=y_pred,
                    target=y,
                    task='multiclass',
                    num_classes=n_classes
                )

                f1 = f1_score(
                    preds=y_pred,
                    target=y,
                    task='multiclass',
                    num_classes=n_classes
                )

                rec = recall(
                    preds=y_pred,
                    target=y,
                    task='multiclass',
                    num_classes=n_classes
                )

                prec = precision(
                    preds=y_pred,
                    target=y,
                    task='multiclass',
                    num_classes=n_classes
                )

                auc = auroc(
                    preds=y_pred_proba,
                    target=y,
                    task='multiclass',
                    num_classes=n_classes
                )

            # Regression Metrics
            mae = mean_absolute_error(
                preds=y_pred_c,
                target=y_c
            )

            mse = mean_squared_error(
                preds=y_pred_c,
                target=y_c
            )

            ccc = concordance_corrcoef(
                preds=y_pred_c,
                target=y_c
            )

            krcc = kendall_rank_corrcoef(
                preds=y_pred_c,
                target=y_c
            )

            pcc = pearson_corrcoef(
                preds=y_pred_c,
                target=y_c
            )
            bootstrap_metrics['pcc'] = bootstrap_analysis(y_c, y_pred_c, pcc.item(), metric_name="pearson_corrcoef", bins=bts_nbins)

            r2s = r2_score(
                preds=y_pred_c,
                target=y_c
            )

            rse = relative_squared_error(
                preds=y_pred_c,
                target=y_c
            )

            scc = spearman_corrcoef(
                preds=y_pred_c,
                target=y_c
            )

            # Append test AUC to the test metrics
            eval_metrics["acc"] = [acc.item()]
            eval_metrics["f1"] = [f1.item()]
            eval_metrics["rec"] = [rec.item()]
            eval_metrics["prec"] = [prec.item()]
            eval_metrics["auc"] = [auc.item()]
            eval_metrics["mae"] = [mae.item()]
            eval_metrics["mse"] = [mse.item()]
            eval_metrics["ccc"] = [ccc.item()]
            eval_metrics["krcc"] = [krcc.item()]
            eval_metrics["pcc"] = [pcc.item()]
            eval_metrics["r2s"] = [r2s.item()]
            eval_metrics["rse"] = [rse.item()]
            eval_metrics["scc"] = [scc.item()]

            eval_metrics_info_df = pd.DataFrame.from_dict(eval_metrics)
            eval_metrics_info_df.to_csv(os.path.join(checkpoint_dir, 'clinical-subtype-analysis', 'results', f"{eval_name}_{c_subtype_name}_eval_metrics_info_kf{fold}.csv"))

            # Build DataFrame for pingouin-stats
            icc_samples = list()
            icc_judges = list()
            icc_scores = list()

            for i, s in enumerate(list(y_pred_c.numpy())):
                icc_samples.append(i)
                icc_judges.append('A')
                icc_scores.append(s)

            for j, c in enumerate(list(y_c.numpy())):
                icc_samples.append(j)
                icc_judges.append('B')
                icc_scores.append(c)

            icc_data = {
                'icc_samples':icc_samples,
                'icc_judges':icc_judges,
                'icc_scores':icc_scores
            }
            icc_data_df = pd.DataFrame.from_dict(icc_data)

            eval_icc = pg.intraclass_corr(data=icc_data_df, targets='icc_samples', raters='icc_judges', ratings='icc_scores').round(3)
            eval_icc.to_csv(os.path.join(checkpoint_dir, 'clinical-subtype-analysis', 'results', f"{eval_name}_{c_subtype_name}_eval_icc_info_kf{fold}.csv"))


            # Convert bootstrap metrics to DataFrame
            bootstrap_df = pd.DataFrame()

            for metric_name, metric_data in bootstrap_metrics.items():
                bootstrap_df.loc[0, f"{metric_name}"] = metric_data['value']
                bootstrap_df.loc[0, f"{metric_name}_lower"] = metric_data['lower_bound']
                bootstrap_df.loc[0, f"{metric_name}_upper"] = metric_data['upper_bound']
                metric_results = metric_data['results']
                metric_results_df = pd.DataFrame.from_dict(
                    {'results':metric_results}
                )
                metric_results_df.to_csv(os.path.join(checkpoint_dir, 'clinical-subtype-analysis', 'results', f"{eval_name}_{c_subtype_name}_bootstrap_results_{metric_name}_kf{fold}.csv"))

            # Save bootstrap metrics
            bootstrap_df.to_csv(os.path.join(checkpoint_dir, 'clinical-subtype-analysis', 'results', f"{eval_name}_{c_subtype_name}_bootstrap_metrics_kf{fold}.csv"))