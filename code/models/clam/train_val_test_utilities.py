# Imports
import os
import numpy as np
import pandas as pd
from tqdm import tqdm

# PyTorch Imports
import torch
import torch.nn as nn
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

# WandB Imports
import wandb



# Function: Get optimizer
def get_optim(model, optimizer, lr, weight_decay):
    if optimizer == "Adam":
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr, weight_decay=weight_decay)
    elif optimizer == 'SGD':
        optimizer = optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=lr, momentum=0.9, weight_decay=weight_decay)
    else:
        raise NotImplementedError
    return optimizer



# Function: Train-Validation Model Pipeline
def train_val_pipeline(datasets, config_json, device, experiment_dir, checkpoint_fname, wandb_project_name, early_stopping=False, patience=20, stop_epoch=50, verbose=True):

    # Load the parameters from the configuration JSON
    n_classes = config_json["data"]["n_classes"]
    dropout = config_json["hyperparameters"]["dropout"]
    dropout_prob = config_json["hyperparameters"]["dropout_prob"]
    model_size = config_json["hyperparameters"]["model_size"]
    model_type = config_json["hyperparameters"]["model_type"]
    verbose = config_json["verbose"]
    optimizer = config_json["hyperparameters"]["optimizer"]
    lr = config_json["hyperparameters"]["lr"]
    weight_decay = config_json["hyperparameters"]["weight_decay"]
    num_workers = config_json["data"]["num_workers"]
    pin_memory = config_json["data"]["pin_memory"]
    epochs = config_json["hyperparameters"]["epochs"]
    early_stopping = config_json["hyperparameters"]["early_stopping"]
    encoding_size = config_json['data']['encoding_size']
    features_ = config_json["features"]
    task_type = config_json["task_type"]


    # Build a configuration dictionary for WandB
    wandb_project_config = {
        "n_classes":n_classes,
        "dropout":dropout,
        "dropout_prob":dropout_prob,
        "model_size":model_size,
        "model_type":model_type,
        "verbose":verbose,
        "optimizer":optimizer,
        "lr":lr,
        "weight_decay":weight_decay,
        "num_workers":num_workers,
        "pin_memory":pin_memory,
        "epochs":epochs,
        "early_stopping":early_stopping,
        "encoding_size":encoding_size,
        "features":features_,
        "task_type":task_type
    }

    # Initialize WandB
    wandb_run = wandb.init(
        project="dl-prediction-brca-tiph-ext",
        name=wandb_project_name,
        config=wandb_project_config
    )
    assert wandb_run is wandb.run



    # Get data splits
    train_set, val_set = datasets


    # Get loss function
    if task_type in ("classification", "clinical_subtype_classification"):
        loss_fn = nn.CrossEntropyLoss()
    elif task_type == "regression":
        loss_fn = nn.MSELoss()


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
    

    # Move into model into device
    model.to(device=device)

    if verbose:
        summary(model)

    
    # Get and load the optimizer
    optimizer = get_optim(
        model=model,
        optimizer=optimizer,
        lr=lr,
        weight_decay=weight_decay
    )
    

    # Create DataLoaders 
    # FIXME: The code is currently optimized for batch_size == 1)
    train_loader = DataLoader(
        dataset=train_set,
        batch_size=1,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory
    )
    val_loader = DataLoader(
        dataset=val_set,
        batch_size=1,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory
    )


    # Tracking parameters
    tracking_params = {"min_val_loss":np.inf}
    if early_stopping:
        tracking_params["patience"] = patience
        tracking_params["stop_epoch"] = stop_epoch
        tracking_params["early_stopping_counter"] = 0
        tracking_params["early_stop"] = False


    # Training Pipeline
    for epoch in range(epochs):
        train_loop_clam(
            epoch=epoch, 
            model=model, 
            loader=train_loader, 
            optimizer=optimizer, 
            n_classes=n_classes,
            task_type=task_type,
            loss_fn=loss_fn,
            device=device,
            wandb_run=wandb_run
        )
        validate_loop_clam(
            model=model, 
            loader=val_loader, 
            n_classes=n_classes, 
            task_type=task_type,
            tracking_params=tracking_params, 
            loss_fn=loss_fn, 
            experiment_dir=experiment_dir,
            checkpoint_fname=checkpoint_fname,
            device=device,
            wandb_run=wandb_run
        )

        # Stop training according to the early stopping parameters
        if early_stopping:
            if tracking_params["early_stop"]: 
                break

    return



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



# Function: Test Model Pipeline
def test_pipeline(test_set, config_json, device, checkpoint_dir, fold, bts_nbins):

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
    test_metrics = dict()


    # Initialize variables 
    test_y_pred = list()
    test_y_pred_proba = list()
    test_y_pred_proba_ = list()
    test_y_pred_c = list()
    test_y = list()
    test_y_c = list()

    # Get batch of data
    for _, input_data_dict in enumerate(test_loader):

        if task_type == "classification":
            features, ssgsea_scores = input_data_dict['features'].to(device), input_data_dict['ssgsea_scores'].to(device)
            output_dict = model(features)
            logits, y_pred, y_proba, y_proba_ = output_dict['logits'], output_dict['y_pred'], output_dict["y_proba"], output_dict["y_proba_"]
            test_y_pred_c.extend(list(logits.cpu().detach().numpy()))
            test_y_pred.extend(list(y_pred.cpu().detach().numpy()))
            test_y.extend(list(ssgsea_scores.cpu().detach().numpy()))
            test_y_pred_proba.extend(list(y_proba.cpu().detach().numpy()))
            test_y_pred_proba_.append(y_proba_.cpu().detach().numpy())
            # print(test_y_pred_proba_)

        
        elif task_type == "clinical_subtype_classification":
            features, c_subtypes = input_data_dict['features'].to(device), input_data_dict['c_subtype_label'].to(device)
            output_dict = model(features)
            logits, y_pred, y_proba = output_dict['logits'], output_dict['y_pred'], output_dict['y_proba']
            test_y_pred.extend(list(y_pred.cpu().detach().numpy()))
            test_y.extend(list(c_subtypes.cpu().detach().numpy()))
            test_y_pred_proba.extend(list(y_proba.cpu().detach().numpy()))

        elif task_type == "regression":
            features, ssgsea_scores, ssgsea_scores_bin = input_data_dict["features"].to(device), input_data_dict["ssgsea_scores"].to(device), input_data_dict["ssgsea_scores_bin"].to(device)
            output_dict = model(features)
            logits = output_dict['logits']
            y_pred = torch.where(logits > 0, 1.0, 0.0)
            y_pred_proba = F.sigmoid(logits)
            test_y_pred_c.extend(list(logits.squeeze(0).cpu().detach().numpy()))
            test_y_pred.extend(list(y_pred.squeeze(0).cpu().detach().numpy()))
            test_y.extend(list(ssgsea_scores_bin.cpu().detach().numpy()))
            test_y_c.extend(list(ssgsea_scores.cpu().detach().numpy()))
            test_y_pred_proba.extend(list(y_pred_proba.squeeze(0).cpu().detach().numpy()))


    # Compute metrics
    bootstrap_metrics = {}
    test_y_pred = torch.from_numpy(np.array(test_y_pred))
    test_y = torch.from_numpy(np.array(test_y))
    test_y_pred_proba = torch.from_numpy(np.array(test_y_pred_proba))
    test_y_pred_proba_ = torch.from_numpy(np.array(test_y_pred_proba_))
    # print(test_y_pred_proba_.shape, test_y_pred_proba_)
    
    if task_type == "regression":
        test_y_pred_c = torch.from_numpy(np.array(test_y_pred_c))
        test_y_c = torch.from_numpy(np.array(test_y_c))
    elif task_type == "classification":
        test_y_pred_c = torch.from_numpy(np.array(test_y_pred_c))

    if n_classes == 2:
        if len(test_y_pred.shape) == 2:
            test_y_pred = test_y_pred.squeeze()
        
        acc = accuracy(
            preds=test_y_pred,
            target=test_y,
            task='binary'
        )

        f1 = f1_score(
            preds=test_y_pred,
            target=test_y,
            task='binary'
        )

        rec = recall(
            preds=test_y_pred,
            target=test_y,
            task='binary'
        )

        prec = precision(
            preds=test_y_pred,
            target=test_y,
            task='binary'
        )

        # Calculate bootstrap metrics for classification
        bootstrap_metrics['acc'] = bootstrap_analysis(test_y, test_y_pred, acc.item(), task='binary', metric_name="accuracy", bins=bts_nbins)
        bootstrap_metrics['f1'] = bootstrap_analysis(test_y, test_y_pred, f1.item(), task='binary', metric_name="f1_score", bins=bts_nbins)
        bootstrap_metrics['rec'] = bootstrap_analysis(test_y, test_y_pred, rec.item(), task='binary', metric_name="recall", bins=bts_nbins)
        bootstrap_metrics['prec'] = bootstrap_analysis(test_y, test_y_pred, prec.item(), task='binary', metric_name="precision", bins=bts_nbins)
        


        # Note: Original implementation uses softmax for 2 classes, so we need to compute AUROC this way
        if task_type == "classification":
            # print(test_y_pred_proba_.squeeze().shape)
            auc = auroc(
                preds=test_y_pred_proba_.squeeze(),
                target=test_y,
                num_classes=2,
                task='multiclass'
            )
            bootstrap_metrics['auc'] = bootstrap_analysis(test_y, test_y_pred_proba_.squeeze(), auc.item(), task='multiclass', metric_name="auroc-clf-2c", bins=bts_nbins)

        else:
            auc = auroc(
                preds=test_y_pred_proba,
                target=test_y,
                task='binary'
            )
            bootstrap_metrics['auc'] = bootstrap_analysis(test_y, test_y_pred_proba, auc.item(), task='binary', metric_name="auroc", bins=bts_nbins)

    else:
        acc = accuracy(
            preds=test_y_pred,
            target=test_y,
            task='multiclass',
            num_classes=n_classes
        )

        f1 = f1_score(
            preds=test_y_pred,
            target=test_y,
            task='multiclass',
            num_classes=n_classes
        )

        rec = recall(
            preds=test_y_pred,
            target=test_y,
            task='multiclass',
            num_classes=n_classes
        )

        prec = precision(
            preds=test_y_pred,
            target=test_y,
            task='multiclass',
            num_classes=n_classes
        )

        auc = auroc(
            preds=test_y_pred_proba,
            target=test_y,
            task='multiclass',
            num_classes=n_classes
        )
    
    if task_type == "regression":
        mae = mean_absolute_error(
            preds=test_y_pred_c,
            target=test_y_c
        )
        
        mse = mean_squared_error(
            preds=test_y_pred_c,
            target=test_y_c
        )
        # bootstrap_metrics['mse'] = bootstrap_analysis(test_y_c, test_y_pred_c, mse.item(), metric_name="mean_squared_error", bins=bts_nbins)

        ccc = concordance_corrcoef(
            preds=test_y_pred_c,
            target=test_y_c
        )

        krcc = kendall_rank_corrcoef(
            preds=test_y_pred_c,
            target=test_y_c
        )

        pcc = pearson_corrcoef(
            preds=test_y_pred_c,
            target=test_y_c
        )
        bootstrap_metrics['pcc'] = bootstrap_analysis(test_y_c, test_y_pred_c, pcc.item(), metric_name="pearson_corrcoef", bins=bts_nbins)

        r2s = r2_score(
            preds=test_y_pred_c,
            target=test_y_c
        )

        rse = relative_squared_error(
            preds=test_y_pred_c,
            target=test_y_c
        )

        scc = spearman_corrcoef(
            preds=test_y_pred_c,
            target=test_y_c
        )



    # Append test AUC to the test metrics
    test_metrics["acc"] = [acc.item()]
    test_metrics["f1"] = [f1.item()]
    test_metrics["rec"] = [rec.item()]
    test_metrics["prec"] = [prec.item()]
    test_metrics["auc"] = [auc.item()]
    
    if task_type == "regression":
        test_metrics["mae"] = [mae.item()]
        test_metrics["mse"] = [mse.item()]
        test_metrics["ccc"] = [ccc.item()]
        test_metrics["krcc"] = [krcc.item()]
        test_metrics["pcc"] = [pcc.item()]
        test_metrics["r2s"] = [r2s.item()]
        test_metrics["rse"] = [rse.item()]
        test_metrics["scc"] = [scc.item()]

        # Build DataFrame for pingouin-stats
        icc_samples = list()
        icc_judges = list()
        icc_scores = list()

        for i, s in enumerate(list(test_y_pred_c.numpy())):
            icc_samples.append(i)
            icc_judges.append('A')
            icc_scores.append(s)
        
        for j, c in enumerate(list(test_y_c.numpy())):
            icc_samples.append(j)
            icc_judges.append('B')
            icc_scores.append(c)
        
        icc_data = {
            'icc_samples':icc_samples,
            'icc_judges':icc_judges,
            'icc_scores':icc_scores
        }
        icc_data_df = pd.DataFrame.from_dict(icc_data)
        test_icc = pg.intraclass_corr(data=icc_data_df, targets='icc_samples', raters='icc_judges', ratings='icc_scores').round(3)
        bootstrap_metrics['icc'] = bootstrap_analysis(test_y_c, test_y_pred_c, test_icc.values[2,2], metric_name="intraclass_corr", bins=bts_nbins)


    if task_type == "regression":
        return test_metrics, test_y_c.numpy(), test_y_pred_c.numpy(), test_icc, bootstrap_metrics
    else:
        return test_metrics, None, None, None, bootstrap_metrics



# Function: Inference Model Pipeline
def inference_pipeline(test_set, config_json, device, checkpoint_dir, fold):

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

    # Initialize variables 
    test_case_ids = list()
    test_svs_paths = list()
    test_y_pred = list()
    test_y_pred_proba = list()
    test_y_pred_c = list()
    test_y = list()
    test_y_c = list()

    # Get batch of data
    for _, input_data_dict in enumerate(test_loader):

        if task_type == "classification":
            case_id, svs_path, features, ssgsea_scores = input_data_dict["case_id"], input_data_dict["svs_path"], input_data_dict['features'].to(device), input_data_dict['ssgsea_scores'].to(device)
            output_dict = model(features)
            logits, y_pred, y_proba = output_dict['logits'], output_dict['y_pred'], output_dict["y_proba"]
            test_y_pred_c.extend(list(logits.cpu().detach().numpy()))
            test_y_pred.extend(list(y_pred.cpu().detach().numpy()))
            test_y.extend(list(ssgsea_scores.cpu().detach().numpy()))
            test_y_pred_proba.extend(list(y_proba.detach().numpy()))
            test_case_ids.extend(case_id)
            test_svs_paths.extend(svs_path)
        
        elif task_type == "clinical_subtype_classification":
            case_id, svs_path, features, c_subtypes = input_data_dict["case_id"], input_data_dict["svs_path"], input_data_dict['features'].to(device), input_data_dict['c_subtype_label'].to(device)
            output_dict = model(features)
            logits, y_pred, y_proba = output_dict['logits'], output_dict['y_pred'], output_dict['y_proba']
            test_y_pred.extend(list(y_pred.cpu().detach().numpy()))
            test_y.extend(list(c_subtypes.cpu().detach().numpy()))
            test_y_pred_proba.extend(list(y_proba.cpu().detach().numpy()))
            test_case_ids.extend(case_id)
            test_svs_paths.extend(svs_path)

        elif task_type == "regression":
            case_id, svs_path, features, ssgsea_scores, ssgsea_scores_bin = input_data_dict["case_id"], input_data_dict["svs_path"], input_data_dict["features"].to(device), input_data_dict["ssgsea_scores"].to(device), input_data_dict["ssgsea_scores_bin"].to(device)
            output_dict = model(features)
            logits = output_dict['logits']
            y_pred = torch.where(logits > 0, 1.0, 0.0)
            y_pred_proba = F.sigmoid(logits)
            test_y_pred_c.extend(list(logits.squeeze(0).cpu().detach().numpy()))
            test_y_pred.extend(list(y_pred.squeeze(0).cpu().detach().numpy()))
            test_y.extend(list(ssgsea_scores_bin.cpu().detach().numpy()))
            test_y_c.extend(list(ssgsea_scores.cpu().detach().numpy()))
            test_y_pred_proba.extend(list(y_pred_proba.squeeze(0).cpu().detach().numpy()))
            test_case_ids.extend(case_id)
            test_svs_paths.extend(svs_path)



        # Create a test inference dictionary
        test_inference_dict = {
            "case_id":test_case_ids,
            "svs_path":test_svs_paths,
            "test_y":test_y,
            "test_y_c":test_y_c,
            "test_y_pred":test_y_pred,
            "test_y_pred_c":test_y_pred_c,
            "test_y_pred_proba":test_y_pred_proba
        }


    return test_inference_dict



# Function: Train Loop for CLAM
def train_loop_clam(epoch, model, loader, optimizer, n_classes, task_type, loss_fn, device, wandb_run):

    # Put model into training mode
    model.train()

    # Initialize variables 
    train_loss = 0.
    train_y_pred = list()
    train_y_pred_proba = list()
    train_y = list()

    # Get batch of data
    for _, input_data_dict in enumerate(loader):

        if task_type == "classification":
            features, ssgsea_scores = input_data_dict['features'].to(device), input_data_dict['ssgsea_scores'].to(device)
            output_dict = model(features)
            logits, y_pred, y_proba = output_dict['logits'], output_dict['y_pred'], output_dict['y_proba']
            train_y_pred.extend(list(y_pred.cpu().detach().numpy()))
            train_y.extend(list(ssgsea_scores.cpu().detach().numpy()))
            train_y_pred_proba.extend(list(y_proba.cpu().detach().numpy()))
            loss = loss_fn(logits, ssgsea_scores)
        
        elif task_type == "clinical_subtype_classification":
            features, c_subtypes = input_data_dict['features'].to(device), input_data_dict['c_subtype_label'].to(device)
            output_dict = model(features)
            logits, y_pred, y_proba = output_dict['logits'], output_dict['y_pred'], output_dict['y_proba']
            train_y_pred.extend(list(y_pred.cpu().detach().numpy()))
            train_y.extend(list(c_subtypes.cpu().detach().numpy()))
            train_y_pred_proba.extend(list(y_proba.cpu().detach().numpy()))
            loss = loss_fn(logits, c_subtypes)

        elif task_type == "regression":
            features, ssgsea_scores, ssgsea_scores_bin = input_data_dict["features"].to(device), input_data_dict["ssgsea_scores"].to(device), input_data_dict["ssgsea_scores_bin"].to(device)
            output_dict = model(features)
            logits = output_dict['logits']
            y_pred = torch.where(logits > 0, 1.0, 0.0)
            y_pred_proba = F.sigmoid(logits)
            train_y_pred.extend(list(y_pred.squeeze(0).cpu().detach().numpy()))
            train_y.extend(list(ssgsea_scores_bin.cpu().detach().numpy()))
            train_y_pred_proba.extend(list(y_pred_proba.squeeze(0).cpu().detach().numpy()))
            loss = loss_fn(logits.squeeze(0), ssgsea_scores.float())
        
        # Get loss values and update records
        loss_value = loss.item()
        train_loss += loss_value

        # Log batch metrics to WandB
        wandb_run.log({"train_batch_loss":loss_value})
        
        # Backpropagation 
        loss.backward()

        # Update weights
        optimizer.step()
        optimizer.zero_grad()



    # Calculate loss and error for epoch
    train_loss /= len(loader)


    # Compute metrics
    train_y_pred = torch.from_numpy(np.array(train_y_pred))
    train_y = torch.from_numpy(np.array(train_y))
    train_y_pred_proba = torch.from_numpy(np.array(train_y_pred_proba))

    if n_classes == 2:
        acc = accuracy(
            preds=train_y_pred,
            target=train_y,
            task='binary'
        )

        f1 = f1_score(
            preds=train_y_pred,
            target=train_y,
            task='binary'
        )

        rec = recall(
            preds=train_y_pred,
            target=train_y,
            task='binary'
        )

        prec = precision(
            preds=train_y_pred,
            target=train_y,
            task='binary'
        )

        auc = auroc(
            preds=train_y_pred_proba,
            target=train_y,
            task='binary'
        )

    else:
        acc = accuracy(
            preds=train_y_pred,
            target=train_y,
            task='multiclass',
            num_classes=n_classes
        )

        f1 = f1_score(
            preds=train_y_pred,
            target=train_y,
            task='multiclass',
            num_classes=n_classes
        )

        rec = recall(
            preds=train_y_pred,
            target=train_y,
            task='multiclass',
            num_classes=n_classes
        )

        prec = precision(
            preds=train_y_pred,
            target=train_y,
            task='multiclass',
            num_classes=n_classes
        )

        auc = auroc(
            preds=train_y_pred_proba,
            target=train_y,
            task='multiclass',
            num_classes=n_classes
        )

    
    # Log metrics into W&B
    wandb_run.log(
        {
            "train_epoch":epoch,
            "train_loss":train_loss,
            "train_acc":acc,
            "train_f1":f1,
            "train_rec":rec,
            "train_prec":prec,
            "train_auc":auc
        }
    )

    return



# Function: Validation Loop for CLAM
def validate_loop_clam(model, loader, n_classes, task_type, tracking_params, loss_fn, experiment_dir, checkpoint_fname, device, wandb_run):

    # Put model into evaluation 
    model.eval()
    
    # Initialize variables to track values
    val_loss = 0.
    val_y_pred = list()
    val_y_pred_proba = list()
    val_y = list()


    # Go through data batches and get metric values
    with torch.no_grad():
        for _, input_data_dict in enumerate(loader):
            if task_type == "classification":
                features, ssgsea_scores = input_data_dict['features'].to(device), input_data_dict['ssgsea_scores'].to(device)
                output_dict = model(features)
                logits, y_pred, y_proba = output_dict['logits'], output_dict['y_pred'], output_dict['y_proba']
                val_y_pred.extend(list(y_pred.cpu().detach().numpy()))
                val_y.extend(list(ssgsea_scores.cpu().detach().numpy()))
                val_y_pred_proba.extend(list(y_proba.cpu().detach().numpy()))
                loss = loss_fn(logits, ssgsea_scores)

            elif task_type == "clinical_subtype_classification":
                features, c_subtypes = input_data_dict['features'].to(device), input_data_dict['c_subtype_label'].to(device)
                output_dict = model(features)
                logits, y_pred, y_proba = output_dict['logits'], output_dict['y_pred'], output_dict['y_proba']
                val_y_pred.extend(list(y_pred.cpu().detach().numpy()))
                val_y.extend(list(c_subtypes.cpu().detach().numpy()))
                val_y_pred_proba.extend(list(y_proba.cpu().detach().numpy()))
                loss = loss_fn(logits, c_subtypes)

            elif task_type == "regression":
                features, ssgsea_scores, ssgsea_scores_bin = input_data_dict["features"].to(device), input_data_dict["ssgsea_scores"].to(device), input_data_dict["ssgsea_scores_bin"].to(device)
                output_dict = model(features)
                logits = output_dict['logits']
                y_pred = torch.where(logits > 0, 1.0, 0.0)
                y_pred_proba = F.sigmoid(logits)
                val_y_pred.extend(list(y_pred.squeeze(0).cpu().detach().numpy()))
                val_y.extend(list(ssgsea_scores_bin.cpu().detach().numpy()))
                val_y_pred_proba.extend(list(y_pred_proba.squeeze(0).cpu().detach().numpy()))
                loss = loss_fn(logits.squeeze(0), ssgsea_scores.float())

            loss_value = loss.item()
            val_loss += loss_value

            # Log batch metrics to WandB
            wandb_run.log({"val_batch_loss":loss_value})


    # Updated final validation loss
    val_loss /= len(loader)

    # Compute metrics
    val_y_pred = torch.from_numpy(np.array(val_y_pred))
    val_y = torch.from_numpy(np.array(val_y))
    val_y_pred_proba = torch.from_numpy(np.array(val_y_pred_proba))

    if n_classes == 2:
        acc = accuracy(
            preds=val_y_pred,
            target=val_y,
            task='binary'
        )

        f1 = f1_score(
            preds=val_y_pred,
            target=val_y,
            task='binary'
        )

        rec = recall(
            preds=val_y_pred,
            target=val_y,
            task='binary'
        )

        prec = precision(
            preds=val_y_pred,
            target=val_y,
            task='binary'
        )

        auc = auroc(
            preds=val_y_pred_proba,
            target=val_y,
            task='binary'
        )

    else:
        acc = accuracy(
            preds=val_y_pred,
            target=val_y,
            task='multiclass',
            num_classes=n_classes
        )

        f1 = f1_score(
            preds=val_y_pred,
            target=val_y,
            task='multiclass',
            num_classes=n_classes
        )

        rec = recall(
            preds=val_y_pred,
            target=val_y,
            task='multiclass',
            num_classes=n_classes
        )

        prec = precision(
            preds=val_y_pred,
            target=val_y,
            task='multiclass',
            num_classes=n_classes
        )

        auc = auroc(
            preds=val_y_pred_proba,
            target=val_y,
            task='multiclass',
            num_classes=n_classes
        )

    
    # Log metrics into W&B
    wandb_run.log(
        {
            "val_loss":val_loss,
            "val_acc":acc,
            "val_f1":f1,
            "val_rec":rec,
            "val_prec":prec,
            "val_auc":auc
        }
    )


    # Save checkpoints based on tracking_params parameters
    if tracking_params is not None:
        
        assert experiment_dir
        
        if val_loss < tracking_params["min_val_loss"]:
            tracking_params["min_val_loss"] = val_loss
            torch.save(model.state_dict(), os.path.join(experiment_dir, checkpoint_fname))
            if "early_stop" in tracking_params.keys():
                tracking_params["counter"] = 0
        else:
            if "early_stop" in tracking_params.keys():
                tracking_params["counter"] += 1
                if tracking_params["counter"] >= tracking_params["patience"]:
                    tracking_params["early_stop"] = True

    return
