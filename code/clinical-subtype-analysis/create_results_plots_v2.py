# Imports
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D



# Constants
SUBTYPES_TABLE_DICT = {
    "HER2-/HR-":"her2neg_hrneg",
    "HER2-/HR+":"her2neg_hrpos",
    "HER2+/HR-":"her2pos_hrneg",
    "HER2+/HR+":"her2pos_hrpos"
}

MODELS_LIST = [
    "HER2-/HR-",
    "HER2-/HR+",
    "HER2+/HR-",
    "HER2+/HR+"
]



# Function: Get average task dataframe
def get_avg_task_df(all_tasks_df, all_tasks, task, models, metrics, immune_task_indices, metabolic_task_indices, tumor_task_indices):

    # Creates a DataFrame for average tasks based on the specified task type.
    avg_task_df = pd.DataFrame({
        'task':list(),
        'model':list(),
        'metric':list(),
        'value':list(),
        'ci_lower':list(),
        'ci_upper':list()
    })

    for model_name in models:
        for metric_name in metrics:
            if task == 'Average Across Tasks':
                task_mask = ~all_tasks_df['task'].isin(['Average Across Tasks', 'Average Across Immune Tasks', 'Average Across Metabolic Tasks', 'Average Across Tumor Tasks'])
            elif task == 'Average Across Immune Tasks':
                task_mask = all_tasks_df['task'].isin([all_tasks[i] for i in immune_task_indices])
            elif task == 'Average Across Metabolic Tasks':
                task_mask = all_tasks_df['task'].isin([all_tasks[i] for i in metabolic_task_indices])
            elif task == 'Average Across Tumor Tasks':
                task_mask = all_tasks_df['task'].isin([all_tasks[i] for i in tumor_task_indices])
            else:
                continue

            model_mask = (all_tasks_df['model'] == model_name)
            metric_mask = (all_tasks_df['metric'] == metric_name)

            filtered_data = all_tasks_df[task_mask & model_mask & metric_mask]

            avg_value = filtered_data['value'].mean()
            avg_ci_lower = filtered_data['ci_lower'].mean()
            avg_ci_upper = filtered_data['ci_upper'].mean()

            avg_task_df = pd.concat([avg_task_df, pd.DataFrame({
                'task':[task],
                'model':[model_name],
                'metric':[metric_name],
                'value':[avg_value],
                'ci_lower':[avg_ci_lower],
                'ci_upper':[avg_ci_upper]
            })], ignore_index=True)

    return avg_task_df



# Function: Compute the average across tasks
def compute_avg_act(arr_vals):

    # Computes the average across tasks for a given array of values.
    sorted_vals = np.zeros_like(arr_vals)

    for c in range(arr_vals.shape[1]):
        row_ = arr_vals[:, c]
        # print(row_)
        row_sorted = np.sort(row_)
        sorted_vals[:, c] = row_sorted

    # Create a new column with the average across tasks
    avg_col = np.mean(sorted_vals, axis=1)
    # print(avg_col.shape, avg_col)

    return avg_col



# Function: Create grouped boxplots
def plot_comparison_boxplots(
        point_df,
        violin_df,
        models,
        metrics,
        task,
        title='Impact of Training Dataset on Model Performance',
        ylim=(0.0, 1.05),
        palette='viridis',
        dodge_width=0.7,
        output_path=""):
    """
    Generates grouped boxplots for model performance comparison.
    This function is purely for visualization and contains no data calculations.
    Args:
        point_df (pd.DataFrame): DataFrame with point estimates and CIs.
        violin_df (pd.DataFrame): DataFrame with bootstrap samples. Can be empty.
        models (list): The list of models to plot, in order.
        metrics (list): The list of metrics to plot, in order.
        output_path (str or Path): The full path to save the output image.
        title (str, optional): The title for the plot.
        ylim (tuple, optional): The Y-axis limits for the plot.
        palette (str, optional): The name of the Matplotlib colormap to use for models.
        dodge_width (float, optional): The amount of space to dodge models within a group.
        output_path (str or Path, optional): The full path to save the output image.
    """

    assert os.path.isdir(output_path), f"Output path {output_path} does not exist."

    if point_df.empty:
        print("Error: Cannot generate plot with empty point data.")
        return

    # --- Plotting Setup ---
    FONT_SIZES = {'title': 18, 'axis_label': 14, 'tick_label': 12, 'legend': 12}

    # FIXME: Automate style selection based on background color
    # print(plt.style.available)
    # plt.style.use('seaborn-whitegrid')
    plt.style.use('seaborn-v0_8-whitegrid')
    # Use a colormap for a flexible number of models
    colormap = plt.get_cmap(palette)
    model_colors = {model: colormap(i / len(models)) for i, model in enumerate(models)}
    fig, ax = plt.subplots(figsize=(12, 8))


    # --- Calculate X positions to group models under each metric ---
    x_positions_dict = {}
    for metric_idx, metric_name in enumerate(metrics):
        # Determine which models have data for the current metric to group them correctly
        models_with_data = [m for m in models if m in point_df[point_df['metric'] == metric_name]['model'].unique()]
        num_models_with_data = len(models_with_data)
        offsets = np.linspace(-dodge_width / 2, dodge_width / 2, num=num_models_with_data) if num_models_with_data > 1 else [0]
        for i, model in enumerate(models_with_data):
            x_positions_dict[(metric_name, model)] = metric_idx + offsets[i]

    # --- Draw Grouped Boxplots ---
    if violin_df is not None and not violin_df.empty:
        for (metric_name, model), x_pos in x_positions_dict.items():
            model_data = violin_df[(violin_df['metric'] == metric_name) & (violin_df['model'] == model)]
            point_data = point_df[(point_df['metric'] == metric_name) & (point_df['model'] == model)]
            point = point_data.iloc[0]
            if not model_data.empty:
                box = ax.boxplot(model_data['value'], positions=[x_pos],
                                 widths=0.1, patch_artist=True, showfliers=True)
                for patch in box['boxes']:
                    patch.set_facecolor(model_colors.get(model, 'gray'))
                    patch.set_edgecolor('black')
                    patch.set_alpha(0.7)
                for median in box['medians']:
                    media_loc = np.zeros((2, 2))
                    media_loc[0,0] = x_pos - 0.05
                    media_loc[1,0] = point['value']
                    media_loc[0,1] = x_pos + 0.05
                    media_loc[1,1] = point['value']
                    median.set_data(media_loc)
                    median.set(color='black', linewidth=2)
                for i, whisker in enumerate(box['whiskers']):
                    whisker_data = whisker.get_xydata().T
                    if i == 0:
                        whisker_data[1,1] = point['ci_lower']
                    else:
                        whisker_data[1,1] = point['ci_upper']
                    whisker.set_data(whisker_data)
                    whisker.set(color='black', linewidth=1.5, linestyle='--')
                for i, cap in enumerate(box['caps']):
                    cap_data = cap.get_xydata().T
                    if i == 0:
                        cap_data[1,0] = point['ci_lower']
                        cap_data[1,1] = point['ci_lower']
                    else:
                        cap_data[1,0] = point['ci_upper']
                        cap_data[1,1] = point['ci_upper']
                    cap.set_data(cap_data)
                    cap.set(color='black', linewidth=1.5)
                for flier in box['fliers']:
                    flier.set(marker='', color='black', alpha=0.5, markersize=3)

    # --- Draw Error Bars and Points ---
    for (metric_name, model), x_pos in x_positions_dict.items():
        point_data = point_df[(point_df['metric'] == metric_name) & (point_df['model'] == model)]
        if not point_data.empty:
            point = point_data.iloc[0]
            # error = [[point['value'] - point['ci_lower']], [point['ci_upper'] - point['value']]]
            # ax.errorbar(x=x_pos, y=point['value'], yerr=error, color=model_colors.get(model, 'black'),
            #             fmt='D', capsize=5, capthick=2, elinewidth=2, markersize=6, zorder=10)
            ax.text(x_pos, point['ci_upper'] + 0.01,
                    f"{point['value']:.3f}\n[{point['ci_lower']:.2f}, {point['ci_upper']:.2f}]",
                    ha='center', va='bottom', fontsize=FONT_SIZES['tick_label'], color=model_colors.get(model, 'black'),
                    zorder=11)
            # ax.text(x_pos, point['ci_upper'] + 0.01,
            #         f"{point['value']:.3f}",
            #         ha='center', va='bottom', fontsize=FONT_SIZES['tick_label'], color=model_colors.get(model, 'black'),
            #         zorder=11)


    # --- Finalize Plot Formatting ---
    ax.grid(visible=False, axis='both', which='both')
    # for i in range(len(metrics) - 1):
    #     ax.axvline(x=i + 0.5, color='#E6E6E6', linestyle='--', linewidth=1, zorder=-1)
    ax.set_xticks(range(len(metrics)))
    ax.set_xticklabels(metrics, fontsize=FONT_SIZES['tick_label'])
    ax.set_title(title, fontsize=FONT_SIZES['title'], pad=20, weight='bold')
    ax.set_ylabel('Metric Value (with 95% CI)', fontsize=FONT_SIZES['axis_label'])
    ax.set_ylim(bottom=ylim[0], top=ylim[1])
    ax.tick_params(axis='y', labelsize=FONT_SIZES['tick_label'])

    # Build a legend based on models that were actually plotted
    legend_handles = [Line2D([0], [0], marker='D', color='w',
                            markerfacecolor=color, markersize=12) for model, color in model_colors.items()]
    ax.legend(handles=legend_handles, labels=model_colors.keys(), loc='lower left', ncol=1, fontsize=FONT_SIZES['legend'])
    plt.tight_layout()

    # --- Save Figure ---
    try:
        plt.savefig(os.path.join(output_path, f'{task}.png'), dpi=600, bbox_inches='tight')
    except Exception as e:
        print(f"Error saving figure: {e}")
    plt.close('all')

    return



if __name__ == "__main__":

    # Imports
    import argparse
    import json

    # CLI
    parser = argparse.ArgumentParser(description='Explainable Multimodal Prediction of Breast Cancer Tumor and Immune Phenotypes from Histopathology.')
    parser.add_argument('--config', type=str, help='Path to config file.', default='src/config/figures/model_checkpoints.json')
    parser.add_argument('--output_path', type=str, help='Path to save the output plots.', default='results/manuscript/figures')
    parser.add_argument('--metric', type=str, help='List of metrics to plot.', choices=['AUC', 'PCC'], required=True)
    args = parser.parse_args()



    # Create output directory if it doesn't exist and add table to the output path
    output_path = os.path.join(args.output_path, "v2", args.metric)
    os.makedirs(output_path, exist_ok=True)

    # Models based on Table
    models = MODELS_LIST
    # print(models)

    # Metrics
    metrics = [args.metric]
    metrics_dict = {idx:name for idx, name in enumerate(metrics)}
    # print(metrics_dict)


    # Tasks
    all_tasks = [
        'Angiogenesis', # 0
        'Antigen Processing and Presentation', # 1
        'B-cell Proliferation', # 2
        'Cell Cycle', # 3
        'Epithelial Mesenchymal Transition', # 4
        'Fatty Acid Metabolism', # 5
        'Glycolysis', # 6
        'Immunosuppression', # 7
        'Oxidative Phosphorylation', # 8
        'T-cell Mediated Cytotoxicity', # 9
        'Average Across Tasks',
        'Average Across Immune Tasks',
        'Average Across Metabolic Tasks',
        'Average Across Tumor Tasks'
    ]

    # Indices of immune-related tasks in the tasks list
    immune_task_indices = np.array([1, 2, 7, 9])

    # Indices of metabolic-related tasks in the tasks list
    metabolic_task_indices = np.array([5, 6, 8])

    # Indices of tumor-related tasks in the tasks list
    tumor_task_indices = np.array([0, 1, 3, 4])


    # Open config file
    with open(args.config, 'r') as f:
        config = json.load(f)

    # Let's build our task dataframe and violin dataframe
    # We will use these to create the all plots without extra .XLSX or .CSV
    # Task
    task = list()
    model = list()
    metric = list()
    value = list()
    ci_lower = list()
    ci_upper = list()

    # Violin
    v_task = list()
    v_model = list()
    v_metric = list()
    v_value = list()

    for task_idx, task_ in enumerate(list(config.keys())):
        for model_name in config[task_].keys():

            # Open checkpoint path
            ckpt_path = config[task_][model_name]

            if model_name in models:
                bootstrap_metrics = pd.read_csv(os.path.join(ckpt_path, "clinical-subtype-analysis/results", f"test_{SUBTYPES_TABLE_DICT[model_name]}_bootstrap_metrics_kf0.csv"))
                bootstrap_results_auc = pd.read_csv(os.path.join(ckpt_path, "clinical-subtype-analysis/results", f"test_{SUBTYPES_TABLE_DICT[model_name]}_bootstrap_results_auc_kf0.csv"))
                bootstrap_results_pcc = pd.read_csv(os.path.join(ckpt_path, "clinical-subtype-analysis/results", f"test_{SUBTYPES_TABLE_DICT[model_name]}_bootstrap_results_pcc_kf0.csv"))

                # All Metrics
                task += [task_] * 1
                model += [model_name] * 1

                # Metric data
                if metrics[0] == 'AUC':
                    metric += ['AUC']
                    value += [bootstrap_metrics['auc'].values[0]]
                    ci_lower += [bootstrap_metrics['auc_lower'].values[0]]
                    ci_upper += [bootstrap_metrics['auc_upper'].values[0]]
                elif metrics[0] == 'PCC':
                    metric += ['PCC']
                    value += [bootstrap_metrics['pcc'].values[0]]
                    ci_lower += [bootstrap_metrics['pcc_lower'].values[0]]
                    ci_upper += [bootstrap_metrics['pcc_upper'].values[0]]

                # Violin data
                if metrics[0] == 'AUC':
                    v_task += [task_] * len(bootstrap_results_auc)
                    v_model += [model_name] * len(bootstrap_results_auc)
                    v_metric += ['AUC'] * len(bootstrap_results_auc)
                    v_value += list(bootstrap_results_auc['results'].values)

                elif metrics[0] == 'PCC':
                    v_task += [task_] * len(bootstrap_results_pcc)
                    v_model += [model_name] * len(bootstrap_results_pcc)
                    v_metric += ['PCC'] * len(bootstrap_results_pcc)
                    v_value += list(bootstrap_results_pcc['results'].values)


    # Create DataFrames
    all_tasks_df = pd.DataFrame({
        'task':task,
        'model':model,
        'metric':metric,
        'value':value,
        'ci_lower':ci_lower,
        'ci_upper':ci_upper
    })
    # print(table_df.head())
    # print(table_df.shape)


    all_tasks_violin_df = pd.DataFrame({
        'task':v_task,
        'model':v_model,
        'metric':v_metric,
        'value':v_value
    })
    # print(violin_df.head())
    # print(violin_df.shape)



    # For each task, create the plots
    for task in all_tasks:

        # Task
        if task not in ('Average Across Tasks', 'Average Across Immune Tasks', 'Average Across Metabolic Tasks', 'Average Across Tumor Tasks'):
            task_mask = (all_tasks_df['task'] == task)
            task_df = all_tasks_df[task_mask]
        else:
            task_df = get_avg_task_df(all_tasks_df, all_tasks, task, models, metrics, immune_task_indices, metabolic_task_indices, tumor_task_indices)

        # Violin
        violin_mask = (all_tasks_violin_df['task'] == task)
        violin_df = all_tasks_violin_df[violin_mask]




        # Create violin_dict for Average Across Tasks
        if task == 'Average Across Tasks':
            violin_df = pd.DataFrame.from_dict(violin_dict_avg)
        elif task == 'Average Across Immune Tasks':
            violin_df = pd.DataFrame.from_dict(violin_dict_avg_immune)
        elif task == 'Average Across Metabolic Tasks':
            violin_df = pd.DataFrame.from_dict(violin_dict_avg_metabolic)
        elif task == 'Average Across Tumor Tasks':
            violin_df = pd.DataFrame.from_dict(violin_dict_avg_tumor)


        # Start creating violin_dict for Average Across Tasks
        if task == 'Angiogenesis':
            violin_dict = {
                'task':list(),
                'model':list(),
                'metric':list(),
                'value':list()
            }
            task_ = violin_df['task'].values
            model = violin_df['model'].values
            metric = violin_df['metric'].values
            value = violin_df['value'].values
            violin_dict['task'].extend(task_)
            violin_dict['model'].extend(model)
            violin_dict['metric'].extend(metric)
            violin_dict['value'].extend(value)

        elif task == 'T-cell Mediated Cytotoxicity':
            task_ = violin_df['task'].values
            model = violin_df['model'].values
            metric = violin_df['metric'].values
            value = violin_df['value'].values
            violin_dict['task'].extend(task_)
            violin_dict['model'].extend(model)
            violin_dict['metric'].extend(metric)
            violin_dict['value'].extend(value)

            # Now, it's time to build the matrices with the average values
            violin_df_raw = pd.DataFrame(violin_dict)
            # print(violin_df_raw.head())

            metrics_matrix = np.zeros((1000, 10, len(models), len(metrics)))
            for t_id, t_n in enumerate(all_tasks):
                if t_n in ('Average Across Tasks', 'Average Across Immune Tasks', 'Average Across Metabolic Tasks', 'Average Across Tumor Tasks'):
                    continue
                else:
                    t_mask = (violin_df_raw['task'] == t_n)
                    for m_id, m_n in enumerate(models):
                        m_mask = (violin_df_raw['model'] == m_n)
                        for metric_id, metric_name in metrics_dict.items():
                            # print(metric_id, metric_name)
                            metric_mask = (violin_df_raw['metric'] == metric_name)
                            filtered_data = violin_df_raw[t_mask & m_mask & metric_mask]
                            filtered_data = filtered_data.values
                            # print(filtered_data.shape)
                            # print(filtered_data)
                            if filtered_data.shape[0] == 1000:
                                metrics_matrix[:, t_id, m_id, metric_id] = filtered_data[:, 3]
            # print(metrics_matrix.shape)
            # print(metrics_matrix)


            # Create a dictionary for Average Across Tasks
            violin_dict_avg = {
                'task': list(),
                'model': list(),
                'metric': list(),
                'value': list()
            }
            violin_dict_avg['task'].extend(['Average Across Tasks'] * len(models) * len(metrics) * 1000)

            # Create a dictionary for Average Across Immune Tasks
            violin_dict_avg_immune = {
                'task': list(),
                'model': list(),
                'metric': list(),
                'value': list()
            }
            violin_dict_avg_immune['task'].extend(['Average Across Immune Tasks'] * len(models) * len(metrics) * 1000)


            # Create a dictionary for Average Across Metabolic Tasks
            violin_dict_avg_metabolic = {
                'task': list(),
                'model': list(),
                'metric': list(),
                'value': list()
            }
            violin_dict_avg_metabolic['task'].extend(['Average Across Metabolic Tasks'] * len(models) * len(metrics) * 1000)


            # Create a dictionary for Average Across Tumor Tasks
            violin_dict_avg_tumor = {
                'task': list(),
                'model': list(),
                'metric': list(),
                'value': list()
            }
            violin_dict_avg_tumor['task'].extend(['Average Across Tumor Tasks'] * len(models) * len(metrics) * 1000)


            # Models
            for m_idx, m_name in enumerate(models):

                # Average Across Tasks
                violin_dict_avg['model'].extend([m_name] * len(metrics) * 1000)
                if metric[0] == 'AUC':
                    auc_ = metrics_matrix[:, :, m_idx, 0]
                    auc_avg = compute_avg_act(auc_)
                    violin_dict_avg['metric'].extend(['AUC'] * 1000)
                    violin_dict_avg['value'].extend(auc_avg)
                elif metric[0] == 'PCC':
                    pcc_ = metrics_matrix[:, :, m_idx, 0]
                    pcc_avg = compute_avg_act(pcc_)
                    violin_dict_avg['metric'].extend(['PCC'] * 1000)
                    violin_dict_avg['value'].extend(pcc_avg)

                # Average Across Immune Tasks
                violin_dict_avg_immune['model'].extend([m_name] * len(metrics) * 1000)
                if metric[0] == 'AUC':
                    auc_immune = metrics_matrix[:, immune_task_indices, m_idx, 0]
                    auc_immune_avg = compute_avg_act(auc_immune)
                    violin_dict_avg_immune['metric'].extend(['AUC'] * 1000)
                    violin_dict_avg_immune['value'].extend(auc_immune_avg)

                elif metric[0] == 'PCC':
                    pcc_immune = metrics_matrix[:, immune_task_indices, m_idx, 0]
                    pcc_immune_avg = compute_avg_act(pcc_immune)
                    violin_dict_avg_immune['metric'].extend(['PCC'] * 1000)
                    violin_dict_avg_immune['value'].extend(pcc_immune_avg)


                # Average Across Metabolic Tasks
                violin_dict_avg_metabolic['model'].extend([m_name] * len(metrics) * 1000)
                if metric[0] == 'AUC':
                    auc_metabolic = metrics_matrix[:, metabolic_task_indices, m_idx, 0]
                    auc_metabolic_avg = compute_avg_act(auc_metabolic)
                    violin_dict_avg_metabolic['metric'].extend(['AUC'] * 1000)
                    violin_dict_avg_metabolic['value'].extend(auc_metabolic_avg)

                elif metric[0] == 'PCC':
                    pcc_metabolic = metrics_matrix[:, metabolic_task_indices, m_idx, 0]
                    pcc_metabolic_avg = compute_avg_act(pcc_metabolic)
                    violin_dict_avg_metabolic['metric'].extend(['PCC'] * 1000)
                    violin_dict_avg_metabolic['value'].extend(pcc_metabolic_avg)


                # Average Across Tumor Tasks
                violin_dict_avg_tumor['model'].extend([m_name] * len(metrics) * 1000)
                if metric[0] == 'AUC':
                    auc_tumor = metrics_matrix[:, tumor_task_indices, m_idx, 0]
                    auc_tumor_avg = compute_avg_act(auc_tumor)
                    violin_dict_avg_tumor['metric'].extend(['AUC'] * 1000)
                    violin_dict_avg_tumor['value'].extend(auc_tumor_avg)

                elif metric[0] == 'PCC':
                    pcc_tumor = metrics_matrix[:, tumor_task_indices, m_idx, 0]
                    pcc_tumor_avg = compute_avg_act(pcc_tumor)
                    violin_dict_avg_tumor['metric'].extend(['PCC'] * 1000)
                    violin_dict_avg_tumor['value'].extend(pcc_tumor_avg)

        else:
            task_ = violin_df['task'].values
            model = violin_df['model'].values
            metric = violin_df['metric'].values
            value = violin_df['value'].values
            violin_dict['task'].extend(task_)
            violin_dict['model'].extend(model)
            violin_dict['metric'].extend(metric)
            violin_dict['value'].extend(value)


        if metrics[0] == 'AUC':
            ylim = (-0.2, 1.2)
        else:
            ylim = (-1.25, 1.2)


        # Drop nan values from task_df and violin_df
        violin_df = violin_df.dropna(subset=['value'])

        plot_comparison_boxplots(
            point_df=task_df,
            violin_df=violin_df,
            models=models,
            metrics=metrics,
            task=task,
            title=task,
            ylim=ylim,
            output_path=output_path
        )
