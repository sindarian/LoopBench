import json
from collections import defaultdict, Counter

import matplotlib.pyplot as plt
import seaborn as sns
import os
import logging
import math
import tensorflow
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D

from matplotlib.offsetbox import AnchoredText
from scipy.sparse import load_npz

from util.constants import PLOT_DIR, METRICS_DIR, OUTPUT_DIR, PATCH_SIZE, RESOLUTION
from util.logger import Logger
import numpy as np
import tensorflow as tf

from util.utils import compute_pixel_distance

LOGGER = Logger(name='plotting', level=logging.DEBUG).get_logger()

def load_histories(metric_dir):
    histories = []
    model_names = []
    patch_sizes = []
    resolutions = []

    files = [f for f in os.listdir(metric_dir) if os.path.isfile(os.path.join(metric_dir, f)) and f.endswith('.json')]
    for file in files:
        path = os.path.join(metric_dir, file)
        with open(path, 'r') as f:
            content = f.read()
            history = json.loads(content)
            histories.append(history)
        model_names.append(file.split('_')[0])
    return histories, model_names, patch_sizes, resolutions

def plot_training_history(metric_dir=os.path.join(OUTPUT_DIR, METRICS_DIR), title='Training Metrics Comparison'):
    histories, labels, _, _ = load_histories(metric_dir)
    colors = [
        'tab:blue',
        'tab:orange',
        'tab:green',
        'tab:red',
        'tab:purple',
        'tab:brown',
        'tab:pink',
        'tab:gray',
        'tab:olive',
        'tab:cyan',
        'gold',
        'darkblue',
        'limegreen',
        'crimson',
        'teal'
    ]

    # Extract metric pairs
    all_metrics = list(histories[0].keys())
    paired_metrics = [(m, f"val_{m}") for m in all_metrics if not m.startswith('val')]

    n_plots = len(paired_metrics)
    rows = math.ceil(n_plots / 2)
    fig, axes = plt.subplots(rows, 2, figsize=(14, rows * 5))
    axes = axes.flatten()

    for i, (train_metric, val_metric) in enumerate(paired_metrics):
        ax = axes[i]
        for model_idx, history in enumerate(histories):
            color = colors[model_idx]
            label_prefix = labels[model_idx]

            # Plot training metric
            if train_metric in history:
                ax.plot(history[train_metric], linestyle='-', color=color,
                        label=f"{label_prefix} {train_metric}")
            # Plot validation metric
            if val_metric in history:
                ax.plot(history[val_metric], linestyle=':', color=color,
                        label=f"{label_prefix} {val_metric}")

        ax.set_title(train_metric)
        ax.set_xlabel('Epoch')
        ax.set_ylabel(train_metric)
        ax.grid(True)
        ax.legend()

    # Turn off unused subplots if any
    for j in range(len(paired_metrics), len(axes)):
        axes[j].axis('off')

    plt.suptitle(title, fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.97])

    text = (f'Patch Size: {PATCH_SIZE}'
            f'\nResolution: {RESOLUTION}')

    fig.text(0.7, .97,
             text,
             va='bottom', ha='left', fontsize=10,
             bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="black"))
    _save_plot('training_evaluation.png')
    plt.show()

def _group_histories_by_patch(histories, model_names, patch_sizes):
    """Organize histories by patch size with labels."""
    data_map = defaultdict(list)
    for i, hist in enumerate(histories):
        label = f'{model_names[i]}'
        ps = patch_sizes[i]
        data_map[ps].append((label, hist))
    return data_map

def _group_histories_by_resolution(histories, model_names, resolutions):
    """Organize histories by patch size with labels."""
    data_map = defaultdict(list)
    for i, hist in enumerate(histories):
        label = f'{model_names[i]}'
        res = resolutions[i]
        data_map[res].append((label, hist))
    return data_map

def _assign_colors(model_names):
    """Assign a consistent color per model from matplotlib default cycle."""
    color_cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']
    model_color_map = {}
    for name in model_names:
        model = name.split()[0].lower()
        if model not in model_color_map:
            model_color_map[model] = color_cycle[len(model_color_map) % len(color_cycle)]
    return model_color_map

def _get_color(label, color_map):
    """Get color for a label based on its model name prefix."""
    return color_map.get(label, 'gray')

def _create_legend_handles(color_map):
    """Create legend handles for LoopNet, CNN and line styles."""
    orange_line = Line2D([], [], color=color_map.get('loopnet', 'orange'), linestyle='-', label='LoopNet')
    blue_line = Line2D([], [], color=color_map.get('cnn', 'blue'), linestyle='-', label='CNN')
    solid_line = Line2D([], [], color='black', linestyle='-', label='Training')
    dashed_line = Line2D([], [], color='black', linestyle='--', label='Validation')
    return [orange_line, blue_line, solid_line, dashed_line]

def _plot_subplot(ax,
                  metric,
                  metric_label,
                  col_name,
                  data_list,
                  color_map,
                  figure_type,
                  is_first_row,
                  is_first_col,
                  is_last_row):
    ax.grid(True)

    # set the title for each column
    if is_first_row:
        ax.set_title(f'{figure_type} {col_name}', fontsize=13)

    # set the y axis label for each row
    if is_first_col:
        ax.set_ylabel(metric_label, fontsize=12)

    for label, hist in data_list:
        color = _get_color(label, color_map)
        if metric in hist:
            ax.plot(hist[metric], color=color, linestyle='-')
        if f"val_{metric}" in hist:
            ax.plot(hist[f"val_{metric}"], color=color, linestyle='--')

    # set the x axis label for each column
    if is_last_row:
        ax.set_xlabel("Epoch", fontsize=12)

def plot_training_history_all_data_splits(metric_dir: str):
    histories, model_names, patch_sizes, resolutions = load_histories(metric_dir)

    metrics = ['loss', 'ROC_AUC', 'PR_AUC']
    metric_labels = ['Loss', 'ROC AUC', 'PR AUC']

    if len(set(patch_sizes)) != 1:
        unique_column_names = sorted(set(patch_sizes))
        data_map = _group_histories_by_patch(histories, model_names, patch_sizes)
        figure_type = 'Patch Size'
    else:
        unique_column_names = sorted(set(resolutions))
        data_map = _group_histories_by_resolution(histories, model_names, resolutions)
        figure_type = 'Resolution'

    color_map = _assign_colors(model_names)

    # the figure will be a metric x patch_size grid
    n_rows, n_cols = len(metrics), len(unique_column_names)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows), sharex=True)

    # flatten the axes
    if n_rows == 1: axes = [axes]
    if n_cols == 1: axes = [[ax] for ax in axes]

    # plot the data
    for row_idx, (metric, metric_label) in enumerate(zip(metrics, metric_labels)):
        for col_idx, col_name in enumerate(unique_column_names):
            ax = axes[row_idx][col_idx]
            _plot_subplot(
                ax,
                metric,
                metric_label,
                col_name,
                data_map.get(col_name, []),
                color_map,
                figure_type,
                is_first_row=(row_idx == 0),
                is_first_col=(col_idx == 0),
                is_last_row=(row_idx == n_rows - 1),
            )

    # set up the legend
    legend_handles = _create_legend_handles(color_map)
    fig.legend(
        handles=legend_handles,
        loc='center right',
        bbox_to_anchor=(0.95, 0.5),
        fontsize=18
    )

    fig.tight_layout(rect=[0, 0, 0.85, 0.95])
    fig.suptitle(f'Training Metrics by {figure_type} and Model', fontsize=16)

    # show and save
    plt.show()
    fname = figure_type.lower().replace(' ', '_')
    _save_plot(f'model_performance_over_{fname}.png')
    plt.close()

def plot_pixel_counts(data,
                      title: str = 'Default Title',
                      limit: int = None,
                      plot_neg: bool = False,
                      patch_size: int = 224,
                      resolution: int = 10000):
    data_size = len(data if not limit else data[:limit])
    indices = np.arange(data_size)
    positives = [t[0] for t in data[:data_size]]

    plt.figure(figsize=(12, 6))
    plt.bar(indices, positives, color='green', label='Positives')
    if plot_neg:
        negatives = [t[1] for t in data[:data_size]]
        plt.bar(indices, negatives, bottom=positives, color='red', alpha=0.5, label='Negatives')

    plt.xlabel('Patch')
    plt.ylabel('Number of Pixels')
    plt.title(f'Pixel Labels per Patch for {title}')
    plt.text(1.05, 1.0,
             f'Max pixel count: {patch_size * patch_size} pixels\nPatch Size: {patch_size}\nResolution: {resolution}',
             transform=plt.gca().transAxes, va='top', ha='left', fontsize=10,
             bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="black"))
    plt.legend()
    plt.tight_layout()
    plt.grid(True)
    plt.show()
    _save_plot(f'pixel_counts_ps{patch_size}_res{resolution}.png')
    plt.close()

def plot_top_positive_patches(train_counts,
                              top_n: int = 4,
                              patch_size: int = 224,
                              resolution: int = 10000,
                              data_to_plot: str = 'x'):
    """
    Plots heatmaps of the top `top_n` Hi-C matrices with the highest positive pixel count
    in a 2x2 grid. No colorbar is displayed.

    Args:
        train_counts (List[Tuple[int, int]]): List of (pos_count, neg_count) tuples.
        generator (HiCDatasetGenerator): A tf.keras.utils.Sequence generator.
        top_n (int): Number of top images to visualize (default: 4).
    """
    # Custom white-orange-red colormap
    white_orange_red = LinearSegmentedColormap.from_list(
        "white_orange_red", ["white", "orange", "red"]
    )

    # Get indices of top N images based on positive count
    top_pos = sorted(train_counts,
                         key=lambda i: i[0],
                         reverse=True)[:top_n]

    fig, axes = plt.subplots(2, 2, figsize=(12, 12))
    fig.suptitle(f'Top {top_n} positive patches\nPatch Size: {patch_size}\nResolution: {resolution}')
    axes = axes.flatten()

    for i, data in enumerate(top_pos):
        if data_to_plot == 'x':
            matrix = data[2][0] # index as [0] to remove the batch dimension
            # matrix = tf.squeeze(matrix)
        else:
            matrix = tensorflow.reshape(data[3][0], (patch_size,patch_size))
        ax = axes[i]

        sns.heatmap(matrix,
                    cmap=white_orange_red,
                    square=True,
                    cbar=False,
                    ax=ax)

        # Black border
        for _, spine in ax.spines.items():
            spine.set_visible(True)
            spine.set_linewidth(2)
            spine.set_color('black')

        ax.set_title('Highest-Scoring Patch' if i == 0 else f'{_ordinal(i + 1)} Highest-Scoring Patch')
        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.set_xticks([])
        ax.set_yticks([])

    # Hide any unused subplots
    for j in range(top_n, len(axes)):
        axes[j].axis('off')

    plt.tight_layout()
    plt.show()
    _save_plot(f'top_{top_n}_interaction_heatmaps_{data_to_plot}_ps{patch_size}_res{resolution}.png')
    plt.close()

def _ordinal(n):
    return f"{n}{'tsnrhtdd'[(n//10%10!=1)*(n%10<4)*n%10::4]}"

def _save_plot(file_name: str):
    plt.savefig(os.path.join(OUTPUT_DIR, PLOT_DIR, file_name))

def plot_coordinate_scatter(start_tuples, patch_size, mb, filename):
    """
    Create a simple scatter plot of coordinate points.

    Parameters:
    -----------
    start_tuples : list of tuples
        List of (x, y) coordinates to plot as points
    filename : str
        Output filename for the plot
    """

    # Extract x and y coordinates
    x_coords = [coord[0] for coord in start_tuples]
    y_coords = [coord[1] for coord in start_tuples]

    # Create scatter plot
    plt.figure(figsize=(10, 10))
    plt.scatter(x_coords, y_coords)
    # plt.gca().invert_yaxis()
    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')
    plt.title(f'Sampled Patches')
    plt.grid(True, alpha=0.3)
    # plt.axis('equal')

    info_text = f'Total Points: {len(start_tuples)}\nPatch Size: {patch_size}x{patch_size}\nMB: {mb}'
    plt.text(0.02, 0.98, info_text, transform=plt.gca().transAxes,
             verticalalignment='bottom', horizontalalignment='left', fontsize=10,
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    plt.gca().invert_yaxis()

    # Save and show
    plt.tight_layout()
    _save_plot(filename)
    plt.show()

def plot_actual_vs_sampled_patch(matrices, title, plot_titles, fname):
    fig_heat, axes_heat = plt.subplots(1, 2, figsize=(16,10))
    axes_heat = axes_heat.flatten()

    for idx, (matrix, plot_title) in enumerate(zip(matrices, plot_titles)):
        # Generate heatmap on subplot
        ax = axes_heat[idx]
        im = ax.imshow(matrix, cmap='hot')
        plt.colorbar(im, ax=ax)
        ax.set_title(plot_title)
        ax.set_xticks([])
        ax.set_yticks([])


    # Format and show heatmap figure
    fig_heat.suptitle(title, fontsize=16)
    fig_heat.tight_layout()
    fig_heat.show()
    fig_heat.savefig(os.path.join(OUTPUT_DIR, PLOT_DIR, 'patch_diag_investigation', fname))

def _plot_pixel_accuracies(matrices, titles, color_bar_labels, fname, model_name, nrows=2, ncols=2, hmap_size=(12, 10)):
    fig_heat, axes_heat = plt.subplots(nrows, ncols, figsize=hmap_size)
    axes_heat = axes_heat.flatten() if nrows > 1 else [axes_heat]

    for idx, (matrix, title, bar_label) in enumerate(zip(matrices, titles, color_bar_labels)):
        # Generate heatmap on subplot
        _plot_heatmap_on_axis(matrix, axes_heat[idx], title, bar_label)

    # Format and show heatmap figure
    fig_heat.suptitle(f'Evaluation of {model_name} Predictions', fontsize=16)
    fig_heat.tight_layout()
    fig_heat.show()
    fig_heat.savefig(os.path.join(OUTPUT_DIR, PLOT_DIR, 'distance_accuracy_investigation', fname))

def _plot_distance_plots(matrices, titles, ylabel, fname, model_name, nrows=2, ncols=2, lplot_size=(16, 12)):
    fig_line, axes_line = plt.subplots(nrows, ncols, figsize=lplot_size)
    axes_line = axes_line.flatten() if nrows > 1 else [axes_line]

    for idx, (matrix, title) in enumerate(zip(matrices, titles)):

        # Compute distance results
        distance_results = compute_pixel_distance(matrix)

        # Generate line plot on subplot
        _plot_distances_on_axis(distance_results, axes_line[idx], title, ylabel)

    # Format and show line plot figure
    fig_line.suptitle(f'Evaluation of {model_name} Predictions', fontsize=16)
    fig_line.tight_layout()
    fig_line.show()
    fig_line.savefig(os.path.join(OUTPUT_DIR, PLOT_DIR, 'distance_accuracy_investigation', fname))

def generate_plots(count_matrix, tp, fp, tn, fn, model_name):
    _plot_pixel_accuracies(matrices=[count_matrix],
                           titles=['Avg Pixel Accuracy per Positive Prediction'],
                           color_bar_labels=['Average Accuracy'],
                           fname=f'{model_name}_pixel_acc_by_pixel.png',
                           model_name=model_name,
                           nrows=1, ncols=1, hmap_size=(6, 6))
    _plot_pixel_accuracies(matrices=[tp, fp, tn, fn],
                           titles=['Avg True Positive Rate per Pixel',
                                  'Avg False Positive Rate per Pixel',
                                  'Avg True Negative Rate per Pixel',
                                  'Avg False Negative Rate per Pixel'],
                           color_bar_labels=['Avg True Positive Rate',
                                            'Avg False Positive Rate',
                                            'Avg True Negative Rate',
                                            'Avg False Negative Rate'],
                           fname=f'{model_name}_pn_rate_by_pixel.png',
                           model_name=model_name)

    _plot_distance_plots(matrices=[count_matrix],
                         titles=['Avg Accuracy by Distance'],
                         ylabel='Accuracy',
                         fname=f'{model_name}_acc_by_distance.png',
                         model_name=model_name,
                         nrows=1, ncols=1, lplot_size=(12, 8))
    _plot_distance_plots(matrices=[tp, fp, tn, fn],
                         titles=['Avg True Positive Rate by Distance',
                                  'Avg False Positive Rate by Distance',
                                  'Avg True Negative Rate by Distance',
                                  'Avg False Negative Rate by Distance'],
                         ylabel='Rate',
                         fname=f'{model_name}_pn_rate_by_distance.png',
                         model_name=model_name)

def _plot_heatmap_on_axis(dist_matx, ax, title, bar_label):
    # heatmap
    im = ax.imshow(dist_matx, cmap='hot')

    # colorbar
    plt.colorbar(im, label=bar_label, ax=ax)

    # title
    ax.set_title(title)

    # remove the axis ticks
    ax.set_xticks([])
    ax.set_yticks([])


def _plot_distances_on_axis(distance_results, ax, title, ylabel):
    for metric in distance_results.keys():
        unsorted_udists = distance_results[metric]['udists']
        unsorted_uaccuracies = distance_results[metric]['uaccuracies']

        sorted_data = sorted(zip(unsorted_udists, unsorted_uaccuracies))

        ax.plot([point[0] for point in sorted_data], [point[1] for point in sorted_data], label=metric)

    ax.set_title(title)
    ax.set_xlabel('Distance')
    ax.set_ylabel(ylabel)
    ax.legend()
    ax.grid(True, alpha=0.3)


def plot_chromosome_labels(chromosomes, dataset_dir:str = 'dataset/hela_100', use_original: bool = True):
    counts = []

    for chrom in chromosomes:
        if use_original:
            labels = np.load(os.path.join(dataset_dir, f'labels.{chrom}.npy'))
        else:
            labels = load_npz(os.path.join(dataset_dir, f'{chrom}_ground_truth.npz')).toarray()

        counts.append(np.sum(labels))

    plt.bar(chromosomes, counts)
    plt.xlabel('Chromosome')
    plt.ylabel('Positive Labels')
    plt.title('Positive Labels by Chromosome')
    plt.show()

def plot_diagonal_distance_histogram(matrix, title: str):
    matrix = np.array(matrix, dtype=bool)

    # Get indices of all True values at once
    rows, cols = np.where(matrix)

    distances = cols - rows

    c = Counter(distances)
    c = sorted(c.items())
    x = [x[0] for x in c]
    y = [x[1] for x in c]

    # Create side-by-side plots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    fig.suptitle(title, fontsize=16)

    # Bar plot
    ax1.bar(x, y, alpha=0.7, color='lightcoral', edgecolor='black')
    ax1.set_xlabel('Distance from Main Diagonal in Pixels')
    ax1.set_ylabel('Number of True Values')
    ax1.grid(True, alpha=0.3)

    # Violin plot
    ax2.violinplot([distances], positions=[0], widths=0.8)
    ax2.set_xlabel('Distribution')
    ax2.set_ylabel('Distance from Main Diagonal')
    ax2.set_xticks([0])
    ax2.set_xticklabels(['Distances'])
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

    fname = title.replace(' ', '_')
    fname = fname.replace(':', '_')
    fig.savefig(os.path.join(OUTPUT_DIR, PLOT_DIR, 'distance_accuracy_investigation', 'chrom_pos_distances', f'{fname}.png'))

def plot_heatmap(matrix, title="Heatmap", cmap='hot', figsize=(8, 6)):
    """
    Plot a heatmap for a 2D matrix.

    Args:
        matrix: 2D numpy array or list
        title: Plot title
        cmap: Colormap (e.g., 'viridis', 'hot', 'coolwarm', 'plasma')
        figsize: Figure size as (width, height)
    """
    plt.figure(figsize=figsize)
    plt.imshow(matrix, cmap=cmap, aspect='auto')
    plt.colorbar()
    plt.title(title)
    plt.show()

def plot_raw_crhom(contacts, title: str = 'Raw Chromosome Interactions'):
    plt.hist(contacts, edgecolor="black", log=True)
    plt.xlabel("Contact Value")
    plt.ylabel("Frequency (log scale)")
    plt.title("Distribution of Contact Values")
    plt.show()
