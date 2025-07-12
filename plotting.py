import json
import matplotlib.pyplot as plt
import os
import logging
from logger import Logger
import numpy as np

LOGGER = Logger(name='plotting', level=logging.DEBUG).get_logger()

PLOT_DIR = 'plots/'

def load_histories(metric_dir):
    histories = []
    model_names = []
    files = [f for f in os.listdir(metric_dir) if os.path.isfile(os.path.join(metric_dir, f)) and f.endswith('.json')]
    for file in files:
        path = os.path.join(metric_dir, file)
        with open(path, 'r') as f:
            content = f.read()
            history = json.loads(content)
            histories.append(history)
        model_names.append(file.split('_')[0])
    return histories, model_names

def plot_training_history(metric_dir):
    histories, labels = load_histories(metric_dir)
    # labels = ['efn', 'res']
    colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red']

    # Extract metric pairs
    all_metrics = list(histories[0].keys())
    paired_metrics = [(m, f"val_{m}") for m in all_metrics if not m.startswith('val')]

    import math

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

    plt.suptitle('Training Metrics Comparison', fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.97])
    plt.savefig(PLOT_DIR + 'training_evaluation.png')
    plt.show()

def plot_pixel_counts(data, title, limit=10, plot_neg=False):
    print(len(data))
    positives = [t[0] for t in data[:limit]]
    indices = np.arange(len(positives))
    print(len(positives))
    print(len(indices))

    plt.figure(figsize=(12, 6))
    plt.bar(indices, positives, color='green', label='Positives')
    if plot_neg:
        negatives = [t[1] for t in data[:limit]]
        plt.bar(indices, negatives, bottom=positives, color='red', alpha=0.5, label='Negatives')

    plt.xlabel('Image')
    plt.ylabel('Pixel Counts')
    plt.title(f'Pixel Counts per Image (Positives and Negatives) for {title}')
    plt.legend()
    plt.tight_layout()
    plt.show()