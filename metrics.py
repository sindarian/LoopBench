import keras.backend
import tensorflow as tf
from scipy.stats import gmean
import numpy as np
from sklearn.metrics import (
    roc_auc_score, average_precision_score,
    accuracy_score, recall_score, precision_score,
    confusion_matrix
)
from tensorflow.keras.metrics import Metric, BinaryAccuracy, AUC, Recall, Precision
from tensorflow.python.keras.metrics import TrueNegatives, FalsePositives


def compute_auc(ypred, ytrue):
    pred_vector = ypred.flatten()
    true_vector = ytrue.flatten()
    ap = average_precision_score(true_vector, pred_vector)
    auc = roc_auc_score(true_vector, pred_vector)
    return auc, ap


def compute_all_metrics(ypred, ytrue, threshold=0.5, avg_metric='geo_mean'):
    # Flatten
    pred_vector = ypred.flatten()
    true_vector = ytrue.flatten()

    # Threshold for binary predictions
    binary_preds = (pred_vector >= threshold).astype(int)

    # Standard metrics
    binary_acc = accuracy_score(true_vector, binary_preds)
    roc_auc = roc_auc_score(true_vector, pred_vector) if len(np.unique(true_vector)) > 1 else np.nan
    pr_auc = average_precision_score(true_vector, pred_vector)
    recall = recall_score(true_vector, binary_preds, zero_division=0)
    precision = precision_score(true_vector, binary_preds, zero_division=0)

    # Specificity = TN / (TN + FP)
    tn, fp, fn, tp = confusion_matrix(true_vector, binary_preds, labels=[0, 1]).ravel()
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0

    metrics_to_avg = [binary_acc, roc_auc, pr_auc, recall, precision, specificity]

    # Average model performance metric. Can be either geo_mean or avg_perf
    model_avg = None
    if avg_metric == 'geo_mean':
        geo_calc = GeoMeanCalculator()
        model_avg = geo_calc.compute(metrics_to_avg)
    elif avg_metric == 'avg_perf':
        avg_calc = AverageCalculator()
        model_avg = avg_calc.compute(metrics_to_avg)

    return {
        "binary_accuracy": binary_acc,
        "ROC_AUC": roc_auc,
        "PR_AUC": pr_auc,
        "recall": recall,
        "precision": precision,
        "specificity": specificity,
        avg_metric: model_avg
    }


class AverageMetric(Metric):
    def __init__(self, name='avg_perf', **kwargs):
        super().__init__(name=name, **kwargs)

        # Initialize the individual metrics
        self.binary_acc = BinaryAccuracy(name='binary_accuracy_internal', threshold=0.5)
        self.roc_auc = AUC(curve="ROC", name='roc_auc_internal')
        self.pr_auc = AUC(curve="PR", name='pr_auc_internal')
        self.recall = Recall(name='recall_internal')
        self.precision = Precision(name='precision_internal')
        self.specificity = Specificity(name='specificity_internal')

        # State to store the average
        self.average = self.add_weight(name='average', initializer='zeros')

    def update_state(self, y_true, y_pred, sample_weight=None):
        # Update all individual metrics
        self.roc_auc.update_state(y_true, y_pred, sample_weight)
        self.pr_auc.update_state(y_true, y_pred, sample_weight)
        self.recall.update_state(y_true, y_pred, sample_weight)
        self.precision.update_state(y_true, y_pred, sample_weight)

        # Calculate the weighted average
        binary_acc = self.binary_acc.result()
        roc_value = self.roc_auc.result()
        pr_value = self.pr_auc.result()
        recall_value = self.recall.result()
        precision_value = self.precision.result()
        specificity = self.specificity.result()

        avg_perf = (binary_acc + roc_value + pr_value + recall_value + precision_value + specificity) / 6

        self.average.assign(avg_perf)

    def result(self):
        return self.average

    def reset_state(self):
        # Reset all metrics
        self.binary_acc.reset_state()
        self.roc_auc.reset_state()
        self.pr_auc.reset_state()
        self.recall.reset_state()
        self.precision.reset_state()
        self.specificity.reset_state()
        self.average.assign(0.0)

class AverageCalculator:
    def __init__(self):
        pass

    def compute(self, metrics):
        return np.mean(metrics)

class Specificity(Metric):
    def __init__(self, name='specificity', **kwargs):
        super().__init__(name=name, **kwargs)

        # Initialize the individual metrics
        self.tn = TrueNegatives(name='true_negatives')
        self.fp = FalsePositives(name='false_positives')

        # State to store the average
        self.specificity = self.add_weight(name='specificity', initializer='zeros')

    def update_state(self, y_true, y_pred, sample_weight=None):
        # Update all individual metrics
        self.tn.update_state(y_true, y_pred, sample_weight)
        self.fp.update_state(y_true, y_pred, sample_weight)

        # Calculate the weighted average
        tn = self.tn.result()
        fp = self.fp.result()

        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0

        self.specificity.assign(specificity)

    def result(self):
        return self.specificity

    def reset_state(self):
        # Reset all metrics
        self.tn.reset_state()
        self.fp.reset_state()
        self.specificity.assign(0.0)

class GeometricMeanMetric(Metric):
    def __init__(self, name="geo_mean", **kwargs):
        super(GeometricMeanMetric, self).__init__(name=name, **kwargs)

        self.binary_acc = BinaryAccuracy(name="binary_accuracy", threshold=0.5)
        self.roc_auc = AUC(curve="ROC", name="ROC_AUC")
        self.pr_auc = AUC(curve="PR", name="PR_AUC")
        self.recall = Recall()
        self.precision = Precision()
        self.specificity = Specificity()

        self.geo_mean = self.add_weight(name='geo_mean', initializer='zeros')

    def update_state(self, y_true, y_pred, sample_weight=None):
        self.binary_acc.update_state(y_true, y_pred, sample_weight)
        self.roc_auc.update_state(y_true, y_pred, sample_weight)
        self.pr_auc.update_state(y_true, y_pred, sample_weight)
        self.recall.update_state(y_true, y_pred, sample_weight)
        self.precision.update_state(y_true, y_pred, sample_weight)
        self.specificity.update_state(y_true, y_pred, sample_weight)

        values = [
            self.binary_acc.result(),
            self.roc_auc.result(),
            self.pr_auc.result(),
            self.recall.result(),
            self.precision.result(),
            self.specificity.result()
        ]
        values = [tf.where(v > 0, v, keras.backend.epsilon()) for v in values]

        self.geo_mean.assign(tf.exp(tf.reduce_mean(tf.math.log(values))))


    def result(self):
        return self.geo_mean

    def reset_state(self):
        self.binary_acc.reset_state()
        self.roc_auc.reset_state()
        self.pr_auc.reset_state()
        self.recall.reset_state()
        self.precision.reset_state()
        self.specificity.reset_state()
        self.geo_mean.assign(0.0)

class GeoMeanCalculator:
    def __init__(self, epsilon=keras.backend.epsilon()):
        self.epsilon = epsilon

    def compute(self, metrics):
        return gmean([m if m != 0 else self.epsilon for m in metrics])

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score, roc_auc_score, precision_score, recall_score,
    confusion_matrix, roc_curve, precision_recall_curve, auc
)

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score, roc_auc_score, average_precision_score,
    recall_score, precision_score, confusion_matrix
)

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score, roc_auc_score, average_precision_score,
    recall_score, precision_score, confusion_matrix
)


def plot_per_sample_metrics(ypred, ytrue, threshold=0.5):
    """
    Computes and plots per-sample binary classification metrics.

    Each sample is expected to be of shape (D,), and the full input shape is (N, D).

    Parameters:
        ypred (np.ndarray): Predicted probabilities, shape (N, D)
        ytrue (np.ndarray): True binary labels, shape (N, D)
        threshold (float): Threshold for binary predictions
    """
    N = ypred.shape[0]
    metrics_by_name = {
        "binary_accuracy": [],
        "ROC_AUC": [],
        "PR_AUC": [],
        "recall": [],
        "precision": [],
        "specificity": []
    }

    for i in range(N):
        pred = ypred[i].flatten()
        true = ytrue[i].flatten().astype(int)
        binary_pred = (pred >= threshold).astype(int)

        # Binary metrics
        acc = accuracy_score(true, binary_pred)
        try:
            roc_auc = roc_auc_score(true, pred)
        except:
            roc_auc = np.nan
        pr_auc = average_precision_score(true, pred)
        recall = recall_score(true, binary_pred, zero_division=0)
        precision = precision_score(true, binary_pred, zero_division=0)

        # Specificity
        tn, fp, fn, tp = confusion_matrix(true, binary_pred, labels=[0, 1]).ravel()
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0

        # Store
        metrics_by_name["binary_accuracy"].append(acc)
        metrics_by_name["ROC_AUC"].append(roc_auc)
        metrics_by_name["PR_AUC"].append(pr_auc)
        metrics_by_name["recall"].append(recall)
        metrics_by_name["precision"].append(precision)
        metrics_by_name["specificity"].append(specificity)

    # Plot each metric separately
    num_metrics = len(metrics_by_name)
    fig, axes = plt.subplots(num_metrics, 1, figsize=(10, 2.5 * num_metrics), sharex=True)

    # for ax, (metric_name, values) in zip(axes, metrics_by_name.items()):
    #     ax.plot(values, marker='o', linewidth=1.5)
    #     ax.set_ylabel(metric_name)
    #     ax.grid(True)

    sample_indices = np.arange(N)
    for ax, (metric_name, values) in zip(axes, metrics_by_name.items()):
        ax.scatter(sample_indices, values, alpha=0.8)
        ax.set_ylabel(metric_name)
        ax.grid(True)

    axes[-1].set_xlabel("Sample index")
    fig.suptitle("Per-sample Metric Values", fontsize=14)
    plt.tight_layout()
    plt.show()


