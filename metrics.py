from sklearn.metrics import average_precision_score
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


def compute_all_metrics(ypred, ytrue, threshold=0.5):
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

    # Average metric (same as in your class)
    avg_perf = np.nanmean([
        binary_acc, roc_auc, pr_auc,
        recall, precision, specificity
    ])

    return {
        "binary_accuracy": binary_acc,
        "ROC_AUC": roc_auc,
        "PR_AUC": pr_auc,
        "recall": recall,
        "precision": precision,
        "specificity": specificity,
        "avg_perf": avg_perf
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