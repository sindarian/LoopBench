import json
import logging
import os

import numpy as np
from keras.callbacks import EarlyStopping
from keras.models import load_model
from sklearn.metrics import roc_curve, auc, precision_recall_curve

from model.custom_layers import ClipByValue
from util.constants import METRICS_DIR, OUTPUT_DIR, MODELS_DIR, PATCH_SIZE, RESOLUTION
from util.logger import Logger
from metrics import compute_all_metrics, Specificity, AverageMetric, plot_per_sample_metrics


class BaseModel(object):
        def __init__(self, model_name='base_model',
                     save_as='base_model',
                     ext='.h5',
                     patch_size=PATCH_SIZE,
                     min_delta=0.0001,
                     patience=7,
                     avg_metric='geo_mean',
                     note=""):
            if avg_metric != 'geo_mean' and avg_metric != 'avg_perf':
                raise ValueError("Only geo_mean and avg_perf are supported as the average metric")

            self.model = None
            self.history = None
            self.model_name = model_name
            self.save_as = save_as
            self.ext = ext
            self.patch_size = patch_size
            self.monitor_metric = "val_" + avg_metric
            self.avg_metric = avg_metric
            self.min_delta = min_delta
            self.patience = patience
            self.note = note
            self.best_epoch = 0

            self.LOGGER = Logger(name=self.model_name, level=logging.DEBUG).get_logger()

        def build(self, image_upper_bound):
            pass

        def train(self, train_gen, val_gen, epochs):
            early_stop = EarlyStopping(monitor=self.monitor_metric,
                                                              min_delta=self.min_delta,
                                                              patience=self.patience,
                                                              verbose=1,
                                                              mode='max',
                                                              restore_best_weights=True)
            history = self.model.fit(train_gen,
                                     validation_data=val_gen,
                                     epochs=epochs,
                                     workers=1,
                                     use_multiprocessing=True,
                                     callbacks=[early_stop],
                                     verbose=1)
            self.best_epoch = early_stop.best_epoch
            self.history = history.history
            self.save_model(history)

        def test(self, test_gen):
            y_pred = self.model.predict(test_gen, workers=4, use_multiprocessing=True, verbose=1)

            # collect true labels from generator
            y_true_batches = []
            for _, y_batch in test_gen:
                y_true_batches.append(y_batch)
            y_true = np.concatenate(y_true_batches, axis=0)

            all_test_metrics = compute_all_metrics(y_pred, y_true.astype('bool'), avg_metric=self.avg_metric)

            fpr, tpr, thresholds = roc_curve(y_true.flatten(), y_pred.flatten())
            roc_auc = auc(fpr, tpr)

            precision, recall, thresholds = precision_recall_curve(y_true.flatten(), y_pred.flatten())
            pr_auc = auc(recall, precision)

            # self.plot_curves((fpr, tpr, roc_auc), (recall, precision, pr_auc))

            return all_test_metrics

        def save_model(self, history):
            print(f'Saving {self.model_name} model and training history')

            metric_file_name = f'{self.save_as}_training_metrics.json'
            with open(os.path.join(OUTPUT_DIR, METRICS_DIR, metric_file_name), 'w') as f:
                json.dump(history.history, f)

            self.model.save(os.path.join(OUTPUT_DIR,
                                         MODELS_DIR,
                                         self.save_as + self.ext))

        def load(self):
            model_path = os.path.join(OUTPUT_DIR, MODELS_DIR, self.save_as + self.ext)
            if os.path.exists(model_path):
                model = load_model(model_path, custom_objects={'ClipByValue': ClipByValue,
                                                               'Specificity': Specificity,
                                                               'AverageMetric': AverageMetric})
            else:
                raise FileNotFoundError(f'Model {model_path} not found')

            return model

        def plot_curves(self, roc_data, pr_data):
            """
            Plot ROC and PR curves side by side.

            Parameters:
            -----------
            roc_data : tuple
                (fpr, tpr, roc_auc)
            pr_data : tuple
                (recall, precision, pr_auc)
            """
            import matplotlib.pyplot as plt

            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
            fig.suptitle(self.note, fontsize=14, fontweight='bold', y=0.98)

            # ROC Curve
            fpr, tpr, roc_auc = roc_data
            ax1.plot(fpr, tpr, color='blue', lw=2,
                     label=f'ROC curve (AUC = {roc_auc:.4f})')
            ax1.plot([0, 1], [0, 1], color='red', lw=2,
                     linestyle='--', label='Random Classifier')
            ax1.set_xlim([0.0, 1.0])
            ax1.set_ylim([0.0, 1.05])
            ax1.set_xlabel('False Positive Rate', fontsize=14)
            ax1.set_ylabel('True Positive Rate', fontsize=14)
            ax1.set_title('Receiver Operating Characteristic (ROC) Curve',
                          fontsize=16, fontweight='bold')
            ax1.legend(loc="lower right", fontsize=12)
            ax1.grid(alpha=0.3)

            # PR Curve
            recall, precision, pr_auc = pr_data
            ax2.plot(recall, precision, color='blue', lw=2,
                     label=f'PR curve (AUC = {pr_auc:.4f})')
            ax2.set_xlim([0.0, 1.0])
            ax2.set_ylim([0.0, 1.05])
            ax2.set_xlabel('Recall', fontsize=14)
            ax2.set_ylabel('Precision', fontsize=14)
            ax2.set_title('Precision-Recall Curve', fontsize=16, fontweight='bold')
            ax2.legend(loc="best", fontsize=12)
            ax2.grid(alpha=0.3)

            plt.tight_layout()
            plt.show()