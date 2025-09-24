import json
import logging
import os

import numpy as np
from keras.callbacks import EarlyStopping
from keras.models import load_model

from model.custom_layers import ClipByValue
from util.constants import METRICS_DIR, OUTPUT_DIR, MODELS_DIR, PATCH_SIZE, RESOLUTION
from util.logger import Logger
from metrics import compute_auc, compute_all_metrics, Specificity, AverageMetric


class BaseModel(object):
        def __init__(self, model_name='base_model',
                     save_as='base_model',
                     ext='.h5',
                     patch_size=PATCH_SIZE,
                     monitor_metric='val_avg_perf',
                     min_delta=0.0001,
                     patience=7):
            self.model = None
            self.model_name = model_name
            self.save_as = save_as
            self.ext = ext
            self.patch_size = patch_size
            self.monitor_metric = monitor_metric
            self.min_delta = min_delta
            self.patience = patience
            self.LOGGER = Logger(name=self.model_name, level=logging.DEBUG).get_logger()

        def build(self, image_upper_bound):
            pass

        def train(self, train_gen, val_gen, epochs):
            history = self.model.fit(train_gen,
                                     validation_data=val_gen,
                                     epochs=epochs,
                                     workers=1,
                                     use_multiprocessing=True,
                                     callbacks=[EarlyStopping(monitor=self.monitor_metric,
                                                              min_delta=self.min_delta,
                                                              patience=self.patience,
                                                              verbose=1,
                                                              mode='max',
                                                              restore_best_weights=True)
                                                ],
                                     verbose=1)

            self.save_model(history)

        def test(self, test_gen):
            y_pred = self.model.predict(test_gen, workers=4, use_multiprocessing=True, verbose=1)

            # collect true labels from generator
            y_true_batches = []
            for _, y_batch in test_gen:
                y_true_batches.append(y_batch)
            y_true = np.concatenate(y_true_batches, axis=0)

            all_test_metrics = compute_all_metrics(y_pred, y_true.astype('bool'))
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