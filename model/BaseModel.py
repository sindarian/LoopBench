import json
import logging
import os

import numpy as np
from keras.callbacks import EarlyStopping

from util.constants import METRICS_DIR, OUTPUT_DIR
from util.logger import Logger
from metrics import compute_auc


class BaseModel(object):
        def __init__(self, model_name='base_model'):
            self.model = None
            self.model_name = model_name
            self.LOGGER = Logger(name=self.model_name, level=logging.DEBUG).get_logger()

        def build(self, image_upper_bound):
            pass

        def train(self, train_gen, val_gen, epochs):
            self.LOGGER.info(f'Training {self.model_name}')
            history = self.model.fit(train_gen,
                                     validation_data=val_gen,
                                     epochs=epochs,
                                     workers=4,
                                     use_multiprocessing=True,
                                     callbacks=[EarlyStopping(monitor='val_PR_AUC',
                                                              min_delta=0.0001,
                                                              patience=5,
                                                              verbose=1,
                                                              mode='max',
                                                              restore_best_weights=True)],
                                     verbose=1)

            self.save_model(history)

        def predict(self, test_gen):
            self.LOGGER.info(f'Predicting with trained {self.model_name}')

            y_pred = self.model.predict(test_gen, workers=4, use_multiprocessing=True, verbose=1)

            # collect true labels from generator
            y_true_batches = []
            for _, y_batch in test_gen:
                y_true_batches.append(y_batch)
            y_true = np.concatenate(y_true_batches, axis=0)

            test_auc, test_ap = compute_auc(y_pred, y_true.astype('bool'))
            self.LOGGER.info('Test AUC is {}. Test AP is {}.'.format(test_auc, test_ap))

        def save_model(self, history):
            self.LOGGER.info(f'Saving {self.model_name} model and training history')

            metric_file_name = f'{self.model_name}_training_metrics.json'
            with open(os.path.join(OUTPUT_DIR, METRICS_DIR, metric_file_name), 'w') as f:
                json.dump(history.history, f)

            self.model.save(f'models/{self.model_name}.h5')