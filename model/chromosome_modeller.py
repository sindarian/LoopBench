import logging
from typing import List

import numpy as np
import tensorflow as tf

from generators.chromosome_generator import ChromosomeGenerator
from model.BaseModel import BaseModel
from tensorflow.keras import backend as K
from util.logger import Logger

LOGGER = Logger(name='ChromosomeModeller', level=logging.DEBUG).get_logger()

class ChromosomeModeller:
    def __init__(self,
                 model,
                 train_generator: ChromosomeGenerator,
                 val_generator: ChromosomeGenerator,
                 test_generator: ChromosomeGenerator,
                 epochs: int = 50):
        self.model = model
        self.train_generator = train_generator
        self.val_generator = val_generator
        self.test_generator = test_generator
        self.epochs = epochs

    def run(self, train_original: bool = False):
        if train_original:
            LOGGER.info(f'Training original model for {self.model.model_name}')
            upper_bound = self.estimate_upper_bound(self.train_generator)
            self.model.build_original(upper_bound)
        else:
            LOGGER.info(f'Training model for {self.model.model_name}')
            self.model.build()
        self.model.train(self.train_generator, self.val_generator, self.epochs)
        all_test_metrics = self.model.test(self.test_generator)
        LOGGER.info(f'Test Scores for {self.model.model_name}: {all_test_metrics}')
        K.clear_session()
        return all_test_metrics

    def run_n_times(self, num_runs: int = 3, drop_worst=1, train_original: bool = False):
        if drop_worst >= num_runs:
            raise ValueError('drop_worst must be less than num_repeats')

        test_metrics = []
        for i in range(num_runs):
            LOGGER.info(f'\nFitting model: {format(self.model.model_name)} - repeat {i + 1}')
            run_metrics = self.run(train_original)
            test_metrics.append(tuple(run_metrics.values()))

        sorted_metrics = sorted(test_metrics, key=lambda x: x[-1]) # sorted in ascending order
        best_run = sorted_metrics[-1]

        # take average of remaining metrics with the lowest excluded
        grouped_metrics = zip(*sorted_metrics[drop_worst:]) if drop_worst > 0 else zip(*sorted_metrics)
        average_metrics = [np.mean(metric) for metric in grouped_metrics]

        return average_metrics, best_run

    def estimate_upper_bound(self, generator, percentile=0.996, max_pixels=2000000):
        collected = []

        for batch_x, _ in generator:
            pixels = tf.reshape(batch_x['patch'], [-1]).numpy()
            collected.extend(pixels.tolist())

            if len(collected) >= max_pixels:
                break

        collected = np.array(collected[:max_pixels])
        return np.quantile(collected, percentile)