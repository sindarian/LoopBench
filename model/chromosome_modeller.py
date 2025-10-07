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
            # if we're training the original GILoop model, calculate the
            # upperbound for the whole train dataset
            LOGGER.info(f'Training original model for {self.model.model_name}')
            upper_bound = self.estimate_upper_bound(self.train_generator)
            self.model.build_original(upper_bound)
        else:
            # otherwise if we're training the new models, then the data
            # was sampled with the upper bound of each chromosome
            LOGGER.info(f'Training model for {self.model.model_name}')
            self.model.build()

        # train teh model with the generators
        self.model.train(self.train_generator, self.val_generator, self.epochs)

        # the history object is saved as a file and on the model object itself. For
        # evaluation of different models, extract the final train and val metrics from it
        train_metrics, val_metrics = self.extract_final_train_metrics()

        # and get the test metrics
        test_metrics = self.model.test(self.test_generator)

        LOGGER.info(f'#########################################################')
        LOGGER.info(f'Final Scores for model: {self.model.model_name}')
        LOGGER.info(f'Train: {train_metrics}')
        LOGGER.info(f'Validation: {val_metrics}')
        LOGGER.info(f'Test: {test_metrics}')
        LOGGER.info(f'#########################################################')

        # reset the keras session
        K.clear_session()
        return train_metrics, val_metrics, test_metrics


    def estimate_upper_bound(self, generator, percentile=0.996, max_pixels=2000000):
        collected = []

        for batch_x, _ in generator:
            pixels = tf.reshape(batch_x['patch'], [-1]).numpy()
            collected.extend(pixels.tolist())

            if len(collected) >= max_pixels:
                break

        collected = np.array(collected[:max_pixels])
        return np.quantile(collected, percentile)

    def run_n_times(self, num_runs: int = 3, drop_worst=1, train_original: bool = False):
        if drop_worst >= num_runs:
            raise ValueError('drop_worst must be less than num_repeats')

        # list to store the final metrics for each repeat run
        all_runs_data = []

        # run the model num_runs times
        for i in range(num_runs):
            LOGGER.info(f'\nFitting model: {format(self.model.model_name)} - repeat {i + 1}')
            train_metrics, val_metrics, test_metrics = self.run(train_original)

            # Store all metrics for this run
            all_runs_data.append({
                'train': train_metrics,
                'val': val_metrics,
                'test': test_metrics
            })

        # compute the average metrics for the model
        avg_train_metrics, avg_val_metrics, avg_test_metrics = self.compute_avg_metrics(all_runs_data, drop_worst)

        return avg_train_metrics, avg_val_metrics, avg_test_metrics

    def extract_final_train_metrics(self):
        train_metrics = {}
        val_metrics = {}
        for metric in self.model.history.keys():
            if 'loss' not in metric:
                if 'val_' in metric:
                    val_metrics[metric] = self.model.history[metric][-1]
                else:
                    train_metrics[metric] = self.model.history[metric][-1]

        return train_metrics, val_metrics

    def compute_avg_metrics(self, all_runs_data, drop_worst):
        # Calculate performance score for each run (average of train and val avg_metrics)
        run_scores = []
        for i, run_data in enumerate(all_runs_data):
            train_avg_metric = run_data['train'][self.model.avg_metric]
            val_avg_metric = run_data['val']['val_' + self.model.avg_metric]
            avg_avg_metric = (train_avg_metric + val_avg_metric) / 2
            run_scores.append((avg_avg_metric, i, run_data))

        # Sort by performance score (ascending order, so worst runs are first)
        run_scores.sort(key=lambda x: x[0])

        # Keep the best runs (drop the worst ones)
        best_runs = run_scores[drop_worst:] if drop_worst > 0 else run_scores

        # Extract the data from best runs
        best_runs_data = [run_data for _, _, run_data in best_runs]

        # Get all metric names from the first run
        metric_names = list(best_runs_data[0]['train'].keys())

        # Calculate averages for each metric across the remaining runs
        avg_train_metrics = {}
        avg_val_metrics = {}
        avg_test_metrics = {}

        for metric in metric_names:
            # Calculate average for train metrics
            train_values = [run['train'][metric] for run in best_runs_data]
            avg_train_metrics[metric] = np.mean(train_values)

            # Calculate average for val metrics
            val_values = [run['val']['val_' +  metric] for run in best_runs_data]
            avg_val_metrics[metric] = np.mean(val_values)

            # Calculate average for test metrics
            test_values = [run['test'][metric] for run in best_runs_data]
            avg_test_metrics[metric] = np.mean(test_values)

        return avg_train_metrics, avg_val_metrics, avg_test_metrics

    # def run_n_times(self, num_runs: int = 3, drop_worst=1, train_original: bool = False):
    #     if drop_worst >= num_runs:
    #         raise ValueError('drop_worst must be less than num_repeats')
    #
    #     # lists to store the final metrics for each repeat run
    #     all_final_train_metrics = []
    #     all_final_val_metrics = []
    #     all_final_test_metrics = []
    #
    #     # run the model num_runs times
    #     for i in range(num_runs):
    #         LOGGER.info(f'\nFitting model: {format(self.model.model_name)} - repeat {i + 1}')
    #         train_metrics, val_metrics, test_metrics = self.run(train_original)
    #
    #         # Store the final metrics
    #         all_final_train_metrics.append(tuple(train_metrics.values()))
    #         all_final_val_metrics.append(tuple(val_metrics.values()))
    #         all_final_test_metrics.append(tuple(test_metrics.values()))
    #
    #     # compute the average metrcs for the model
    #     avg_train_metrics = self.compute_avg_metrics(all_final_train_metrics, drop_worst)
    #     avg_val_metrics = self.compute_avg_metrics(all_final_val_metrics, drop_worst)
    #     avg_test_metrics = self.compute_avg_metrics(all_final_test_metrics, drop_worst)
    #
    #     return avg_train_metrics, avg_val_metrics, avg_test_metrics

    # def extract_final_train_metrics(self):
    #     train_metrics = {}
    #     val_metrics = {}
    #     for metric in self.model.history.keys():
    #         if 'val_' in metric:
    #             val_metrics[metric] = self.model.history[metric][-1]
    #         else:
    #             train_metrics[metric] = self.model.history[metric][-1]
    #
    #     return train_metrics, val_metrics

    # def compute_avg_metrics(self, metrics, drop_worst):
    #     sorted_metrics = sorted(metrics, key=lambda x: x[-1])  # sorted in ascending order
    #     # best_run = sorted_metrics[-1]
    #
    #     # take the average of remaining metrics with the lowest excluded
    #     grouped_metrics = zip(*sorted_metrics[drop_worst:]) if drop_worst > 0 else zip(*sorted_metrics)
    #     average_metrics = [np.mean(metric) for metric in grouped_metrics]
    #
    #     return average_metrics