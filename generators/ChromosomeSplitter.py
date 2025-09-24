import logging
import os
from pathlib import Path

import numpy as np
from typing import List, Iterable

from numpy.random.mtrand import random_sample
from scipy.sparse import load_npz

from generators.chromosome_generator import ChromosomeGenerator
from util.logger import Logger
from util.plotting.plotting import plot_pixel_counts, plot_top_positive_patches
from util.utils import count_pos_neg_distributions

LOGGER = Logger(name='ChromosomeLoader', level=logging.DEBUG).get_logger()

class ChromosomeSplitter:
    def __init__(self,
                 chromosomes: List[str] = None,
                 data_dir: str = 'dataset/hela_100',
                 patch_size: int = 64,
                 batch_size: int = 32,
                 shuffle_train: bool = True,
                 split_ratios: Iterable[float] = (0.7, 0.2, 0.1),
                 include_diagonal: bool = False,
                 process_as_chromosomes: bool = False):

        if chromosomes is None:
            self.chromosomes = [str(i) for i in range(1, 23)] + ['X']
        else:
            self.chromosomes = chromosomes

        self.data_dir = data_dir
        self.patch_size = patch_size
        self.batch_size = batch_size
        self.shuffle_train = shuffle_train
        self.split_ratios = split_ratios
        self.include_diagonal = include_diagonal
        self.process_as_chromosomes = process_as_chromosomes

    def create_data_splits(self):
        if self.process_as_chromosomes:
            self.compute_data_split_indices_npzy()
        else:
            self.compute_data_split_images()

    def compute_data_split_indices_npzy(self):
        chrom_data = []
        for chromosome in self.chromosomes:
            patch_indices = np.load(Path(self.data_dir) / f'{chromosome}_patches.npy')
            chrom_data.append((chromosome, patch_indices))

        total_samples = len(chrom_data)

        # Shuffle
        np.random.shuffle(chrom_data)

        # Compute split boundaries
        train_end = int(self.split_ratios[0] * total_samples)
        val_end = train_end + int(self.split_ratios[1] * total_samples)

        train_idx = chrom_data[:train_end]
        train_idx.extend(self.upsample_dataset(train_idx, .3, 100))

        val_idx = chrom_data[train_end:val_end]
        test_idx = chrom_data[val_end:]

        '''
        collected = []
        import tensorflow as tf
        [collected.extend(tf.reshape(sample[0], [-1]).numpy()) for sample in train_idx]
        collected = np.array(collected[:2000000])
        upper_bound = np.quantile(collected, 0.996)
        train_idx_copy = []
        for idx, i in enumerate(train_idx):
            # clip and rescale
            # data = ClipByValue(upper_bound)(i[0]).numpy()
            # data = Rescale(1./upper_bound)(data)

            # only rescale
            # data = Rescale(1. / upper_bound)(i[0])

            # only clip
            data = ClipByValue(upper_bound)(i[0]).numpy()

            train_idx_copy.append((data, i[1]))
        '''

        # return train_idx_copy, val_idx, test_idx
        return train_idx, val_idx, test_idx

    def compute_data_split_images(self):
        # Total number of samples
        chrom_lengths = [np.load(os.path.join(self.data_dir, f'imageset.{chromosome}.npy'), mmap_mode='r').shape[0]
                         for chromosome in self.chromosomes]
        total_samples = sum(chrom_lengths)
        all_indices = np.arange(total_samples)

        # Shuffle
        np.random.shuffle(all_indices)

        # Compute split boundaries
        train_end = int(self.split_ratios[0] * total_samples)
        val_end = train_end + int(self.split_ratios[1] * total_samples)

        train_idx = all_indices[:train_end]
        val_idx = all_indices[train_end:val_end]
        test_idx = all_indices[val_end:]

        return train_idx, val_idx, test_idx

    def visualize_data(self, generator: ChromosomeGenerator, limit: int = None, plot_neg: bool = False):
        # compute distributions
        train_counts = count_pos_neg_distributions(generator, patch_size=self.patch_size)

        # plot distributions
        plot_pixel_counts(train_counts, generator.name, limit, plot_neg, patch_size=self.patch_size)

        # plot the heatmap for the top 4 images
        plot_top_positive_patches(train_counts, 4,  patch_size=self.patch_size)
        plot_top_positive_patches(train_counts, 4, data_to_plot='y',  patch_size=self.patch_size)