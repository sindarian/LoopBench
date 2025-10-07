import logging
import os
import pickle
from itertools import product
from pathlib import Path

import numpy as np
from typing import List, Iterable

from numpy.random.mtrand import random_sample
from scipy.sparse import load_npz

from generators.chromosome_generator import ChromosomeGenerator
from util.constants import PATCH_SIZE, RESOLUTION
from util.logger import Logger
from util.plotting.plotting import plot_pixel_counts, plot_top_positive_patches
from util.utils import count_pos_neg_distributions

LOGGER = Logger(name='ChromosomeLoader', level=logging.DEBUG).get_logger()

class ChromosomeLoader:
    def __init__(self,
                 chromosomes: List[str] = None,
                 data_dir: str = 'dataset/hela_100',
                 patch_size: int = 64,
                 resolution: int = RESOLUTION,
                 batch_size: int = 32,
                 shuffle_train: bool = True,
                 split_ratios: Iterable[float] = (0.7, 0.2, 0.1),
                 include_diagonal: bool = False,
                 use_original: bool = True,
                 experiment: bool = False,
                 thresholds = None,
                 normalizations = None):

        if chromosomes is None:
            self.chromosomes = [str(i) for i in range(1, 23)] + ['X']
        else:
            self.chromosomes = chromosomes

        self.data_dir = data_dir
        self.patch_size = patch_size
        self.resolution = resolution
        self.batch_size = batch_size
        self.shuffle_train = shuffle_train
        self.split_ratios = split_ratios
        self.include_diagonal = include_diagonal
        self.use_original = use_original

        self.experiment = experiment
        if self.experiment:
            self.thresholds = thresholds
            self.normalizations = normalizations
            self.all_sampling_param_combos = list(product(self.thresholds, self.normalizations))

    def compute_and_save_data_splits(self, overwrite=False, upsample=False, strategy=None, factor=1, threshold=100):
        # if self.use_original:
        # it doesn't matter what threshold/normalization is used here. the patches are the same across all datasets,
        # only the contact matrices are different
        train_idx, val_idx, test_idx, upsampled_idx = self.split_data(self.thresholds[0], self.normalizations[0],
                                                                      upsample, strategy, factor, threshold)

        # save the same non-upsampled train, val, test split for each dataset
        for quantile, method in self.all_sampling_param_combos:
            self.write_data_splits(quantile, method, train_idx, val_idx, test_idx, upsampled_idx, strategy, overwrite)

    def create_experiment_generators(self, upsample=False, strategy=None, factor=1, threshold=100, resolution=None):
        # a dict will be returned that contains the generators for all experiments
        generators = {}

        # for each experiment dataset
        for quantile, method in self.all_sampling_param_combos:
            # load the split data
            train_data, val_data, test_data = self.load_split_data(quantile, method, upsample, strategy, factor, threshold)

            # create a generator for each split
            train_gen = self.create_generator(train_data, shuffle=self.shuffle_train, name='Train', resolution=resolution)
            val_gen = self.create_generator(val_data, shuffle=False, name='Val', resolution=resolution)
            test_gen = self.create_generator(test_data, shuffle=False, name='Test', resolution=resolution)

            # add the generators into the dictionary indexed by the quantile threshold and normalization method
            generators[f'{quantile}_{method}'] = [train_gen, val_gen, test_gen]

        return generators

    def combine_gens(self, generators, shuffle):

        # gen1_data = [(i[0], i[1], gen1.resolution) for i in gen1.indices]
        # gen2_data = [(i[0], i[1], gen2.resolution) for i in gen2.indices]
        # gen3_data = [(i[0], i[1], gen3.resolution) for i in gen3.indices]
        # all_data = gen1_data + gen2_data + gen3_data

        all_data = []
        for gen in generators:
            all_data += [(i[0], i[1], i[2]) for i in gen.indices]

        return self.create_generator(all_data, shuffle=shuffle, name='Combined')

    def create_generator(self, data, shuffle, name, resolution=None):
        return ChromosomeGenerator(
            indices=data,
            chromosomes=self.chromosomes,
            name=name,
            patch_size=self.patch_size,
            batch_size=self.batch_size,
            shuffle=shuffle,
            include_diagonal=self.include_diagonal,
            use_original=self.use_original,
            resolution=resolution)

    def load_split_data(self, quantile, method, upsample=False, strategy=None, factor=1, threshold=100):
        path = Path(str(self.data_dir) + f'_{quantile}_quant_{method}_ps_{self.patch_size}_rs_{self.resolution}')

        # load the metadata for the dataset
        metadata_path = path / 'dataset_metadata.pkl'

        if metadata_path.exists():
            # load existing pickle
            with open(metadata_path, "rb") as f:
                dataset_metadata = pickle.load(f)
        else:
            raise Exception(f"No metadata file found for dataset {path}")

        # load the data splits
        train_split = dataset_metadata['splits']['train']
        val_split = dataset_metadata['splits']['val']
        test_split = dataset_metadata['splits']['test']

        # load the upsampled data if it exists
        saved_upsampled_data = []
        if strategy in dataset_metadata['splits']:
            LOGGER.info(f"Upsampled data already saved for strategy: {strategy}")
            saved_upsampled_data = dataset_metadata['splits'][strategy]


        train_data = []
        val_data = []
        test_data = []

        # next load the chromosome contacts and ground truth
        for chromosome in self.chromosomes:
            contacts = load_npz(path / f'{chromosome}_contacts.npz')
            ground_truth = load_npz(path / f'{chromosome}_ground_truth.npz')

            train_data += self.load_patches(chromosome, train_split, contacts, ground_truth)
            val_data += self.load_patches(chromosome, val_split, contacts, ground_truth)
            test_data += self.load_patches(chromosome, test_split, contacts, ground_truth)

        if len(saved_upsampled_data) > 0:
            train_data.extend(saved_upsampled_data)
        elif upsample:
            upsampled_data = self.upsample_dataset(train_data, factor=factor, threshold=threshold, strategy=strategy)
            train_data.extend(upsampled_data)

            LOGGER.info(f"No upsampled data saved for strategy: {strategy}. Saving upsampled data from this run.")
            dataset_metadata['splits'][strategy] = upsampled_data
            # pickle.dump(dataset_metadata, f)
            with open(metadata_path, "wb") as f:
                pickle.dump(dataset_metadata, f)

        return train_data, val_data, test_data

    def load_patches(self, chromosome, data_split, contacts, ground_truth):
        chrom_patches = [data for data in data_split if data[0] == chromosome]
        patches = []

        for chrom, patch in chrom_patches:
            start_row = patch[0]
            start_col = patch[1]
            end_row = patch[2]
            end_col = patch[3]

            patches.append((contacts[start_row:end_row, start_col:end_col].toarray(),
                            ground_truth[start_row:end_row, start_col:end_col].toarray(),
                            self.resolution))
            if len(patches[0][0]) == 0:
                LOGGER.info(f"Empty patch found for chromosome {chrom} at {patch}")

        return patches

    def create_chromosome_generators(self, upsample=False, factor=1, threshold=100, strategy="balanced"):
        if self.use_original:
            train_data, val_data, test_data = self.compute_data_split_images()
        else:
            train_data, val_data, test_data = self.compute_data_split_indices_npzy(upsample, factor, threshold,
                                                                                   strategy)

        train_gen = self.create_generator(train_data, shuffle=self.shuffle_train, name='Train')
        val_gen = self.create_generator(val_data, shuffle=False, name='Val')
        test_gen = self.create_generator(test_data, shuffle=False, name='Test')

        # # Create generators
        # train_gen = ChromosomeGenerator(
        #                                 indices=train_idx,
        #                                 chromosomes=self.chromosomes,
        #                                 name='Train Generator',
        #                                 patch_size=self.patch_size,
        #                                 batch_size=self.batch_size,
        #                                 shuffle=self.shuffle_train,
        #                                 include_diagonal=self.include_diagonal,
        #                                 process_as_chromosome=self.process_as_chromosomes)
        # val_gen = ChromosomeGenerator(
        #                               indices=val_idx,
        #                               chromosomes=self.chromosomes,
        #                               name='Validation Generator',
        #                               patch_size=self.patch_size,
        #                               batch_size=self.batch_size,
        #                               include_diagonal=self.include_diagonal,
        #                               process_as_chromosome=self.process_as_chromosomes)
        # test_gen = ChromosomeGenerator(
        #                                indices=test_idx,
        #                                chromosomes=self.chromosomes,
        #                                name='Test Generator',
        #                                patch_size=self.patch_size,
        #                                batch_size=self.batch_size,
        #                                include_diagonal=self.include_diagonal,
        #                                process_as_chromosome=self.process_as_chromosomes)

        return train_gen, val_gen, test_gen

    # def create_chromosome_generators(self):
    #     if self.process_as_chromosomes:
    #         train_idx, val_idx, test_idx = self.compute_data_split_indices_npzy()
    #     else:
    #         train_idx, val_idx, test_idx = self.compute_data_split_images()
    #
    #     # Create generators
    #     train_gen = ChromosomeGenerator(
    #                                     indices=train_idx,
    #                                     chromosomes=self.chromosomes,
    #                                     name='Train Generator',
    #                                     patch_size=self.patch_size,
    #                                     batch_size=self.batch_size,
    #                                     shuffle=self.shuffle_train,
    #                                     include_diagonal=self.include_diagonal,
    #                                     process_as_chromosome=self.process_as_chromosomes)
    #     val_gen = ChromosomeGenerator(
    #                                   indices=val_idx,
    #                                   chromosomes=self.chromosomes,
    #                                   name='Validation Generator',
    #                                   patch_size=self.patch_size,
    #                                   batch_size=self.batch_size,
    #                                   include_diagonal=self.include_diagonal,
    #                                   process_as_chromosome=self.process_as_chromosomes)
    #     test_gen = ChromosomeGenerator(
    #                                    indices=test_idx,
    #                                    chromosomes=self.chromosomes,
    #                                    name='Test Generator',
    #                                    patch_size=self.patch_size,
    #                                    batch_size=self.batch_size,
    #                                    include_diagonal=self.include_diagonal,
    #                                    process_as_chromosome=self.process_as_chromosomes)
    #
    #     return train_gen, val_gen, test_gen

    def split_data(self, quantile, method, upsample=False, strategy=None, factor=1, threshold=100):
        chrom_data = []
        path = Path(str(self.data_dir) + f'_{quantile}_quant_{method}_ps_{self.patch_size}_rs_{self.resolution}')

        for chromosome in self.chromosomes:
            patch_indices = np.load(path / f'{chromosome}_patches.npy')
            for patch in patch_indices:
                chrom_data.append((chromosome, patch))

        # Shuffle
        np.random.shuffle(chrom_data)

        # get total number of samples across all the data
        total_samples = len(chrom_data)

        # Compute split boundaries
        train_end = int(self.split_ratios[0] * total_samples)
        val_end = train_end + int(self.split_ratios[1] * total_samples)

        train_indices = chrom_data[:train_end]

        upsampled_indices = None
        if upsample:
            upsampled_indices = self.upsample_dataset(train_indices, factor, threshold, strategy)

        val_indices = chrom_data[train_end:val_end]
        test_indices = chrom_data[val_end:]

        return train_indices, val_indices, test_indices, upsampled_indices

    def write_data_splits(self, quantile, method, train_indices, val_indices, test_indices, upsampled_idx, strategy, overwrite=False):
        path = Path(str(self.data_dir) + f'_{quantile}_quant_{method}_ps_{self.patch_size}_rs_{self.resolution}')
        metadata_path = path / 'dataset_metadata.pkl'

        if metadata_path.exists():
            with open(metadata_path, "rb") as f:
                dataset_metadata = pickle.load(f)
        else:
            raise Exception(f"No metadata file found for dataset {path}")

        if 'splits' not in dataset_metadata or overwrite:
            dataset_metadata['splits'] = \
                {
                    'train': train_indices,
                    'val': val_indices,
                    'test': test_indices
                }

            if upsampled_idx is not None and (strategy not in dataset_metadata['splits'] or overwrite):
                dataset_metadata['splits'][strategy] = upsampled_idx

            # Save back to pickle
            with open(metadata_path, "wb") as f:
                pickle.dump(dataset_metadata, f)
        else:
            return # nothing to do

    def compute_data_split_indices_npzy(self, upsample=False, factor=1, threshold=100, strategy=None):
        chrom_data = []
        for chromosome in self.chromosomes:
            patch_indices = np.load(Path(self.data_dir) / f'{chromosome}_patches.npy')
            contacts = load_npz(Path(self.data_dir) / f'{chromosome}_contacts.npz')
            ground_truth = load_npz(Path(self.data_dir) / f'{chromosome}_ground_truth.npz')
            for patch in patch_indices:

                # extract the patch indexes
                start_row = patch[0]
                start_col = patch[1]
                end_row = patch[2]
                end_col = patch[3]

                # extract the sparse patch/gt and convert to dense
                patch = contacts[start_row:end_row, start_col:end_col].toarray()
                label = ground_truth[start_row:end_row, start_col:end_col].toarray()

                chrom_data.append((patch, label))

        total_samples = len(chrom_data)

        # Shuffle
        np.random.shuffle(chrom_data)

        # Compute data split indices
        train_end = int(self.split_ratios[0] * total_samples)
        val_end = train_end + int(self.split_ratios[1] * total_samples)

        # Split the patches and ground truth
        train_data = chrom_data[:train_end]
        if upsample:
            train_data.extend(self.upsample_dataset(train_data, factor, threshold, strategy))

        val_data = chrom_data[train_end:val_end]
        test_data = chrom_data[val_end:]

        return train_data, val_data, test_data

    def upsample_dataset(self, data, factor=1, threshold=100, strategy=None):
        viable_data = [d for d in data if np.sum(d[1]) > threshold]

        if strategy == 'random':
            num_new_samples = factor * len(viable_data)
            new_samples_indxs = np.random.choice(np.arange(len(viable_data)), size=num_new_samples, replace=True)
            new_samples = [viable_data[i] for i in new_samples_indxs]
        elif strategy == 'balanced':
            new_samples = viable_data * factor
        else:
            raise Exception(f"Invalid upsampling strategy: {strategy}")

        return new_samples

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