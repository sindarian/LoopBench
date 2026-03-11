import logging
import os
import pickle
from itertools import product
from pathlib import Path

import numpy as np
from typing import List, Iterable, Tuple

from numpy.random.mtrand import random_sample
from scipy.sparse import load_npz

from generators.chromosome_generator import ChromosomeGenerator
from util.constants import PATCH_SIZE, RESOLUTION
from util.logger import Logger
from util.plotting.plotting import plot_pixel_counts, plot_top_positive_patches
from util.utils import count_pos_neg_distributions

import sparse

LOGGER = Logger(name='ChromosomeLoader', level=logging.DEBUG).get_logger()

class ChromosomeLoader:
    """
    Manages loading, splitting, and generator creation for Hi-C contact patch datasets.

    Supports two data pipelines:
        - Original (use_original=True): Loads patches from per-chromosome .npy files
          on disk, splitting by global integer indices.
        - Experiment (use_original=False): Loads sparse COO contact and ground truth
          patches from .npz files, converting them to dense arrays for training.

    In experiment mode, all combinations of quantile thresholds and normalization
    strategies are tracked and can be iterated over to create per-scenario generators.
    """
    def __init__(
        self,
        chromosomes: List[str] = None,
        data_dir: str = 'dataset/hela_100',
        patch_size: int = 64,
        resolution: int = RESOLUTION,
        batch_size: int = 32,
        shuffle_train: bool = True,
        split_ratios: Tuple[float, ...] = (0.7, 0.2, 0.1),
        include_diagonal: bool = False,
        use_original: bool = True,
        experiment: bool = False,
        thresholds: List[float] = None,
        normalizations: List[str] = None,
        is_train: bool = True,
        norm_method: str = 'all',
    ):
        """
        Args:
            chromosomes (List[str]): Chromosomes to load. Defaults to chr1–22 and X.
            data_dir (str): Base directory for dataset files. In experiment mode, this is
                            combined with threshold/normalization params to form dataset paths.
            patch_size (int): Height and width of each square contact patch. Defaults to 64.
            resolution (int): Hi-C resolution in base pairs. Defaults to RESOLUTION.
            batch_size (int): Number of samples per batch. Defaults to 32.
            shuffle_train (bool): If True, shuffles the training generator each epoch. Defaults to True.
            split_ratios (Tuple[float, ...]): Train/val/test split ratios. Use (0.7, 0.3) for
                                              train/val only, or (0.7, 0.2, 0.1) for all three splits.
            include_diagonal (bool): If True, loads diagonal distance flags alongside patches.
                                     Only used in the original pipeline. Defaults to False.
            use_original (bool): If True, uses the original file-based pipeline.
                                 If False, uses the sparse .npz experiment pipeline. Defaults to True.
            experiment (bool): If True, enables experiment mode with threshold/normalization combos.
            thresholds (List[float]): Quantile thresholds for experiment mode (e.g. [0.96]).
            normalizations (List[str]): Normalization strategies for experiment mode (e.g. ['log,zscore']).
            is_train (bool): If True, applies training-mode behavior during data loading. Defaults to True.
            norm_method (str): Normalization method key passed to the data pipeline. Defaults to 'all'.
        """
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
        self.is_train = is_train
        self.norm_method = norm_method

        self.experiment = experiment
        if self.experiment:
            self.thresholds = thresholds
            self.normalizations = normalizations
            # Precompute all (threshold, normalization) combinations for grid iteration
            self.all_sampling_param_combos = list(product(self.thresholds, self.normalizations))

      def compute_and_save_data_splits(
        self,
        overwrite: bool = False,
        upsample: bool = False,
        strategy: Optional[str] = None,
        factor: int = 1,
        threshold: int = 100,
    ) -> None:
        """
        Computes train/val/test splits from the first threshold/normalization combo
        and writes them to the metadata pickle for all experiment datasets.

        Patch indices are identical across all threshold/normalization combinations
        since only the contact values differ, not the patch positions.

        Args:
            overwrite (bool): If True, overwrites existing splits in metadata. Defaults to False.
            upsample (bool): If True, generates and saves upsampled training indices. Defaults to False.
            strategy (str | None): Upsampling strategy to apply ('random', 'balanced', or None).
            factor (int): Upsampling factor. Defaults to 1.
            threshold (int): Minimum positive pixel count for a patch to be eligible for upsampling.
                             Defaults to 100.
        """
        # Compute splits once using the first combo — patch positions are shared across all datasets
        train_idx, val_idx, test_idx, upsampled_idx = self.split_data(self.thresholds[0], self.normalizations[0],
                                                                      upsample, strategy, factor, threshold)

        # Write the same splits to each dataset's metadata file
        for quantile, method in self.all_sampling_param_combos:
            self.write_data_splits(quantile, method, train_idx, val_idx, test_idx, upsampled_idx, strategy, overwrite)

    def create_experiment_generators(
        self,
        upsample: bool = False,
        strategy: Optional[str] = None,
        factor: int = 1,
        threshold: int = 100,
        resolution: Optional[int] = None,
    ) -> dict:
        """
        Creates train/val/test generators for each (threshold, normalization) combination
        and returns them in a dict keyed by scenario name.

        Args:
            upsample (bool): If True, applies upsampling to the training split. Defaults to False.
            strategy (str | None): Upsampling strategy ('random', 'balanced', or None).
            factor (int): Upsampling factor. Defaults to 1.
            threshold (int): Minimum positive pixel count for upsampling eligibility. Defaults to 100.
            resolution (int | None): Resolution passed to each generator as a per-sample feature.

        Returns:
            dict: Keys are scenario strings (e.g. '0.96_log_zscore'), values are
                  [train_gen, val_gen, test_gen] lists of ChromosomeGenerators.
        """
        # a dict will be returned that contains the generators for all experiments
        generators = {}

        # for each experiment dataset
        for quantile, method in self.all_sampling_param_combos:
            # load the split data
            train_data, val_data, test_data = self.compute_data_split_indices_npzy(quantile, method, upsample, factor, threshold,
                                                                                   strategy)

            # create a generator for each split
            train_gen = self.create_generator(train_data, shuffle=self.shuffle_train, name='Train', resolution=resolution)
            val_gen = self.create_generator(val_data, shuffle=False, name='Val', resolution=resolution)
            test_gen = self.create_generator(test_data, shuffle=False, name='Test', resolution=resolution)

            # add the generators into the dictionary indexed by the quantile threshold and normalization method
            generators[f'{quantile}_{method}'] = [train_gen, val_gen, test_gen]

        return generators

     def combine_gens(
        self,
        generators: List[ChromosomeGenerator],
        shuffle: bool,
    ) -> ChromosomeGenerator:
        """
        Combines multiple generators into a single generator by concatenating their indices.
        Uses the first generator's configuration as the base.

        Args:
            generators (List[ChromosomeGenerator]): Generators to combine.
            shuffle (bool): If True, shuffles the combined indices at epoch end.

        Returns:
            ChromosomeGenerator: Single generator containing all indices from the input generators.
        """
        combined_indices = []
        for gen in generators:
            combined_indices.extend(gen.indices)
        
        base = generators[0]
        base.indices = combined_indices
        return base

    def create_generator(
        self,
        data: List,
        shuffle: bool,
        name: str,
        resolution: Optional[int] = None,
    ) -> ChromosomeGenerator:
        """
        Instantiates a ChromosomeGenerator for the given data split.

        Args:
            data (List): Patch indices or patch tuples for this split.
            shuffle (bool): If True, shuffles the generator at epoch end.
            name (str): Display name for the generator (e.g. 'Train', 'Val', 'Test').
            resolution (int | None): Resolution passed as a per-sample feature. Defaults to None.

        Returns:
            ChromosomeGenerator: Configured generator for the given split.
        """
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

    def load_split_data(
        self,
        quantile: float,
        method: str,
        upsample: bool = False,
        strategy: Optional[str] = None,
        factor: int = 1,
        threshold: int = 100,
    ) -> Tuple[List, List, List]:
        """
        Loads pre-saved train/val/test splits from metadata and assembles patch tuples
        by slicing sparse contact and ground truth matrices per chromosome.
        Optionally applies or reuses upsampled training data.

        Args:
            quantile (float): Quantile threshold identifying the dataset to load.
            method (str): Normalization method identifying the dataset to load.
            upsample (bool): If True, applies upsampling to the training split. Defaults to False.
            strategy (str | None): Upsampling strategy ('random', 'balanced', or None).
            factor (int): Upsampling factor. Defaults to 1.
            threshold (int): Minimum positive pixel count for upsampling eligibility. Defaults to 100.

        Returns:
            Tuple[List, List, List]: train_data, val_data, test_data — each a list of
                (contact_patch, ground_truth_patch, resolution) tuples.

        Raises:
            Exception: If no metadata file is found for the specified dataset path.
        """
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
            contacts = sparse.load_npz(path / f'{chromosome}_contacts.npz')
            ground_truth = sparse.load_npz(path / f'{chromosome}_ground_truth.npz')

            # print(dataset_metadata['splits'])
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

    def load_patches(
        self,
        chromosome: str,
        data_split: List[Tuple],
        contacts,
        ground_truth,
    ) -> List[Tuple[np.ndarray, np.ndarray, int]]:
        """
        Extracts dense contact and ground truth patches for a specific chromosome
        by slicing the sparse matrices using the patch boundary indices from the split.

        Args:
            chromosome (str): Chromosome identifier to filter patches by.
            data_split (List[Tuple]): Full data split containing (chromosome, patch_bounds) tuples.
            contacts: Sparse COO contact matrix of shape (N, patch_size, patch_size).
            ground_truth: Sparse COO ground truth matrix of shape (N, patch_size, patch_size).

        Returns:
            List[Tuple[np.ndarray, np.ndarray, int]]: List of
                (contact_patch, ground_truth_patch, resolution) tuples for this chromosome.
        """
        # Filter to only the patches belonging to this chromosome
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

    def create_chromosome_generators(
        self,
        upsample: bool = False,
        factor: int = 1,
        threshold: int = 100,
        strategy: str = "balanced",
    ) -> Tuple[ChromosomeGenerator, Optional[ChromosomeGenerator], Optional[ChromosomeGenerator]]:
        """
        Creates train/val/test generators using either the original image pipeline
        or the experiment sparse .npz pipeline depending on use_original.

        Args:
            upsample (bool): If True, applies upsampling to the training split. Defaults to False.
            factor (int): Upsampling factor. Defaults to 1.
            threshold (int): Minimum positive pixel count for upsampling eligibility. Defaults to 100.
            strategy (str): Upsampling strategy ('random' or 'balanced'). Defaults to 'balanced'.

        Returns:
            Tuple[ChromosomeGenerator, ChromosomeGenerator | None, ChromosomeGenerator | None]:
                Train, val, and test generators. Val and test may be None depending on split_ratios.
        """
        if self.use_original:
            train_data, val_data, test_data = self.compute_data_split_images()
        else:
            train_data, val_data, test_data = self.compute_data_split_indices_npzy(None, None, upsample, factor, threshold,
                                                                                   strategy)

        train_gen = self.create_generator(train_data, shuffle=self.shuffle_train, name='Train')
        val_gen = self.create_generator(val_data, shuffle=False, name='Val') if val_data is not None else None
        test_gen = self.create_generator(test_data, shuffle=False, name='Test') if test_data is not None else None

        return train_gen, val_gen, test_gen

    def split_data(
        self,
        quantile: float,
        method: str,
        upsample: bool = False,
        strategy: Optional[str] = None,
        factor: int = 1,
        threshold: int = 100,
    ) -> Tuple[List, List, List, Optional[List]]:
        """
        Loads all patch positions for each chromosome from sparse .npz files, shuffles them,
        and splits them into train/val/test sets according to split_ratios.

        Args:
            quantile (float): Quantile threshold identifying the dataset.
            method (str): Normalization method identifying the dataset.
            upsample (bool): If True, generates upsampled training indices. Defaults to False.
            strategy (str | None): Upsampling strategy ('random', 'balanced', or None).
            factor (int): Upsampling factor. Defaults to 1.
            threshold (int): Minimum positive pixel count for upsampling eligibility. Defaults to 100.

        Returns:
            Tuple[List, List, List, List | None]:
                train_indices, val_indices, test_indices, upsampled_indices (or None if not upsampling).
        """
        chrom_data = []
        path = Path(str(self.data_dir) + f'_{quantile}_quant_{method}_ps_{self.patch_size}_rs_{self.resolution}')

        for chromosome in self.chromosomes:
            patch_indices = sparse.load_npz(path / f'{chromosome}_contacts.npz').todense()

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

    def write_data_splits(
        self,
        quantile: float,
        method: str,
        train_indices: List,
        val_indices: List,
        test_indices: List,
        upsampled_idx: Optional[List],
        strategy: Optional[str],
        overwrite: bool = False,
    ) -> None:
        """
        Persists train/val/test split indices to the dataset metadata pickle.
        Skips writing if splits already exist and overwrite is False.

        Args:
            quantile (float): Quantile threshold identifying the target dataset.
            method (str): Normalization method identifying the target dataset.
            train_indices (List): Training split patch indices.
            val_indices (List): Validation split patch indices.
            test_indices (List): Test split patch indices.
            upsampled_idx (List | None): Upsampled training indices to save, or None.
            strategy (str | None): Upsampling strategy key used to store upsampled indices.
            overwrite (bool): If True, overwrites existing splits. Defaults to False.
        """
        path = Path(str(self.data_dir) + f'_{quantile}_quant_{method}_ps_{self.patch_size}_rs_{self.resolution}')
        metadata_path = path / 'dataset_metadata.pkl'

        if metadata_path.exists():
            with open(metadata_path, "rb") as f:
                dataset_metadata = pickle.load(f)
        else:
            dataset_metadata = {}

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
        

    def load_normalization_params(
        self,
        filepath: str = 'dataset/hela_100/dataset_metadata.pkl',
    ) -> dict:
        """
        Loads and returns the full dataset metadata dictionary from a pickle file.

        Args:
            filepath (str): Path to the metadata pickle file.

        Returns:
            dict: Full metadata dictionary including normalization params and splits.
        """
        with open(filepath, 'rb') as f:
            metadata = pickle.load(f)

        return metadata

    def compute_data_split_indices_npzy(
        self,
        quantile: Optional[float],
        method: Optional[str],
        upsample: bool = False,
        factor: int = 1,
        threshold: int = 100,
        strategy: Optional[str] = None,
    ) -> Tuple[List, Optional[List], Optional[List]]:
        """
        Loads all contact and ground truth patches from sparse .npz files, converts them
        to dense arrays, shuffles and splits them into train/val/test sets.

        When quantile and method are None, loads directly from data_dir without
        constructing an experiment-specific path.

        Args:
            quantile (float | None): Quantile threshold identifying the dataset, or None for base path.
            method (str | None): Normalization method identifying the dataset, or None for base path.
            upsample (bool): If True, appends upsampled patches to training data. Defaults to False.
            factor (int): Upsampling factor. Defaults to 1.
            threshold (int): Minimum positive pixel count for upsampling eligibility. Defaults to 100.
            strategy (str | None): Upsampling strategy ('random', 'balanced', or None).

        Returns:
            Tuple[List, List | None, List | None]:
                train_data, val_data (or None), test_data (or None) — each a list of
                (contact_patch, ground_truth_patch) dense array tuples.
        """
        chrom_data = []

        if quantile and method:
            path = Path(str(self.data_dir) + f'_{quantile}_quant_{method}_ps_{self.patch_size}_rs_{self.resolution}')
        else:
            path = Path(self.data_dir)

        for chromosome in self.chromosomes:
            contacts = sparse.load_npz(path / f'{chromosome}_contacts.npz')
            ground_truth = sparse.load_npz(path / f'{chromosome}_ground_truth.npz')

            for patch, label in zip(contacts, ground_truth):
                patch = patch.todense()
                label = label.todense()
                chrom_data.append((patch, label))

        total_samples = len(chrom_data)

        # Shuffle
        np.random.shuffle(chrom_data)

        # Compute data split indices
        train_end = int(self.split_ratios[0] * total_samples)
        val_end = train_end + int(self.split_ratios[1] * total_samples) if len(self.split_ratios) > 1 else None

        # Split the patches and ground truth
        train_data = chrom_data[:train_end]
        if upsample:
            train_data.extend(self.upsample_dataset(train_data, factor, threshold, strategy))

        val_data = chrom_data[train_end:val_end] if len(self.split_ratios) > 1 else None
        if len(self.split_ratios) == 3:
            test_data = chrom_data[val_end:]
        else:
            test_data = None

        return train_data, val_data, test_data

    def normalize_dataset(
        self,
        dataset: List[Tuple],
        mean: float,
        std: float,
        chrm_zscores: Optional[dict] = None,
    ) -> List[Tuple[np.ndarray, np.ndarray]]:
        """
        Applies log normalization followed by z-score standardization to a dataset.
        Supports both global and per-chromosome normalization statistics.

        Args:
            dataset (List[Tuple]): List of (chromosome, patch, label) tuples.
            mean (float): Global mean for z-score normalization.
            std (float): Global standard deviation for z-score normalization.
            chrm_zscores (dict | None): Per-chromosome {'mean', 'std'} stats. If provided,
                                        overrides the global mean/std. Defaults to None.

        Returns:
            List[Tuple[np.ndarray, np.ndarray]]: List of (normalized_patch, label) tuples.
        """
        result = []
        for item in dataset:
            # if len(item) == 3:
            chromosome, patch, label = item
            log_patch = np.log(patch + 1)

            if chrm_zscores is not None:
                # Use chromosome-specific stats
                chrom_mean = chrm_zscores[chromosome]['mean']
                chrom_std = chrm_zscores[chromosome]['std']
                z_patch = (log_patch - chrom_mean) / chrom_std
            else:
                # Use global stats
                z_patch = (log_patch - mean) / std

            result.append((z_patch, label))

        return result

    def upsample_dataset(
        self,
        data: List[Tuple],
        factor: int = 1,
        threshold: int = 100,
        strategy: Optional[str] = None,
    ) -> List[Tuple]:
        """
        Upsamples patches with sufficient positive (loop) pixels using the specified strategy.

        Args:
            data (List[Tuple]): List of (patch, label) or (patch, label, resolution) tuples.
            factor (int): Number of times to replicate or resample viable patches. Defaults to 1.
            threshold (int): Minimum number of positive pixels for a patch to be upsampling-eligible.
                             Defaults to 100.
            strategy (str | None): Upsampling strategy:
                - 'random': Randomly samples factor * len(viable) patches with replacement.
                - 'balanced': Replicates all viable patches exactly factor times.

        Returns:
            List[Tuple]: New upsampled patch tuples to be appended to the training set.

        Raises:
            Exception: If an unrecognized strategy is provided.
        """
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

    def compute_data_split_images(self) -> Tuple[np.ndarray, Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Computes train/val/test split indices for the original image-based pipeline by
        counting total patches across all chromosome .npy files and splitting by index.

        Returns:
            Tuple[np.ndarray, np.ndarray | None, np.ndarray | None]:
                Shuffled train, val, and test index arrays. Val and test may be None
                depending on split_ratios length.
        """
        # Total number of samples
        chrom_lengths = [np.load(os.path.join(self.data_dir, f'imageset.{chromosome}.npy'), mmap_mode='r').shape[0]
                         for chromosome in self.chromosomes]
        total_samples = sum(chrom_lengths)
        all_indices = np.arange(total_samples)

        # Shuffle
        np.random.shuffle(all_indices)

        # Compute split boundaries
        train_end = int(self.split_ratios[0] * total_samples)
        val_end = train_end + int(self.split_ratios[1] * total_samples) if len(self.split_ratios) > 1 else None

        train_idx = all_indices[:train_end]
        val_idx = all_indices[train_end:val_end] if len(self.split_ratios) > 1 else None
        test_idx = all_indices[val_end:] if len(self.split_ratios) > 2 else None

        return train_idx, val_idx, test_idx

    def visualize_data(
        self,
        generator: ChromosomeGenerator,
        limit: Optional[int] = None,
        plot_neg: bool = False,
    ) -> None:
        """
        Computes and plots positive/negative patch distributions and top positive patch heatmaps
        for a given generator.

        Args:
            generator (ChromosomeGenerator): Generator to visualize.
            limit (int | None): Maximum number of samples to include in the distribution plot.
            plot_neg (bool): If True, includes negative patches in the distribution plot.
        """
        # compute distributions
        train_counts = count_pos_neg_distributions(generator, patch_size=self.patch_size)

        # plot distributions
        plot_pixel_counts(train_counts, generator.name, limit, plot_neg, patch_size=self.patch_size)

        # plot the heatmap for the top 4 images
        plot_top_positive_patches(train_counts, 4,  patch_size=self.patch_size)
        plot_top_positive_patches(train_counts, 4, data_to_plot='y',  patch_size=self.patch_size)