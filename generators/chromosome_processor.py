import logging
import os
import pathlib
import pickle
import time
from itertools import product
import sparse

import numpy as np
from typing import List, Dict, Generator
from pathlib import Path

from scipy.sparse import csc_matrix, save_npz, hstack, vstack
from tqdm import tqdm

from gutils import get_raw_graph, block_sampling, block_sampling_mustache, create_ground_truth, get_raw_graph_optimized
from util.constants import PATCH_SIZE, RESOLUTION

from util.logger import Logger
from util.plotting import plotting

from typing import Tuple

LOGGER = Logger(name='ChromosomeSampler', level=logging.DEBUG).get_logger()

class ChromosomeProcessor:
    """
    Processes Hi-C contact matrices into patch-based datasets for loop prediction.

    Supports two processing pipelines:
        - Original (GILoop): Extracts fixed-size patches from raw contact matrices
          using block sampling, saving them as numpy arrays.
        - Experiment: Applies quantile thresholding and normalization to contact
          matrices, then extracts and saves patches as sparse COO arrays alongside
          ground truth labels and dataset metadata.

    Both pipelines support optional Mustache-preprocessed data as an additional
    or alternative data source.
    """

    def __init__(
        self,
        chromosome_list: List[str],
        bedpe_dict: Dict,
        contact_data_dir: str = 'data/txt_hela_100',
        genome_assembly: str = 'hg38',
        patch_size: int = PATCH_SIZE,
        resolution: int = RESOLUTION,
        output_dir: str = 'dataset/hela_100',
        plot_chrom: bool = False,
        use_mustache: bool = False,
        use_giloop: bool = True,
        only_giloop_diag: bool = False,
        only_mustache_diag: bool = False,
        h5_file: str = 'contact_matrices.h5',
        mustache_data_dir: str = 'data/mustache_outputs',
        experiment: bool = False,
        is_test: bool = False,
    ):
        """
        Args:
            chromosome_list (List[str]): Chromosomes to process (e.g. ['1', '2', 'X']).
            bedpe_dict (Dict): Parsed BEDPE loop annotations keyed by chromosome.
            contact_data_dir (str): Directory containing raw Hi-C contact text files.
            genome_assembly (str): Genome assembly name used to locate chrom size files (e.g. 'hg38').
            patch_size (int): Size of square patches to extract from contact matrices.
            resolution (int): Hi-C resolution in base pairs (e.g. 10000 for 10kb).
            output_dir (str): Directory to save processed patches, labels, and metadata.
            plot_chrom (bool): If True, plots sampled chromosomes and loop distributions.
            use_mustache (bool): If True, includes Mustache-preprocessed patches in sampling.
            use_giloop (bool): If True, includes GILoop-sampled patches in sampling.
            only_giloop_diag (bool): If True, restricts GILoop sampling to diagonal patches only.
            only_mustache_diag (bool): If True, restricts Mustache sampling to diagonal patches only.
            h5_file (str): Filename for the HDF5 contact matrix cache.
            mustache_data_dir (str): Directory containing pre-computed Mustache outputs.
            experiment (bool): If True, skips output directory creation (used in experiment pipelines).
            is_test (bool): If True, applies test-mode normalization when loading contact matrices.
        """
        self.chromosome_list = chromosome_list
        self.bedpe_dict = bedpe_dict
        self.contact_data_dir = contact_data_dir
        self.genome_assembly = genome_assembly
        self.patch_size = patch_size
        self.resolution = resolution
        self.output_dir = Path(output_dir)
        self.plot_chrom = plot_chrom
        self.use_mustache = use_mustache
        self.use_giloop = use_giloop
        self.only_giloop_diag = only_giloop_diag
        self.only_mustache_diag = only_mustache_diag
        self.h5_path = Path(os.path.join(self.output_dir, h5_file))
        self.mustache_data_dir = mustache_data_dir
        self.is_test = is_test

        self.experiment = experiment

        # Create output directory structure unless running in experiment mode
        if not self.experiment:
            self.output_dir.mkdir(parents=True, exist_ok=True)
            self.h5_path.parent.mkdir(parents=True, exist_ok=True)

    def _get_start_tuples_for_chromosome(self, matrix) -> List[Tuple[int, int]]:
        """
        Generates all valid (row, col) patch start coordinates for a chromosome matrix.
        Sampling is restricted to a 2Mb genomic window above the diagonal to focus on
        biologically relevant short-range interactions.

        Args:
            matrix: Loaded chromosome contact matrix with header metadata.

        Returns:
            List[Tuple[int, int]]: List of (row, col) start coordinates in base-pair units.
        """
        if matrix._filtered:
            header_len = len(matrix.get_cropped_headers())
        else:
            header_len = len(matrix.get_headers())
        segment_count = self._get_segment_count(header_len, self.patch_size)

        # Initialize an empty list to store the coordinate tuples.
        start_tuples = []

        # Iterate through each possible starting segment `i` for the first dimension.
        for i in range(segment_count):
            # Calculate the right-most edge for the second dimension `j`.
            # This limits sampling to a 2-megabase genomic window for efficiency and relevance.
            mb = 2000000
            right_edge = min(segment_count, int(i + (mb / self.resolution / self.patch_size)) + 2)  # tiled
            # Iterate from the diagonal (`i`) to the calculated right edge.
            for j in range(i, right_edge):
                # Append the valid (row, col) start coordinate tuple, multiplied by patch size.
                start_tuples.append((i * self.patch_size, j * self.patch_size))
        return start_tuples

    def _get_segment_count(self,
                           cropped_header_length: int,
                           patch_size: int
    ) -> int:
        """
        Calculates the number of non-overlapping segments needed to tile a chromosome.
        Uses ceiling division to ensure full coverage when the chromosome length is
        not perfectly divisible by the patch size.

        Args:
            cropped_header_length (int): Number of bins in the (optionally cropped) chromosome.
            patch_size (int): Size of each patch in bins.

        Returns:
            int: Number of segments required to cover the chromosome.
        """
        # Check if the length is not perfectly divisible by the patch size.
        if cropped_header_length % patch_size != 0:
            # If not, perform ceiling division by adding 1 to the integer division result.
            return int(cropped_header_length / patch_size) + 1
        else:
            # If it divides perfectly, return the exact division result.
            return int(cropped_header_length / patch_size)

    def _process_single_chromosome(
        self,
        chromosome: str,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Processes a single chromosome by sampling patches from GILoop and/or Mustache
        data sources and concatenating the results.

        Args:
            chromosome (str): Chromosome identifier (e.g. '1', 'X').

        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray]:
                - patches: Array of shape (N, patch_size, patch_size) containing contact patches.
                - diagonals: Array of shape (N,) containing diagonal distances per patch.
                - labels: Array of shape (N, patch_size, patch_size) containing ground truth labels.
        """
        LOGGER.info(f"Processing chromosome: {chromosome}")

        chromosome_patches = []
        chromosome_diagonals = []
        chromosome_labels = []

        if self.use_giloop:
            giloop_patches, giloop_diags, giloop_labels = self._sample_giloop(chromosome, only_diag=self.only_giloop_diag)
            chromosome_patches.append(giloop_patches)
            chromosome_diagonals.append(giloop_diags)
            chromosome_labels.append(giloop_labels)

        if self.use_mustache:
            mustache_patches, mustache_diags, mustache_labels = self._sample_mustache(chromosome, only_diag=self.only_mustache_diag)
            if len(mustache_patches) != 0 and len(mustache_diags) != 0 and len(mustache_labels) != 0:
                chromosome_patches.append(mustache_patches)
                chromosome_diagonals.append(mustache_diags)
                chromosome_labels.append(mustache_labels)

        patches_array = np.concatenate(chromosome_patches, axis=0)
        diagonals_array = np.concatenate(chromosome_diagonals, axis=0)
        labels_array = np.concatenate(chromosome_labels, axis=0)

        LOGGER.info(f'Completed chromosome {chromosome}: {patches_array.shape[0]} patches saved')
        return patches_array,diagonals_array,  labels_array

    def _sample_giloop(
        self,
        chromosome: str,
        only_diag: bool = False,
    ) -> Tuple[List, List, List]:
        """
        Samples contact patches from a chromosome matrix using the GILoop block sampling
        strategy.

        Args:
            chromosome (str): Chromosome identifier (e.g. '1', 'X').
            only_diag (bool): If True, restricts sampling to diagonal patches only.

        Returns:
            Tuple[List, List, List]:
                - patches: List of (patch_size, patch_size) float32 contact arrays.
                - diagonals: List of diagonal distances (col - row) per patch.
                - labels: List of (patch_size, patch_size) bool ground truth arrays.
        """
        LOGGER.info('Sampling for GILoop')
        # Load matrix for this chromosome
        matrix = self._load_matrix_for_chromosome_original(chromosome)

        # Get start tuples for this chromosome
        start_tuples = self._get_start_tuples_for_chromosome(matrix)
        if only_diag:
            start_tuples = [st for st in start_tuples if st[0] == st[1]]

        LOGGER.info(f'Found {len(start_tuples)} patches for chromosome {chromosome}')

        # init arrays
        patches = []
        diagonals = []
        labels = []

        # Create a copy of bedpe_list for this chromosome to avoid modification issues
        bedpe_copy = {k: set(v) for k, v in self.bedpe_dict.items()}

        # self.plot_chrom = True
        if self.plot_chrom:
            chrom_image, chrom_loops, _ = block_sampling(matrix, (0,), matrix.mat.shape[0], bedpe_copy)
            self.plot_sampled_crhom_and_labels(chromosome, chrom_image, chrom_loops, 'GILoop')

        # Process each patch
        for i, starts_tuple in enumerate(tqdm(start_tuples)):
            try:
                patch_start = (starts_tuple[0],) if starts_tuple[0] == starts_tuple[1] else starts_tuple
                subgraph, label, _ = block_sampling(matrix, patch_start, self.patch_size, bedpe_copy)

                # Handle diagonal vs off-diagonal cases
                is_on_diagonal = starts_tuple[0] == starts_tuple[1]
                if is_on_diagonal:
                    # Diagonal case - use full patch
                    patch = subgraph.astype(np.float32)
                    patch_label = label.astype(np.bool_)
                else:
                    # Off-diagonal case - slice to get cross-interactions
                    patch = subgraph[:self.patch_size, self.patch_size:].astype(np.float32)
                    patch_label = label[:self.patch_size, self.patch_size:].astype(np.bool_)

                patches.append(patch)
                diagonals.append(starts_tuple[1] - starts_tuple[0])
                labels.append(patch_label)

            except Exception as e:
                LOGGER.error(f'Error processing patch {i} at {starts_tuple}: {e}')
                continue

        return patches, diagonals, labels

    def _sample_mustache(
        self,
        chromosome: str,
        only_diag: bool = False,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Samples contact patches from pre-computed Mustache-processed data for a chromosome.
        Skips the chromosome silently if the expected image or label files are not found.

        Args:
            chromosome (str): Chromosome identifier (e.g. '1', 'X').
            only_diag (bool): If True, restricts sampling to diagonal patches only.

        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray]:
                - patches: Array of shape (N, patch_size, patch_size) of float32 contact patches.
                - diagonals: Array of shape (N,) of diagonal distances per patch.
                - labels: Array of shape (N, patch_size, patch_size) of ground truth labels.
        """
        LOGGER.info('Sampling for mustache')

        # init arrays
        patches = []
        diagonals = []
        labels = []

        # load the mustache data for the given chromosome
        image_file = f'mustache.{chromosome}.image.npy'
        label_file = f'mustache.{chromosome}.label.npy'

        image_path = os.path.join(self.mustache_data_dir, image_file)
        label_path = os.path.join(self.mustache_data_dir, label_file)

        if os.path.exists(image_path) and os.path.exists(label_path):
            mustache_image = np.load(image_path)
            mustache_labels = np.load(label_path)

            LOGGER.info(
                f'Loaded mustache data for chr{chromosome}: '
                f'image shape: {mustache_image.shape}, '
                f'labels shape: {mustache_labels.shape}, '
                f'total loops: {np.sum(mustache_labels)} ')

            start_tuples = self.get_mustache_starts(mustache_image)
            if only_diag:
                start_tuples = [st for st in start_tuples if st[0] == st[1]]

            if self.plot_chrom:
                chrom_image = block_sampling_mustache(mustache_image, (0,), mustache_image.shape[0], is_labels=False)
                chrom_loops = block_sampling_mustache(mustache_labels, (0,), mustache_labels.shape[0], is_labels=True)
                self.plot_sampled_crhom_and_labels(chromosome, chrom_image, chrom_loops, 'Mustache')

            # Process each patch
            for i, starts_tuple in enumerate(tqdm(start_tuples)):
                patch_start = (starts_tuple[0],) if starts_tuple[0] == starts_tuple[1] else starts_tuple
                sampled_image = block_sampling_mustache(mustache_image, patch_start, self.patch_size)
                final_label = block_sampling_mustache(mustache_labels, patch_start, self.patch_size, is_labels=True)

                final_patch = sampled_image.astype(np.float32)

                patches.append(final_patch)
                labels.append(final_label)

                diagonals.append(starts_tuple[1] - starts_tuple[0])

        # Convert to numpy arrays
        patches_array = np.array(patches)
        diagonals_array = np.array(diagonals)
        labels_array = np.array(labels)

        LOGGER.info(f'Total sampled loops from Mustache data: {np.sum(labels_array)}')
        return patches_array, diagonals_array, labels_array

    def plot_sampled_crhom_and_labels(
        self,
        chromosome: str,
        chrom_image: np.ndarray,
        chrom_loops: np.ndarray,
        data_origin: str,
    ) -> None:
        """
        Plots the loop distance histogram for a sampled chromosome.

        Args:
            chromosome (str): Chromosome identifier used in the plot title.
            chrom_image (np.ndarray): Sampled contact matrix (currently unused).
            chrom_loops (np.ndarray): Ground truth loop label matrix.
            data_origin (str): Data source label used for the plot title (e.g. 'hela', 'gm12878').
        """

        if data_origin == 'hela':
            title_origin = 'HeLa'
        elif data_origin == 'gm12878':
            title_origin = 'GM12878'
        loop_dist_title = f'{title_origin} Loop Distance from Main Diagonal (Chrom-{chromosome})'
        plotting.plot_diagonal_distance_histogram(chrom_loops, title=loop_dist_title, chrom=chromosome, dataset_name=data_origin)

    def get_mustache_starts(self, matrix: np.ndarray) -> List[Tuple[int, int]]:
        """
        Generates all valid (row, col) patch start coordinates for a Mustache matrix,
        restricted to a 2Mb genomic window above the diagonal.

        Args:
            matrix (np.ndarray): 2D Mustache contact matrix.

        Returns:
            List[Tuple[int, int]]: List of (row, col) start coordinates in base-pair units.
        """
        segment_count = self._get_segment_count(matrix.shape[0], self.patch_size)

        # Initialize an empty list to store the coordinate tuples.
        start_tuples = []

        # Iterate through each possible starting segment `i` for the first dimension.
        for i in range(segment_count):
            # Calculate the right-most edge for the second dimension `j`.
            # This limits sampling to a 2-megabase genomic window for efficiency and relevance.
            mb = 2000000
            right_edge = min(segment_count, int(i + (mb / self.resolution / self.patch_size)) + 2)  # tiled
            # Iterate from the diagonal (`i`) to the calculated right edge.
            for j in range(i, right_edge):
                # Append the valid (row, col) start coordinate tuple, multiplied by patch size.
                start_tuples.append((i * self.patch_size, j * self.patch_size))
        return start_tuples

    def _chromosome_patch_generator(self) -> Generator[Tuple[str, np.ndarray, np.ndarray, np.ndarray], None, None]:
        """
        Lazily yields processed patch data for each chromosome in chromosome_list.

        Yields:
            Tuple[str, np.ndarray, np.ndarray, np.ndarray]:
                - chromosome: Chromosome identifier.
                - patches: Contact patch array of shape (N, patch_size, patch_size).
                - diagonals: Diagonal distance array of shape (N,).
                - labels: Ground truth label array of shape (N, patch_size, patch_size).
        """
        for chromosome in self.chromosome_list:
            start = time.time()
            patches, diagonals, labels = self._process_single_chromosome(chromosome)
            end = time.time()
            LOGGER.info(
                f'Time taken to sample the chrom {chromosome}: {end - start} seconds, {(end - start) / 60} minutes; original method used: true')
            yield chromosome, patches, diagonals, labels

    def save_chromosome_data(
        self,
        chromosome: str,
        patches: np.ndarray,
        diagonals: np.ndarray,
        labels: np.ndarray,
    ) -> None:
        """
        Saves patches, diagonals, and labels for a chromosome as separate .npy files.

        Args:
            chromosome (str): Chromosome identifier used in the output filenames.
            patches (np.ndarray): Contact patch array of shape (N, patch_size, patch_size).
            diagonals (np.ndarray): Diagonal distance array of shape (N,).
            labels (np.ndarray): Ground truth label array of shape (N, patch_size, patch_size).
        """

        # Create filenames
        patches_file = self.output_dir / f'imageset.{chromosome}.npy'
        diagonals_file = self.output_dir / f'diagonals.{chromosome}.npy'
        labels_file = self.output_dir / f'labels.{chromosome}.npy'

        # Save arrays
        np.save(patches_file, patches)
        np.save(diagonals_file, diagonals)
        np.save(labels_file, labels)

        LOGGER.info(f'Saved chromosome {chromosome}: {patches_file} {patches.shape}, '
              f'{diagonals_file} {diagonals.shape},'
              f' {labels_file} {labels.shape}')

    def process_all_chromosomes_as_patches(self) -> None:
        """
        Runs the original GILoop patch-based pipeline over all chromosomes.
        Patches, diagonals, and labels are saved as .npy files in the output directory.
        """
        LOGGER.info(f'Starting processing of {len(self.chromosome_list)} chromosomes')
        LOGGER.info(f'Output directory: {self.output_dir}')

        total_patches = 0
        successful_chromosomes = 0

        for chromosome, patches, diagonals, labels in self._chromosome_patch_generator():
            self.save_chromosome_data(chromosome, patches, diagonals, labels)
            total_patches += patches.shape[0]
            successful_chromosomes += 1

        LOGGER.info(f'\nProcessing complete:')
        LOGGER.info(f'  Successfully processed: {successful_chromosomes}/{len(self.chromosome_list)} chromosomes')
        LOGGER.info(f'  Total patches saved: {total_patches}')
        LOGGER.info(f'  Files saved in: {self.output_dir}')

    def process_all_chromosomes_as_chromosomes(
        self,
        threshold: float = 0.75,
        normalization: str = 'log2,clip',
    ) -> None:
        """
        Runs the experiment pipeline over all chromosomes. Applies quantile thresholding
        and normalization, extracts sparse patches and ground truth, then saves them
        alongside dataset metadata.

        Args:
            threshold (float): Quantile threshold for contact value filtering. Defaults to 0.75.
            normalization (str): Normalization pipeline to apply (e.g. 'log2,clip'). Defaults to 'log2,clip'.
        """
        for chromosome in tqdm(self.chromosome_list, desc=f'Processing chromosomes'):
            start = time.time()
            contacts, chrom_upper_bound, mean, std = self._load_matrix_for_chromosome_optimized(chromosome, threshold, normalization)

            padded_contacts, padded_ground_truth, patch_indices = self.create_contact_dependent_data(contacts, chromosome)

            # Extract and convert patches to sparse COO format to reduce memory usage
            dense_patches = self.extract_patches(padded_contacts, patch_indices)
            contacts_path = Path(self.output_dir) / f'dense_{chromosome}_contacts.npz'
            sparse_patches = sparse.COO.from_numpy(dense_patches)

            dense_ground_truth = self.extract_patches(padded_ground_truth, patch_indices)
            sparse_ground_truth = sparse.COO.from_numpy(dense_ground_truth)

            end = time.time()
            LOGGER.info(
                f'Time taken to sample the chrom {chromosome}: {end - start} seconds, {(end - start) / 60} minutes; original method used: false')

            # np.save(contacts_path, dense_patches)

            metadata = None
            if mean and std:
                metadata = \
                    {
                        'mean': mean,
                        'std': std
                    }

            self._save(chromosome=chromosome,
                       contact_matrix=padded_contacts,
                       ground_truth=padded_ground_truth,
                       patches=patch_indices,
                       output_dir=self.output_dir,
                       sparse_patches=sparse_patches,
                       sparse_ground_truth=sparse_ground_truth,
                       metadata=metadata)

    def extract_patches(
        self,
        padded_contacts: np.ndarray,
        patch_indices: List[Tuple[int, int, int, int]],
    ) -> np.ndarray:
        """
        Extracts fixed-size patches from a padded 2D contact matrix.

        Args:
            padded_contacts (np.ndarray): 2D contact matrix of shape (height, width).
            patch_indices (List[Tuple[int, int, int, int]]): List of (start_x, start_y, end_x, end_y)
                patch boundary tuples.

        Returns:
            np.ndarray: 3D array of shape (num_patches, patch_size, patch_size) in float32.
        """
        num_patches = len(patch_indices)

        # Initialize 3D matrix
        patches = np.zeros((num_patches, PATCH_SIZE, PATCH_SIZE), dtype=np.float32)

        # Extract each patch
        for i, (start_x, start_y, end_x, end_y) in enumerate(patch_indices):
            patches[i] = padded_contacts[start_x:end_x, start_y:end_y]

        return patches

    def run_sampling_experiments(
        self,
        thresholds: List[float],
        normalizations: List[str],
    ) -> None:
        """
        Runs the experiment sampling pipeline for all combinations of chromosomes,
        quantile thresholds, and normalization strategies. Each combination is saved
        to a uniquely named output directory encoding its parameters.

        Args:
            thresholds (List[float]): Quantile thresholds to apply during sampling.
            normalizations (List[str]): Normalization pipelines to apply (e.g. ['log,zscore', 'clip']).
        """
        all_sampling_param_combos = product(self.chromosome_list, thresholds, normalizations)

        for sampling_param_group in all_sampling_param_combos:
            chromosome = sampling_param_group[0]
            threshold = sampling_param_group[1]
            norm_method = sampling_param_group[2]
            LOGGER.info(f'Processing chromosome {chromosome} with: threshold={threshold}, normalization={norm_method}, patch size={self.patch_size}, resolution={self.resolution}')

            contacts, chrom_upper_bound, mean, std = self._load_matrix_for_chromosome_optimized(chromosome, threshold, norm_method)

            padded_contacts, padded_ground_truth, patch_indices = self.create_contact_dependent_data(contacts, chromosome)

            # Build the output path encoding all sampling parameters
            path = Path((str(self.output_dir)) + f'_{threshold}_quant_{norm_method.replace(",", "_")}_ps_{self.patch_size}_rs_{self.resolution}')
            LOGGER.info(f'Saving sampled data in {str(path)}')

            if not path.exists():
                path.mkdir(parents=True, exist_ok=True)

            dense_patches = self.extract_patches(padded_contacts, patch_indices)
            sparse_patches = sparse.COO.from_numpy(dense_patches)

            dense_ground_truth = self.extract_patches(padded_ground_truth, patch_indices)
            sparse_ground_truth = sparse.COO.from_numpy(dense_ground_truth)

            metadata = None
            if mean and std:
                metadata = \
                    {
                        'mean': mean,
                        'std': std
                    }

            self._save(chromosome=chromosome,
                       contact_matrix=padded_contacts,
                       ground_truth=padded_ground_truth,
                       patches=patch_indices,
                       output_dir=path,
                       sparse_patches=sparse_patches,
                       sparse_ground_truth=sparse_ground_truth,
                       metadata=metadata)

    def create_contact_dependent_data(
        self,
        contacts,
        chromosome: str = None,
    ) -> Tuple[np.ndarray, np.ndarray, List[Tuple[int, int, int, int]]]:
        """
        Derives ground truth labels and patch indices from a loaded contact matrix,
        then pads both the contact matrix and ground truth to ensure full patch coverage.

        Args:
            contacts: Loaded contact matrix object with a `.mat` attribute.
            chromosome (str): Chromosome identifier used for logging.

        Returns:
            Tuple[np.ndarray, np.ndarray, List[Tuple[int, int, int, int]]]:
                - padded_contacts: Padded contact matrix as a 2D float array.
                - padded_ground_truth: Padded ground truth label matrix as a 2D bool array.
                - patch_indices: List of (start_x, start_y, end_x, end_y) patch boundary tuples.
        """
        # create the ground truth using the contacts and bedfile
        ground_truth, _ = create_ground_truth(contacts, contacts.mat.shape[0], self.bedpe_dict, False)

        # self.plot_sampled_crhom_and_labels(chromosome, None, ground_truth, self.contact_data_dir.split('_')[-2])

        # create the patch indices
        start_positions = self._get_start_tuples_for_chromosome(contacts)
        LOGGER.info(f'Found {len(start_positions)} patches for chromosome {chromosome}')
        patch_indices = [(start[0],
                          start[1],
                          start[0] + self.patch_size,
                          start[1] + self.patch_size) for start in start_positions]

        # Pad the contact and ground truth data by the patch size
        padded_contacts = self.pad_matrix_bottom_right(contacts.mat,
                                                       pad_size=self.patch_size)
        padded_ground_truth = self.pad_matrix_bottom_right(ground_truth,
                                                           pad_size=self.patch_size).astype(bool)

        return padded_contacts, padded_ground_truth, patch_indices

    def _load_matrix_for_chromosome_optimized(
        self,
        chromosome_name: str,
        threshold: float = 0.75,
        normalization: str = 'log2,clip',
    ) -> Tuple:
        """
        Loads and normalizes a Hi-C contact matrix for a chromosome using the optimized pipeline.

        Args:
            chromosome_name (str): Chromosome identifier (e.g. '1', 'X').
            threshold (float): Quantile threshold for contact value filtering. Defaults to 0.75.
            normalization (str): Normalization pipeline to apply (e.g. 'log2,clip'). Defaults to 'log2,clip'.

        Returns:
            Tuple: (contact_matrix, chrom_upper_bound, mean, std)
        """
        # load the genome assembly file
        chrom_size_path = '{}.chrom.sizes'.format(self.genome_assembly)

        # Load the high-resolution ("image") matrix
        image_matrix, chrom_upper_bound, mean, std = get_raw_graph_optimized(chromosome_name, self.contact_data_dir,
                                                                  self.resolution, chrom_size_path, self.plot_chrom,
                                                                  threshold, normalization, self.is_test)

        return image_matrix, chrom_upper_bound, mean, std

    def _load_matrix_for_chromosome_original(
        self,
        chromosome_name: str,
        filter: bool = True,
    ):
        """
        Loads a raw Hi-C contact matrix for a chromosome using the original GILoop pipeline.

        Args:
            chromosome_name (str): Chromosome identifier (e.g. '1', 'X').
            filter (bool): If True, applies default filtering to the contact matrix. Defaults to True.

        Returns:
            Contact matrix object compatible with block_sampling and header methods.
        """
        chrom_size_path = '{}.chrom.sizes'.format(self.genome_assembly)
        image_matrix = get_raw_graph(chromosome_name, self.contact_data_dir, self.resolution, chrom_size_path, filter)
        return image_matrix

    def pad_matrix_bottom_right(
        self,
        matrix: np.ndarray,
        pad_size: int,
        pad_value: int = 0,
    ) -> np.ndarray:
        """
        Pads a 2D matrix on the bottom and right edges with a constant value.
        Used to prevent out-of-bounds slicing when extracting patches near chromosome edges.

        Args:
            matrix (np.ndarray): 2D input matrix to pad.
            pad_size (int): Number of rows/columns to add to the bottom and right.
            pad_value (int): Constant value to fill the padded region. Defaults to 0.

        Returns:
            np.ndarray: Padded matrix of shape (rows + pad_size, cols + pad_size).
        """
        padded = np.pad(
            matrix,
            pad_width=((0, pad_size), (0, pad_size)),
            mode='constant',
            constant_values=pad_value
        )

        return padded

    def _save(
        self,
        chromosome: str,
        contact_matrix: csc_matrix,
        ground_truth: csc_matrix,
        patches: List[Tuple[int, int, int, int]],
        output_dir: pathlib.Path,
        metadata: dict = None,
        sparse_patches=None,
        sparse_ground_truth=None,
    ) -> None:
        """
        Saves sparse contact patches, sparse ground truth, and optional metadata for a
        chromosome to the specified output directory. Metadata is accumulated in a shared
        dataset_metadata.pkl file, updating any existing entries for the chromosome.

        Args:
            chromosome (str): Chromosome identifier used in output filenames.
            contact_matrix (csc_matrix): Full padded contact matrix (not saved directly).
            ground_truth (csc_matrix): Full padded ground truth matrix (not saved directly).
            patches (List[Tuple[int, int, int, int]]): Patch boundary tuples (not saved directly).
            output_dir (pathlib.Path): Directory to save output files.
            metadata (dict | None): Optional per-chromosome metadata (e.g. mean, std).
            sparse_patches: Sparse COO array of contact patches to save.
            sparse_ground_truth: Sparse COO array of ground truth patches to save.
        """
        # Save sparse contact patches and ground truth as .npz files
        contacts_path = Path(output_dir) / f'{chromosome}_contacts.npz'
        sparse.save_npz(contacts_path, sparse_patches)

        ground_truth_path = Path(output_dir) / f'{chromosome}_ground_truth.npz'
        sparse.save_npz(ground_truth_path, sparse_ground_truth)

        # Accumulate per-chromosome metadata into the shared dataset metadata pickle
        metadata_path = Path(output_dir) / f'dataset_metadata.pkl'
        if metadata:
            if metadata_path.exists():
                with open(metadata_path, "rb") as f:
                    dataset_metadata = pickle.load(f)
            else:
                dataset_metadata = {}

            dataset_metadata[chromosome] = metadata

            with open(metadata_path, "wb") as f:
                pickle.dump(dataset_metadata, f)