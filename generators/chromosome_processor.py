import logging
import os
import pathlib
import pickle
from itertools import product

import numpy as np
from typing import List, Dict, Generator
from pathlib import Path

from scipy.sparse import csc_matrix, save_npz
from tqdm import tqdm

from gutils import get_raw_graph, block_sampling, block_sampling_mustache, create_ground_truth, get_raw_graph_optimized
from util.constants import PATCH_SIZE, RESOLUTION

from util.logger import Logger
from util.plotting import plotting

from typing import Tuple

LOGGER = Logger(name='ChromosomeSampler', level=logging.DEBUG).get_logger()

class ChromosomeProcessor:
    """TensorFlow data iterator for processing Hi-C chromosomes and saving to files."""

    def __init__(self,
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
                 experiment: bool = False):

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
        self.experiment = experiment

        # Create output directory
        if not self.experiment:
            self.output_dir.mkdir(parents=True, exist_ok=True)
            self.h5_path.parent.mkdir(parents=True, exist_ok=True)

    def _get_start_tuples_for_chromosome(self, matrix):
        """Generate start tuples for a single chromosome."""
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

    def _get_segment_count(self, cropped_header_length: int, patch_size: int) -> int:
        """Calculate number of segments needed to cover the chromosome."""
        # Check if the length is not perfectly divisible by the patch size.
        if cropped_header_length % patch_size != 0:
            # If not, perform ceiling division by adding 1 to the integer division result.
            return int(cropped_header_length / patch_size) + 1
        else:
            # If it divides perfectly, return the exact division result.
            return int(cropped_header_length / patch_size)

    def _process_single_chromosome(self, chromosome: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Process a single chromosome and return patches and labels."""
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

        LOGGER.info(f'  Completed chromosome {chromosome}: {patches_array.shape[0]} patches saved')
        return patches_array,diagonals_array,  labels_array

    def _sample_giloop(self, chromosome: str, only_diag: bool = False):
        LOGGER.info('sampling for GILoop...')
        # Load matrix for this chromosome
        matrix = self._load_matrix_for_chromosome_original(chromosome)

        # Get start tuples for this chromosome
        start_tuples = self._get_start_tuples_for_chromosome(matrix)
        if only_diag:
            start_tuples = [st for st in start_tuples if st[0] == st[1]]

        LOGGER.info(f'  Found {len(start_tuples)} patches for chromosome {chromosome}')

        # init arrays
        patches = []
        diagonals = []
        labels = []

        # Create a copy of bedpe_list for this chromosome to avoid modification issues
        bedpe_copy = {k: set(v) for k, v in self.bedpe_dict.items()}

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
                LOGGER.error(f'  Error processing patch {i} at {starts_tuple}: {e}')
                continue

        return patches, diagonals, labels

    def _sample_mustache(self, chromosome: str, only_diag: bool = False):
        LOGGER.info('sampling for mustache...')

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

    def plot_sampled_crhom_and_labels(self, chromosome, chrom_image, chrom_loops, data_origin):
        loop_actual_title = f'CHRM-{chromosome} ({data_origin}): All Loops'
        plotting.plot_heatmap(chrom_loops, title=loop_actual_title)

        loop_dist_title = f'CHRM-{chromosome} ({data_origin}): Loop Distance from Main Diagonal'
        plotting.plot_diagonal_distance_histogram(chrom_loops, title=loop_dist_title)

    def get_mustache_starts(self, matrix):
        """Generate start tuples for a single chromosome."""
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
        """Generator that yields (chromosome_name, patches, labels) for each chromosome."""
        for chromosome in self.chromosome_list:
            patches, diagonals, labels = self._process_single_chromosome(chromosome)
            yield chromosome, patches, diagonals, labels

    def save_chromosome_data(self, chromosome: str, patches: np.ndarray, diagonals: np.ndarray, labels: np.ndarray):
        """Save patches and labels for a chromosome to separate numpy files."""

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

    def process_all_chromosomes_as_patches(self):
        """Process all chromosomes and save to files."""
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

    def process_all_chromosomes_as_chromosomes(self, threshold=0.75, normalization='log2,clip'):
        for chromosome in tqdm(self.chromosome_list, desc=f'Processing chromosomes'):
            # load the contacts
            contacts, chrom_upper_bound = self._load_matrix_for_chromosome_optimized(chromosome, threshold, normalization)

            padded_contacts, padded_ground_truth, patch_indices = self.create_contact_dependent_data(contacts)

            self._save(chromosome=chromosome,
                       contact_matrix=padded_contacts,
                       ground_truth=padded_ground_truth,
                       patches=patch_indices,
                       output_dir=self.output_dir)

    def run_sampling_experiments(self, thresholds, normalizations):
        all_sampling_param_combos = product(self.chromosome_list, thresholds, normalizations)

        for sampling_param_group in all_sampling_param_combos:
            chromosome = sampling_param_group[0]
            threshold = sampling_param_group[1]
            norm_method = sampling_param_group[2]
            print(f'\nProcessing chromosome {chromosome} with threshold {threshold} and {norm_method} normalization')

            # load the contacts
            contacts, chrom_upper_bound = self._load_matrix_for_chromosome_optimized(chromosome, threshold, norm_method)

            padded_contacts, padded_ground_truth, patch_indices = self.create_contact_dependent_data(contacts)

            # create a path just for this sampled data
            path = Path((str(self.output_dir)) + f'_{threshold}_quant_{norm_method.replace(",", "_")}')
            if not path.exists():
                path.mkdir(parents=True, exist_ok=True)

            chrom_metadata = \
                {
                    'quantile_threshold': threshold,
                    'upper_bound': chrom_upper_bound
                }

            self._save(chromosome=chromosome,
                       contact_matrix=padded_contacts,
                       ground_truth=padded_ground_truth,
                       patches=patch_indices,
                       metadata=chrom_metadata,
                       output_dir=path)

    def create_contact_dependent_data(self, contacts):
        # create the ground truth using the contacts and bedfile
        ground_truth, _ = create_ground_truth(contacts, contacts.mat.shape[0], self.bedpe_dict, False)

        # create the patch indices
        start_positions = self._get_start_tuples_for_chromosome(contacts)
        patch_indices = [(start[0],
                          start[1],
                          start[0] + self.patch_size,
                          start[1] + self.patch_size) for start in start_positions]

        padded_contacts = self.pad_matrix_bottom_right(contacts.mat,
                                                       pad_size=64)
        padded_ground_truth = self.pad_matrix_bottom_right(ground_truth,
                                                           pad_size=64).astype(bool)

        return padded_contacts, padded_ground_truth, patch_indices

    # Define a function to load matrix for each chromosome
    def _load_matrix_for_chromosome_optimized(self, chromosome_name, threshold=0.75, normalization='log2,clip'):
        # load the genome assembly file
        chrom_size_path = '{}.chrom.sizes'.format(self.genome_assembly)

        # Load the high-resolution ("image") matrix
        image_matrix, chrom_upper_bound = get_raw_graph_optimized(chromosome_name, self.contact_data_dir,
                                                                  self.resolution, chrom_size_path, self.plot_chrom,
                                                                  threshold, normalization)

        return image_matrix, chrom_upper_bound

    def _load_matrix_for_chromosome_original(self, chromosome_name, filter=True):
        # load the genome assembly file
        chrom_size_path = '{}.chrom.sizes'.format(self.genome_assembly)

        # Load the high-resolution ("image") matrix
        image_matrix = get_raw_graph(chromosome_name, self.contact_data_dir, self.resolution, chrom_size_path, filter)

        return image_matrix

    def pad_matrix_bottom_right(self, matrix: csc_matrix, pad_size: int) -> csc_matrix:
        """
        Pad the given sparse CSC matrix on the bottom and right with a specified value.

        Parameters:
            matrix (csc_matrix): The input sparse matrix.
            pad_size (int): The number of rows and columns to add.
            pad_value (int): The value to use for padding. Default is -1.

        Returns:
            csc_matrix: The padded sparse matrix.
        """
        rows, cols = matrix.shape
        new_rows = rows + pad_size
        new_cols = cols + pad_size

        # Create a new sparse matrix filled with the pad_value
        # This is done by constructing a COOrdinate matrix with all the padding values
        pad_matrix = csc_matrix((new_rows, new_cols))

        # Insert original matrix into the top-left corner
        pad_matrix[:rows, :cols] = matrix

        return pad_matrix

    def _save(self, chromosome: str, contact_matrix: csc_matrix, ground_truth: csc_matrix,
              patches: List[Tuple[int, int, int, int]], output_dir: pathlib.Path, metadata: dict = None):
        contacts_path = Path(output_dir) / f'{chromosome}_contacts.npz'
        save_npz(contacts_path, contact_matrix)

        ground_truth_path = Path(output_dir) / f'{chromosome}_ground_truth.npz'
        save_npz(ground_truth_path, ground_truth)

        patches_path = Path(output_dir) / f'{chromosome}_patches'
        np.save(patches_path, patches)

        if metadata:
            metadata_path = Path(output_dir) / f'dataset_metadata.pkl'
            if metadata_path.exists():
                # load existing pickle
                with open(metadata_path, "rb") as f:
                    dataset_metadata = pickle.load(f)
            else:
                dataset_metadata = {}

            # add/update entry for this chromosome
            dataset_metadata[chromosome] = metadata

            # save back to pickle
            with open(metadata_path, "wb") as f:
                pickle.dump(dataset_metadata, f)
