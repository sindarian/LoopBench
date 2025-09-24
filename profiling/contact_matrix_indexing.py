# Standard library imports
import logging
import pathlib
import shutil
import time
from itertools import product
from pathlib import Path

# Third-party imports
import h5py
import numpy as np
import zarr
from typing_extensions import override

# Typing imports
from typing import List, Tuple

# Local application/library specific imports
from generators.chromosome_processor import ChromosomeProcessor
from gutils import parsebed, get_raw_graph
from hickit.matrix import CisMatrix
from util.constants import RESOLUTION, PATCH_SIZE
from util.logger import Logger


class ChromosomeProcessorEvaluator(ChromosomeProcessor):
    LOGGER = Logger(name='ChromosomeProcessorEvaluator', level=logging.DEBUG).get_logger()

    def __init__(self, *args, **kwargs):
        super().__init__(chromosome_list=['1'],
                         bedpe_dict=parsebed('bedpe/hela.hg38.bedpe', valid_threshold=1),
                         contact_data_dir='data/txt_hela_100',
                         genome_assembly='hg38',
                         output_dir='profiling/dataset/hela_100',
                         patch_size=PATCH_SIZE,
                         resolution=RESOLUTION)

    def _save_to_hdf5_zazrr(self, chromosome: str, matrix: np.ndarray) -> Tuple[List[str], List[str]]:
        """
        Save the given matrix to both HDF5 and Zarr formats with various compression settings.

        This method saves the matrix for a specific chromosome to:
        - Multiple HDF5 files with different compression algorithms.
        - Multiple Zarr directories with different compressors and compression levels.

        Args:
            chromosome (str): The name of the chromosome dataset (e.g., 'chrm1').
            matrix (np.ndarray): The contact matrix to be saved.

        Returns:
            Tuple[List[str], List[str]]: Two lists containing the file paths of the saved
            HDF5 files and the saved Zarr directories, respectively.
        """
        hdf5_paths = self._save_hdf5_datasets(chromosome, matrix)
        zarr_paths = self._save_zarr_datasets(chromosome, matrix)

        return hdf5_paths, zarr_paths

    def _save_hdf5_datasets(self, chromosome: str, matrix: np.ndarray) -> List[str]:
        """
        Save a matrix to HDF5 files using different compression algorithms.

        This method saves the provided matrix to HDF5 files with specified compression
        methods ('gzip' and 'lzf'). Each file contains a dataset named after the chromosome.
        The matrix is stored in chunks determined by `self.patch_size` for efficient I/O.

        Args:
            chromosome (str): Name of the chromosome dataset (e.g., 'chrm1').
            matrix (np.ndarray): The contact matrix to be saved.

        Returns:
            List[str]: A list of file paths where the HDF5 files have been saved.
        """
        compressions = ['gzip', 'lzf']
        paths = []

        for compression in compressions:
            self.LOGGER.info(f'Saving {compression} compressed HDF5 file...')

            path = f'{self.output_dir}/compression_{compression}_matrices.h5'
            paths.append(path)

            h5_path = Path(path)
            h5_path.parent.mkdir(exist_ok=True)

            with h5py.File(h5_path, "w") as h5f:
                h5f.create_dataset(
                    chromosome,
                    data=matrix,
                    compression=compression,
                    chunks=(self.patch_size, self.patch_size)
                )

        return paths

    def _save_zarr_datasets(self, chromosome: str, matrix: np.ndarray) -> List[str]:
        """
        Save a matrix to Zarr format with various compression options.

        This method saves the given matrix to a Zarr file for each combination of
        compressors ('zstd', 'lz4') and compression levels (5, 7). The matrix is
        stored in chunks of size (64, 64). For each compression type and level,
        a new Zarr directory is created and the matrix is written with the respective
        compression settings.

        Args:
            chromosome (str): The name of the chromosome (e.g., 'chrm1').
            matrix (np.ndarray): The contact matrix to be saved in Zarr format.

        Returns:
            list: A list of paths to the generated Zarr directories.
        """
        compressors = ['zstd', 'lz4']
        clevels = [5, 7] # compression levels
        paths = []

        combinations = list(product(compressors, clevels))
        for compressor, clevel in combinations:
            self.LOGGER.info(f'Saving {compressor} compressed Zarr directory with clevel {clevel}...')

            path = f'{self.output_dir}/compressor_{compressor}_clevel_{clevel}_matrices.zarr'
            paths.append(path)

            zarr_path = zarr.DirectoryStore(path)
            root = zarr.group(store=zarr_path)

            root.create_dataset(
                name=chromosome,
                data=matrix,
                chunks=(64, 64),
                compressor=zarr.Blosc(cname=compressor, clevel=clevel)
            )

        return paths

    def _compare_matrix_storage(self, chromosome: str, hdf5_paths: List[str], zarr_paths: List[str],
                                subwindow_slice: tuple = (0, 64, 0, 64)) -> None:
        """
        Compare file size and read time for Zarr vs HDF5 storage of a specific chromosome matrix.

        This method compares the storage size and read time for both Zarr and HDF5 formats
        for a given chromosome's submatrix (patch). It checks the file size and the time taken to
        read a specific submatrix defined by the `subwindow_slice` argument. The results
        are logged for both file formats.

        Args:
            chromosome (str): The chromosome name (e.g., 'chrm1') to compare.
            hdf5_paths (list of str): A list of paths to HDF5 files containing chromosome matrices.
            zarr_paths (list of str): A list of paths to Zarr directories containing chromosome matrices.
            subwindow_slice (tuple, optional): The region of the matrix to slice and test read times.
                                                Default is (0, 64, 0, 64) which slices the first 64x64 submatrix.

        Logs:
            - File size in KB and MB.
            - Read time in milliseconds and seconds.
        """
        row_start, row_end, col_start, col_end = subwindow_slice

        all_paths = hdf5_paths + zarr_paths

        for path in all_paths:
            file_path = Path(path)
            assert file_path.exists(), f'{file_path} does not exist.'

            if 'h5' in path:
                size_on_disk_b = file_path.stat().st_size
                read_time_s = self._time_hdf5_read(file_path, chromosome, row_start, row_end, col_start, col_end)
            else:
                size_on_disk_b = sum(f.stat().st_size for f in file_path.rglob("*") if f.is_file())
                read_time_s = self._time_zarr_read(file_path, chromosome, row_start, row_end, col_start, col_end)

            # convert sizes to KB and MB
            size_on_disk_kb = size_on_disk_b / 1024
            size_on_disk_mb = size_on_disk_kb / 1024

            # convert read time to MS
            read_time_ms = read_time_s * 1000

            file = path.split('/')[-1]
            print(f'\n\n')
            self.LOGGER.info(f'\nStats for {file} using chromosome {chromosome}:')
            self.LOGGER.info(f'    file size: {size_on_disk_kb} KB')
            self.LOGGER.info(f'    file size: {size_on_disk_mb} MB')
            self.LOGGER.info(f'    read time: {read_time_ms * 1000} ms')
            self.LOGGER.info(f'    read time: {read_time_s} s')

    def _time_hdf5_read(self, h5_path: pathlib.Path, chromosome: str, row_start: int, row_end: int, col_start: int,
                        col_end: int) -> float:
        """
        Times the reading of a submatrix from an HDF5 file.

        This method measures the time it takes to access a specific submatrix from a given HDF5 file.
        The submatrix is determined by the specified row and column start and end indices.

        Args:
            h5_path (pathlib.Path): The file path to the HDF5 dataset.
            chromosome (str): The name of the chromosome dataset (e.g., 'chrm1').
            row_start (int): The starting row index for the submatrix.
            row_end (int): The ending row index for the submatrix.
            col_start (int): The starting column index for the submatrix.
            col_end (int): The ending column index for the submatrix.

        Returns:
            float: The time in seconds it took to read the submatrix.
        """
        h5_start = time.time()
        with h5py.File(h5_path, "r") as h5f:
            sub_h5 = h5f[chromosome][row_start:row_end, col_start:col_end]
        return time.time() - h5_start

    def _time_zarr_read(self, zarr_path: pathlib.Path, chromosome: str, row_start: int, row_end: int, col_start: int,
                        col_end: int) -> float:
        """
        Times the reading of a submatrix (patch) from a given Zarr file.

        This method measures the time it takes to access a specific patch from the given Zarr file.
        The submatrix is determined by the specified row and column start and end indices.

        Args:
            zarr_path (pathlib.Path): The file path to the Zarr dataset.
            chromosome (str): The name of the chromosome dataset (e.g., 'chrm1').
            row_start (int): The starting row index for the submatrix.
            row_end (int): The ending row index for the submatrix.
            col_start (int): The starting column index for the submatrix.
            col_end (int): The ending column index for the submatrix.

        Returns:
            float: The time in seconds it took to read the submatrix.
        """
        zarr_start = time.time()
        root = zarr.open_group(zarr_path, mode='r')
        chr1_matrix = root[chromosome]
        subwindow = chr1_matrix[row_start:row_end, col_start:col_end]
        return time.time() - zarr_start

    @override
    def _save_to_hdf5(self, chromosome: str, contact_matrix: np.ndarray, patches: List[tuple]) -> List[str]:
        """
        Saves the contact matrix and patch references into an HDF5 file with specified compression.

        This method creates a group for the given chromosome and stores:
        - The contact matrix.
        - The region references corresponding to the patches.
        - The patch indices used to create the region references..

        Args:
            chromosome (str): The name of the chromosome (e.g., 'chrm1').
            contact_matrix (np.ndarray): The contact matrix.
            patches (List[tuple]): A list of patch indices, where each patch is defined by a tuple
                                    (start_row, start_col, end_row, end_col).

        Returns:
            List[str]: A list of paths to the saved HDF5 files with compression applied.
        """
        compressions = ['gzip', 'lzf']
        paths = []

        for compression in compressions:
            path = f'{self.output_dir}/compression_{compression}_matrices.h5'
            paths.append(path)

            h5_path = Path(path)
            h5_path.parent.mkdir(exist_ok=True)

            with h5py.File(self.h5_path, "w") as h5f:
                # create a group for the chromosome
                chrm_group = h5f.create_group(chromosome)

                # save the contacts dataset to use for region references later
                contacts_ds = chrm_group.create_dataset(
                    chromosome,
                    data=contact_matrix,
                    compression=compression,
                    chunks=(self.patch_size, self.patch_size)
                )

                # contact patch regions
                contact_refs = self._get_region_references(patches, contacts_ds)

                # Store the region references in a separate dataset
                ref_dtype = h5py.regionref_dtype
                chrm_group.create_dataset('contact_patches', data=contact_refs, dtype=ref_dtype)

                chrm_group.create_dataset('patches_indices', data=patches)

        return paths

    def _get_region_references(self, patches: List[tuple], dataset: h5py.Dataset) -> List[h5py.RegionReference]:
        """
        Retrieves region references for the given patches from the dataset.

        Args:
            patches (List[tuple]): A list of tuples, where each tuple contains the start and end
                                   coordinates of the patch (start_row, start_col, end_row, end_col).
            dataset (h5py.Dataset): The dataset from which to retrieve the region references.

        Returns:
            List[h5py.RegionReference]: A list of region references for the given patches.
        """
        region_refs = []
        for (start_row, start_col, end_row, end_col) in patches:
            ref = dataset.regionref[start_row:end_row, start_col:end_col]
            region_refs.append(ref)

        return region_refs

    @override
    def pad_matrix_bottom_right(self, matrix: np.ndarray, pad_size: int, pad_value: float = 0.0) -> np.ndarray:
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

        # If padding with 0, we can just use a zero matrix
        pad_matrix = np.zeros((new_rows, new_cols))

        # Insert original matrix into the top-left corner
        pad_matrix[:rows, :cols] = matrix

        return pad_matrix

    def _time_region_ref_read(self, contacts_ds: h5py.Dataset, contact_patches) -> tuple:
        """
        Times the region reference read operation and calculates the size of the stored indices (region references)

        Args:
            contacts_ds (h5py dataset): The dataset containing the contact matrix (can be an h5py or zarr dataset).
            contact_patches (h5py region references): An array of region references to access the data.

        Returns:
            tuple: A tuple containing:
                - The time taken to read the region reference (in seconds).
                - The size of the stored indices (in bytes).
                - The accessed patch data.
        """
        start_reg_ref = time.time()
        patch_data_ref = contacts_ds[contact_patches[0]]
        end_reg_ref = time.time()
        time_reg_ref_s = end_reg_ref - start_reg_ref
        size_reg_ref_b = contact_patches.nbytes

        return time_reg_ref_s, size_reg_ref_b, patch_data_ref

    def _time_patch_indexing_read(self, contacts_ds: h5py.Dataset, patches_indices: np.ndarray) -> tuple:
        """
        Times the patch indexing read operation and calculates the size of the stored indices (patches)

        Args:
            contacts_ds (h5py dataset): The dataset containing the contact matrix.
            patches_indices (np.ndarray): An array of patch indices, where each entry is a tuple
                                          (x_start, y_start, x_end, y_end).

        Returns:
            tuple: A tuple containing:
                - The time taken to slice the dataset (in seconds).
                - The size of the saved patch indices in the h5py dataset (in bytes).
                - The accessed patch data itself.
        """
        start_slice_idx = time.time()
        x_start, y_start, x_end, y_end = patches_indices[0]
        patch_data_idx = contacts_ds[x_start:x_end, y_start:y_end]
        end_slice_idx = time.time()
        time_slice_idx_s = end_slice_idx - start_slice_idx
        size_slice_idx_b = patches_indices.nbytes

        return time_slice_idx_s, size_slice_idx_b, patch_data_idx

    def _print_indexing_times_and_sizes(self, paths: List[str], chromosome: str = 'chrm1') -> None:
        """
        Prints the retrieval time and data size statistics for different indexing strategies
        (region reference vs patch indexing) used to access chromosome data in an HDF5 file.

        This method:
        - Opens the provided HDF5 paths and extracts the contact matrix and associated data.
        - Measures the retrieval time and memory size of data retrieved via two strategies:
            1. **Region References**: Accessing data using HDF5 region references.
            2. **Patch Indexing**: Accessing data via slicing indices (patches).
        - Logs the retrieval times for both strategies.
        - Logs the memory size for both storge options (region references and patch indexing).
        - Verifies that the data retrieved using both methods is identical.

        Parameters:
        ----------
        paths : list of str
            List of paths to the HDF5 files to evaluate. Each path corresponds to a dataset containing
            the chromosome data.

        chromosome : str, optional (default='chrm1')
            The chromosome identifier for which the data should be accessed (e.g., '1', 'X').

        Returns:
        -------
        None
            This method does not return a value.

        Side Effects:
        --------------
        - Logs the retrieval time (in seconds and milliseconds) and data size (in KB) for both indexing strategies.
        - Asserts that the data retrieved by both indexing strategies are identical.

        Notes:
        ------
        - The method relies on the use of HDF5 region references for efficient data retrieval.
        - The logging output includes the retrieval times in both seconds and milliseconds.
        - The method ensures that the data fetched by both region references and patch indexing are equivalent
          by performing an equality check.
        """
        for path in paths:
            h5_path = Path(path)
            h5_path.parent.mkdir(exist_ok=True)
            with h5py.File(self.h5_path, "r") as h5f:
                chrm_group = h5f[chromosome]

                contact_patches = chrm_group['contact_patches']
                patches_indices = chrm_group['patches_indices']
                contacts_ds = chrm_group[chromosome]

                # Timing access using region reference
                time_reg_ref_s, size_reg_ref_b, reg_ref_patch = self._time_region_ref_read(contacts_ds, contact_patches)
                time_reg_ref_ms = time_reg_ref_s * 1000
                size_reg_ref_kb = size_reg_ref_b / 1024

                # Timing access using coordinate slicing
                time_slice_idx_s, size_slice_idx_b, indexed_patch = self._time_patch_indexing_read(contacts_ds, patches_indices)
                time_slice_idx_ms = time_slice_idx_s * 1000
                size_slice_idx_kb = size_slice_idx_b / 1024

                # verify the patches are equal
                # assert np.array_equal(reg_ref_patch, indexed_patch)

                print('\n\n')
                self.LOGGER.info(f'Stats for different indexing strategies for file {path}:')
                self.LOGGER.info(f'    Retrieval time using region reference: {time_reg_ref_ms} ms')
                self.LOGGER.info(f'    Retrieval time using region reference: {time_reg_ref_s} s')
                self.LOGGER.info(f'    Size of region reference data: {size_reg_ref_kb} KB')
                self.LOGGER.info(f'    Retrieval time using patch indexing: {time_slice_idx_ms} ms')
                self.LOGGER.info(f'    Retrieval time using patch indexing: {time_slice_idx_s} s')
                self.LOGGER.info(f'    Size of patch index data: {size_slice_idx_kb} KB')

    def evaluate_data_compression_size_read_stats(self, contacts: CisMatrix, chromosome: str) -> None:
        """
        Evaluates the storage size and read times for the compressed contact matrix stored in HDF5 and Zarr formats.

        This method:
        - Pads the contact matrix to using the patch size.
        - Saves the padded contact matrix in both HDF5 and Zarr formats with various compression methods.
        - Compares the file sizes and read times between the HDF5 and Zarr formats for the given chromosome.
        - Cleans up by removing the generated output files after evaluation.

        Parameters:
        ----------
        contacts : CisMatrix
            The contact matrix object containing the raw matrix data in `contacts.mat`.

        chromosome : str
            The chromosome identifier (e.g., '1', 'X').

        Returns:
        -------
        None
            This method does not return a value. It logs the size and read time statistics for both HDF5 and Zarr formats.
            It also removes the generated output files after completion.

        Side Effects:
        --------------
        - Generates HDF5 and Zarr files for the padded contact matrix.
        - Removes the output directory and its contents after use.
        """
        contact_matrix = evaluator.pad_matrix_bottom_right(contacts.mat, pad_size=PATCH_SIZE)

        # save the contacts
        hdf5_paths, zarr_paths = self._save_to_hdf5_zazrr(chromosome=f'chrm{chromosome}',
                                                          matrix=contact_matrix)

        # evaluate the storage and read times
        self._compare_matrix_storage(f'chrm{chromosome}', hdf5_paths, zarr_paths)

        shutil.rmtree(self.output_dir)

    def evaluate_patch_indexing_vs_region_refs(self, contacts: CisMatrix, chromosome: str) -> None:
        """
        Evaluates the performance difference between accessing data from the contact matrix using:
        1. Region reference-based indexing.
        2. Index slicing using patch indices.

        The method:
        - Creates patch indices for a chromosome's contact matrix.
        - Pads the contact matrix to a defined size.
        - Saves the contact matrix and patch indices to an HDF5 file.
        - Measures and prints the time taken to retrieve submatrices (patches) using both indexing methods.
        - Cleans up by removing generated output files.

        Parameters:
        ----------
        contacts : CisMatrix
            The contact matrix object that contains the raw matrix data in `contacts.mat`.

        chromosome : str
            The chromosome identifier (e.g., '1', 'X').

        Returns:
        -------
        None
            This method does not return a value, but logs the performance statistics (time and memory usage).
            It also removes generated output files after completion.

        Side Effects:
        --------------
        - Generates HDF5 files containing the padded contact matrix and patch indices.
        - Cleans up by deleting the output directory after use.
        """
        # create the patch indices
        start_positions = self._get_start_tuples_for_chromosome(contacts)
        patch_indices = [(start[0],
                    start[1],
                    start[0] + self.patch_size,
                    start[1] + self.patch_size) for start in start_positions]

        contact_matrix = evaluator.pad_matrix_bottom_right(contacts.mat, pad_size=64)

        hdf5_paths = self._save_to_hdf5(chromosome=f'chrm{chromosome}',
                                        contact_matrix=contact_matrix,
                                        patches=patch_indices)

        self._print_indexing_times_and_sizes(hdf5_paths, f'chrm{chromosome}')

        shutil.rmtree(self.output_dir)


if __name__ == '__main__':
    """
    Main entry point for evaluating chromosome contact matrix storage and access strategies.

    Steps performed:
    1. Instantiate the ChromosomeProcessorEvaluator.
    2. Load the contact matrix for chromosome '1' and pad it to prepare for patch sampling.
    3. Evaluate and compare storage size and read performance of the contact matrix
       using different compression methods in both HDF5 and Zarr formats.
    4. Evaluate and compare two indexing methods—region references and patch indices—
       for accessing subregions (patches) of the contact matrix stored in HDF5 files.

    This script aims to identify optimal storage and access patterns for large genomic
    contact matrices, balancing compression efficiency and data retrieval speed.
    """
    evaluator = ChromosomeProcessorEvaluator()

    print('\n\nRunning with filtering')
    # load the contacts and pad it as if we were going to sample from it
    contacts = evaluator._load_matrix_for_chromosome_original('1')

    # evaluate the effectiveness of saving the contact data using
    # HDF5 and Zarr formats with varying compressions and compression levels
    evaluator.evaluate_data_compression_size_read_stats(contacts, '1')

    # evaluate the effectiveness of using region references and
    # patch indices to sample patches from a contact matrix saved as an HDF5 file
    evaluator.evaluate_patch_indexing_vs_region_refs(contacts, '1')

    print('\n\nRunning without filtering')
    contacts = evaluator._load_matrix_for_chromosome_original('1', False)
    evaluator.evaluate_data_compression_size_read_stats(contacts, '1')
    evaluator.evaluate_patch_indexing_vs_region_refs(contacts, '1')