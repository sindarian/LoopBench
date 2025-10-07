import runpy
import sys
from collections import defaultdict

import numpy as np
from keras.layers import Rescaling
from scipy.sparse import diags, csc_matrix, csr_matrix
import os
from subprocess import run
import pandas as pd

import mustace_orig
from hickit.reader import get_chrom_sizes, get_headers
from hickit.matrix import CisMatrix
from model.custom_layers import ClipByValue
from mustache import regulator, mustache_norm
from util.constants import RESOLUTION
from util.plotting.plotting import plot_raw_crhom


def parsebed(chiafile, res=10000, lower=1, upper=5000000, valid_threshold=1):
    """
    Parses a ChIA-PET BEDPE file to extract significant genomic interactions.

    This function reads a file containing paired-end sequencing data, which represents
    physical interactions between different genomic loci. It filters these interactions
    based on their genomic distance and how many times they were observed[cite: 1, 4]. This creates
    the ground-truth dataset of "loops" used for labeling the Hi-C patches[cite: 1, 4].

    Note: The function hard-codes a "chr" prefix to chromosome names to prevent
    downstream bugs. It also filters out mitochondrial DNA ('M').

    Parameters:
    - chiafile (str): Path to the BEDPE-style file.
    - res (int): The resolution in base pairs to bin the coordinates.
    - lower (int): The minimum genomic distance for an interaction to be kept.
    - upper (int): The maximum genomic distance for an interaction to be kept.
    - valid_threshold (int): The minimum number of times an interaction must be
      observed to be considered valid.

    Returns:
    - dict: A dictionary where keys are chromosome names (e.g., 'chr1') and values
      are sets of valid interaction tuples `(locus_a, locus_b)`.
    """
    # Use a defaultdict to automatically create a list for each new chromosome encountered.
    coords = defaultdict(list)
    # Convert the upper distance threshold from base pairs to bin units.
    upper = upper // res
    # Open and read the specified BEDPE file line by line.
    with open(chiafile) as o:
        for line in o:
            # Strip whitespace and split the line into columns.
            s = line.rstrip().split()
            # Extract the start positions of the two interacting loci.
            a, b = float(s[1]), float(s[4])
            # Convert float positions to integers.
            a, b = int(a), int(b)
            # Ensure that `a` is always the smaller coordinate.
            if a > b:
                a, b = b, a
            # Convert the genomic coordinates to binned integer indices based on the resolution.
            a //= res
            b //= res
            # Filter the interaction: must be longer than `lower` and shorter than `upper`, and not on the mitochondrial chromosome.
            if (b - a > lower) and (b - a < upper) and 'M' not in s[0]:
                # Standardize the chromosome name by ensuring it has a "chr" prefix.
                chrom = 'chr' + s[0].lstrip('chr')
                # Add the binned coordinate tuple to the list for the corresponding chromosome.
                coords[chrom].append((a, b))

    # Create a new dictionary to store only the interactions that meet the validity threshold.
    valid_coords = dict()
    # Iterate through each chromosome's collected coordinates.
    for c in coords:
        # Get the list of all interactions for the current chromosome.
        current_set = set(coords[c])
        valid_set = set()
        # For each unique interaction, check if it occurs frequently enough.
        for coord in current_set:
            # If the interaction appears at least `valid_threshold` times, it's considered valid.
            if coords[c].count(coord) >= valid_threshold:
                valid_set.add(coord)
        # Store the set of valid interactions for the chromosome.
        valid_coords[c] = valid_set
    return valid_coords


def autofill_indicators(indicators, full_size):
    """
        Pads indicator DataFrames to a uniform size.

        When sampling a patch at the very end of a chromosome, the resulting data may be
        smaller than the standard `patch_size`. This function ensures that the corresponding
        metadata DataFrame (`indicator`) is padded to the expected `full_size`. It fills
        the extra rows with a locus value of -1 to signify that they are padding and do
        not correspond to real genomic coordinates.

        Parameters:
        - indicators (list[pd.DataFrame]): A list of indicator DataFrames to process.
        - full_size (int): The target size that all indicators should have.

        Returns:
        - list[pd.DataFrame]: The list of indicators, now all padded to `full_size`.
        """
    # Iterate through the list of indicator DataFrames with their indices.
    for i, indicator in enumerate(indicators):
        # Check if the current indicator is smaller than the required full size.
        if len(indicator) < full_size:
            # Calculate how much padding is needed.
            padded_size = full_size - len(indicator)
            # Create a dictionary for the padding data.
            padder = {
                # Use the same chromosome name for all padded rows.
                'chrom': [indicator['chrom'].unique()[0]] * padded_size,
                # Use -1 as a placeholder for the genomic locus to indicate padding.
                'locus': [-1] * padded_size
            }
            # Convert the padding dictionary to a DataFrame.
            padder = pd.DataFrame(padder)
            # Concatenate the original indicator with the new padding DataFrame.
            padded_indicator = pd.concat([indicator, padder]).reset_index(drop=True)
            # Replace the original indicator in the list with the new padded version.
            indicators[i] = padded_indicator
    return indicators


def block_sampling(matrix, starts_tuple, continuous_len, bedpe_list, filter=True):
    """
        Extracts a square sub-matrix (patch) and its corresponding labels.

        This is a central function called by the main patch generation loop. It takes a
        large chromosome matrix and a set of starting coordinates, then extracts the
        corresponding smaller block of data. It also calls a helper function to generate
        the ground-truth `label` matrix for this specific patch based on the `bedpe_list`.
        If the requested block is at the edge of the chromosome, it is padded to ensure
        consistent sizing.

        Parameters:
        - matrix (CisMatrix): The matrix object to sample from.
        - starts_tuple (tuple): A tuple of start indices for the patch.
        - continuous_len (int): The desired dimension of the patch (i.e., patch size).
        - bedpe_list (dict): The dictionary of ground-truth interactions from `parsebed`.
        - filter (bool): Whether to get headers from the filtered or original matrix.

        Returns:
        - tuple: A tuple containing:
            - subgraph (np.ndarray): The extracted numerical data patch.
            - label (np.ndarray): The boolean ground-truth label patch.
            - position_indicator (pd.DataFrame): DataFrame with genomic coordinates for the patch.
        """
    # Get the full size of the chromosome matrix dimension.
    full_size = matrix.mat.shape[0]
    # This list will hold all the row/column indices to be sampled from the main matrix.
    sampled_indices = []
    # This list will hold tuples defining the contiguous regions that were sampled.
    regions_tuples = []
    # Iterate through the start positions provided in the tuple (can be one for on-diagonal, two for off-diagonal).
    for start in starts_tuple:
        # If the sample fits completely within the matrix boundaries.
        if start + continuous_len <= full_size:
            # Add the full range of indices to the list.
            sampled_indices += list(range(start, start + continuous_len))
            # Record the region that was sampled.
            regions_tuples.append((start, start + continuous_len))
        else:
            # If the sample goes past the edge, take indices only up to the end of the matrix.
            sampled_indices += list(range(start, full_size))
            # Record the truncated region that was sampled.
            regions_tuples.append((start, full_size))
    # Slice the main matrix to extract the subgraph using the collected indices.
    subgraph = matrix.mat[sampled_indices, :][:, sampled_indices]
    # Check whether to use the filtered (cropped) or original headers for coordinates.
    if filter:
        position_indicator = matrix.get_cropped_headers().copy().reset_index(drop=True)
    else:
        position_indicator = matrix.headers.copy().reset_index(drop=True)
    # Select the rows from the headers that correspond to the sampled indices.
    position_indicator = position_indicator.iloc[sampled_indices].reset_index(drop=True)
    # Generate the boolean label matrix for this specific subgraph.
    label = get_label_for_continuous_subgraph(
        position_indicator, bedpe_list, continuous_len,
        pd.unique(matrix.headers['chrom'])[0]
    )
    # If the extracted subgraph is smaller than the desired size (due to hitting an edge).
    if subgraph.shape[0] < len(starts_tuple) * continuous_len: # padding
        # Pad both the subgraph and its label matrix with zeros to match the full size.
        subgraph, label = padding(subgraph, label, len(starts_tuple) * continuous_len)
    # Return the final data, ensuring correct data types.
    return subgraph.astype('float32'), label.astype('bool'), position_indicator

def create_ground_truth(matrix, matrix_len, bedpe_list, filter=True):
    if filter:
        position_indicator = matrix.get_cropped_headers().copy().reset_index(drop=True)
    else:
        position_indicator = matrix.headers.copy().reset_index(drop=True)

    ground_truth = get_label_for_continuous_subgraph(position_indicator,
                                                     bedpe_list,
                                                     matrix_len,
                                                     pd.unique(matrix.headers['chrom'])[0],
                                                     optimized=True)
    return ground_truth, position_indicator

def block_sampling_mustache(matrix, starts_tuple, patch_size, is_labels=False):
    # Get the full size of the chromosome matrix dimension.
    full_size = matrix.shape[0]
    # This list will hold all the row/column indices to be sampled from the main matrix.
    sampled_indices = []
    # Iterate through the start positions provided in the tuple (can be one for on-diagonal, two for off-diagonal).
    for start in starts_tuple:
        # If the sample fits completely within the matrix boundaries.
        if start + patch_size <= full_size:
            # Add the full range of indices to the list.
            sampled_indices += list(range(start, start + patch_size))
        else:
            # If the sample goes past the edge, take indices only up to the end of the matrix.
            sampled_indices += list(range(start, full_size))

    patch = np.full((patch_size, patch_size), False, dtype=np.bool_) if is_labels \
        else np.full((patch_size, patch_size), -1, dtype=np.float64)

    if len(starts_tuple) == 1:
        sampled_patch = matrix[sampled_indices, :][:, sampled_indices]
        actual_size = len(sampled_indices)
        patch[:actual_size, :actual_size] = sampled_patch
    else:
        half_length = len(sampled_indices) // 2
        sampled_patch = matrix[sampled_indices[:half_length], :][:, sampled_indices[half_length:]]
        patch[:sampled_patch.shape[0], :sampled_patch.shape[1]] = sampled_patch

    return patch

def get_label_for_continuous_subgraph(position_indicator, bedpe_list, continuous_len, chrom_name, optimized=False):
    """
    Generates a boolean label matrix for a given patch.

    This function determines the ground truth for a patch. It iterates through the
    list of known interactions (`bedpe_list`) for the given chromosome. If an
    interaction's coordinates fall within the genomic region covered by the patch,
    the corresponding pixel in the `label` matrix is set to `True`.

    A critical side-effect of this function is that it **destructively modifies** the
    `bedpe_list` by removing the interactions it has just labeled. This is likely an
    optimization to speed up subsequent searches for later patches.

    Parameters:
    - position_indicator (pd.DataFrame): DataFrame with genomic coordinates for the patch's bins.
    - bedpe_list (dict): The dictionary of ground-truth interactions.
    - continuous_len (int): The size of the patch.
    - chrom_name (str): The name of the chromosome being processed.

    Returns:
    - np.ndarray: A boolean matrix representing the ground truth for the patch.
    """
    # Reconstruct the chromosome name with the 'chr' prefix.
    current_chrom = 'chr' + chrom_name

    # Get the set of ground-truth interactions for the current chromosome.
    current_set = bedpe_list[current_chrom] if current_chrom in bedpe_list else {}

    # These lists will store the matrix coordinates of valid interactions.
    edge_list_row = []
    edge_list_col = []

    # Determine the genomic span of the patch. It can be unsymmetric for off-diagonal patches.
    if len(position_indicator) > continuous_len:
        # The first `continuous_len` rows of the indicator define the row span.
        row_locus_span = (position_indicator['locus'].iloc[0], position_indicator['locus'].iloc[continuous_len-1] + 1)
        # The remaining rows define the column span.
        col_locus_span = (position_indicator['locus'].iloc[continuous_len], position_indicator['locus'].iloc[-1] + 1)
    else: # This is a symmetric, on-diagonal patch.
        row_locus_span = (position_indicator['locus'].iloc[0], position_indicator['locus'].iloc[-1] + 1)
        col_locus_span = row_locus_span

    # Store the row and column spans as a single region definition.
    region = [row_locus_span, col_locus_span]

    # This list will track interactions that have been found within this patch.
    marked_truth = []

    # Iterate through all ground-truth interactions for this chromosome.
    for truth in current_set:

        # Check if the interaction falls within the genomic boundaries of the current patch.
        if is_entry_in_genomic_region(truth, region):

            # Also check if both ends of the interaction correspond to valid (non-filtered) bins.
            if is_entry_valid_in_cropped_map(truth, position_indicator):

                # If valid, find the matrix indices for the interaction's loci and store them.
                edge_list_row.append(list(position_indicator['locus']).index(truth[0]))
                edge_list_col.append(list(position_indicator['locus']).index(truth[1]))

            # Mark this interaction for removal, regardless of whether it was in a cropped bin.
            marked_truth.append(truth)

    # CRITICAL: Remove the found interactions from the global set to speed up future searches.
    # for marked in marked_truth:
    #     current_set.remove(marked)

    # Filter out any padded rows from the position indicator (where locus is -1).
    position_indicator = position_indicator[position_indicator['locus'] >= 0]

    # Get the actual size of the indicator after filtering.
    subgraph_size = len(position_indicator)

    # Initialize an empty boolean label matrix.
    if optimized:
        label = csc_matrix((subgraph_size, subgraph_size), dtype='int')
    else:
        label = np.zeros((subgraph_size, subgraph_size), dtype='bool')

    # Set the corresponding pixels in the label matrix to True.
    if optimized:
        label[edge_list_row, edge_list_col] = 1
    else:
        label[edge_list_row, edge_list_col] = True

    # Symmetrize the label matrix.
    if optimized:
        label = label + label.transpose() - diags(label.diagonal())
    else:
        label = np.triu(label) + np.tril(label.T, 1)

    # return label.astype('bool')
    return label


def is_entry_in_genomic_region(entry, genomic_region):
    """Checks if a genomic interaction `entry` is located within a `genomic_region`."""
    # Check if the first locus of the entry is within the region's first dimension span.
    if (genomic_region[0][0] <= entry[0] < genomic_region[0][1]) and \
            (genomic_region[1][0] <= entry[1] < genomic_region[1][1]): # Check the second locus against the second dimension.
        return True
    else:
        return False

def is_entry_valid_in_cropped_map(entry, position_indicator):
    """Checks if both loci of an interaction `entry` exist in the (potentially filtered) `position_indicator`."""
    # Check if both loci are present in the 'locus' column of the indicator DataFrame.
    if entry[0] in list(position_indicator['locus']) and entry[1] in list(position_indicator['locus']):
        return True
    else:
        return False

def is_ascent_order(the_list):
    """Checks if a list of items is in ascending order. Note: This function is not used by other functions in this script."""
    the_max = None
    for item in the_list:
        # Initialize the_max with the first item.
        if the_max is None:
            the_max = item
        # If the current item is greater than the max so far, update the max.
        elif item > the_max:
            the_max = item
        # If the current item is not greater, the list is not in ascending order.
        else:
            return False
    return True


def padding(subgraph, label, subgraph_size):
    """Pads a subgraph and its label matrix to a specified size with zeros."""
    # Calculate the number of rows/columns to add as padding.
    padding_len = subgraph_size - subgraph.shape[0]
    # Pad the subgraph using numpy.pad with a constant value of 0.
    # subgraph = np.pad(subgraph, [(0, padding_len), (0, padding_len)], mode='constant')
    subgraph = np.pad(subgraph, [(0, padding_len), (0, padding_len)], mode='constant', constant_values=(-1))
    # Pad the label matrix similarly.
    label = np.pad(label, [(0, padding_len), (0, padding_len)], mode='constant', constant_values=(False))
    return subgraph, label

def padding_mustache(subgraph, subgraph_size, is_labels):
    """Pads a subgraph and its label matrix to a specified size with zeros."""
    # Calculate the number of rows/columns to add as padding.
    padding_len = subgraph_size - subgraph.shape[0]
    # Pad the subgraph using numpy.pad with a constant value of 0.
    if not is_labels:
        subgraph = np.pad(subgraph, [(0, padding_len), (0, padding_len)], mode='constant', constant_values=(-1))
    else:
        subgraph = np.pad(subgraph, [(0, padding_len), (0, padding_len)], mode='constant', constant_values=(False))
    return subgraph

def padding_center(subgraph, subgraph_size):
    """Pads a subgraph and its label matrix to a specified size with zeros."""
    # Calculate the number of rows/columns to add as padding.
    padding_len = subgraph_size - subgraph.shape[0]
    # Pad the subgraph using numpy.pad with a constant value of 0.
    # subgraph = np.pad(subgraph, [(0, padding_len), (0, padding_len)], mode='constant')
    subgraph = np.pad(subgraph, [(0, padding_len), (0, padding_len)], mode='constant', constant_values=(-1))
    # Pad the label matrix similarly.
    return subgraph

def get_raw_graph(chrom_name, txt_dir, resolution, chrom_sizes_path, filter_by_nan=True):
    """
        Loads a full chromosome's Hi-C matrix from a text file.

        This function serves as the entry point for data loading. It orchestrates the reading
        of a simple tab-separated text file, creates a NumPy matrix, and wraps it in a
        `CisMatrix` object from the `hickit` library. This object associates the raw
        numerical data with genomic coordinate information (headers).

        Parameters:
        - chrom_name (str): The chromosome to load (e.g., '1').
        - txt_dir (str): The directory containing the matrix text files.
        - resolution (int): The resolution of the data in base pairs.
        - chrom_sizes_path (str): Path to the file containing chromosome sizes.
        - filter_by_nan (bool): If True, filters out rows/columns with a high percentage of NaN values.

        Returns:
        - CisMatrix: A matrix object containing the data and associated genomic headers.
        """
    # Create the interaction matrix from the text file data.
    X = create_interaction_matrix(chrom_name, txt_dir, chrom_sizes_path, resolution)

    # Convert the matrix to a numpy array.
    A = np.asarray(X)

    # Get the genomic headers (coordinate information) for this chromosome.
    headers = get_headers([chrom_name], get_chrom_sizes(chrom_sizes_path), resolution)

    # Create a CisMatrix object, which bundles the matrix data with its headers.
    matrix = CisMatrix(headers[headers['chrom'] == chrom_name], A, resolution)

    # Optionally filter out rows/columns that are mostly empty (NaNs).
    if filter_by_nan:
        matrix.filter_by_nan_percentage(0.9999)

    return matrix

def get_raw_graph_optimized(chrom_name, txt_dir, resolution, chrom_sizes_path, plot_chrom=False, threshold=0.75,
                            normalization='log2,clip'):
    # Create the interaction matrix
    interactions, chrom_upper_bound = create_sparse_csc_interaction_matrix(chrom_name, txt_dir, chrom_sizes_path, resolution,
                                                        plot_chrom, threshold, normalization)

    # Get the genomic headers (coordinate information) for this chromosome.
    headers = get_headers([chrom_name], get_chrom_sizes(chrom_sizes_path), resolution)

    # Create a CisMatrix object, which bundles the matrix data with its headers.
    matrix = CisMatrix(headers[headers['chrom'] == chrom_name], interactions, resolution)

    return matrix, chrom_upper_bound


def initialise_mat(chr_index, resolution, chrom_size_path):
    """Initializes an empty square numpy matrix for a given chromosome."""
    # Get the sizes of all chromosomes from the provided file.
    chrom_sizes = get_chrom_sizes(chrom_size_path)

    # Calculate the number of bins (loci) for the given chromosome and resolution.
    nloci = int((chrom_sizes[chr_index] / resolution) + 1)

    # Create a square matrix of zeros with the calculated dimensions.
    mat = np.zeros((nloci, nloci))
    return mat

def initialise_sparse_csc_mat(chr_index, resolution, chrom_size_path):
    """Initializes an empty square numpy matrix for a given chromosome."""
    # Get the sizes of all chromosomes from the provided file.
    chrom_sizes = get_chrom_sizes(chrom_size_path)

    # Calculate the number of bins (loci) for the given chromosome and resolution.
    nloci = int((chrom_sizes[chr_index] / resolution) + 1)

    # Create a square matrix of zeros with the calculated dimensions.
    mat = csc_matrix((nloci, nloci))
    return mat

def read_txt_data(txt_dir, i):
    """Reads a simple text contact file into a pandas DataFrame."""
    # Construct the full path to the contact file.
    path_to_txt = os.path.join(txt_dir, 'chr{}.contact.txt'.format(i))
    # Read the tab-separated file, which has no header, and return its values as a numpy array.
    txt_data = pd.read_csv(path_to_txt, sep='\t', header=None).values
    return txt_data

def read_contact_txt_data(txt_dir, i):
    """Reads a simple text contact file into a pandas DataFrame."""
    # Construct the full path to the contact file.
    path_to_txt = os.path.join(txt_dir, 'chr{}.contact.txt'.format(i))
    # Read the tab-separated file, which has no header, and return its values as a numpy array.
    txt_data = pd.read_csv(path_to_txt, sep='\t', header=None, names=["rows", "cols", "reads"])
    return txt_data


def create_interaction_matrix(chr_index, txt_dir, chrom_size_path, resolution=RESOLUTION):
    """
    Creates a dense interaction matrix from sparse text data.

    This reads a 3-column text file (locus1, locus2, value) and populates
    a dense 2D numpy matrix with the interaction values.
    """
    # Read the sparse data from the text file.
    txt_data = read_txt_data(txt_dir, chr_index)
    # Initialize an empty square matrix for the chromosome.
    mat = initialise_mat(chr_index, resolution, chrom_size_path)
    # Convert the genomic coordinates in the first two columns to integer row/column indices.
    rows = (txt_data[:, 0] / resolution).astype(int)
    cols = (txt_data[:, 1] / resolution).astype(int)
    # Get the interaction values from the third column.
    data = txt_data[:, 2]
    # Populate the matrix at the specified row/column indices with the interaction data.
    mat[rows, cols] = data

    mat = np.triu(mat) + np.tril(mat.T, 1)

    return mat

def create_sparse_csc_interaction_matrix(chr_index, txt_dir, chrom_size_path, resolution=RESOLUTION, plot_chrom=False,
                                         threshold=0.75, normalization='log2,clip'):
    # Read the sparse data from the text file
    txt_data = read_contact_txt_data(txt_dir, chr_index)
    # drop any NaNs from the normalized Hi-C file
    txt_data = txt_data.dropna()
    # collapse using sum
    # txt_data = txt_data.groupby(['rows', 'cols'], as_index=False)['reads'].sum()
    # convert the returned interaction dataframe to a numpy object
    # interaction matrix rows = 0, interaction matrix cols = 1, interaction matrix values = 2
    contact_data = txt_data.values

    # Convert the genomic coordinates in the first two columns to integer row/column indices.
    rows = (contact_data[:, 0] / resolution).astype(int)
    cols = (contact_data[:, 1] / resolution).astype(int)

    contacts = contact_data[:, 2]

    if plot_chrom:
        print(f'Chromosome {chr_index}:\n'
              f'max = {max(contact_data[:, 2])}\n'
              f'min = {min(contact_data[:, 2])}\n'
              f'mean = {np.mean(contact_data[:, 2])}\n')
        plot_raw_crhom(contacts)

    upper_bound = np.quantile(contacts, threshold)

    # init an empty sparse matrix and populate it with the normalized values
    sparse_mat = initialise_sparse_csc_mat(chr_index, resolution, chrom_size_path)
    used_mustache = False

    norms = normalization.split(',')
    for norm in norms:
        if norm == 'log2':
            contacts = np.log2(contacts + 1)
        elif norm == 'log':
            contacts = np.log(contacts + 1)
        elif norm == 'clip':
            contacts = ClipByValue(upper_bound)(contacts)
        elif norm == 'divide':
            contacts = contacts / upper_bound
        elif norm == 'zscore':
            contacts = (contacts - np.mean(contacts)) / np.std(contacts) #TODO: store mean/std dev in pkl, would including these as input to network improve perf?
        elif norm == 'mustache':
            used_mustache = True
            chrom_contacts = csc_matrix(sparse_mat, copy=True)
            chrom_contacts[rows, cols] = contacts
            contacts = mustache_norm(chrom_contacts, res=resolution, pt=0.01, distance_filter=2000000, num_processes=5)

            ### using orig
            argv = [
                "-f", "./data/hela_s3.hic",
                "-r", "10kb",
                "-pt", "0.01",
                "-p", "1",
                "-d", "2000000",
                "-ch", f"{chr_index}",
            ]
            # contacts = mustace_orig.main(argv)
        #     if argv is None:
        #         argv = []
        #     old_argv = sys.argv
        #     try:
        #         # Run and capture module globals to retrieve full_chrom
        #         sys.argv = ["mustace_orig"] + list(argv)
        #         mod_globals = runpy.run_module("mustace_orig", run_name="__main__")
        #         full_chrom = mod_globals.get("full_chrom")
        #     finally:
        #         sys.argv = old_argv
        #
        #     if full_chrom is None:
        #         raise RuntimeError("mustache run did not produce 'full_chrom' in module globals")
        # else:
        #     raise ValueError('Normalization method not recognized')

    if not used_mustache:
        sparse_mat[rows, cols] = contacts
        # Symmetrize the matrix by adding its transpose to itself.
        sparse_mat = sparse_mat + sparse_mat.transpose() - diags(sparse_mat.diagonal())
    else:
        sparse_mat = csc_matrix(contacts)

    return sparse_mat, upper_bound

def hic_to_intra_txt(juicer_path, file_path, out_dir, chrom, norm='KR', resolution=10000, hic_type='oe'):
    """
    Utility to convert .hic files to the text format used by this pipeline.

    This is a helper function for upstream data preparation. It uses the Juicer Tools
    command-line program to dump an intra-chromosomal contact matrix from a standard
    `.hic` file into a simple tab-separated `.txt` file[cite: 17, 18]. This is the format
    that the `create_interaction_matrix` function expects.

    Parameters:
    - juicer_path (str): Path to the 'juicer_tools.jar' executable.
    - file_path (str): Path to the input .hic file.
    - out_dir (str): Directory to save the output text file.
    - chrom (str): The chromosome to extract.
    - norm (str): The normalization to apply (e.g., 'KR').
    - resolution (int): The resolution to extract the data at.
    - hic_type (str): The type of data to dump (e.g., 'oe' for observed/expected).
    """
    """
    This function extract information from the .hic file and
    dump it to .txt files.
    """
    print('Start extracting matrix from the hic file...')
    print('Extracting intra-chromosomal interactions from chr {} ...'.format(chrom))
    # Define the output path for the text file.
    out_path = os.path.join(out_dir, 'chr{}.contact.txt'.format(chrom))
    # Construct the command-line arguments to call the Juicer Tools java program.
    cmd = [
        'java', '-jar', juicer_path, 'dump', hic_type,
        norm, file_path, chrom, chrom, 'BP', str(resolution), out_path
    ]
    # Run the command.
    run(cmd, shell=False)
