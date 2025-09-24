import logging
import os.path
import tensorflow as tf
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from tqdm import tqdm
import numpy as np
import pandas as pd

from itertools import product

from tqdm import tqdm

import gutils
from util.constants import RESOLUTION, PATCH_SIZE, SAMPLED_DATA_DIR, PATCH_SIZES, RESOLUTIONS, OUTPUT_DIR, PLOT_DIR
from gutils import get_raw_graph, block_sampling, parsebed
import random

from util.plotting.plotting import plot_actual_vs_sampled_patch, plot_coordinate_scatter


def get_segment_count(cropped_header_length, patch_size):
    """
        Calculates the number of segments (patches) that can fit into a given length.

        This is a utility function to determine how many patches of `patch_size`
        are needed to cover the entire length of a chromosome's contact map,
        represented by `cropped_header_length`. It effectively performs a ceiling
        division to ensure the entire length is covered.

        Parameters:
        - cropped_header_length (int): The total size of the dimension to be segmented.
        - patch_size (int): The size of each segment or patch.

        Returns:
        - int: The total number of segments required.
        """
    # Check if the length is not perfectly divisible by the patch size. 
    if cropped_header_length % patch_size != 0:
        # If not, perform ceiling division by adding 1 to the integer division result. 
        return int(cropped_header_length / patch_size) + 1
    else:
        # If it divides perfectly, return the exact division result. 
        return int(cropped_header_length / patch_size)


def get_patches_different_downsampling_rate(chrom_name,
                                            patch_size,
                                            graph_txt_dir,
                                            image_txt_dir,
                                            resolution,
                                            chrom_sizes_path,
                                            bedpe_list):
    """
        Generates training samples by extracting corresponding patches from high-resolution
        and low-resolution Hi-C maps for a given chromosome.

        This is the core function of the script. It loads a high-resolution ("image")
        and a low-resolution ("graph") contact matrix. It aligns them, then samples
        2D patches based on a list of start coordinates. For each sample, it extracts
        the image patch, the graph patch, and a boolean label matrix derived from a
        list of known genomic interactions (`bedpe_list`). It also generates a
        detailed index (`indicators`) mapping patch pixels to genomic loci.

        The function handles on-diagonal (intra-domain) and off-diagonal (inter-domain)
        sampling differently, constructing a 2x2 block structure for the graph patches.

        Parameters:
        - chrom_name (str): The name of the chromosome to process (e.g., 'chr1').
        - patch_size (int): The dimension of the square patches to extract (e.g., 64).
        - graph_txt_dir (str): Directory containing the low-resolution matrices.
        - image_txt_dir (str): Directory containing the high-resolution matrices.
        - resolution (int): The genomic resolution of the matrices in base pairs (e.g., 10000).
        - chrom_sizes_path (str): Path to the file containing chromosome sizes.
        - bedpe_list (list): A pre-parsed list of genomic interactions used for labeling.

        Returns:
        - tuple: A tuple containing:
            - image_set (np.ndarray): A stack of high-resolution patches.
            - graph_set (np.ndarray): A stack of low-resolution patches.
            - labels (np.ndarray): A stack of boolean label matrices.
            - indicators (pd.DataFrame): A DataFrame mapping patch indices to genomic coordinates.
        """
    # Load the high-resolution ("image") matrix, which serves as the input to the cnn
    image_matrix = get_raw_graph(chrom_name, image_txt_dir, resolution, chrom_sizes_path, filter_by_nan=False)
    # Load the low-resolution ("graph") matrix, which serves as the input to the gnn
    graph_matrix = get_raw_graph(chrom_name, graph_txt_dir, resolution, chrom_sizes_path, filter_by_nan=False)

    # Filter out noisy rows/columns (bins) from the low-resolution graph matrix. 
    graph_matrix.filter_by_nan_percentage(0.9999)
    # Get the genomic coordinates (headers) that remain after filtering. 
    unified_cropped_headers = graph_matrix.get_cropped_headers()
    # Get a boolean vector indicating which genomic loci were kept. 
    unified_loci_existence = graph_matrix.get_loci_existence_vector()
    # Apply the exact same filter to the high-resolution image matrix to ensure they are perfectly aligned. 
    image_matrix.mat = image_matrix.mat[unified_loci_existence, :][:, unified_loci_existence]
    # Update the image matrix's metadata to reflect this alignment. 
    image_matrix._filtered = True
    image_matrix._cropped_headers = unified_cropped_headers
    image_matrix._loci_existence = unified_loci_existence

    # Calculate how many patches are needed to span the filtered chromosome length.
    segment_count = get_segment_count(len(image_matrix.get_cropped_headers()), patch_size)
    # Generate all the (row, column) starting coordinates for patch extraction.
    start_tuples = get_start_tuples(segment_count, patch_size, resolution)
    # Pre-allocate numpy arrays to store the results for efficiency.
    image_set = np.zeros((len(start_tuples), patch_size, patch_size), dtype='float32')
    # The graph set patches are 2x the size to hold the concatenated context.
    graph_set = np.zeros((len(start_tuples), 2*patch_size, 2*patch_size), dtype='float32')
    # Pre-allocate the array for the boolean label matrices.
    labels = np.zeros((len(start_tuples), patch_size, patch_size), dtype='bool')
    # Initialize a list to store metadata (genomic coordinates) for each patch. 
    indicators = []

    # Loop over every generated start coordinate tuple to extract a patch.
    print(f'Chromosome {chrom_name}: Generating image set, graph set, labels, and indicators')
    num_plots = 0
    print(f'start_tuples: {start_tuples}')
    count = 0
    for i in tqdm(range(len(start_tuples))):
        tuple = start_tuples[i]

        # Check if the patch is on the main diagonal of the chromosome matrix.
        if tuple[0] == tuple[1]:
            # For on-diagonal patches, sample a single block from the graph matrix.
            # g, l, p = block_sampling(graph_matrix, (tuple[0],), patch_size, bedpe_list)
            # Create a larger 2x2 container for the graph patch.
            # graph = np.zeros((patch_size*2, patch_size*2))
            # Populate all four quadrants with the same on-diagonal block `g`.
            # graph[:patch_size, :patch_size] = g
            # graph[patch_size:, patch_size:] = g
            # graph[:patch_size, patch_size:] = g
            # graph[patch_size:, :patch_size] = g
            # Store the assembled 2x2 graph patch.
            # graph_set[i, :, :] = graph
            # Sample the corresponding block from the high-res image matrix.
            g, l, p = block_sampling(image_matrix, (tuple[0],), patch_size, bedpe_list)
            # Store the image patch.
            image_set[i, :, :] = g
            # Store the labels (only generated once from the first `block_sampling` call).
            labels[i, :, :] = l
            # Pad the metadata if the patch was at the edge of the chromosome.
            p = gutils.autofill_indicators([p], patch_size)[0]
            # Duplicate the metadata to match the 2x2 structure of the graph patch.
            p = pd.concat([p, p])
        else:
            # For off-diagonal patches, sample the full 2x2 block from the graph matrix directly.
            # graph_set[i, :, :], l, p = block_sampling(graph_matrix, tuple, patch_size, bedpe_list)
            g, l, p = block_sampling(image_matrix, tuple, patch_size, bedpe_list)
            # Extract the top-right quadrant from the sampled image block.
            image_set[i, :, :] = g[:patch_size, patch_size:]
            # Extract the corresponding quadrant from the label matrix.
            labels[i, :, :] = l[:patch_size, patch_size:]
            # Pad the metadata for the full 2x2 block if necessary.
            p = gutils.autofill_indicators([p], 2 * patch_size)[0]

            # expected_rows = list(range(tuple[0], tuple[0]+patch_size))
            # expected_cols = list(range(tuple[1], tuple[1] + patch_size))
            # sampled_patch = image_set[i, :, :]
            # real_patch = image_matrix.mat[expected_rows, :][:, expected_cols].astype(np.float32)
            # if num_plots < 10:
            #     plot_actual_vs_sampled_patch([real_patch, sampled_patch], 'Actual vs Patch',
            #               ['Actual Data', 'Sampled Data'], f'patch_{tuple[0]}_{tuple[1]}.png')
            #     num_plots += 1
            # assert real_patch.all() == sampled_patch.all()

            # if g.all() != image_set[i, :, :].all():
            #     print(f'\n\nUnique values in array: {np.unique(g[:patch_size, patch_size:])}\n\n')
            #     print(count)
            #     continue

        # Sanity check to ensure the metadata has the correct length. 
        assert len(p) == 2 * patch_size
        # Add the patch's final metadata to the list of all indicators. 
        indicators.append(p)
        count += 1
    # Concatenate all individual indicator DataFrames into a single large DataFrame. 
    indicators = pd.concat(indicators)
    # Post-process the graph set. This is a critical step for the modeling task. 
    for i, graph in enumerate(graph_set):
        # Zero out the top-left and bottom-right quadrants. 
        graph_set[i, :patch_size, :patch_size] = 0
        graph_set[i, patch_size:, patch_size:] = 0
    # Final sanity check to ensure the total number of indicator rows matches the data. 
    assert len(indicators) == len(graph_set) * 2 * patch_size
    # Return the complete set of data required for training. 
    return image_set, graph_set, labels, indicators

def get_boolean_graph_property(chrom_name, patch_size, txt_dir, resolution, chrom_sizes_path):
    """
        Identifies which potential patches are on the matrix diagonal.

        This function generates a boolean array indicating whether a patch, defined
        by a starting tuple from `get_start_tuples`, is on the main diagonal of the
        contact matrix (i.e., its start and end coordinates are the same).
        This is likely a feature generation step, creating a simple binary feature
        for each patch that a downstream model could use.

        Note: This function appears to be unused in the main `run_sample_patches`
        workflow, which calculates this property directly. It may be a remnant
        of a previous approach or used in a different context.

        Parameters:
        - chrom_name (str): The name of the chromosome to process.
        - patch_size (int): The dimension of the patches.
        - txt_dir (str): The directory containing the matrix data.
        - resolution (int): The genomic resolution of the matrix.
        - chrom_sizes_path (str): Path to the file containing chromosome sizes.

        Returns:
        - np.ndarray: A 1D numpy array of booleans, where `True` or `1`
          indicates an on-diagonal patch.
        """
    print('Chromosome {}'.format(chrom_name))
    # Load the raw matrix data. 
    matrix = get_raw_graph(chrom_name, txt_dir, resolution, chrom_sizes_path)
    # Calculate the number of segments for this chromosome. 
    segment_count = get_segment_count(len(matrix.get_cropped_headers()), patch_size)
    # Get the list of all start tuples that will be sampled. 
    start_tuples = get_start_tuples(segment_count, patch_size, resolution)
    # Create a numpy array to store the boolean property for each tuple. 
    graph_nodes_identical = np.zeros((len(start_tuples),))
    # Iterate through each tuple to check its property. 
    for i, tup in enumerate(start_tuples):
        # If the start row and column are the same, it's an on-diagonal patch. 
        if tup[0] == tup[1]:
            # Mark this patch as identical (on-diagonal) with a 1. 
            graph_nodes_identical[i] = 1
            # Return the resulting feature vector. 
    return graph_nodes_identical


def get_start_tuples(segment_count, patch_size, resolution):
    """
        Generates coordinate pairs for sampling patches from a matrix.

        This function creates a list of (row, column) start indices for all the
        patches to be extracted from a chromosome matrix. To make computation
        feasible, it doesn't generate all possible pairs. Instead, for each
        starting segment `i`, it only pairs it with segments `j` that are nearby
        (within a 2Mb window, as defined by `2000000/resolution`). This focuses
        sampling on locally relevant interactions, which is standard in Hi-C analysis.

        Parameters:
        - segment_count (int): The total number of segments along one dimension of the matrix.
        - patch_size (int): The size of each patch in bins.
        - resolution (int): The resolution of the data in base pairs per bin.

        Returns:
        - list: A list of tuples, where each tuple is a pair of start coordinates
          for patch sampling.
        """
    # Initialize an empty list to store the coordinate tuples. 
    start_tuples = []
    # Iterate through each possible starting segment `i` for the first dimension. 
    for i in range(segment_count):
        # Calculate the right-most edge for the second dimension `j`. 
        # This limits sampling to a 2-megabase genomic window for efficiency and relevance.
        mb = 2000000
        # mb = 10000000
        right_edge = min(segment_count, int(i+(mb/resolution/patch_size))+2) # tiled
        # Iterate from the diagonal (`i`) to the calculated right edge. 
        for j in range(i, right_edge):
            # Append the valid (row, col) start coordinate tuple, multiplied by patch size. 
            start_tuples.append((i*patch_size, j*patch_size))
    return start_tuples


def run_sample_patches(dataset_path,
                       assembly,
                       bedpe_path,
                       image_txt_dir,
                       graph_txt_dir,
                       chroms,
                       patch_size,
                       resolution,
                       sampled_data_dir):
    """
        Main execution function to generate and save a dataset of patches for multiple chromosomes.

        This function orchestrates the entire data generation pipeline. It sets up file paths,
        parses a BEDPE file containing ground truth interactions, and then iterates through
        a list of chromosomes. For each chromosome, it calls
        `get_patches_different_downsampling_rate` to generate the image, graph, label,
        and indicator data. Finally, it saves these artifacts to disk in a structured
        dataset directory, creating separate `.npy` and `.csv` files for each chromosome.

        Parameters:
        - dataset_name (str): The name for the output dataset directory.
        - assembly (str): The genome assembly version (e.g., 'hg19'), used to find the chrom sizes file.
        - bedpe_path (str): Path to the BEDPE file with interaction data.
        - image_txt_dir (str): Path to the high-resolution input data.
        - graph_txt_dir (str): Path to the low-resolution input data.
        - chroms (list): A list of chromosome names to process (e.g., ['chr1', 'chr2']).
        """
    # Construct the path to the chromosome sizes file based on the genome assembly. 
    chrom_size_path = '{}.chrom.sizes'.format(assembly)
    # Parse the BEDPE file to get a dictionary of ground-truth interactions (loops). 
    bedpe_list = parsebed(bedpe_path, valid_threshold=1)
    # Define the path for the output dataset directory.
    # dataset_path = os.path.join(sampled_data_dir, dataset_name)
    # Create the output directory; fail if it already exists to prevent accidental overwrites.
    os.makedirs(dataset_path, exist_ok=False)
    # Iterate over each chromosome specified in the `chroms` list.
    for cn in chroms:
        # Call the main patch generation function to get all data for the current chromosome.
        image_set, graph_set, labels, indicators = get_patches_different_downsampling_rate(chrom_name=cn,
                                                                                           patch_size=patch_size,
                                                                                           graph_txt_dir=graph_txt_dir,
                                                                                           image_txt_dir=image_txt_dir,
                                                                                           resolution=resolution,
                                                                                           chrom_sizes_path=chrom_size_path,
                                                                                           bedpe_list=bedpe_list)
        # Save the high-resolution image patches as a numpy array.
        np.save(os.path.join(dataset_path, 'imageset.{}.npy'.format(cn)), image_set.astype('float32'))
        # Save the boolean label patches as a numpy array of integers.
        np.save(os.path.join(dataset_path, 'labels.{}.npy'.format(cn)), labels.astype('int'))

        # Save the genomic coordinate metadata to a CSV file. 
        indicators.to_csv(os.path.join(dataset_path, 'indicators.{}.csv'.format(cn)))
        # Save the low-resolution graph patches as a numpy array. 
        np.save(os.path.join(dataset_path, 'graphset.{}.npy'.format(cn)), graph_set.astype('float32'))

        # Initialize a boolean array to flag on-diagonal patches. 
        graph_nodes_identical = np.zeros((len(graph_set),), dtype='bool')
        # Iterate through each generated patch by its index. 
        for idx in range(len(graph_set)):
            # Check if the patch is on-diagonal. This is done by comparing the genomic locus
            # of the first bin in the first half of the patch with the first bin in the second half.
            # If they are identical, the patch was an on-diagonal sample that was duplicated. 
            if indicators.iloc[idx * (2 * 64)]['locus'] == indicators.iloc[idx * (2 *PATCH_SIZE) + PATCH_SIZE]['locus']:
                # If they are the same, set the flag for this patch to True. [cite: 10, 11]
                graph_nodes_identical[idx] = True
        # Save the boolean flags as an integer numpy array. 
        np.save(os.path.join(dataset_path, 'graph_identical.{}.npy'.format(cn)), graph_nodes_identical.astype('int'))