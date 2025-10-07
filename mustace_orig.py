#!/usr/bin/env python3
# Import necessary standard libraries
import argparse
import os
import sys
import re
import math
import warnings
import time
import struct
from collections import defaultdict

# Import necessary standard libraries
import pandas as pd
import numpy as np

# Import libraries for handling Hi-C data
import hicstraw
import cooler

# Import scientific and statistical libraries from SciPy
from scipy.stats import expon
from scipy.ndimage import gaussian_filter
from scipy.ndimage.filters import maximum_filter
from scipy.signal import convolve2d
import scipy.ndimage.measurements as scipy_measurements
from scipy import sparse

# Import library for multiple testing correction
from statsmodels.stats.multitest import multipletests

# Import multiprocessing library for parallel execution
from multiprocessing import Process, Manager


def parseBP(s):
    """
    Parses a string representing a number of base pairs, which can include 'kb' or 'mb' suffixes.

    :param s: The input string (e.g., '5000', '10kb', '2mb').
    :type s: str
    :return: The number of base pairs as an integer, or False if the string is invalid.
    :rtype: int or bool
    """
    # Return False if the string is empty or None
    if not s:
        return False
    # If the string is purely numeric, convert to integer and return
    if s.isnumeric():
        return int(s)
    # Convert string to lowercase for case-insensitive matching
    s = s.lower()
    # Check for 'kb' (kilobase) suffix
    if "kb" in s:
        # Split the string at 'kb' to get the numeric part
        n = s.split("kb")[0]
        # If the numeric part is not a valid number, return False
        if not n.isnumeric():
            return False
        # Return the numeric part multiplied by 1000
        return int(n) * 1000
    # Check for 'mb' (megabase) suffix
    elif "mb" in s:
        # Split the string at 'mb' to get the numeric part
        n = s.split("mb")[0]
        # If the numeric part is not a valid number, return False
        if not n.isnumeric():
            return False
        # Return the numeric part multiplied by 1,000,000
        return int(n) * 1000000
    # If no valid format is found, return False
    return False


def parse_args(args):
    """
    Sets up and parses command-line arguments using argparse.

    :param args: Command-line arguments from sys.argv[1:].
    :type args: list
    :return: An object containing the parsed command-line arguments.
    :rtype: argparse.Namespace
    """
    # Initialize the argument parser with a description
    parser = argparse.ArgumentParser(description="Check the help flag")

    # Define command-line arguments
    parser.add_argument("-f",
                        "--file",
                        dest="f_path",
                        help="REQUIRED: Contact map",
                        required=False)
    parser.add_argument("-d",
                        "--distance",
                        dest="distFilter",
                        help="REQUIRED: Maximum distance (in bp) allowed between loop loci",
                        required=False)
    parser.add_argument("-o",
                        "--outfile",
                        dest="outdir",
                        help="Name of the output file.\
                       Output is a numpy binary.",
                        required=False)
    parser.add_argument("-r",
                        "--resolution",
                        dest="resolution",
                        help="REQUIRED: Resolution used for the contact maps",
                        required=True)
    parser.add_argument("-bed", "--bed", dest="bed",
                        help="BED file for HiC-Pro type input",
                        default="",
                        required=False)
    parser.add_argument("-m", "--matrix", dest="mat",
                        help="MATRIX file for HiC-Pro type input",
                        default="",
                        required=False)
    parser.add_argument("-b", "--biases", dest="biasfile",
                        help="RECOMMENDED: biases calculated by\
                        ICE or KR norm for each locus for contact map are read from BIASFILE",
                        required=False)
    parser.add_argument(
        "-cz",
        "--chromosomeSize",
        default="",
        dest="chrSize_file",
        help="RECOMMENDED: .hic corressponfing chromosome size file.",
        required=False)
    parser.add_argument(
        "-norm",
        "--normalization",
        default=False,
        dest="norm_method",
        help="RECOMMENDED: Hi-C  normalization method (KR, VC,...).",
        required=False)
    parser.add_argument(
        "-st",
        "--sparsityThreshold",
        dest="st",
        type=float,
        default=0.88,
        help="OPTIONAL: Mustache filters out contacts in sparse areas, you can relax this for sparse datasets(i.e. -st 0.8). Default value is 0.88.",
        required=False)
    parser.add_argument(
        "-pt",
        "--pThreshold",
        dest="pt",
        type=float,
        default=0.2,
        help="OPTIONAL: P-value threshold for the results in the final output. Default is 0.2",
        required=False)
    parser.add_argument(
        "-sz",
        "--sigmaZero",
        dest="s_z",
        type=float,
        default=1.6,
        help="OPTIONAL: sigma0 value for the method. DEFAULT is 1.6. \
        Experimentally chosen for 5Kb resolution",
        required=False)
    parser.add_argument("-oc", "--octaves", dest="octaves", default=2,
                        type=int,
                        help="OPTIONAL: Octave count for the method. \
                        DEFAULT is 2.",
                        required=False)
    parser.add_argument("-i", "--iterations", dest="s", default=10,
                        type=int,
                        help="OPTIONAL: iteration count for the method. \
                        DEFAULT is 10. Experimentally chosen for \
                        5Kb resolution",
                        required=False)
    parser.add_argument("-p", "--processes", dest="nprocesses", default=4, type=int,
                        help="OPTIONAL: Number of parallel processes to run. DEFAULT is 4. Increasing this will also increase the memory usage",
                        required=False)
    parser.add_argument(
        "-ch",
        "--chromosome",
        dest="chromosome",
        nargs='+',
        help="REQUIRED: Specify which chromosome to run the program for. Optional for cooler files.",
        default='n',
        required=False)
    parser.add_argument(
        "-ch2",
        "--chromosome2",
        dest="chromosome2",
        nargs='+',
        help="Optional: Specify the second chromosome for interchromosomal analysis.",
        default='n',
        required=False)
    parser.add_argument("-v",
                        "--verbose",
                        dest="verbose",
                        type=bool,
                        default=True,
                        help="OPTIONAL: Verbosity of the program",
                        required=False)
    # Parse the provided arguments and return the result
    return parser.parse_args(args)


def kth_diag_indices(a, k):
    """
    Returns the indices of the k-th diagonal of a square matrix 'a'.

    :param a: The input square numpy array.
    :type a: numpy.ndarray
    :param k: The diagonal to find indices for (k=0 is the main diagonal, k>0 is above, k<0 is below).
    :type k: int
    :return: A tuple of two arrays (rows, cols) representing the indices of the diagonal.
    :rtype: tuple
    """
    # Get the indices of the main diagonal
    rows, cols = np.diag_indices_from(a)
    # For diagonals below the main diagonal (k < 0)
    if k < 0:
        # Return row indices starting from -k and column indices up to k
        return rows[-k:], cols[:k]
    # For diagonals above the main diagonal (k > 0)
    elif k > 0:
        # Return row indices up to -k and column indices starting from k
        return rows[:-k], cols[k:]
    # For the main diagonal (k = 0)
    else:
        # Return the full row and column indices
        return rows, cols


def is_chr(s, c):
    # if 'X' == c or 'chrX':
    #    return 'X' in c
    # if 'Y' == c:
    #    return 'Y' in c
    """
    Checks if two chromosome name strings refer to the same chromosome, ignoring the 'chr' prefix.

    :param s: First chromosome name string.
    :type s: str or int
    :param c: Second chromosome name string.
    :type c: str or int
    :return: True if the chromosome names are equivalent, False otherwise.
    :rtype: bool
    """
    # Compare the string representations after removing any 'chr' prefix
    return str(c).replace('chr', '') == str(s).replace('chr', '')  # re.findall("[1-9][0-9]*", str(s))


def get_sep(f):
    """
    Guesses the column separator (delimiter) of a text file by reading the first few lines.

    :param f: The path to the file.
    :type f: str
    :return: The detected separator ('\t', ' ', or ',').
    :rtype: str
    :raises FileNotFoundError: If a separator cannot be determined.
    """
    # Open the file for reading
    with open(f) as file:
        # Iterate over each line in the file
        for line in file:
            # Check for tab separator
            if "\t" in line:
                return '\t'
            # Check for space separator in a stripped line
            if " " in line.strip():
                return ' '
            # Check for comma separator
            if "," in line:
                return ','
            # If line has only one column, assume space as separator
            if len(line.split(' ')) == 1:
                return ' '
            # Stop after checking the first few lines
            break
    # If no separator is found, raise an error
    raise FileNotFoundError


def read_bias(f, chromosome, res):
    """
    Reads a bias vector file (e.g., from KR or ICE normalization) and returns it as a dictionary.

    :param f: Path to the bias file.
    :type f: str
    :param chromosome: The chromosome to read biases for.
    :type chromosome: str
    :param res: The resolution in base pairs.
    :type res: int
    :return: A dictionary mapping bin index to bias value, or False if no file is provided.
    :rtype: collections.defaultdict or bool
    """
    # Create a defaultdict that returns 1.0 for missing keys
    d = defaultdict(lambda: 1.0)
    # Proceed only if a file path is provided
    if f:
        # Guess the separator for the bias file
        sep = get_sep(f)
        # Open the bias file
        with open(f) as file:
            # Enumerate through each line of the file
            for pos, line in enumerate(file):
                # Strip whitespace and split the line by the separator
                line = line.strip().split(sep)
                # Handle 3-column format (chr, start, bias)
                if len(line) == 3:
                    # Check if the line corresponds to the correct chromosome
                    if is_chr(line[0], chromosome):
                        # Convert the bias value to a float
                        val = float(line[2])
                        # (float(line[1]) // res) == Get the bin index by dividing the genomic position by the resolution
                        # Check if the bias value is not NaN
                        if not np.isnan(val):
                            # If bias is too low, treat it as infinite (unmappable region)
                            if val < 0.2:
                                d[(float(line[1]) // res)] = np.Inf
                            else:
                                d[(float(line[1]) // res)] = val
                        else:
                            # If bias is NaN, treat it as infinite
                            d[(float(line[1]) // res)] = np.Inf
                # Handle 1-column format (bias values only)
                elif len(line) == 1:
                    # Convert the bias value to a float
                    val = float(line[0])
                    # pos == Use the line number as the bin index
                    # Check if the bias value is not NaN
                    if not np.isnan(val):
                        # If bias is too low, treat it as infinite
                        if val < 0.2:
                            d[pos] = np.Inf
                        else:
                            d[pos] = val
                    else:
                        # If bias is NaN, treat it as infinite
                        d[pos] = np.Inf
        # Return the dictionary of biases
        return d
    # If no file path was given, return False
    return False


def read_pd(f, distance_in_bp, bias, chromosome, res):
    """
    Reads a contact map from a text file (e.g., HiC-Pro format) into sparse matrix coordinates.

    :param f: Path to the contact map file.
    :type f: str
    :param distance_in_bp: Maximum interaction distance to consider.
    :type distance_in_bp: int
    :param bias: Path to the bias file.
    :type bias: str
    :param chromosome: The chromosome to read data for.
    :type chromosome: str
    :param res: The resolution in base pairs.
    :type res: int
    :return: A tuple of three numpy arrays (x, y, val) for sparse matrix representation.
    :rtype: tuple
    """
    # Guess the file's column separator
    sep = get_sep(f)
    # Read the file into a pandas DataFrame
    df = pd.read_csv(f, sep=sep, header=None)
    # Drop rows with any missing values
    df.dropna(inplace=True)
    # Handle 5-column BEDPE-like format (chr1, start1, end1, chr2, start2, end2, value)
    if df.shape[1] == 5:
        # Filter for rows matching the specified chromosome for the first locus
        df = df[np.vectorize(is_chr)(df[0], chromosome)]
        # If no interactions remain, print a message and return
        if df.shape[0] == 0:
            print('Could\'t read any interaction for this chromosome!')
            return
        # Filter for rows matching the specified chromosome for the second locus
        df = df[np.vectorize(is_chr)(df[2], chromosome)]
        # Filter by interaction distance
        df = df.loc[np.abs(df[1] - df[3]) <= ((distance_in_bp / res + 1) * res), :]
        # Convert genomic coordinates to bin indices
        df[1] //= res
        df[3] //= res
        # Read and apply bias correction if a bias file is provided
        bias = read_bias(bias, chromosome, res)
        if bias:
            # Get bias factors for the first set of bins
            factors = np.vectorize(bias.get)(df[1], 1)
            # Apply bias correction
            df[4] = np.divide(df[4], factors)
            # Get bias factors for the second set of bins
            factors = np.vectorize(bias.get)(df[3], 1)
            # Apply bias correction
            df[4] = np.divide(df[4], factors)
        # Filter out zero or negative contact values
        df = df.loc[df[4] > 0, :]
        # Define x and y coordinates for the sparse matrix (ensure x <= y)
        x = np.min(df.loc[:, [1, 3]], axis=1)
        y = np.max(df.loc[:, [1, 3]], axis=1)
        val = np.array(df[4])
    # Handle 3-column format (bin1, bin2, value)
    elif df.shape[1] == 3:
        # Filter by interaction distance in bins
        df = df.loc[np.abs(df[1] - df[0]) <= ((distance_in_bp / res + 1) * res), :]
        # Convert genomic coordinates (if not already binned) to bin indices
        df[0] //= res
        df[1] //= res
        # Read and apply bias correction if a bias file is provided
        bias = read_bias(bias, chromosome, res)
        if bias:
            # Get bias factors for the first set of bins
            factors = np.vectorize(bias.get)(df[0], 1)
            # Apply bias correction
            df[2] = np.divide(df[2], factors)
            # Get bias factors for the second set of bins
            factors = np.vectorize(bias.get)(df[1], 1)
            # Apply bias correction
            df[2] = np.divide(df[2], factors)
        # Filter out zero or negative contact values
        df = df.loc[df[2] > 0, :]
        # Define x and y coordinates for the sparse matrix (ensure x <= y)
        x = np.min(df.loc[:, [0, 1]], axis=1)
        y = np.max(df.loc[:, [0, 1]], axis=1)
        val = np.array(df[2])

    # Return the sparse matrix coordinates
    return x, y, val


def read_hic_file(f, norm_method, CHRM_SIZE, distance_in_bp, chr1, chr2, res):
    """
    Reads contact data from a .hic file for a specific region, handling chunking for large chromosomes.

    :param f: Path to the .hic file.
    :type f: str
    :param norm_method: Normalization to apply (e.g., 'KR', 'VC').
    :type norm_method: str
    :param CHRM_SIZE: The size of the chromosome in base pairs.
    :type CHRM_SIZE: int
    :param distance_in_bp: The maximum interaction distance to consider.
    :type distance_in_bp: int
    :param chr1: The name of the first chromosome.
    :type chr1: str
    :param chr2: The name of the second chromosome.
    :type chr2: str
    :param res: The resolution in base pairs.
    :type res: int
    :return: A tuple of three numpy arrays (x, y, val) for sparse matrix representation.
    :rtype: tuple
    """
    # If chromosome size is not provided, get it from the .hic file
    if not CHRM_SIZE:
        hic = hicstraw.HiCFile(f)
        chromosomes = hic.getChromosomes()
        chrSize_in_bp = {}
        # Populate a dictionary with chromosome names and lengths
        for i in range(1, len(chromosomes)):
            chrSize_in_bp["chr" + str(chromosomes[i].name).replace("chr", '')] = chromosomes[i].length
        # Standardize the chromosome key
        chr_key = "chr" + chr1.replace("chr", '')
        if chr_key in chrSize_in_bp:
            CHRM_SIZE = chrSize_in_bp[chr_key]
        else:
            raise NameError('wrong chromosome name!')

    # Define chunk size for reading data, ensuring it's large enough
    CHUNK_SIZE = max(2 * distance_in_bp / res, 2000)
    start = 0
    end = min(CHRM_SIZE, CHUNK_SIZE * res)  # CHUNK_SIZE*res
    result = []
    val = []

    # Loop to read the chromosome in chunks
    while start < CHRM_SIZE:
        print(int(start), int(end))
        # Use hicstraw to fetch data for the current chunk
        if not norm_method:
            # Default to KR normalization if not specified
            temp = hicstraw.straw("observed", "KR", f, str(chr1) + ":" + str(int(start)) + ":" + str(int(end)),
                                  str(chr2) + ":" + str(int(start)) + ":" + str(int(end)), "BP", res)
        else:
            # Use the specified normalization method
            temp = hicstraw.straw("observed", str(norm_method), f,
                                  str(chr1) + ":" + str(int(start)) + ":" + str(int(end)),
                                  str(chr2) + ":" + str(int(start)) + ":" + str(int(end)), "BP", res)
        # If no data is returned for the chunk, advance to the next chunk
        if len(temp) == 0:
            # Advance start position, with overlap to not miss boundary interactions
            start = min(start + CHUNK_SIZE * res - distance_in_bp, CHRM_SIZE)
            if end == CHRM_SIZE - 1:
                break
            else:
                end = min(end + CHUNK_SIZE * res - distance_in_bp, CHRM_SIZE - 1)
            continue

        # Process the fetched data
        if result == []: # For the first chunk
            # Initialize the result list with bin coordinates and counts
            cur_block = [[int(record.binX), int(record.binY), record.counts] for record in temp]
            result.append([x[0] for x in cur_block])
            result.append([x[1] for x in cur_block])
            result.append([x[2] for x in cur_block])
            # Store the current block to avoid duplicates in overlapping chunks
            prev_block = set([(record.binX, record.binY, record.counts) for record in temp])
        else: # For subsequent chunks
            # Create a set of interactions for the current block
            cur_block = set([(int(record.binX), int(record.binY), record.counts) for record in temp])
            # Find new interactions by taking the set difference with the previous block
            to_add_list = list(cur_block - prev_block)
            del prev_block
            # Append the new interactions to the result list
            result[0] += [x[0] for x in to_add_list]
            result[1] += [x[1] for x in to_add_list]
            result[2] += [x[2] for x in to_add_list]
            # Update prev_block for the next iteration
            prev_block = cur_block
            del cur_block

        # Advance to the next chunk with overlap
        start = min(start + CHUNK_SIZE * res - distance_in_bp, CHRM_SIZE)
        if end == CHRM_SIZE - 1:
            break
        else:
            end = min(end + CHUNK_SIZE * res - distance_in_bp, CHRM_SIZE - 1)

    # If no contacts were found at all, return empty lists
    if len(result) == 0:
        print(f'There is no contact in chrmosome {chr1} to work on.')
        return [], [], []

    # Convert lists to numpy arrays and bin coordinates to pixel indices
    x = np.array(result[0]) // res
    y = np.array(result[1]) // res
    val = np.array(result[2])

    # Remove any NaN values that might have occurred
    nan_indx = np.logical_or.reduce((np.isnan(result[0]), np.isnan(result[1]), np.isnan(result[2])))
    x = x[~nan_indx]
    y = y[~nan_indx]
    val = val[~nan_indx]
    x = x.astype(int)
    y = y.astype(int)

    # Check again if any valid data remains
    if len(val) == 0:
        print(f'There is no contact in chrmosome {chr1} to work on.')
        return [], [], []
    else:
        # Ensure remaining NaN values in counts are set to 0
        val[np.isnan(val)] = 0

    # For intrachromosomal maps, filter by distance and remove zero counts
    if (chr1 == chr2):
        dist_f = np.logical_and(np.abs(x - y) <= distance_in_bp / res, val > 0)
        x = x[dist_f]
        y = y[dist_f]
        val = val[dist_f]

    # Return final sparse data if any contacts remain
    if len(val > 0):
        return x, y, val
    else:
        print(f'There is no contact in chrmosome {chr1} to work on.')
        return [], [], []


def read_cooler(f, distance_in_bp, chr1, chr2, cooler_balance):
    """
    Reads contact data from a .cool file, handling chunking for large chromosomes.

    :param f: Path to the .cool file.
    :type f: str
    :param distance_in_bp: The maximum interaction distance to consider.
    :type distance_in_bp: int
    :param chr1: The name of the first chromosome.
    :type chr1: str
    :param chr2: The name of the second chromosome.
    :type chr2: str
    :param cooler_balance: Whether to apply cooler's native balancing (weight column).
    :type cooler_balance: bool or str
    :return: A tuple of (x, y, val, res) for sparse data and resolution.
    :rtype: tuple
    """
    # Open the cooler file
    clr = cooler.Cooler(f)
    # Get resolution from the cooler file
    res = clr.binsize
    print(f'Your cooler data resolution is {res}')
    # Validate chromosome names
    if chr1 not in clr.chromnames or chr2 not in clr.chromnames:
        raise NameError('wrong chromosome name!')
    # Get chromosome size and define chunking parameters
    CHRM_SIZE = clr.chromsizes[chr1]
    CHUNK_SIZE = max(2 * distance_in_bp / res, 2000)
    start = 0
    end = min(CHUNK_SIZE * res, CHRM_SIZE)  # CHUNK_SIZE*res
    result = []
    val = []
    ###########################
    # Process intrachromosomal contacts
    if chr1 == chr2:
        # try:
        # normVec = clr.bins()['weight'].fetch(chr1)
        # result = clr.matrix(balance=True,sparse=True).fetch(chr1)#as_pixels=True, join=True
        # Loop to read the chromosome in chunks
        while start < CHRM_SIZE:
            print(int(start), int(end))
            # Fetch the matrix for the current chunk, applying balancing if requested
            if not cooler_balance:
                temp = clr.matrix(balance=True, sparse=True).fetch((chr1, int(start), int(end)))
            else:
                temp = clr.matrix(balance=cooler_balance, sparse=True).fetch((chr1, int(start), int(end)))
            # Work with the upper triangle of the matrix to avoid duplicate contacts
            temp = sparse.triu(temp)
            # Replace any NaN/inf values with 0
            np.nan_to_num(temp, copy=False, nan=0, posinf=0, neginf=0)
            # Convert start coordinate from bp to pixel index
            start_in_px = int(start / res)
            # If the chunk is empty, advance to the next one
            if len(temp.row) == 0:
                start = min(start + CHUNK_SIZE * res - distance_in_bp, CHRM_SIZE)
                if end == CHRM_SIZE - 1:
                    break
                else:
                    end = min(end + CHUNK_SIZE * res - distance_in_bp, CHRM_SIZE - 1)
                continue

            # Adjust coordinates to be relative to the whole chromosome
            row_coords = start_in_px + temp.row
            col_coords = start_in_px + temp.col
            # Aggregate contacts, avoiding duplicates from overlapping chunks
            if result == []: # First chunk
                result += [list(row_coords), list(col_coords), list(temp.data)]
                prev_block = set(
                    [(x, y, v) for x, y, v in zip(row_coords, col_coords, temp.data)])
            else: # Subsequent chunks
                cur_block = set(
                    [(x, y, v) for x, y, v in zip(row_coords, col_coords, temp.data)])
                # Find new interactions by taking the set difference
                to_add_list = list(cur_block - prev_block)
                del prev_block
                # Append new interactions
                result[0] += [x[0] for x in to_add_list]
                result[1] += [x[1] for x in to_add_list]
                result[2] += [x[2] for x in to_add_list]
                prev_block = cur_block
                del cur_block

            # Advance to the next chunk with overlap
            start = min(start + CHUNK_SIZE * res - distance_in_bp, CHRM_SIZE)
            if end == CHRM_SIZE - 1:
                break
            else:
                end = min(end + CHUNK_SIZE * res - distance_in_bp, CHRM_SIZE - 1)

        # If no contacts were found, return empty lists
        if len(result) == 0:
            print(f'There is no contact in chrmosome {chr1} to work on.')
            return [], [], [], res

        # Convert result lists to numpy arrays
        x = np.array(result[0])
        y = np.array(result[1])
        val = np.array(result[2])
    # Process interchromosomal contacts (no chunking needed)
    else:
        # Fetch the entire interchromosomal matrix
        result = clr.matrix(balance=True, sparse=True).fetch(chr1, chr2)
        result = sparse.triu(result)
        # Ensure all NaN/inf values are 0
        np.nan_to_num(result, copy=False, nan=0, posinf=0, neginf=0)
        x = result.row
        y = result.col
        val = result.data

    ##########################
    # If no data, return empty
    if len(val) == 0:
        print(f'There is no contact in chrmosome {chr1} to work on.')
        return [], [], [], res
    else:
        # Ensure remaining NaNs are zero
        val[np.isnan(val)] = 0

    # For intrachromosomal maps, filter by distance and remove zero counts
    if (chr1 == chr2):
        dist_f = np.logical_and(np.abs(x - y) <= distance_in_bp / res, val > 0)
        x = x[dist_f]
        y = y[dist_f]
        val = val[dist_f]
    # Return final sparse data and resolution
    if len(val > 0):
        return np.array(x), np.array(y), np.array(val), res
    else:
        print(f'There is no contact in chrmosome {chr1} to work on.')
        return [], [], [], res


def read_mcooler(f, distance_in_bp, chr1, chr2, res, cooler_balance):
    """
    Reads contact data from a .mcool file for a specific resolution, handling chunking.

    :param f: Path to the .mcool file.
    :type f: str
    :param distance_in_bp: The maximum interaction distance to consider.
    :type distance_in_bp: int
    :param chr1: The name of the first chromosome.
    :type chr1: str
    :param chr2: The name of the second chromosome.
    :type chr2: str
    :param res: The resolution to extract from the .mcool file.
    :type res: int
    :param cooler_balance: Whether to apply cooler's native balancing.
    :type cooler_balance: bool or str
    :return: A tuple of three numpy arrays (x, y, val) for sparse matrix representation.
    :rtype: tuple
    """
    # Construct the URI to access a specific resolution within the mcool file
    uri = '%s::/resolutions/%s' % (f, res)
    # Open the cooler object from the URI
    clr = cooler.Cooler(uri)

    # Validate chromosome names
    if chr1 not in clr.chromnames or chr2 not in clr.chromnames:
        raise NameError('wrong chromosome name!')

    # Get chromosome size and define chunking parameters
    CHRM_SIZE = clr.chromsizes[chr1]
    CHUNK_SIZE = max(2 * distance_in_bp / res, 2000)
    start = 0
    end = min(CHRM_SIZE, CHUNK_SIZE * res)  # CHUNK_SIZE*res
    result = []
    val = []

    # Process intrachromosomal contacts
    if chr1 == chr2:
        try:
            # Loop to read the chromosome in chunks
            while start < CHRM_SIZE:
                print(int(start), int(end))
                # Fetch the matrix for the current chunk with balancing
                if not cooler_balance:
                    temp = clr.matrix(balance=True, sparse=True).fetch((chr1, int(start), int(end)))
                else:
                    temp = clr.matrix(balance=cooler_balance, sparse=True).fetch((chr1, int(start), int(end)))
                # Work with the upper triangle and clean data
                temp = sparse.triu(temp)
                np.nan_to_num(temp, copy=False, nan=0, posinf=0, neginf=0)
                start_in_px = int(start / res)

                # If chunk is empty, move to the next
                if len(temp.row) == 0:
                    start = min(start + CHUNK_SIZE * res - distance_in_bp, CHRM_SIZE)
                    if end == CHRM_SIZE - 1:
                        break
                    else:
                        end = min(end + CHUNK_SIZE * res - distance_in_bp, CHRM_SIZE - 1)

                    continue

                # Adjust coordinates and aggregate contacts, avoiding duplicates
                row_coords = start_in_px + temp.row
                col_coords = start_in_px + temp.col

                if result == []: # First chunk
                    result += [list(row_coords), list(col_coords), list(temp.data)]
                    prev_block = set(
                        [(x, y, v) for x, y, v in zip(row_coords, col_coords, temp.data)])
                else: # Subsequent chunks
                    cur_block = set(
                        [(x, y, v) for x, y, v in zip(row_coords, col_coords, temp.data)])
                    to_add_list = list(cur_block - prev_block)
                    del prev_block
                    result[0] += [x[0] for x in to_add_list]
                    result[1] += [x[1] for x in to_add_list]
                    result[2] += [x[2] for x in to_add_list]
                    prev_block = cur_block
                    del cur_block

                # Advance to the next chunk with overlap
                start = min(start + CHUNK_SIZE * res - distance_in_bp, CHRM_SIZE)
                if end == CHRM_SIZE - 1:
                    break
                else:
                    end = min(end + CHUNK_SIZE * res - distance_in_bp, CHRM_SIZE - 1)
        except:
            # Catch any errors during file reading
            raise NameError('Reading from the file failed!')

        # If no contacts found, return empty lists
        if len(result) == 0:
            print(f'There is no contact in chrmosome {chr1} to work on.')
            return [], [], []

        # Convert to numpy arrays
        x = np.array(result[0])
        y = np.array(result[1])
        val = np.array(result[2])
    # Process interchromosomal contacts
    else:
        # Fetch the entire interchromosomal matrix
        result = clr.matrix(balance=True, sparse=True).fetch(chr1, chr2)
        result = sparse.triu(result)
        np.nan_to_num(result, copy=False, nan=0, posinf=0, neginf=0)
        x = result.row
        y = result.col
        val = result.data

    # Final data cleaning and filtering
    if len(val) == 0:
        print(f'There is no contact in chrmosome {chr1} to work on.')
        return [], [], []
    else:
        val[np.isnan(val)] = 0

    # For intrachromosomal, filter by distance
    if (chr1 == chr2):
        dist_f = np.logical_and(np.abs(x - y) <= distance_in_bp / res, val > 0)
        x = x[dist_f]
        y = y[dist_f]
        val = val[dist_f]

    # Return final sparse data
    if len(val > 0):
        return np.array(x), np.array(y), np.array(val)
    else:
        print(f'There is no contact in chrmosome {chr1} to work on.')
        return [], [], []


def get_diags(map):
    """
    Calculates the mean and standard deviation for each diagonal of a contact map.

    :param map_matrix: The input contact map as a numpy matrix.
    :type map_matrix: numpy.ndarray
    :return: A tuple of two dictionaries (means, stds) where keys are diagonal indices.
    :rtype: tuple
    """
    # Initialize dictionaries to store means and standard deviations
    means = {}
    stds = {}
    # Iterate through each diagonal of the matrix
    for i in range(len(map)):
        # Extract the i-th diagonal
        diag = map.diagonal(i)
        # Filter out zero values, which represent non-contacts
        diag = diag[diag != 0]
        # If the diagonal has non-zero elements
        if len(diag) > 0:
            # Calculate mean and standard deviation
            mean = np.mean(diag)
            # Store the mean, handling potential NaN values
            std = np.std(diag) if np.std(diag) != 0 else 1
            if math.isnan(mean):
                means[i] = 0
            else:
                means[i] = mean
            if math.isnan(std):
                stds[i] = 1
            else:
                stds[i] = std
        else:
            # If diagonal is empty, set mean to 0 and std to 1
            means[i] = 0
            stds[i] = 1
    # Return the dictionaries of means and standard deviations
    return means, stds


def normalize_sparse(x, y, v, resolution, distance_in_px):
    """
    Performs distance-based z-score normalization on sparse Hi-C data.
    For each diagonal, it calculates a local mean and std to normalize pixel values.

    :param x: Array of row indices (bin 1).
    :type x: numpy.ndarray
    :param y: Array of column indices (bin 2).
    :type y: numpy.ndarray
    :param v: Array of contact values.
    :type v: numpy.ndarray
    :param resolution: The resolution in base pairs.
    :type resolution: int
    :param distance_in_px: The maximum interaction distance in pixels (bins).
    :type distance_in_px: int
    :return: A list of weights used for p-value correction (currently not fully implemented).
    :rtype: list
    """
    # Determine the size of the matrix from the maximum coordinate
    n = max(max(x), max(y)) + 1
    pval_weights = []
    # Calculate the genomic distance for each contact
    distances = np.abs(y - x)

    # Use a more sophisticated local normalization for large chromosomes
    if (n - distance_in_px) * resolution > 2000000:
        with warnings.catch_warnings():
            # Ignore runtime warnings that can occur with division by zero
            warnings.simplefilter('ignore', category=RuntimeWarning)
            # Define a filter size (e.g., 2Mb window)
            filter_size = int(2000000 / resolution)
            # Iterate through each diagonal up to the max distance
            for d in range(2 + distance_in_px):
                # Get indices for contacts on the current diagonal
                indices = distances == d
                # Create a dense array for the diagonal's values
                vals = np.zeros(n - d)
                vals[x[indices]] = v[indices] + 0.001 # Add small epsilon to avoid log(0)
                if vals.size == 0:
                    continue
                # Calculate global mean and std for the diagonal as a fallback
                std = np.std(v[indices])
                mean = np.mean(v[indices])
                if math.isnan(mean):
                    mean = 0
                if math.isnan(std):
                    std = 1

                # Use a 1D convolution to calculate local stats
                kernel = np.ones(filter_size)
                # Count non-zero entries in the window
                counts = np.convolve(vals != 0, kernel, mode='same')

                # Calculate local sum and sum of squares
                s = np.convolve(vals, kernel, mode='same')
                s2 = np.convolve(vals ** 2, kernel, mode='same')
                # Calculate local variance
                local_var = (s2 - s ** 2 / counts) / (counts - 1)

                # Fallback to global std for unstable local variance calculations
                std2 = std ** 2
                np.nan_to_num(local_var, copy=False,
                              neginf=std2, posinf=std2, nan=std2)

                # Calculate local mean
                local_mean = s / counts
                # Fallback to global mean where local counts are too low
                local_mean[counts < 30] = mean
                local_var[counts < 30] = std2
                np.nan_to_num(local_mean, copy=False,
                              neginf=mean, posinf=mean, nan=mean)

                # Get local std deviation
                local_std = np.sqrt(local_var)
                # Normalize the values for the current diagonal
                vals[x[indices]] -= local_mean[x[indices]]
                vals[x[indices]] /= local_std[x[indices]]
                np.nan_to_num(vals, copy=False, nan=0, posinf=0, neginf=0)
                # Apply a weight based on the diagonal's mean intensity
                vals = vals * (1 + math.log(1 + mean, 30))
                pval_weights += [1 + math.log(1 + mean, 30)]
                # Update the original value array with the normalized values
                v[indices] = vals[x[indices]]
    else:
        # Use simpler global normalization for smaller matrices
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', category=RuntimeWarning)
            np.nan_to_num(v, copy=False, neginf=0, posinf=0, nan=0)
            distance_in_px = min(distance_in_px, n)
            # Iterate through each diagonal
            for d in range(distance_in_px):
                indices = distances == d
                # Calculate global mean and std for the diagonal
                std = np.std(v[indices])
                mean = np.mean(v[indices])
                if math.isnan(mean):
                    mean = 0
                if math.isnan(std):
                    std = 1
                # Normalize values on the diagonal (z-score)
                v[indices] = (v[indices] - mean) / std
                np.nan_to_num(v, copy=False, nan=0, posinf=0, neginf=0)
    return pval_weights


def inter_normalize_map(vals):
    """
    Performs z-score normalization on an entire inter-chromosomal map.

    :param vals: The contact values of the map.
    :type vals: numpy.ndarray
    """
    # Calculate global mean and standard deviation
    m = np.mean(vals)
    s = np.std(vals)
    # Apply z-score normalization
    cmap -= m
    cmap /= s
    # Clean up any resulting NaN or infinity values
    np.nan_to_num(cmap, copy=False, nan=0, posinf=0, neginf=0)

def mustache(c, chromosome, chromosome2, res, pval_weights, start, end, mask_size, distance_in_px, octave_values, st,
             pt, block_num, num_blocks):
    """
        The core Mustache algorithm for detecting loop-like features in a Hi-C map block.
        This function implements a scale-space search for blob-like features using a Difference of Gaussians (DoG) pyramid.

        :param c: The input contact map block (a dense numpy matrix).
        :type c: numpy.ndarray
        :param chromosome: Name of the first chromosome.
        :type chromosome: str
        :param chromosome2: Name of the second chromosome.
        :type chromosome2: str
        :param res: Resolution in base pairs.
        :type res: int
        :param pval_weights: Weights for p-value adjustment (not fully used).
        :type pval_weights: list
        :param start: The starting pixel coordinate of this block in the full chromosome map.
        :type start: int
        :param end: The ending pixel coordinate of this block.
        :type end: int
        :param mask_size: Size of the overlapping region to mask to avoid double-counting.
        :type mask_size: int
        :param distance_in_px: Maximum interaction distance in pixels.
        :type distance_in_px: int
        :param octave_values: A list of starting sigma values for each octave in the scale-space search.
        :type octave_values: list
        :param st: Sparsity threshold for filtering results.
        :type st: float
        :param pt: P-value threshold for final loop calls.
        :type pt: float
        :return: A list of significant loops, each with format [x, y, p-value, scale].
        :rtype: list
        """
    # Identify non-zero pixels in the upper triangle of the map (potential loop candidates)
    nz = np.logical_and(c != 0, np.triu(c, 4))
    # A stricter definition of non-zero for filtering later
    nz_temp = np.logical_and.reduce((c != 0, np.triu(c, 4) > 0, np.tril(c, distance_in_px) > 0))
    # If the matrix is too sparse, exit early
    if np.sum(nz) < 50: # TODO experiment with crhm1, if LcMax takes too long, return 0s otherwise return LcMax
        print(f'Matrix is too sparse: {np.sum(nz)}')
        return np.zeros(c.shape), np.zeros(c.shape)
        # return []

    # Set the lower triangle and far-diagonal elements to a marker value (2) to ignore them
    c[np.tril_indices_from(c, 4)] = 2
    # if chromosome == chromosome2:
    #     c[np.triu_indices_from(c, k=(distance_in_px + 1))] = 2

    # Initialize arrays to store results for each candidate pixel
    pAll = np.ones_like(c[nz]) * 2 # P-values, initialized to a placeholder value
    Scales = np.ones_like(pAll) # Detection scale (sigma)
    vAll = np.zeros_like(pAll) # DoG response value
    s = 10 # Number of scales to check per octave
    # curr_filter = 1
    scales = {} # Dictionary to store sigma values
    # Lcmax = np.array(c.shape)
    Lcmax = None

    # --- Scale-Space Extrema Detection ---
    # Iterate over each octave (a set of scales)
    for o in octave_values:
        scales[o] = {}
        # --- First scale in the octave ---
        sigma = o
        w = 2 * math.ceil(2 * sigma) + 1 # Kernel width
        t = (((w - 1) / 2) - 0.5) / sigma # Truncation for Gaussian filter
        Gp = gaussian_filter(c, o, truncate=t, order=0) # Previous Gaussian-smoothed image
        scales[o][1] = sigma

        # --- Second scale ---
        sigma = o * 2 ** ((2 - 1) / s)
        w = 2 * math.ceil(2 * sigma) + 1
        t = (((w - 1) / 2) - 0.5) / sigma
        Gc = gaussian_filter(c, sigma, truncate=t, order=0) # Current Gaussian-smoothed image
        scales[o][2] = sigma

        # First Difference of Gaussians (DoG) image
        Lp = Gp - Gc
        Gp = [] # Free memory

        sigma = o * 2 ** ((3 - 1) / s)
        w = 2 * math.ceil(2 * sigma) + 1
        t = (((w - 1) / 2) - 0.5) / sigma
        Gn = gaussian_filter(c, sigma, truncate=t, order=0) # Next Gaussian-smoothed image
        scales[o][3] = sigma

        # Second DoG image
        # Lp = Gp - Gc
        Lc = Gc - Gn
        if Lcmax is None:
            Lcmax = Lc.copy()
        else:
            Lcmax = np.maximum(Lcmax, Lc)

        # Find local maxima in the DoG images
        locMaxP = maximum_filter(
            Lp, footprint=np.ones((3, 3)), mode='constant')
        locMaxC = maximum_filter(
            Lc, footprint=np.ones((3, 3)), mode='constant')

        # Iterate through the remaining scales in the octave
        for i in range(3, s + 2):
            # curr_filter += 1
            Gc = Gn # Shift images: current becomes previous

            # Calculate next scale's sigma and apply Gaussian filter
            sigma = o * 2 ** ((i) / s)
            w = 2 * math.ceil(2 * sigma) + 1
            t = ((w - 1) / 2 - 0.5) / sigma
            Gn = gaussian_filter(c, sigma, truncate=t, order=0)
            scales[o][i + 1] = sigma

            # Create the next DoG image
            Ln = Gc - Gn

            # Fit an exponential distribution to the absolute DoG values to model the background
            dist_params = expon.fit(np.abs(Lc[nz]))
            # Calculate p-values for the current DoG image based on the fitted distribution
            pval = 1 - expon.cdf(np.abs(Lc[nz]), *dist_params)

            # Find local maxima in the next DoG image
            locMaxN = maximum_filter(
                Ln, footprint=np.ones((3, 3)), mode='constant')

            # --- Identify Scale-Space Extrema ---
            # An extremum is a pixel that is a local maximum in both space (3x3 neighborhood)
            # and scale (compared to adjacent DoG images Lp and Ln).
            willUpdate = np.logical_and \
                .reduce((Lc[nz] > vAll, # Is the DoG response stronger than any found so far?
                         Lc[nz] == locMaxC[nz], # Is it a local maximum in the current 2D slice?
                         np.logical_or(Lp[nz] == locMaxP[nz],  Ln[nz] == locMaxN[nz]), # Is it also an extremum in an adjacent scale?
                         Lc[nz] > locMaxP[nz], # Is it also an extremum in an adjacent scale?
                         Lc[nz] > locMaxN[nz])) # Is it greater than the response in the next scale?

            # If a pixel is a new extremum, store its value, scale, and p-value
            vAll[willUpdate] = Lc[nz][willUpdate]
            Scales[willUpdate] = scales[o][i]
            pAll[willUpdate] = pval[willUpdate]

            # Move to the next scale
            Lp = Lc
            Lc = Ln
            locMaxP = locMaxC
            locMaxC = locMaxN

    # --- Filtering and Clustering ---
    # Find all pixels that were identified as candidate loops
    pFound = pAll != 2
    if len(pFound) < 10000: # Exit if too few candidates
        # return []
        print(f'Too few candidates found for {chromosome}:{pFound}.')
        return np.zeros(c.shape), np.zeros(c.shape)

    # Apply Benjamini-Hochberg FDR correction to the p-values
    _, pCorrect, _, _ = multipletests(pAll[pFound], method='fdr_bh')
    pAll[pFound] = pCorrect

    #################
    # o = np.ones_like(c)
    # o[nz] = pAll
    # x, y = np.where(nz_temp)
    # o[x,y]*=np.array(pval_weights)[y-x]
    # o[x,y]/=10
    # pAll = o[nz]
    #################
    # Reconstruct the full p-value matrix
    o = np.ones_like(c)
    o[nz] = pAll
    # Count significant pixels below the p-value threshold
    sig_count = np.sum(o < pt)  # change
    # Get the coordinates of the significant pixels, sorted by p-value
    x, y = np.unravel_index(np.argsort(o.ravel()), o.shape)

    # Reconstruct the full scale matrix
    so = np.ones_like(c)
    so[nz] = Scales

    # Keep only the top 'sig_count' pixels
    x = x[:sig_count]
    y = y[:sig_count]
    xyScales = so[x, y]

    # --- Sparsity Filter ---
    # Filter out loops in sparse regions of the map
    nonsparse = x != 0 # Initialize a boolean mask
    for i in range(len(xyScales)):
        # Check the density of valid contacts in a small window around the candidate
        s = math.ceil(xyScales[i])
        c1 = np.sum(nz[x[i] - s:x[i] + s + 1, y[i] - s:y[i] + s + 1]) / \
             ((2 * s + 1) ** 2)
        # Check density in a larger window
        s = 2 * s
        c2 = np.sum(nz[x[i] - s:x[i] + s + 1, y[i] - s:y[i] + s + 1]) / \
             ((2 * s + 1) ** 2)
        # If either region is too sparse, mark the pixel for removal
        if c1 < st or c2 < 0.6:
            nonsparse[i] = False
    x = x[nonsparse]
    y = y[nonsparse]

    if len(x) == 0:
        return []

    # --- Enrichment Filter (for intrachromosomal only) ---
    def nz_mean(vals):
        return np.mean(vals[vals != 0])

    def diag_mean(k, map):
        return nz_mean(map[kth_diag_indices(map, k)])

    # if chromosome == chromosome2:
    #     # Calculate the mean value of the diagonal corresponding to each candidate loop
    #     means = np.vectorize(diag_mean, excluded=['map'])(k=y - x, map=c)
    #     # Keep only pixels that are significantly enriched over their diagonal's background
    #     passing_indices = c[x, y] > 2 * means  # change
    #     if len(passing_indices) == 0 or np.sum(passing_indices) == 0:
    #         return []
    #     x = x[passing_indices]
    #     y = y[passing_indices]

    # --- Clustering ---
    # Cluster adjacent significant pixels to report a single representative loop per cluster
    # label_matrix = np.zeros((np.max(y) + 2, np.max(y) + 2), dtype=np.float32) # 780 loops found for chrmosome=1, fdr<0.01 in 84.45sec
    label_matrix = np.zeros_like(c, dtype=np.float32) # 780 loops found for chrmosome=1, fdr<0.01 in 84.73sec
    # Mark significant pixels with their p-value and their 8-neighbors with a marker
    label_matrix[x, y] = o[x, y] + 1
    # print('before neighbors: ', np.unique(label_matrix).size)
    label_matrix[x + 1, y] = 2
    label_matrix[x + 1, y + 1] = 2
    label_matrix[x, y + 1] = 2
    label_matrix[x - 1, y] = 2
    label_matrix[x - 1, y - 1] = 2
    label_matrix[x, y - 1] = 2
    label_matrix[x + 1, y - 1] = 2
    label_matrix[x - 1, y + 1] = 2
    # print('after neighbors: ', np.unique(label_matrix).size)

    # Use scipy's labeling function to find connected components (clusters)
    num_features = scipy_measurements.label(
        label_matrix, output=label_matrix, structure=np.ones((3, 3)))
    # print('after clustering: ', num_features)

    # ground_truth = np.zeros_like(c)

    out = []
    # For each found cluster
    for label in range(1, num_features + 1):
        # Find all pixels belonging to this cluster
        indices = np.argwhere(label_matrix == label)
        # Find the pixel within the cluster that has the lowest p-value
        i = np.argmin(o[indices[:, 0], indices[:, 1]])
        _x, _y = indices[i, 0], indices[i, 1]
        # if _x >= start + mask_size or _y >= start + mask_size:
        #     ground_truth[_x, _y] = 1
        # Append this representative pixel to the output list, adjusting coordinates back to full chromosome scale
        out.append([_x + start, _y + start, o[_x, _y], so[_x, _y]])

    # print('ground_truth: ', np.sum(ground_truth))
    # print('out: ' , len(out))

    # np.save(f'./mustache/output/mustache.{chromosome}.image.{block_num}.npy', Lc)
    # ground_truth = np.zeros_like(c)
    # ground_truth[_x, _y] = 1
    return out, Lcmax


def regulator(f, norm_method, CHRM_SIZE, outdir, bed="",
              res=5000,
              sigma0=1.6,
              s=10,
              pt=0.1,
              st=0.88,
              octaves=2,
              verbose=True,
              nprocesses=4,
              distance_filter=2000000,
              bias=False,
              chromosome='n',
              chromosome2=None):
    """
        Main controller function that orchestrates the loop calling process.
        It reads the data, normalizes it, splits it into chunks, and runs the Mustache algorithm in parallel.

        :param f: Path to the contact map file.
        :type f: str
        :param norm_method: Normalization method to use.
        :type norm_method: str
        :param CHRM_SIZE: Size of the chromosome in base pairs.
        :type CHRM_SIZE: int
        :param outdir: Path to the output directory/file.
        :type outdir: str
        :param bed: Path to the BED file for HiC-Pro format.
        :type bed: str
        :param res: Resolution in base pairs.
        :type res: int
        :param sigma0: Starting sigma for scale-space search.
        :type sigma0: float
        :param s: Number of scales per octave.
        :type s: int
        :param pt: P-value threshold.
        :type pt: float
        :param st: Sparsity threshold.
        :type st: float
        :param octaves: Number of octaves for scale-space search.
        :type octaves: int
        :param verbose: Verbosity flag.
        :type verbose: bool
        :param nprocesses: Number of parallel processes to use.
        :type nprocesses: int
        :param distance_filter: Maximum interaction distance in base pairs.
        :type distance_filter: int
        :param bias: Path to the bias file.
        :type bias: str or bool
        :param chromosome: Name of the first chromosome.
        :type chromosome: str
        :param chromosome2: Name of the second chromosome (for inter-chromosomal analysis).
        :type chromosome2: str or None
        :return: A list of all significant loops found.
        :rtype: list
        """
    # Default to intrachromosomal analysis if chromosome2 is not specified
    if not chromosome2 or chromosome2 == 'n':
        chromosome2 = chromosome

    # Check for valid input format for interchromosomal analysis
    if (chromosome != chromosome2) and not ((('.hic' in f) or ('.cool' in f) or ('.mcool' in f))):
        print(
            "Interchromosomal analysis is only supported for .hic and .cool input formats.")
        raise FileNotFoundError

    # Define the sigma values for the scale-space search based on octaves
    octave_values = [sigma0 * (2 ** i) for i in range(octaves)]
    distance_in_bp = distance_filter

    print("Reading contact map...")
    # Read the contact map based on file type
    if f.endswith(".hic"):
        x, y, v = read_hic_file(f, norm_method, CHRM_SIZE, distance_in_bp, chromosome, chromosome2, res)
    elif f.endswith(".cool"):
        x, y, v, res = read_cooler(f, distance_in_bp, chromosome, chromosome2, norm_method)
    elif f.endswith(".mcool"):
        x, y, v = read_mcooler(f, distance_in_bp, chromosome, chromosome2, res, norm_method)
    else:
        x, y, v = read_pd(f, distance_in_bp, bias, chromosome, res)

    # If no data was read, return an empty list
    if len(v) == 0:
        return []
    print("Normalizing contact map...")
    distance_in_px = int(math.ceil(distance_in_bp // res))

    # --- Intrachromosomal analysis ---
    if chromosome == chromosome2:
        # Get the full matrix size
        n = max(max(x), max(y)) + 1
        pval_weights = normalize_sparse(x, y, v, res, distance_in_px)

        # Normalize the sparse data
        CHUNK_SIZE = max(2 * distance_in_px, 2000)

        # --- Chunking the matrix for parallel processing ---
        overlap_size = distance_in_px

        # Determine start and end coordinates for each chunk
        if n <= CHUNK_SIZE:
            start = [0]
            end = [n]
        else:
            start = [0]
            end = [CHUNK_SIZE]

            # Create overlapping chunks to avoid missing loops at boundaries
            while end[-1] < n:
                start.append(end[-1] - overlap_size)
                end.append(start[-1] + CHUNK_SIZE)
            end[-1] = n
            start[-1] = end[-1] - CHUNK_SIZE

        print("Loop calling...")
        # Use a Manager to share a list between processes
        with Manager() as manager:
            o = manager.list() # Shared list to store results
            arrs = manager.list()
            gts = manager.list()
            i = 0
            processes = []
            # Iterate over each chunk
            for i in range(len(start)):
                # Create the dense matrix for the current block
                # create the currnet block
                indx = np.logical_and.reduce((x >= start[i], x < end[i], y >= start[i], y < end[i]))
                xc = x[indx] - start[i]
                yc = y[indx] - start[i]
                vc = v[indx]
                cc = np.zeros((CHUNK_SIZE, CHUNK_SIZE))
                cc[xc, yc] = vc

                # Create and start a new process for this chunk
                p = Process(target=process_block, args=(
                    i, start, end, overlap_size, cc, chromosome, chromosome2, res, pval_weights, distance_in_px,
                    octave_values, o, st, pt, arrs, gts))
                p.start()
                processes.append(p)

                # Wait for processes to finish if the pool is full or it's the last chunk
                if len(processes) >= nprocesses or i == (len(start) - 1):
                    for p in processes:
                        p.join()
                    processes = []
            # o_corrected = [[e[0],e[1],e[2]/pval_weights[e[1]-e[0]],e[3]] for e in list(o)]

            # Return the aggregated results from all processes
            return list(o), np.asarray(list(arrs), dtype=np.float32), np.asarray(list(gts), dtype=np.int64)

    # --- Interchromosomal analysis ---
    else:
        # Normalize the entire map at once (no chunking)
        n1 = max(x) + 1
        n2 = max(y) + 1
        inter_normalize_map(x, y, v, res)
        # Note: The actual processing for interchromosomal is not fully implemented here
        # It would require creating a dense matrix and calling mustache, but this path is not completed.
        # For now, it just normalizes and would then end.


def process_block(i, start, end, overlap_size, cc, chromosome, chromosome2, res, pval_weights, distance_in_px,
                  octave_values, o, st, pt, arrs, gts):
    """
        A wrapper function executed by each parallel process. It runs the Mustache algorithm on a single block.

        :param i: The index of the block being processed.
        :type i: int
        :param start: List of start coordinates for all blocks.
        :type start: list
        :param end: List of end coordinates for all blocks.
        :type end: list
        :param overlap_size: The size of the overlap between blocks.
        :type overlap_size: int
        :param cc: The dense contact map for the current block.
        :type cc: numpy.ndarray
        :param chromosome: Name of the first chromosome.
        :type chromosome: str
        :param chromosome2: Name of the second chromosome.
        :type chromosome2: str
        :param res: Resolution in base pairs.
        :type res: int
        :param pval_weights: P-value weights.
        :type pval_weights: list
        :param distance_in_px: Max distance in pixels.
        :type distance_in_px: int
        :param octave_values: Sigma values for the scale-space search.
        :type octave_values: list
        :param o: The shared list to which results are appended.
        :type o: multiprocessing.Manager.list
        :param st: Sparsity threshold.
        :type st: float
        :param pt: P-value threshold.
        :type pt: float
        """
    num_blocks = len(start)
    print("Starting block ", i + 1, "/", num_blocks , "...", sep='')
    # Determine the mask size to avoid double-counting loops in overlapping regions
    if i == 0:
        mask_size = -1 # No mask for the first block
    elif i == len(start) - 1:
        mask_size = end[i - 1] - start[i] # Mask the overlap with the previous block
    else:
        mask_size = overlap_size

    # Run the core loop detection algorithm on the block
    loops, Lcmax = mustache(
        cc, chromosome, chromosome2, res, pval_weights, start[i], end[i], mask_size, distance_in_px, octave_values, st,
        pt, i + 1, num_blocks)

    # Append valid loops to the shared output list, ensuring they are not from the masked region
    ground_truth = np.zeros_like(cc)
    for loop in list(loops):
        if loop[0] >= start[i] + mask_size or loop[1] >= start[i] + mask_size:
            o.append([loop[0], loop[1], loop[2], loop[3]])
            ground_truth[[loop[0]-start[i]], loop[1]-start[i]] = 1
    arrs.append(Lcmax)
    gts.append(ground_truth)

    # print('filtered o: ', len(o))
    # print('ground_truth: ', np.sum(ground_truth))
    # np.save(f'./mustache/output/mustache.{chromosome}.labels.{i + 1}.npy', ground_truth)
    print("Block", i + 1, "done.")


def main(arg_list):
    """
    The main entry point of the script. It parses arguments, sets up parameters,
    iterates through chromosomes, calls the main regulator function, and writes the output.
    """
    # Record the start time for performance measurement
    start_time = time.time()
    # Parse command-line arguments
    # args = parse_args(sys.argv[1:])
    args = parse_args(arg_list)
    print("\n")

    # Determine the input file path
    f = args.f_path
    if args.bed and args.mat:
        f = args.mat

    # Check if the input file exists
    if not os.path.exists(f):
        print("Error: Couldn't find the specified contact files")
        return

    # Parse resolution string into an integer
    res = parseBP(args.resolution)
    if not res:
        print("Error: Invalid resolution")
        return
    CHR_LIST_FLAG = False
    CHR_COOL_FLAG = False
    CHR_HIC_FLAG = False

    # --- Chromosome Handling ---
    # Determine which chromosomes to process based on input flags
    if not args.chromosome or args.chromosome == 'n':
        # For cooler files, we can automatically get the chromosome list
        if f.endswith(".cool") or f.endswith(".mcool"):
            CHR_COOL_FLAG = True
        elif f.endswith(".hic"):
            # For .hic files, get chromosome list from the file header
            CHR_HIC_FLAG = True
        elif len(args.chromosome > 1):
            print("Error: For this data type you should enter only one chromosome name.")
            return
        else:
            print("Error: Please enter the chromosome name.")
            return
    elif len(args.chromosome) > 1:
        CHR_LIST_FLAG = True

    distFilter = parseBP(args.distFilter)  # change
    # --- Distance Filter Handling ---
    # Set a sensible default for the distance filter if not provided by the user
    if not distFilter:
        if 200 * res >= 2000000:
            distFilter = 200 * res
            print("The distance limit is set to {}bp".format(200 * res))
        elif 2000 * res <= 2000000:
            distFilter = 2000 * res
            print("The distance limit is set to {}bp".format(2000 * res))
        else:
            distFilter = 2000000
            print("The distance limit is set to 2Mbp")
    # Enforce reasonable bounds on the user-provided distance filter
    elif distFilter < 200 * res:
        print("The distance limit is set to {}bp".format(200 * res))
        distFilter = 200 * res
    elif distFilter > 10000 * res:
        print("The distance limit is set to {}bp".format(10000 * res))
        distFilter = 10000 * res
    elif distFilter > 10000000:
        distFilter = 10000000
        print("The distance limit is set to 10Mbp")

    # distFilter = 4000000
    chrSize_in_bp = False
    if CHR_COOL_FLAG:
        # extract all the chromosome names big enough to run mustache on
        chr_list = []
        if f.endswith(".cool"):
            clr = cooler.Cooler(f)
        else:  # mcooler
            uri = '%s::/resolutions/%s' % (f, res)
            clr = cooler.Cooler(uri)
        for i, chrm in enumerate(clr.chromnames):
            if clr.chromsizes[i] > 1000000:
                chr_list.append(chrm)
    elif CHR_HIC_FLAG:
        hic = hicstraw.HiCFile(f)
        chromosomes = hic.getChromosomes()
        chr_list = [chromosomes[i].name for i in range(1, len(chromosomes))]
        chrSize_in_bp = {}
        for i in range(1, len(chromosomes)):
            chrSize_in_bp["chr" + str(chromosomes[i].name).replace("chr", '')] = chromosomes[i].length
    else:
        chr_list = args.chromosome.copy()

    if (args.chromosome2 and args.chromosome2 != 'n') and (len(chr_list) != len(args.chromosome2)):
        print("Error: the same number of chromosome1 and chromosome2 should be provided.")
        return
    elif type(args.chromosome2) == list:
        chr_list2 = args.chromosome2.copy()
    else:
        chr_list2 = chr_list.copy()

    CHRM_SIZE = False
    if args.chrSize_file and (not chrSize_in_bp):
        csz_file = args.chrSize_file
        csz = pd.read_csv(csz_file, header=None, sep='\t')
        chrSize_in_bp = {}
        for i in range(csz.shape[0]):
            chrSize_in_bp["chr" + str(csz.iloc[i, 0]).replace('chr', '')] = csz.iloc[i, 1]

    first_chr_to_write = True
    # --- Main Processing Loop ---
    # Iterate through each specified chromosome pair
    for i, (chromosome, chromosome2) in enumerate(zip(chr_list, chr_list2)):
        if chrSize_in_bp:
            CHRM_SIZE = chrSize_in_bp["chr" + str(chromosome).replace('chr', '')]
        biasf = False
        if args.biasfile:
            if os.path.exists(args.biasfile):
                biasf = args.biasfile
            else:
                print("Error: Couldn't find specified bias file")
                return
        # Call the main regulator function to find loops
        o, arrs, gts = regulator(f, args.norm_method, CHRM_SIZE, args.outdir,
                      bed=args.bed,
                      res=res,
                      sigma0=args.s_z,
                      s=args.s,
                      verbose=args.verbose,
                      pt=args.pt,
                      st=args.st,
                      distance_filter=distFilter,
                      nprocesses=args.nprocesses,
                      bias=biasf,
                      chromosome=chromosome,
                      chromosome2=chromosome2,
                      octaves=args.octaves)
        full_chrom = reconstruct_diagonal_matrix(arrs, int(math.ceil(distFilter // res)))
        full_chrom = np.triu(full_chrom) + np.tril(full_chrom.T)
        full_gt = reconstruct_diagonal_matrix(gts, int(math.ceil(distFilter // res)))
        full_gt = np.triu(full_gt) + np.tril(full_gt.T)
        return full_chrom
        # np.save(f'./mustache/output/mustache.{chromosome}.image.npy', full_chrom)
        # np.save(f'./mustache/output/mustache.{chromosome}.label.npy', full_gt)
        # if i == 0:
        #     # Open the output file and write the header
        #     with open(args.outdir, 'w') as out_file:
        #         out_file.write(
        #             "BIN1_CHR\tBIN1_START\tBIN1_END\tBIN2_CHROMOSOME\tBIN2_START\tBIN2_END\tFDR\tDETECTION_SCALE\n")
        # if o == []:
        #     # Format and write each significant loop to the output file
        #     print("{0} loops found for chrmosome={1}, fdr<{2} in {3}sec".format(len(o), chromosome, args.pt,
        #                                                                         "%.2f" % (time.time() - start_time)))
        #     start_time = time.time()
        #     continue
        #
        # # if first_chr_to_write:
        # #    first_chr_to_write = False
        # print("{0} loops found for chrmosome={1}, fdr<{2} in {3}sec".format(len(o), chromosome, args.pt,
        #                                                                     "%.2f" % (time.time() - start_time)))
        #
        # with open(args.outdir, 'a') as out_file:
        #     # out_file.write( "BIN1_CHR\tBIN1_START\tBIN1_END\tBIN2_CHROMOSOME\tBIN2_START\tBIN2_END\tFDR\tDETECTION_SCALE\n")
        #     for significant in o:
        #         out_file.write(
        #             str(chromosome) + '\t' + str(significant[0] * res) + '\t' + str((significant[0] + 1) * res) + '\t' +
        #             str(chromosome2) + '\t' + str(significant[1] * res) + '\t' + str(
        #                 (significant[1] + 1) * res) + '\t' + str(significant[2]) +
        #             '\t' + str(significant[3]) + '\n')
        # # else:
        # #    print("{0} loops found for chrmosome={1}, fdr<{2} in {3}sec".format(len(o),chromosome,args.pt,"%.2f" % (time.time()-old_time)))
        # #    with open(args.outdir, 'a') as out_file:
        # #        for significant in o:
        # #            out_file.write(str(chromosome)+'\t' + str(significant[0]*res) + '\t' + str((significant[0]+1)*res) + '\t' +
        # #	                   str(chromosome2) + '\t' + str(significant[1]*res) + '\t' + str((significant[1]+1)*res) + '\t' + str(significant[2]) +
        # #		           '\t' + str(significant[3]) + '\n')
        # start_time = time.time()


def reconstruct_diagonal_matrix(blocks, overlap_size):
    """
    Reconstructs a full diagonal matrix from overlapping square blocks.

    :param blocks: Array of shape (n_blocks, block_size, block_size)
    :type blocks: numpy.ndarray
    :param overlap_size: Number of pixels that overlap between adjacent blocks
    :type overlap_size: int
    :return: The reconstructed full matrix
    :rtype: numpy.ndarray
    """
    n_blocks, block_size, _ = blocks.shape

    # how much we advance for each block
    step_size = block_size - overlap_size

    # total size of the reconstructed matrix
    total_size = step_size * (n_blocks - 1) + block_size
    result = np.zeros((total_size, total_size))

    for i in range(n_blocks):
        start_pos = i * step_size
        end_pos = start_pos + block_size

        if i == 0:
            # copy first block
            result[start_pos:end_pos, start_pos:end_pos] = blocks[i]
        else:
            # take element-wise maximum in overlapping regions
            result[start_pos:end_pos, start_pos:end_pos] = np.maximum(
                result[start_pos:end_pos, start_pos:end_pos],
                blocks[i]
            )

    return result

# if __name__ == '__main__':
#     main()
