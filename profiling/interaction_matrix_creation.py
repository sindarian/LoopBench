import os
import shutil
import sys
import time
from pathlib import Path

import h5py
import numpy as np
import cupy as cp
from resource import getrusage, RUSAGE_SELF
from scipy.sparse import diags, lil_matrix, csr_matrix, csc_matrix, save_npz

from gutils import read_txt_data
from hickit.matrix import CisMatrix
from hickit.reader import get_headers
from hickit.utils import get_chrom_sizes
from util.constants import RESOLUTION

def initialise_mat(chr_index, resolution, chrom_size_path):
    """Initializes an empty square numpy matrix for a given chromosome."""
    # Get the sizes of all chromosomes from the provided file.
    chrom_sizes = get_chrom_sizes(chrom_size_path)

    # Calculate the number of bins (loci) for the given chromosome and resolution.
    nloci = int((chrom_sizes[chr_index] / resolution) + 1)

    # Create a square matrix of zeros with the calculated dimensions.
    mat = np.zeros((nloci, nloci))
    return mat

def initialise_cupy_mat(chr_index, resolution, chrom_size_path):
    """Initializes an empty square numpy matrix for a given chromosome."""
    # Get the sizes of all chromosomes from the provided file.
    chrom_sizes = get_chrom_sizes(chrom_size_path)

    # Calculate the number of bins (loci) for the given chromosome and resolution.
    nloci = int((chrom_sizes[chr_index] / resolution) + 1)

    # Create a square matrix of zeros with the calculated dimensions.
    mat = cp.zeros((nloci, nloci))
    return mat

def initialise_sparse_csr_mat(chr_index, resolution, chrom_size_path):
    """Initializes an empty square numpy matrix for a given chromosome."""
    # Get the sizes of all chromosomes from the provided file.
    chrom_sizes = get_chrom_sizes(chrom_size_path)

    # Calculate the number of bins (loci) for the given chromosome and resolution.
    nloci = int((chrom_sizes[chr_index] / resolution) + 1)

    # Create a square matrix of zeros with the calculated dimensions.
    mat = csr_matrix((nloci, nloci))
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

def initialise_sparse_lil_mat(chr_index, resolution, chrom_size_path):
    """Initializes an empty square numpy matrix for a given chromosome."""
    # Get the sizes of all chromosomes from the provided file.
    chrom_sizes = get_chrom_sizes(chrom_size_path)

    # Calculate the number of bins (loci) for the given chromosome and resolution.
    nloci = int((chrom_sizes[chr_index] / resolution) + 1)

    # Create a square matrix of zeros with the calculated dimensions.
    mat = lil_matrix((nloci, nloci))
    return mat

def initialise_h5py_mat(chr_index, resolution, chrom_size_path):
    """Initializes an empty square numpy matrix for a given chromosome."""
    # Get the sizes of all chromosomes from the provided file.
    chrom_sizes = get_chrom_sizes(chrom_size_path)

    # Calculate the number of bins (loci) for the given chromosome and resolution.
    nloci = int((chrom_sizes[chr_index] / resolution) + 1)

    # Create a square matrix of zeros with the calculated dimensions.
    mat = np.zeros((nloci, nloci))

    h5_path = Path(os.path.join('profiling/dataset/hela_100', f'interaction_matrix_hdf5.h5'))
    h5_path.parent.mkdir(parents=True, exist_ok=True)

    with h5py.File(h5_path, "w") as h5f:
        chrm_group = h5f.create_group(f'chrm{chr_index}')
        chrm_group.create_dataset(
            'interactions',
            data=mat,
            compression="lzf"
        )

    return h5_path

def create_interaction_matrix(chr_index, txt_dir, chrom_size_path, resolution=RESOLUTION, filter=True):
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

    if filter:
        return filter_matrix(chr_index, chrom_size_path, resolution, mat).mat
    else:
        return mat

def create_cp_interaction_matrix(chr_index, txt_dir, chrom_size_path, resolution=RESOLUTION, filter=True):
    """
    Creates a dense interaction matrix from sparse text data.

    This reads a 3-column text file (locus1, locus2, value) and populates
    a dense 2D numpy matrix with the interaction values.
    """
    # Read the sparse data from the text file.
    txt_data = read_txt_data(txt_dir, chr_index)

    # Initialize an empty square matrix for the chromosome.
    mat = initialise_cupy_mat(chr_index, resolution, chrom_size_path)

    # Convert the genomic coordinates in the first two columns to integer row/column indices.
    rows = (txt_data[:, 0] / resolution).astype(int)
    cols = (txt_data[:, 1] / resolution).astype(int)

    # Get the interaction values from the third column.
    data = txt_data[:, 2]

    # Populate the matrix at the specified row/column indices with the interaction data.
    mat[rows, cols] = data

    diag = mat.diagonal()
    mat = mat + mat.T
    idx = cp.arange(mat.shape[0])
    mat[idx, idx] -= diag

    if filter:
        return filter_matrix(chr_index, chrom_size_path, resolution, mat).mat
    else:
        return mat

def create_sparse_csr_interaction_matrix(chr_index, txt_dir, chrom_size_path, resolution=RESOLUTION, filter=True):
    # Read the sparse data from the text file.
    txt_data = read_txt_data(txt_dir, chr_index)
    sparse_mat = initialise_sparse_csr_mat(chr_index, resolution, chrom_size_path)

    # Convert the genomic coordinates in the first two columns to integer row/column indices.
    rows = (txt_data[:, 0] / resolution).astype(int)
    cols = (txt_data[:, 1] / resolution).astype(int)

    # Get the interaction values from the third column.
    data = txt_data[:, 2]

    # Populate the matrix at the specified row/column indices with the interaction data.
    sparse_mat[rows, cols] = data

    # Symmetrize the matrix by adding its transpose to itself.
    sparse_mat = sparse_mat + sparse_mat.transpose() - diags(sparse_mat.diagonal())

    if filter:
        sparse_mat = csr_matrix(np.nan_to_num(sparse_mat.toarray(), nan=0.0))

    return sparse_mat

def create_sparse_csc_interaction_matrix(chr_index, txt_dir, chrom_size_path, resolution=RESOLUTION, filter=True):
    # Read the sparse data from the text file.
    txt_data = read_txt_data(txt_dir, chr_index)
    sparse_mat = initialise_sparse_csc_mat(chr_index, resolution, chrom_size_path)

    # Convert the genomic coordinates in the first two columns to integer row/column indices.
    rows = (txt_data[:, 0] / resolution).astype(int)
    cols = (txt_data[:, 1] / resolution).astype(int)

    # Get the interaction values from the third column.
    data = txt_data[:, 2]

    # Populate the matrix at the specified row/column indices with the interaction data.
    sparse_mat[rows, cols] = data

    # Symmetrize the matrix by adding its transpose to itself.
    sparse_mat = sparse_mat + sparse_mat.transpose() - diags(sparse_mat.diagonal())

    if filter:
        sparse_mat = csc_matrix(np.nan_to_num(sparse_mat.toarray(), nan=0.0))

    return sparse_mat

def create_sparse_lil_interaction_matrix(chr_index, txt_dir, chrom_size_path, resolution=RESOLUTION, filter=True):
    # Read the sparse data from the text file.
    txt_data = read_txt_data(txt_dir, chr_index)
    sparse_mat = initialise_sparse_lil_mat(chr_index, resolution, chrom_size_path)

    # Convert the genomic coordinates in the first two columns to integer row/column indices.
    rows = (txt_data[:, 0] / resolution).astype(int)
    cols = (txt_data[:, 1] / resolution).astype(int)

    # Get the interaction values from the third column.
    data = txt_data[:, 2]

    # Populate the matrix at the specified row/column indices with the interaction data.
    sparse_mat[rows, cols] = data

    # Symmetrize the matrix by adding its transpose to itself.
    sparse_mat = sparse_mat + sparse_mat.transpose() - diags(sparse_mat.diagonal())

    if filter:
        dense = sparse_mat.toarray()
        dense = np.nan_to_num(dense, nan=0.0)
        sparse_mat = lil_matrix(dense)

    return sparse_mat


def create_hdf5_interaction_matrix(chr_index, txt_dir, chrom_size_path, resolution=RESOLUTION, filter=True):
    """
    Creates a dense interaction matrix from sparse text data.

    This reads a 3-column text file (locus1, locus2, value) and populates
    a dense 2D numpy matrix with the interaction values.
    """
    # Read the sparse data from the text file.
    txt_data = read_txt_data(txt_dir, chr_index)
    # Initialize an empty square matrix for the chromosome.
    h5_path = initialise_h5py_mat(chr_index, resolution, chrom_size_path)
    size = None
    with h5py.File(h5_path, "r+") as h5f:
        # Convert the genomic coordinates in the first two columns to integer row/column indices.
        rows = (txt_data[:, 0] / resolution).astype(int)
        cols = (txt_data[:, 1] / resolution).astype(int)

        # Get the interaction values from the third column.
        data = txt_data[:, 2]

        # Populate the matrix at the specified row/column indices with the interaction data.
        hmat = h5f[f'chrm{chr_index}']['interactions']
        group = h5f[f'chrm{chr_index}']
        # hmat[rows, cols] = data

        arr = hmat[...]
        arr[rows, cols] = data

        arr = np.triu(arr) + np.tril(arr.T, 1)

        if filter:
            arr = filter_matrix(chr_index, chrom_size_path, resolution, arr).mat

            # delete old dataset if it exists
            if "interactions" in group:
                del group["interactions"]

            # create new dataset with correct shape
            group.create_dataset(
                "interactions",
                data=arr,
                compression="lzf"
            )
        else:
            hmat[:, :] = arr  # Write back

            # Write back the symmetrized matrix to the HDF5 dataset
            h5f[f'chrm{chr_index}']['interactions'][:, :] = hmat

    return h5_path.stat().st_size

def filter_matrix(chr_index, chrom_size_path, resolution, mat):
    # Get the genomic headers (coordinate information) for this chromosome.
    headers = get_headers([chr_index], get_chrom_sizes(chrom_size_path), resolution)

    # Create a CisMatrix object, which bundles the matrix data with its headers.
    matrix = CisMatrix(headers[headers['chrom'] == chr_index], mat, resolution)
    matrix.filter_by_nan_percentage(0.9999)

    return matrix

def time_interaction_matrix_creation(chrom_name, txt_dir, resolution, chrom_sizes_path, method='numpy', filter=True):
    method_funcs = {
        'numpy': create_interaction_matrix,
        'cupy': create_cp_interaction_matrix,
        'csr': create_sparse_csr_interaction_matrix,
        'csc': create_sparse_csc_interaction_matrix,
        'lil': create_sparse_lil_interaction_matrix,
        'hdf5': create_hdf5_interaction_matrix,
    }

    start = time.time()
    X = method_funcs[method](chrom_name, txt_dir, chrom_sizes_path, resolution, filter)
    stop = time.time()

    total_time = stop - start
    if method == 'csr' or method == 'csc':
        print(f'\nSparse {method} Interation Matrix: {total_time} s, '
              f'{(X.data.nbytes + X.indptr.nbytes + X.indices.nbytes) / 1024 / 1024 / 1024} GB')
    elif method == 'lil':
        print(f'\nSparse {method} Interation Matrix: {total_time} s, '
              f'(data size will be equivalent to size on disk) GB')
    elif method == 'hdf5':
        # return values for the hdf5 method is the file size
        print(f'\nHDF5 File size on disk: {total_time} s, {X / 1024 / 1024 / 1024} GB')
    else:
        print(f'\n{method} Interaction Matrix: {total_time} s, {X.nbytes / 1024 / 1024 / 1024} GB')

    print()
    print(f'\nPeak Memory Usage (KB): {getrusage(RUSAGE_SELF).ru_maxrss}')
    print(f'Peak Memory Usage (MB): {getrusage(RUSAGE_SELF).ru_maxrss / 1024}')
    print(f'Peak Memory Usage (GB): {getrusage(RUSAGE_SELF).ru_maxrss / 1024 / 1024}')

    return X

if __name__ == '__main__':
    method = 'numpy'

    filter_vals = [True, False]
    if len(sys.argv) > 1:
        method = sys.argv[1]

    for filter_val in filter_vals:
        print()
        print('*******')
        print(f'Filtering: {filter_val}')
        print('*******')
        interaction_matrix = time_interaction_matrix_creation(chrom_name='1',
                                                              txt_dir='data/txt_hela_100',
                                                              resolution=RESOLUTION,
                                                              chrom_sizes_path='hg38.chrom.sizes',
                                                              method=method,
                                                              filter=filter_val)

        output_dir = Path('profiling/dataset/hela_100')
        output_dir.mkdir(parents=True, exist_ok=True)

        save_path = output_dir / f'interaction_matrix_{method}'

        if method == 'hdf5':
            save_path = save_path.with_suffix('.h5')
            with h5py.File(save_path, "r+") as h5f:
                dataset = h5f[f'chrm1']['interactions']

                size_bytes = dataset.size * dataset.dtype.itemsize
                size_kb = size_bytes / 1024
                size_mb = size_kb / 1024
                size_gb = size_mb / 1024

                print()
                print(f'In mem size (bytes): {size_bytes}')
                print(f'In mem size (KB): {size_kb}')
                print(f'In mem size (MB): {size_mb}')
                print(f'In mem size (GB): {size_gb}')
        else:
            if method == 'cupy':
                # CuPy matrix needs to be moved to host before saving
                save_path = save_path.with_suffix('.npy')
                np.save(save_path, cp.asnumpy(interaction_matrix))
            elif method in ['csr', 'csc']:
                # Save sparse matrices using .npz format
                save_path = save_path.with_suffix('.npz')
                save_npz(save_path, interaction_matrix)
            elif method == 'lil':
                import pickle
                with open(save_path, "wb") as f:
                    pickle.dump(interaction_matrix, f)
            else:
                # Save NumPy dense matrix
                save_path = save_path.with_suffix('.npy')
                np.save(save_path, interaction_matrix)

            size_bytes = save_path.stat().st_size
            size_kb = size_bytes / 1024
            size_mb = size_kb / 1024
            size_gb = size_mb / 1024

            print()
            print(f'File size on disk (bytes): {size_bytes}')
            print(f'File size on disk (KB): {size_kb}')
            print(f'File size on disk (MB): {size_mb}')
            print(f'File size on disk (GB): {size_gb}')

        shutil.rmtree(output_dir)
