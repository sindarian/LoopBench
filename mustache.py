import time
from multiprocessing import Process, Manager

import math
import numpy as np
from scipy.ndimage import gaussian_filter, maximum_filter
from scipy.stats import expon
from statsmodels.stats.multitest import multipletests
import scipy.ndimage.measurements as scipy_measurements


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

# def mustache(c, chromosome, chromosome2, res, pval_weights, start, end, mask_size, distance_in_px, octave_values, st,
#              pt, block_num, num_blocks):
def mustache(c, start, distance_in_px, octave_values, st, pt):
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
    if np.sum(nz) < 50:
        print(f'###### too sparse: {np.sum(nz)}')
        # return []
        return np.zeros(c.shape), np.zeros(c.shape)

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
        print(f'###### to few candidates: {pFound}')
        # return []
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
        print(f'###### len(x) == 0')
        # return []
        return np.zeros(c.shape), np.zeros(c.shape)

    # --- Enrichment Filter (for intrachromosomal only) ---
    def nz_mean(vals):
        return np.mean(vals[vals != 0])

    def diag_mean(k, map):
        return nz_mean(map[kth_diag_indices(map, k)])

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

    out = []
    # For each found cluster
    for label in range(1, num_features + 1):
        # Find all pixels belonging to this cluster
        indices = np.argwhere(label_matrix == label)
        # Find the pixel within the cluster that has the lowest p-value
        i = np.argmin(o[indices[:, 0], indices[:, 1]])
        _x, _y = indices[i, 0], indices[i, 1]
        # Append this representative pixel to the output list, adjusting coordinates back to full chromosome scale
        out.append([_x + start, _y + start, o[_x, _y], so[_x, _y]])
    return out, Lcmax

def process_block(i, start, cc, overlap_size, distance_in_px, octave_values, st, pt, arrs, data, start_i, end_i, pos):
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

    # Run the core loop detection algorithm on the block
    _, Lcmax = mustache(cc, start[i], distance_in_px, octave_values, st, pt)
    arrs.append(Lcmax)
    data[i] = \
        {
            'dog': Lcmax,
            'start_end': [start_i, end_i]
        }
    pos.append((start_i, end_i))

    print("Block", i + 1, "done.")

def regulator(contacts,
              res=10000,
              sigma0=1.6,
              pt=0.1,
              st=0.88,
              octaves=2,
              distance_filter=2000000,
              nprocesses=4):

    # Define the sigma values for the scale-space search based on octaves
    octave_values = [sigma0 * (2 ** i) for i in range(octaves)]
    distance_in_px = int(math.ceil(distance_filter // res))

    # Get the full matrix size
    n = contacts.shape[0]

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
        arrs = manager.list()
        data = manager.dict()
        pos = manager.list()
        # i = 0
        processes = []
        # Iterate over each chunk
        for i in range(len(start)):
        # for i in range(10):
            # Create the dense matrix for the current block
            # idx_end = start[i] + CHUNK_SIZE if start[i] + CHUNK_SIZE <= n else n
            # cc = contacts[start[i]:idx_end, start[i]:idx_end].toarray()
            cc = contacts[start[i]:end[i], start[i]:end[i]].toarray()
            print(f'({start[i]}, {end[i]})')

            # Create and start a new process for this chunk
            p = Process(target=process_block, args=(i, start, cc, overlap_size, distance_in_px, octave_values, st, pt, arrs, data, start[i], end[i], pos))
            p.start()
            processes.append(p)

            # Wait for processes to finish if the pool is full or it's the last chunk
            if len(processes) >= nprocesses or i == (len(start) - 1):
                for p in processes:
                    p.join()
                processes = []

        # Return the aggregated results from all processes
        return np.asarray(list(arrs), dtype=np.float32), np.asarray(pos)

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

def mustache_norm(contacts,
                  res=10000,
                  sigma0=1.6,
                  pt=0.1,
                  st=0.88,
                  octaves=2,
                  distance_filter=2000000,
                  num_processes=4):
    start = time.time()
    arrs, pos = regulator(contacts, res, sigma0, pt, st, octaves, distance_filter, num_processes)
    end = time.time()
    print(f"mustache_norm execution time (s): {end - start}")

    # full_chrom = reconstruct_diagonal_matrix(arrs, int(math.ceil(distance_filter // res)))

    new_contacts = np.zeros(contacts.shape)
    for arr, start_end in zip(arrs, sorted(pos, key=lambda x: x[0])):
        start, end = start_end
        new_contacts[start:end, start:end] = arr[:,:]

    # full_chrom = np.triu(full_chrom) + np.tril(full_chrom.T)
    return new_contacts