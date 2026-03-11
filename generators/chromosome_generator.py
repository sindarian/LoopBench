import os
from typing import List

import numpy as np
import tensorflow
from tensorflow.keras.backend import epsilon
import sparse

class ChromosomeGenerator(tensorflow.keras.utils.Sequence):
        """
    A TensorFlow Keras data generator for Hi-C contact patch datasets.

    Supports two data loading modes:
        - Original (use_original=True): Loads patches directly from per-chromosome
          .npy files on disk using memory-mapped I/O, applying log normalization.
        - Experiment (use_original=False): Loads pre-assembled (x, y) or
          (x, y, resolution) patch tuples stored in memory as a list.

    Inherits from keras.utils.Sequence to ensure safe multiprocessing and
    correct epoch-end shuffling during training.
    """

    def __init__(
        self,
        indices: List,
        chromosomes: List[str] = None,
        data_dir: str = 'dataset/hela_100',
        name: str = 'Default Generator',
        patch_size: int = 64,
        batch_size: int = 32,
        shuffle: bool = False,
        include_diagonal: bool = False,
        use_original: bool = True,
        upper_bound: Optional[float] = None,
        resolution: int = 10000,
    ):
            """
        Args:
            indices (List): Sample indices for the original pipeline (int indices into
                            cumulative chromosome arrays) or patch tuples for the
                            experiment pipeline.
            chromosomes (List[str]): Chromosomes to load. Defaults to chr1–22 and X.
            data_dir (str): Directory containing per-chromosome .npy files.
            name (str): Display name for the generator used in logging.
            patch_size (int): Height and width of each square contact patch. Defaults to 64.
            batch_size (int): Number of samples per batch. Defaults to 32.
            shuffle (bool): If True, shuffles indices at the end of each epoch. Defaults to False.
            include_diagonal (bool): If True, loads diagonal distance flags alongside patches.
                                     Only used in the original pipeline. Defaults to False.
            use_original (bool): If True, uses the original file-based loading pipeline.
                                 If False, uses the in-memory experiment patch list. Defaults to True.
            upper_bound (float | None): Optional upper bound for contact value clipping. Currently unused.
            resolution (int): Hi-C resolution in base pairs. Used as a per-sample feature
                              in multi-resolution mode. Defaults to 10000.
        """
        if chromosomes is None:
            self.chromosomes = [str(i) for i in range(1, 23)] + ['X']
        else:
            self.chromosomes = chromosomes

        self.data_dir = data_dir
        self.patch_size = patch_size
        self.name = name
        self.batch_size = batch_size
        self.shuffle = shuffle

        if use_original:
            # Precompute per-chromosome patch counts and cumulative lengths for index mapping
            self.chrom_lengths = [np.load(os.path.join(data_dir, f'imageset.{chrom}.npy'), mmap_mode='r').shape[0]
                                  for chrom in chromosomes]
            self.cum_lengths = np.cumsum(self.chrom_lengths)

        self.indices = indices
        self.on_epoch_end()
        self.include_diagonal = include_diagonal
        self.use_original = use_original
        self.upper_bound = upper_bound
        self.resolution = resolution

    def __len__(self) -> int:
        """Returns the number of batches per epoch."""
        return int(np.ceil(len(self.indices) / self.batch_size))

    def on_epoch_end(self) -> None:
        """Shuffles indices at the end of each epoch if shuffle is enabled."""
        if self.shuffle:
            np.random.shuffle(self.indices)

    def __getitem__(self, idx: int) -> Tuple[dict, np.ndarray]:
        """
        Returns a single batch of inputs and labels.

        Inputs are returned as a dict with key 'patch' (and optionally 'resolution'
        in multi-resolution mode). Labels are flattened to shape (batch, patch_size^2, 1).

        Args:
            idx (int): Batch index.

        Returns:
            Tuple[dict, np.ndarray]:
                - x: Dict with key 'patch' of shape (batch, patch_size, patch_size),
                     and optionally 'resolution' of shape (batch,).
                - y: Label array of shape (batch, patch_size * patch_size, 1).
        """
        batch_indices = self.indices[idx * self.batch_size:(idx + 1) * self.batch_size]

        if self.use_original:
            x_batch, y_batch, diagonal_batch = self.read_patch_images(batch_indices)
            res = None
        else:
            x_batch, y_batch, res = self.read_patch_list(batch_indices)

        # Flatten labels to (batch, patch_size^2, 1) for compatibility with the loss function
        y_batch_flat = y_batch.reshape(len(batch_indices), -1)
        y_batch_flat = y_batch_flat[:, :, np.newaxis]

        # without resolution
        x = {'patch': x_batch}

        x = {'patch': x_batch}
        if res is not None:
            x ['resolution'] = np.array(res)

        return x, y_batch_flat

    def copy(
        self,
        batch_size: int = None,
        shuffle: bool = None,
    ) -> 'ChromosomeGenerator':
        """
        Creates a shallow copy of this generator with optional overrides for
        batch size and shuffle behavior. All other configuration is preserved.

        Args:
            batch_size (int | None): Override batch size. Uses current value if None.
            shuffle (bool | None): Override shuffle setting. Uses current value if None.

        Returns:
            ChromosomeGenerator: New generator instance with the same data and config.
        """
        return ChromosomeGenerator(chromosomes=self.chromosomes,
                                   data_dir=self.data_dir,
                                   patch_size=self.patch_size,
                                   indices=self.indices,
                                   name=self.name,
                                   batch_size=batch_size if batch_size is not None else self.batch_size,
                                   shuffle=shuffle if shuffle is not None else self.shuffle,
                                   include_diagonal=self.include_diagonal,
                                   use_original=self.use_original,
                                   resolution=self.resolution)

    def read_patch_images(
        self,
        batch_indices: List[int],
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Loads a batch of patches from per-chromosome .npy files using memory-mapped I/O.
        Maps global sample indices to per-chromosome local indices using cumulative lengths,
        then applies log normalization to the contact patches.

        Args:
            batch_indices (List[int]): Global sample indices for this batch.

        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray]:
                - x_batch: Log-normalized contact patches of shape (batch, patch_size, patch_size).
                - y_batch: Binary label patches of shape (batch, patch_size, patch_size).
                - diagonal_batch: Diagonal distance flags of shape (batch, 1).
                  Only populated if include_diagonal is True.
        """
        x_batch = np.zeros((len(batch_indices), self.patch_size, self.patch_size), dtype='float32')
        diagonal_batch = np.zeros((len(batch_indices), 1), dtype='int')
        y_batch = np.zeros((len(batch_indices), self.patch_size, self.patch_size), dtype='float32')

        for i, sample_idx in enumerate(batch_indices):
            chrom_idx = np.searchsorted(self.cum_lengths, sample_idx, side='right')
            if chrom_idx == 0:
                local_idx = sample_idx
            else:
                local_idx = sample_idx - self.cum_lengths[chrom_idx - 1]

            img_path = os.path.join(self.data_dir, f'imageset.{self.chromosomes[chrom_idx]}.npy')
            lbl_path = os.path.join(self.data_dir, f'labels.{self.chromosomes[chrom_idx]}.npy')

            imageset = np.load(img_path, mmap_mode='r')
            labels = np.load(lbl_path, mmap_mode='r')

            x_batch[i] = np.log(imageset[local_idx] + 1 + epsilon())
            y_batch[i] = labels[local_idx].astype('int')

            # Load diagonal data if needed
            if self.include_diagonal:
                diagonal_path = os.path.join(self.data_dir, f'diagonals.{self.chromosomes[chrom_idx]}.npy')
                diagonal_flags = np.load(diagonal_path, mmap_mode='r')
                diagonal_batch[i] = int(diagonal_flags[local_idx])

        return x_batch, y_batch, diagonal_batch

    def read_patch_list(
        self,
        batch_indices: List,
    ) -> Tuple[np.ndarray, np.ndarray, Optional[List]]:
        """
        Loads a batch of patches from in-memory patch tuples. Each entry in batch_indices
        is expected to be a (x_patch, y_patch) or (x_patch, y_patch, resolution) tuple.

        Args:
            batch_indices (List): List of (x_patch, y_patch) or (x_patch, y_patch, resolution)
                                  tuples for this batch.

        Returns:
            Tuple[np.ndarray, np.ndarray, List | None]:
                - x_batch: Contact patches of shape (batch, patch_size, patch_size).
                - y_batch: Label patches of shape (batch, patch_size, patch_size).
                - res: List of per-sample resolutions if present, otherwise None.
        """
        x_batch = np.zeros((len(batch_indices), self.patch_size, self.patch_size), dtype='float32')
        y_batch = np.zeros((len(batch_indices), self.patch_size, self.patch_size), dtype='float32')
        res = []

        for i, idx in enumerate(batch_indices):
            # r=None
            # idx is a tuple of two numpy arrays (x_patch, y_patch)
            # x_patch, y_patch, r = idx # only for multi-res
            if len(idx) == 2:
                x_patch, y_patch = idx
                r = None
            else:
                x_patch, y_patch, r = idx  # only for multi-res

            x_batch[i] = x_patch.astype('float32')
            y_batch[i] = y_patch.astype('float32')
            # res.append(r) # only for multi-res
            if r is not None:
                res.append(r)

            # x_patch, y_patch = idx
            # x_batch[i] = x_patch.astype('float32')
            # y_batch[i] = y_patch.astype('float32')

        return x_batch, y_batch, res if len(res) > 0 else None
        # return x_batch, x_batch, res
