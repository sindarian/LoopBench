import os
from typing import List

import numpy as np
import tensorflow
from tensorflow.keras.backend import epsilon

class ChromosomeGenerator(tensorflow.keras.utils.Sequence):
    def __init__(self,
                 indices: List[int],
                 chromosomes: List[str] = None,
                 data_dir: str = 'dataset/hela_100',
                 name: str = 'Default Generator',
                 patch_size: int = 64,
                 batch_size: int = 32,
                 shuffle: bool = False,
                 include_diagonal: bool = False,
                 use_original: bool = True,
                 upper_bound = None,
                 resolution = 10000):

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
            self.chrom_lengths = [np.load(os.path.join(data_dir, f'imageset.{chrom}.npy'), mmap_mode='r').shape[0]
                                  for chrom in chromosomes]
            self.cum_lengths = np.cumsum(self.chrom_lengths)
        self.indices = indices
        self.on_epoch_end()
        self.include_diagonal = include_diagonal
        self.use_original = use_original
        self.upper_bound = upper_bound
        self.resolution = resolution

    def __len__(self):
        return int(np.ceil(len(self.indices) / self.batch_size))

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indices)

    def __getitem__(self, idx):
        batch_indices = self.indices[idx * self.batch_size:(idx + 1) * self.batch_size]

        if self.use_original:
            x_batch, y_batch, diagonal_batch = self.read_patch_images(batch_indices)
        else:
            x_batch, y_batch, res = self.read_patch_list(batch_indices)


        y_batch_flat = y_batch.reshape(len(batch_indices), -1)
        y_batch_flat = y_batch_flat[:, :, np.newaxis]

        # without resolution
        x = {'patch': x_batch}

        # with resolution
        # res_batch = np.full((len(batch_indices), 1), self.resolution, dtype='float32')
        x = {'patch': x_batch, 'resolution': np.array(res)}

        return x, y_batch_flat
        # return x, y_batch

    def copy(self, batch_size=None, shuffle=None):
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

    def read_patch_images(self, batch_indices):
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

    def read_patch_list(self, batch_indices):
        x_batch = np.zeros((len(batch_indices), self.patch_size, self.patch_size), dtype='float32')
        y_batch = np.zeros((len(batch_indices), self.patch_size, self.patch_size), dtype='float32')
        res = []

        for i, idx in enumerate(batch_indices):
            # r=None
            # idx is a tuple of two numpy arrays (x_patch, y_patch)
            x_patch, y_patch, r = idx # only for multi-res
            # if len(idx) == 2:
            #     x_patch, y_patch = idx
            # else:
            #     x_patch, y_patch, r = idx
            x_batch[i] = x_patch.astype('float32')
            y_batch[i] = y_patch.astype('float32')
            res.append(r) # only for multi-res

            # x_patch, y_patch = idx
            # x_batch[i] = x_patch.astype('float32')
            # y_batch[i] = y_patch.astype('float32')

        return x_batch, y_batch, res
        # return x_batch, x_batch, res
