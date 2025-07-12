import numpy as np
import os
import tensorflow as tf

class HiCDatasetGenerator(tf.keras.utils.Sequence):
    def __init__(self, chrom_names, data_dir, patch_size, indices, name, batch_size=32, shuffle=True):
        self.chrom_names = chrom_names
        self.data_dir = data_dir
        self.patch_size = patch_size
        self.name = name
        self.batch_size = batch_size
        self.shuffle = shuffle

        self.chrom_lengths = [np.load(os.path.join(data_dir, f'imageset.{cn}.npy'), mmap_mode='r').shape[0]
                              for cn in chrom_names]
        self.cum_lengths = np.cumsum(self.chrom_lengths)

        self.indices = indices
        self.on_epoch_end()

    def __len__(self):
        return int(np.ceil(len(self.indices) / self.batch_size))

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indices)

    def __getitem__(self, idx):
        batch_indices = self.indices[idx * self.batch_size:(idx + 1) * self.batch_size]
        x_batch = np.zeros((len(batch_indices), self.patch_size, self.patch_size), dtype='float32')
        y_batch = np.zeros((len(batch_indices), self.patch_size, self.patch_size), dtype='int')

        for i, sample_idx in enumerate(batch_indices):
            chrom_idx = np.searchsorted(self.cum_lengths, sample_idx, side='right')
            if chrom_idx == 0:
                local_idx = sample_idx
            else:
                local_idx = sample_idx - self.cum_lengths[chrom_idx - 1]

            img_path = os.path.join(self.data_dir, f'imageset.{self.chrom_names[chrom_idx]}.npy')
            lbl_path = os.path.join(self.data_dir, f'labels.{self.chrom_names[chrom_idx]}.npy')

            imageset = np.load(img_path, mmap_mode='r')
            labels = np.load(lbl_path, mmap_mode='r')

            x_batch[i] = np.log(imageset[local_idx] + 1)
            y_batch[i] = labels[local_idx].astype('int')

        y_batch_flat = y_batch.reshape(len(batch_indices), -1)[..., np.newaxis]
        return x_batch, y_batch_flat
