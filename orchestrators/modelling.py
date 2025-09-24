import logging
import os

from tensorflow.keras.models import load_model


import numpy as np
import tensorflow as tf

from generators.chromosome_generator import ChromosomeGenerator
from model.custom_layers import ClipByValue
from predict import run_output_predictions
from train import train_loop_net, train_run_cnn
from util.constants import PATCH_SIZES, RESOLUTIONS, OUTPUT_DIR, METRICS_DIR, ONE_PATCHE_MANY_RES, MANY_PATCHES_ONE_RES, \
    MODELS_DIR, PATCH_SIZE, RESOLUTION
from util.logger import Logger
from util.plotting import plotting
from util.plotting.plotting import plot_pixel_counts, plot_top_positive_patches
from util.utils import count_pos_neg_distributions

LOGGER = Logger(name='data_sampler', level=logging.DEBUG).get_logger()

def train_models(cell_line, chroms, patch_sizes, resolutions):
    # for patch_size, resolution in list(product(patch_sizes, resolutions)):
    # split the data into generators
    train_gen, val_gen, test_gen = split_data(cell_line, chroms, PATCH_SIZE, RESOLUTION)

    # estimate the upper bound of the training data
    # this is used to scale the data and clip outliers when the model is built
    x_train_upper_bound = _estimate_upper_bound(train_gen, percentile=0.996)

    # build, train, and run a CNN and LoopNet for loop calling
    # train_run_cnn(train_gen=train_gen.copy(),
    #               val_gen=val_gen.copy(),
    #               test_gen=test_gen.copy(),
    #               clip_value=x_train_upper_bound,
    #               patch_size=PATCH_SIZE,
    #               resolution=RESOLUTION,
    #               epochs=1)
    train_loop_net(train_gen=train_gen.copy(),
                   val_gen=val_gen.copy(),
                   test_gen=test_gen.copy(),
                   clip_value=x_train_upper_bound,
                   patch_size=PATCH_SIZE,
                   resolution=RESOLUTION,
                   epochs=10)
    plotting.plot_training_history()

def test_models(cnn_path, loopnet_path, cell_line, assembly, chroms):
    cnn = load_model(cnn_path, custom_objects={'ClipByValue': ClipByValue})
    loopnet = load_model(loopnet_path, custom_objects={'ClipByValue': ClipByValue})

    LOGGER.info('Testing the CNN on the target cell line')
    # Predict on the target cell line
    run_output_predictions(model=cnn,
                           target_dataset_name=cell_line,
                           target_assembly=assembly,
                           chroms=chroms)

    LOGGER.info('Testing LoopNet on the target cell line')
    # Predict on the target cell line
    run_output_predictions(model=loopnet,
                           target_dataset_name=cell_line,
                           target_assembly=assembly,
                           chroms=chroms)

def _estimate_upper_bound(generator, percentile=0.996, max_pixels=2000000):
    collected = []

    for batch_x, _ in generator:
        pixels = tf.reshape(batch_x['patch'], [-1]).numpy()
        collected.extend(pixels.tolist())

        if len(collected) >= max_pixels:
            break
    # for batch_x, _ in generator:
    #     pixels = tf.reshape(batch_x, [-1]).numpy()
    #     collected.extend(pixels.tolist())
    #
    #     if len(collected) >= max_pixels:
    #         break

    collected = np.array(collected[:max_pixels])
    upper_bound = np.quantile(collected, percentile)
    LOGGER.info(f'x_train_upper_bound: {upper_bound}')
    return upper_bound

def _create_hic_generators(chrom_names, data_dir, patch_size,
                          split_ratios=(0.7, 0.2, 0.1),
                          batch_size=32):
    # Total number of samples
    chrom_lengths = [np.load(os.path.join(data_dir, f'imageset.{cn}.npy'), mmap_mode='r').shape[0]
                     for cn in chrom_names]
    total_samples = sum(chrom_lengths)
    all_indices = np.arange(total_samples)

    # Shuffle
    np.random.shuffle(all_indices)

    # Compute split boundaries
    train_end = int(split_ratios[0] * total_samples)
    val_end = train_end + int(split_ratios[1] * total_samples)

    train_idx = all_indices[:train_end]
    val_idx = all_indices[train_end:val_end]
    test_idx = all_indices[val_end:]

    # Create generators
    train_gen = ChromosomeGenerator(chromosomes=chrom_names,
                                    data_dir=data_dir,
                                    patch_size=patch_size,
                                    indices=train_idx,
                                    batch_size=batch_size,
                                    name='Train Generator',
                                    shuffle=True)
    val_gen = ChromosomeGenerator(chromosomes=chrom_names,
                                  data_dir=data_dir,
                                  patch_size=patch_size,
                                  indices=val_idx,
                                  name='Validation Generator',
                                  batch_size=batch_size,
                                  shuffle=False)
    test_gen = ChromosomeGenerator(chromosomes=chrom_names,
                                   data_dir=data_dir,
                                   patch_size=patch_size,
                                   indices=test_idx,
                                   batch_size=batch_size,
                                   name='Test Generator',
                                   shuffle=False)

    return train_gen, val_gen, test_gen

def split_data(cell_line, chroms, patch_size, resolution):
    LOGGER.info(
        f'Create Generators for Train, Val, and Test Data - patch size {patch_size} w/ resolution {resolution}')
    train_gen, val_gen, test_gen = _create_hic_generators(
        chrom_names=chroms,
        data_dir=os.path.join(f'dataset', cell_line),
        patch_size=patch_size,
        split_ratios=(0.7, 0.2, 0.1),  # train, val, test
        batch_size=8
    )

    # visualize the training data distribution
    _visualize_data(generator=train_gen.copy(1, False), patch_size=patch_size, resolution=resolution)

    return train_gen, val_gen, test_gen

def _visualize_data(generator: ChromosomeGenerator,
                    patch_size: int,
                    resolution: int,
                    limit: int = None,
                    plot_neg: bool = False):
    # compute distributions
    train_counts = count_pos_neg_distributions(generator, patch_size)

    # plot distributions
    plot_pixel_counts(train_counts, generator.name, limit, plot_neg, patch_size, resolution)

    # plot the heatmap for the top 4 images
    plot_top_positive_patches(train_counts, 4, patch_size, resolution, 'x')
    plot_top_positive_patches(train_counts, 4, patch_size, resolution, 'y')