import numpy as np
import os
import random
from collections import OrderedDict
import pandas as pd
from sklearn.metrics import f1_score
import tensorflow as tf

from constants import SEED, PATCH_SIZE
from hi_c_data_generator import HiCDatasetGenerator
from logger import Logger
import logging

from util.plotting.plotting import plot_pixel_counts

LOGGER = Logger(name='utils', level=logging.DEBUG).get_logger()

np.random.seed(SEED)
random.seed(SEED)

def cool2txt(cooler_path):
    pass

def get_best_threshold(y_score, y_true, thresholds):
    largest_f1 = 0
    best_thresh = None
    for thresh in thresholds:
        y_pred_binary = (y_score>thresh)
        flanking_f1 = f1_score(y_true.flatten(), y_pred_binary.flatten())
        if flanking_f1 > largest_f1:
            largest_f1 = flanking_f1
            best_thresh = thresh
    return best_thresh

def get_chrom_pred_df(chrom_name, chrom_proba, threshold, the_headers, resolution=10000):
    assert chrom_proba.shape[0] == len(the_headers)
    chrom_name = 'chr' + chrom_name
    chrom_binary_pred = (chrom_proba > threshold)
    pos_coords = np.argwhere(chrom_binary_pred)
    locus1_start = pos_coords[:, 0] * resolution
    locus1_end = (pos_coords[:, 0] + 1) * resolution
    locus2_start = pos_coords[:, 1] * resolution
    locus2_end = (pos_coords[:, 1] + 1) * resolution
    the_dict = OrderedDict()
    the_dict['chrom1'] = chrom_name
    the_dict['locus1_start'] = locus1_start
    the_dict['locus1_end'] = locus1_end
    the_dict['chrom2'] = chrom_name
    the_dict['locus2_start'] = locus2_start
    the_dict['locus2_end'] = locus2_end
    df = pd.DataFrame(the_dict)
    return df

def output_chrom_pred_to_bedpe(chrom_name, chrom_proba, threshold, full_headers, output_dir, resolution):
    assert chrom_proba.shape[0] == len(full_headers)
    chrom_name = 'chr' + chrom_name
    chrom_binary_pred = (chrom_proba > threshold)
    pos_coords = np.argwhere(chrom_binary_pred)
    locus1_start = pos_coords[:, 0] * resolution
    locus1_end = (pos_coords[:, 0] + 1) * resolution
    locus2_start = pos_coords[:, 1] * resolution
    locus2_end = (pos_coords[:, 1] + 1) * resolution
    the_dict = OrderedDict()
    the_dict['chrom1'] = chrom_name
    the_dict['locus1_start'] = locus1_start
    the_dict['locus1_end'] = locus1_end
    the_dict['chrom2'] = chrom_name
    the_dict['locus2_start'] = locus2_start
    the_dict['locus2_end'] = locus2_end
    df = pd.DataFrame(the_dict)
    os.makedirs(output_dir, exist_ok=True)
    bedpe_path = os.path.join(output_dir, '{}.GILoop_pred.bedpe'.format(chrom_name))
    df.to_csv(bedpe_path, sep='\t', header=False, index=False)

def get_chrom_proba(chrom_name_to_score, chrom_sizes, resolution, pred_results, labels, indicator_path, identical_path, patch_size):
    hic_size = int(chrom_sizes[chrom_name_to_score]/resolution) + 1
    score_matrix = np.zeros((hic_size, hic_size), dtype=np.float32)
    ground_truth = np.zeros((hic_size, hic_size), dtype=np.int32)
    graph_identicals = np.load(identical_path)
    indicators = pd.read_csv(
        indicator_path,
        sep=',', index_col=0, dtype={'chrom': 'str'}
    )
    loci_indicators = indicators['locus'].values
    pred_results = pred_results.reshape((-1, patch_size, patch_size))
    labels = labels.reshape((-1, patch_size, patch_size))
    for i, pred_patch in enumerate(pred_results):
        patch_label = labels[i]
        if graph_identicals[i]:
            pred_patch = (pred_patch + pred_patch.transpose()) / 2
        current_patch_indicators = loci_indicators[i*patch_size*2:(i*patch_size*2)+patch_size*2]
        for row in range(patch_size):
            for col in range(patch_size):
                score = pred_patch[row, col]
                y = patch_label[row, col]
                row_locus = current_patch_indicators[row]
                col_locus = current_patch_indicators[patch_size + col]
                if row_locus >= 0 and col_locus >= 0:
                    score_matrix[row_locus, col_locus] = score
                    ground_truth[row_locus, col_locus] = y
    score_matrix = np.triu(score_matrix) + np.tril(score_matrix.T, 1)
    ground_truth = np.triu(ground_truth) + np.tril(ground_truth.T, 1)
    return score_matrix, ground_truth

def normalise_graphs(adjs):
    for i, adj in enumerate(adjs):
        np.fill_diagonal(adj, 0)
        assert np.diag(adj).sum() == 0
        adj_ = adj + np.eye(adj.shape[0])
        rowsum = np.sum(adj_, axis=1)
        degree_mat_inv_sqrt = np.diag(np.power(rowsum, -0.5).flatten())
        adj_normalized = (adj_ @ degree_mat_inv_sqrt).transpose() @ (degree_mat_inv_sqrt)
        adjs[i] = adj_normalized
    return adjs

def scale_hic(hic, max_hic_value):
    hic[hic>max_hic_value] = max_hic_value
    hic = hic * (1/max_hic_value)
    return hic

def read_data_with_motif(chrom_names, data_dir, patch_size):
    total_cnt = 0
    for cn in chrom_names:
        _ = np.load(os.path.join(data_dir, 'imageset.{}.npy'.format(cn)))
        total_cnt += len(_)
    imageset = np.zeros((total_cnt, patch_size, patch_size), dtype='float32')
    graphset = np.zeros((total_cnt, 2 * patch_size, 2 * patch_size), dtype='float32')
    labels = np.zeros((total_cnt, patch_size, patch_size), dtype='bool')
    node_features = None

    current_start = 0
    for cn in chrom_names:
        current_image = np.load(os.path.join(data_dir, 'imageset.{}.npy'.format(cn)))
        current_graph = np.load(os.path.join(data_dir, 'graphset.{}.npy'.format(cn)))
        current_y = np.load(os.path.join(data_dir, 'labels.{}.npy'.format(cn)))

        kmer_features = np.load(os.path.join(data_dir, 'node_features.{}.npy'.format(cn)))
        motif_features = np.load(os.path.join(data_dir, 'motif_features.{}.npy'.format(cn)))
        current_features = np.concatenate((kmer_features, motif_features), axis=-1)

        if node_features is None:
            node_features = np.zeros((total_cnt, 2 * patch_size, current_features.shape[2]), dtype='float32')

        current_end = current_start + len(current_image)
        imageset[current_start:current_end, :, :] = current_image
        graphset[current_start:current_end, :, :] = current_graph
        labels[current_start:current_end, :, :] = current_y
        node_features[current_start:current_end, :, :] = current_features
        current_start = current_end

    return np.log(imageset + 1), np.log(graphset + 1), labels.astype('int'), node_features


def get_split_dataset(dataset_dir, image_size, seed, chroms):
    images, graphs, y, features = read_data_with_motif(chroms, dataset_dir, image_size)

    train_bound = int(images.shape[0] * 0.8)
    val_bound = int(images.shape[0] * 0.9)

    indices = np.arange(images.shape[0])
    np.random.shuffle(indices)

    train_indices = indices[:train_bound]
    train_images = images[train_indices]
    train_graphs = graphs[train_indices]
    train_y = y[train_indices]
    train_features = features[train_indices]

    val_indices = indices[train_bound:val_bound]
    val_images = images[val_indices]
    val_graphs = graphs[val_indices]
    val_y = y[val_indices]
    val_features = features[val_indices]

    test_indices = indices[val_bound:]
    test_images = images[test_indices]
    test_graphs = graphs[test_indices]
    test_y = y[test_indices]
    test_features = features[test_indices]

    return train_images, train_graphs, train_features, train_y, val_images, val_graphs, val_features, val_y, test_images, test_graphs, test_features, test_y


def read_image_data(chrom_names, data_dir, patch_size):
    total_cnt = 0
    for cn in chrom_names:
        _ = np.load(os.path.join(data_dir, 'imageset.{}.npy'.format(cn)))
        total_cnt += len(_)
    imageset = np.zeros((total_cnt, patch_size, patch_size), dtype='float32')
    labels = np.zeros((total_cnt, patch_size, patch_size), dtype='bool')

    current_start = 0
    for cn in chrom_names:
        current_image = np.load(os.path.join(data_dir, 'imageset.{}.npy'.format(cn)))
        current_y = np.load(os.path.join(data_dir, 'labels.{}.npy'.format(cn)))

        current_end = current_start + len(current_image)
        imageset[current_start:current_end, :, :] = current_image
        labels[current_start:current_end, :, :] = current_y
        current_start = current_end
    return np.log(imageset + 1), labels.astype('int')


def get_split_imageset(dataset_dir, image_size, chroms):
    x, y = read_image_data(chroms, dataset_dir, image_size)

    # get all the data indices
    indices = np.arange(x.shape[0])
    LOGGER.debug(f'indices: {indices}')

    # define the training data as 80% of the total data
    #            validation data as 10% of the total data
    #            test data as the remainder
    train_bound = int(x.shape[0] * 0.8)
    val_bound = int(x.shape[0] * 0.9)
    LOGGER.debug(f'train_bound: {train_bound}')
    LOGGER.debug(f'val_bound: {val_bound}')

    # shuffle the indices for randomness
    np.random.shuffle(indices)

    # split the data into train, test, and validation
    x_train, y_train = split_data(x, y, indices[:train_bound])
    x_val, y_val = split_data(x, y, indices[train_bound:val_bound])
    x_test, y_test = split_data(x, y, indices[val_bound:])

    print(x_train)

    # return the split data
    return x_train, y_train, x_val, y_val, x_test, y_test

def split_data(data, labels, selected_indices):
    # TODO: explain
    x = data[selected_indices]
    y = labels[selected_indices]
    return x, y

def read_graph_data(chrom_names, data_dir, patch_size):
    total_cnt = 0
    for cn in chrom_names:
        _ = np.load(os.path.join(data_dir, 'imageset.{}.npy'.format(cn)))
        total_cnt += len(_)
    graphset = np.zeros((total_cnt, 2 * patch_size, 2 * patch_size), dtype='float32')
    labels = np.zeros((total_cnt, patch_size, patch_size), dtype='bool')
    node_features = None

    current_start = 0
    for cn in chrom_names:
        current_graph = np.load(os.path.join(data_dir, 'graphset.{}.npy'.format(cn)))
        current_y = np.load(os.path.join(data_dir, 'labels.{}.npy'.format(cn)))
        kmer_features = np.load(os.path.join(data_dir, 'node_features.{}.npy'.format(cn)))
        motif_features = np.load(os.path.join(data_dir, 'motif_features.{}.npy'.format(cn)))
        current_features = np.concatenate((kmer_features, motif_features), axis=-1)

        if node_features is None:
            node_features = np.zeros((total_cnt, 2 * patch_size, current_features.shape[2]), dtype='float32')

        current_end = current_start + len(current_graph)
        graphset[current_start:current_end, :, :] = current_graph
        labels[current_start:current_end, :, :] = current_y
        node_features[current_start:current_end, :, :] = current_features
        current_start = current_end

    return np.log(graphset + 1), labels.astype('int'), node_features


def get_split_graphset(dataset_dir, patch_size, seed, chroms):
    graphs, y, features = read_graph_data(chroms, dataset_dir, patch_size)
    train_bound = int(graphs.shape[0] * 0.8)
    val_bound = int(graphs.shape[0] * 0.9)

    indices = np.arange(graphs.shape[0])
    np.random.shuffle(indices)

    train_indices = indices[:train_bound]
    train_graphs = graphs[train_indices]
    train_y = y[train_indices]
    train_features = features[train_indices]

    val_indices = indices[train_bound:val_bound]
    val_graphs = graphs[val_indices]
    val_y = y[val_indices]
    val_features = features[val_indices]

    test_indices = indices[val_bound:]
    test_graphs = graphs[test_indices]
    test_y = y[test_indices]
    test_features = features[test_indices]

    return train_graphs, train_features, train_y, val_graphs, val_features, val_y, test_graphs, test_features, test_y

def count_pos_neg_distributions(generator, patch_size):
    batch_counts = []
    total_pixels = patch_size * patch_size

    for _, y_batch in generator:
        pos_count = np.sum(y_batch)
        neg_count = total_pixels - pos_count
        batch_counts.append((pos_count, neg_count))

    return batch_counts

def estimate_upper_bound(generator, percentile=0.996, max_pixels=2000000):
    collected = []

    for batch_x, _ in generator:
        pixels = tf.reshape(batch_x, [-1]).numpy()
        collected.extend(pixels.tolist())

        if len(collected) >= max_pixels:
            break

    collected = np.array(collected[:max_pixels])
    return np.quantile(collected, percentile)

def create_hic_generators(chrom_names, data_dir, patch_size,
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
    train_gen = HiCDatasetGenerator(chrom_names=chrom_names,
                                    data_dir=data_dir,
                                    patch_size=patch_size,
                                    indices=train_idx,
                                    batch_size=batch_size,
                                    name='Train Generator',
                                    shuffle=True)
    val_gen = HiCDatasetGenerator(chrom_names=chrom_names,
                                  data_dir=data_dir,
                                  patch_size=patch_size,
                                  indices=val_idx,
                                  name='Validation Generator',
                                  batch_size=batch_size,
                                  shuffle=False)
    test_gen = HiCDatasetGenerator(chrom_names=chrom_names,
                                   data_dir=data_dir,
                                   patch_size=patch_size,
                                   indices=test_idx,
                                   batch_size=batch_size,
                                   name='Test Generator',
                                   shuffle=False)

    return train_gen, val_gen, test_gen

def visualize_data(generator: HiCDatasetGenerator, limit: int = None, plot_neg: bool = False):
    # Compute distributions
    train_counts = count_pos_neg_distributions(generator, PATCH_SIZE)

    # Plot distributions
    plot_pixel_counts(train_counts, generator.name, limit, plot_neg)