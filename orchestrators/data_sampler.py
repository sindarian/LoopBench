import logging
import os
from itertools import product

import sample_patches
from util.constants import PATCH_SIZES, RESOLUTIONS
from util.logger import Logger

LOGGER = Logger(name='data_sampler', level=logging.DEBUG).get_logger()

def sample_data(cell_line,
                assembly,
                bedpe_path,
                image_txt_dir,
                graph_txt_dir,
                chroms,
                patch_sizes: list = PATCH_SIZES,
                resolutions: list = RESOLUTIONS):
    for patch_size, resolution in list(product(patch_sizes, resolutions)):
        LOGGER.info(f'SAMPLING CELL LINE - {cell_line} w/ {assembly}')
        LOGGER.info(f'Sampling Parameters - patch size {patch_size} w/ resolution {resolution}')

        dataset_path = os.path.join(f'dataset', cell_line)

        sample_patches.run_sample_patches(dataset_path=dataset_path,
                                                 assembly=assembly,
                                                 bedpe_path=bedpe_path,
                                                 image_txt_dir=image_txt_dir,
                                                 graph_txt_dir=graph_txt_dir,
                                                 chroms=chroms,
                                                 patch_size=patch_size,
                                                 resolution=resolution,
                                                 sampled_data_dir=None)
        # run_generate_node_features(dataset_path, chroms, assembly)