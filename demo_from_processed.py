import os

from util.plotting import plotting
from util.constants import SAMPLED_DATA_DIR
from sample_patches import run_sample_patches
from train import train_run_cnn, train_loop_net
from util.logger import Logger
import logging

from util.utils import PATCH_SIZE, create_hic_generators, estimate_upper_bound, visualize_data

LOGGER = Logger(name='demo_from_processed', level=logging.DEBUG).get_logger()


if __name__ == '__main__':
    ################################################################################
    ##  Define the macros. Change the variables below according to what you need  ##
    ################################################################################

    # Define the unique ID for this run of experiment
    # (i.e. the unique name of the model trained in this experiment)
    run_id = 'demo'

    # Specify the genome assemblies (reference genome) of the source and target Hi-C/ChIA-PET
    # In this demo, source and target cell lines are sequenced with different assembly genomes
    source_assembly = 'hg19'
    target_assembly = 'hg38'

    # The path to the ChIA-PET annotation files
    source_bedpe_path = 'bedpe/gm12878.tang.ctcf-chiapet.hg19.bedpe'

    # In the real-world scenarios, the target cell line typically does not have ChIA-PET labels
    # In that case, please specify an empty target .bedpe file as a placeholder
    # Comment or uncomment the following two lines as needed
    target_bedpe_path = 'bedpe/hela.hg38.bedpe'
    # target_bedpe_path = 'bedpe/placeholder.bedpe'  # Uncomment this in the case where target label is unavailable

    mode = 'test'
    if 'placeholder' in target_bedpe_path:
        mode = 'realworld'

    # Specify the directory that contains the images and graphs of the source cell line
    # You can use a mixed downsampling rate, but here we use an identical sequencing depth
    # for both graph and image
    source_image_data_dir = 'data/txt_gm12878_50'
    source_graph_data_dir = 'data/txt_gm12878_50'

    # Specify the data dir for target cell line
    target_image_data_dir = 'data/txt_hela_100'
    target_graph_data_dir = 'data/txt_hela_100'

    # Name the sampled datasets with unique identifiers you like
    source_dataset_name = 'gm12878_50'
    target_dataset_name = 'hela_100'

    # Define the chromosomes we draw training data from
    source_chroms = [str(i) for i in range(1, 23)] + ['X']

    # Define the chromosomes of the target cell line we want to predict on
    target_chroms = \
        [str(i) for i in range(1, 18)] + \
        [str(i) for i in range(19, 23)] + ['X'] # Chr18 of HeLa-S3 is absent in the Hi-C file

    # Set the threshold cutting off the probability map to generate the final annotations
    threshold = 0.48

    # Set the path to the output file, where saves the annotations
    output_path = 'predictions/demo.bedpe'


    ##############################################################################
    ###               The GILoop core algorithm starts from here               ###
    ##############################################################################

    resample_data = False # resample usually when you adjust the PATCH_SIZE or RESOLUTION

    if resample_data:
        LOGGER.info('SAMPLING SOURCE CELL LINE - GM12878 w/ hg19')
        # Sample patches for source cell line
        run_sample_patches(dataset_name=source_dataset_name,
                           assembly=source_assembly,
                           bedpe_path=source_bedpe_path,
                           image_txt_dir=source_image_data_dir,
                           graph_txt_dir=source_graph_data_dir,
                           chroms=source_chroms)

        LOGGER.info('Sampling the target cell line - HeLa100 w/ hg38')
        # Sample patches for target cell line
        run_sample_patches(dataset_name=target_dataset_name,
                           assembly=target_assembly,
                           bedpe_path=target_bedpe_path,
                           image_txt_dir=target_image_data_dir,
                           graph_txt_dir=target_graph_data_dir,
                           chroms=target_chroms)

    LOGGER.info('Create Generators for Train, Val, and Test Data')
    train_gen, val_gen, test_gen = create_hic_generators(
        chrom_names=source_chroms,
        data_dir=os.path.join(SAMPLED_DATA_DIR, source_dataset_name),
        patch_size=PATCH_SIZE,
        split_ratios=(0.8, 0.1, 0.1), # train, val, test
        batch_size=8
    )

    # visualize the training data distribution
    visualize_data(train_gen)

    # estimate the upper bound of the training data
    # this is used to scale the data and clip outliers when the model is built
    x_train_upper_bound = estimate_upper_bound(train_gen, percentile=0.996)
    print("99.6% upper bound:", x_train_upper_bound)

    # build, train, and run a CNN and LoopNet for loop calling
    train_run_cnn(train_gen=train_gen,
                  val_gen=val_gen,
                  test_gen=test_gen,
                  clip_value=x_train_upper_bound,
                  epochs=2)
    train_loop_net(train_gen=train_gen,
                   val_gen=val_gen,
                   test_gen=test_gen,
                   clip_value=x_train_upper_bound,
                   epochs=2)

    # LOGGER.info('Testing the U-net models')
    # Predict on the target cell line
    # run_output_predictions(
    #     run_id,
    #     'Finetune',
    #     threshold,
    #     target_dataset_name,
    #     target_assembly,
    #     target_chroms,
    #     output_path,
    #     mode
    # )

    plotting.plot_training_history()