import os

from generators.chromosome_loader import ChromosomeLoader
from generators.chromosome_processor import ChromosomeProcessor
from gutils import parsebed
from model.cnn import UNet
from model.loop_net import LoopNet
from model.chromosome_modeller import ChromosomeModeller
from orchestrators.modelling import  test_models
from util.plotting import plotting
from util.constants import OUTPUT_DIR, MODELS_DIR, PATCH_SIZE, RESOLUTION
from util.logger import Logger
import logging

LOGGER = Logger(name='demo_from_processed', level=logging.DEBUG).get_logger()

def sample_data(use_original: bool = True, threshold: float = 0.95, normalization: str = "clip"):
    # Create processor
    chrom_processor = ChromosomeProcessor(
        chromosome_list=target_chroms,
        bedpe_dict=parsebed(target_bedpe_path, valid_threshold=1),
        contact_data_dir=target_image_data_dir,
        genome_assembly=target_assembly,
        output_dir='dataset/hela_100',
        patch_size=PATCH_SIZE,
        resolution=RESOLUTION,
        plot_chrom=False,
        use_giloop=True
    )

    if use_original:
        LOGGER.info("Sampling chromosomes with original sampling logic")
        chrom_processor.process_all_chromosomes_as_patches()
    else:
        LOGGER.info(f'Sampling chromosomes with new sampling logic using threshold: {threshold}'
                    f' and normalization: {normalization}')
        chrom_processor.process_all_chromosomes_as_chromosomes(threshold, normalization)

def load_data(batch: int = 8, split_ratios: tuple = (0.7, 0.2, 0.1), use_original: bool = True, upsample: bool = False,
              factor: int = 3, threshold: int = 10, strategy: str = "balanced"):
    # load the data into generators
    loader = ChromosomeLoader(chromosomes=target_chroms,
                              data_dir='dataset/hela_100',
                              patch_size=PATCH_SIZE,
                              batch_size=batch,
                              split_ratios=split_ratios,
                              use_original=use_original)

    train_gen, val_gen, test_gen = loader.create_chromosome_generators(upsample=upsample,
                                                                       factor=factor,
                                                                       threshold=threshold,
                                                                       strategy=strategy)

    return train_gen, val_gen, test_gen


def run_original(batch: int = 8, split_ratios: tuple = (0.7, 0.2, 0.1), use_original: bool = True, epochs: int = 30,
                 resample: bool = False):
    if resample:
        sample_data(use_original)

    # # load the data into generators
    # loader = ChromosomeLoader(chromosomes=target_chroms,
    #                           data_dir='dataset/hela_100',
    #                           patch_size=PATCH_SIZE,
    #                           batch_size=batch,
    #                           split_ratios=split_ratios,
    #                           use_original=use_original)
    #
    # train_gen, val_gen, test_gen = loader.create_chromosome_generators()

    train_gen, val_gen, test_gen = load_data(batch, split_ratios)

    giloop_modeller = ChromosomeModeller(model=UNet(patch_size=PATCH_SIZE),
                                         train_generator=train_gen,
                                         val_generator=val_gen,
                                         test_generator=test_gen,
                                         epochs=epochs)
    # giloop_modeller.run(use_original)
    giloop_modeller.run_n_times(num_runs=5, drop_worst=2, train_original=True)

def run_new_models(batch: int = 8, split_ratios: tuple = (0.7, 0.2, 0.1), use_original: bool = False,
                   threshold: float = 0.95, normalization: str = "clip", epochs: int = 30, resample: bool = False):
    if resample:
        sample_data(use_original, threshold, normalization)

    # # load the data into generators
    # loader = ChromosomeLoader(chromosomes=target_chroms,
    #                           data_dir='dataset/hela_100',
    #                           patch_size=PATCH_SIZE,
    #                           batch_size=batch,
    #                           split_ratios=split_ratios,
    #                           use_original=use_original)
    #
    # train_gen, val_gen, test_gen = loader.create_chromosome_generators(upsample=True,
    #                                                                    factor=3,
    #                                                                    threshold=10,
    #                                                                    strategy="balanced")

    train_gen, val_gen, test_gen = load_data(batch, split_ratios, use_original, True, 3, 10, "balanced")

    unet_modeller = ChromosomeModeller(model=UNet(model_name="UNet",
                                                  save_as="unet",
                                                  patch_size=PATCH_SIZE),
                                       train_generator=train_gen,
                                       val_generator=val_gen,
                                       test_generator=test_gen,
                                       epochs=epochs)
    loopnet_modeller = ChromosomeModeller(model=LoopNet(patch_size=PATCH_SIZE),
                                          train_generator=train_gen,
                                          val_generator=val_gen,
                                          test_generator=test_gen,
                                          epochs=epochs)
    # unet_modeller.run(use_original)
    unet_modeller.run_n_times(num_runs=5, drop_worst=2, train_original=use_original)
    # loopnet_modeller.run(use_original)
    loopnet_modeller.run_n_times(num_runs=5, drop_worst=2, train_original=use_original)

if __name__ == '__main__':
    # Define the unique ID for this run of experiment
    # (i.e. the unique name of the model trained in this experiment)
    run_id = 'demo'

    # Specify the genome assemblies (reference genome) of the source and target Hi-C/ChIA-PET
    # In this demo, source and target cell lines are sequenced with different assembly genomes
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
    target_image_data_dir = 'data/txt_hela_100'
    target_graph_data_dir = 'data/txt_hela_100'

    target_dataset_name = 'hela_100'

    # Define the chromosomes of the target cell line we want to predict on
    target_chroms = \
        [str(i) for i in range(1, 18)] + \
        [str(i) for i in range(19, 23)] + ['X'] # Chr18 of HeLa-S3 is absent in the Hi-C file

    # Set the threshold cutting off the probability map to generate the final annotations
    threshold = 0.48

    # Set the path to the output file, where saves the annotations
    output_path = 'predictions/demo.bedpe'

    # # Create processor
    # chrom_processor = ChromosomeProcessor(
    #     chromosome_list=target_chroms,
    #     bedpe_dict=parsebed(target_bedpe_path, valid_threshold=1),
    #     contact_data_dir=target_image_data_dir,
    #     genome_assembly=target_assembly,
    #     output_dir='dataset/hela_100',
    #     patch_size=PATCH_SIZE,
    #     resolution=RESOLUTION,
    #     plot_chrom=False,
    #     use_giloop=True
    # )
    #
    # if use_original:
    #     chrom_processor.process_all_chromosomes_as_patches()
    # else:
    #     chrom_processor.process_all_chromosomes_as_chromosomes(threshold=0.95, normalization="clip")
    # # load the data into generators
    # loader = ChromosomeLoader(chromosomes=target_chroms,
    #                           data_dir='dataset/hela_100',
    #                           patch_size=PATCH_SIZE,
    #                           batch_size=8,
    #                           split_ratios=(0.7, 0.2, 0.1),
    #                           use_original=use_original)
    # train_gen, val_gen, test_gen = loader.create_chromosome_generators(upsample=True,
    #                                                                    factor=3,
    #                                                                    threshold=10,
    #                                                                    strategy="balanced")
    #
    # # plot the data
    # # plotting.plot_chromosome_labels(target_chroms, use_original=use_original)
    # # data_to_plot, _, _ = loader.create_chromosome_generators(upsample=False)
    # # loader.visualize_data(data_to_plot.copy(1))
    #
    # unet_modeller = ChromosomeModeller(model=UNet(patch_size=PATCH_SIZE),
    #                               train_generator=train_gen,
    #                               val_generator=val_gen,
    #                               test_generator=test_gen,
    #                               epochs=10)
    # loopnet_modeller = ChromosomeModeller(model=LoopNet(patch_size=PATCH_SIZE),
    #                                    train_generator=train_gen,
    #                                    val_generator=val_gen,
    #                                    test_generator=test_gen,
    #                                    epochs=10)
    # unet_modeller.run(use_original)
    # loopnet_modeller.run(use_original)

    # plot the training history

    run_original(resample=False, epochs=1)
    run_new_models(resample=False, epochs=1)
    plotting.plot_training_history(title='Training Metrics Comparison')

    # if TEST:
        # test_models(cnn_path=os.path.join(OUTPUT_DIR, MODELS_DIR, 'cnn_ps224_res10000.h5'),
        #             loopnet_path=os.path.join(OUTPUT_DIR, MODELS_DIR, 'loopnet_ps224_res10000.h5'),
        #             cell_line=target_dataset_name,
        #             assembly=target_assembly,
        #             chroms=target_chroms)