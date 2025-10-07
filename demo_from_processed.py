from typing import Tuple, Literal

from generators.chromosome_generator import ChromosomeGenerator
from generators.chromosome_loader import ChromosomeLoader
from generators.chromosome_processor import ChromosomeProcessor
from gutils import parsebed
from model.cnn import UNet
from model.loop_net import LoopNet
from model.chromosome_modeller import ChromosomeModeller
from util.plotting import plotting
from util.constants import OUTPUT_DIR, MODELS_DIR, PATCH_SIZE, RESOLUTION
from util.logger import Logger
import logging

from util.utils import print_metrics_table

LOGGER = Logger(name='demo_from_processed', level=logging.DEBUG).get_logger()


def sample_data(
    use_original: bool = True,
    threshold: float = 0.95,
    normalization: Literal["log2", "clip", "divide"] = "clip"
) -> None:
    """
    Sample and optionally normalize the chromosome dataset.

    Args:
        use_original (bool): Whether to use the original dataset format.
        threshold (float): Confidence threshold for filtering samples.
        normalization (str): Method used for normalization.

    Returns:
        None
    """
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

def load_data(
    batch: int = 8,
    split_ratios: Tuple[float, float, float] = (0.7, 0.2, 0.1),
    use_original: bool = True,
    upsample: bool = False,
    factor: int = 3,
    threshold: int = 10,
    strategy: Literal["balanced", "random", None] = "balanced"
) -> (ChromosomeGenerator, ChromosomeGenerator, ChromosomeGenerator):
    """
    Load and prepare the sampled dataset for training and evaluation.

    Args:
        batch (int): Batch size for data loading.
        split_ratios (Tuple[float, float, float]): Ratios for train, validation, and test sets.
        use_original (bool): Whether to use the original dataset format.
        upsample (bool): Whether to upsample underrepresented classes.
        factor (int): Upsampling factor (if upsample is True).
        threshold (int): Minimum sample count threshold per class.
        strategy (str): Sampling strategy to use. Options: "balanced", "random", None".

    Returns:
        train generator, validation generator, and test generator.
    """
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


def run_original(
    batch: int = 8,
    split_ratios: Tuple[float, float, float] = (0.7, 0.2, 0.1),
    epochs: int = 30,
    resample: bool = False,
    num_runs: int = 5,
    drop_worst: int = 2
) -> None:
    """
    Run training using the original model setup.

    Args:
        batch (int): Batch size used during training.
        split_ratios (Tuple[float, float, float]): Ratios for train, validation, and test splits.
        epochs (int): Number of training epochs.
        resample (bool): Whether to apply data resampling.
        num_runs (int): Number of times to run the model
        drop_worst (int): Number of lowest performing runs to drop the average model performance calculation

    Returns:
        None
    """
    use_original = True
    if resample:
        sample_data(use_original)

    train_gen, val_gen, test_gen = load_data(batch, split_ratios)

    giloop_modeller = ChromosomeModeller(model=UNet(patch_size=PATCH_SIZE),
                                         train_generator=train_gen,
                                         val_generator=val_gen,
                                         test_generator=test_gen,
                                         epochs=epochs)
    avg_metrics, best_metrics = giloop_modeller.run_n_times(num_runs, drop_worst, train_original=use_original)
    print('Metrics Table for Original GILoop')
    print_metrics_table(avg_metrics, best_metrics, None)


def run_new_models(
        batch: int = 8,
        split_ratios: Tuple[float, float, float] = (0.7, 0.2, 0.1),
        threshold: float = 0.95,
        normalization: Literal["log2", "clip", "divide"] = "clip",
        epochs: int = 30,
        resample: bool = False,
        num_runs: int = 5,
        drop_worst: int = 2,
        run_Unet: bool = True,
        run_LoopNet: bool = True
) -> None:
    """
    Run training on new models with specified parameters.

    Args:
        batch (int): Batch size used during training.
        split_ratios (tuple): Train, validation, and test split ratios.
        threshold (float): Confidence threshold for predictions.
        normalization (str): Method used to normalize data.
        epochs (int): Number of training epochs.
        resample (bool): Whether to apply data resampling.
        num_runs (int): Number of times to run the model
        drop_worst (int): Number of lowest performing runs to drop the average model performance calculation

    Returns:
        None
    """
    use_original = False
    if resample:
        sample_data(use_original, threshold, normalization)

    train_gen, val_gen, test_gen = load_data(batch, split_ratios, use_original, True, 3, 10, "random")

    if run_Unet:
        unet_modeller = ChromosomeModeller(model=UNet(model_name="UNet",
                                                      save_as="unet",
                                                      patch_size=PATCH_SIZE),
                                           train_generator=train_gen,
                                           val_generator=val_gen,
                                           test_generator=test_gen,
                                           epochs=epochs)
        avg_train_metrics, avg_val_metrics, avg_test_metrics = unet_modeller.run_n_times(num_runs, drop_worst, train_original=use_original)
        print('Metrics Table for Enhanced GILoop')
        print_metrics_table(avg_train_metrics, avg_val_metrics, avg_test_metrics, [threshold, normalization])

    if run_LoopNet:
        loopnet_modeller = ChromosomeModeller(model=LoopNet(patch_size=PATCH_SIZE),
                                              train_generator=train_gen,
                                              val_generator=val_gen,
                                              test_generator=test_gen,
                                              epochs=epochs)

        avg_train_metrics, avg_val_metrics, avg_test_metrics = loopnet_modeller.run_n_times(num_runs, drop_worst, train_original=use_original)
        print('Metrics Table for LoopNet')
        print_metrics_table(avg_train_metrics, avg_val_metrics, avg_test_metrics, [threshold, normalization])

if __name__ == '__main__':
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
    # target_chroms = ['1']

    # Set the threshold cutting off the probability map to generate the final annotations
    # threshold = 0.48

    # Set the path to the output file, where saves the annotations
    output_path = 'predictions/demo.bedpe'

    # run_original(resample=False, epochs=30, num_runs=4, drop_worst=1)
    run_new_models(resample=False, epochs=30, num_runs=4, drop_worst=1, run_Unet=False)
    plotting.plot_training_history(title='Training Metrics Comparison')