import logging
import time
from typing import Tuple, Literal

from generators.chromosome_generator import ChromosomeGenerator
from generators.chromosome_loader import ChromosomeLoader
from generators.chromosome_processor import ChromosomeProcessor
from gutils import parsebed
from model.cnn import UNet
from model.loop_net import LoopNet
from model.chromosome_modeller import ChromosomeModeller
from util.plotting import plotting
from util.constants import PATCH_SIZE, RESOLUTION
from util.logger import Logger

from util.utils import print_metrics_table

LOGGER = Logger(name='demo_from_processed', level=logging.DEBUG).get_logger()

def sample_data(
    train_data: dict,
    test_data: dict | None,
    use_original: bool = True,
    threshold: float = 0.95,
    normalization: str = "log,zscore",
    chroms: List[str] = None,
) -> None:
    """
    Processes and samples Hi-C contact data for training and optionally a separate test
    cell line. Supports both the original patch-based pipeline and the newer
    chromosome-level processing pipeline with quantile thresholding and normalization.

    Args:
        train_data (dict): Training dataset config with keys: 'cell_line', 'assembly',
                           'bedfile', 'contacts', 'output_dir'.
        test_data (dict | None): Test dataset config with the same keys as train_data.
                                 If None, only the training data is processed.
        use_original (bool): If True, uses the original patch-based processing pipeline.
                             If False, uses chromosome-level processing with threshold
                             and normalization. Defaults to True.
        threshold (float): Quantile threshold for filtering samples during chromosome-level
                           processing. Only used when use_original is False. Defaults to 0.95.
        normalization (str): Normalization pipeline to apply during chromosome-level processing
                             (e.g. 'log,zscore'). Only used when use_original is False.
                             Defaults to 'log,zscore'.
        chroms (List[str]): Chromosomes to process. Defaults to ['1'].

    Returns:
        None
    """
    if chroms is None:
        chroms = ['1']

    train_data_processor = ChromosomeProcessor(
        chromosome_list=chroms,
        bedpe_dict=parsebed(train_data['bedfile'], valid_threshold=1),
        contact_data_dir=train_data['contacts'],
        genome_assembly=train_data['assembly'],
        output_dir=train_data['output_dir'],
        patch_size=PATCH_SIZE,
        resolution=RESOLUTION,
        plot_chrom=False,
        use_giloop=True
    )

    start = time.time()
    if use_original:
        train_data_processor.process_all_chromosomes_as_patches()
    else:
        train_data_processor.process_all_chromosomes_as_chromosomes(threshold, normalization)

    end = time.time()
    LOGGER.info(
        f'Time taken to sample the entire train dataset: {end - start} seconds, {(end - start)/60} minutes; original method used: {use_original}')

    if test_data is not None:
        test_data_processor = ChromosomeProcessor(
            chromosome_list=chroms,
            bedpe_dict=parsebed(test_data['bedfile'], valid_threshold=1),
            contact_data_dir=test_data['contacts'],
            genome_assembly=test_data['assembly'],
            output_dir=test_data['output_dir'],
            patch_size=PATCH_SIZE,
            resolution=RESOLUTION,
            plot_chrom=False,
            use_giloop=True,
            is_test=True
        )

        start = time.time()
        if use_original:
            test_data_processor.process_all_chromosomes_as_patches()
        else:
            test_data_processor.process_all_chromosomes_as_chromosomes(threshold, normalization)

        end = time.time()
        LOGGER.info(
            f'Time taken to sample the entire test dataset: {end - start} seconds, {(end - start) / 60} minutes; original method used: {use_original}')

def load_data(
    batch: int = 8,
    use_original: bool = True,
    upsample: bool = False,
    factor: int = 3,
    threshold: int = 10,
    strategy: Literal["balanced", "random"] | None = "balanced",
    train_data_dir: str = 'dataset/hela_100',
    test_data_dir: str | None = 'dataset/gm12878_50',
    norm_method: str = 'all',
    chroms: List[str] = None,
) -> Tuple[ChromosomeGenerator, ChromosomeGenerator, ChromosomeGenerator]:
    """
    Initializes ChromosomeLoaders for training and optionally a separate test cell line,
    then creates and returns train/val/test generators.

    When test_data_dir is provided, the train loader uses a 70/30 train/val split and
    the test generator is sourced entirely from the separate test dataset. Otherwise,
    a 70/20/10 train/val/test split is applied to the training data.

    Args:
        batch (int): Batch size for data loading. Defaults to 8.
        use_original (bool): If True, uses the original dataset format. Defaults to True.
        upsample (bool): If True, upsamples underrepresented classes in training data. Defaults to False.
        factor (int): Upsampling factor applied when upsample is True. Defaults to 3.
        threshold (int): Minimum sample count threshold per class for upsampling. Defaults to 10.
        strategy (Literal["balanced", "random"] | None): Upsampling strategy to use. Defaults to "balanced".
        train_data_dir (str): Path to the training dataset directory. Defaults to 'dataset/hela_100'.
        test_data_dir (str | None): Path to a separate test dataset directory. If None, test data
                                    is sourced from the training split. Defaults to 'dataset/gm12878_50'.
        norm_method (str): Normalization method passed to the data loader. Defaults to 'all'.
        chroms (List[str]): Chromosomes to load. Defaults to ['1'].

    Returns:
        Tuple[ChromosomeGenerator, ChromosomeGenerator, ChromosomeGenerator]:
            Train, validation, and test generators.
    """
    if chroms is None:
        chroms = ['1']

    # Load training data — use 70/30 split if a separate test set is provided, else 70/20/10
    train_loader = ChromosomeLoader(chromosomes=chroms,
                              data_dir=train_data_dir,
                              patch_size=PATCH_SIZE,
                              batch_size=batch,
                              split_ratios=(0.7, 0.2, 0.1) if not test_data_dir else (0.7, 0.3),
                              # split_ratios=(1.0,),
                              use_original=use_original,
                              is_train=True,
                              # is_train=False,
                              norm_method=norm_method)

    train_gen, val_gen, test_gen = train_loader.create_chromosome_generators(upsample=upsample,
                                                                       factor=factor,
                                                                       threshold=threshold,
                                                                       strategy=strategy)

    if test_data_dir:
        # Load the separate test cell line dataset using the full dataset as the test split
        test_loader = ChromosomeLoader(chromosomes=chroms,
                                        data_dir=test_data_dir,
                                        patch_size=PATCH_SIZE,
                                        batch_size=batch,
                                        split_ratios=(1.0,),
                                        use_original=use_original,
                                        is_train=False,
                                        norm_method=norm_method,
                                        shuffle_train=False)
        test_gen, _, _ = test_loader.create_chromosome_generators()
        
    return train_gen, val_gen, test_gen


def run_original(
    train_data: dict,
    test_data: dict | None,
    batch: int = 8,
    split_ratios: Tuple[float, float, float] = (0.7, 0.2, 0.1),
    epochs: int = 30,
    resample: bool = False,
    num_runs: int = 5,
    drop_worst: int = 2,
    patch_size: int = PATCH_SIZE,
    resolution: int = RESOLUTION,
    train: bool = True,
    note: str = "",
    chroms: List[str] = None,
) -> None:
    """
    Trains and evaluates the original GILoop UNet model using the original preprocessing
    pipeline. Optionally resamples data before training. Results are averaged across
    multiple runs with the worst runs optionally dropped.

    Args:
        train_data (dict): Training dataset config with keys: 'cell_line', 'assembly',
                           'bedfile', 'contacts', 'output_dir'.
        test_data (dict | None): Test dataset config with the same keys as train_data.
                                 If None, the training data is used for testing.
        batch (int): Batch size used during training. Defaults to 8.
        split_ratios (Tuple[float, float, float]): Train/val/test split ratios. Defaults to (0.7, 0.2, 0.1).
        epochs (int): Number of training epochs per run. Defaults to 30.
        resample (bool): If True, resamples and saves data before training. Defaults to False.
        num_runs (int): Number of times to train the model; results are averaged. Defaults to 5.
        drop_worst (int): Number of worst runs to drop before averaging metrics. Defaults to 2.
        patch_size (int): Patch size used for model input. Defaults to PATCH_SIZE.
        resolution (int): Hi-C resolution used for data loading. Defaults to RESOLUTION.
        train (bool): If True, trains and evaluates the model. Defaults to True.
        note (str): Optional note appended to the model name for logging/identification. Defaults to "".
        chroms (List[str]): Chromosomes to sample and train on. Defaults to ['1'].

    Returns:
        None
    """
    if chroms is None:
        chroms = ['1']

    use_original = True

    if resample:
        sample_data(train_data, test_data, use_original, chroms=chroms)

    if train:
        train_gen, val_gen, test_gen = load_data(batch,
                                                 train_data_dir=train_data['output_dir'],
                                                 test_data_dir=test_data['output_dir'] if test_data is not None else None,
                                                 chroms=chroms)

        giloop_modeller = ChromosomeModeller(model=UNet(patch_size=PATCH_SIZE, note=note),
                                             train_generator=train_gen,
                                             val_generator=val_gen,
                                             test_generator=test_gen,
                                             epochs=epochs)

        avg_train_metrics, avg_val_metrics, avg_test_metrics = giloop_modeller.run_n_times(num_runs, drop_worst,
                                                                                         train_original=use_original)
        LOGGER.info('Metrics Table for Original GILoop')
        print_metrics_table(avg_train_metrics, avg_val_metrics, avg_test_metrics, ["0.996", "original"],
                            patch_size, resolution)

def run_new_models(
    train_data: dict,
    test_data: dict | None,
    batch: int = 8,
    split_ratios: Tuple[float, float, float] = (0.7, 0.2, 0.1),
    threshold: float = 0.96,
    normalization: str = "log,zscore",
    epochs: int = 30,
    resample: bool = False,
    num_runs: int = 5,
    drop_worst: int = 2,
    run_Unet: bool = True,
    run_LoopNet: bool = True,
    patch_size: int = 64,
    resolution: int = 10000,
    norm_method: str = 'all',
    note: str = "",
    chroms: List[str] = None,
) -> None:
    """
    Trains and evaluates the EnhancedUNet and/or LoopNet models using the specified
    data, normalization, and training configuration. Optionally resamples data before training.
    Results are averaged across multiple runs with the worst run(s) optionally dropped.

    Args:
        train_data (dict): Training dataset config with keys: 'cell_line', 'assembly',
                           'bedfile', 'contacts', 'output_dir'.
        test_data (dict | None): Test dataset config with the same keys as train_data.
                                 If None, the training data is used for testing.
        batch (int): Batch size used during training. Defaults to 8.
        split_ratios (Tuple[float, float, float]): Train/val/test split ratios. Defaults to (0.7, 0.2, 0.1).
        threshold (float): Quantile threshold used for data sampling. Defaults to 0.96.
        normalization (str): Normalization pipeline to apply (e.g. 'log,zscore'). Defaults to 'log,zscore'.
        epochs (int): Number of training epochs per run. Defaults to 30.
        resample (bool): If True, resamples and saves data before training. Defaults to False.
        num_runs (int): Number of times to train the model; results are averaged. Defaults to 5.
        drop_worst (int): Number of worst runs to drop before averaging metrics. Defaults to 2.
        run_Unet (bool): If True, trains and evaluates the EnhancedUNet model. Defaults to True.
        run_LoopNet (bool): If True, trains and evaluates the LoopNet model. Defaults to True.
        patch_size (int): Patch size used for model input. Defaults to 64.
        resolution (int): Hi-C resolution used for data loading. Defaults to 10000.
        norm_method (str): Normalization method passed to the data loader. Defaults to 'all'.
        note (str): Optional note appended to the model name for logging/identification. Defaults to "".
        chroms (List[str]): Chromosomes to sample and train on. Defaults to ['1'].

    Returns:
        None
    """
    if chroms is None:
        chroms = ['1']

    use_original = False

    if resample:
        sample_data(train_data, test_data, use_original, threshold, normalization, chroms=chroms)

    if run_Unet or run_LoopNet:
        # Load train/val/test generators with random upsampling (factor=3, threshold=10)
        train_gen, val_gen, test_gen = load_data(batch, use_original, True, 3,
                                                 10, "random",
                                                 train_data_dir=train_data['output_dir'],
                                                 test_data_dir=test_data['output_dir'] if test_data else None,
                                                 norm_method=norm_method,
                                                 chroms=chroms)
        split_scenario_name = [threshold, normalization]

        if run_Unet:
            unet_modeller = ChromosomeModeller(model=UNet(model_name="UNet",
                                                          save_as="unet",
                                                          patch_size=PATCH_SIZE,
                                                          note="UNet: " + note),
                                               train_generator=train_gen,
                                               val_generator=val_gen,
                                               test_generator=test_gen,
                                               epochs=epochs)
            avg_train_metrics, avg_val_metrics, avg_test_metrics = unet_modeller.run_n_times(num_runs, drop_worst, train_original=use_original)
            LOGGER.info('Metrics Table for Enhanced GILoop')
            print_metrics_table(avg_train_metrics, avg_val_metrics, avg_test_metrics, split_scenario_name, patch_size, resolution)

        if run_LoopNet:
            loopnet_modeller = ChromosomeModeller(model=LoopNet(patch_size=PATCH_SIZE, note="LoopNet: " + note),
                                                  train_generator=train_gen,
                                                  val_generator=val_gen,
                                                  test_generator=test_gen,
                                                  epochs=epochs)

            avg_train_metrics, avg_val_metrics, avg_test_metrics = loopnet_modeller.run_n_times(num_runs, drop_worst, train_original=use_original)
            LOGGER.info('Metrics Table for LoopNet')
            print_metrics_table(avg_train_metrics, avg_val_metrics, avg_test_metrics, split_scenario_name, patch_size, resolution)

if __name__ == '__main__':
    # All valid chromosomes for HeLa-S3 — Chr18 is excluded as it is absent in the Hi-C file
    chroms = \
        [str(i) for i in range(1, 18)] + \
        [str(i) for i in range(19, 23)] + ['X'] # Chr18 of HeLa-S3 is absent in the Hi-C file

    # Training dataset: HeLa-S3 cell line mapped to hg38
    train_data = \
        {
            'cell_line': 'hela_100',
            'assembly': 'hg38',
            'bedfile': 'bedpe/hela.hg38.bedpe',
            'contacts': 'data/txt_hela_100',
            'output_dir': 'dataset/hela_100'
        }

    # Test dataset: GM12878 cell line mapped to hg19 for cross-cell-line generalization evaluation
    test_data = \
        {
            'cell_line': 'gm12878_50',
            'assembly': 'hg19',
            'bedfile': 'bedpe/gm12878.tang.ctcf-chiapet.hg19.bedpe',
            'contacts': 'data/txt_gm12878_50',
            'output_dir': 'dataset/gm12878_50'
        }

    # Evaluate original GILoop, EnhancedUNet, and LoopNet on HeLa only (train and test on same cell line)
    run_original(train_data, None, resample=True, train=True, epochs=30, num_runs=3, drop_worst=1,
                 note="GILoop: HeLa ROC and PR Curves", chroms=chroms)
    run_new_models(train_data, None, resample=True, epochs=30, num_runs=3, drop_worst=1,
                   run_Unet=True, run_LoopNet=True, note="HeLa ROC and PR Curves", chroms=chroms)

    # Evaluate original GILoop, EnhancedUNet, and LoopNet trained on HeLa and tested on GM12878
    run_original(train_data, test_data, resample=False, train=True, epochs=30, num_runs=3, drop_worst=1,
                 note="GILoop: HeLa + GM ROC and PR Curves", chroms=chroms)
    run_new_models(train_data, test_data, resample=False, epochs=30, num_runs=3, drop_worst=1,
                   run_Unet=True, run_LoopNet=True, note="HeLa + GM ROC and PR Curves", chroms=chroms)

    # Plot aggregated training metrics across all runs
    plotting.plot_training_history(title='Training Metrics Comparison')