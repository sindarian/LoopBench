from itertools import product

from generators.chromosome_loader import ChromosomeLoader
from generators.chromosome_processor import ChromosomeProcessor
from gutils import parsebed
from model.cnn import UNet
from model.chromosome_modeller import ChromosomeModeller
from model.identity_map import IDMap
from util.constants import PATCH_SIZE, RESOLUTION
from util.logger import Logger
import logging

from util.utils import print_metrics_table

LOGGER = Logger(name='experiment_runner', level=logging.DEBUG).get_logger()


def init_experiment_thresholds(
    lower_bound: int | float,
    upper_bound: int | float,
    increment: int | float,
    existing_thresholds: List[float] | None,
) -> List[float]:
    """
    Generates a list of threshold values between lower_bound and upper_bound,
    spaced by increment, scaled to a 0-1 range (divided by 100).
    Optionally prepends any existing thresholds to the result.

    Args:
        lower_bound (int | float): Starting value for threshold generation (e.g. 10 for 0.10).
        upper_bound (int | float): Ending value (exclusive) for threshold generation (e.g. 90 for 0.90).
        increment (int | float): Step size between each threshold (e.g. 10 for 0.10 steps).
        existing_thresholds (List[float] | None): Optional list of pre-existing thresholds to prepend.

    Returns:
        List[float]: A list of threshold values scaled between 0 and 1.
    """
    # Build evenly-spaced thresholds from lower_bound to upper_bound (exclusive),
    # stepping by increment, then scale each value to a 0–1 range by dividing by 100
    thresholds = [(lower_bound + (increment * inc_lower_by)) / 100 for inc_lower_by in
                  range(int((upper_bound - lower_bound) / increment))]

    # If existing thresholds are provided, prepend them to the newly generated list
    if existing_threshods:
        thresholds = existing_threshods + thresholds

    return thresholds

def sample_data(
    thresholds: List[float],
    norms: List[str],
    patch_size: int = PATCH_SIZE,
    resolution: int = RESOLUTION,
    chroms: List[str] = None,
) -> None:
    """
    Initializes a ChromosomeProcessor and runs sampling experiments for all combinations
    of the given quantile thresholds and normalization strategies.

    Args:
        thresholds (List[float]): Quantile thresholds to apply during sampling.
        norms (List[str]): Normalization strategies to apply (e.g. ['log,zscore', 'clip']).
        patch_size (int): Patch size to use when extracting Hi-C contact patches. Defaults to PATCH_SIZE.
        resolution (int): Hi-C resolution to use for sampling. Defaults to RESOLUTION.
        chroms (List[str]): Chromosomes to process. Defaults to ['1'].

    Returns:
        None
    """
    if chroms is None:
        chroms = ['1']

    chrom_processor = ChromosomeProcessor(
        chromosome_list=chroms,
        bedpe_dict=parsebed(bedpe_path, valid_threshold=1),
        contact_data_dir=image_data_dir,
        genome_assembly=assembly,
        patch_size=patch_size,
        resolution=resolution,
        output_dir=output_dir,
        plot_chrom=False,
        experiment=True)
    chrom_processor.run_sampling_experiments(thresholds=thesholds, normalizations=norms)

def load_sampled_data(
    thresholds: List[float],
    norms: List[str],
    patch_size: int = PATCH_SIZE,
    resolution: int = RESOLUTION,
    chroms: List[str] = None,
) -> ChromosomeLoader:
    """
    Initializes and returns a ChromosomeLoader configured for the given thresholds,
    normalizations, patch size, and resolution.

    Args:
        thresholds (List[float]): Quantile thresholds used to filter the sampled data.
        norms (List[str]): Normalization strategies applied to the data (e.g. ['log,zscore']).
        patch_size (int): Patch size to use for data loading. Defaults to PATCH_SIZE.
        resolution (int): Hi-C resolution to use for data loading. Defaults to RESOLUTION.
        chroms (List[str]): Chromosomes to load. Defaults to ['1'].

    Returns:
        ChromosomeLoader: Configured loader instance ready for split computation and generator creation.
    """
    if chroms is None:
        chroms = ['1']

    loader = ChromosomeLoader(chromosomes=chroms,
                              data_dir=output_dir,
                              patch_size=patch_size,
                              resolution=resolution,
                              batch_size=8,
                              split_ratios=(0.7, 0.2, 0.1),
                              include_diagonal=False,
                              use_original=False,
                              experiment=True,
                              thresholds=thresholds,
                              normalizations=norms)
    return loader

def train_evaluate_model(
    loader: ChromosomeLoader,
    upsample_strategy: str | None,
    avg_metric: str = "avg_perf",
    patch_size: int = PATCH_SIZE,
    resolution: int = RESOLUTION,
    num_runs: int = 3,
    drop_worst: int = 1,
    epochs: int = 30,
) -> None:
    """
    Creates experiment generators from the loader and trains/evaluates a UNet model
    for each experiment scenario, repeating runs to average out variance.

    Args:
        loader (ChromosomeLoader): Loaded dataset object containing sampled Hi-C data.
        upsample_strategy (str | None): Upsampling strategy to apply (e.g. 'random', 'balanced').
                                        If None, upsampling is disabled.
        avg_metric (str): Metric used to evaluate and compare model performance. Defaults to 'avg_perf'.
        patch_size (int): Patch size used for the UNet model input. Defaults to PATCH_SIZE.
        resolution (int): Hi-C resolution used for the model. Defaults to RESOLUTION.
        num_runs (int): Number of times to train the model; results are averaged. Defaults to 3.
        drop_worst (int): Number of worst runs to drop before averaging metrics. Defaults to 1.
        epochs (int): Number of training epochs per run. Defaults to 30.

    Returns:
        None
    """

    LOGGER.info(f'Loading the experiment generators for upsampling strategy {upsample_strategy}')
    generators = loader.create_experiment_generators(upsample=True if upsample_strategy else False,
                                                     strategy=upsample_strategy,
                                                     factor=3,
                                                     threshold=10)
    for scenario in generators:
        modeller = ChromosomeModeller(model=UNet(model_name=f'unet_{scenario}_ps_{patch_size}_res_{resolution}',
                                                 save_as=f'unet_{scenario}_ps_{patch_size}_res_{resolution}',
                                                 patch_size=patch_size,
                                                 avg_metric=avg_metric,
                                                 patience=10),
                                      train_generator=generators[scenario][0],
                                      val_generator=generators[scenario][1],
                                      test_generator=generators[scenario][2],
                                      epochs=epochs)
        avg_train_metrics, avg_val_metrics, avg_test_metrics = modeller.run_n_times(num_runs, drop_worst)
        print_metrics_table(avg_train_metrics, avg_val_metrics, avg_test_metrics, scenario.split("_"), patch_size, resolution)

        """
        An Identity map was evaluated with the data processed by
        mustache, but it ultimately didn't yoield meaningful results.
        """
        """
        if loader.normalizations == ['mustache']:
            modeller = ChromosomeModeller(model=IDMap(patch_size=patch_size,
                                                      avg_metric=avg_metric,
                                                      patience=10),
                                          train_generator=generators[scenario][0],
                                          val_generator=generators[scenario][1],
                                          test_generator=generators[scenario][0],
                                          epochs=30)
            avg_train_metrics, avg_val_metrics, avg_test_metrics = modeller.run_n_times(3, 1)
            print_metrics_table(avg_train_metrics, avg_val_metrics, avg_test_metrics,
                                'mustache'.split("_"),
                                patch_size, resolution)

            idm_gen = loader.combine_gens([generators[scenario][0], generators[scenario][1], generators[scenario][2]], shuffle=False)
            idm_gen.batch_size = 1
            modeller.model.build()
            modeller.model.test(idm_gen)
        """

# def create_generators(
#     loader: ChromosomeLoader,
#     upsample_strategy: str | None,
# ) -> dict:
#     """
#     Computes and saves data splits for the given loader, then creates and returns
#     train/val/test generators with optional upsampling.

#     Args:
#         loader (ChromosomeLoader): Loaded dataset object containing sampled Hi-C data.
#         upsample_strategy (str | None): Upsampling strategy to apply (e.g. 'random', 'balanced').
#                                         If None, upsampling is disabled.

#     Returns:
#         dict: A dictionary of train/val/test generators keyed by experiment scenario.
#     """
#     loader.compute_and_save_data_splits(overwrite=False)

#     LOGGER.info(f'Loading the experiment generators for upsampling strategy {upsample_strategy}')
#     generators = loader.create_experiment_generators(upsample=True if upsample_strategy else False,
#                                                      strategy=upsample_strategy,
#                                                      factor=3,
#                                                      threshold=10)

#     return generators

def exp1_exhaustive_quantile_norm_search(
    sample: bool,
    train: bool,
    upsample_strategies: List[str | None] = None,
) -> None:
    """
    Runs an exhaustive grid search over quantile thresholds and normalization strategies.
    Optionally samples data and/or trains and evaluates models for every
    combination of threshold, normalization, and upsampling strategy.

    Args:
        sample (bool): If True, samples and saves data for all threshold/normalization combos.
        train (bool): If True, trains and evaluates a model for each upsampling strategy.
        upsample_strategies (list | None): Upsampling strategies to iterate over.
                                           Defaults to the complete list of valid strategies ['random', 'balanced', None].
    """
    # sample data with thresholds from 50 to 90 (steps of 5) then 90 to 99 (steps of 1)
    exhaustive_thresholds = init_experiment_thresholds(50, 95, 5, None)
    exhaustive_thresholds = init_experiment_thresholds(91, 100, 1, exhaustive_thresholds)

    # all normalizations to test with
    exhaustive_normalizations = ['clip', 'divide']

    # default the upsampling to all use cases
    if upsample_strategies is None:
        upsample_strategies = ['random', 'balanced', None]

    if sample:
        sample_data(exhaustive_thresholds, exhaustive_normalizations)

    if train:
        for upsample_strategy in upsample_strategies:
            loader = load_sampled_data(exhaustive_thresholds, exhaustive_normalizations)
            loader.compute_and_save_data_splits(overwrite=False)
            train_evaluate_model(loader, upsample_strategy)

def exp2_evaluate_best_quantile_norm_with_log2(
    params: List[Tuple[float, str, str]],
    resample: bool,
    train: bool,
    avg_metric: str = "geo_mean",
) -> None:
    """
    Experiment 2: Evaluates the best quantile threshold from Experiment 1 across
    multiple log-based normalization pipelines to identify the optimal normalization strategy.

    Args:
        params (List[Tuple[float, str, str]]): List of (quantile, normalization, upsample_strategy)
            tuples defining each combination to evaluate.
            Example: [(0.96, 'log,zscore', 'random'), ...]
        resample (bool): If True, samples and saves data for each quantile/normalization combo.
        train (bool): If True, trains and evaluates a model for each combo.
        avg_metric (str): Metric used to evaluate model performance. Defaults to 'geo_mean'.

    Returns:
        None
    """
    # Sample data for all quantile/normalization combinations
    for quantile, normalization, upsample_strategy in params:
        if resample:
            sample_data([quantile], [normalization])

    # Train and evaluate a model for each quantile/normalization/upsampling combination
    for quantile, normalization, upsample_strategy in params:
        if train:
            loader = load_sampled_data([quantile], [normalization.replace(",", "_")])
            loader.compute_and_save_data_splits(overwrite=False)
            train_evaluate_model(loader, upsample_strategy, avg_metric=avg_metric)

def exp3_and4_evaluate_training_across_patch_resolution(
    params: dict,
    resample: bool,
    train: bool,
    chroms: List[str] = None,
) -> None:
    """
    Experiments 3 & 4: Evaluates model performance across all combinations of patch sizes
    and resolutions using a fixed normalization and upsampling strategy.

    Exp 3 performs a grid search over multiple patch sizes and resolutions.
    Exp 4 reuses this function with a single patch size and resolution to train
    the base enhanced model on all chromosomes.

    Args:
        params (dict): Experiment configuration containing:
            - threshold (float): Quantile threshold for data sampling.
            - normalization (str): Normalization pipeline (e.g. 'log,zscore').
            - upsampling_strategy (str): Upsampling strategy (e.g. 'random').
            - patch_sizes (List[int]): Patch sizes to evaluate (e.g. [32, 64, 128, 224]).
            - resolutions (List[int]): Hi-C resolutions to evaluate (e.g. [5000, 10000, 15000]).
        resample (bool): If True, samples and saves data for all patch size/resolution combos.
        train (bool): If True, trains and evaluates a model for each patch size/resolution combo.
        chroms (List[str]): Chromosomes to sample and train on. Defaults to ['1'].

    Returns:
        None
    """
    LOGGER.info(f'Experiment 3 & 4: Train a Model for Each Resolution on one chromosome (3) or all chromosomes (4)')

    if chroms is None:
        chroms = ['1']

    # Pull the run params out into variables
    threshold = params['threshold']
    norm = params['normalization']
    strategy = params['upsampling_strategy']
    patch_sizes = params['patch_sizes']
    resolutions = params['resolutions']

    if resample:
        # Sample data for every patch size and resolution combination
        for patch_size, res in product(patch_sizes, resolutions):
            sample_data([threshold], [norm], patch_size, res, chroms=chroms)

    if train:
        # Train and evaluate a model for every patch size and resolution combination
        for patch_size, res in product(patch_sizes, resolutions):
            LOGGER.info(f'Experiment 3: threshold: {threshold}, norm: {norm}, patch_size: {patch_size}, res: {res}')
            loader = load_sampled_data([threshold], [norm.replace(",", "_")],
                                       patch_size=patch_size, resolution=res, chroms=chroms)
            train_evaluate_model(loader, strategy, avg_metric="geo_mean", patch_size=patch_size, resolution=res,
                                 num_runs=3, drop_worst=1)

def exp5_train_across_resolutions(
    resample: bool,
    train: bool,
    chroms: List[str] = None,
) -> None:
    """
    Experiment 5: Trains a resolution-aware model by combining data from multiple
    Hi-C resolutions (5kb, 10kb, 15kb) into a single multi-resolution generator.
    Uses a fixed quantile threshold and log+zscore normalization.

    Args:
        resample (bool): If True, samples and saves data for all resolution/patch size combos.
        train (bool): If True, builds multi-resolution generators and trains a UNet model.
        chroms (List[str]): Chromosomes to sample and train on. Defaults to ['1'].

    Returns:
        None
    """
    LOGGER.info(f'Experiment 5: Resolution Aware Model')
    
    if chroms is None:
        chroms = ['1']

    threshold = 0.96
    norm = 'log,zscore'
    path_norm = norm.replace(",", "_")
    scenario = f'{threshold}_{path_norm}'

    if resample:
        # Sample data at each resolution for the fixed patch size
        for patch_size, res in product([64], [5000, 10000, 15000]):
            sample_data([threshold], [norm], patch_size, res, chroms=chroms)

    if train:
        # Load sampled data for each resolution
        loader_10k = load_sampled_data([threshold], [path_norm],
                                   patch_size=64, resolution=10000, chroms=chroms)
        loader_5k = load_sampled_data([threshold], [path_norm],
                                       patch_size=64, resolution=5000, chroms=chroms)
        loader_15k = load_sampled_data([threshold], [path_norm],
                                       patch_size=64, resolution=15000, chroms=chroms)

        # Create train/val/test generators for each resolution with random upsampling
        gen_10k = loader_10k.create_experiment_generators(upsample=True,
                                                         strategy='random',
                                                         factor=3,
                                                         threshold=10,
                                                          resolution=10000)
        gen_5k = loader_5k.create_experiment_generators(upsample=True,
                                                                 strategy='random',
                                                                 factor=3,
                                                                 threshold=10,
                                                        resolution=5000)
        gen_15k = loader_15k.create_experiment_generators(upsample=True,
                                                                 strategy='random',
                                                                 factor=3,
                                                                 threshold=10,
                                                          resolution=15000)

        # Unpack train/val/test generators for each resolution
        train_5k, val_5k, test_5k = list(gen_5k[scenario])[:3]
        train_10k, val_10k, test_10k = list(gen_10k[scenario])[:3]
        train_15k, val_15k, test_15k = list(gen_15k[scenario])[:3]

        # train_5k = list(gen_5k[scenario])[0]
        # val_5k = list(gen_5k[scenario])[1]
        # test_5k = list(gen_5k[scenario])[2]

        # train_10k = list(gen_10k[scenario])[0]
        # val_10k = list(gen_10k[scenario])[1]
        # test_10k = list(gen_10k[scenario])[2]

        # train_15k = list(gen_15k[scenario])[0]
        # val_15k = list(gen_15k[scenario])[1]
        # test_15k = list(gen_15k[scenario])[2]

        # Combine generators across all resolutions into unified train/val/test generators
        train_gen = loader_10k.combine_gens([train_5k, train_10k, train_15k], shuffle=True)
        val_gen = loader_10k.combine_gens([val_5k, val_10k, val_15k], shuffle=False)
        test_gen = loader_10k.combine_gens([test_5k, test_10k, test_15k], shuffle=False)

        # Train a UNet model on the combined multi-resolution data
        modeller = ChromosomeModeller(model=UNet(model_name=f'unet_multi_res',
                                                 save_as=f'unet_multi_res',
                                                 patch_size=64,
                                                 avg_metric="geo_mean"),
                                      train_generator=train_gen,
                                      val_generator=val_gen,
                                      test_generator=test_gen,
                                      epochs=30)
        avg_train_metrics, avg_val_metrics, avg_test_metrics = modeller.run_n_times(3, 1)
        print_metrics_table(avg_train_metrics, avg_val_metrics, avg_test_metrics, None, 64,
                            "combined")

def exp6_giloop_and_mustache(
    sample: bool = True,
    train: bool = True,
    chroms: List[str] = None,
    avg_metric: str = "geo_mean",
    epochs: int = 30,
) -> None:
    """
    Experiment 6: Compares three normalization pipelines for loop detection:
        1. Custom normalization only (log + zscore)
        2. Mustache preprocessing only
        3. Combined custom normalization + Mustache

    All combinations are run at a fixed quantile threshold, patch size,
    and resolution, with random upsampling.

    Args:
        sample (bool): If True, samples and saves data for all normalization/resolution combos.
        train (bool): If True, trains and evaluates a model for each normalization/upsampling combo.
        chroms (List[str]): Chromosomes to sample data from. Defaults to ['1'].
        avg_metric (str): Metric used to evaluate model performance. Defaults to 'geo_mean'.
        epochs (int): Number of training epochs. Defaults to 30.

    Returns:
        None
    """
    LOGGER.info(f'Experiment 6: Comparison of custom normalization vs Mustache processing vs custom norm and Mustache')
    thresholds = [0.96]
    normalizations = ['log,zscore', 'mustache', 'log,zscore,mustache']
    upsampling_strategy = ['random']
    patch_sizes = [64]
    resolutions = [10000]

    if sample:
        # Sample data for every combination of quantile, normalization, patch size, and resolution
        for quantile, normalization, patch_size, resolution in product(thresholds,
                                                                       normalizations,
                                                                       patch_sizes,
                                                                       resolutions):
            sample_data([quantile], [normalization], patch_size, resolution, chroms)

    if train:
        # Train a model for each quantile/normalization/upsampling combination
        for quantile, normalization, upsample_strategy in product(thresholds,
                                                                  normalizations,
                                                                  upsampling_strategy):
            # Replace commas in normalization string to match saved file naming convention
            norm_file_name = normalization.replace(",", "_")
            loader = load_sampled_data([quantile], [norm_file_name])
            loader.compute_and_save_data_splits()
            train_evaluate_model(loader, upsample_strategy, avg_metric=avg_metric, epochs=epochs)

if __name__ == '__main__':

    # Reference genome assembly used for all experiments
    assembly = 'hg38'

    # Path to the BEDPE file containing ground truth loop annotations for HeLa-S3
    bedpe_path = 'bedpe/hela.hg38.bedpe'

    # Directory containing the raw Hi-C contact map text files
    image_data_dir = 'data/txt_hela_100'

    # Dataset to run experiments on
    dataset_name = 'hela_100'

    # Root output directory for all sampled/processed experiment datasets
    output_dir = f'experiment_datasets/{dataset_name}'

    # All valid chromosomes for HeLa-S3 — Chr18 is excluded as it is absent in the Hi-C file
    all_chroms = \
        [str(i) for i in range(1, 18)] + \
        [str(i) for i in range(19, 23)] + ['X']  # Chr18 of HeLa-S3 is absent in the Hi-C file

    # Exp 2: best quantile (0.96) and best upsampling strategy (random) from Exp 1 paired with various log-based normalization pipelines
    exp2_params = [(0.96, 'log,clip', 'random'), (0.96, 'log,divide', 'random'),
                   (0.96, 'log,clip,divide', 'random'), (0.96, 'log,divide,zscore', 'random'),
                   (0.96, 'log,clip,zscore', 'random'), (0.96, 'log,zscore', 'random'),
                   (0.96, 'zscore', 'random')]

    # Exp 3: grid search over patch sizes and resolutions using the best norm strategy from Exp 2
    exp3_params = \
    {
        'threshold': 0.96,
        'normalization': 'log,zscore',
        'upsampling_strategy': 'random',
        'patch_sizes': [32, 64, 128, 224],
        'resolutions': [5000, 10000, 15000]
    }

    # Exp 4: same setup as Exp 3 but fixed to a single patch size/resolution,
    # trained on all chromosomes to produce the base enhanced model
    exp4_params = \
        {
            'threshold': 0.96,
            'normalization': 'log,zscore',
            'upsampling_strategy': 'random',
            'patch_sizes': [64],
            'resolutions': [10000]
        }

    # Define all experiments to run in sequence
    exps = [
            exp1_exhaustive_quantile_norm_search(sample=True, train=True),
            exp2_evaluate_best_quantile_norm_with_log2(exp2_params, True, True, avg_metric="avg_perf"),
            exp3_and4_evaluate_training_across_patch_resolution(exp3_params, True, True),
            exp3_and4_evaluate_training_across_patch_resolution(exp4_params, True, True, chroms=all_chroms),
            exp5_train_across_resolutions(True, True, all_chroms),
            exp6_giloop_and_mustache(True, True, chroms=all_chroms, epochs=1)
           ]

    # Execute each experiment in order
    for exp in exps:
        exp

    # run the exhaustive search for the best quantile and norm method
    
    # exp1_exhaustive_quantile_norm_search(sample=False, train=True)

    # run the best params from the exhaustive search with log2 norm
    # exp2_params = [(0.96, 'log,clip', 'random'), (0.96, 'log,divide', 'random'),
    #                    (0.96, 'log,clip,divide', 'random'), (0.96, 'log,divide,zscore', 'random'),
    #                    (0.96, 'log,clip,zscore', 'random'), (0.96, 'log,zscore', 'random'),
    #                    (0.96, 'zscore', 'random')]
    # exp2_evaluate_best_quantile_norm_with_log2(exp2_params, False, True, avg_metric="avg_perf")

    # exp3_params = \
    #     {
    #         'threshold': 0.96,
    #         'normalization': 'log,zscore',
    #         'upsampling_strategy': 'random',
    #         'patch_sizes': [32, 64, 128, 224],
    #         'resolutions': [5000, 10000, 15000]
    #     }
    # exp3_evaluate_training_across_patch_resolution(exp3_params, False, True)

    # reuse the code from exp3 for exp4: training the "base enhanced model" on all chroms
    # exp4_params = \
    #     {
    #         'threshold': 0.96,
    #         'normalization': 'log,zscore',
    #         'upsampling_strategy': 'random',
    #         'patch_sizes': [64],
    #         'resolutions': [10000]
    #     }
    # exp3_evaluate_training_across_patch_resolution(exp4_params, True, True, chroms=all_chroms)

    # experiment 5: compare the results from exp4 to that of exp5 (one resolution all chroms vs all resolutions all chroms)
    # exp5_train_across_resolutions(False, True, all_chroms)

    # experiment 6: compare the behavior of giloop sampling, mustache smapling, giloop + mustache sampling, and verify mustache efficiency by itself
    # exp6_giloop_and_mustache(False, True, chroms=all_chroms, epochs=30)