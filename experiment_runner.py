from generators.chromosome_loader import ChromosomeLoader
from generators.chromosome_processor import ChromosomeProcessor
from gutils import parsebed
from model.cnn import UNet
from model.chromosome_modeller import ChromosomeModeller
from util.constants import PATCH_SIZE
from util.logger import Logger
import logging

LOGGER = Logger(name='experiment_runner', level=logging.DEBUG).get_logger()


def init_experiment_thresholds(lower_bound, upper_bound, increment, existing_threshods):
    thresholds = [(lower_bound + (increment * inc_lower_by)) / 100 for inc_lower_by in
                  range(int((upper_bound - lower_bound) / increment))]

    if existing_threshods:
        thresholds = existing_threshods + thresholds

    return thresholds

def print_metrics_table(avg_metrics, best_metrics, split_scenario_name):
    metric_names = [
        "binary_accuracy",
        "ROC_AUC",
        "PR_AUC",
        "recall",
        "precision",
        "specificity",
        "avg_perf"
    ]

    # Scenario header
    print(f"\nMetrics for scenario: quantile={split_scenario_name[0]}, "
          f"normalization={','.join(split_scenario_name[1:])}")

    # Table header
    print(f"{'Metric':<15} {'Average':>10} {'Best':>10}")
    print("-" * 37)

    # Table rows
    for name, avg_val, best_val in zip(metric_names, avg_metrics, best_metrics):
        print(f"{name:<15} {avg_val:10.4f} {best_val:10.4f}")
    print()


def run_exhaustive_quartiles(resample, train):
    # for sampling data with a quantile from 50 to 90 in steps of 5
    lower_bound = 50
    upper_bound = 95
    increment = 5
    exhaustive_thresholds = init_experiment_thresholds(lower_bound, upper_bound, increment, None)

    # for sampling data with a quantile from 90 to 99 in steps of 1
    lower_bound = 91 # 90 already sampled
    upper_bound = 100
    increment = 1
    exhaustive_thresholds = init_experiment_thresholds(lower_bound, upper_bound, increment, exhaustive_thresholds)

    exhaustive_normalizations = ['clip', 'divide']

    if resample:
        chrom_processor = ChromosomeProcessor(
            chromosome_list=['1'],
            bedpe_dict=parsebed(bedpe_path, valid_threshold=1),
            contact_data_dir=image_data_dir,
            genome_assembly=assembly,
            output_dir=output_dir,
            plot_chrom=False,
            experiment=True)

        chrom_processor.run_sampling_experiments(thresholds=exhaustive_thresholds, normalizations=exhaustive_normalizations)

    if train:
        upsample_strategies = [(True, 'random'), (True, 'balanced'), (False, None)]

        for upsample_strategy in upsample_strategies:
            loader = ChromosomeLoader(chromosomes=['1'],
                                      data_dir=output_dir,
                                      patch_size=PATCH_SIZE,
                                      batch_size=8,
                                      split_ratios=(0.7, 0.2, 0.1),
                                      include_diagonal=False,
                                      use_original=False,
                                      experiment=True,
                                      thresholds=exhaustive_thresholds,
                                      normalizations=exhaustive_normalizations)
            loader.compute_and_save_data_splits(overwrite=False)

            print(f'Loading the experiment generators for upsampling strategy {upsample_strategy[1]}')
            generators = loader.create_experiment_generators(upsample=upsample_strategy[0],
                                                             strategy=upsample_strategy[1],
                                                             factor=3,
                                                             threshold=10)

            print('Running all configured scenarios...')
            for scenario in generators:
                modeller = ChromosomeModeller(model=UNet(model_name=f'cnn_{scenario}',
                                                         save_as=f'cnn_{scenario}',
                                                         patch_size=PATCH_SIZE),
                                              train_generator=generators[scenario][0],
                                              val_generator=generators[scenario][1],
                                              test_generator=generators[scenario][2],
                                              epochs=30)
                avg_metrics, best_metrics = modeller.run_n_times(3, 1)
                print_metrics_table(avg_metrics, best_metrics, scenario.split("_"))

def run_enhanced_norms(run_params, resample, train):
    for quantile, normalization, upsample_strategy in run_params:
        if resample:
            chrom_processor = ChromosomeProcessor(
                chromosome_list=['1'],
                bedpe_dict=parsebed(bedpe_path, valid_threshold=1),
                contact_data_dir=image_data_dir,
                genome_assembly=assembly,
                output_dir=output_dir,
                plot_chrom=False,
                experiment=True)
            chrom_processor.run_sampling_experiments(thresholds=[quantile], normalizations=[normalization])

        if train:
            loader = ChromosomeLoader(chromosomes=['1'],
                                      data_dir=output_dir,
                                      patch_size=PATCH_SIZE,
                                      batch_size=8,
                                      split_ratios=(0.7, 0.2, 0.1),
                                      include_diagonal=False,
                                      use_original=False,
                                      experiment=True,
                                      thresholds=[quantile],
                                      normalizations=[normalization.replace(",", "_")])
            loader.compute_and_save_data_splits(overwrite=False)

            print(f'Loading the experiment generators for upsampling strategy {upsample_strategy}')
            generators = loader.create_experiment_generators(upsample=True if upsample_strategy else False,
                                                             strategy=upsample_strategy,
                                                             factor=3,
                                                             threshold=10)
            for scenario in generators:
                modeller = ChromosomeModeller(model=UNet(model_name=f'unet_{scenario}',
                                                         save_as=f'unet_{scenario}',
                                                         patch_size=PATCH_SIZE),
                                              train_generator=generators[scenario][0],
                                              val_generator=generators[scenario][1],
                                              test_generator=generators[scenario][2],
                                              epochs=30)
                avg_metrics, best_metrics = modeller.run_n_times(3, 1)
                print_metrics_table(avg_metrics, best_metrics, scenario.split("_"))


if __name__ == '__main__':

    assembly = 'hg38'

    bedpe_path = 'bedpe/hela.hg38.bedpe'

    image_data_dir = 'data/txt_hela_100'

    dataset_name = 'hela_100'

    output_dir = f'experiment_datasets/{dataset_name}'

    # run the exhaustive search for the best quantile and norm method
    run_exhaustive_quartiles(False, True)

    # run the best params from the exhaustive search with log2 norm
    params = [(0.96, 'log2,clip', 'random'), (0.55, 'log2,divide', 'random'),
              (0.95, 'log2,clip', 'balanced'), (0.5, 'log2,divide', 'balanced'),]
    run_enhanced_norms(params, False, True)
