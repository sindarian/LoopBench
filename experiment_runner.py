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

def init_experiment_thresholds(lower_bound, upper_bound, increment, existing_threshods):
    thresholds = [(lower_bound + (increment * inc_lower_by)) / 100 for inc_lower_by in
                  range(int((upper_bound - lower_bound) / increment))]

    if existing_threshods:
        thresholds = existing_threshods + thresholds

    return thresholds

def sample_data(thesholds, norms, patch_size=PATCH_SIZE, resolution=RESOLUTION, chroms=['1']):
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

def load_sampled_data(thresholds, norms, patch_size=PATCH_SIZE, resolution=RESOLUTION, chroms=['1']):
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

def train_evaluate_model(loader, upsample_strategy, avg_metric="avg_perf", patch_size=PATCH_SIZE, resolution=RESOLUTION,
                         num_runs=3, drop_worst=1):
    loader.compute_and_save_data_splits(overwrite=False)

    print(f'Loading the experiment generators for upsampling strategy {upsample_strategy}')
    generators = loader.create_experiment_generators(upsample=True if upsample_strategy else False,
                                                     strategy=upsample_strategy,
                                                     factor=3,
                                                     threshold=10)
    for scenario in generators:
        # modeller = ChromosomeModeller(model=UNet(model_name=f'unet_{scenario}_ps_{patch_size}_res_{resolution}',
        #                                          save_as=f'unet_{scenario}_ps_{patch_size}_res_{resolution}',
        #                                          patch_size=patch_size,
        #                                          avg_metric=avg_metric,
        #                                          patience=10),
        #                               train_generator=generators[scenario][0],
        #                               val_generator=generators[scenario][1],
        #                               test_generator=generators[scenario][2],
        #                               epochs=30)
        # avg_train_metrics, avg_val_metrics, avg_test_metrics = modeller.run_n_times(num_runs, drop_worst)
        # print_metrics_table(avg_train_metrics, avg_val_metrics, avg_test_metrics, scenario.split("_"), patch_size, resolution)

        if loader.normalizations == ['mustache']:
            modeller = ChromosomeModeller(model=IDMap(patch_size=patch_size,
                                                      avg_metric=avg_metric,
                                                      patience=10),
                                          train_generator=generators[scenario][0],
                                          val_generator=generators[scenario][1],
                                          test_generator=generators[scenario][0],
                                          epochs=30)
            # avg_train_metrics, avg_val_metrics, avg_test_metrics = modeller.run_n_times(3, 1)
            # print_metrics_table(avg_train_metrics, avg_val_metrics, avg_test_metrics,
            #                     'mustache'.split("_"),
            #                     patch_size, resolution)

            idm_gen = loader.combine_gens([generators[scenario][0], generators[scenario][1], generators[scenario][2]], shuffle=False)
            idm_gen.batch_size = 1
            modeller.model.build()
            modeller.model.test(idm_gen)

def create_generators(loader, upsample_strategy):
    loader.compute_and_save_data_splits(overwrite=False)

    print(f'Loading the experiment generators for upsampling strategy {upsample_strategy}')
    generators = loader.create_experiment_generators(upsample=True if upsample_strategy else False,
                                                     strategy=upsample_strategy,
                                                     factor=3,
                                                     threshold=10)

    return generators

def exp1_exhaustive_quantile_norm_search(sample, train, upsample_strategies=None):
    # sample data with thresholds from 50 to 90 (steps of 5) then 90 to 99 (steps of 1
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

def exp2_evaluate_best_quantile_norm_with_log2(params, resample, train, avg_metric="geo_mean"):
    for quantile, normalization, upsample_strategy in params:
        if resample:
            sample_data([quantile], [normalization])

    for quantile, normalization, upsample_strategy in params:
        if train:
            loader = load_sampled_data([quantile], [normalization.replace(",", "_")])
            loader.compute_and_save_data_splits(overwrite=False)
            train_evaluate_model(loader, upsample_strategy, avg_metric=avg_metric)

def exp3_evaluate_training_across_patch_resolution(params, resample, train, chroms=['1']):
    # threshold, norm, strategy = params

    # pull the run params out into variables
    threshold = params['threshold']
    norm = params['normalization']
    strategy = params['upsampling_strategy']
    patch_sizes = params['patch_sizes']
    resolutions = params['resolutions']

    if resample:
        for patch_size, res in product(patch_sizes, resolutions):
            sample_data([threshold], [norm], patch_size, res, chroms=chroms)

    if train:
        for patch_size, res in product(patch_sizes, resolutions):
            LOGGER.info(f'Experiment 3: threshold: {threshold}, norm: {norm}, patch_size: {patch_size}, res: {res}')
            loader = load_sampled_data([threshold], [norm.replace(",", "_")],
                                       patch_size=patch_size, resolution=res, chroms=chroms)
            train_evaluate_model(loader, strategy, avg_metric="geo_mean", patch_size=patch_size, resolution=res,
                                 num_runs=3, drop_worst=1)

def exp5_train_across_resolutions(resample, train, chroms=['1']):
    LOGGER.info(f'Experiment 5: Resolution aware model')
    threshold = 0.96
    norm = 'log,zscore'
    path_norm = norm.replace(",", "_")
    scenario = f'{threshold}_{path_norm}'

    if resample:
        for patch_size, res in product([64], [5000, 10000, 15000]):
            sample_data([threshold], [norm], patch_size, res, chroms=chroms)

    if train:
        # loaders
        loader_10k = load_sampled_data([threshold], [path_norm],
                                   patch_size=64, resolution=10000, chroms=chroms)
        loader_5k = load_sampled_data([threshold], [path_norm],
                                       patch_size=64, resolution=5000, chroms=chroms)
        loader_15k = load_sampled_data([threshold], [path_norm],
                                       patch_size=64, resolution=15000, chroms=chroms)

        # generators
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

        train_5k = list(gen_5k[scenario])[0]
        val_5k = list(gen_5k[scenario])[1]
        test_5k = list(gen_5k[scenario])[2]

        train_10k = list(gen_10k[scenario])[0]
        val_10k = list(gen_10k[scenario])[1]
        test_10k = list(gen_10k[scenario])[2]

        train_15k = list(gen_15k[scenario])[0]
        val_15k = list(gen_15k[scenario])[1]
        test_15k = list(gen_15k[scenario])[2]

        train_gen = loader_10k.combine_gens([train_5k, train_10k, train_15k], shuffle=True)
        val_gen = loader_10k.combine_gens([val_5k, val_10k, val_15k], shuffle=False)
        test_gen = loader_10k.combine_gens([test_5k, test_10k, test_15k], shuffle=False)

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

def exp6_giloop_and_mustache(sample=True, train=True, chroms=['1'], avg_metric="geo_mean"):
    thresholds = [0.96]
    # normalizations = ['log,zscore', 'mustache']
    normalizations = ['mustache']
    upsampling_strategy = ['random']
    patch_sizes = [64]
    resolutions = [10000]

    if sample:
        for quantile, normalization, patch_size, resolution in product(thresholds,
                                                                       normalizations,
                                                                       patch_sizes,
                                                                       resolutions):
            sample_data([quantile], [normalization], patch_size, resolution, chroms)

    if train:
        all_generators = {'train': [], 'val': [], 'test': []}

        for quantile, normalization, upsample_strategy in product(thresholds,
                                                                  normalizations,
                                                                  upsampling_strategy):
            norm_file_name = normalization.replace(",", "_")
            loader = load_sampled_data([quantile], [norm_file_name])
            loader.compute_and_save_data_splits()
            train_evaluate_model(loader, upsample_strategy, avg_metric=avg_metric)

        #     generator = create_generators(loader, upsample_strategy)
        #     for gen in generator:
        #         all_generators['train'].append(generator[gen][0])
        #         all_generators['val'].append(generator[gen][1])
        #         all_generators['test'].append(generator[gen][2])
        #
        # # compare both
        # combined_train_gen = loader.combine_gens(all_generators['train'], shuffle=True)
        # combined_val_gen = loader.combine_gens(all_generators['val'], shuffle=False)
        # combined_test_gen = loader.combine_gens(all_generators['test'], shuffle=False)
        #
        # modeller = ChromosomeModeller(model=UNet(model_name=f'unet_combined',
        #                                          save_as=f'unet_combined',
        #                                          patch_size=patch_sizes[0],
        #                                          avg_metric=avg_metric,
        #                                          patience=10),
        #                               train_generator=combined_train_gen,
        #                               val_generator=combined_val_gen,
        #                               test_generator=combined_test_gen,
        #                               epochs=30)
        # avg_train_metrics, avg_val_metrics, avg_test_metrics = modeller.run_n_times(3, 1)
        # print_metrics_table(avg_train_metrics, avg_val_metrics, avg_test_metrics, 'log,zscore,mustache'.split("_"),
        #                     patch_sizes[0], resolutions[0])

if __name__ == '__main__':

    assembly = 'hg38'

    bedpe_path = 'bedpe/hela.hg38.bedpe'

    image_data_dir = 'data/txt_hela_100'

    dataset_name = 'hela_100'

    output_dir = f'experiment_datasets/{dataset_name}'

    all_chroms = \
        [str(i) for i in range(1, 18)] + \
        [str(i) for i in range(19, 23)] + ['X']  # Chr18 of HeLa-S3 is absent in the Hi-C file

    # run the exhaustive search for the best quantile and norm method
    # exp1_exhaustive_quantile_norm_search(False, True)

    # run the best params from the exhaustive search with log2 norm
    exp2_params = [(0.96, 'log,clip', 'random'), (0.96, 'log,divide', 'random'),
                       (0.96, 'log,clip,divide', 'random'), (0.96, 'log,divide,zscore', 'random'),
                       (0.96, 'log,clip,zscore', 'random'), (0.96, 'log,zscore', 'random'),
                       (0.96, 'zscore', 'random')]
    # exp2_evaluate_best_quantile_norm_with_log2(exp2_params, False, True, avg_metric="avg_perf")

    exp3_params = \
        {
            'threshold': 0.96,
            'normalization': 'log,zscore',
            'upsampling_strategy': 'random',
            'patch_sizes': [32, 64, 128, 224],
            'resolutions': [5000, 10000, 15000]
        }
    # exp3_evaluate_training_across_patch_resolution(exp3_params, False, True)

    # reuse the code from exp3 for exp4: training the "base enhanced model" on all chroms
    exp4_params = \
        {
            'threshold': 0.96,
            'normalization': 'log,zscore',
            'upsampling_strategy': 'random',
            'patch_sizes': [64],
            'resolutions': [10000]
        }
    # exp3_evaluate_training_across_patch_resolution(exp4_params, True, True, chroms=all_chroms)

    # experiment 5: compare the results from exp4 to that of exp5 (one resolution all chroms vs all resolutions all chroms)
    # exp5_train_across_resolutions(False, True, all_chroms)

    # experiment 6: compare the behavior of giloop sampling, mustache smapling, giloop + mustache sampling, and verify mustache efficiency by itself
    exp6_giloop_and_mustache(False, True)