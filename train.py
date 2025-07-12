from cnn import cnn_run, build_cnn
from tensorflow.keras import backend as K

from logger import Logger
import logging

from loop_net import LoopNet
from metric_configs import efficient_net_hyperparams, resnet_hyperparams
from utils import PATCH_SIZE, \
    estimate_upper_bound, create_hic_generators, visualize_data, LOG_LEVEL

LOGGER = Logger(name='train', level=logging.DEBUG).get_logger()

def train_run_cnn(train_gen,
                  val_gen,
                  test_gen,
                  clip_value,
                  epochs: int = 50):
    # cnn_run(chroms, dataset_name, epochs)
    giloop_cnn = build_cnn(clip_value)
    cnn_run(giloop_cnn, train_gen, val_gen, test_gen, epoch=epochs)
    K.clear_session()

def train_loop_net(
            dataset_name: str,
            train_gen,
            val_gen,
            test_gen,
            clip_value,
            model_name: str = 'resnet50',
            hyper_params: dict = resnet_hyperparams,
            epochs: int = 50):
    # data directory to read HiC maps from
    # dataset_dir = 'dataset/' + dataset_name

    # create generators to feed the train, val, and test data to the model
    # train_gen, val_gen, test_gen = create_hic_generators(
        # chrom_names=chroms,
        # data_dir=dataset_dir,
        # patch_size=PATCH_SIZE,
        # split_ratios=(0.8, 0.1, 0.1), # train, val, test
        # batch_size=batch_size
    # )

    # visualize the training data distribution
    visualize_data(train_gen)

    # estimate the upper bound of the training data
    # this is used to scale the data and clip outliers when the model is built
    # x_train_upper_bound = estimate_upper_bound(train_gen)
    # print("Percentile upper bound:", x_train_upper_bound)

    net = LoopNet(model_name=model_name)
    net.build(clip_value, hyper_params)
    net.train(train_gen, val_gen, epochs)
    net.predict(test_gen)