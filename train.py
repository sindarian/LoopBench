import os

from keras.models import load_model
from tensorflow.keras import backend as K

from generators.chromosome_generator import ChromosomeGenerator
from model.cnn import UNet
from model.custom_layers import ClipByValue
from util.constants import PATCH_SIZE, RESOLUTION, OUTPUT_DIR, MODELS_DIR
from util.logger import Logger
import logging

from model.loop_net import LoopNet

LOGGER = Logger(name='train', level=logging.DEBUG).get_logger()

def train_run_cnn(train_gen: ChromosomeGenerator,
                  val_gen: ChromosomeGenerator,
                  test_gen: ChromosomeGenerator,
                  clip_value: float,
                  patch_size: int,
                  resolution: int,
                  epochs: int = 50):
    giloop_cnn = UNet(save_as=f'cnn_ps{patch_size}_res{resolution}', patch_size=patch_size, resolution=resolution)
    giloop_cnn.build(clip_value)
    giloop_cnn.train(train_gen, val_gen, epochs)
    giloop_cnn.predict(test_gen)
    K.clear_session()

def train_loop_net(train_gen: ChromosomeGenerator,
                   val_gen: ChromosomeGenerator,
                   test_gen: ChromosomeGenerator,
                   clip_value: float,
                   patch_size: int,
                   resolution: int,
                   epochs: int = 50):
    loop_net = LoopNet(save_as=f'loopnet_ps{patch_size}_res{resolution}', patch_size=patch_size,)

    loop_net.build(clip_value)
    # loop_net.build_center(clip_value)
    # loop_net.build_center_global_avg_pooling(clip_value)
    # loop_net.build_center_crop_pooling(clip_value)

    loop_net.train(train_gen, val_gen, epochs)
    loop_net.predict(test_gen)
    K.clear_session()