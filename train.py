from tensorflow.keras import backend as K

from model.cnn import GILoopCnn
from util.logger import Logger
import logging

from model.loop_net import LoopNet

LOGGER = Logger(name='train', level=logging.DEBUG).get_logger()

def train_run_cnn(train_gen,
                  val_gen,
                  test_gen,
                  clip_value,
                  epochs: int = 50):
    giloop_cnn = GILoopCnn()
    giloop_cnn.build(clip_value)
    giloop_cnn.train(train_gen, val_gen, epochs)
    giloop_cnn.predict(test_gen)
    K.clear_session()

def train_loop_net(
            train_gen,
            val_gen,
            test_gen,
            clip_value,
            epochs: int = 50):

    loop_net = LoopNet()
    loop_net.build(clip_value)
    loop_net.train(train_gen, val_gen, epochs)
    loop_net.predict(test_gen)