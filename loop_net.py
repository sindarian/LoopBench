import json
import logging

import numpy as np
from keras.layers import BatchNormalization
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import (Input, Dropout, Activation, Reshape, GaussianNoise, Dense, Rescaling, ReLU, Conv2D,
                                     Concatenate, Flatten)
from tensorflow.keras.layers import concatenate
from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.keras.metrics import BinaryAccuracy, AUC
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.optimizers.schedules import PolynomialDecay
from tensorflow.python.keras.layers import UpSampling2D

from custom_layers import ClipByValue
from logger import Logger
from metric_configs import EARLY_STOP, efficient_net_hyperparams
from metrics import compute_auc
from utils import PATCH_SIZE, FLATTENED_PATCH_SIZE, LOG_LEVEL


class LoopNet:

    def __init__(self, model_name='efnb0'):
        self.model = None
        self.model_name = model_name
        self.LOGGER = Logger(name='LoopNet_' + self.model_name, level=logging.DEBUG).get_logger()

    def build(self, image_upper_bound, hyper_params=efficient_net_hyperparams):
        I = Input(shape=(PATCH_SIZE, PATCH_SIZE))
        x = ClipByValue(image_upper_bound)(I)
        x = Rescaling(1. / image_upper_bound)(x)
        x = Reshape((PATCH_SIZE, PATCH_SIZE, 1))(x)
        x = GaussianNoise(0.05)(x)
        x = Concatenate()([x, x, x]) # convert the grayscale input to 3 dim

        if self.model_name == 'efnb0':
            base_model = EfficientNetB0(include_top=False, weights='imagenet', input_tensor=x)

            # feature extraction layers from EfficientNetB0 for U-Net skip connections
            skip_names = [
                "block2a_activation",
                "block3a_activation",
                "block4a_activation",
                "block6a_activation",
            ]
            skip_outputs = [base_model.get_layer(name).output for name in skip_names]
            x = base_model.output

            x = self.upsample_block(x, skip_outputs)

            cnn_logits = Conv2D(1, 1, padding='same', name='logits')(x)
            cnn_sigmoid = Activation('sigmoid', name='sigmoid')(cnn_logits)
            cnn_sigmoid = Flatten()(cnn_sigmoid)
        else: # resnet50
            base_model = ResNet50(include_top=False, weights='imagenet', input_tensor=x)

            # feature extraction layers from ResNet50 for U-Net skip connections
            skip_names = [
                "conv1_relu",
                "conv2_block3_out",
                "conv3_block4_out",
                "conv4_block6_out",
            ]
            skip_outputs = [base_model.get_layer(name).output for name in skip_names]
            x = base_model.get_layer("conv5_block3_out").output

            x = self.upsample_block(x, skip_outputs)

            cnn_logits = Conv2D(1, 1, padding='same', name='logits')(x)
            cnn_sigmoid = Activation('sigmoid', name='sigmoid')(cnn_logits)
            cnn_sigmoid = Flatten()(cnn_sigmoid)

        model = Model(inputs=[I], outputs=[cnn_sigmoid])
        model.compile(
            optimizer=Adam(learning_rate=PolynomialDecay(initial_learning_rate=hyper_params['optimizer']['lr'],
                                                         decay_steps=hyper_params['optimizer']['decay'],
                                                         end_learning_rate=hyper_params['optimizer']['end_lr'],
                                                         power=hyper_params['optimizer']['power'])
                           ),
            loss=BinaryCrossentropy(),
            loss_weights=FLATTENED_PATCH_SIZE,
            metrics=[BinaryAccuracy(name='binary_accuracy', threshold=0.5),
                     AUC(curve="ROC", name='ROC_AUC'),
                     AUC(curve="PR", name='PR_AUC')]
        )
        model.summary()
        self.model = model
        return model

    def train(self, train_gen, val_gen, epochs):
        history = self.model.fit(train_gen,
                                 validation_data=val_gen,
                                 epochs=epochs,
                                 workers=4,
                                 use_multiprocessing=True,
                                 callbacks=[EarlyStopping(monitor='val_PR_AUC',
                                                          min_delta=0.0001,
                                                          patience=5,
                                                          verbose=1,
                                                          mode='max',
                                                          restore_best_weights=True)],
                                 verbose=1)

        # save the training history
        with open(f'metrics/{self.model_name}_training_metrics.json', 'w') as f:
            json.dump(history.history, f)

        # TODO: fix RESNET50 model saving
        # Save architecture
        # with open(f'models/{model_name}_arch.json', 'w') as f:
        #     f.write(model.to_json())
        # Save weights separately
        # model.save_weights(f'models/{model_name}_weights.h5')

    def predict(self, test_gen):
        # make predictions
        y_pred = self.model.predict(test_gen, workers=4, use_multiprocessing=True, verbose=1)

        # collect true labels from generator
        y_true_batches = []
        for _, y_batch in test_gen:
            y_true_batches.append(y_batch)
        y_true = np.concatenate(y_true_batches, axis=0)

        test_auc, test_ap = compute_auc(y_pred, y_true.astype('bool'))
        self.LOGGER.info('Test AUC is {}. Test AP is {}.'.format(test_auc, test_ap))

    def upsample_block(self, x, skip_outputs):
        filters = [512, 256, 128, 64]

        for i in range(4):
            x = UpSampling2D((2, 2))(x)
            x = Conv2D(filters[i], 3, padding='same', activation='relu')(x)
            x = BatchNormalization()(x)
            x = Conv2D(filters[i], 3, padding='same', activation='relu')(x)
            x = BatchNormalization()(x)
            x = Dropout(0.3)(x)

            # check that skip and x are same spatial size
            skip = skip_outputs[3 - i]
            if skip.shape[1:3] != x.shape[1:3]:
                skip = UpSampling2D(size=(x.shape[1] // skip.shape[1], x.shape[2] // skip.shape[2]))(skip)

            x = concatenate([x, skip])

        # final upsample to 224x224
        x = UpSampling2D((2, 2))(x)
        x = Conv2D(32, 3, padding='same', activation='relu')(x)
        x = BatchNormalization()(x)
        x = Conv2D(16, 3, padding='same')(x)
        x = BatchNormalization()(x)

        return x
