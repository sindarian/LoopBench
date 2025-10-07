#!/usr/bin/env python3
import tensorflow as tf
from keras.layers import Rescaling, GlobalAveragePooling2D, Concatenate, Lambda
from tensorflow.keras.applications.efficientnet import preprocess_input
from tensorflow.keras.layers import (Input, Dropout, ReLU, Activation, Conv2D, MaxPooling2D, Reshape,
                                     UpSampling2D, GaussianNoise, Dense)
from tensorflow.keras.layers import concatenate

from tensorflow.keras.metrics import BinaryAccuracy, AUC, Precision, Recall

from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.optimizers.schedules import PolynomialDecay
from tensorflow_addons.losses import SigmoidFocalCrossEntropy

from metrics import Specificity, GeometricMeanMetric
from model.custom_layers import ClipByValue
from model.loop_net import AverageMetric
from util.constants import PATCH_SIZE
from model.BaseModel import BaseModel

class UNet(BaseModel):
    def __init__(self, model_name='GILoop', save_as='giloop', patch_size=PATCH_SIZE, avg_metric="geo_mean", patience=7):
        super().__init__(model_name, save_as, patch_size=patch_size, avg_metric=avg_metric, patience=patience)

    def build(self, show_summary=False): # Enhanced
        I = Input(shape=(self.patch_size, self.patch_size), name='patch')

        x = Reshape((self.patch_size, self.patch_size, 1))(I)
        x = GaussianNoise(0.05)(x)
        conv1 = Conv2D(32,
                       3,
                       activation='relu',
                       padding='same',
                       kernel_regularizer=tf.keras.regularizers.l2(0.000001))(x)
        conv1 = Conv2D(32,
                       3,
                       activation='relu',
                       padding='same',
                       kernel_regularizer=tf.keras.regularizers.l2(0.0001))(conv1)
        pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
        conv2 = Conv2D(64,
                       3,
                       activation='relu',
                       padding='same',
                       kernel_regularizer=tf.keras.regularizers.l2(0.0001))(pool1)
        conv2 = Conv2D(64,
                       3,
                       activation='relu',
                       padding='same',
                       kernel_regularizer=tf.keras.regularizers.l2(0.0001))(conv2)
        drop2 = Dropout(0.3)(conv2)
        pool2 = MaxPooling2D(pool_size=(2, 2))(drop2)
        conv3 = Conv2D(128,
                       3,
                       activation='relu',
                       padding='same',
                       kernel_regularizer=tf.keras.regularizers.l2(0.0001))(pool2)
        conv3 = Conv2D(128,
                       3,
                       activation='relu',
                       padding='same',
                       kernel_regularizer=tf.keras.regularizers.l2(0.0001))(conv3)
        drop3 = Dropout(0.3)(conv3)
        pool3 = MaxPooling2D(pool_size=(2, 2))(drop3)
        conv4 = Conv2D(256,
                       3,
                       activation='relu',
                       padding='same',
                       kernel_regularizer=tf.keras.regularizers.l2(0.0001))(pool3)
        conv4 = Conv2D(256,
                       3,
                       activation='relu',
                       padding='same',
                       kernel_regularizer=tf.keras.regularizers.l2(0.0001))(conv4)
        drop4 = Dropout(0.5)(conv4)
        pool4 = MaxPooling2D(pool_size=(2, 2))(drop4)
        conv5 = Conv2D(512,
                       3,
                       activation='relu',
                       padding='same',
                       kernel_regularizer=tf.keras.regularizers.l2(0.0001))(pool4)
        conv5 = Conv2D(512,
                       3,
                       activation='relu',
                       padding='same',
                       kernel_regularizer=tf.keras.regularizers.l2(0.0001))(conv5)
        drop5 = Dropout(0.5)(conv5)
        up6 = Conv2D(256,
                     2,
                     activation='relu',
                     padding='same',
                     kernel_regularizer=tf.keras.regularizers.l2(0.0001))(UpSampling2D(size=(2, 2))(drop5))
        merge6 = concatenate([drop4, up6], axis=3)
        conv6 = Conv2D(256,
                       3,
                       activation='relu',
                       padding='same',
                       kernel_regularizer=tf.keras.regularizers.l2(0.0001))(merge6)
        conv6 = Conv2D(256,
                       3,
                       activation='relu',
                       padding='same',
                       kernel_regularizer=tf.keras.regularizers.l2(0.0001))(conv6)

        up7 = Conv2D(128,
                     2,
                     activation='relu',
                     padding='same',
                     kernel_regularizer=tf.keras.regularizers.l2(0.0001))(UpSampling2D(size=(2, 2))(conv6))
        merge7 = concatenate([conv3, up7], axis=3)
        conv7 = Conv2D(128,
                       3,
                       activation='relu',
                       padding='same',
                       kernel_regularizer=tf.keras.regularizers.l2(0.0001))(merge7)
        conv7 = Conv2D(128,
                       3,
                       activation='relu',
                       padding='same',
                       kernel_regularizer=tf.keras.regularizers.l2(0.0001))(conv7)
        up8 = Conv2D(64,
                     2,
                     activation='relu',
                     padding='same',
                     kernel_regularizer=tf.keras.regularizers.l2(0.0001))(UpSampling2D(size=(2, 2))(conv7))
        merge8 = concatenate([conv2, up8], axis=3)
        conv8 = Conv2D(64,
                       3,
                       activation='relu',
                       padding='same',
                       kernel_regularizer=tf.keras.regularizers.l2(0.0001))(merge8)
        conv8 = Conv2D(64,
                       3,
                       activation='relu',
                       padding='same',
                       kernel_regularizer=tf.keras.regularizers.l2(0.0001))(conv8)
        up9 = Conv2D(32,
                     2,
                     activation='relu',
                     padding='same',
                     kernel_regularizer=tf.keras.regularizers.l2(0.0001))(UpSampling2D(size=(2, 2))(conv8))
        merge9 = concatenate([conv1, up9], axis=3)
        conv9 = Conv2D(32,
                       3,
                       activation='relu',
                       padding='same',
                       kernel_regularizer=tf.keras.regularizers.l2(0.0001))(merge9)
        conv9 = Conv2D(16,
                       3,
                       padding='same',
                       kernel_regularizer=tf.keras.regularizers.l2(0.0001))(conv9)
        conv10 = conv9

        x = Conv2D(64, (1, 1), activation='relu',
                   kernel_regularizer=tf.keras.regularizers.l2(0.0001))(conv10)
        x = Dropout(0.15)(x)
        x = Conv2D(32, (1, 1), activation='relu',
                   kernel_regularizer=tf.keras.regularizers.l2(0.0001))(x)
        x = Dropout(0.15)(x)
        cnn_logits = Conv2D(1, (1, 1), activation=None, name='logits',
                            kernel_regularizer=tf.keras.regularizers.l2(0.0001))(x)
        cnn_sigmoid = Activation('sigmoid')(cnn_logits)

        cnn_sigmoid = Reshape((self.patch_size * self.patch_size, 1))(cnn_sigmoid)

        model = Model(name="GILoop_CNN", inputs=[I], outputs=[cnn_sigmoid])

        # # Input shape: say (None, 10) for vectors of length 10
        # inp = layers.Input(shape=(mustache,))
        # out = layers.Lambda(lambda x: x)(inp)  # Identity mapping
        #
        # model = models.Model(inputs=inp, outputs=out)

        model.compile(
            loss=SigmoidFocalCrossEntropy(from_logits=False,
                                                    alpha=0.5,
                                                    gamma=1.2,
                                                    reduction=tf.keras.losses.Reduction.SUM_OVER_BATCH_SIZE),
            loss_weights=self.patch_size*self.patch_size,
            optimizer=Adam(learning_rate=PolynomialDecay(initial_learning_rate=0.0001,
                                                         decay_steps=2000 * 20,
                                                         end_learning_rate=0.00001,
                                                         power=2.0
                                                         )),
            metrics=[BinaryAccuracy(name='binary_accuracy', threshold=0.5),
                     AUC(curve="ROC", name='ROC_AUC'),
                     AUC(curve="PR", name='PR_AUC'),
                     Recall(),
                     Precision(),
                     Specificity(),
                     GeometricMeanMetric() if self.avg_metric == "geo_mean" else AverageMetric()]
        )

        if show_summary:
            model.summary()
        self.model = model

    # def build(self, show_summary=False):  # NO L2
    #     I = Input(shape=(self.patch_size, self.patch_size), name='patch')
    #
    #     x = Reshape((self.patch_size, self.patch_size, 1))(I)
    #     x = GaussianNoise(0.05)(x)
    #     conv1 = Conv2D(32,
    #                    3,
    #                    activation='relu',
    #                    padding='same')(x)
    #     conv1 = Conv2D(32,
    #                    3,
    #                    activation='relu',
    #                    padding='same')(conv1)
    #     pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    #     conv2 = Conv2D(64,
    #                    3,
    #                    activation='relu',
    #                    padding='same')(pool1)
    #     conv2 = Conv2D(64,
    #                    3,
    #                    activation='relu',
    #                    padding='same')(conv2)
    #     drop2 = Dropout(0.3)(conv2)
    #     pool2 = MaxPooling2D(pool_size=(2, 2))(drop2)
    #     conv3 = Conv2D(128,
    #                    3,
    #                    activation='relu',
    #                    padding='same')(pool2)
    #     conv3 = Conv2D(128,
    #                    3,
    #                    activation='relu',
    #                    padding='same')(conv3)
    #     drop3 = Dropout(0.3)(conv3)
    #     pool3 = MaxPooling2D(pool_size=(2, 2))(drop3)
    #     conv4 = Conv2D(256,
    #                    3,
    #                    activation='relu',
    #                    padding='same')(pool3)
    #     conv4 = Conv2D(256,
    #                    3,
    #                    activation='relu',
    #                    padding='same')(conv4)
    #     drop4 = Dropout(0.5)(conv4)
    #     pool4 = MaxPooling2D(pool_size=(2, 2))(drop4)
    #     conv5 = Conv2D(512,
    #                    3,
    #                    activation='relu',
    #                    padding='same')(pool4)
    #     conv5 = Conv2D(512,
    #                    3,
    #                    activation='relu',
    #                    padding='same')(conv5)
    #     drop5 = Dropout(0.5)(conv5)
    #     up6 = Conv2D(256,
    #                  2,
    #                  activation='relu',
    #                  padding='same')(UpSampling2D(size=(2, 2))(drop5))
    #     merge6 = concatenate([drop4, up6], axis=3)
    #     conv6 = Conv2D(256,
    #                    3,
    #                    activation='relu',
    #                    padding='same')(merge6)
    #     conv6 = Conv2D(256,
    #                    3,
    #                    activation='relu',
    #                    padding='same')(conv6)
    #
    #     up7 = Conv2D(128,
    #                  2,
    #                  activation='relu',
    #                  padding='same')(UpSampling2D(size=(2, 2))(conv6))
    #     merge7 = concatenate([conv3, up7], axis=3)
    #     conv7 = Conv2D(128,
    #                    3,
    #                    activation='relu',
    #                    padding='same')(merge7)
    #     conv7 = Conv2D(128,
    #                    3,
    #                    activation='relu',
    #                    padding='same')(conv7)
    #     up8 = Conv2D(64,
    #                  2,
    #                  activation='relu',
    #                  padding='same')(UpSampling2D(size=(2, 2))(conv7))
    #     merge8 = concatenate([conv2, up8], axis=3)
    #     conv8 = Conv2D(64,
    #                    3,
    #                    activation='relu',
    #                    padding='same')(merge8)
    #     conv8 = Conv2D(64,
    #                    3,
    #                    activation='relu',
    #                    padding='same')(conv8)
    #     up9 = Conv2D(32,
    #                  2,
    #                  activation='relu',
    #                  padding='same')(UpSampling2D(size=(2, 2))(conv8))
    #     merge9 = concatenate([conv1, up9], axis=3)
    #     conv9 = Conv2D(32,
    #                    3,
    #                    activation='relu',
    #                    padding='same')(merge9)
    #     conv9 = Conv2D(16,
    #                    3,
    #                    padding='same')(conv9)
    #     conv10 = conv9
    #
    #     x = Conv2D(64, (1, 1), activation='relu')(conv10)
    #     x = Dropout(0.15)(x)
    #     x = Conv2D(32, (1, 1), activation='relu')(x)
    #     x = Dropout(0.15)(x)
    #     cnn_logits = Conv2D(1, (1, 1), activation=None, name='logits')(x)
    #     cnn_sigmoid = Activation('sigmoid')(cnn_logits)
    #
    #     cnn_sigmoid = Reshape((self.patch_size * self.patch_size, 1))(cnn_sigmoid)
    #
    #     model = Model(name="GILoop_CNN", inputs=[I], outputs=[cnn_sigmoid])
    #
    #     model.compile(
    #         loss=SigmoidFocalCrossEntropy(from_logits=False,
    #                                       alpha=0.5,
    #                                       gamma=1.2,
    #                                       reduction=tf.keras.losses.Reduction.SUM_OVER_BATCH_SIZE),
    #         loss_weights=self.patch_size * self.patch_size,
    #         optimizer=Adam(learning_rate=PolynomialDecay(initial_learning_rate=0.0001,
    #                                                      decay_steps=2000 * 20,
    #                                                      end_learning_rate=0.00001,
    #                                                      power=2.0
    #                                                      )),
    #         metrics=[BinaryAccuracy(name='binary_accuracy', threshold=0.5),
    #                  AUC(curve="ROC", name='ROC_AUC'),
    #                  AUC(curve="PR", name='PR_AUC'),
    #                  Recall(),
    #                  Precision(),
    #                  Specificity(),
    #                  GeometricMeanMetric() if self.avg_metric == "geo_mean" else AverageMetric()]
    #     )
    #
    #     if show_summary:
    #         model.summary()
    #     self.model = model

    def build_original(self, image_upper_bound, show_summary=False):
        self.LOGGER.info(f'Building {self.model_name} model')
        I = Input(shape=(self.patch_size, self.patch_size), name='patch')

        x = ClipByValue(image_upper_bound)(I)
        x = Rescaling(1 / image_upper_bound)(x)

        x = Reshape((self.patch_size, self.patch_size, 1))(x)
        x = GaussianNoise(0.05)(x)
        conv1 = Conv2D(32,
                       3,
                       activation='relu',
                       padding='same',
                       kernel_regularizer=tf.keras.regularizers.l2(0.0001))(x)
        conv1 = Conv2D(32,
                       3,
                       activation='relu',
                       padding='same',
                       kernel_regularizer=tf.keras.regularizers.l2(0.0001))(conv1)
        pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
        conv2 = Conv2D(64,
                       3,
                       activation='relu',
                       padding='same',
                       kernel_regularizer=tf.keras.regularizers.l2(0.0001))(pool1)
        conv2 = Conv2D(64,
                       3,
                       activation='relu',
                       padding='same',
                       kernel_regularizer=tf.keras.regularizers.l2(0.0001))(conv2)
        drop2 = Dropout(0.3)(conv2)
        pool2 = MaxPooling2D(pool_size=(2, 2))(drop2)
        conv3 = Conv2D(128,
                       3,
                       activation='relu',
                       padding='same',
                       kernel_regularizer=tf.keras.regularizers.l2(0.0001))(pool2)
        conv3 = Conv2D(128,
                       3,
                       activation='relu',
                       padding='same',
                       kernel_regularizer=tf.keras.regularizers.l2(0.0001))(conv3)
        drop3 = Dropout(0.3)(conv3)
        pool3 = MaxPooling2D(pool_size=(2, 2))(drop3)
        conv4 = Conv2D(256,
                       3,
                       activation='relu',
                       padding='same',
                       kernel_regularizer=tf.keras.regularizers.l2(0.0001))(pool3)
        conv4 = Conv2D(256,
                       3,
                       activation='relu',
                       padding='same',
                       kernel_regularizer=tf.keras.regularizers.l2(0.0001))(conv4)
        drop4 = Dropout(0.5)(conv4)
        pool4 = MaxPooling2D(pool_size=(2, 2))(drop4)
        conv5 = Conv2D(512,
                       3,
                       activation='relu',
                       padding='same',
                       kernel_regularizer=tf.keras.regularizers.l2(0.0001))(pool4)
        conv5 = Conv2D(512,
                       3,
                       activation='relu',
                       padding='same',
                       kernel_regularizer=tf.keras.regularizers.l2(0.0001))(conv5)
        drop5 = Dropout(0.5)(conv5)
        up6 = Conv2D(256,
                     2,
                     activation='relu',
                     padding='same',
                     kernel_regularizer=tf.keras.regularizers.l2(0.0001))(UpSampling2D(size=(2, 2))(drop5))
        merge6 = concatenate([drop4, up6], axis=3)
        conv6 = Conv2D(256,
                       3,
                       activation='relu',
                       padding='same',
                       kernel_regularizer=tf.keras.regularizers.l2(0.0001))(merge6)
        conv6 = Conv2D(256,
                       3,
                       activation='relu',
                       padding='same',
                       kernel_regularizer=tf.keras.regularizers.l2(0.0001))(conv6)

        up7 = Conv2D(128,
                     2,
                     activation='relu',
                     padding='same',
                     kernel_regularizer=tf.keras.regularizers.l2(0.0001))(UpSampling2D(size=(2, 2))(conv6))
        merge7 = concatenate([conv3, up7], axis=3)
        conv7 = Conv2D(128,
                       3,
                       activation='relu',
                       padding='same',
                       kernel_regularizer=tf.keras.regularizers.l2(0.0001))(merge7)
        conv7 = Conv2D(128,
                       3,
                       activation='relu',
                       padding='same',
                       kernel_regularizer=tf.keras.regularizers.l2(0.0001))(conv7)
        up8 = Conv2D(64,
                     2,
                     activation='relu',
                     padding='same',
                     kernel_regularizer=tf.keras.regularizers.l2(0.0001))(UpSampling2D(size=(2, 2))(conv7))
        merge8 = concatenate([conv2, up8], axis=3)
        conv8 = Conv2D(64,
                       3,
                       activation='relu',
                       padding='same',
                       kernel_regularizer=tf.keras.regularizers.l2(0.0001))(merge8)
        conv8 = Conv2D(64,
                       3,
                       activation='relu',
                       padding='same',
                       kernel_regularizer=tf.keras.regularizers.l2(0.0001))(conv8)
        up9 = Conv2D(32,
                     2,
                     activation='relu',
                     padding='same',
                     kernel_regularizer=tf.keras.regularizers.l2(0.0001))(UpSampling2D(size=(2, 2))(conv8))
        merge9 = concatenate([conv1, up9], axis=3)
        conv9 = Conv2D(32,
                       3,
                       activation='relu',
                       padding='same',
                       kernel_regularizer=tf.keras.regularizers.l2(0.0001))(merge9)
        conv9 = Conv2D(16,
                       3,
                       padding='same',
                       kernel_regularizer=tf.keras.regularizers.l2(0.0001))(conv9)
        conv10 = conv9

        image_embedding = Reshape((self.patch_size*self.patch_size, -1), name='embedding')(conv10)
        image_decode = ReLU()(image_embedding)
        image_decode = Dropout(0.3)(image_decode)
        image_decode = Dense(32,
                             activation='relu',
                             kernel_regularizer=tf.keras.regularizers.l2(0.0001))(image_decode)
        image_decode = Dropout(0.3)(image_decode)
        image_decode = Dense(16,
                             activation='relu',
                             kernel_regularizer=tf.keras.regularizers.l2(0.0001))(image_decode)
        image_decode = Dropout(0.3)(image_decode)
        image_decode = Dense(8,
                             activation='relu',
                             kernel_regularizer=tf.keras.regularizers.l2(0.0001))(image_decode)
        image_decode = Dropout(0.3)(image_decode)
        cnn_logits = Dense(1,
                           name='logits',
                           kernel_regularizer=tf.keras.regularizers.l2(0.0001))(image_decode)
        cnn_sigmoid = Activation('sigmoid')(cnn_logits)

        model = Model(name="GILoop", inputs=[I], outputs=[cnn_sigmoid])
        model.compile(
            loss=
            SigmoidFocalCrossEntropy(from_logits=False,
                                     alpha=0.5,
                                     gamma=1.2,
                                     reduction=tf.keras.losses.Reduction.SUM_OVER_BATCH_SIZE),
            loss_weights=self.patch_size * self.patch_size,
            optimizer=Adam(learning_rate=PolynomialDecay(initial_learning_rate=0.001,
                                                         decay_steps=2000 * 20,
                                                         end_learning_rate=0.00005,
                                                         power=2.0
                                                         )),
            metrics=[BinaryAccuracy(name='binary_accuracy', threshold=0.5),
                     AUC(curve="ROC", name='ROC_AUC'),
                     AUC(curve="PR", name='PR_AUC'),
                     Recall(),
                     Precision(),
                     Specificity(),
                     GeometricMeanMetric()]
        )

        if show_summary:
            model.summary()

        self.model = model