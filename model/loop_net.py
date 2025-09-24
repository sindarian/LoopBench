from keras.layers import UpSampling2D
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import (Input, Dropout, Activation, Reshape, GaussianNoise, Conv2D,
                                     Concatenate)
from tensorflow.keras.layers import concatenate
from tensorflow.keras.metrics import BinaryAccuracy

from metrics import Specificity, AverageMetric
from util.constants import PATCH_SIZE
from model.BaseModel import BaseModel

from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.optimizers.schedules import PolynomialDecay
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.keras.metrics import AUC, Recall, Precision


class LoopNet(BaseModel):

    def __init__(self,
                 model_name: str = 'LoopNet',
                 save_as: str = 'loopnet',
                 patch_size: int = PATCH_SIZE):
        super().__init__(model_name, save_as, patch_size=patch_size)

    def build(self, show_summary=False):
        patch_input = Input(shape=(self.patch_size, self.patch_size), name='patch')

        x = Reshape((self.patch_size, self.patch_size, 1))(patch_input)
        x = GaussianNoise(0.05)(x)
        x = Concatenate()([x, x, x]) # convert the grayscale input to 3 dim

        base_model = ResNet50(include_top=False,
                              weights='imagenet',
                              input_tensor=x,
                              input_shape=(self.patch_size, self.patch_size, 3))

        # feature extraction layers from ResNet50 for U-Net skip connections
        skip_names = [
            "conv1_relu",
            "conv2_block3_out",
            "conv3_block4_out",
            "conv4_block6_out",
        ]
        skip_outputs = [base_model.get_layer(name).output for name in skip_names]
        x = base_model.get_layer("conv5_block3_out").output

        filters = [512, 256, 128, 64]

        for i in range(4):
            x = UpSampling2D((2, 2))(x)
            x = Conv2D(filters[i], 3, padding='same', activation='relu')(x)
            x = BatchNormalization()(x)
            x = Conv2D(filters[i], 3, padding='same', activation='relu')(x)
            x = BatchNormalization()(x)
            x = Dropout(0.3)(x)

            skip = skip_outputs[3 - i]

            x = concatenate([x, skip])

        x = UpSampling2D((2, 2))(x)
        x = Conv2D(32, 3, padding='same', activation='relu')(x)
        x = BatchNormalization()(x)
        x = Conv2D(16, 3, padding='same')(x)
        x = BatchNormalization()(x)

        cnn_logits = Conv2D(1, 1, padding='same')(x)
        cnn_sigmoid = Activation('sigmoid')(cnn_logits)
        cnn_sigmoid = Reshape((self.patch_size * self.patch_size, -1), name='embedding')(cnn_sigmoid)

        model = Model(name=self.model_name, inputs=[patch_input], outputs=[cnn_sigmoid])

        model.compile(
            optimizer=Adam(learning_rate=PolynomialDecay(initial_learning_rate=0.001,
                                                         decay_steps=2000 * 20,
                                                         end_learning_rate=0.00005,
                                                         power=2.0)),
            loss=BinaryCrossentropy(),
            loss_weights=self.patch_size*self.patch_size,
            metrics=[BinaryAccuracy(name='binary_accuracy', threshold=0.5),
                     AUC(curve="ROC", name='ROC_AUC'),
                     AUC(curve="PR", name='PR_AUC'),
                     Recall(),
                     Precision(),
                     Specificity(),
                     AverageMetric()
            ]
        )

        if show_summary:
            model.summary()
        self.model = model
