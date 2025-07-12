from keras.layers import BatchNormalization
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import (Input, Dropout, Activation, Reshape, GaussianNoise, Rescaling, Conv2D,
                                     Concatenate, Flatten)
from tensorflow.keras.layers import concatenate
from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.keras.metrics import BinaryAccuracy, AUC
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.optimizers.schedules import PolynomialDecay
from tensorflow.python.keras.layers import UpSampling2D

from util.constants import FLATTENED_PATCH_SIZE
from custom_layers import ClipByValue
from model.BaseModel import BaseModel
from util.utils import PATCH_SIZE


class LoopNet(BaseModel):

    def __init__(self, model_name='LoopNet_resnet50'):
        super().__init__(model_name)

    def build(self, image_upper_bound):
        I = Input(shape=(PATCH_SIZE, PATCH_SIZE))
        x = ClipByValue(image_upper_bound)(I)
        x = Rescaling(1. / image_upper_bound)(x)
        x = Reshape((PATCH_SIZE, PATCH_SIZE, 1))(x)
        x = GaussianNoise(0.05)(x)
        x = Concatenate()([x, x, x]) # convert the grayscale input to 3 dim

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

            x = concatenate([x, skip])

        # final upsample to 224x224
        x = UpSampling2D((2, 2))(x)
        x = Conv2D(32, 3, padding='same', activation='relu')(x)
        x = BatchNormalization()(x)
        x = Conv2D(16, 3, padding='same')(x)
        x = BatchNormalization()(x)

        cnn_logits = Conv2D(1, 1, padding='same', name='logits')(x)
        cnn_sigmoid = Activation('sigmoid', name='sigmoid')(cnn_logits)
        cnn_sigmoid = Flatten()(cnn_sigmoid)

        model = Model(name="LoopNet_RestNet50", inputs=[I], outputs=[cnn_sigmoid])
        model.compile(
            optimizer=Adam(learning_rate=PolynomialDecay(initial_learning_rate=0.001,
                                                         decay_steps=2000 * 20,
                                                         end_learning_rate=0.00005,
                                                         power=2.0)
                           ),
            loss=BinaryCrossentropy(),
            loss_weights=FLATTENED_PATCH_SIZE,
            metrics=[BinaryAccuracy(name='binary_accuracy', threshold=0.5),
                     AUC(curve="ROC", name='ROC_AUC'),
                     AUC(curve="PR", name='PR_AUC')]
        )

        model.summary()
        self.model = model
