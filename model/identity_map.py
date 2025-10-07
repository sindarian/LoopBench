# #!/usr/bin/env python3
# import tensorflow as tf
# from keras.layers import Rescaling, GlobalAveragePooling2D, Concatenate, Lambda, Flatten
# from tensorflow.keras.applications.efficientnet import preprocess_input
# from tensorflow.keras.layers import (Input, Dropout, ReLU, Activation, Conv2D, MaxPooling2D, Reshape,
#                                      UpSampling2D, GaussianNoise, Dense)
# from tensorflow.keras.layers import concatenate
#
# from tensorflow.keras.metrics import BinaryAccuracy, AUC, Precision, Recall
#
# from tensorflow.keras.models import Model
# from tensorflow.keras.optimizers import Adam
# from tensorflow.keras.optimizers.schedules import PolynomialDecay
# from tensorflow_addons.losses import SigmoidFocalCrossEntropy
#
# from metrics import Specificity, GeometricMeanMetric
# from model.loop_net import AverageMetric
# from util.constants import PATCH_SIZE
# from model.BaseModel import BaseModel
#
# class IDMap(BaseModel):
#     def __init__(self, model_name='Identity_Map', save_as='identity_map', patch_size=PATCH_SIZE, avg_metric="geo_mean", patience=7):
#         super().__init__(model_name, save_as, patch_size=patch_size, avg_metric=avg_metric, patience=patience)
#
#     def build(self, show_summary=False): # Enhanced
#         inp = Input(shape=(self.patch_size, self.patch_size), name='patch')
#         identity_out = Lambda(lambda x: x)(inp)  # Identity mapping
#         flattened_out = Flatten()(identity_out)  # Flatten to match y_batch_flat shape
#         reshaped_out = Reshape((-1, 1))(identity_out)
#         out = Activation('sigmoid')(flattened_out)
#
#         model = Model(name=self.model_name, inputs=inp, outputs=out)
#
#         model.compile(
#             loss=SigmoidFocalCrossEntropy(from_logits=False,
#                                                     alpha=0.5,
#                                                     gamma=1.2,
#                                                     reduction=tf.keras.losses.Reduction.SUM_OVER_BATCH_SIZE),
#             loss_weights=self.patch_size*self.patch_size,
#             # optimizer=Adam(learning_rate=PolynomialDecay(initial_learning_rate=0.0001,
#             #                                              decay_steps=2000 * 20,
#             #                                              end_learning_rate=0.00001,
#             #                                              power=2.0
#             #                                              )),
#             metrics=[
#                 BinaryAccuracy(name='binary_accuracy', threshold=0.5),
#                      AUC(curve="ROC", name='ROC_AUC'),
#                      AUC(curve="PR", name='PR_AUC'),
#                      Recall(),
#                      Precision(),
#                      Specificity(),
#                      GeometricMeanMetric() if self.avg_metric == "geo_mean" else AverageMetric()]
#         )
#
#         # if show_summary:
#         model.summary()
#         self.model = model

#!/usr/bin/env python3
import tensorflow as tf
from keras.layers import Rescaling, GlobalAveragePooling2D, Concatenate, Lambda, Flatten
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
from model.loop_net import AverageMetric
from util.constants import PATCH_SIZE
from model.BaseModel import BaseModel

class IDMap(BaseModel):
    def __init__(self, model_name='Identity_Map', save_as='identity_map', patch_size=PATCH_SIZE, avg_metric="geo_mean", patience=7):
        super().__init__(model_name, save_as, patch_size=patch_size, avg_metric=avg_metric, patience=patience)

    def build(self, show_summary=False):
        inp = Input(shape=(self.patch_size, self.patch_size), name='patch')
        identity_out = Lambda(lambda x: x)(inp)
        flattened_out = Flatten()(identity_out)
        # out = Activation('sigmoid')(flattened_out)

        model = Model(name=self.model_name, inputs=inp, outputs=flattened_out)

        model.compile(
            # loss=SigmoidFocalCrossEntropy(from_logits=False,
            #                                         alpha=0.5,
            #                                         gamma=1.2,
            #                                         reduction=tf.keras.losses.Reduction.SUM_OVER_BATCH_SIZE),
            # loss_weights=self.patch_size*self.patch_size,
            # optimizer=Adam(learning_rate=PolynomialDecay(initial_learning_rate=0.0001,
            #                                              decay_steps=2000 * 20,
            #                                              end_learning_rate=0.00001,
            #                                              power=2.0
            #                                              )),
            metrics=[
                BinaryAccuracy(name='binary_accuracy', threshold=0.5),
                     AUC(curve="ROC", name='ROC_AUC'),
                     AUC(curve="PR", name='PR_AUC'),
                     Recall(),
                     Precision(),
                     Specificity(),
                     GeometricMeanMetric() if self.avg_metric == "geo_mean" else AverageMetric()]
        )

        # if show_summary:
        model.summary()
        self.model = model
