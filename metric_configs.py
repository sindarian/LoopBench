from tensorflow.keras.metrics import BinaryAccuracy, AUC
from tensorflow.keras.optimizers.schedules import PolynomialDecay
from tensorflow_addons.losses import SigmoidFocalCrossEntropy
from tensorflow.keras.optimizers import Adam
import tensorflow as tf
from utils import PATCH_SIZE

OPTIMIZER = Adam(learning_rate=PolynomialDecay(initial_learning_rate=0.001,
                                                     decay_steps=2000 * 20,
                                                     end_learning_rate=0.00005,
                                                     power=2.0))
LOSS_WEIGHTS = {'sigmoid': PATCH_SIZE * PATCH_SIZE}
LOSS = {'sigmoid': SigmoidFocalCrossEntropy(from_logits=False,
                                            alpha=0.98,
                                            gamma=2.0,
                                            reduction=tf.keras.losses.Reduction.SUM_OVER_BATCH_SIZE) # average total loss over the batch size
        }
METRICS = {'sigmoid': [BinaryAccuracy(name='binary_accuracy', threshold=0.5),
                             AUC(curve="ROC", name='ROC_AUC'),
                             AUC(curve="PR", name='PR_AUC')
                       ]
           }
# use validation AUC of precision-recall for stopping
EARLY_STOP = tf.keras.callbacks.EarlyStopping(monitor='val_PR_AUC',
                                              min_delta=0.0001,
                                              patience=5,
                                              verbose=1,
                                              mode='max',
                                              restore_best_weights=True)

efficient_net_hyperparams = {
    'base_model': {
        'include_top': True
    },
    'loss': {
        'alpha': 0.9,
        'gamma': 1.5,
        'reduction': tf.keras.losses.Reduction.SUM_OVER_BATCH_SIZE
    },
    'optimizer': {
        'lr': 0.001,
        'decay': 2000 * 20,
        'end_lr': 0.00005,
        'power': 2.0
    }
}

resnet_hyperparams = {
    'base_model': {
        'include_top': False
    },
    'loss': {
        'alpha': 0.9,#0.5,
        'gamma': 1.2,
        'reduction': tf.keras.losses.Reduction.SUM_OVER_BATCH_SIZE
    },
    'optimizer': {
        'lr': 0.001,
        'decay': 2000 * 20,
        'end_lr': 0.00005,
        'power': 2.0
    }
}