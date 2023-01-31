"""
    * Installation of the training environment on a server with a GPU:
conda install tensorflow-gpu=1.14
conda install python=3.7
conda install ipython
pip install -U segmentation-models
conda install opencv
"""

import os
import cv2
import numpy as np

from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
# from tensorflow.keras.losses import categorical_crossentropy
from tensorflow.keras import backend as K

%env SM_FRAMEWORK=tf.keras  # to use before importing segmentation_models
import segmentation_models as sm

# os.environ["CUDA_VISIBLE_DEVICES"] = "1"

PATH = '/mnt/data/benoit/dataset_seg/'

# import tensorflow as tf
# gpus = tf.config.experimental.list_physical_devices('GPU')
# for gpu in gpus:
#     tf.config.experimental.set_memory_growth(gpu, True)


def load_dataset(path, indexes=None):

  if indexes is None:
    indexes = np.unique([int(f[:-5]) for f in os.listdir(path)])

  X = np.zeros((len(indexes) * 1, 128, 128, 3))
  Y = np.zeros((len(indexes) * 1, 128, 128, 2))

  for i, index in enumerate(indexes):
      if i % 1000 == 0:
          print(i)

      x = cv2.cvtColor(cv2.imread(path + '{}x.png'.format(index)), cv2.COLOR_BGR2RGB)
      y = cv2.imread(path + '{}y.png'.format(index), 0)  # 0 to read as grey scale

      x2 = (x / 255.).astype(np.float64)
      y2 = (y / 255.)[..., np.newaxis]
      y2 = np.concatenate((y2, 1 - y2), axis=-1)

      X[i] = x2
      Y[i] = y2

  return X, Y


def dice_coef(y_true, y_pred, smooth=1e-3):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return K.mean((2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth))

X_train, Y_train = load_dataset(PATH + 'train/')
X_valid, Y_valid = load_dataset(PATH + 'valid/')

# path = 'train/'
# subset_size = 3000
# all_indexes = np.unique([int(f[:-5]) for f in os.listdir(path)])
# np.random.shuffle(all_indexes)
# indexes_batches = [all_indexes[(k * subset_size):((k + 1) * subset_size)] for k in range(int(len(all_indexes) / subset_size) + 1)]

BACKBONE = 'vgg16'
n_classes = 2
activation = 'softmax'
weights = None  # 'imagenet'

# https://github.com/qubvel/segmentation_models/blob/master/segmentation_models/losses.py

BATCH_SIZE = 16
LR = 0.0001
EPOCHS = 10
OPTIMIZER = Adam(LR)
custom_loss = sm.losses.CategoricalCELoss()
#metric = sm.metrics.IOUScore
metric = dice_coef

earlystopper = EarlyStopping(patience=5, verbose=1)
checkpointer = ModelCheckpoint(PATH + 'UNET_deepberry.h5', verbose=1, save_best_only=True)

model = sm.Unet(backbone_name=BACKBONE, encoder_weights=weights, classes=n_classes, activation=activation)

model.compile(optimizer=OPTIMIZER, loss=custom_loss, metrics=[metric])

history = model.fit(X_train, Y_train, validation_data=(X_valid, Y_valid), batch_size=BATCH_SIZE, epochs=EPOCHS
                    ,callbacks=[earlystopper, checkpointer])

# model.save(PATH + 'model.h5')