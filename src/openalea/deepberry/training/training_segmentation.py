"""
# Installation of the training environment with a GPU:
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
from tensorflow.keras import backend as K

# %env SM_FRAMEWORK=tf.keras  # might be necessary before importing segmentation_models on a server
import segmentation_models as sm


def load_dataset(path, indexes=None):
    """

    Parameters
    ----------
    path : str
        path containing couples of input (x) and output (y) images for training a segmentation model: 0x.png, 0y.png,
        1x.png, 1y.png, 2x.png, 2y.png, ...
    indexes : list
        allow to only load a subpart of the dataset (to avoid "out of memory" errors). For example, if 5 is contained
        in indexes, 5x.png and 5y.png will be loaded. Default: None (the entire dataset will be loaded).

    Returns
    -------

    """

    if indexes is None:
        indexes = np.unique([int(f[:-5]) for f in os.listdir(path)])

    # it's faster to create a full-zeros array at the beginning, already having the final desired size.
    X = np.zeros((len(indexes) * 1, 128, 128, 3))
    Y = np.zeros((len(indexes) * 1, 128, 128, 2))

    for i, index in enumerate(indexes):

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


def segmentation_training(x_train, y_train, x_valid, y_valid,
                          backbone='vgg16', weights=None, batch_size=16, lr=0.0001, epochs=50, output_path='model.h5'):

    n_classes = 2  # berry and non-berry
    optimizer = Adam(lr)
    # for other losses, see https://github.com/qubvel/segmentation_models/blob/master/segmentation_models/losses.py
    custom_loss = sm.losses.CategoricalCELoss()
    metric = dice_coef

    earlystopper = EarlyStopping(patience=5, verbose=1)
    checkpointer = ModelCheckpoint(output_path, verbose=1, save_best_only=True)

    model = sm.Unet(backbone_name=backbone, encoder_weights=weights, classes=n_classes, activation='softmax')

    model.compile(optimizer=optimizer, loss=custom_loss, metrics=[metric])

    _ = model.fit(x_train, y_train, validation_data=(x_valid, y_valid), batch_size=batch_size, epochs=epochs,
                  callbacks=[earlystopper, checkpointer])


if __name__ == '__main__':

    # os.environ["CUDA_VISIBLE_DEVICES"] = "1"

    # from tensorflow.config.experimental import list_physical_devices, set_memory_growth
    # gpus = list_physical_devices('GPU')
    # for gpu in gpus:
    #     set_memory_growth(gpu, True)

    PATH = '/mnt/data/benoit/dataset_seg/'  # server training
    # PATH = 'data/grapevine/dataset/dataset_seg/'  # local training (/!\ it might fail because of memory issues, and
    # be very slow without GPU)

    x_train, y_train = load_dataset(PATH + 'train/')
    x_valid, y_valid = load_dataset(PATH + 'valid/')

    segmentation_training(x_train=x_train, y_train=y_train, x_valid=x_valid, y_valid=y_valid,
                          output_path='segmentation_model.h5')

