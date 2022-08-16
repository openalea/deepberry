import os
import cv2
import numpy as np

#import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

from tensorflow.keras.losses import categorical_crossentropy
from tensorflow.keras import backend as K
from tensorflow.keras.optimizers import Adam
import segmentation_models as sm

os.environ["CUDA_VISIBLE_DEVICES"]="1"

data_path = 'data/segmentation/train/'
indexes = np.unique([int(f[:-5]) for f in os.listdir(data_path)])

train_indexes = np.random.choice(indexes, int(0.8 * len(indexes)), replace=False)[::100]

X, Y = [], []

for i in train_indexes:

    x = cv2.cvtColor(cv2.imread(data_path + '{}x.png'.format(i)), cv2.COLOR_BGR2RGB)
    y = cv2.cvtColor(cv2.imread(data_path + '{}y.png'.format(i)), cv2.COLOR_BGR2RGB)

    for do_rotate in [False, True]:
        for do_flip in np.random.choice(['noflip', 0, 1, -1], 2, replace=False):

            x2, y2 = x.copy(), y.copy()
            if do_rotate:
                x2 = cv2.rotate(x2, cv2.ROTATE_90_CLOCKWISE)
                y2 = cv2.rotate(y2, cv2.ROTATE_90_CLOCKWISE)

            if do_flip != 'noflip':
                x2 = cv2.flip(x2, int(do_flip))
                y2 = cv2.flip(y2, int(do_flip))

            x2 = (x2 / 255.).astype(np.float64)
            y2 = (y2[:, :, 0] / 255.)[..., np.newaxis]
            y2 = np.concatenate((y2, 1 - y2), axis=-1)

            X.append(x2)
            Y.append(y2)

X = np.array(X)
Y = np.array(Y)

# ==============================================================================================================

BACKBONE = 'vgg16'
n_classes = 2
activation = 'softmax'
weights = None # 'imagenet'

model = sm.Unet(backbone_name=BACKBONE, encoder_weights=weights, classes=n_classes, activation=activation)


def dice_coef(y_true, y_pred, smooth=1e-3):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return K.mean((2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth))

BATCH_SIZE = 16
LR = 0.0001
EPOCHS = 100
OPTIMIZER = Adam(LR)
custom_loss = sm.losses.CategoricalCELoss()
#metric = sm.metrics.IOUScore
metric = dice_coef

# Model compilation
model.compile(optimizer=OPTIMIZER, loss=custom_loss, metrics=[metric])

# Train model
earlystopper = EarlyStopping(patience=5, verbose=1)
###checkpointer = ModelCheckpoint('UNET_VGG16_RGB.h5', verbose=1, save_best_only=True)
# si on precise pas le chemin ou on a monté le drive (ici /content/ggdrive/), la sauvegarde marchera, mais le fichier h5 sera pas visible en se baladant dans le Drive ! :
checkpointer = ModelCheckpoint('/data/UNET_VGG16_RGB.h5', verbose=1, save_best_only=True)
history = model.fit(X, Y, validation_split=0.2, batch_size=BATCH_SIZE, epochs=EPOCHS, callbacks=[earlystopper, checkpointer], verbose=1)





























