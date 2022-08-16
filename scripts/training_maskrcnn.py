import os
import numpy as np

# import sys
# import random
# import math
# import re
# import time
# import cv2
# import matplotlib
# import matplotlib.pyplot as plt

# os.environ["CUDA_VISIBLE_DEVICES"] = "1"
#
# import tensorflow as tf
# gpus = tf.config.experimental.list_physical_devices('GPU')
# for gpu in gpus:
#     tf.config.experimental.set_memory_growth(gpu, True)
# print(gpus)

# import tensorflow as tf
# gpu_options = tf.GPUOptions(allow_growth=True)
# session = tf.InteractiveSession(config=tf.ConfigProto(gpu_options=gpu_options))

# Import Mask RCNN
from mrcnn.config import Config
from mrcnn import utils
import mrcnn.model as modellib
from mrcnn import visualize
from mrcnn.model import log

import pandas as pd
import skimage
import warnings


def ellipse_interpolation(x, y, w, h, a, n_points=100):
    """
    Return a number n_points  of coordinates x,y for the ellipse of parameters x,y,w,h,a
    (a is in degrees)
    """
    lsp = np.linspace(0, 2 * np.pi, n_points)
    ell = np.array([w / 2 * np.cos(lsp), h / 2 * np.sin(lsp)])
    a_rad = a * np.pi / 180
    r_rot = np.array([[np.cos(a_rad), -np.sin(a_rad)], [np.sin(a_rad), np.cos(a_rad)]])
    rot = np.zeros((2, ell.shape[1]))
    for i in range(ell.shape[1]):
        rot[:, i] = np.dot(r_rot, ell[:, i])
    points_x, points_y = x + rot[0, :], y + rot[1, :]
    return points_x, points_y


class ShapesConfig(Config):
    """Configuration for training on the toy shapes dataset.
    Derives from the base Config class and overrides values specific
    to the toy shapes dataset.
    """
    # Give the configuration a recognizable name
    NAME = "shapes"

    # Train on 1 GPU and 8 images per GPU. We can put multiple images on each
    # GPU because the images are small. Batch size is 8 (GPUs * images/GPU).
    GPU_COUNT = 1
    IMAGES_PER_GPU = 8

    # Number of classes (including background)
    NUM_CLASSES = 1 + 1  # background + n

    # Use small images for faster training. Set the limits of the small side
    # the large side, and that determines the image shape.
    IMAGE_MIN_DIM = 448
    IMAGE_MAX_DIM = 448

    # Use smaller anchors because our image and objects are small
    RPN_ANCHOR_SCALES = (8, 16, 32, 64, 128)  # anchor side in pixels

    # Reduce training ROIs per image because the images are small and have
    # few objects. Aim to allow ROI sampling to pick 33% positive ROIs.
    TRAIN_ROIS_PER_IMAGE = 32

    # Use a small epoch since the data is simple
    STEPS_PER_EPOCH = 100

    # use small validation steps since the epoch is small
    VALIDATION_STEPS = 5


config = ShapesConfig()
config.display()


class CustomDataset(utils.Dataset):

    def load_custom(self, subset='train'):

        self.add_class("object", 1, "Berry")

        df_label = pd.read_csv('data/maskrcnn/{}_maskrcnn.csv'.format(subset))

        for image_name in os.listdir('data/maskrcnn/{}'.format(subset)):

          s_label = df_label[df_label['image_name'] == image_name]

          polygons = []
          for _, row in s_label.iterrows():
            pts_x, pts_y = ellipse_interpolation(row['ell_x'], row['ell_y'], row['ell_w'], row['ell_h'], row['ell_a'], n_points=100)
            polygons.append([pts_x, pts_y])

          self.add_image(
              "object",  ## for a single class just add the name here
              image_id=image_name,  # use file name as a unique image id
              path='data/maskrcnn/{}/'.format(subset) + image_name,
              width=448, height=448,
              polygons=polygons,
              num_ids=[1] * len(polygons)
              )
          print(image_name, len(s_label))

    def load_mask(self, image_id):
        """Generate instance masks for an image.
       Returns:
        masks: A bool array of shape [height, width, instance count] with
            one mask per instance.
        class_ids: a 1D array of class IDs of the instance masks.
        """

        # Convert polygons to a bitmap mask of shape
        # [height, width, instance_count]
        info = self.image_info[image_id]
        num_ids = info['num_ids']
        mask = np.zeros([info["height"], info["width"], len(info["polygons"])],
                        dtype=np.uint8)
        for i, polygone in enumerate(info["polygons"]):
          pts_x, pts_y = polygone
          rr, cc = skimage.draw.polygon(pts_y, pts_x)
          mask[rr, cc, i] = 1

        num_ids = np.array(num_ids, dtype=np.int32)
        return mask, num_ids

    def image_reference(self, image_id):
        """Return the path of the image."""
        info = self.image_info[image_id]
        return info["path"]


# Training dataset
dataset_train = CustomDataset()
dataset_train.load_custom(subset='train')
dataset_train.prepare()

dataset_val = CustomDataset()
dataset_val.load_custom(subset='valid')
dataset_val.prepare()

model = modellib.MaskRCNN(mode="training", config=config,
                          model_dir='data/maskrcnn/')

# # Which weights to start with?
# init_with = "imagenet"  # imagenet, coco, or last
#
# if init_with == "imagenet":
#     model.load_weights(model.get_imagenet_weights(), by_name=True)
# elif init_with == "coco":
#     # Load weights trained on MS COCO, but skip layers that
#     # are different due to the different number of classes
#     # See README for instructions to download the COCO weights
#     model.load_weights(COCO_MODEL_PATH, by_name=True,
#                        exclude=["mrcnn_class_logits", "mrcnn_bbox_fc",
#                                 "mrcnn_bbox", "mrcnn_mask"])
# elif init_with == "last":
#     # Load the last model you trained and continue training
#     model.load_weights(model.find_last(), by_name=True)

warnings.filterwarnings("ignore")

# # Passing layers="heads" freezes all layers except the head layers.
# model.train(dataset_train, dataset_val,
#             learning_rate=config.LEARNING_RATE,
#             epochs=1,
#             layers='heads')

# Fine tune all layers: passing layers="all" trains all layers
model.train(dataset_train, dataset_val,
            learning_rate=config.LEARNING_RATE / 10,
            epochs=50,
            layers="all")