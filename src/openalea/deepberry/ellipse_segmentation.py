""" Detection and segmentation of partially-hidden berry contours as ellipses, on a image of grapevine cluster """

import cv2
import numpy as np
import pandas as pd

# TODO : load segmentation model with opencv instead of keras ?
#  https://jeanvitor.com/tensorflow-object-detecion-opencv/
from tensorflow.keras.models import load_model

from openalea.deepberry.utils import nms, ellipse_interpolation

# These parameters are used to generate the training dataset and models. They can't be changed after for prediction.
VIGNETTE_SIZE_DET = 416  # Size of the square sub-image inputs for the detection model
VIGNETTE_SIZE_SEG = 128  # Size of the square sub-images inputs and outputs for the segmentation model
BERRY_SCALING_SEG = 0.75  # Ratio to standardise detection-output scaling before feeding them to the segmentation.


def berry_detection(image, model, score_threshold=0.89, max_box_size=150, ratio_threshold=2.5, nms_threshold=0.7):
    """
    Detection of bounding boxes around grapevine berries

    Parameters
    ----------
    image : 3D array
        image of a grapevine cluster
    model : cv2.dnn.DetectionModel
        object detection model, trained to detect berries on (VIGNETTE_SIZE_DET, VIGNETTE_SIZE_DET, 3) image inputs.
    max_box_size : int
        maximum expected value for the length of the boxes surrounding the berries, in pixels.
    score_threshold : float
        only the detected boxes with a confidence score above this threshold are saved. in [0, 1]. This is the most
        important parameter, it depends on the model used and should be reevaluated after training a new model.
    ratio_threshold : float
        only the detected boxes with a length/width ratio under this threshold are saved. >1.
    nms_threshold : float
        non-maximum suppression is used to avoid having detected boxes with an IoU above this threshold. in [0,1].

    Returns
    -------
    pandas.core.frame.DataFrame
        Each row corresponds to a predicted box, described by the following columns:
        "x", "y" : center coordinates
        "w" : width
        "h" : height
        "score" : confidence score
    """

    model.setInputParams(size=(VIGNETTE_SIZE_DET, VIGNETTE_SIZE_DET), scale=1 / 255, swapRB=False)

    px_spacing = VIGNETTE_SIZE_DET - max_box_size

    Y = list(np.arange(0, image.shape[0] - VIGNETTE_SIZE_DET, px_spacing)) + [image.shape[0] - VIGNETTE_SIZE_DET]
    X = list(np.arange(0, image.shape[1] - VIGNETTE_SIZE_DET, px_spacing)) + [image.shape[1] - VIGNETTE_SIZE_DET]
    res = []
    for y_corner in Y:
        for x_corner in X:
            vignette = image[y_corner:(y_corner + VIGNETTE_SIZE_DET), x_corner:(x_corner + VIGNETTE_SIZE_DET)]
            _, scores, boxes = model.detect(vignette, score_threshold, nms_threshold)
            for score, box in zip(scores, boxes):
                (x, y, w, h) = box
                if max((w, h)) / min((w, h)) < ratio_threshold:
                    res.append([x + x_corner, y + y_corner, w, h, score])
    res = pd.DataFrame(res, columns=['x', 'y', 'w', 'h', 'score'])

    # non-maximum suppression
    scores = np.array(res['score'])
    polygons = []
    for _, row in res.iterrows():
        x, y, w, h = row[['x', 'y', 'w', 'h']]
        polygons.append(np.array([[x, x, x + w, x + w, x], [y, y + h, y + h, y, y]]).T)
    to_keep = nms(polygons=polygons, scores=scores, threshold=nms_threshold)
    res = res.iloc[to_keep]

    return res


def berry_segmentation(image, model, boxes, nms_threshold_ell=0.7):
    """
    Ellipse-segmentation of grapevine berries

    Parameters
    ----------
    image : 3D array
        image of a grapevine cluster
    model : keras model
        segmentation model, trained to segment berries on (VIGNETTE_SIZE_SEG, VIGNETTE_SIZE_SEG, 3) image inputs, with
        a berry rescaling ratio equal to BERRY_SCALING_SEG
    boxes : pandas.core.frame.DataFrame
        output from berry_detection function applied on the same image
    nms_threshold_ell : float. in [0, 1].
        non-maximum suppression is used to avoid having segmented ellipses with an IoU above this threshold. in [0,1].

    Returns
    -------
    pandas.core.frame.DataFrame
        Each row corresponds to a segmented ellipse, described by the following columns:
        "score" : confidence score (for the detection step, this value was already contained in "boxes" input)
        "ell_x", "ell_y" : ellipse center coordinates
        "ell_h" : length ellipse major axis
        "ell_w" : length of ellipse minor axis
        "ell_a" : ellipse rotation angle
    """

    ds = int(VIGNETTE_SIZE_SEG / 2)

    res = []

    seg_vignettes = {}
    for row_index, row in boxes.iterrows():

        x, y, w, h, score = row[['x', 'y', 'w', 'h', 'score']]
        x_vignette, y_vignette = round(x + (w / 2)), round(y + (h / 2))
        zoom = (VIGNETTE_SIZE_SEG * BERRY_SCALING_SEG) / np.max((w, h))

        ya, yb = int(round(y_vignette - (ds / zoom))), int(round(y_vignette + (ds / zoom)))
        xa, xb = int(round(x_vignette - (ds / zoom))), int(round(x_vignette + (ds / zoom)))

        # check if enough space to unzoom in case of big berry
        enough_space = (0 <= ya) and (yb < image.shape[0]) and (0 <= xa) and (xb <= image.shape[1])

        if enough_space:
            seg_vignette = cv2.resize(image[ya:yb, xa:xb], (VIGNETTE_SIZE_SEG, VIGNETTE_SIZE_SEG))
            seg_vignette = (seg_vignette / 255.).astype(np.float64)
            seg_vignettes[row_index] = seg_vignette

    if seg_vignettes:

        # segmentation mask prediction (all vignettes at once ~= 2x faster)
        multi_seg = model.predict(np.array(list(seg_vignettes.values())), verbose=0)
        multi_seg = multi_seg[:, :, :, 0]
        multi_seg = (multi_seg > 0.5).astype(np.uint8) * 255  # important to get correct contours !

        for row_index, seg in zip(seg_vignettes.keys(), multi_seg):

            # again
            x, y, w, h, score = boxes.loc[row_index][['x', 'y', 'w', 'h', 'score']]
            x_vignette, y_vignette = round(x + (w / 2)), round(y + (h / 2))
            zoom = (VIGNETTE_SIZE_SEG * BERRY_SCALING_SEG) / np.max((w, h))

            # extraction of the mask edges
            edges, _ = cv2.findContours(seg, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            if edges:
                edges = edges[np.argmax([len(e) for e in edges])][:, 0, :]  # takes longest contour if several areas

            if len(edges) >= 5:  # cv2.fitEllipse() requires >= 5 points

                # fit ellipse in the zoomed / centered vignette space
                (ell_x, ell_y), (ell_w, ell_h), ell_a = cv2.fitEllipse(edges)

                # rescale to raw image space
                ell_x_raw, ell_y_raw = x_vignette + ((ell_x - ds) / zoom), y_vignette + ((ell_y - ds) / zoom)
                ell_w_raw, ell_h_raw = ell_w / zoom, ell_h / zoom
                ell_a_raw = ell_a

                res.append([ell_x_raw, ell_y_raw, ell_w_raw, ell_h_raw, ell_a_raw, score])

    res = pd.DataFrame(res, columns=['ell_x', 'ell_y', 'ell_w', 'ell_h', 'ell_a', 'score'])

    # non-maximum suppression
    scores = np.array(res['score'])
    polygons = []
    for _, row in res.iterrows():
        xe, ye, we, he, ae = row[['ell_x', 'ell_y', 'ell_w', 'ell_h', 'ell_a']]
        polygons.append(ellipse_interpolation(x=xe, y=ye, w=we, h=he, a=ae, n_points=30).T)
    to_keep = nms(polygons=polygons, scores=scores, threshold=nms_threshold_ell)
    res = res.iloc[to_keep]

    return res


def load_berry_models(dir):
    """
    Load the detection and segmentation models

    Parameters
    ----------
    dir : str
        directory containing two files for the detection model (detection.cfg, detection.weights) and one file for the
        segmentation model (segmentation.h5)

    Returns
    -------
    (cv2.dnn.DetectionModel, keras_model)
    """

    # yolov4 object detection model (https://github.com/AlexeyAB/darknet)
    weights_path = dir + '/detection.weights'
    config_path = dir + '/detection.cfg'
    net = cv2.dnn.readNetFromDarknet(config_path, weights_path)
    model_det = cv2.dnn_DetectionModel(net)

    # U-net segmentation model
    model_seg = load_model(dir + '/segmentation.h5', custom_objects={'dice_coef': None}, compile=False)

    return model_det, model_seg
