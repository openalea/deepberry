import cv2
import numpy as np
import pandas as pd

# TODO : load unet model with opencv instead of keras ? using https://jeanvitor.com/tensorflow-object-detecion-opencv/
from tensorflow.keras.models import load_model

from deepberry.src.openalea.deepberry.utils import nms

# These parameters are used to generate the training dataset and models. They can't be changed after for prediction.
VIGNETTE_SIZE_DET = 416
VIGNETTE_SIZE_SEG = 128
BERRY_SCALING_SEG = 0.75


def berry_detection(image, model, max_box_size=150, score_threshold=0.985, ratio_threshold=2.5, nms_threshold=0.7):
    """
    max_box_size : maximum box height or width in the dataset
    nms_threshold : nms is never higher than 0.7 in the ground-truth annotated dataset
    ratio_threshold: in ground-truth dataset; 0.5% of values > 2., 0.03% > 2.5
    score_threshold: needs to be re-evaluated if using a new model
    """

    px_spacing = VIGNETTE_SIZE_DET - max_box_size

    model.setInputParams(size=(VIGNETTE_SIZE_DET, VIGNETTE_SIZE_DET), scale=1 / 255, swapRB=False)

    Y = list(np.arange(0, image.shape[0] - VIGNETTE_SIZE_DET, px_spacing)) + [image.shape[0] - VIGNETTE_SIZE_DET]
    X = list(np.arange(0, image.shape[1] - VIGNETTE_SIZE_DET, px_spacing)) + [image.shape[1] - VIGNETTE_SIZE_DET]
    res = []
    for y_corner in Y:
        for x_corner in X:
            vignette = image[y_corner:(y_corner + VIGNETTE_SIZE_DET), x_corner:(x_corner + VIGNETTE_SIZE_DET)]
            classes, scores, boxes = model.detect(vignette, score_threshold, nms_threshold)
            for score, box in zip(scores, boxes):
                (x, y, w, h) = box
                if max((w, h)) / min((w, h)) < ratio_threshold:  # check if box has normal length/width ratio
                    res.append([x + x_corner, y + y_corner, w, h, score])
    res = pd.DataFrame(res, columns=['x', 'y', 'w', 'h', 'score'])

    res = nms(res, nms_threshold=nms_threshold, ellipse=False)

    return res


def berry_segmentation(image, model, boxes, nms_threshold_ell=0.7):

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

    res = nms(res, nms_threshold=nms_threshold_ell, ellipse=True)

    return res


def load_berry_models(path):

    # yolov4 object detection model
    weights_path = path + '/detection.weights'
    config_path = path + '/detection.cfg'
    net = cv2.dnn.readNetFromDarknet(config_path, weights_path)
    model_det = cv2.dnn_DetectionModel(net)

    # U-net segmentation model
    model_seg = load_model(path + '/segmentation.h5', custom_objects={'dice_coef': None}, compile=False)

    return model_det, model_seg

