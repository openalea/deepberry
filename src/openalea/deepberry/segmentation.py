import numpy as np
import cv2
import pandas as pd

from deepberry.src.openalea.deepberry.utils import ellipse_interpolation, nms
#from openalea.deepberry.utils import ellipse_interpolation, nms

from keras.models import load_model
from keras import backend as K


def detect_berry(image, model, vignette_size=(416, 416), px_spacing=270,
                 score_threshold=0.95, ratio_threshold=2.5, nms_threshold=0.7):
    """
    px_spacing = 270 ~= 416 - 150 (150 = max box width of a berry)
    nms_threshold : nms is never higher than 0.7 in the ground-truth annotated dataset
    ratio_threshold: in ground-truth dataset; 0.5% of values > 2., 0.03% > 2.5
    """

    model.setInputParams(size=vignette_size, scale=1 / 255, swapRB=False)

    h_vignette, w_vignette = vignette_size

    Y = list(np.arange(0, image.shape[0] - h_vignette, px_spacing)) + [image.shape[0] - h_vignette]
    X = list(np.arange(0, image.shape[1] - w_vignette, px_spacing)) + [image.shape[1] - w_vignette]
    res = []
    for y_corner in Y:
        for x_corner in X:
            vignette = image[y_corner:(y_corner + h_vignette), x_corner:(x_corner + w_vignette)]
            classes, scores, boxes = model.detect(vignette, score_threshold, nms_threshold)
            for score, box in zip(scores, boxes):
                (x, y, w, h) = box
                if max((w, h)) / min((w, h)) < ratio_threshold:  # check if box has normal length/width ratio
                    res.append([x + x_corner, y + y_corner, w, h, score])
    res = pd.DataFrame(res, columns=['x', 'y', 'w', 'h', 'score'])

    res_nms = nms(res, nms_threshold=nms_threshold)

    return res_nms


def segment_berry(image, model, boxes, dim_ratio_threshold=2., seg_size=128, score_threshold=0.95):

    res = []
    res_ellipse = {'raw': [], 'fit': []}

    # filter box detections
    boxes_filter = boxes.copy()
    for index, row in boxes_filter.iterrows():
        x, y, w, h, score = row[['x', 'y', 'w', 'h', 'score']]
        remove_row = False
        # check if box has low prediction
        if score < score_threshold:
            remove_row = True
        # check if box has abnormal length/width ratio
        if max((w, h)) / min((w, h)) > dim_ratio_threshold:
            remove_row = True
        # check if box is too close from image border
        x_center, y_center = round(x + w / 2), round(y + h / 2)
        if not (int(seg_size / 2) < y_center < image.shape[0] - int(seg_size / 2)) or \
                not (int(seg_size / 2) < x_center < image.shape[1] - int(seg_size / 2)):  # TODO
            remove_row = True
        # remove box if one of the previous conditions is met
        if remove_row:
            boxes_filter.drop(index, inplace=True)

    seg_vignettes = []
    for _, row in boxes_filter.iterrows():
        x, y, w, h, score = row[['x', 'y', 'w', 'h', 'score']]
        x_center, y_center = round(x + w / 2), round(y + h / 2)

        # crop vignette
        seg_vignette = image[(int(y_center) - int(seg_size / 2)):(int(y_center) + int(seg_size / 2)),
                       (int(x_center) - int(seg_size / 2)):(int(x_center) + int(seg_size / 2))]
        seg_vignette = (seg_vignette / 255.).astype(np.float64)

        seg_vignettes.append(seg_vignette)

    # segmentation mask prediction (all vignettes at once =~ 2x faster)
    multi_seg = model.predict(np.array(seg_vignettes), verbose=0)
    multi_seg = multi_seg[:, :, :, 0]
    multi_seg = (multi_seg > 0.5).astype(np.uint8) * 255  # important to get correct contours !

    for (_, row), seg in zip(boxes_filter.iterrows(), multi_seg):
        x, y, w, h, score = row[['x', 'y', 'w', 'h', 'score']]
        x_center, y_center = round(x + w / 2), round(y + h / 2)

        # extraction of the mask edges
        edges, _ = cv2.findContours(seg, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        if edges:
            edges = edges[np.argmax([len(e) for e in edges])][:, 0, :]  # takes longest contour if several areas

        if len(edges) >= 5:  # cv2.fitEllipse() requires >= 5 points
            # convert to whole image space
            edges_img = edges + np.array([x_center - int(seg_size / 2), y_center - int(seg_size / 2)])
            # fit ellipse
            (ell_x, ell_y), (ell_w, ell_h), ell_a = cv2.fitEllipse(edges_img)
            res.append([ell_x, ell_y, ell_w, ell_h, ell_a, score])

            """ just for visu """
            res_ellipse['raw'].append(edges_img)
            lsp_x, lsp_y = ellipse_interpolation(x=ell_x, y=ell_y, w=ell_w, h=ell_h, a=ell_a, n_points=100)
            res_ellipse['fit'].append(np.array([lsp_x, lsp_y]).T)

    res = pd.DataFrame(res, columns=['ell_x', 'ell_y', 'ell_w', 'ell_h', 'ell_a', 'score'])

    # TODO : remove res_ellipse
    return res, res_ellipse


def segment_berry_scaled(image, model, boxes, seg_size=128):

    res = []
    res_ellipse = {'raw': [], 'fit': []}

    scale_factor = 0.75

    seg_vignettes = {}
    for row_index, row in boxes.iterrows():

        x, y, w, h, score = row[['x', 'y', 'w', 'h', 'score']]
        x_vignette, y_vignette = round(x + (w / 2)), round(y + (h / 2))
        zoom = (seg_size * scale_factor) / np.max((w, h))

        ds = int(seg_size / 2)
        ya, yb = int(y_vignette - round(ds / zoom)), int(y_vignette + round(ds / zoom))
        xa, xb = int(x_vignette - round(ds / zoom)), int(x_vignette + round(ds / zoom))

        # check if box is not too close from image border
        condition1 = (ds < y_vignette < image.shape[0] - ds) and (ds < x_vignette < image.shape[1] - ds)
        # check if enough space to unzoom in case of big berry
        condition2 = (0 <= ya) and (yb < image.shape[0]) and (0 <= xa) and (xb <= image.shape[1])

        if condition1 and condition2:
            seg_vignette = cv2.resize(image[ya:yb, xa:xb], (128, 128))
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
            zoom = (seg_size * scale_factor) / np.max((w, h))

            # extraction of the mask edges
            edges, _ = cv2.findContours(seg, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            if edges:
                edges = edges[np.argmax([len(e) for e in edges])][:, 0, :]  # takes longest contour if several areas

            if len(edges) >= 5:  # cv2.fitEllipse() requires >= 5 points

                edges_img = edges + np.array([x_vignette - int(seg_size / 2), y_vignette - int(seg_size / 2)])
                # fit ellipse
                (ell_x, ell_y), (ell_w, ell_h), ell_a = cv2.fitEllipse(edges_img)
                ell_w /= zoom
                ell_h /= zoom
                res.append([ell_x, ell_y, ell_w, ell_h, ell_a, score])

                """ just for visu """
                res_ellipse['raw'].append(edges_img)
                lsp_x, lsp_y = ellipse_interpolation(x=ell_x, y=ell_y, w=ell_w, h=ell_h, a=ell_a, n_points=100)
                res_ellipse['fit'].append(np.array([lsp_x, lsp_y]).T)

    res = pd.DataFrame(res, columns=['ell_x', 'ell_y', 'ell_w', 'ell_h', 'ell_a', 'score'])

    # TODO : remove res_ellipse
    return res, res_ellipse


def dice_coef(y_true, y_pred, smooth=1e-3):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return K.mean((2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth))


def load_models_berry(path):

    # yolov4 object detection model
    weights_path = path + '/detection.weights'
    config_path = path + '/detection.cfg'
    net = cv2.dnn.readNetFromDarknet(config_path, weights_path)
    model_det = cv2.dnn_DetectionModel(net)

    # U-net segmentation model
    model_seg = load_model(path + '/segmentation.h5', custom_objects={'dice_coef': dice_coef})

    return model_det, model_seg

# ===============================================================================================================


#MODEL_SEG = load_model('data/grapevine/UNET_VGG16_RGB_normal.h5', custom_objects={'dice_coef': dice_coef})


if __name__ == '__main__':

    import matplotlib.pyplot as plt

    # ===== run prediction ======================================================================================

    # img_folder = 'data/grapevine/grapevine22/'
    # img_folder = 'data/grapevine/rgb/rotation2/'
    # img_folder = 'data/grapevine/rgb/326/'
    # img_folder = ''
    # files = ['data/grapevine/non_elliptique.png']
    # files = [f for f in os.listdir(img_folder) if 'png' in f]
    # files.sort(key=lambda f: int(f.split('_')[0]))

    path = 'V:/DYN2020-05-15/2488/d715d002-1c05-4d7e-b6b2-581aca86d88f.png'

    img = cv2.imread(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # ===== YOLOv4 berry detection ================================================================================

    res_det = detect_berry(image=img, model=MODEL_DET, score_threshold=0.5)

    plt.figure()
    # plt.xlim((1000, 1800))
    # plt.ylim((1800, 1000))
    plt.imshow(img)
    for _, row in res_det.iterrows():
        x, y, w, h, score = row
        if score > 0.5:
            plt.plot([x, x, x + w, x + w, x], [y, y + h, y + h, y, y], 'r-')

    # ===== U-NET berry segmentation ==============================================================================

    plt.figure(path)
    plt.imshow(img)
    for _, box in res_det[res_det['score'] > 0.95].iterrows():
        x, y, w, h = box[['x', 'y', 'w', 'h']]
        plt.plot([x, x, x + w, x + w, x], [y, y + h, y + h, y, y], 'b-')

    for col, model_seg in zip(['orange', 'red'], [MODEL_SEG, MODEL_SEG_SCALED]):
        res, res_ellipse = segment_berry_scaled(image=img, model=model_seg, boxes=res_det)

        # for ell in res_ellipse['raw']:
        #     ell = np.concatenate((ell, np.array([ell[-0]])))
        #     plt.plot(ell[:, 0], ell[:, 1], 'b-', linewidth=0.4)
        for ell in res_ellipse['fit']:
            ell = np.concatenate((ell, np.array([ell[-0]])))
            plt.plot(ell[:, 0], ell[:, 1], '-', color=col, linewidth=0.5)

    # plt.savefig('data/grapevine/test.png.png'.format(k_file), dpi=300)
    # plt.close('all')

    # ===== mask for image registration ===========================================================================

    mask = np.float32(img * 0)

    for _, row in res.iterrows():

        mask = cv2.ellipse(mask,
                    (round(row['ell_x']), round(row['ell_y'])),
                    (round(row['ell_w'] / 2 * 1.), round(row['ell_h'] / 2 * 1.)),
                    row['ell_a'], 0., 360,
                    (1, 1, 1), -1)

    plt.imshow((img * mask).astype(np.uint8))

