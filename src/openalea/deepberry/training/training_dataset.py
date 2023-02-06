import json

import cv2
import numpy as np
import pandas as pd

from deepberry.src.openalea.deepberry.ellipse_segmentation import VIGNETTE_SIZE_DET, VIGNETTE_SIZE_SEG, \
    BERRY_SCALING_SEG
from deepberry.src.openalea.deepberry.utils import ellipse_interpolation


# ===== data augmentation =============================================================================================


def hsv_variation(img, variation=[20, 20, 50]):
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)

    for i, var in enumerate(variation):
        value = np.random.randint(-var, var)
        if value >= 0:
            lim = 255 - value
            hsv[:, :, i][hsv[:, :, i] > lim] = 255
            hsv[:, :, i][hsv[:, :, i] <= lim] += value
        else:
            lim = 0 - value
            hsv[:, :, i][hsv[:, :, i] < lim] = 0
            hsv[:, :, i][hsv[:, :, i] >= lim] -= - value

    img = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
    return img


def adjust_contrast_brightness(img, contrast:float=1.0, brightness:int=0):
    """
    Adjusts contrast and brightness of an uint8 image.
    contrast:  float (0.0,  inf) with 1.0 leaving the contrast as is
    brightness: int [-255, 255] with 0 leaving the brightness as is
    """
    b = brightness + int(round(255 * (1 - contrast) / 2))
    return cv2.addWeighted(img, contrast, img, 0, b)


# ===== converting annotations ========================================================================================


def labelme_json_postprocessing(json_file):
    """
    load and convert a labelme .json annotation file to a dataframe, with each row containing the box
    (box_x, box_y, box_w, box_h) and ellipse (ell_x, ell_y, ell_w, ell_h, ell_a) parameters of the correspondind berry.

    Parameters
    ----------
    json_file: str
        path to a .json file generated with the annotation tool "labelme". This file must contain all the polygon
        annotations from a single image. Each polygon label corresponds to one berry.

    Returns
    -------
    pandas.DataFrame
    """

    dfi = []

    # load annotation (.json from Labelme or with the same format)
    with open(json_file) as f:
        anot = json.load(f)

    for shape in anot['shapes']:

        # extract x,y annotated around a berry
        points = np.array(shape['points']).reshape((-1, 1, 2)).astype(int)  # reshaping & int

        # a) ellipse label: fit ellipse equation (5 parameters) to the annotated x,y points with opencv
        (ell_x, ell_y), (ell_w, ell_h), ell_a = cv2.fitEllipse(points)

        # b) box label:
        lsp_x, lsp_y = ellipse_interpolation(x=ell_x, y=ell_y, w=ell_w, h=ell_h, a=ell_a, n_points=100)
        box_x, box_y, box_w, box_h = ell_x, ell_y, max(lsp_x) - min(lsp_x), max(lsp_y) - min(lsp_y)

        dfi.append([ell_x, ell_y, ell_w, ell_h, ell_a, box_x, box_y, box_w, box_h])

    dfi = pd.DataFrame(dfi, columns=['ell_x', 'ell_y', 'ell_w', 'ell_h', 'ell_a', 'box_x', 'box_y', 'box_w', 'box_h'])

    return dfi


# ===== training dataset generation ===================================================================================


def generate_detection_vignette(image, labels, output_path, target_berry_center=None, random_center_proba=0.1):
    """" yolov4 object detection """

    shift_berry_center = 200

    # determine the center point of the vignette on the image
    if np.random.random() < random_center_proba:
        # random
        wi = np.random.randint(VIGNETTE_SIZE_DET / 2, image.shape[1] - VIGNETTE_SIZE_DET / 2)
        hi = np.random.randint(VIGNETTE_SIZE_DET / 2, image.shape[0] - VIGNETTE_SIZE_DET / 2)
    else:
        # close to a berry center
        x, y = target_berry_center if target_berry_center is not None else \
            labels.sample()[['box_x', 'box_y']].values[0]
        xmin, xmax = x - shift_berry_center, x + shift_berry_center
        ymin, ymax = y - shift_berry_center, y + shift_berry_center
        wi = int(min(max(np.random.randint(xmin, xmax), VIGNETTE_SIZE_DET / 2),
                     image.shape[1] - (VIGNETTE_SIZE_DET / 2)))
        hi = int(min(max(np.random.randint(ymin, ymax), VIGNETTE_SIZE_DET / 2),
                     image.shape[0] - (VIGNETTE_SIZE_DET / 2)))

    # crop a vignette around the chosen point
    ha, hb = hi - int(VIGNETTE_SIZE_DET / 2), hi + int(VIGNETTE_SIZE_DET / 2)
    wa, wb = wi - int(VIGNETTE_SIZE_DET / 2), wi + int(VIGNETTE_SIZE_DET / 2)
    vignette = image[ha:hb, wa:wb]

    # image augmentation
    vignette = hsv_variation(vignette, variation=[20, 20, 20])
    contrast = np.random.choice([np.random.uniform(0.7, 1), np.random.uniform(1, 1.5)])
    vignette = adjust_contrast_brightness(vignette, contrast=contrast)

    # random data augmentation flip among 8 possibilities
    # (It's not clear if it's already implemented in Yolov4...)
    aug = {'horizontal_flip': np.random.choice([True, False]),
           'vertical_flip': np.random.choice([True, False]),
           '90_flip': np.random.choice([True, False])}

    # apply the image flip
    if aug['horizontal_flip']:
        vignette = cv2.flip(vignette, 1)
    if aug['vertical_flip']:
        vignette = cv2.flip(vignette, 0)
    if aug['90_flip']:
        vignette = cv2.rotate(vignette, cv2.ROTATE_90_CLOCKWISE)

    # create and save the annotation .txt file, with the format needed to train Yolov4
    boxes_in = {}
    with open(output_path + '.txt', 'w') as out:
        for _, row in labels.iterrows():
            x, y, w, h = row['box_x'], row['box_y'], row['box_w'], row['box_h']

            # check if box is // 100% INCLUDED // in the vignette space
            box_in = wa + w/2 < x < wb - w/2 and ha + h/2 < y < hb - h/2

            if box_in:
                x_center, y_center = (x - wa) / VIGNETTE_SIZE_DET, (y - ha) / VIGNETTE_SIZE_DET
                width, height = w / VIGNETTE_SIZE_DET, h / VIGNETTE_SIZE_DET

                # if image was flipped, annotation has to be flipped too
                if aug['horizontal_flip']:
                    x_center = 1 - x_center
                if aug['vertical_flip']:
                    y_center = 1 - y_center
                if aug['90_flip']:
                    x_center, y_center = 1 - y_center, x_center
                    width, height = height, width

                box_class = 0  # default, because only one class
                out.write('{} {} {} {} {}\n'.format(box_class, x_center, y_center, width, height))

            # save
            boxes_in[row.name] = box_in

    # save vignette
    cv2.imwrite(output_path + '.png', cv2.cvtColor(vignette.astype(np.uint8), cv2.COLOR_RGB2BGR))

    return boxes_in


def generate_segmentation_vignette(image, label, output_path):
    """
    image segmentation
    """

    shift = 0.1

    # random shift to the berry center position
    dx = shift * (2 * np.random.random() - 1) * label['box_w']
    dy = shift * (2 * np.random.random() - 1) * label['box_h']

    # zoom factor to have a berry with a constant size (with random noise added)
    zoom = (VIGNETTE_SIZE_SEG * BERRY_SCALING_SEG) / max(label[['box_h', 'box_w']])
    zoom *= (1 + shift * (2 * np.random.random() - 1))

    # deduce the vignette crop coordinates based on center shift and zoom
    ya = round(label['box_y'] + dy - ((VIGNETTE_SIZE_SEG / 2) / zoom))
    yb = round(label['box_y'] + dy + ((VIGNETTE_SIZE_SEG / 2) / zoom))
    xa = round(label['box_x'] + dx - ((VIGNETTE_SIZE_SEG / 2) / zoom))
    xb = round(label['box_x'] + dx + ((VIGNETTE_SIZE_SEG / 2) / zoom))

    # check if there is enough space to crop the vignette
    enough_space = (0 <= ya) and (yb < image.shape[0]) and (0 <= xa) and (xb <= image.shape[1])

    if enough_space:

        input_vignette = cv2.resize(image[ya:yb, xa:xb], (VIGNETTE_SIZE_SEG, VIGNETTE_SIZE_SEG))

        output_vignette = cv2.ellipse(img=np.float32(input_vignette[:, :, 2] * 0),
                                      center=(round((VIGNETTE_SIZE_SEG / 2) - dx * zoom),
                                              round((VIGNETTE_SIZE_SEG / 2) - dy * zoom)),
                                      axes=(round((label['ell_w'] / 2) * zoom),
                                            round((label['ell_h'] / 2) * zoom)),
                                      angle=label['ell_a'],
                                      startAngle=0.,
                                      endAngle=360,
                                      color=255,
                                      thickness=-1)

        # data augmentation: rgb color / contrast
        input_vignette = hsv_variation(input_vignette, variation=[20, 20, 20])
        contrast = np.random.choice([np.random.uniform(0.7, 1), np.random.uniform(1, 1.5)])
        input_vignette = adjust_contrast_brightness(input_vignette, contrast=contrast)

        # data augmentation : flip and rotation
        do_rotate = np.random.choice([True, False])
        do_flip = np.random.choice(['no flip', 0, 1, -1])
        if do_rotate:
            input_vignette = cv2.rotate(input_vignette, cv2.ROTATE_90_CLOCKWISE)
            output_vignette = cv2.rotate(output_vignette, cv2.ROTATE_90_CLOCKWISE)
        if do_flip != 'no flip':
            input_vignette = cv2.flip(input_vignette, int(do_flip))
            output_vignette = cv2.flip(output_vignette, int(do_flip))

        cv2.imwrite(output_path + 'x.png',
                    cv2.cvtColor(input_vignette.astype(np.uint8), cv2.COLOR_RGB2BGR))
        cv2.imwrite(output_path + 'y.png',
                    cv2.cvtColor(output_vignette.astype(np.uint8), cv2.COLOR_RGB2BGR))

    return enough_space


if __name__ == '__main__':

    import os

    # dir containing the following folders: image_train, image_valid, label_train, label_valid
    DIR_DATASET = 'data/grapevine/dataset/'

    # ===== 1) merge all berry annotations in an easy-to-use dataframe ================================================

    df = []
    for dataset in ['train', 'valid']:

        anot_files = os.listdir(DIR_DATASET + 'label_{}'.format(dataset))

        for file in anot_files:
            print(file)

            dfi = labelme_json_postprocessing(DIR_DATASET + 'label_{}/{}'.format(dataset, file))
            dfi['image_name'] = file.replace('.json', '.png')
            dfi['dataset'] = dataset
            df.append(dfi)

    df = pd.concat(df)

    df.to_csv(DIR_DATASET + 'grapevine_annotation.csv', index=False)

    # ===== generate the training dataset for the detection model (Yolov4) ============================================

    # load annotations (generated in previous code section)
    anot_all = pd.read_csv(DIR_DATASET + 'grapevine_annotation.csv')

    n_vignettes_dic = {'train': 15_000, 'valid': 1_500}  # number of vignettes to generate

    dir = DIR_DATASET + 'dataset_yolo_TEST'
    if not os.path.isdir(dir):
        os.mkdir(dir)

    # remove warning for df.loc[...] = value (default='warn')
    pd.options.mode.chained_assignment = None

    for dataset in ['train', 'valid']:

        saving_path = dir + '/' + dataset
        if not os.path.isdir(saving_path):
            os.mkdir(saving_path)

        anot = anot_all[anot_all['dataset'] == dataset]

        # counter to ensure that all berries appear a similar number of times in the generated vignette dataset
        anot['reps'] = 0

        # load images
        names = anot['image_name'].unique()
        images = {n: cv2.cvtColor(cv2.imread(DIR_DATASET + 'image_{}/'.format(dataset) + n), cv2.COLOR_BGR2RGB)
                  for n in names}

        # remove berries with box going out from image (should not happen, or very rare)
        indexes = []
        for _, row in anot.iterrows():
            x, y, w, h = row['box_x'], row['box_y'], row['box_w'], row['box_h']
            h_img, w_img = images[row['image_name']].shape[:2]
            if not (0 + w / 2 < x < w_img - w / 2 and 0 + h / 2 < y < h_img - h / 2):
                indexes.append(row.name)
                print('removing one box annotation')
        anot = anot.drop(indexes)

        for k_vignette in range(n_vignettes_dic[dataset]):

            # select row from the berry which appears the least amount of time in the vignettes dataset
            row_rarest_berry = anot.sort_values('reps').iloc[0]

            berries_added = generate_detection_vignette(image=images[row_rarest_berry['image_name']],
                                                        labels=anot[
                                                            anot['image_name'] == row_rarest_berry['image_name']],
                                                        target_berry_center=list(row_rarest_berry[['box_x', 'box_y']]),
                                                        output_path='{}/img{}'.format(saving_path, k_vignette))

            # update the counter
            for rowname, added in berries_added.items():
                anot.loc[rowname, 'reps'] += int(added)

    # ===== generate segmentation training dataset ====================================================================

    # load annotations (generated in previous code section)
    df = pd.read_csv(DIR_DATASET + 'grapevine_annotation.csv')

    n_berry_reps = {'train': 10, 'valid': 1}

    dir = DIR_DATASET + 'dataset_seg_TEST'
    if not os.path.isdir(dir):
        os.mkdir(dir)

    for dataset in ['train', 'valid']:

        k_vignette = 0

        saving_path = dir + '/' + dataset
        if not os.path.isdir(saving_path):
            os.mkdir(saving_path)

        anot = df[df['dataset'] == dataset]

        for image_name in np.unique(anot['image_name']):

            s = df[df['image_name'] == image_name]
            image = cv2.cvtColor(cv2.imread(DIR_DATASET + 'image_{}/'.format(dataset) + image_name), cv2.COLOR_BGR2RGB)

            for _, row in s.iterrows():
                for _ in range(n_berry_reps[dataset]):

                    vignette_done = generate_segmentation_vignette(image=image,
                                                                   label=row,
                                                                   output_path='{}/{}'.format(saving_path, k_vignette))

                    if vignette_done:
                        k_vignette += 1

    # =================================================================================================================
