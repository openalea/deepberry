"""

# darknet detector train training.data config_1class.cfg -dont_show
# sudo docker run --rm -v $PWD:/workspace -w /workspace \daisukekobayashi/darknet:cpu darknet detector train training.data config_1class.cfg -dont_show
# sudo docker run --runtime=nvidia --rm -v $PWD:/workspace -w /workspace \daisukekobayashi/darknet:gpu darknet detector train training.data config_1class.cfg -dont_show

fit ellipse : https://stackoverflow.com/questions/61467807/detect-ellipse-parameters-from-a-given-elliptical-mask
plot ellipse : https://stackoverflow.com/questions/10952060/plot-ellipse-with-matplotlib-pyplot-python

a) en utilisant le docker (pour arreter; sudo docker stop), sur la carte gpu n°1
sudo docker run --gpus '"device=1"' --runtime=nvidia --rm -v $PWD:/workspace -w /workspace \
daisukekobayashi/darknet:gpu darknet detector train training.data config_1class.cfg -dont_show

b) en utilisant direct l'executable, modifié pour sauver toutes les 500 iters (pour arreter: ctrl C), sur les 2 gpu:
darknet500 detector train training.data config_1class.cfg -dont_show -gpus 0,1

pour sauver l'avancement dans un fichier txt :
| tee results.txt

======== COMMANDE YOLOv4 ============================================================================================

darknet500 detector train training.data config_1class.cfg -dont_show | tee results.txt

=====================================================================================================================

# TODO in model_training/README
HOW TO TRAIN DEEBBERRY MODELS:
- create a folder with the following 4 subfolders : image_train, image_valid, label_train, label_valid. The path of the
main folder must be indicated in the PATH variable below. image_train / image_valid must contain RGB images of
grapevine. label_train / label_valid must contain annotation .json files with the same name as their corresponding
image. Each annotation file needs to have the same format as the .json files from the annotation tool Labelme.
-run this script (1- generate annot df, 2- generate yolo dataset, 3- generate unet dataset)
- zip yolo dataset, put in on google drive, also put the .cfg file on drive, run the colab script for detection
--> save model with the highest AP (Average Precision).
- zip unet dataset, put in on colab, run the colab script for segmentation
---> save model with the lowest validation loss

"""

import json
import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
import pandas as pd
from skimage import io

from deepberry.src.openalea.deepberry.segmentation import VIGNETTE_SIZE_DET, VIGNETTE_SIZE_SEG, BERRY_SCALING_SEG
from deepberry.src.openalea.deepberry.utils import hsv_variation, adjust_contrast_brightness, ellipse_interpolation
from deepberry.src.openalea.deepberry.color import berry_features_extraction

# dir containing the following folders: image_train, image_valid, label_train, label_valid
DIR_DATASET = 'data/grapevine/dataset/'

# image format
H_IMG, W_IMG = 2448, 2048

# ===== 1) merge all berry annotations in an easy-to-use dataframe ===================================================

df = []
for dataset in ['train', 'valid']:

    anot_files = os.listdir(DIR_DATASET + 'label_{}'.format(dataset))

    for file in anot_files:
        print(file)

        # img = plt.imread(PATH + 'image_{}/'.format(dataset) + file.replace('.json', '.png'))
        # plt.figure(file)
        # plt.imshow(img)

        # load annotation (json from labelme)
        with open(DIR_DATASET + 'label_{}/{}'.format(dataset, file)) as f:
            anot = json.load(f)

        for shape in anot['shapes']:

            # extract x,y annotated around a berry
            points = np.array(shape['points']).reshape((-1, 1, 2)).astype(int)  # reshaping & int

            # fit ellipse equation (5 parameters) to the annotated x,y points with opencv
            (ell_x, ell_y), (ell_w, ell_h), ell_a = cv2.fitEllipse(points)

            # a) n points of the rotated ellipse
            lsp_x, lsp_y = ellipse_interpolation(x=ell_x, y=ell_y, w=ell_w, h=ell_h, a=ell_a, n_points=100)
            # b) deduce box coordinates (x,y = box center)
            box_x, box_y, box_w, box_h = ell_x, ell_y, max(lsp_x) - min(lsp_x), max(lsp_y) - min(lsp_y)

            # plt.plot(list(x_anot) + [x_anot[0]], list(y_anot) + [y_anot[0]], 'ro-')
            # plt.plot(lsp_x, lsp_y, 'blue')
            # plt.plot([xmin, xmin, xmax, xmax, xmin], [ymin, ymax, ymax, ymin, ymin], 'r-')

            image_name = file.replace('.json', '.png')
            df.append([ell_x, ell_y, ell_w, ell_h, ell_a, box_x, box_y, box_w, box_h, image_name, dataset])

df = pd.DataFrame(df, columns=['ell_x', 'ell_y', 'ell_w', 'ell_h', 'ell_a', 'box_x', 'box_y', 'box_w', 'box_h',
                               'image_name', 'dataset'])
df.to_csv(DIR_DATASET + 'grapevine_annotation.csv', index=False)

# ===== generate the training dataset for the detection model (Yolov4) ================================================

# load annotations (generated in previous code section)
df = pd.read_csv(DIR_DATASET + 'grapevine_annotation.csv')

n_vignettes_dic = {'train': 15_000, 'valid': 1_500}  # number of vignettes to generate

dir = DIR_DATASET + 'dataset_yolo'
if not os.path.isdir(dir):
    os.mkdir(dir)

for dataset in ['train', 'valid']:

    n_vignettes = n_vignettes_dic[dataset]

    saving_path = dir + '/' + dataset
    if not os.path.isdir(saving_path):
        os.mkdir(saving_path)

    anot = df[df['dataset'] == dataset]
    anot['reps'] = 0

    # remove berries with ellipse going out from image (should not happen, or very rare)
    indexes = []
    for i, row in anot.iterrows():
        x, y, w, h = row['box_x'], row['box_y'], row['box_w'], row['box_h']
        if not (0 + w / 2 < x < W_IMG - w / 2 and 0 + h / 2 < y < H_IMG - h / 2):
            indexes.append(row.name)
    anot = anot.drop(indexes)

    # remove warning for df.loc[...] = value (default='warn')
    pd.options.mode.chained_assignment = None

    # load images
    names = anot['image_name'].unique()
    images = {n: cv2.cvtColor(cv2.imread(DIR_DATASET + 'image_{}/'.format(dataset) + n), cv2.COLOR_BGR2RGB)
              for n in names}

    for k_image in range(n_vignettes):

        berry_selec = anot.sort_values('reps').iloc[0]
        print(k_image, len(anot[anot['reps'] == 0]), min(anot['reps']),
              round(np.mean(anot['reps']), 2), np.median(anot['reps']), max(anot['reps']))

        image = images[berry_selec['image_name']]

        # center of the vignette
        if np.random.random() < 0.10:
            wi = np.random.randint(VIGNETTE_SIZE_DET / 2, image.shape[1] - VIGNETTE_SIZE_DET / 2)
            hi = np.random.randint(VIGNETTE_SIZE_DET / 2, image.shape[0] - VIGNETTE_SIZE_DET / 2)
        else:
            xmin, xmax = berry_selec['box_x'] - 200, berry_selec['box_x'] + 200
            ymin, ymax = berry_selec['box_y'] - 200, berry_selec['box_y'] + 200
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
        with open(saving_path + '/img{}.txt'.format(k_image), 'w') as out:
            for _, row in anot[anot['image_name'] == berry_selec['image_name']].iterrows():
                x, y, w, h = row['box_x'], row['box_y'], row['box_w'], row['box_h']
                # check if box is // 100% INCLUDED // in the vignette space
                if wa + w/2 < x < wb - w/2 and ha + h/2 < y < hb - h/2:
                    x_center, y_center = (x - wa) / VIGNETTE_SIZE_DET, (y - ha) / VIGNETTE_SIZE_DET
                    width, height = w / VIGNETTE_SIZE_DET, h / VIGNETTE_SIZE_DET

                    # count berry occurence
                    anot.loc[row.name, 'reps'] += 1

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

        # save vignette
        io.imsave(saving_path + '/img{}.png'.format(k_image), vignette.astype(np.uint8))
        k_image += 1

# ===== add hue in anot df ================================================================
# TODO in paper/valid_seg (effect of color)

df = pd.read_csv(DIR_DATASET + 'grapevine_annotation.csv')
new_df = []
for image_name in df['image_name'].unique():
    print(image_name)
    s = df[df['image_name'] == image_name]
    img = cv2.cvtColor(cv2.imread(DIR_DATASET + 'images/{}'.format(image_name)), cv2.COLOR_BGR2RGB)
    new_s = berry_features_extraction(img, s)
    new_df.append(new_s)
new_df = pd.concat(new_df)

# hue = ((180 - np.array(new_df['hue'])) - 100) % 180
# new_df['hue'] = hue

new_df.to_csv(DIR_DATASET + 'grapevine_annotation.csv', index=False)

# ===== generate segmentation training dataset =======================================================================

# load annotations (generated in previous code section)
df = pd.read_csv(DIR_DATASET + 'grapevine_annotation.csv')

n_berry_reps = {'train': 10, 'valid': 1}
shift = 0.1

dir = DIR_DATASET + 'dataset_seg'
if not os.path.isdir(dir):
    os.mkdir(dir)

for dataset in ['train', 'valid']:

    img_counter = 0

    saving_path = dir + '/' + dataset
    if not os.path.isdir(saving_path):
        os.mkdir(saving_path)

    anot = df[df['dataset'] == dataset]

    for image_name in np.unique(anot['image_name']):

        s = df[df['image_name'] == image_name]
        img = cv2.cvtColor(cv2.imread(DIR_DATASET + 'image_{}/'.format(dataset) + image_name), cv2.COLOR_BGR2RGB)

        for _, row in s.iterrows():
            for _ in range(n_berry_reps[dataset]):

                # random shift to the berry center position
                dx = shift * (2 * np.random.random() - 1) * row['box_w']
                dy = shift * (2 * np.random.random() - 1) * row['box_h']

                # zoom factor to have a berry with a constant size
                zoom = (VIGNETTE_SIZE_SEG * BERRY_SCALING_SEG) / max(row[['box_h', 'box_w']])
                # add some random noise to this factor
                zoom *= (1 + shift * (2 * np.random.random() - 1))

                # deduce the vignette crop coordinates based on center shift and zoom
                ya = round(row['box_y'] + dy - ((VIGNETTE_SIZE_SEG / 2) / zoom))
                yb = round(row['box_y'] + dy + ((VIGNETTE_SIZE_SEG / 2) / zoom))
                xa = round(row['box_x'] + dx - ((VIGNETTE_SIZE_SEG / 2) / zoom))
                xb = round(row['box_x'] + dx + ((VIGNETTE_SIZE_SEG / 2) / zoom))

                # check if there is enough space to crop the vignette
                enough_space = (0 <= ya) and (yb < H_IMG) and (0 <= xa) and (xb <= W_IMG)

                if enough_space:

                    input_vignette = cv2.resize(img[ya:yb, xa:xb], (VIGNETTE_SIZE_SEG, VIGNETTE_SIZE_SEG))

                    output_vignette = cv2.ellipse(np.float32(input_vignette[:, :, 2] * 0),
                                                  (round((VIGNETTE_SIZE_SEG / 2) - dx * zoom),
                                                   round((VIGNETTE_SIZE_SEG / 2) - dy * zoom)),
                                                  (round((row['ell_w'] / 2) * zoom),
                                                   round((row['ell_h'] / 2) * zoom)),
                                                  row['ell_a'], 0., 360, 255, 2)  # TODO 2 or -1 ?

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

                    io.imsave(saving_path + '/{}x.png'.format(img_counter), input_vignette.astype(np.uint8))
                    io.imsave(saving_path + '/{}y.png'.format(img_counter), output_vignette.astype(np.uint8))
                    img_counter += 1

                else:
                    print('border')























