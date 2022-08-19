"""
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

"""

# TODO : image w x h x 4 ??

import json
import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
import pandas as pd
from skimage import io

from deepberry.src.openalea.deepberry.utils import hsv_variation, adjust_contrast_brightness, ellipse_interpolation

PATH = 'data/grapevine/dataset/'

VALIDATION_IMAGE_SUBSET = ['ARCH2021-05-27_323_3706_180.png',
                           'DYN2020-05-15_7243_2465_330.png',
                           'ARCH2022-05-18_684_5601_240.png',
                           'DYN2020-05-15_7235_2422_330.png',
                           'ARCH2022-05-18_1705_5601_330.png']

new_valid = ['DYN2020-05-15_7244_2636_330.png',
             'DYN2020-05-15_7238_2388_330.png',
             'ARCH2021-05-27_7760_3986_240.png',
             'ARCH2021-05-27_7763_3704_270.png',
             'ARCH2022-05-18_328_5928_0.png',
             'ARCH2022-05-18_531_5931_30.png',
             'ARCH2022-05-18_576_5832_0.png',
             'ARCH2022-05-18_1563_5769_30.png',
             'ARCH2022-05-18_1611_5924_0.png',
             'ARCH2022-05-18_1634_5924_30.png']

# ===== annotation dataframe (and visualisation) ===================================================================


anot_files = os.listdir(PATH + 'annotation/')

#shape = anot['shapes'][34]
#file = 'DYN2020-05-15_7244_2636_330.json'
#plt.xlim((1520, 1590))
#plt.ylim((1635, 1703))

df = []
for file in anot_files[::12]:

    img = plt.imread(PATH + 'images/' + file.replace('.json', '.png'))
    plt.figure(file)
    plt.imshow(img)

    # load annotation (json from labelme)
    with open(PATH + 'annotation/' + file) as f:
        anot = json.load(f)

    for shape in anot['shapes']:

        x_anot = np.array(shape['points'])[:, 0].reshape([-1, 1])
        y_anot = np.array(shape['points'])[:, 1].reshape([-1, 1])

        # fit ellipse equation (5 parameters) to the annotated x,y points with opencv
        points = np.hstack((x_anot, y_anot))
        points = np.array([[p] for p in points]).astype(int)  # reshape & int
        #points = cv2.convexHull(points)
        (ell_x, ell_y), (ell_w, ell_h), ell_a = cv2.fitEllipse(points)

        # a) n points of the rotated ellipse
        lsp_x, lsp_y = ellipse_interpolation(x=ell_x, y=ell_y, w=ell_w, h=ell_h, a=ell_a, n_points=100)
        # b) box coordinates (x,y = box center)
        xmin, xmax, ymin, ymax = np.min(lsp_x), np.max(lsp_x), np.min(lsp_y), np.max(lsp_y)
        box_x, box_y, box_w, box_h = (xmax + xmin) / 2, (ymax + ymin) / 2, xmax - xmin, ymax - ymin

        plt.plot(list(x_anot) + [x_anot[0]], list(y_anot) + [y_anot[0]], 'ro-')
        plt.plot(lsp_x, lsp_y, 'blue')
        #plt.plot([xmin, xmin, xmax, xmax, xmin], [ymin, ymax, ymax, ymin, ymin], 'b-')

        df.append([ell_x, ell_y, ell_w, ell_h, ell_a, box_x, box_y, box_w, box_h, file.replace('.json', '.png')])

# TODO : box_x != ell_x ?
df = pd.DataFrame(df, columns=['ell_x', 'ell_y', 'ell_w', 'ell_h', 'ell_a', 'box_x', 'box_y', 'box_w', 'box_h', 'image_name'])
df.to_csv(PATH + 'grapevine_annotation.csv', index=False)

plt.hist(df['ell_h'] / df['ell_w'], bins=50)
plt.hist(np.max((df['box_w'], df['box_h']), axis=0) / np.min((df['box_w'], df['box_h']), axis=0), bins=50)

# ===== explore nms in the dataset ========================================================

from grapevine.utils import get_iou

df = pd.read_csv(PATH + 'grapevine_annotation.csv')
for name in df['image_name'].unique():
    iou_list = []
    s = df[df['image_name'] == name]
    for _, row1 in s.iterrows():
        for _, row2 in s.iterrows():
            box1 = {'x1': row1['box_x'], 'x2': row1['box_x'] + row1['box_w'],
                    'y1': row1['box_y'], 'y2': row1['box_y'] + row1['box_h']}
            box2 = {'x1': row2['box_x'], 'x2': row2['box_x'] + row2['box_w'],
                    'y1': row2['box_y'], 'y2': row2['box_y'] + row2['box_h']}
            iou = get_iou(box1, box2)
            if iou not in [0., 1.]:
                iou_list.append(iou)
    if iou_list:
        print(round(sorted(iou_list)[-1], 2), name)

# ==== test ell size ====================================

df['size'] = np.max(df[['box_w', 'box_h']], axis=1)
df.sort_values('size')

# ===== generate segmentation training dataset ===========================================

# TODO : 1% annotated box > 128px. > 160 devrait suffire.

dataset_folder = 'training_dataset_segmentation/'
if not os.path.isdir(PATH + dataset_folder):
    os.mkdir(PATH + dataset_folder)

for subset in ['train', 'valid']:

    saving_path = PATH + dataset_folder + subset + '_scaled/'
    if not os.path.isdir(saving_path):
        os.mkdir(saving_path)

    anot = df[df['image_name'].isin(VALIDATION_IMAGE_SUBSET)] if subset == 'valid' else \
        df[~df['image_name'].isin(VALIDATION_IMAGE_SUBSET)]

    size = 128
    shift = 0.1

    for image_name in np.unique(anot['image_name']):

        s = df[df['image_name'] == image_name]
        img = io.imread(PATH + 'images/' + image_name)[:, :, :3]

        for k, row in s.iterrows():

            # ===== new method ===============================================================

            # round(round(2448 * z) / z) : conserve 2448 si z > 1., pb a partir de z ~< 0.8

            zoom = (size * 0.75) / np.max(row[['ell_w', 'ell_h']])
            zoom = zoom * (1 + shift * (2 * np.random.random() - 1))

            x_vignette = round(row['ell_x'] + shift * (2 * np.random.random() - 1) * row['box_w'])
            y_vignette = round(row['ell_y'] + shift * (2 * np.random.random() - 1) * row['box_h'])

            ya = y_vignette - round(int(size / 2) / zoom)
            yb = y_vignette + round(int(size / 2) / zoom)
            xa = x_vignette - round(int(size / 2) / zoom)
            xb = x_vignette + round(int(size / 2) / zoom)

            # a vignette of shape (size, size) can be positionned around x,y
            condition1 = int(size / 2) < y_vignette < 2448 - int(size / 2) and int(size / 2) < x_vignette < 2048 - int(size / 2)
            # enough space to unzoom if big berry
            condition2 = (0 <= ya) and (yb < 2448) and (0 <= xa) and (xb <= 2448)

            if condition1 and condition2:

                input_vignette = cv2.resize(img[ya:yb, xa:xb], (128, 128))

                # TODO shift!! (x,y) ?
                label = cv2.ellipse(np.float32(img * 0),
                                    (round(row['ell_x']), round(row['ell_y'])),
                                    (round(zoom * (row['ell_w'] / 2)), round(zoom * (row['ell_h'] / 2))),
                                    row['ell_a'], 0., 360,
                                    (255, 255, 255), -1)

                output_vignette = label[(y_vignette - int(size / 2)):(y_vignette + int(size / 2)),
                                        (x_vignette - int(size / 2)):(x_vignette + int(size / 2))]

                io.imsave(saving_path + '{}x.png'.format(k), input_vignette.astype(np.uint8))
                io.imsave(saving_path + '{}y.png'.format(k), output_vignette.astype(np.uint8))

            else:
                print('border')







            # mask = (output_vignette[:, :, 0] > 0.5).astype(np.uint8) * 255
            # edges, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            # if edges:
            #     edges = edges[np.argmax([len(e) for e in edges])][:, 0, :]  # takes longest contour if several areas
            #
            # plt.figure()
            # plt.imshow(output_vignette)
            #
            # plt.figure()
            # plt.imshow(input_vignette)
            # plt.plot(edges[:, 0], edges[:, 1], 'r-')
            #
            # # back to original scale
            # edges_img = edges + np.array([x_vignette - int(size / 2), y_vignette - int(size / 2)])
            # (ell_x, ell_y), (ell_w, ell_h), ell_a = cv2.fitEllipse(edges_img)
            # ell_w /= zoom
            # ell_h /= zoom
            #
            # plt.figure()
            # lsp_x, lsp_y = ellipse_interpolation(x=ell_x, y=ell_y, w=ell_w, h=ell_h, a=ell_a, n_points=100)
            # plt.imshow(img)
            # plt.plot(lsp_x, lsp_y, 'r-')

            # ===== old method ================================================================

            label = cv2.ellipse(np.float32(img * 0),
                                (round(row['ell_x']), round(row['ell_y'])),
                                (round(row['ell_w'] / 2), round(row['ell_h'] / 2)),
                                row['ell_a'], 0., 360,
                                (255, 255, 255), -1)

            x_vignette = round(row['ell_x'] + shift * np.random.random() * row['box_w'])
            y_vignette = round(row['ell_y'] + shift * np.random.random() * row['box_h'])

            if 64 < y_vignette < 2448 - 64 and 64 < x_vignette < 2048 - 64:

                input_vignette = img[(y_vignette - int(size / 2)):(y_vignette + int(size / 2)),
                               (x_vignette - int(size / 2)):(x_vignette + int(size / 2))]
                output_vignette = label[(y_vignette - int(size / 2)):(y_vignette + int(size / 2)),
                               (x_vignette - int(size / 2)):(x_vignette + int(size / 2))]

                io.imsave(saving_path + '{}x.png'.format(k), input_vignette.astype(np.uint8))
                io.imsave(saving_path + '{}y.png'.format(k), output_vignette.astype(np.uint8))

            else:
                print('border')


# =====

import os
import numpy as np
import cv2

data_path = PATH + 'training_dataset_segmentation/train/'
indexes = np.unique([int(f[:-5]) for f in os.listdir(data_path)])
train_indexes = np.random.choice(indexes, int(0.9 * len(indexes)), replace=False)

X = np.zeros((len(train_indexes) * 4, 128, 128, 3))
Y = np.zeros((len(train_indexes) * 4, 128, 128, 2))

i_image = 0
for i in sorted(train_indexes):
    print(i)
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

            X[i_image] = x2
            Y[i_image] = y2
            i_image += 1

np.save(PATH + 'training_dataset_segmentation/X.npy', X)
np.save(PATH + 'training_dataset_segmentation/Y.npy', Y)

# ==== visu segmentation dataset ======================================================================================

img_mean = np.zeros((128, 128))
folder = PATH + 'training_dataset_segmentation/train/'
files = [folder + n for n in os.listdir(folder) if 'y' in n]
for f in files:
    img = io.imread(f)[:, :, 0]
    img_mean += img / 2550
plt.imshow(img_mean / np.max(img_mean) * 255.)

# ===== generate detection training dataset ===========================================================================

"""
- TODO voir l'effet des parametres yolo de data aug (blur, zoom, crop, enlever mosaic etc)
(blur : default blur_kernel=31 dans yolo, mais valeur peut etre changée)

- Rotation : en principe c'est pas possible quand on a que les box (pour ce que yolo l'implemente que pour la partie 
classif). Mais la vu qu'on a les ellipses, on peut l'implementer, ca a juste l'air galère.
Methode : on part de l'image, on la tourne de k degrés, et on modifie tout le tableau d'anot en conséquences (ell_a
facile, mais comment on fait pour ell_x et ell_y ?)

-Autre facon de gérer les problèmes de bord ?
"""

H, W = 416, 416  # vignette size

dataset_folder = 'training_dataset/'
if not os.path.isdir(PATH + dataset_folder):
    os.mkdir(PATH + dataset_folder)

saving_path = PATH + dataset_folder + 'valid/'
if not os.path.isdir(saving_path):
    os.mkdir(saving_path)

anot = df[df['image_name'].isin(VALIDATION_IMAGE_SUBSET)]
anot['reps'] = 0

# remove berries with ellipse going out from image (should not happen, or very rare)
indexes = []
for i, row in anot.iterrows():
    x, y, w, h = row['box_x'], row['box_y'], row['box_w'], row['box_h']
    if not (0 + w / 2 < x < 2048 - w / 2 and 0 + h / 2 < y < 2448 - h / 2):
        indexes.append(row.name)
anot = anot.drop(indexes)

# remove warning for df.loc[...] = value (default='warn')
pd.options.mode.chained_assignment = None

#k_image = 0
for k_image in range(100):  # 7973
#for image_name in anot['image_name'].unique():

    berry_selec = anot.sort_values('reps').iloc[0]
    print(k_image, len(anot[anot['reps'] == 0]), min(anot['reps']), round(np.mean(anot['reps']), 2), np.median(anot['reps']), max(anot['reps']))

    img = cv2.imread(PATH + 'images/' + berry_selec['image_name'])
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # plt.imshow(img)
    # plt.plot(berry_selec['box_x'], berry_selec['box_y'], 'r*')

    # xmin, xmax = min(anot_selec['box_x']), max(anot_selec['box_x'])
    # ymin, ymax = min(anot_selec['box_y']), max(anot_selec['box_y'])

    #n_rep = max((20, int((xmax - xmin) * (ymax - ymin) * 0.0002)))
    n_rep = 1

    for _ in range(n_rep):

        image = hsv_variation(img, variation=[20, 20, 20])
        contrast = np.random.choice([np.random.uniform(0.7, 1), np.random.uniform(1, 1.5)])
        image = adjust_contrast_brightness(image, contrast=contrast)
        #io.imsave(path + 'augmentation_example/aug{}.png'.format(k), image)
        # plt.figure('augmented')
        # plt.imshow(img2)

        # center of the vignette
        if np.random.random() < 0.10:  # 0.01
            wi = np.random.randint(W / 2, image.shape[1] - W / 2)
            hi = np.random.randint(H / 2, image.shape[0] - H / 2)
        else:
            # wi = int(min(max(np.random.randint(xmin, xmax), W / 2), image.shape[1] - W / 2))
            # hi = int(min(max(np.random.randint(ymin, ymax), H / 2), image.shape[0] - H / 2))
            xmin, xmax = berry_selec['box_x'] - 200, berry_selec['box_x'] + 200
            ymin, ymax = berry_selec['box_y'] - 200, berry_selec['box_y'] + 200
            wi = int(min(max(np.random.randint(xmin, xmax), W / 2), image.shape[1] - W / 2))
            hi = int(min(max(np.random.randint(ymin, ymax), H / 2), image.shape[0] - H / 2))


        # crop a vignette around the chosen point
        ha, hb = hi - int(H / 2), hi + int(H / 2)
        wa, wb = wi - int(W / 2), wi + int(W / 2)
        vignette = image[ha:hb, wa:wb]

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
        with open(saving_path + 'img{}.txt'.format(k_image), 'w') as out:
            for _, row in anot[anot['image_name'] == berry_selec['image_name']].iterrows():
                x, y, w, h = row['box_x'], row['box_y'], row['box_w'], row['box_h']
                # check if box is // 100% INCLUDED // in the vignette space
                if wa + w/2 < x < wb - w/2 and ha + h/2 < y < hb - h/2:
                    x_center, y_center = (x - wa) / W, (y - ha) / H
                    width, height = w / W, h / H

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

                    box_class = 0
                    out.write('{} {} {} {} {}\n'.format(box_class, x_center, y_center, width, height))

        # save vignette
        io.imsave(saving_path + 'img{}.png'.format(k_image), vignette.astype(np.uint8))
        k_image += 1

# ===== generate Mask-RCNN dataset ===========================================================================

H, W = 448, 448  # vignette size

dataset_folder = 'training_dataset/'
if not os.path.isdir(PATH + dataset_folder):
    os.mkdir(PATH + dataset_folder)

saving_path = PATH + dataset_folder + 'valid_maskrcnn/'
if not os.path.isdir(saving_path):
    os.mkdir(saving_path)

df = pd.read_csv(PATH + 'grapevine_annotation.csv')

anot = df[df['image_name'].isin(VALIDATION_IMAGE_SUBSET)]
anot['reps'] = 0

# remove berries with ellipse going out from image (should not happen, or very rare)
indexes = []
for i, row in anot.iterrows():
    x, y, w, h = row['box_x'], row['box_y'], row['box_w'], row['box_h']
    if not (0 + w / 2 < x < 2048 - w / 2 and 0 + h / 2 < y < 2448 - h / 2):
        indexes.append(row.name)
anot = anot.drop(indexes)

# remove warning for df.loc[...] = value (default='warn')
pd.options.mode.chained_assignment = None

df_label = []
for k_image in range(1000):  # 7973
#for image_name in anot['image_name'].unique():

    berry_selec = anot.sort_values('reps').iloc[0]
    print(k_image, len(anot[anot['reps'] == 0]), min(anot['reps']), round(np.mean(anot['reps']), 2), np.median(anot['reps']), max(anot['reps']))

    img = cv2.imread(PATH + 'images/' + berry_selec['image_name'])
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # plt.imshow(img)
    # plt.plot(berry_selec['box_x'], berry_selec['box_y'], 'r*')

    # xmin, xmax = min(anot_selec['box_x']), max(anot_selec['box_x'])
    # ymin, ymax = min(anot_selec['box_y']), max(anot_selec['box_y'])

    #n_rep = max((20, int((xmax - xmin) * (ymax - ymin) * 0.0002)))
    n_rep = 1

    for _ in range(n_rep):

        image = hsv_variation(img, variation=[20, 20, 20])
        contrast = np.random.choice([np.random.uniform(0.7, 1), np.random.uniform(1, 1.5)])
        image = adjust_contrast_brightness(image, contrast=contrast)
        #io.imsave(path + 'augmentation_example/aug{}.png'.format(k), image)
        # plt.figure('augmented')
        # plt.imshow(img2)

        # center of the vignette
        if np.random.random() < 0.10:  # 0.01
            wi = np.random.randint(W / 2, image.shape[1] - W / 2)
            hi = np.random.randint(H / 2, image.shape[0] - H / 2)
        else:
            # wi = int(min(max(np.random.randint(xmin, xmax), W / 2), image.shape[1] - W / 2))
            # hi = int(min(max(np.random.randint(ymin, ymax), H / 2), image.shape[0] - H / 2))
            xmin, xmax = berry_selec['box_x'] - 200, berry_selec['box_x'] + 200
            ymin, ymax = berry_selec['box_y'] - 200, berry_selec['box_y'] + 200
            wi = int(min(max(np.random.randint(xmin, xmax), W / 2), image.shape[1] - W / 2))
            hi = int(min(max(np.random.randint(ymin, ymax), H / 2), image.shape[0] - H / 2))

        # crop a vignette around the chosen point
        ha, hb = hi - int(H / 2), hi + int(H / 2)
        wa, wb = wi - int(W / 2), wi + int(W / 2)
        vignette = image[ha:hb, wa:wb]

        for _, row in anot[anot['image_name'] == berry_selec['image_name']].iterrows():
            x, y, w, h = row['box_x'], row['box_y'], row['box_w'], row['box_h']
            # check if box is // 100% INCLUDED // in the vignette space
            if wa + w/2 < x < wb - w/2 and ha + h/2 < y < hb - h/2:

                df_label.append([row['ell_x'] - wa, row['ell_y'] - ha,
                                 row['ell_w'], row['ell_h'], row['ell_a'],
                                 'img{}.png'.format(k_image)])

                # count berry occurence
                anot.loc[row.name, 'reps'] += 1

        # save vignette
        io.imsave(saving_path + 'img{}.png'.format(k_image), vignette.astype(np.uint8))
        k_image += 1

df_label = pd.DataFrame(df_label, columns=['ell_x', 'ell_y', 'ell_w', 'ell_h', 'ell_a', 'image_name'])
df_label.to_csv(PATH + 'training_dataset/valid_maskrcnn.csv', index=False)

# ===== visualize training dataset ==========================================================================

folder = PATH + 'training_dataset/train/'

for k in range(10):

    i = np.random.randint(len(os.listdir(folder)) / 2)
    img = io.imread(folder + 'img{}.png'.format(i))
    with open(folder + 'img{}.txt'.format(i), 'r') as file:
        anots = file.readlines()

    plt.figure()
    plt.imshow(img)
    for anot in anots:
        side, x, y, w, h = [float(k) for k in anot[:-2].split(' ')]
        w2, h2 = w * img.shape[1], h * img.shape[0]
        x_corner, y_corner = round(x * img.shape[1] - w2/2), round(y * img.shape[0] - h2/2)
        X = [x_corner, x_corner, x_corner + w2, x_corner + w2, x_corner]
        Y = [y_corner, y_corner + h2, y_corner + h2, y_corner, y_corner]
        plt.plot(X, Y, 'g-')

# ===== training on deepl ===================================================================================

import os

# create train.txt and valid.txt files
with open('/home/bdaviet/train.txt', 'w') as out:
  images = [f for f in os.listdir('train') if f[-4:] == '.png']
  for f in images:
    out.write('train/{}\n'.format(f))

with open('/home/bdaviet/valid.txt', 'w') as out:
  images = [f for f in os.listdir('valid') if f[-4:] == '.png']
  for f in images:
    out.write('valid/{}\n'.format(f))

# darknet detector train training.data config_1class.cfg -dont_show

# sudo docker run --rm -v $PWD:/workspace -w /workspace \daisukekobayashi/darknet:cpu darknet detector train training.data config_1class.cfg -dont_show
# sudo docker run --runtime=nvidia --rm -v $PWD:/workspace -w /workspace \daisukekobayashi/darknet:gpu darknet detector train training.data config_1class.cfg -dont_show

# ===== temporal annotation =================================================================================

with open('data/grapevine/temporal/baie2.json') as f:
    anot = json.load(f)

for image in np.array(list(anot.keys())):
    t = int(image.split('.')[1])
    region = anot[image]['regions'][0]
    X = np.array(region['shape_attributes']['all_points_x']).reshape([-1, 1])
    Y = np.array(region['shape_attributes']['all_points_y']).reshape([-1, 1])

    A = np.hstack([X**2, X * Y, Y**2, X, Y])
    b = np.ones_like(X)
    x = np.linalg.lstsq(A, b)[0].squeeze()

    area = np.pi * np.sqrt(np.abs(1 / x[0])) * np.sqrt(np.abs(1 / x[2]))
    print(t, area)
    #plt.plot(t, area, 'k.')

    # original points
    plt.plot(X, Y, 'k.')

    # ellipse fit
    x_coord = np.linspace(min(X) - 100, max(X) + 100, 300)
    y_coord = np.linspace(min(Y) - 100, max(Y) + 100, 300)
    X_coord, Y_coord = np.meshgrid(x_coord, y_coord)
    Z_coord = x[0] * X_coord ** 2 + x[1] * X_coord * Y_coord + x[2] * Y_coord ** 2 + x[3] * X_coord + x[4] * Y_coord
    plt.contour(X_coord, Y_coord, Z_coord, levels=[1], colors=('r'), linewidths=2)


# ===== edge detection ====================================================================================

import cv2

# Read the original image
img = cv2.imread('data/grapevine/grapevine.png')
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
# Display original image
# cv2.imshow('Original', img)
# cv2.waitKey(0)

# Convert to graycsale
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Blur the image for better edge detection
img_blur = cv2.GaussianBlur(img_gray, (3, 3), 0)

# Sobel Edge Detection
sobelxy = cv2.Sobel(src=img_blur, ddepth=cv2.CV_64F, dx=3, dy=3, ksize=5)  # Combined X and Y Sobel Edge Detection

# Display Sobel Edge Detection Images
# cv2.imshow('Sobel X Y using Sobel() function', sobelxy)
# cv2.waitKey(0)

# Canny Edge Detection
edges = cv2.Canny(image=img_blur, threshold1=10, threshold2=50)  # Canny Edge Detection

# Display Canny Edge Detection Image
plt.imshow(np.stack((img, cv2.merge((255 - edges, 255 - edges, 255 - edges)))).min(0))


x = np.array([2, 1, 3, 0, 0])
border = 10
x_coord = np.linspace(min(X) - border, max(X) + border, 300)
y_coord = np.linspace(min(Y) - border, max(Y) + border, 300)
X_coord, Y_coord = np.meshgrid(x_coord, y_coord)
Z_coord = x[0] * X_coord ** 2 + x[1] * X_coord * Y_coord + x[2] * Y_coord ** 2 + x[3] * X_coord + x[4] * Y_coord
plt.axis('equal')
plt.contour(X_coord, Y_coord, Z_coord, '*', levels=[1], colors=('r'), linewidths=1)
plt.grid()

























