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

# TODO in model_training/README
HOW TO TRAIN DEEBBERRY MODELS:
- create a folder with the following 4 subfolders : image_train, image_valid, label_train, label_valid. The path of the
main folder must be indicated in the PATH variable below. image_train / image_valid must contain RGB images of
grapevine. label_train / label_valid must contain annotation .json files with the same name as their corresponding
image. Each annotation file needs to have the same format as the .json files from the annotation tool Labelme.
-run this script (1- generate annot df, 2- generate yolo dataset, 3- generate unet dataset)
- zip yolo dataset, put in on google drive, also put the .cfg file on drive, run the colab script for detection
- zip unet dataset, put in on colab, run the colab script for segmentation
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
import pandas as pd
from skimage import io

from deepberry.src.openalea.deepberry.utils import hsv_variation, adjust_contrast_brightness, ellipse_interpolation

from deepberry.src.openalea.deepberry.color import mean_hue_berry

PATH = 'data/grapevine/dataset/'

# TODO remove
fd = 'data/grapevine/dataset/dataset_yolo/train/'
for f in os.listdir(fd):
    if int(f[3:].split('.')[0]) > 15000:
        os.remove(fd + f)


# ===== 1) merge all berry annotations in an easy-to-use dataframe ===================================================

df = []
for dataset in ['train', 'valid']:

    anot_files = os.listdir(PATH + 'label_{}'.format(dataset))

    for file in anot_files:
        print(file)

        # img = plt.imread(PATH + 'image_{}/'.format(dataset) + file.replace('.json', '.png'))
        # plt.figure(file)
        # plt.imshow(img)

        # load annotation (json from labelme)
        with open(PATH + 'label_{}/{}'.format(dataset, file)) as f:
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
df.to_csv(PATH + 'grapevine_annotation.csv', index=False)

# ===== generate the training dataset for the detection model (Yolov4) ================================================

# load annotations (generated in previous code section)
df = pd.read_csv(PATH + 'grapevine_annotation.csv')

H_img, W_img = 2448, 2048  # image size
H_vig, W_vig = 416, 416  # vignette size

dir = PATH + 'dataset_yolo'
if not os.path.isdir(dir):
    os.mkdir(dir)

for dataset, n_vignettes_total in zip(['train', 'valid'], [15000, 1500]):

    saving_path = dir + '/' + dataset
    if not os.path.isdir(saving_path):
        os.mkdir(saving_path)

    anot = df[df['dataset'] == dataset]
    anot['reps'] = 0

    # remove berries with ellipse going out from image (should not happen, or very rare)
    indexes = []
    for i, row in anot.iterrows():
        x, y, w, h = row['box_x'], row['box_y'], row['box_w'], row['box_h']
        if not (0 + w / 2 < x < W_img - w / 2 and 0 + h / 2 < y < H_img - h / 2):
            indexes.append(row.name)
    anot = anot.drop(indexes)

    # remove warning for df.loc[...] = value (default='warn')
    pd.options.mode.chained_assignment = None

    # load images
    names = anot['image_name'].unique()
    images = {n: cv2.cvtColor(cv2.imread(PATH + 'image_{}/'.format(dataset) + n), cv2.COLOR_BGR2RGB) for n in names}

    for k_image in range(n_vignettes_total):

        berry_selec = anot.sort_values('reps').iloc[0]
        print(k_image, len(anot[anot['reps'] == 0]), min(anot['reps']),
              round(np.mean(anot['reps']), 2), np.median(anot['reps']), max(anot['reps']))

        image = images[berry_selec['image_name']]

        # center of the vignette
        if np.random.random() < 0.10:
            wi = np.random.randint(W_vig / 2, image.shape[1] - W_vig / 2)
            hi = np.random.randint(H_vig / 2, image.shape[0] - H_vig / 2)
        else:
            xmin, xmax = berry_selec['box_x'] - 200, berry_selec['box_x'] + 200
            ymin, ymax = berry_selec['box_y'] - 200, berry_selec['box_y'] + 200
            wi = int(min(max(np.random.randint(xmin, xmax), W_vig / 2), image.shape[1] - (W_vig / 2)))
            hi = int(min(max(np.random.randint(ymin, ymax), H_vig / 2), image.shape[0] - (H_vig / 2)))

        # crop a vignette around the chosen point
        ha, hb = hi - int(H_vig / 2), hi + int(H_vig / 2)
        wa, wb = wi - int(W_vig / 2), wi + int(W_vig / 2)
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
                    x_center, y_center = (x - wa) / W_vig, (y - ha) / H_vig
                    width, height = w / W_vig, h / H_vig

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

import cv2
df = pd.read_csv(PATH + 'grapevine_annotation.csv')
new_df = []
for image_name in df['image_name'].unique():
    print(image_name)
    s = df[df['image_name'] == image_name]
    img = cv2.cvtColor(cv2.imread(PATH + 'images/{}'.format(image_name)), cv2.COLOR_BGR2RGB)
    new_s = mean_hue_berry(img, s)
    new_df.append(new_s)
new_df = pd.concat(new_df)

# hue = ((180 - np.array(new_df['hue'])) - 100) % 180
# new_df['hue'] = hue

new_df.to_csv(PATH + 'grapevine_annotation.csv', index=False)

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
























