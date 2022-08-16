import pandas as pd
import numpy as np
from skimage import io
import cv2
import os

from grapevine.prediction import berry_is_black

PATH = 'data/grapevine/'

anot = pd.read_csv(PATH + 'dataset/grapevine_annotation.csv')

#image_name = np.random.choice(anot['image_name'])

for channel in ['r', 'g', 'b', 'h', 's', 'v']:
    if not os.path.isdir(PATH + 'visu_color3/' + channel):
        os.mkdir(PATH + 'visu_color3/' + channel)

for image_name in anot['image_name'].unique():

    s = anot[anot['image_name'] == image_name]

    img = io.imread(PATH + 'dataset/images/' + image_name)[:, :, :3]

    for k in range(10):

        row = s.sample().iloc[0]

        mask = cv2.ellipse(np.float32(img[:, :, 0] * 0),
                            (round(row['ell_x']), round(row['ell_y'])),
                            (round(row['ell_w'] / 2), round(row['ell_h'] / 2)),
                            row['ell_a'], 0., 360,
                            (1), -1)

        pixels = img[mask == 1]
        pixels_hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)[mask == 1]

        print(len(pixels_hsv), 'pixels')

        medians = np.median(pixels, axis=0)
        medians_hsv = np.median(pixels_hsv, axis=0)

        green = medians[1]
        hue = medians_hsv[0]
        if 35 < green < 80 and hue <= 41:

            v = img * np.dstack([mask, mask, mask]).astype('uint8')
            v2 = v[(round(row['box_y']) - round(0.55 * row['box_h'])):(round(row['box_y']) + round(0.55 * row['box_h'])),
                 (round(row['box_x']) - round(0.55 * row['box_w'])):(round(row['box_x']) + round(0.55 * row['box_w']))]
            v2[v2 == [0, 0, 0]] = 255

            # plt.figure(str(medians))
            # plt.imshow(v2)

            if v2.size != 0:
                rd_letters = ''.join(np.random.choice(list('abcdefghijklmnopqrstuvwxyz'), 5))
                for k, channel in enumerate(['r', 'g', 'b']):
                    io.imsave(PATH + 'visu_color3/{}/{}_{}.png'.format(channel, int(medians[k]), rd_letters), v2)
                for k, channel in enumerate(['h', 's', 'v']):
                    io.imsave(PATH + 'visu_color3/{}/{}_{}.png'.format(channel, int(medians_hsv[k]), rd_letters), v2)


# ==========================================================================================================

for image_name in anot['image_name'].unique():

    s = anot[anot['image_name'] == image_name]

    img = io.imread(PATH + 'dataset/images/' + image_name)[:, :, :3]

    for k in range(10):

        row = s.sample().iloc[0]

        black = berry_is_black(image=img, ell_parameters=list(row[['ell_x', 'ell_y', 'ell_w', 'ell_h', 'ell_a']]))

        mask = cv2.ellipse(np.float32(img[:, :, 0] * 0),
                           (round(row['ell_x']), round(row['ell_y'])),
                           (round(row['ell_w'] / 2), round(row['ell_h'] / 2)),
                           row['ell_a'], 0., 360,
                           (1), -1)

        v = img * np.dstack([mask, mask, mask]).astype('uint8')
        v2 = v[(round(row['box_y']) - round(0.55 * row['box_h'])):(
                    round(row['box_y']) + round(0.55 * row['box_h'])),
             (round(row['box_x']) - round(0.55 * row['box_w'])):(
                         round(row['box_x']) + round(0.55 * row['box_w']))]
        v2[v2 == [0, 0, 0]] = 255

        if v2.size != 0:
            rd_letters = ''.join(np.random.choice(list('abcdefghijklmnopqrstuvwxyz'), 5))
            if black:
                io.imsave(PATH + 'visu_color4/black/{}.png'.format(rd_letters), v2)
            else:
                io.imsave(PATH + 'visu_color4/green/{}.png'.format(rd_letters), v2)







































