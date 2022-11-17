import os
import matplotlib.pyplot as plt
import numpy as np
import cv2
import pandas as pd

import shutil

from deepberry.src.openalea.deepberry.utils import ellipse_interpolation, get_image_path


PATH = 'data/grapevine/results/'

colors = {'WW': 'dodgerblue', 'WD1': 'orange', 'WD2': 'red', 'normal': 'black', 'odium': 'green'}
symbols = {'A02-PL6': '.:', 'BARESA': '*-', 'PRIMITIV': 'o--'}

index = pd.read_csv('data/grapevine/image_index.csv')

PALETTE = np.array(
    [[255, 0, 0], [0, 255, 0], [0, 0, 255], [255, 255, 0], [204, 121, 167], [0, 158, 115],
     [0, 114, 178], [230, 159, 0], [140, 86, 75], [0, 255, 255], [255, 0, 100], [0, 77, 0], [100, 0, 255],
     [100, 0, 0], [0, 0, 100], [100, 100, 0], [0, 100, 100], [100, 0, 100], [255, 100, 100]])
PALETTE = np.array(20 * list(PALETTE) + [[0, 0, 0]])

# ===== image + ellipses ============================================================================================

# plantid = 415
# task = 5944
# angle = 240
# s = image_index[(image_index['plantid'] == plantid) & (image_index['task'] == task) & (image_index['angle'] == angle)]

# 'V:/ARCH2022-05-18/5928/dd3debb4-5e13-4430-b83b-dd25c71f0b61.png'

s = df[df['exp'] == 'ARCH2022-05-18'].sample().iloc[0]  # random ellipse

img_path = get_image_path(index, s['plantid'], s['task'], s['angle'], disk='Z:/')
img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
ellipses = pd.read_csv('X:/phenoarch_cache/cache_{}/{}/{}_{}.csv'.format(s['exp'], s['plantid'], s['task'], s['angle']))

plt.figure()
plt.imshow(img)
for _, ell in ellipses.iterrows():
    lsp_x, lsp_y = ellipse_interpolation(x=ell['ell_x'], y=ell['ell_y'], w=ell['ell_w'],
                                         h=ell['ell_h'], a=ell['ell_a'], n_points=100)
    plt.plot(lsp_x, lsp_y, 'red', linewidth=1)
    # if ell['black']:
    #     plt.plot(ell['ell_x'], ell['ell_y'], 'wx')












