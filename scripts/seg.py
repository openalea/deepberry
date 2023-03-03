"""
Quick berry detection & segmentation on an image
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt

from deepberry.src.openalea.deepberry.utils import ellipse_interpolation
from deepberry.src.openalea.deepberry.ellipse_segmentation import load_berry_models, berry_detection, \
    berry_segmentation

model_detection, model_segmentation = load_berry_models('deepberry/examples/data/model')
image = cv2.cvtColor(cv2.imread('deepberry/examples/data/image/image.png'), cv2.COLOR_BGR2RGB)

# image = cv2.resize(image, (np.array(image.shape)[:2][::-1] / 8).astype(int))

score_threshold_detection = 0.5
res_det = berry_detection(image=image, model=model_detection, score_threshold=score_threshold_detection)
res_seg = berry_segmentation(image=image, model=model_segmentation, boxes=res_det, nms_threshold_ell=0.7)

plt.imshow(image)
for _, (xe, ye, we, he, ae) in res_seg[['ell_x', 'ell_y', 'ell_w', 'ell_h', 'ell_a']].iterrows():
    lsp_x, lsp_y = ellipse_interpolation(x=xe, y=ye, w=we, h=he, a=ae, n_points=50)
    plt.plot(lsp_x, lsp_y, 'r-')























