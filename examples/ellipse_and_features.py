import cv2

import pandas as pd

import os
import matplotlib.pyplot as plt
# os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from deepberry.src.openalea.deepberry.detection_and_segmentation import berry_detection, berry_segmentation, \
    load_berry_models
from deepberry.src.openalea.deepberry.features_extraction import berry_features_extraction

from deepberry.src.openalea.deepberry.utils import ellipse_interpolation

# TODO remove =======================================

"""
data (PhenoArch 2021 - plantid 7794 - task 3786 - angle 120)
"""

# plantid = 7794

exp = 'ARCH2021-05-27'
res = pd.read_csv('X:/phenoarch_cache/cache_{0}/full_results_temporal_{0}.csv'.format(exp))

selec = res[(res['plantid'] == 7794) & (res['angle'] == 120)]
for task in selec.sort_values('timestamp')['task'].unique()[::2]:
    s = selec[selec['task'] == task]
    s2 = s[['ell_x', 'ell_y', 'ell_w', 'ell_h', 'ell_a', 'score', 'hue_scaled', 'area', 'volume', 'roundness']]
    s.to_csv('deepberry/examples/data/temporal/{}.csv'.format(s['timestamp'].iloc[0]))


index = pd.read_csv('data/grapevine/image_index.csv')
index = index[index['imgangle'].notnull()]

s_index = index[(index['exp'] == exp) & (index['plantid'] == 7794) & (index['imgangle'] == 120)]

img = cv2.cvtColor(cv2.imread('Z:/{}/{}/{}.png'.format(exp, row_img['taskid'], row_img['imgguid'])),
                   cv2.COLOR_BGR2RGB)

cv2.imwrite('deepberry/examples/data/image.png', cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

"""
PREREQUISITES
"""

"""
Load models and image
"""

model_detection, model_segmentation = load_berry_models('deepberry/examples/data/model')
image = cv2.cvtColor(cv2.imread('deepberry/examples/data/image.png'), cv2.COLOR_BGR2RGB)

score_threshold_detection = 0.89

"""
Display image
"""

plt.imshow(image)

"""
PIPELINE
"""

# berry detection (Yolov4 deep-learning model)
res_det = berry_detection(image=image, model=model_detection, score_threshold=score_threshold_detection)

plt.imshow(image)
for _, (x, y, w, h, score) in res_det[['x', 'y', 'w', 'h', 'score']].iterrows():
    plt.plot([x, x, x + w, x + w, x], [y, y + h, y + h, y, y], 'r-')

# berry ellipse-segmentation (U-Net deep-learning model)
res_seg = berry_segmentation(image=image, model=model_segmentation, boxes=res_det)

plt.imshow(image)
for _, (xe, ye, we, he, ae) in res_seg[['ell_x', 'ell_y', 'ell_w', 'ell_h', 'ell_a']].iterrows():
    lsp_x, lsp_y = ellipse_interpolation(x=xe, y=ye, w=we, h=he, a=ae, n_points=50)
    plt.plot(lsp_x, lsp_y, 'r-')

# berry features extraction (area/volume, color, roundness)
res = berry_features_extraction(image=image, ellipses=res_seg)

fig = plt.figure()
for k, feature in enumerate(['volume', 'hue_scaled', 'roundness']):
    ax = plt.subplot(3, 1, k + 1)
    ax.set_title(feature)
    plt.hist(res[feature], 30)
fig.tight_layout()





