import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from shapely.geometry import Polygon
from sklearn.metrics import r2_score

from deepberry.src.openalea.deepberry.utils import ellipse_interpolation
from deepberry.src.openalea.deepberry.detection_and_segmentation import berry_detection, VIGNETTE_SIZE_SEG, BERRY_SCALING_SEG

# https://github.com/yfpeng/object_detection_metrics
# (which is adapted from https://github.com/rafaelpadilla/Object-Detection-Metrics)
from podm.metrics import get_pascal_voc_metrics, MetricPerClass, get_bounding_boxes, BoundingBox
from podm.box import intersection_over_union

from keras.models import load_model


# ===================================================================================================================

path = 'Y:/lepseBinaries/Trained_model/deepberry/new/'
weights_path = path + '/detection.weights'
config_path = path + '/detection.cfg'
net = cv2.dnn.readNetFromDarknet(config_path, weights_path)
model_det = cv2.dnn_DetectionModel(net)

seg_name = 'segmentation_ell3.h5'

model = load_model(path + seg_name, custom_objects={'dice_coef': None}, compile=False)

image_name = 'ARCH2021-05-27_7791_3835_300.png'
# image_name = 'ARCH2022-05-18_1685_5601_180.png'

image = cv2.cvtColor(cv2.imread('data/grapevine/dataset/image_valid/' + image_name), cv2.COLOR_BGR2RGB)

boxes = berry_detection(image=image, model=model_det, score_threshold=0.985)


ds = int(VIGNETTE_SIZE_SEG / 2)
res = []
seg_vignettes = {}

for row_index, row in boxes.iterrows():
    x, y, w, h, score = row[['x', 'y', 'w', 'h', 'score']]
    x_vignette, y_vignette = round(x + (w / 2)), round(y + (h / 2))
    zoom = (VIGNETTE_SIZE_SEG * BERRY_SCALING_SEG) / np.max((w, h))

    ya, yb = round(y_vignette - (ds / zoom)), round(y_vignette + (ds / zoom))
    xa, xb = round(x_vignette - (ds / zoom)), round(x_vignette + (ds / zoom))

    # check if enough space to unzoom in case of big berry
    enough_space = (0 <= ya) and (yb < image.shape[0]) and (0 <= xa) and (xb <= image.shape[1])

    if enough_space:
        seg_vignette = cv2.resize(image[ya:yb, xa:xb], (VIGNETTE_SIZE_SEG, VIGNETTE_SIZE_SEG))
        seg_vignette = (seg_vignette / 255.).astype(np.float64)
        seg_vignettes[row_index] = seg_vignette
    else:
        print('not enough space')

if seg_vignettes:

    # segmentation mask prediction (all vignettes at once ~= 2x faster)
    multi_seg = model.predict(np.array(list(seg_vignettes.values())), verbose=0)
    # multi_seg = multi_seg[:, :, :, 0]
    # multi_seg = (multi_seg > 0.5).astype(np.uint8) * 255  # important to get correct contours !

    plt.figure()
    K = 6
    Kj = 2
    for enum, k in enumerate(range(Kj * (K ** 2), (Kj + 1) * (K ** 2))):
        ax = plt.subplot(K, K, enum + 1)
        # plt.imshow(list(seg_vignettes.values())[k])
        # seg = (multi_seg[k] > 0.5).astype(int)
        seg = np.argmax(multi_seg[k], axis=2)
        # plt.imshow(seg * 255.)
        # edges, _ = cv2.findContours(multi_seg[k], cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        # if edges:
        #     edges = edges[np.argmax([len(e) for e in edges])][:, 0, :]  # takes longest contour if several areas
        edges = np.array(np.where(seg == 0))[::-1].T

        plt.imshow(list(seg_vignettes.values())[k])
        edges = edges[::1]
        plt.plot(edges[:, 0], edges[:, 1], 'r.', markersize=0.7)
        # plt.imshow(seg)

#     for row_index, seg in zip(seg_vignettes.keys(), multi_seg):
#
#         # again
#         x, y, w, h, score = boxes.loc[row_index][['x', 'y', 'w', 'h', 'score']]
#         x_vignette, y_vignette = round(x + (w / 2)), round(y + (h / 2))
#         zoom = (VIGNETTE_SIZE_SEG * BERRY_SCALING_SEG) / np.max((w, h))
#
#         # extraction of the mask edges
#         # edges, _ = cv2.findContours(seg, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
#         # if edges:
#         #     edges = edges[np.argmax([len(e) for e in edges])][:, 0, :]  # takes longest contour if several areas
#         edges = np.array(np.where(seg == 255))[::-1].T
#
#         if len(edges) >= 5:  # cv2.fitEllipse() requires >= 5 points
#
#             # fit ellipse in the zoomed / centered vignette space
#             (ell_x, ell_y), (ell_w, ell_h), ell_a = cv2.fitEllipse(edges)
#
#             # rescale to raw image space
#             ell_x_raw, ell_y_raw = x_vignette + ((ell_x - ds) / zoom), y_vignette + ((ell_y - ds) / zoom)
#             ell_w_raw, ell_h_raw = ell_w / zoom, ell_h / zoom
#             ell_a_raw = ell_a
#
#             res.append([ell_x_raw, ell_y_raw, ell_w_raw, ell_h_raw, ell_a_raw, score])
#
#         else:
#             print('len(edges) < 5')
#
# res = pd.DataFrame(res, columns=['ell_x', 'ell_y', 'ell_w', 'ell_h', 'ell_a', 'score'])


# ====================================================================================================================

from math import sin, cos, atan2, pi, fabs
import numpy as np
import matplotlib.pyplot as plt

# https://gist.github.com/michalpelka/ddb66f9c3109f5fc183a6cab917d949b


def ellipe_tan_dot(rx, ry, px, py, theta):
    '''Dot product of the equation of the line formed by the point
    with another point on the ellipse's boundary and the tangent of the ellipse
    at that point on the boundary.
    '''
    return ((rx ** 2 - ry ** 2) * cos(theta) * sin(theta) -
            px * rx * sin(theta) + py * ry * cos(theta))


def ellipe_tan_dot_derivative(rx, ry, px, py, theta):
    '''The derivative of ellipe_tan_dot.
    '''
    return ((rx ** 2 - ry ** 2) * (cos(theta) ** 2 - sin(theta) ** 2) -
            px * rx * cos(theta) - py * ry * sin(theta))


def estimate_distance(x, y, rx, ry, x0=0, y0=0, angle=0, error=1e-5):
    '''Given a point (x, y), and an ellipse with major - minor axis (rx, ry),
    its center at (x0, y0), and with a counter clockwise rotation of
    `angle` degrees, will return the distance between the ellipse and the
    closest point on the ellipses boundary.
    '''
    x -= x0
    y -= y0
    if angle:
        # rotate the points onto an ellipse whose rx, and ry lay on the x, y
        # axis
        angle = -pi / 180. * angle
        x, y = x * cos(angle) - y * sin(angle), x * sin(angle) + y * cos(angle)

    theta = atan2(rx * y, ry * x)
    while fabs(ellipe_tan_dot(rx, ry, x, y, theta)) > error:
        theta -= ellipe_tan_dot(
            rx, ry, x, y, theta) / \
            ellipe_tan_dot_derivative(rx, ry, x, y, theta)

    px, py = rx * cos(theta), ry * sin(theta)
    return ((x - px) ** 2 + (y - py) ** 2) ** .5


# ====================================================================================================================

from deepberry.src.openalea.deepberry.utils import ellipse_interpolation

for k in range(100)[::10]:

    seg = multi_seg[k]
    edges, _ = cv2.findContours(seg, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    if edges:
        edges = edges[np.argmax([len(e) for e in edges])][:, 0, :]  # takes longest contour if several areas

    dist_threshold = 1.
    min_size = 0.5
    best_score = float('inf')
    best_parameters = [None] * 5
    for _ in range(50):

        indexes = np.random.choice(np.arange(len(edges)), 5, replace=False)
        random_points = edges[indexes]

        # fit to random_points
        (ell_x, ell_y), (ell_w, ell_h), ell_a = cv2.fitEllipse(random_points)
        # evaluate vs all points
        xy = ellipse_interpolation(x=ell_x, y=ell_y, w=ell_w, h=ell_h, a=ell_a, n_points=5000).T
        dists = np.array([np.min(np.sqrt(np.sum((xy - p) ** 2, axis=1))) for p in edges])

        close_points = edges[dists < dist_threshold]
        if len(close_points) > int(min_size * len(edges)):

            # fit to close_points
            (ell_x, ell_y), (ell_w, ell_h), ell_a = cv2.fitEllipse(close_points)
            # evaluate vs close_points
            xy = ellipse_interpolation(x=ell_x, y=ell_y, w=ell_w, h=ell_h, a=ell_a, n_points=5000).T
            dists = np.array([np.min(np.sqrt(np.sum((xy - p) ** 2, axis=1))) for p in close_points])

            score = np.sum(dists)
            if score < best_score:
                best_score = score
                best_parameters = ell_x, ell_y, ell_w, ell_h, ell_a
            print(score)
    print(f'best_score: {best_score:.2f}')


    (ell_x, ell_y), (ell_w, ell_h), ell_a = cv2.fitEllipse(edges)
    xy = ellipse_interpolation(x=ell_x, y=ell_y, w=ell_w, h=ell_h, a=ell_a, n_points=100).T

    ell_x, ell_y, ell_w, ell_h, ell_a = best_parameters
    xy_ransac = ellipse_interpolation(x=ell_x, y=ell_y, w=ell_w, h=ell_h, a=ell_a, n_points=100).T

    plt.figure()
    plt.gca().set_aspect('equal', adjustable='box')
    plt.imshow(list(seg_vignettes.values())[k])
    plt.plot(edges[:, 0], edges[:, 1], '.', color='yellow')
    plt.plot(xy[:, 0], xy[:, 1], 'r-')
    plt.plot(xy[:, 0], xy[:, 1], '-', color='grey')
    plt.plot(xy_ransac[:, 0], xy_ransac[:, 1], 'r-')

plt.figure()
plt.imshow(multi_seg[k])
plt.plot(xy[:, 0], xy[:, 1], 'r-')


# ===================================================================================================================

plt.gca().set_aspect('equal', adjustable='box')  # same x y scale
plt.plot(xy[:, 0], xy[:, 1], 'k-')
plt.plot(edges[:, 0], edges[:, 1], 'k.')
plt.scatter(close_points[:, 0], close_points[:, 1], c=dists[dists < th])
plt.colorbar()


plt.figure()
plt.gca().set_aspect('equal', adjustable='box')  # same x y scale
ell_x, ell_y, ell_w, ell_h, ell_a = 50, 50, 35, 80, -45
lsp_x, lsp_y = ellipse_interpolation(x=ell_x, y=ell_y, w=ell_w, h=ell_h, a=ell_a, n_points=300)
plt.plot(lsp_x, lsp_y, 'k-')




import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
rx, ry = ell_h / 2, ell_w / 2  # major, minor ellipse axis
x0, y0 = ell_x, ell_y  # center point of the ellipse
angle = - ell_a  # ellipse's rotation counter clockwise
sx, sy = s = 100, 100  # size of the canvas background

dist = np.zeros(s)
for x in range(sx):
    for y in range(sy):
        dist[x, y] = estimate_distance(x, y, rx, ry, x0, y0, angle)

plt.figure()
plt.imshow(dist.T, extent=(0, sx, 0, sy), origin="lower")
plt.colorbar()
ax = plt.gca()
ellipse = Ellipse(xy=(x0, y0), width=2 * rx, height=2 * ry, angle=angle,
                  edgecolor='r', fc='None', linestyle='dashed')
ax.add_patch(ellipse)
plt.show()











