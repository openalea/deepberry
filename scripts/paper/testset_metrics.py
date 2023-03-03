import cv2
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import linregress

from deepberry.src.openalea.deepberry.utils import ellipse_interpolation
from deepberry.src.openalea.deepberry.training.training_dataset import labelme_json_postprocessing

# podm : https://github.com/yfpeng/object_detection_metrics (pip install object_detection_metrics)
# (adapted from https://github.com/rafaelpadilla/Object-Detection-Metrics)
from podm.metrics import get_pascal_voc_metrics, BoundingBox
from podm.box import intersection_over_union

IOU_THRESHOLD = 0.5

DIR_PREDS = 'data/grapevine/paper/test_set_predictions/'

DIR_OUTPUT = 'data/grapevine/paper/'


def get_boxes(df):
    boxes = []
    for _, row in df.iterrows():
        bb = BoundingBox.of_bbox(image=row['image_name'], category='berry', xtl=row['x1'], ytl=row['y1'],
                                 xbr=row['x2'], ybr=row['y2'], score=row['score'])
        boxes.append(bb)

    return boxes


def get_metrics(df, iou_threshold):
    """
    returns a podm.metrics.MetricPerClass with various metrics attributes:
    ap, precision, interpolated_recall, interpolated_precision, tp, fp, num_groundtruth, num_detection
    """
    gt_BoundingBoxes = get_boxes(df[df['type'] == 'obs'])
    pd_BoundingBoxes = get_boxes(df[df['type'] == 'pred'])
    res = get_pascal_voc_metrics(gt_BoundingBoxes, pd_BoundingBoxes, iou_threshold)['berry']
    return res

# ===== new annotations: additional labels for the test set (FP -> TP) ============================================


anot_new = []
fd = 'data/grapevine/paper/anot_test_fp/'
for file in os.listdir(fd):
    dfi = labelme_json_postprocessing(fd + file)
    dfi['image_name'] = file.replace('.json', '.png')
    anot_new.append(dfi)
anot_new = pd.concat(anot_new)


# ===== detection metrics =============================================================================================

include_new_annotations = True

small_threshold = 1000
remove_small = True

det = pd.read_csv(DIR_PREDS + 'detection.csv')
if include_new_annotations:
    add = []
    for _, row in anot_new.iterrows():
        score = 1
        x, y, w, h = row[['box_x', 'box_y', 'box_w', 'box_h']]
        x, y = x - w / 2, y - h / 2
        add.append(['obs', x, y, x + w, y + h, 1, row['image_name']])
    det = pd.concat((det, pd.DataFrame(add, columns=det.columns)))
det['exp'] = [n.split('_')[0] for n in det['image_name']]

if remove_small:
    det['area_box'] = (det['x2'] - det['x1']) * (det['y2'] - det['y1'])
    det = det[det['area_box'] > small_threshold]

# Precision-Recall
metric = get_metrics(det, iou_threshold=IOU_THRESHOLD)
p, r = metric.interpolated_precision, metric.interpolated_recall
f1_score = (2 * p * r) / (p + r)
df_pred = det[det['type'] == 'pred']
plt.plot(r, p, 'k-')
scores = np.array(sorted(df_pred['score'])[::-1])
score_threshold = scores[np.argmax(f1_score)]
plt.title('Precision-Recall curve')
plt.xlabel('Recall = TP / (TP + FN)')
plt.ylabel('Precision = TP / (TP + FP)')
for s in [0.999, score_threshold, 0.5]:
    k = np.argmin(np.abs(scores - s))
    plt.plot(r[k], p[k], 'o', label=round(s, 4))
plt.legend()

# previous graph can be used to select the score threshold and compute automatically the corresponding metrics
precision, recall, f1 = p[np.argmax(f1_score)], r[np.argmax(f1_score)], np.max(f1_score)
print(f'score threshold with max F1-score: s={score_threshold:.4f}')
print(f'precision: {100 * precision:.1f}%, recall: {100 * recall:.1f}%, F1-score: {100 * f1:.1f}%')

# final choice on score_threshold
# score_threshold = 0.985
score_threshold = 0.89

metric_threshold = get_metrics(det[det['score'] > score_threshold], iou_threshold=IOU_THRESHOLD)
print(f"TP = {int(metric_threshold.tp)}, FP = {int(metric_threshold.fp)}, "
      f"FN = {int(len(det[det['type'] == 'obs']) - metric_threshold.tp)}")

# visualise detection for one image
image_name = 'ARCH2022-05-18_22_5928_0.png'
det_n = det[det['image_name'] == image_name]
img = cv2.cvtColor(cv2.imread('data/grapevine/dataset/dataset_raw/image_test/' + image_name), cv2.COLOR_BGR2RGB)
plt.figure()
plt.imshow(img)
for _, row in det_n.iterrows():
    if row['score'] > score_threshold:
        col = 'greenyellow' if row['type'] == 'obs' else 'r'
        linestyle = '--' if row['type'] == 'obs' else '-'
        linewidth = 1. if row['type'] == 'obs' else 1.6
        x1, x2, y1, y2 = row[['x1', 'x2', 'y1', 'y2']]
        plt.plot([x1, x1, x2, x2, x1], [y1, y2, y2, y1, y1], linestyle=linestyle, linewidth=linewidth, color=col)

# ===== segmentation metrics ==========================================================================================

include_new_annotations = True

# load det + seg validation dataset
seg = pd.read_csv(DIR_PREDS + 'segmentation.csv')
if include_new_annotations:
    add = []
    for _, row in anot_new.iterrows():
        score = 1
        x, y, w, h = row[['box_x', 'box_y', 'box_w', 'box_h']]
        ex, ey, ew, eh, ea = row[['ell_x', 'ell_y', 'ell_w', 'ell_h', 'ell_a']]
        x, y = x - w / 2, y - h / 2
        add.append(['obs', x, y, x + w, y + h, ex, ey, ew, eh, ea, 1, row['image_name']])
    seg = pd.concat((seg, pd.DataFrame(add, columns=seg.columns)))
seg['exp'] = [n.split('_')[0] for n in seg['image_name']]


# segmentation variables (on ellipses)
seg['area'] = (seg['ell_w'] / 2) * (seg['ell_h'] / 2) * np.pi
seg['volume'] = (4 / 3) * np.pi * ((np.sqrt(seg['area'] / np.pi)) ** 3)
seg['roundness'] = seg['ell_w'] / seg['ell_h']  # always w <= h

# generate couples of (obs, pred) for seg validation
box_couples = []
tp, fn = 0, 0
seg['det'] = None
for image_name in seg['image_name'].unique():
    obs = seg[(seg['image_name'] == image_name) & (seg['type'] == 'obs')]
    pred = seg[(seg['image_name'] == image_name) & (seg['type'] == 'pred')]
    obs_box = get_boxes(obs)
    pred_box = get_boxes(pred)
    for k1, box1 in enumerate(obs_box):
        iou_list = [intersection_over_union(box1, box2) for box2 in pred_box]
        if max(iou_list) > IOU_THRESHOLD:  # filter couples that are not close enough to be considered as a match
            # a pred was found as a match for this obs : TP
            tp += 1
            box_couples.append([obs.iloc[k1], pred.iloc[np.argmax(iou_list)]])
        else:
            # no pred was found for this obs : FN
            fn += 1

# a) obs vs pred
var = 'area'
# ['DYN2020-05-15'], ['ARCH2021-05-27'], ['ARCH2022-05-18']
selec_box_couples = box_couples
# selec_box_couples = [b for b in box_couples if b[0]['image_name'].split('_')[0] == 'ARCH2022-05-18']
x = np.array([row[var] for row, _ in selec_box_couples])
y = np.array([row[var] for _, row in selec_box_couples])

rmse = np.sqrt(np.sum((x - y) ** 2) / len(x))
r2 = linregress(x, y)[2] ** 2
bias = np.mean(y - x)
mape = 100 * np.mean(np.abs((x - y) / x))

fig, ax = plt.subplots(figsize=(10, 10), dpi=100)
ax.tick_params(axis='both', which='major', labelsize=20)  # axis number size
plt.gca().set_aspect('equal', adjustable='box')  # same x y scale
plt.title('Berry area', fontsize=35)
plt.xlabel('observation (px²)', fontsize=30)
plt.ylabel('prediction (px²)', fontsize=30)
max_value = max(np.concatenate((x, y)))
plt.xlim(-0.02 * max_value, 1.02 * max_value)
plt.ylim(-0.02 * max_value, 1.02 * max_value)
plt.plot([-max_value, 2 * max_value], [-max_value, 2 * max_value], '-', color='grey')
plt.plot(x, y, 'r.', alpha=0.25, markersize=10)
plt.text(0.52, 0.03, f'n = {len(x)} \nR² = {r2:.3f} \nRMSE = {rmse:.1f}px² \nBias = {bias:.1f}px² \nMAPE = {mape:.1f}%',
         horizontalalignment='left', verticalalignment='bottom', transform=ax.transAxes, fontsize=30)

plt.savefig(DIR_OUTPUT + 'val_seg.png', bbox_inches='tight')

# ===== further exploration of detection errors =======================================================================

# load det + seg validation dataset
seg = pd.read_csv(DIR_PREDS + 'segmentation.csv')

# image_names = ['DYN2020-05-15_7233_2424_330.png', 'ARCH2021-05-27_7772_3793_30.png', 'ARCH2021-05-27_7783_3942_0.png',
#                'ARCH2022-05-18_1715_5899_0.png', 'ARCH2022-05-18_2448_6054_60.png', 'DYN2020-05-15_7232_2656_330.png',
#                'DYN2020-05-15_7236_2530_330.png', 'DYN2020-05-15_7238_2388_330.png', 'DYN2020-05-15_7236_2530_330.png',
#                'ARCH2022-05-18_1685_5601_180.png', 'ARCH2021-05-27_7788_3785_270.png']

image_names = list(seg['image_name'].unique())

n_fn, n_fp = 0, 0
areas_fn, areas_fp = [], []
for image_name in image_names:

    dfn = seg[seg['image_name'] == image_name]

    # generate couples of (obs, pred) for seg validation
    box_couples = []
    n, tp = 0, 0

    obs, pred = dfn[dfn['type'] == 'obs'], dfn[dfn['type'] == 'pred']
    obs_box = get_boxes(obs)
    pred_box = get_boxes(pred)
    i_fn, i_tp = [], []
    for k1, box1 in enumerate(obs_box):
        iou_list = [intersection_over_union(box1, box2) for box2 in pred_box]
        if max(iou_list) > IOU_THRESHOLD:  # filter couples that are not close enough to be considered as a match
            # a pred was found as a match for this obs : TP
            i_tp.append(np.argmax(iou_list))
        else:
            # no pred was found for this obs : FN
            i_fn.append(k1)
            areas_fn += [obs.iloc[k1]['area']]

    # if a pred is not a TP, it is a FP
    i_fp = [i for i in range(len(pred_box)) if i not in i_tp]
    areas_fp += [pred.iloc[i]['area'] for i in i_fp]

    # visu
    img = cv2.cvtColor(cv2.imread('data/grapevine/dataset/dataset_raw/image_test/' + image_name), cv2.COLOR_BGR2RGB)
    plt.figure(image_name)
    plt.imshow(img)
    # for _, row in dfn.iterrows():
    #     col = 'greenyellow' if row['type'] == 'obs' else 'r'
    #     linestyle = '--' if row['type'] == 'obs' else '-'
    #     linewidth = 1. if row['type'] == 'obs' else 1.6
    #     x1, x2, y1, y2 = row[['x1', 'x2', 'y1', 'y2']]
    #     plt.plot([x1, x1, x2, x2, x1], [y1, y2, y2, y1, y1], linestyle=linestyle, linewidth=linewidth, color=col)
    #     # xe, ye, we, he, ae = row[['ell_x', 'ell_y', 'ell_w', 'ell_h', 'ell_a']]
    #     # lsp_x, lsp_y = ellipse_interpolation(x=xe, y=ye, w=we, h=he, a=ae, n_points=50)
    #     # plt.plot(lsp_x, lsp_y, linestyle=linestyle, linewidth=linewidth, color=col)

    for i in i_fn:
        row = obs.iloc[i]
        xe, ye, we, he, ae = row[['ell_x', 'ell_y', 'ell_w', 'ell_h', 'ell_a']]
        lsp_x, lsp_y = ellipse_interpolation(x=xe, y=ye, w=we, h=he, a=ae, n_points=50)
        plt.plot(lsp_x, lsp_y, '-', color=('greenyellow' if row['type'] == 'obs' else 'r'))

    for i in i_fp:
        row = pred.iloc[i]
        xe, ye, we, he, ae = row[['ell_x', 'ell_y', 'ell_w', 'ell_h', 'ell_a']]
        lsp_x, lsp_y = ellipse_interpolation(x=xe, y=ye, w=we, h=he, a=ae, n_points=50)
        plt.plot(lsp_x, lsp_y, '-', color=('greenyellow' if row['type'] == 'obs' else 'r'))

    n_fn += len(i_fn)
    n_fp += len(i_fp)

# ======

binwidth = 100
plt.figure()
colors = ['grey', 'red', 'green']
for k, areas in enumerate([seg[seg['type'] == 'obs']['area'], areas_fp, areas_fn]):
    plt.subplot(3, 1, k + 1)
    plt.ylim((0, 28))
    plt.xlim(min(seg['area']) - 500, 9000)  # max(seg['area']) + 500)
    plt.xlabel('Berry area (px²)')
    plt.hist(areas, bins=np.arange(min(areas), max(areas) + binwidth, binwidth), color=colors[k])

