import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score

# https://github.com/yfpeng/object_detection_metrics
# (which is adapted from https://github.com/rafaelpadilla/Object-Detection-Metrics)
from podm.metrics import get_pascal_voc_metrics, BoundingBox
from podm.box import intersection_over_union


IOU_THRESHOLD = 0.75

DIR_VALIDATION = 'data/grapevine/validation/'

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


# def IOU(pol1_xy, pol2_xy):
#     """
#     https://stackoverflow.com/questions/58435218/intersection-over-union-on-non-rectangular-quadrilaterals
#     """
#     # Define each polygon
#     polygon1_shape = Polygon(pol1_xy)
#     polygon2_shape = Polygon(pol2_xy)
#
#     # Calculate intersection and union, and the IOU
#     polygon_intersection = polygon1_shape.intersection(polygon2_shape).area
#     polygon_union = polygon1_shape.area + polygon2_shape.area - polygon_intersection
#     return polygon_intersection / polygon_union


# ===== detection metrics =============================================================================================

val_det = pd.read_csv('data/grapevine/validation/validation_detection.csv')
val_det['exp'] = [n.split('_')[0] for n in val_det['image_name']]

# Precision-Recall
metric = get_metrics(val_det, iou_threshold=IOU_THRESHOLD)
p, r = metric.interpolated_precision, metric.interpolated_recall
f1_score = (2 * p * r) / (p + r)
df_pred = val_det[val_det['type'] == 'pred']
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

# generates metrics / image_name for best_iteration
df_name = []
for image_name in val_det['image_name'].unique():
    dfin = val_det[val_det['image_name'] == image_name]
    metric = get_metrics(dfin, iou_threshold=IOU_THRESHOLD)
    df_name.append([image_name, dfin.iloc[0]['exp'], metric.ap, metric.num_groundtruth])
    plt.plot(metric.interpolated_precision, metric.interpolated_recall, 'k-')
df_name = pd.DataFrame(df_name, columns=['image_name', 'exp', 'ap', 'n']).sort_values('ap')

# visualise detection for one image
image_name = 'ARCH2021-05-27_7794_3942_60.png'
val_det_n = val_det[val_det['image_name'] == image_name]
img = cv2.cvtColor(cv2.imread('data/grapevine/dataset/image_valid/' + image_name), cv2.COLOR_BGR2RGB)
plt.figure()
plt.imshow(img)
for _, row in val_det_n.iterrows():
    if row['score'] > score_threshold:
        col = 'greenyellow' if row['type'] == 'obs' else 'r'
        linestyle = '--' if row['type'] == 'obs' else '-'
        linewidth = 1. if row['type'] == 'obs' else 1.6
        x1, x2, y1, y2 = row[['x1', 'x2', 'y1', 'y2']]
        plt.plot([x1, x1, x2, x2, x1], [y1, y2, y2, y1, y1], linestyle=linestyle, linewidth=linewidth, color=col)

# ===== segmentation metrics ==========================================================================================

# load det + seg validation dataset
val_seg = pd.read_csv(DIR_VALIDATION + 'validation_segmentation.csv')

# segmentation variables (on ellipses)
val_seg['area'] = (val_seg['ell_w'] / 2) * (val_seg['ell_h'] / 2) * np.pi
val_seg['volume'] = (4 / 3) * np.pi * ((np.sqrt(val_seg['area'] / np.pi)) ** 3)
val_seg['roundness'] = val_seg['ell_w'] / val_seg['ell_h']  # always w <= h

# generate couples of (obs, pred) for seg validation
box_couples = []
n, tp = 0, 0
for image_name in val_seg['image_name'].unique():
    dfn = val_seg[val_seg['image_name'] == image_name]
    obs, pred = dfn[dfn['type'] == 'obs'], dfn[dfn['type'] == 'pred']
    obs_box = get_boxes(obs)
    pred_box = get_boxes(pred)
    n += len(pred_box)
    for k2, box2 in enumerate(pred_box):
        iou_list = [intersection_over_union(box1, box2) for box1 in obs_box]
        if max(iou_list) > IOU_THRESHOLD:  # filter couples that are not close enough to be considered as a match
            k1 = np.argmax(iou_list)
            box_couples.append([obs.iloc[k1], pred.iloc[k2]])
            tp += 1

# a) obs vs pred
var = 'area'
# ['DYN2020-05-15'], ['ARCH2021-05-27'], ['ARCH2022-05-18']
selec_box_couples = box_couples
# selec_box_couples = [b for b in box_couples if b[0]['image_name'].split('_')[0] == 'ARCH2022-05-18']
x = np.array([row[var] for row, _ in selec_box_couples])
y = np.array([row[var] for _, row in selec_box_couples])

rmse = np.sqrt(np.sum((x - y) ** 2) / len(x))
r2 = r2_score(x, y)
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
