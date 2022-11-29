import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from shapely.geometry import Polygon
from sklearn.metrics import r2_score

from deepberry.src.openalea.deepberry.detection_and_segmentation import berry_detection, berry_segmentation, load_berry_models
from deepberry.src.openalea.deepberry.utils import ellipse_interpolation, get_iou

# https://github.com/yfpeng/object_detection_metrics
# (which is adapted from https://github.com/rafaelpadilla/Object-Detection-Metrics)
from podm.metrics import get_pascal_voc_metrics, MetricPerClass, get_bounding_boxes, BoundingBox
from podm.box import intersection_over_union


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


def IOU(pol1_xy, pol2_xy):
    """
    https://stackoverflow.com/questions/58435218/intersection-over-union-on-non-rectangular-quadrilaterals
    """
    # Define each polygon
    polygon1_shape = Polygon(pol1_xy)
    polygon2_shape = Polygon(pol2_xy)

    # Calculate intersection and union, and the IOU
    polygon_intersection = polygon1_shape.intersection(polygon2_shape).area
    polygon_union = polygon1_shape.area + polygon2_shape.area - polygon_intersection
    return polygon_intersection / polygon_union


MODEL_DET, MODEL_SEG = load_berry_models('Y:/lepseBinaries/Trained_model/deepberry/new/')

# dir containing image_train, image_valid folders
DIR_DATASET = 'data/grapevine/dataset/'

# dataframe generated with the script generate_training_datasets.py
df_annot = pd.read_csv(DIR_DATASET + 'grapevine_annotation.csv')

# where the files created in this script are saved
DIR_VALIDATION = 'data/grapevine/validation/'

# this parameter is kept fixed for the computation of all metrics
IOU_THRESHOLD = 0.75  # match between box1 and box2 if iou(box1, box2) > IOU_THRESHOLD

# =====================================================================================================================

# backup_fd = 'data/grapevine/validation/backup8/'
#
# iterations = [f.split('_')[1].split('.')[0] for f in os.listdir(backup_fd)]
# iterations = sorted([int(iter) for iter in iterations if iter.isdigit()])
# iterations = [i for i in iterations if i > 5000]
#
# df_res_brut = []
# for k_image, image_name in enumerate(df_annot[df_annot['dataset'] == 'valid']['image_name'].unique()):
#
#     obs = df_annot[df_annot['image_name'] == image_name]
#     img = cv2.cvtColor(cv2.imread('data/grapevine/dataset/image_valid/' + image_name), cv2.COLOR_BGR2RGB)
#
#     for iteration in iterations:
#
#         # yolov4 trained model
#         config_path = 'deepberry/scripts/model_training/detection.cfg'
#         weights_path = 'data/grapevine/validation/backup8/detection_{}.weights'.format(iteration)
#         net = cv2.dnn.readNetFromDarknet(config_path, weights_path)
#         model = cv2.dnn_DetectionModel(net)
#
#         t0 = time.time()
#         pred = detect_berry(image=img, model=model, score_threshold=0.)
#         print(k_image, iteration, round(time.time() - t0, 1))
#
#         for _, row in obs.iterrows():
#             x, y, w, h = row[['box_x', 'box_y', 'box_w', 'box_h']]
#             x, y = x - w / 2, y - h / 2
#             df_res_brut.append([iteration, 'obs', x, y, x + w, y + h, 1, image_name])
#
#         for _, row in pred.iterrows():
#             x, y, w, h, score = row
#             df_res_brut.append([iteration, 'pred', x, y, x + w, y + h, score, image_name])
#
# df_res_brut = pd.DataFrame(df_res_brut, columns=['iteration', 'type', 'x1', 'y1', 'x2', 'y2', 'score', 'image_name'])
#
# df_res_brut.to_csv('data/grapevine/validation/validation_backup8_new.csv', index=False)

# ===== run detection on validation set (no score_threshold) ==========================================================

val_det = []
for k_image, image_name in enumerate(df_annot[df_annot['dataset'] == 'valid']['image_name'].unique()):
    print(image_name)

    obs = df_annot[df_annot['image_name'] == image_name]
    img = cv2.cvtColor(cv2.imread(DIR_DATASET + 'image_valid/' + image_name), cv2.COLOR_BGR2RGB)

    # score_threshold=0 bc it's necessary to save all predictions to compute AP metric (and select optimal threshold)
    pred = berry_detection(image=img, model=MODEL_DET, score_threshold=0.)

    for _, row in obs.iterrows():
        score = 1
        x, y, w, h = row[['box_x', 'box_y', 'box_w', 'box_h']]
        x, y = x - w / 2, y - h / 2
        val_det.append(['obs', x, y, x + w, y + h, score, image_name])

    for _, row in pred.iterrows():
        x, y, w, h, score = row
        val_det.append(['pred', x, y, x + w, y + h, score, image_name])

val_det = pd.DataFrame(val_det, columns=['type', 'x1', 'y1', 'x2', 'y2', 'score', 'image_name'])

val_det.to_csv(DIR_VALIDATION + 'validation_detection.csv', index=False)

# ===== run segmentation on validation set (with the selected score_threshold) ========================================

score_threshold = 0.985

val_seg = []
for k_image, image_name in enumerate(df_annot[df_annot['dataset'] == 'valid']['image_name'].unique()):
    print(image_name)

    # TODO remove
    image_name = 'ARCH2021-05-27_7791_3835_300.png'

    obs = df_annot[df_annot['image_name'] == image_name]
    img = cv2.cvtColor(cv2.imread('data/grapevine/dataset/image_valid/' + image_name), cv2.COLOR_BGR2RGB)

    res_det = berry_detection(image=img, model=MODEL_DET, score_threshold=score_threshold)
    pred = berry_segmentation(image=img, model=MODEL_SEG, boxes=res_det)

    for _, row in obs.iterrows():
        score = 1
        x, y, w, h = row[['box_x', 'box_y', 'box_w', 'box_h']]
        x, y = x - w / 2, y - h / 2
        val_seg.append(['obs',
                       x, y, x + w, y + h,
                       row['ell_x'], row['ell_y'], row['ell_w'], row['ell_h'], row['ell_a'],
                       score, image_name])

    # len(res_det) >= len(pred) (not always equal) ,so we use score as berry identifiers. It's a hack and it's not very
    # safe, but most of the time all berries have different scores for a grape. if not the case, take the closest one
    for _, row_seg in pred.iterrows():
        s_det = res_det[res_det['score'] == row_seg['score']]
        if len(s_det) > 1:
            d = np.sum(np.abs(np.array(s_det[['x', 'y']]) - np.array(row_seg[['ell_x', 'ell_y']])), axis=1)
            row_det = s_det.iloc[np.argmin(d)]
        else:
            row_det = s_det.iloc[0]

        val_seg.append(['pred',
                       row_det['x'], row_det['y'], row_det['x'] + row_det['w'], row_det['y'] + row_det['h'],
                       row_seg['ell_x'], row_seg['ell_y'], row_seg['ell_w'], row_seg['ell_h'], row_seg['ell_a'],
                       row_det['score'], image_name])

val_seg = pd.DataFrame(val_seg, columns=['type',
                                       'x1', 'y1', 'x2', 'y2',
                                       'ell_x', 'ell_y', 'ell_w', 'ell_h', 'ell_a',
                                       'score', 'image_name'])

val_seg.to_csv(DIR_VALIDATION + 'validation_segmentation_ell3.csv')

# ===== display metrics =============================================================================================

df = pd.read_csv('data/grapevine/validation/validation_backup8.csv')
df['exp'] = [n.split('_')[0] for n in df['image_name']]

iterations = df['iteration'].unique()

# generates metrics / model iteration
all_results = {}
for exps in [['DYN2020-05-15'], ['ARCH2021-05-27'], ['ARCH2022-05-18'],
            ['ARCH2021-05-27', 'ARCH2022-05-18', 'DYN2020-05-15']]:
    all_results['_'.join(exps)] = {}
    for i in iterations:
        print(i)
        dfi = df[(df['iteration'] == i) & (df['exp'].isin(exps))]
        all_results['_'.join(exps)][i] = get_metrics(dfi, iou_threshold=IOU_THRESHOLD)

# Average Precision (AP) = f(iteration, exp)
for exps in list(all_results.keys()):
    map = [all_results[exps][i].ap for i in iterations]
    plt.plot(iterations, map, '.-', label=exps)
plt.legend()

# previous graph is used to select the best iteration (i.e. best model)
best_iteration = 16000

# Precision-Recall
dfi = df[df['iteration'] == best_iteration]
metric = get_metrics(dfi, iou_threshold=IOU_THRESHOLD)
p, r = metric.interpolated_precision, metric.interpolated_recall
f1_score = (2 * p * r) / (p + r)
dfi_pred = dfi[dfi['type'] == 'pred']
plt.plot(r, p, 'k-')
score_threshold = sorted(dfi_pred['score'])[np.argmax(f1_score)]
plt.title('Precision-Recall curve')
plt.xlabel('Recall = TP / (TP + FN)')
plt.ylabel('Precision = TP / (TP + FP)')
plt.plot(r[np.argmax(f1_score)], p[np.argmax(f1_score)], 'go', markersize=10)
for s in [0.999, 0.5]:
    k = np.argmin(np.abs(np.array(sorted(dfi_pred['score'])[::-1]) - s))
    plt.plot(r[k], p[k], 'ro')

# previous graph can be used to select the score threshold and compute automatically the corresponding metrics
precision, recall, f1 = p[np.argmax(f1_score)], r[np.argmax(f1_score)], np.max(f1_score)
print(f'score threshold with max F1-score: s={score_threshold:.4f}')
print(f'precision: {100 * precision:.1f}%, recall: {100 * recall:.1f}%, F1-score: {100 * f1:.1f}%')

# final choice on score_threshold
score_threshold = 0.985

# generates metrics / image_name for best_iteration
df_name = []
dfi = df[df['iteration'] == best_iteration]
for image_name in dfi['image_name'].unique():
    dfin = dfi[dfi['image_name'] == image_name]
    metric = get_metrics(dfin, iou_threshold=IOU_THRESHOLD)
    df_name.append([image_name, dfin.iloc[0]['exp'], metric.ap, metric.num_groundtruth])
    plt.plot(metric.interpolated_precision, metric.interpolated_recall, 'k-')
df_name = pd.DataFrame(df_name, columns=['image_name', 'exp', 'ap', 'n']).sort_values('ap')

# visualise detection for one image
image_name = 'ARCH2021-05-27_7794_3942_60.png'
dfin = df[(df['iteration'] == best_iteration) & (df['image_name'] == image_name)]
img = cv2.cvtColor(cv2.imread('data/grapevine/dataset/image_valid/' + image_name), cv2.COLOR_BGR2RGB)
plt.imshow(img)
for _, row in dfin.iterrows():
    if row['score'] > score_threshold:
        col = 'greenyellow' if row['type'] == 'obs' else 'r'
        linestyle = '--' if row['type'] == 'obs' else '-'
        linewidth = 1. if row['type'] == 'obs' else 1.6
        x1, x2, y1, y2 = row[['x1', 'x2', 'y1', 'y2']]
        plt.plot([x1, x1, x2, x2, x1], [y1, y2, y2, y1, y1], linestyle=linestyle, linewidth=linewidth, color=col)

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

# ===== obs vs pred (segmentation)  ==================================================================================
# (use the list of couples generated in the previous code section)

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


# visualise segmentation for one image
image_name = 'ARCH2021-05-27_7783_3757_30.png'
s_seg = val_seg[val_seg['image_name'] == 'ARCH2021-05-27_7783_3757_30.png']
img = cv2.cvtColor(cv2.imread('data/grapevine/dataset/image_valid/' + image_name), cv2.COLOR_BGR2RGB)
plt.imshow(img)
for _, row in s_seg.iterrows():
    col = 'greenyellow' if row['type'] == 'obs' else 'r'
    linestyle = '--' if row['type'] == 'obs' else '-'
    linewidth = 1. if row['type'] == 'obs' else 1.6
    if row['type'] == 'pred':
        x1, x2, y1, y2 = row[['x1', 'x2', 'y1', 'y2']]
        plt.plot([x1, x1, x2, x2, x1], [y1, y2, y2, y1, y1], linestyle=linestyle, linewidth=linewidth, color=col)
    x, y, w, h, a = row[['ell_x', 'ell_y', 'ell_w', 'ell_h', 'ell_a']]
    lsp_x, lsp_y = ellipse_interpolation(x=x, y=y, w=w, h=h, a=a, n_points=100)
    plt.plot(lsp_x, lsp_y, linestyle=linestyle, linewidth=linewidth, color=col)



# ============================================================

plt.xlim((0, 1))
plt.plot(metric.interpolated_recall, metric.interpolated_precision)
scores = np.array(sorted(dfi[dfi['type'] == 'pred']['score'])[::-1])
for s in [0.99, 0.97, 0.95, 0.93, 0.90, 0.5]:
    k = np.argmin(np.abs(scores - s))
    plt.plot(metric.interpolated_recall[k], metric.interpolated_precision[k], 'o', label=s)
plt.legend()


"""
https://jonathan-hui.medium.com/map-mean-average-precision-for-object-detection-45c121a31173
"""

df_res = []
for iteration in [i for i in df['iteration'].unique()]:
    for image_name in df['image_name'].unique():
        selec = df[(df['iteration'] == iteration) & (df['image_name'] == image_name)]

        s = selec[selec['score'] > score_threshold]

        # TODO : sometimes, fp < 0
        tp = 0
        obs, pred = s[s['type'] == 'obs'], s[s['type'] == 'pred']
        for _, row in obs.iterrows():
            box1 = {'x1': row['x1'], 'x2': row['x2'], 'y1': row['y1'], 'y2': row['y2']}
            # x1, y1, x2, y2 = row[['x1', 'y1', 'x2', 'y2']]
            # pol_obs = [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]
            all_iou = []
            for _, row in pred.iterrows():
                box2 = {'x1': row['x1'], 'x2': row['x2'], 'y1': row['y1'], 'y2': row['y2']}
                iou = get_iou(box1, box2) # for boxes, it's faster than the polygon method (and less imports)
                # x1, y1, x2, y2 = row[['x1', 'y1', 'x2', 'y2']]
                # pol_pred = [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]
                # iou = IOU(pol_obs, pol_pred)
                all_iou.append(iou)
            if len(all_iou) != 0:
                if max(all_iou) > IOU_THRESHOLD:
                    tp += 1
        fn = len(obs) - tp
        fp = len(pred) - tp

        print(iteration, image_name, tp, fp, fn)

        df_res.append([image_name, iteration, tp, fn, fp])

        # plt.imshow(img)
        # plt.plot(obs['x'], obs['y'], 'g*')
        # plt.plot(pred['x'], pred['y'], 'r*')

df_res = pd.DataFrame(df_res, columns=['image_name', 'iteration', 'tp', 'fn', 'fp'])


df_res2 = df_res.groupby('iteration').sum().reset_index()
cols_sum = np.array([sum((row['tp'], row['fp'], row['fn'])) for _, row in df_res2.iterrows()])
for col in ['tp', 'fp', 'fn']:
    df_res2[col] /= cols_sum

plt.plot(df_res2['iteration'], df_res2['tp'], 'k', label='TP')
plt.plot(df_res2['iteration'], df_res2['fn'], 'g', label='FN')
plt.plot(df_res2['iteration'], df_res2['fp'], 'r', label='FP')
plt.legend()

precision = df_res2['tp'] / (df_res2['tp'] + df_res2['fp'])
plt.plot(df_res2['iteration'], precision, 'k', label='Precision')
recall = df_res2['tp'] / (df_res2['tp'] + df_res2['fn'])
plt.plot(df_res2['iteration'], recall, 'r', label='Recall')
plt.legend()

# for one iteration:
dfi = df_res[df_res['iteration'] == 24000]
dfi['exp'] = [n.split('_')[0] for n in dfi['image_name']]


# ===== results.txt =============================================================================================

with open(PATH + 'validation/results7.txt') as f:
    lines = f.readlines()

lines = [l for l in lines if 'avg loss' in l]
iterations = [int(l.split(' ')[1][:-1]) for l in lines]
acc1 = [float(l.split(' ')[2][:-1]) for l in lines]
acc2 = [float(l.split(' ')[3][:-1]) for l in lines]

plt.plot(iterations, acc1, 'g.-')
plt.plot([iterations[0], iterations[-1]], [0, 0], 'r-')









