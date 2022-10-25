import os
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
from shapely.geometry import Polygon
from sklearn.metrics import r2_score

from deepberry.src.openalea.deepberry.prediction import detect_berry, segment_berry_scaled, load_models_berry
from deepberry.src.openalea.deepberry.utils import ellipse_interpolation

MODEL_DET, MODEL_SEG = load_models_berry('Y:/lepseBinaries/Trained_model/deepberry/')

PATH = 'data/grapevine/'

df_obs = pd.read_csv(PATH + 'dataset/grapevine_annotation.csv')

backup_folder = 'validation/backup7/'
config_path = 'data/grapevine/config_1class.cfg'

validation_image_subset = ['ARCH2021-05-27_323_3706_180.png',
                           'DYN2020-05-15_7243_2465_330.png',
                           'ARCH2022-05-18_684_5601_240.png',
                           'DYN2020-05-15_7235_2422_330.png',
                           'ARCH2022-05-18_1705_5601_330.png']


# TODO use it in DL codes if it's faster ? (nms)
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

# ===== display metrics for the selected model =====================================================================

res = {}
for image_name in validation_image_subset:
    img = cv2.cvtColor(cv2.imread(PATH + 'dataset/images/' + image_name), cv2.COLOR_BGR2RGB)

    obs = df_obs[df_obs['image_name'] == image_name]

    res_det = detect_berry(image=img, model=MODEL_DET)
    pred, _ = segment_berry_scaled(image=img, model=MODEL_SEG, boxes=res_det)

    obs_ell = [ellipse_interpolation(row['ell_x'], row['ell_y'], row['ell_w'], row['ell_h'], row['ell_a'], n_points=100)
               for _, row in obs.iterrows()]
    pred_ell = [ellipse_interpolation(row['ell_x'], row['ell_y'], row['ell_w'], row['ell_h'], row['ell_a'], n_points=100)
               for _, row in pred.iterrows()]

    plt.figure(image_name)
    plt.gca().set_aspect('equal', adjustable='box')
    for (ell_x, ell_y) in obs_ell:
        plt.plot(ell_x, ell_y, 'k-')
    for (ell_x, ell_y) in pred_ell:
        plt.plot(ell_x, ell_y, 'r-')

    for _, box in res_det.iterrows():
        x, y, w, h = box[['x', 'y', 'w', 'h']]
        plt.plot([x, x, x + h, x + h, x],
                 [y + h, y, y, y + h, y + h], 'r-')

    res[image_name] = {'obs': obs, 'pred': pred}


area_obs, area_pred = [], []
for image_name in res.keys():

    obs, pred = res[image_name]['obs'], res[image_name]['pred']

    obs_ell = [ellipse_interpolation(row['ell_x'], row['ell_y'], row['ell_w'], row['ell_h'], row['ell_a'], n_points=100)
               for _, row in obs.iterrows()]
    pred_ell = [ellipse_interpolation(row['ell_x'], row['ell_y'], row['ell_w'], row['ell_h'], row['ell_a'], n_points=100)
               for _, row in pred.iterrows()]

    plt.figure(image_name)
    plt.gca().set_aspect('equal', adjustable='box')
    for (ell_x, ell_y) in obs_ell:
        plt.plot(ell_x, ell_y, 'k-')
    for (ell_x, ell_y) in pred_ell:
        plt.plot(ell_x, ell_y, 'r-')

    D = np.zeros((len(obs_ell), len(pred_ell)))
    for i1, e1 in enumerate(obs_ell):
        for i2, e2 in enumerate(pred_ell):
            D[i1, i2] = IOU(e1.T, e2.T)
    D2 = D.copy()
    matches, dists = [], []
    for k in range(min(D2.shape)):
        i, j = np.unravel_index(D2.argmax(), D2.shape)
        d = D2[i, j]
        if d > 0.5:
            matches.append([i, j])
            dists.append(d)
        D2[i, :] = float('-inf')
        D2[:, j] = float('-inf')

    # detection accuracy
    tp, fn, fp = len(matches), len(obs) - len(matches), len(pred) - len(matches)
    print(tp, fn, fp)

    # segmentation accuracy
    print(np.mean(dists))

    s_obs = obs.iloc[np.array(matches)[:, 0]]
    area_obs += list((s_obs['ell_w'] / 2) * (s_obs['ell_h'] / 2) * np.pi)
    s_pred = pred.iloc[np.array(matches)[:, 1]]
    area_pred += list((s_pred['ell_w'] / 2) * (s_pred['ell_h'] / 2) * np.pi)

x, y = np.array(area_obs), np.array(area_pred)
plt.plot([0, 10000], [0, 10000], '-', color='grey')
a, b = np.polyfit(x, y, 1)
plt.plot([0, 10000], a*np.array([0, 10000]) + b, '--', color='red', label=f'y = {a:.{2}f}x {b:+.{2}f}')
plt.plot(x, y, 'o', alpha=0.4)
rmse = np.sqrt(np.sum((x - y) ** 2) / len(x))
r2 = r2_score(x, y)
bias = np.mean(y - x)
mape = 100 * np.mean(np.abs((x - y) / x))
print(rmse, r2, bias, mape)



# ==================================================================================================================

iterations = [f.split('class_')[1].split('.')[0] for f in os.listdir(PATH + backup_folder)]
iterations = sorted([int(iter) for iter in iterations if iter.isdigit()])
iterations = [i for i in iterations if i >= 3000 and i <= 140000]

df_res_brut = []
df_res = []
#for k_image, image_name in enumerate(np.random.choice(df[df['plantid'].isin(ZB14_valid)]['image_name'].unique(), 10, replace=False)):
for k_image, image_name in enumerate(validation_image_subset):

    anot_selec = df_obs[df_obs['image_name'] == image_name]
    img = cv2.imread(PATH + 'dataset/images/' + image_name)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    obs = df_obs[df_obs['image_name'] == image_name]

    for iteration in iterations:
        weight_file = 'config_1class_{}.weights'.format(iteration)

        # yolov4 trained model
        weights_path = PATH + backup_folder + weight_file
        net = cv2.dnn.readNetFromDarknet(config_path, weights_path)
        model = cv2.dnn_DetectionModel(net)

        t0 = time.time()
        pred = detect_berry(image=img, model=model)
        t = round(time.time() - t0, 1)
        print(k_image, weight_file, t)

        # plt.figure()
        # plt.imshow(img)

        for _, row in obs.iterrows():
            x, y, w, h = row[['box_x', 'box_y', 'box_w', 'box_h']]
            x, y = x - w / 2, y - h / 2
            df_res_brut.append([iteration, 'obs', x, y, x + w, y + h, 1, image_name])
            # plt.plot([x, x, x + w, x + w, x], [y, y + h, y + h, y, y], 'g-')

        for _, row in pred.iterrows():
            x, y, w, h, score = row
            df_res_brut.append([iteration, 'pred', x, y, x + w, y + h, score, image_name])
            # if score > 0.9:
            #     plt.plot([x, x, x + w, x + w, x], [y, y + h, y + h, y, y], 'r-')

df_res_brut = pd.DataFrame(df_res_brut, columns=['iteration', 'type', 'x1', 'y1', 'x2', 'y2', 'score', 'image_name'])

df_res_brut.to_csv(PATH + 'validation/validation_{}.csv'.format(backup_folder.split('/')[1]), index=False)

# ===== display metrics =============================================================================================

# TODO : impact du score threshold
# TODO : effet du paramètre nms ?
# TODO : quel nms dans mes annotations manuelles ???


score_threshold = 0.9
iou_threshold = 0.6  # match between box1 and box2 if iou(box1, box2) > iou_threshold

res = {}

for training in ['validation/validation_backup7']:

    df = pd.read_csv(PATH + '{}.csv'.format(training))

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
                all_iou = []
                for _, row in pred.iterrows():
                    box2 = {'x1': row['x1'], 'x2': row['x2'], 'y1': row['y1'], 'y2': row['y2']}
                    all_iou.append(get_iou(box1, box2))
                if len(all_iou) != 0:
                    if max(all_iou) > iou_threshold:
                        tp += 1
            fn = len(obs) - tp
            fp = len(pred) - tp

            print(iteration, image_name, training, tp, fp, fn)

            df_res.append([image_name, iteration, tp, fn, fp])

            # plt.imshow(img)
            # plt.plot(obs['x'], obs['y'], 'g*')
            # plt.plot(pred['x'], pred['y'], 'r*')

    df_res = pd.DataFrame(df_res, columns=['image_name', 'iteration', 'tp', 'fn', 'fp'])
    res[training] = df_res

for (_, df_res), symbol in zip(res.items(), ['*-', 'o--', '.:', '^-'][:(len(res.items()))]):
    df_res = df_res[~(df_res['image_name'] == 'ARCH2022-05-18_684_5601_240.png')]
    df_res2 = df_res.groupby('iteration').sum().reset_index()
    cols_sum = np.array([sum((row['tp'], row['fp'], row['fn'])) for _, row in df_res2.iterrows()])
    for col in ['tp', 'fp', 'fn']:
        df_res2[col] /= cols_sum

    plt.plot(df_res2['iteration'], df_res2['tp'], 'k' + symbol, label='TP')
    plt.plot(df_res2['iteration'], df_res2['fn'], 'g' + symbol, label='FN')
    plt.plot(df_res2['iteration'], df_res2['fp'], 'r' + symbol, label='FP')
    plt.legend()


# ===== results.txt =============================================================================================

with open(PATH + 'validation/results7.txt') as f:
    lines = f.readlines()

lines = [l for l in lines if 'avg loss' in l]
iterations = [int(l.split(' ')[1][:-1]) for l in lines]
acc1 = [float(l.split(' ')[2][:-1]) for l in lines]
acc2 = [float(l.split(' ')[3][:-1]) for l in lines]

plt.plot(iterations, acc1, 'g.-')
plt.plot([iterations[0], iterations[-1]], [0, 0], 'r-')









