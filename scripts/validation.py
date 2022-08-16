import os
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time

from grapevine.prediction import detect_berry
from grapevine.utils import get_iou

PATH = 'data/grapevine/'

df_obs = pd.read_csv(PATH + 'dataset/grapevine_annotation.csv')

backup_folder = 'validation/backup7/'
config_path = 'data/grapevine/config_1class.cfg'

validation_image_subset = ['ARCH2021-05-27_323_3706_180.png',
                           'DYN2020-05-15_7243_2465_330.png',
                           'ARCH2022-05-18_684_5601_240.png',
                           'DYN2020-05-15_7235_2422_330.png',
                           'ARCH2022-05-18_1705_5601_330.png']

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









