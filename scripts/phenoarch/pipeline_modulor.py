""" Run the full berry segmentation/tracking pipeline on a server (modulor) and save results in a cache """

import os
import cv2
import time
import pandas as pd
import numpy as np
from multiprocessing import Pool

from deepberry.src.openalea.deepberry.ellipse_segmentation import \
    berry_detection, berry_segmentation, load_berry_models
from deepberry.src.openalea.deepberry.features_extraction import berry_features_extraction
from deepberry.src.openalea.deepberry.temporal import distance_matrix, points_sets_alignment

DIR_INDEX = '/home/daviet/deepberry_data/'
DIR_CACHE = '/mnt/data/phenoarch_cache/'
DIR_MODEL = '/mnt/phenomixNasShare/lepseBinaries/Trained_model/deepberry/'
DIR_IMAGE = '/mnt/phenomixNas/'

MODEL_DET, MODEL_SEG = load_berry_models(DIR_MODEL)

SCORE_THRESHOLD_DET = 0.89  # 0.985

index = pd.read_csv(DIR_INDEX + 'image_index.csv')
index = index[~index['imgangle'].isna()]

# selection of the 3 "high-frequency" genotypes from 2020 / 2021 experiments
selec = index[(index['exp'] == 'DYN2020-05-15') |
              ((index['exp'] == 'ARCH2021-05-27') &
               (index['genotype'].isin(['V.vinifera/V6863_H10-PL3', 'V.vinifera/V6860_DO4-PL4'])))]

# filter problematic tasks
quarantine_tasks = {'DYN2020-05-15': [2339, 2340, 2354, 2362, 2364, 2365, 2366, 2375, 2377, 2378, 2379],
                    'ARCH2021-05-27': [3797, 3798, 3804, 3810, 3811, 3819, 3827, 3829, 3831, 3843],
                    'ARCH2022-05-18': [5742, 5744, 5876, 5877]}
for exp, tasks in quarantine_tasks.items():
    selec = selec[~((selec['exp'] == exp) & (selec['taskid'].isin(tasks)))]

# filter problematic snapshots for temporal (2020 & 2021)
quarantine_temporal = {'DYN2020-05-15': {7232: [2380, 2382, 2384],
                                         7235: [2613, 2619, 2607, 2614, 2620, 2608, 2615, 2621, 2609, 2617, 2623,
                                                2611, 2634, 2636, 2639, 2642, 2637, 2640, 2643, 2638, 2641, 2656],
                                         7238: [2380, 2382, 2384, 2385, 2390, 2394, 2386, 2391, 2395]},
                       'ARCH2021-05-27': {7763: [3835, 3944, 3842, 3836, 3937, 3942, 3932, 3938, 3927, 3968, 3928,
                                                 3934, 3940, 3929, 3979, 3986, 3993, 4008]},
                       'ARCH2022-05-18': {}}

# =====================================================================================================================

selec = index[index['exp'].str.contains('ARCH2022')]
plantids = np.random.choice(np.unique(selec['plantid']), 60, replace=False)
selec = selec[selec['plantid'].isin(plantids)]

# ===== non-temporal: berry detection and segmentation ================================================================

chrono = []

for _, row in selec.iterrows():

    plantid_path = DIR_CACHE + 'cache_{}_NEW/segmentation/{}/'.format(row['exp'], row['plantid'])
    if not os.path.isdir(plantid_path):
        os.makedirs(plantid_path)

    filename = plantid_path + '{}_{}.csv'.format(int(row['taskid']), int(row['imgangle']))
    img_path = DIR_IMAGE + '{}/{}/{}.png'.format(row['exp'], int(row['taskid']), row['imgguid'])

    if not os.path.isfile(filename):

        img_dwn = False
        try:
            img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
            img_dwn = True
        except:
            print('cannot load image', img_path)

        if img_dwn:

            t0 = time.time()
            res_det = berry_detection(image=img, model=MODEL_DET, score_threshold=SCORE_THRESHOLD_DET)
            t1 = time.time()
            res_seg = berry_segmentation(image=img, model=MODEL_SEG, boxes=res_det)
            t2 = time.time()
            res = berry_features_extraction(image=img, ellipses=res_seg)
            t3 = time.time()

            print('{} | det: {:.1f}s, seg: {:.1f}s, features: {:.1f}s (n={})'.format(
                row['plantid'], t1 - t0, t2 - t1, t3 - t2, len(res)))

            chrono.append([t1 - t0, t2 - t1, t3 - t2, len(res)])

            res.to_csv(filename, index=False)

# ===== temporal: berry tracking ======================================================================================


def run_one_plant_temporal(meta):

    exp, plantid, angle, time_period = meta
    print(exp, plantid, angle, time_period)

    time_period_str = '' if time_period is None else '_{}'.format(time_period)

    plantid_t_path = DIR_CACHE + 'cache_{}_NEW/temporal{}/{}/'.format(exp, time_period_str, plantid)
    if not os.path.isdir(plantid_t_path):
        os.makedirs(plantid_t_path)

    plantid_t_path_mat = DIR_CACHE + 'cache_{}_NEW/distance_matrix{}/{}/'.format(exp, time_period_str, plantid)
    if not os.path.isdir(plantid_t_path_mat):
        os.makedirs(plantid_t_path_mat)

    s = selec[(selec['exp'] == exp) & (selec['plantid'] == plantid) & (selec['imgangle'] == angle)]

    tasks = list(s.groupby('taskid')['timestamp'].mean().sort_values().reset_index()['taskid'])

    # quarantine_plantid (temporal)
    if plantid in list(quarantine_temporal[exp].keys()):
        tasks = [t for t in tasks if t not in quarantine_temporal[exp][plantid]]

    if time_period is not None:
        tasks = tasks[::time_period]

    seg_folder = DIR_CACHE + 'cache_{}_NEW/segmentation/{}/'.format(exp, plantid)

    ellipses_sets = []
    tasks_to_remove = []
    for task in tasks:
        seg_path = seg_folder + '{}_{}.csv'.format(task, angle)
        if os.path.isfile(seg_path):
            ellipses = pd.read_csv(seg_path)
            if len(ellipses) > 1:
                ellipses_sets.append(ellipses)
            else:
                # print(plantid, task, angle, 'len = {}'.format(len(ellipses)))
                tasks_to_remove.append(task)
        else:
            # print(plantid, task, angle, 'no seg file')
            tasks_to_remove.append(task)

    tasks = [t for t in tasks if t not in tasks_to_remove]

    points_sets = [np.array(s[['ell_x', 'ell_y']]) for s in ellipses_sets]

    # camera shift TODO more clean
    if exp == 'DYN2020-05-15':
        t_camera = 1592571441  # between tasks 2566 and 2559
        for i_task, task in enumerate(tasks):
            timestamp = s[s['taskid'] == task]['timestamp'].mean()
            if timestamp > t_camera:
                points_sets[i_task] += np.array([-4.2, -437.8])

    # matrix_path = DIR_CACHE + 'cache_{}/distance_matrix/{}_{}_{}.npy'.format(exp, plantid, angle, time_period)
    matrix_path = plantid_t_path_mat + '{}.npy'.format(angle)
    if not os.path.isfile(matrix_path):
        print('computing distance matrix...')
        M = distance_matrix(points_sets)
        np.save(matrix_path, M)
    M = np.load(matrix_path)
    if len(points_sets) != len(M):
        print('wrong matrix size !', plantid)

    berry_ids = points_sets_alignment(points_sets=points_sets, dist_mat=M)
    for k in range(len(ellipses_sets)):
        ellipses_sets[k].loc[:, 'berryid'] = berry_ids[k]
        ellipses_sets[k].loc[:, 'task'] = tasks[k]
    final_res = pd.concat(ellipses_sets)

    # final_res.to_csv(plantid_t_path + '{}_{}.csv'.format(angle, time_period), index=False)
    final_res.to_csv(plantid_t_path + '{}.csv'.format(angle), index=False)


for period in [2, 3, 6, 9, 12, 15][::-1]:

    gb = selec.groupby(['exp', 'plantid', 'imgangle']).size().reset_index()
    metas = [[row['exp'], int(row['plantid']), int(row['imgangle'])] for _, row in gb.iterrows()]

    metas = [m + [period] for m in metas]

    for meta in metas:
        run_one_plant_temporal(meta)

def mp_full_exp(metas, nb_cpu=12):
    with Pool(nb_cpu) as p:
        p.map(run_one_plant_temporal, metas)
mp_full_exp(metas)

# ===== combine results in a single .csv file with metadata ===========================================================

# TODO add t column ?

# exp = 'DYN2020-05-15'
# exp = 'ARCH2021-05-27'
exp = 'ARCH2022-05-18'

# for step in ['segmentation', 'temporal']:
step = 'temporal'
# for time_period in [2, 3, 6, 9, 12, 15]:
for time_period in [None]:

    time_period_str = '' if time_period is None else '_{}'.format(time_period)

    df = []
    exp_path = DIR_CACHE + 'cache_{}/{}{}'.format(exp, step, time_period_str)
    s = index[index['exp'] == exp]
    if os.path.isdir(exp_path):
        for plantid in [int(p) for p in os.listdir(exp_path)]:
            s2 = s[s['plantid'] == plantid]
            genotype, scenario = s2[['genotype', 'scenario']].iloc[0]
            plantid_path = '{}/{}/'.format(exp_path, plantid)

            files = os.listdir(plantid_path)
            # for f in [f for f in files if int(f[:-4].split('_')[1]) == period]:
            for f in files:

                print(plantid_path + f)
                dfi = pd.read_csv(plantid_path + f)

                if step == 'segmentation':
                    task, angle = [int(k) for k in f[:-4].split('_')]
                    timestamp = s2[(s2['taskid'] == task) & (s2['imgangle'] == angle)]['timestamp'].iloc[0]
                    dfi[['exp', 'plantid', 'task', 'timestamp', 'angle', 'genotype', 'scenario']] = \
                        exp, int(plantid), int(task), timestamp, angle, genotype, scenario
                elif step == 'temporal':
                    angle = int(f[:-4])
                    dfi[['exp', 'plantid', 'angle', 'genotype', 'scenario']] = \
                        exp, int(plantid), angle, genotype, scenario
                    task_to_timestamp = {task: s2[(s2['taskid'] == task) & (s2['imgangle'] == angle)]['timestamp'].iloc[0]
                                         for task in dfi['task'].unique()}
                    dfi['timestamp'] = [task_to_timestamp[task] for task in dfi['task']]

                df.append(dfi)

    df = pd.concat(df)

        # df.to_csv(PATH_CACHE + 'cache_{0}/full_results_period{1}_{0}.csv'.format(exp, period), index=False)
    df.to_csv(DIR_CACHE + 'cache_{0}/full_results_{1}_{0}{2}.csv'.format(exp, step, time_period_str), index=False)

# =====================================================================================================================


# for exp in ['DYN2020-05-15', 'ARCH2021-05-27', 'ARCH2022-05-18']:

exp = 'DYN2020-05-15'

cache_path = PATH_CACHE + 'cache_{}/'.format(exp)
if not os.path.isdir(cache_path):
    os.mkdir(cache_path)

exp_df = df[df['exp'] == exp]
if exp == 'ARCH2021-05-27':
    exp_df = exp_df[exp_df['genotype'].isin(['V.vinifera/V6863_H10-PL3', 'V.vinifera/V6860_DO4-PL4'])]

plantids = [int(p) for p in exp_df['plantid'].unique()]

plantid_angle_list = [(p, k * 30, t) for t in [2, 3, 6, 9, 12, 15] for k in range(12) for p in plantids]

mp_full_exp(plantid_angle_list, 12)




