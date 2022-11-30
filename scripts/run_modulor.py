"""
MULTI-PROCESSING

a) DETECTION (cv2)
- run_one_plant() : 90% cpu, 2.1s
- run_one_plant(), cv2.setNumThreads(0) : 1 cpu, 10.6s
- mp_full_exp() : MARCHE PAS
- mp_full_exp(), cv2.setNumThreads(0) : tous les cpu, 11.9s / n_cpu

b) SEGMENTATION (keras)
- run_one_plant() : plusieurs cpu, 3s
- mp_full_exp() : MARCHE PAS
Apparemment il y a pas de moyen facile de retirer le multi-processing pour tensorflow/keras ...
"""

import os
import pandas as pd
import cv2
import numpy as np
import time

from multiprocessing import Pool

from deepberry.src.openalea.deepberry.detection_and_segmentation import berry_detection, berry_segmentation, load_berry_models
from deepberry.src.openalea.deepberry.features_extraction import berry_features_extraction
from deepberry.src.openalea.deepberry.temporal import distance_matrix, points_sets_alignment

# disable multi CPU in opencv2. necessary to run deep-learning (opencv2) and multi-processing at the same time
#cv2.setNumThreads(0)

DIR_INDEX = '/home/daviet/deepberry_data/'
DIR_CACHE = '/mnt/data/phenoarch_cache/'
DIR_MODEL = '/mnt/phenomixNasShare/lepseBinaries/Trained_model/deepberry/new/'

MODEL_DET, MODEL_SEG = load_berry_models(DIR_MODEL)

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

# ===== non-temporal ==============================================================================================

chrono = []

for _, row in selec.iterrows():

    plantid_path = DIR_CACHE + 'cache_{}/segmentation/{}/'.format(row['exp'], row['plantid'])
    if not os.path.isdir(plantid_path):
        os.makedirs(plantid_path)

    filename = plantid_path + '{}_{}.csv'.format(int(row['taskid']), int(row['imgangle']))
    img_path = '/mnt/phenomixNas/{}/{}/{}.png'.format(row['exp'], int(row['taskid']), row['imgguid'])

    if not os.path.isfile(filename):

        img_dwn = False
        try:
            img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
            img_dwn = True
        except:
            print('cannot load image', img_path)

        if img_dwn:

            t0 = time.time()
            res_det = berry_detection(image=img, model=MODEL_DET, score_threshold=0.985)
            t1 = time.time()
            res_seg = berry_segmentation(image=img, model=MODEL_SEG, boxes=res_det)
            t2 = time.time()
            res = berry_features_extraction(image=img, ellipses=res_seg)
            t3 = time.time()

            print('{} | det: {:.1f}s, seg: {:.1f}s, features: {:.1f}s (n={})'.format(
                row['plantid'], t1 - t0, t2 - t1, t3 - t2, len(res)))

            chrono.append([t1 - t0, t2 - t1, t3 - t2, len(res)])

            res.to_csv(filename, index=False)


def run_one_plant(plantid_angle_period):

    plantid, angle, time_period = plantid_angle_period
    print('run_one_plant', plantid, angle, time_period)

    s = exp_df[exp_df['plantid'] == plantid].sort_values('timestamp')

    # ===== temporal ==================================================================================================

    exp = exp_df.iloc[0]['exp']

    plantid_t_path = DIR_CACHE + 'cache_{}/temporal/{}/'.format(exp, plantid)
    if not os.path.isdir(plantid_t_path):
        os.mkdir(plantid_t_path)

    tasks = list(s.groupby('taskid')['timestamp'].mean().sort_values().reset_index()['taskid'])

    # quarantine_plantid (temporal)
    quarantine_t = pd.read_csv(DIR_CACHE + 'cache_{0}/quarantine_temporal_{0}.csv'.format(exp))
    tasks = [t for t in tasks if t not in list(quarantine_t[quarantine_t['plantid'] == plantid]['task'])]

    tasks = tasks[::time_period]

    seg_folder = DIR_CACHE + 'cache_{}/segmentation/{}/'.format(exp, plantid)

    # for angle in [k * 30 for k in range(12)]:

    ellipses_sets = []
    tasks_to_remove = []
    for task in tasks:
        seg_path = seg_folder + '{}_{}.csv'.format(task, angle)
        if os.path.isfile(seg_path):
            ellipses = pd.read_csv(seg_path)
            if len(ellipses) > 1:
                ellipses_sets.append(ellipses)
            else:
                print(plantid, task, angle, 'len = {}'.format(len(ellipses)))
                tasks_to_remove.append(task)
        else:
            print(plantid, task, angle, 'no seg file')
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

    matrix_path = DIR_CACHE + 'cache_{}/distance_matrix/{}_{}_{}.npy'.format(exp, plantid, angle, time_period)
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

    final_res.to_csv(plantid_t_path + '{}_{}.csv'.format(angle, time_period), index=False)



def mp_full_exp(plantids, nb_cpu=11):
    with Pool(nb_cpu) as p:
        p.map(run_one_plant, plantids)

# ===============================================================================================================


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


# ===== full_results.csv ==============================================================================================

exp = 'DYN2020-05-15'
# exp = 'ARCH2021-05-27'
exp = 'ARCH2022-05-18'

# for period in [2, 3, 6, 9, 12, 15]:

    df = []
    exp_path = '/mnt/data/phenoarch_cache/cache_{}/segmentation'.format(exp) # TODO temporal!
    s = index[index['exp'] == exp]
    if os.path.isdir(exp_path):
        for plantid in [int(p) for p in os.listdir(exp_path)]:
            s2 = s[s['plantid'] == plantid]
            genotype, scenario = s2[['genotype', 'scenario']].iloc[0]
            plantid_path = '{}/{}/'.format(exp_path, plantid)

            files = [f for f in os.listdir(plantid_path) if '_' in f]
            # for f in [f for f in files if int(f[:-4].split('_')[1]) == period]:
            for f in files:
                # angle = int(f.split('_')[0])
                task, angle = [int(k) for k in f[:-4].split('_')]

                print(plantid_path + f)
                df_f = pd.read_csv(plantid_path + f)

                # for task in df_f['task'].unique():
                #     df_f_task = df_f[df_f['task'] == task]
                #     task = int(df_f_task.iloc[0]['task'])
                df_f_task = df_f

                timestamp = s2[(s2['taskid'] == task) & (s2['imgangle'] == angle)]['timestamp'].iloc[0]
                df_f_task[['exp', 'plantid', 'task', 'timestamp', 'angle', 'genotype', 'scenario']] = \
                    exp, int(plantid), int(task), timestamp, angle, genotype, scenario
                df.append(df_f_task)
    df = pd.concat(df)

    # df['area'] = (df['ell_w'] / 2) * (df['ell_h'] / 2) * np.pi
    # df['volume'] = (4 / 3) * np.pi * ((np.sqrt(df['area'] / np.pi)) ** 3)
    # df['roundness'] = df['ell_w'] / df['ell_h']  # always w <= h

    tmin_dic = {row['exp']: row['timestamp'] for _, row in df.groupby('exp')['timestamp'].min().reset_index().iterrows()}
    df['t'] = df.apply(lambda row: (row['timestamp'] - tmin_dic[row['exp']]) / 3600 / 24, axis=1)

    # df.to_csv(PATH_CACHE + 'cache_{0}/full_results_period{1}_{0}.csv'.format(exp, period), index=False)
df.to_csv(PATH_CACHE + 'cache_{0}/full_results_{0}.csv'.format(exp), index=False)






