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

from deepberry.src.openalea.deepberry.prediction import detect_berry, segment_berry_scaled, classify_berry, load_models_berry

# disable multi CPU in opencv2. necessary to run deep-learning (opencv2) and multi-processing at the same time
#cv2.setNumThreads(0)

PATH_DATA = '/home/daviet/deepberry_data/'
PATH_CACHE = '/mnt/data/phenoarch_cache/'

MODEL_DET, MODEL_SEG = load_models_berry(PATH_DATA)

index = pd.read_csv(PATH_DATA + 'image_index.csv')

df = index[~index['imgangle'].isna()]

exp = 'ARCH2022-05-18'
cache_path = PATH_CACHE + 'cache_{}/'.format(exp)
if not os.path.isdir(cache_path):
    os.mkdir(cache_path)

exp_df = df[df['exp'] == exp]

genotypes = list(exp_df.groupby(['genotype'])['plantid'].nunique().sort_values()[::-1].reset_index()['genotype'])


def run_one_plant(plantid):

    print('run_one_plant', plantid)

    s = exp_df[exp_df['plantid'] == plantid]
    s = s.sort_values('timestamp')

    plantid_path = cache_path + str(plantid) + '/'
    if not os.path.isdir(plantid_path):
        os.mkdir(plantid_path)

    for _, row in s.iterrows():

        savefile = plantid_path + '{}_{}.csv'.format(int(row['taskid']), int(row['imgangle']))
        img_path = '/mnt/phenomixNas/{}/{}/{}.png'.format(row['exp'], int(row['taskid']), row['imgguid'])

        if not os.path.isfile(savefile):

            img_dwn = False
            try:
                img = cv2.imread(img_path)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img_dwn = True
            except:
                print('bug', img_path)

            if img_dwn:

                t0 = time.time()
                res_det = detect_berry(image=img, model=MODEL_DET)
                t1 = time.time()
                res_seg, _ = segment_berry_scaled(image=img, model=MODEL_SEG, boxes=res_det)
                t2 = time.time()
                res_classif = classify_berry(image=img, ellipses=res_seg)
                t3 = time.time()

                print('{} | det: {:.1f}s, seg: {:.1f}s, classif: {:.1f}s (n={}, black={}%)'.format(plantid,
                    t1 - t0, t2 - t1, t3 - t2, len(res_classif), round(100*np.sum(res_classif['black'])/len(res_classif), 1)))

                res_classif.to_csv(savefile, index=False)


def mp_full_exp(plantids, nb_cpu=11):
    with Pool(nb_cpu) as p:
        p.map(run_one_plant, plantids)

# ===============================================================================================================


for genotype in genotypes:
    print('genotype', genotype)
    s = exp_df[exp_df['genotype'] == genotype]
    for plantid in [int(plantid) for plantid in s['plantid'].unique()]:
        print('plantid', plantid)
        run_one_plant(plantid)

#mp_full_exp(plantids, nb_cpu=11)






