import os
import pandas as pd
import cv2
import numpy as np
import time

from deepberry.src.openalea.deepberry.segmentation import detect_berry, segment_berry_scaled, load_models_berry

MODEL_DET, MODEL_SEG = load_models_berry('Y:/lepseBinaries/Trained_model/deepberry/')

PATH = 'data/grapevine/results/'

index = pd.read_csv('data/grapevine/image_index.csv')

df = index[~index['imgangle'].isna()]

# ===============================================================================================================

# # genotype with most plants
# genotypes = list(df.groupby('genotype').size().reset_index().sort_values(0)['g2'])[::-1]
# #genotypes = ['BARESA', 'BASERRI', 'BELLONE', 'PRIMITIV', 'A02-PL6']
# selec = df2[df2['g2'].isin(genotypes)]

exp = 'ARCH2021-05-27'
if not os.path.isdir(PATH + exp):
    os.mkdir(PATH + exp)

# TODO typo exp vs manip
selec = df[df['exp'] == exp]

for plantid in [int(p) for p in selec['plantid'].unique()]:
    s0 = selec[selec['plantid'] == plantid]

    for grapeid in s0['grapeid'].unique():
        s = s0[s0['grapeid'] == grapeid]

        s = s.sort_values('timestamp')

        if len(s0['grapeid'].unique()) == 1:
            plantid_path = PATH + '{}/{}/'.format(exp, plantid)
        else:
            plantid_path = PATH + '{}/{}_{}/'.format(exp, plantid, int(grapeid))
        if not os.path.isdir(plantid_path):
            os.mkdir(plantid_path)

        for _, row in s.iterrows():

            n_files = sum([len(os.listdir(PATH + exp + '/' + id)) for id in os.listdir(PATH + exp)])
            print(exp, n_files)

            savefile = plantid_path + '{}_{}.csv'.format(int(row['taskid']), int(row['imgangle']))

            img_path = 'V:/{}/{}/{}.png'.format(row['exp'], int(row['taskid']), row['imgguid'])

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

                    print('det: {:.1f}s, seg: {:.1f}s, classif: {:.1f}s (n={}, black={}%)'.format(
                        t1 - t0, t2 - t1, t3 - t2, len(res_classif), round(100*np.sum(res_classif['black'])/len(res_classif), 1)))

                    res_classif.to_csv(savefile, index=False)












