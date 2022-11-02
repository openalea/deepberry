"""
viewtypeid: 1 = top, 2 = side, 3 = xyz

2020: 16k images xyz. 1 genotype x 9 plantes x 150 tasks (50 days)
2021: 40k images xyz. 5 genotype (+ Odium; 6gxe en tout) x 2-8 plantes (tot=25) x 80 tasks (34 days)
2022 (21-07): 65k images xyz. 3000 plants (1700 xyz).

=> Idée : 2022 pour montrer la robustesse à une forte variabilité génétique, 2020-2021 pour le temporel

2022:
24 photos = 2 grappes. Toujours prises en photo dans le meme ordre
24 imgs, puis 12 ? parce-que ce sont rendu compte que trop galere de suivre 2-3 grappes, donc en gardent qu'une
mais laquelle ? je peux surement pas savoir, besoin tableau llorenc avec position des grappes
"""

import os
import numpy as np
import shutil
import matplotlib.pyplot as plt
import pandas as pd
import cv2

import time
import datetime

from deepberry.src.openalea.deepberry.utils import ellipse_interpolation


def date_to_timestamp(date):
    return int(time.mktime(datetime.datetime.strptime(date, '%Y-%m-%d %H:%M:%S').timetuple()))


# ===== PhenoArch 2020 experiment ===================================================================================

df = pd.read_csv('data/copy_from_database/images_grapevine20.csv')
df_plant = pd.read_csv('data/copy_from_database/plants_grapevine20.csv')

df = df[df['viewtypeid'] == 3]  # xyz camera
df['exp'] = 'DYN2020-05-15'

dict_plant = {row['plantid']: row['plantcode'] for _, row in df_plant.iterrows()}
df = df[~df['plantid'].isna()]
df['plantcode'] = [dict_plant[plantid] for plantid in df['plantid']]
df['genotype'] = 'Alexandroouli'
df['scenario'] = 'normal'

# angle correction (angle = 330 for >99% images)
for plantid in df['plantid'].unique():
    selec = df[df['plantid'] == plantid]
    for task in selec['taskid'].unique():
        s = selec[selec['taskid'] == task].sort_values('acquisitiondate')
        if len(s) != 12:
            df.at[s.index, 'imgangle'] = None
        else:
            df.at[s.index, 'imgangle'] = [k * 30 for k in range(12)]

df20 = df.copy()

# ===== PhenoArch 2021 experiment ===================================================================================

df = pd.read_csv('data/copy_from_database/images_grapevine21.csv')
df_plant = pd.read_csv('data/copy_from_database/plants_grapevine21.csv')

df = df[df['viewtypeid'] == 3]  # xyz camera
df['exp'] = 'ARCH2021-05-27'

dict_plant = {row['plantid']: row['plantcode'] for _, row in df_plant.iterrows()}
df = df[~df['plantid'].isna()]
df['plantcode'] = [dict_plant[plantid] for plantid in df['plantid']]
df = df[~df['plantcode'].isin([p for p in df['plantcode'].unique() if 'Vide' in p or 'ChessBoard' in p])]
df['genotype'] = ['/'.join(p.split('/')[1:3]) for p in df['plantcode']]
df['scenario'] = ['odium' if 'Odium' in p else 'normal' for p in df['plantcode']]

# angle correction (angle = 330 often missing because of repeated angle in snapshot)
# and several grapes for some plants (24 or 36 images / snapshot), but it's globally consistent over time.
for plantid in df['plantid'].unique():
    selec = df[df['plantid'] == plantid]
    gb = selec.groupby('taskid').size()
    n_imgs = max(set(gb.values), key=list(gb.values).count)  # most frequent number of images / task
    if n_imgs % 12 != 0:  # True for only 1 plant
        df.at[selec.index, 'imgangle'] = None
    else:
        for task in selec['taskid'].unique():
            s = selec[selec['taskid'] == task].sort_values('acquisitiondate')
            if len(s) != n_imgs:
                df.at[s.index, 'imgangle'] = None
            else:
                if n_imgs == 12:
                    df.at[s.index, 'imgangle'] = [int(k * 30) for k in range(12)]
                elif n_imgs == 24:
                    df.at[s.index, 'imgangle'] = [0, None, 30, None, 60, None, 90, None, None, 120, 150, None,
                                                  None, 180, 210, None, 240, None, 270, None, None, 300, None, 330]
                elif n_imgs == 36:
                    df.at[s.index, 'imgangle'] = [None, None, 0, None, 30, None, None, 60, None, None, None, 90, None,
                                                  120, None, None, None, 150, 180, None, None, 210, None, None, 240,
                                                  None, None, None, None, 270, None, None, 300, None, None, 330]

df21 = df.copy()

# ===== PhenoArch 2022 experiment ===================================================================================

df = pd.read_csv('data/copy_from_database/images_grapevine22.csv')
df_plant = pd.read_csv('data/copy_from_database/plants_grapevine22.csv')

df = df[df['viewtypeid'] == 3]  # xyz camera
df['exp'] = 'ARCH2022-05-18'

dict_plant = {row['plantid']: row['plantcode'] for _, row in df_plant.iterrows()}
df = df[~df['plantid'].isna()]
df['plantcode'] = [dict_plant[plantid] for plantid in df['plantid']]
df = df[~df['plantcode'].str.contains('PhenoDYYN')]
df = df[~df['plantcode'].str.contains('T0')]
df['plantcode'] = [p.replace('MIREILLE', 'V2713/MIREILLE') if len(p.split('/')) != 9 else p for p in df['plantcode']]

#df['genotype_code'] = [n.split('/')[1] for n in df['plantcode']]
df['genotype'] = [n.split('/')[2] for n in df['plantcode']]
df['scenario'] = [n.split('/')[-3] for n in df['plantcode']]

# TODO : 13% images are lost (i.e. unknown angle), could be saved using grape ids
for plantid in df['plantid'].unique():
    selec = df[df['plantid'] == plantid]
    #print([len(selec[selec['taskid'] == task]) for task in selec['taskid'].unique()])
    for task in selec['taskid'].unique():
        s = selec[selec['taskid'] == task].sort_values('acquisitiondate')
        if len(s) != 12:
            df.at[s.index, 'imgangle'] = None
        else:
            df.at[s.index, 'imgangle'] = [k * 30 for k in range(12)]

df22 = df.copy()

# ===== Combine 2020/2021/2022 PhenoArch experiments =================================================================

df = pd.concat((df20, df21, df22))

df['daydate'] = [d[5:10] for d in df['acquisitiondate']]
df['plantid'] = [int(p) for p in df['plantid']]
df['timestamp'] = df.apply(lambda row: date_to_timestamp(row['acquisitiondate']), axis=1)
del df['viewtypeid']  # index only contains viewtypeid = 3

df.to_csv('data/grapevine/image_index2.csv', index=False)

# ====================================================================================================================
# ====================================================================================================================

index1 = pd.read_csv('data/grapevine/image_index.csv')
index2 = pd.read_csv('data/grapevine/image_index2.csv')

# ===== remove 'black' col (modulor) ===============================================================================

n = 0
for exp in ['DYN2020-05-15', 'ARCH2021-05-27', 'ARCH2022-05-18']:
    fd = '/mnt/data/phenoarch_cache/cache_{}'.format(exp)
    files = [fd + '/' + id + '/' + f for id in os.listdir(fd) for f in os.listdir(fd + '/' + id)]
    for f in files:
        df = pd.read_csv(f)
        del df['black']
        df.to_csv(f, index=False)
        n += 1
        print(n, exp)

# ===== update cache ===============================================================================================

n = 0
# for exp in ['DYN2020-05-15', 'ARCH2021-05-27', 'ARCH2022-05-18']:
exp = 'ARCH2021-05-27'
fd = 'X:/phenoarch_cache/cache_{}_OLD'.format(exp)
for id in os.listdir(fd):
    plantid, grapeid = [int(k) for k in id.split('_')] if '_' in id else (int(id), 0)
    s1 = index1[(index1['exp'] == exp) & (index1['plantid'] == plantid) & (index1['grapeid'] == grapeid)]
    s2 = index2[(index2['exp'] == exp) & (index2['plantid'] == plantid)]
    files = os.listdir(fd + '/' + id)
    for f in files:
        task, angle = [int(k) for k in f[:-4].split('_')]

        row1 = s1[(s1['taskid'] == task) & (s1['imgangle'] == angle)].iloc[0]
        row2 = s2[s2['imgguid'] == row1['imgguid']].iloc[0]

        n += 1
        if not np.isnan(row2['imgangle']):
            path1 = fd + '/' + id + '/' + f

            fd2 = 'cache_{}/{}/'.format(exp, plantid)
            if not os.path.isdir(fd2):
                os.mkdir(fd2)
            path2 = fd2 + '{}_{}.csv'.format(task, int(row2['imgangle']))
            shutil.copyfile(path1, path2)

# ===== verify image_index coherency ===============================================================================

selec = index2[index2['exp'] == 'ARCH2021-05-27']
selec = selec[~selec['imgangle'].isna()]
selec = selec[selec['genotype'].isin(['V.vinifera/V6863_H10-PL3', 'V.vinifera/V6860_DO4-PL4'])]

plantid = sorted(selec['plantid'].unique())[2]
s = selec[selec['plantid'] == plantid]
s = s[s['taskid'] == s['taskid'].unique()[10]]
s = s.sort_values('timestamp')
for k, (_, row) in enumerate(s.iterrows()):
    path = 'Z:/{}/{}/{}.png'.format(row['exp'], row['taskid'], row['imgguid'])
    img = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)
    plt.figure(k)
    plt.imshow(img)

# ===== extract images =============================================================================================

selec = index[(index['exp'] == 'DYN2020-05-15') & (index['plantid'] == 7243)]
selec = selec[selec['imgangle'] == 120]

for i, (_, row) in enumerate(selec.iterrows()):
    path1 = 'V:/{}/{}/{}.png'.format(row['exp'], row['taskid'], row['imgguid'])
    path2 = 'data/grapevine/grapevine22/{}.png'.format(row['acquisitiondate']).replace(':', '-')
    shutil.copyfile(path1, path2)

# ===== image frequency / exp ======================================================================================

i = 0
for col, exp in zip(['g', 'r', 'k'], index['exp'].unique()):
    selec = index[index['exp'] == exp]
    plantids = selec['plantid'].unique()
    plantids = sorted(np.random.choice(plantids, int(len(plantids) / 10), replace=False)) if '2022' in exp else plantids
    for plantid in plantids:
        s = selec[selec['plantid'] == plantid]
        gb = s.groupby('taskid')['timestamp'].mean()
        plt.plot(gb.values - min(selec['timestamp']), [i] * len(gb), col+'.-')
        i += 1

