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


# ===== 2020 experiment =====

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

df['grapeid'] = 0

df20 = df.copy()

# ===== 2021 experiment =====

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
    n_imgs = max(set(gb.values), key=list(gb.values).count)
    if n_imgs % 12 != 0:  # True for only 1 plant
        df.at[selec.index, 'imgangle'] = None
        df.at[selec.index, 'grapeid'] = None
    else:
        for task in selec['taskid'].unique():
            s = selec[selec['taskid'] == task].sort_values('acquisitiondate')
            if len(s) != n_imgs:
                df.at[s.index, 'imgangle'] = None
                df.at[s.index, 'grapeid'] = None
            else:
                df.at[s.index, 'imgangle'] = [k * 30 for k in range(12)] * int(n_imgs / 12)
                df.at[s.index, 'grapeid'] = [i for i in range(int(len(s) / 12)) for _ in range(12)]

df21 = df.copy()

# ===== 2022 experiment =====

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

# TODO : 13% images are lost, could be saved with grapeid
for plantid in df['plantid'].unique():
    selec = df[df['plantid'] == plantid]
    #print([len(selec[selec['taskid'] == task]) for task in selec['taskid'].unique()])
    for task in selec['taskid'].unique():
        s = selec[selec['taskid'] == task].sort_values('acquisitiondate')
        if len(s) != 12:
            df.at[s.index, 'imgangle'] = None
        else:
            df.at[s.index, 'imgangle'] = [k * 30 for k in range(12)]

df['grapeid'] = 0

df22 = df.copy()

# ===== all =====

df = pd.concat((df20, df21, df22))

df['daydate'] = [d[5:10] for d in df['acquisitiondate']]
df['timestamp'] = df.apply(lambda row: date_to_timestamp(row['acquisitiondate']), axis=1)

df.to_csv('data/grapevine/image_index.csv', index=False)

# ====================================================================================================================

df = pd.read_csv('data/grapevine/image_index.csv')

# ===== extract images (training set) ================================================================================

# files = [f for f in os.listdir('data/grapevine/dataset/images') if 'ARCH2022' in f] + os.listdir(
#     'data/grapevine/dataset/images/TODO')
# plantids_2022 = np.unique([int(f.split('_')[1]) for f in files])
selec = df[(df['exp'] == 'ARCH2022-05-18') & (~df['imgangle'].isna())]
selec = selec[selec['taskid'] > 6070]
selec = selec[selec['plantid'].isin(selec[selec['plantcode'].isin([p for p in selec['plantcode'] if 'WD2' in p])]['plantid'])]  # WD2
# selec = selec[~selec['plantid'].isin(plantids_2022)]
folder = 'data/grapevine/grapevine22/'
for _, row in selec.sample(300).iterrows():
    path1 = 'V:/ARCH2022-05-18/{}/{}.png'.format(row['taskid'], row['imgguid'])
    path2 = folder + 'ARCH2022-05-18_{}_{}_{}.png'.format(int(row.plantid), int(row.taskid), int(row.imgangle))
    shutil.copyfile(path1, path2)

# ===== extract images =============================================================================================

selec = df[(df['exp'] == 'DYN2020-05-15') & (df['plantid'] == 7243)]
selec = selec[selec['imgangle'] == 120]

for i, (_, row) in enumerate(selec.iterrows()):
    path1 = 'V:/{}/{}/{}.png'.format(row['exp'], row['taskid'], row['imgguid'])
    path2 = 'data/grapevine/grapevine22/{}.png'.format(row['acquisitiondate']).replace(':', '-')
    shutil.copyfile(path1, path2)

# TODO remove
for k, f in enumerate(os.listdir('data/grapevine/gif/')):
    img = cv2.cvtColor(cv2.imread('data/grapevine/gif/' + f), cv2.COLOR_BGR2RGB)
    f2 = f[:10] + f[10:-4].replace('-', ':')
    task = selec[selec['acquisitiondate'] == f2]['taskid'].iloc[0]
    img = cv2.putText(img, str(task), (200, 900), cv2.FONT_HERSHEY_SIMPLEX, 6, (255, 0, 0), 20, cv2.LINE_AA)
    plt.imsave('data/grapevine/gif3/{}_{}.png'.format(k, task), img)


# ==== create gif =======================================================================================
# full gif script for angle

plantid = 7243
exp = 'DYN2020-05-15'
task = 2505

df_img = pd.read_csv('data/grapevine/image_index.csv')
s_img = df_img[(df_img['exp'] == exp) & (df_img['plantid'] == plantid) & (df_img['taskid'] == task)]

df_ell = pd.read_csv('data/grapevine/results/full_results.csv')
s_ell = df_ell[(df_ell['exp'] == exp) & (df_ell['plantid'] == plantid) & (df_ell['task'] == task)]

imgs_gif = []
for _, row in s_img.iterrows():

    path1 = 'V:/{}/{}/{}.png'.format(row['exp'], int(row['taskid']), row['imgguid'])
    img = cv2.cvtColor(cv2.imread(path1), cv2.COLOR_BGR2RGB)

    ellipses = s_ell[s_ell['timestamp'] == row['timestamp']]

    for _, ell in ellipses.iterrows():
        x, y, w, h, a = ell[['ell_x', 'ell_y', 'ell_w', 'ell_h', 'ell_a']]
        lsp_x, lsp_y = ellipse_interpolation(x=x, y=y, w=w, h=h, a=a, n_points=100)
        img = cv2.polylines(img, [np.array([lsp_x, lsp_y]).T.astype('int32')], True, (255, 0, 0), 5)

    img[:150] *= 0
    txt = 'camera angle = {}'.format(int(row['imgangle']))
    img = cv2.putText(img, txt, (300, 120), cv2.FONT_HERSHEY_SIMPLEX, 4, (255, 255, 255), 5, cv2.LINE_AA)

    path2 = 'data/grapevine/gif2/{}.png'.format(int(row['imgangle']))
    plt.imsave(path2, img)

    imgs_gif.append(img)

imgs_gif = [Image.fromarray(np.uint8(img)) for img in imgs_gif]
fps = 2
imgs_gif[0].save('data/grapevine/2020_7243_task2505_{}fps.gif'.format(fps), save_all=True, append_images=imgs_gif[1:],
                 optimize=True, duration=1000/fps, loop=0)



# ===== image frequency / exp ======================================================================================

i = 0
for col, exp in zip(['g', 'r', 'k'], df['exp'].unique()):
    selec = df[df['exp'] == exp]
    plantids = selec['plantid'].unique()
    plantids = sorted(np.random.choice(plantids, int(len(plantids) / 10), replace=False)) if '2022' in exp else plantids
    for plantid in plantids:
        s = selec[selec['plantid'] == plantid]
        gb = s.groupby('taskid')['timestamp'].mean()
        plt.plot(gb.values - min(selec['timestamp']), [i] * len(gb), col+'.-')
        i += 1

# ===== load images (temporal) ==================================================================================

selec = df[(df['exp'] == 'ARCH2021-05-27') & (~df['imgangle'].isna())]

selec = selec[selec['grapeid'] == 0]

# selec.groupby('plantid')[['taskid', 'daydate']].nunique()

plantid = 1029
angle = 150

selec = selec[(selec['plantid'] == plantid) & (selec['imgangle'] == angle)]
selec = selec.sort_values('timestamp')

for i, (_, row) in enumerate(selec.iterrows()):

    path1 = 'V:/{}/{}/{}.png'.format(row['exp'], row['taskid'], row['imgguid'])
    path2 = 'data/grapevine/grapevine22/{}_{}_{}.png'.format(row['exp'], int(row.plantid), i)
    shutil.copyfile(path1, path2)
