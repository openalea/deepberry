"""
viewtypeid: 1 = top, 2 = side, 3 = xyz

2020: 16k images xyz. 1 genotype x 9 plantes x 150 tasks (50 days)
2021: 40k images xyz. 5 genotype (+ Odium; 6gxe en tout) x 2-8 plantes (tot=25) x 80 tasks (34 days)
2022 (21-07): 65k images xyz. 3000 plants (1700 xyz).


2022:
24 photos = 2 grappes. Toujours prises en photo dans le meme ordre
24 imgs, puis 12 ? parce-que ce sont rendu compte que trop galere de suivre 2-3 grappes, donc en gardent qu'une
mais laquelle ? je peux surement pas savoir, besoin tableau llorenc avec position des grappes
"""

import pandas as pd
import time
import datetime


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

df.to_csv('data/grapevine/image_index.csv', index=False)


