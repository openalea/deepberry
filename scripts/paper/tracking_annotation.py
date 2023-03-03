import cv2
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from deepberry.src.openalea.deepberry.utils import ellipse_interpolation

index = pd.read_csv('data/grapevine/image_index.csv')
colors = [np.random.randint(0, 255 + 1, 3) for _ in range(500)]

res_dict = {}
for exp in ['DYN2020-05-15', 'ARCH2021-05-27']:
    dfi = pd.read_csv('X:/phenoarch_cache/cache_{0}/full_results_temporal_{0}.csv'.format(exp))
    if exp == 'DYN2020-05-15':
        dfi = dfi[~(dfi['task'] == 2656)]
    dfi['t'] = (dfi['timestamp'] - min(dfi['timestamp'])) / 3600 / 24
    res_dict[exp] = dfi

# ===== what is annotated =============================================================================================

n_dates = 10

df_anot = []
for exp in ['DYN2020-05-15', 'ARCH2021-05-27']:
    res = res_dict[exp]
    for plantid in res['plantid'].unique():
        selec0 = res[res['plantid'] == plantid]
        angle = selec0.groupby('angle').size().sort_values().index[-1]
        selec = selec0[selec0['angle'] == angle]
        tasks = np.random.choice(selec['task'].unique(), n_dates, replace=False)
        tasks = list(selec[selec['task'].isin(tasks)].groupby('task')['timestamp'].mean().sort_values().index)
        for task in tasks:
            df_anot.append([exp, plantid, angle, task])
df_anot = pd.DataFrame(df_anot, columns=['exp', 'plantid', 'angle', 'task'])

# df_anot.to_csv('data/grapevine/paper/tracking_annotation/anot.csv', index=False)

# =====================================================================================================================

df_anot = pd.read_csv('data/grapevine/paper/tracking_annotation/anot.csv')

# TODO only considers berryids seen at least twice in selected tasks

"""
DYN2020-05-15 (n=623)
7232:
7240: new_45(99) t4->5 / new_52(62) t4->5
7238:
7245: 
7244: new_19 t8->10 / new_36 9->10 / new_44 4->5 9->10 / new_56 5->6 8->9(back)
new_67 1->3 / new_65 3->5 / new_72 6->7->8(back) / new_78 2->3 9->10(back)
7235: new_24 9->10 / 
7233: new_4 6->7
7236: new_13 2->3 4->5(back) / new_62 8->9 /
7243: new_49 8->9 / new_57 4->9 / new_71 1->2 / new_104 1->2 / new_109 4->5->6->7 8->9 (all backs) / 
new_123 1->2

ARCH2021-05-27 (n = 793)
7788: new_20 1->2 4->6 / new_21 4->7 / new_22 3->8
7794:
7791: new_48 5->6 / new_95 5->8 8->9->10 (back) / new_100 2->3
7772:
7763:
7783:
7764:
7781:
7760:
"""
# ===== accuracy ======================================================================================================

anot = []
for exp in df_anot['exp'].unique():
    for plantid in df_anot[df_anot['exp'] == exp]['plantid'].unique():
        tasks = list(df_anot[(df_anot['exp'] == exp) & (df_anot['plantid'] == plantid)]['task'])
        angle = df_anot[(df_anot['exp'] == exp) & (df_anot['plantid'] == plantid)]['angle'].iloc[0]
        anot.append([exp, plantid, angle, tasks])

k_row = 17

exp, plantid, angle, tasks = anot[k_row]

print(exp, plantid)

res = res_dict[exp]

selec = res[(res['exp'] == exp) & (res['plantid'] == plantid) & (res['angle'] == angle) & (res['task'].isin(tasks))]

# gb_id = selec[selec['task'].isin(tasks)].groupby('berryid').size()
# common_berryids = list(gb_id.index[np.where(gb_id >= 2)])
# n += len(common_berryids) - 1

# camera shift
t_camera = 1592571441  # between tasks 2566 and 2559
y_shift = 437
selec.loc[selec['timestamp'] < t_camera, 'ell_y'] += y_shift

gb = selec.groupby('berryid')['ell_y'].mean().sort_values()
dic_newid = {i: k for k, i in enumerate(gb.index)}
selec['berryid_new'] = [dic_newid[i] if i != -1 else i for i in selec['berryid']]

ymin = min(selec[selec['berryid'] != -1]['ell_y']) - 100
ymax = max(selec[selec['berryid'] != -1]['ell_y']) + 100

plt.close('all')
for task in tasks:
    print(task)

    s = selec[selec['task'] == task]
    row_index = index[(index['exp'] == exp) & (index['plantid'] == plantid) &
                      (index['imgangle'] == angle) & (index['taskid'] == task)].iloc[0]
    img = cv2.cvtColor(cv2.imread('Z:/{}/{}/{}.png'.format(exp, task, row_index['imgguid'])), cv2.COLOR_BGR2RGB)

    if exp == 'DYN2020-05-15':
        if s['timestamp'].mean() > t_camera:
            img = np.concatenate((img, np.zeros((y_shift, img.shape[1], img.shape[2])).astype(float)))
        else:
            img = np.concatenate((np.zeros((y_shift, img.shape[1], img.shape[2])).astype(float), img))
            # s['ell_y'] += y_shift

    plt.figure()
    plt.ylim((ymax, ymin))
    plt.imshow(img / 255., alpha=1.)
    for _, row in s.iterrows():
        # if row['berryid'] in common_berryids:
        if row['berryid_new'] != -1:
            x, y, w, h, a = row[['ell_x', 'ell_y', 'ell_w', 'ell_h', 'ell_a']]
            col = colors[int(row['berryid_new'])] / 255
            lsp_x, lsp_y = ellipse_interpolation(x=x, y=y, w=w, h=h, a=a, n_points=100)
            plt.plot(lsp_x, lsp_y, '-', color=col)
            plt.text(x, y, str(row['berryid_new']), fontsize=9, ha='center', va='center',
                     color=col)

















