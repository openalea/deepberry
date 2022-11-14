import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import cv2
import os

from deepberry.src.openalea.deepberry.utils import ellipse_interpolation
from deepberry.src.openalea.deepberry.temporal import distance_matrix, points_sets_alignment

# ====================================================================================================================

PALETTE = np.array(
    [[255, 0, 0], [0, 255, 0], [0, 0, 255], [255, 255, 0], [204, 121, 167], [0, 158, 115],
     [0, 114, 178], [230, 159, 0], [140, 86, 75], [0, 255, 255], [255, 0, 100], [0, 77, 0], [100, 0, 255],
     [100, 0, 0], [0, 0, 100], [100, 100, 0], [0, 100, 100], [100, 0, 100], [255, 100, 100]])
PALETTE = np.array(20 * list(PALETTE) + [[0, 0, 0]])

PATH = 'data/grapevine/results/'

#plant_index = pd.read_csv(PATH + 'image_index_old.csv')

df = pd.read_csv(PATH + 'full_results.csv')
index = pd.read_csv('data/grapevine/image_index.csv')

# 2020
df20 = df[df['exp'] == 'DYN2020-05-15']
# TODO try without this ?
# t_camera = 1592571441  # np.mean(df20.groupby('task')['timestamp'].mean().sort_values()[[2566, 2559]].values)
# df20.loc[df20['timestamp'] > t_camera, ['ell_x', 'ell_y']] += np.array([-4.2, -437.8])
# TODO redo with this
tasks20 = list(df20.groupby('task')['timestamp'].mean().sort_values().reset_index()['task'])
df20 = df20[~((df20['plantid'] == 7232) & (df20['task'].isin(tasks20[:tasks20.index(2385)])))]
df20 = df20[~((df20['plantid'] == 7235) & (df20['task'].isin(tasks20[tasks20.index(2613):])))]
df20 = df20[~((df20['plantid'] == 7238) & (df20['task'].isin(tasks20[:tasks20.index(2387)])))]

t = tasks20[:tasks20.index(2385)] + tasks20[tasks20.index(2613):] + tasks20[:tasks20.index(2387)]

# 2021
df21 = df[df['exp'] == 'ARCH2021-05-27']
df21 = df21[df21['genotype'].isin(['V.vinifera/V6863_H10-PL3', 'V.vinifera/V6860_DO4-PL4'])]
tasks21 = list(df21.groupby('task')['timestamp'].mean().sort_values().reset_index()['task'])
df21 = df21[~((df21['plantid'] == 7763) & (df21['task'].isin(tasks21[tasks21.index(3835):])))]


# ===== accuracy ====================================================================================================

index = pd.read_csv('data/grapevine/image_index.csv')

df = []
for exp in ['DYN2020-05-15', 'ARCH2021-05-27']:

    res = pd.read_csv('X:/phenoarch_cache/cache_{0}/full_results_{0}.csv'.format(exp))

    for plantid in res['plantid'].unique():
        for angle in [k * 30 for k in range(12)]:
            s = res[(res['plantid'] == plantid) & (res['angle'] == angle)]
            acc = np.sum(s['berryid'] != -1) / len(s)
            df.append([exp, plantid, angle, acc])
df = pd.DataFrame(df, columns=['exp', 'plantid', 'angle', 'acc'])

n = 0
plt.ylim((-2, 102))
plt.ylabel('Accuracy (%)')
for exp in df['exp'].unique():
    s0 = df[df['exp'] == exp]
    for plantid in s0['plantid'].unique():
        s = s0[s0['plantid'] == plantid]
        plt.plot([n] * len(s), s['acc'] * 100, 'ko')
        plt.plot([n] * 2, [min(s['acc'] * 100), max(s['acc'] * 100)], 'k-')
        n += 1
    n += 5


# ===== image frequency ==============================================================================================

plantid = 7232
s = df20[df20['plantid'] == plantid]
t = sorted(s.groupby('task')['t'].mean().values)

plt.plot(np.diff(t[::12]), 'ko')

n = []
dt = []
df22 = df[df['exp'] == 'ARCH2022-05-18']
for plantid in df22['plantid'].unique():
    s = df22[df22['plantid'] == plantid]
    t = sorted(s.groupby('task')['t'].mean().values)
    dt += list(np.diff(t))
    n += [len(t)]
    print(plantid, len(t), round(np.mean(np.diff(t)), 1), round(np.median(np.diff(t)), 1))

# ====================================================================================================================

def get_rgb_ellipses(ellipses, index, exp, plantid, task, angle, grapeid=0, save=False):

    s = index[(index['exp'] == exp) & (index['plantid'] == plantid) &
              (index['imgangle'] == angle) & (index['grapeid'] == grapeid)]
    tasks = np.array(s.groupby('taskid')['timestamp'].mean().sort_values().reset_index()['taskid'])
    i_task = np.where(tasks == task)[0][0]
    row_index = s[s['taskid'] == task]

    selec_ellipses = ellipses[(ellipses['exp'] == exp) & (ellipses['plantid'] == plantid) &
                            (ellipses['angle'] == angle) & (ellipses['task'] == task) &
                              (ellipses['grapeid'] == grapeid)]

    if len(row_index) != 1:
        print('several rows !')
        return
    else:
        row_index = row_index.iloc[0]

    path1 = 'Z:/{}/{}/{}.png'.format(exp, task, row_index['imgguid'])

    img = cv2.cvtColor(cv2.imread(path1), cv2.COLOR_BGR2RGB)

    for _, ell in selec_ellipses.iterrows():
        x, y, w, h, a = ell[['ell_x', 'ell_y', 'ell_w', 'ell_h', 'ell_a']]
        lsp_x, lsp_y = ellipse_interpolation(x=x, y=y, w=w, h=h, a=a, n_points=100)
        img = cv2.polylines(img, [np.array([lsp_x, lsp_y]).T.astype('int32')], True, (255, 0, 0), 5)

    if save:
        path2 = 'data/grapevine/temporal/{}_{}_{}_{}_{}.png'.format(exp, plantid, grapeid, i_task, task)
        plt.imsave(path2, img)

    return img


# ===== load images =============================================================================================

gb = df21.groupby(['plantid', 'grapeid']).size().reset_index()

for _, row in gb.iterrows():

    plantid, grapeid = row['plantid'], row['grapeid']

    selec = df21[(df21['plantid'] == plantid) & (df21['grapeid'] == grapeid)]
    angle = selec.groupby('angle').size().sort_values().index[-1]
    selec = selec[selec['angle'] == angle]

    tasks = np.array(selec.groupby('task')['timestamp'].mean().sort_values().reset_index()['task'])

    i_tasks = np.linspace(0, len(tasks) - 1, 10).astype(int)

    for i_task in i_tasks:
        task = tasks[i_task]

        selec_task = selec[selec['task'] == task]

    # selec_index = index[(index['exp'] == exp) & (index['genotype'].isin(['V.vinifera/V6863_H10-PL3', 'V.vinifera/V6860_DO4-PL4']))]
    # gb = selec_index.groupby(['plantid', 'grapeid']).size().sort_values()
    # selec_index = selec_index[(selec_index['plantid'] == plantid) & (selec_index['grapeid'] == grapeid)]

        selec_index = index[(index['exp'] == exp) & (index['plantid'] == plantid) & (index['grapeid'] == grapeid)
                            & (index['imgangle'] == angle) & (index['taskid'] == task)]

        for k, (_, row_img) in enumerate(selec_index.sort_values('timestamp').iterrows()):
            path1 = 'Z:/{}/{}/{}.png'.format(exp, task, row_img['imgguid'])
            path2 = 'data/grapevine/temporal/{}/{}_{}_{}_{}.png'.format(exp, plantid, grapeid, i_task, task)
            img = cv2.cvtColor(cv2.imread(path1), cv2.COLOR_BGR2RGB)
            ellipses = selec_task[selec_task['task'] == task]
            for _, ell in ellipses.iterrows():
                x, y, w, h, a = ell[['ell_x', 'ell_y', 'ell_w', 'ell_h', 'ell_a']]
                lsp_x, lsp_y = ellipse_interpolation(x=x, y=y, w=w, h=h, a=a, n_points=100)
                img = cv2.polylines(img, [np.array([lsp_x, lsp_y]).T.astype('int32')], True, (255, 0, 0), 5)
            plt.imsave(path2, img)


# ===== powerpoint illustration ================================================================================

selec = df21[(df21['plantid'] == 7794) & (df21['grapeid'] == 0) & (df21['angle'] == 180)]
tasks = np.array(selec.groupby('task')['timestamp'].mean().sort_values().reset_index()['task'])

selec_index = index[(index['exp'] == 'ARCH2021-05-27') & (index['plantid'] == 7794) &
                    (index['grapeid'] == 0) & (index['imgangle'] == 180)]

plt.figure()
plt.gca().set_aspect('equal', adjustable='box')
i1, i2 = 0, 50
for i, col in zip([i1, i2], ['r', 'b']):
    path = selec_index[selec_index['taskid'] == tasks[i]].iloc[0]['imgguid']
    full_path = 'Z:/ARCH2021-05-27/{}/{}.png'.format(tasks[i], path)
    img = cv2.cvtColor(cv2.imread(full_path), cv2.COLOR_BGR2RGB)
    s = selec[selec['task'] == tasks[i]]
    # plt.figure()
    # plt.imshow(img)
    plt.plot(s['ell_x'], s['ell_y'], col + 'o')
    for _, row in s.iterrows():
        x, y, w, h, a = row['ell_x'], row['ell_y'], row['ell_w'], row['ell_h'], row['ell_a']
        ell = ellipse_interpolation(x, y, w, h, a, n_points=100)
        # plt.plot(ell[0], ell[1], col + '-')


s1, s2 = selec[selec['task'] == tasks[i1]], selec[selec['task'] == tasks[i2]]

centers1, centers2 = np.array(s1[['ell_x', 'ell_y']]), np.array(s2[['ell_x', 'ell_y']])

reg = AffineRegistration(**{'X': centers1, 'Y': centers2})
centers2_reg = reg.register()[0]
plt.figure()
plt.gca().set_aspect('equal', adjustable='box')
plt.plot(centers1[:, 0], centers1[:, 1], 'r*', markersize=10)
plt.plot(centers2[:, 0], centers2[:, 1], 'bo', markersize=10)
plt.plot(centers2_reg[:, 0], centers2_reg[:, 1], 'bo', markersize=10)

reg = AffineRegistration(**{'X': centers2, 'Y': centers1})
centers1_reg = reg.register()[0]

plt.figure()
plt.gca().set_aspect('equal', adjustable='box')
for _, row in s1.iterrows():
    x, y, w, h, a = row['ell_x'], row['ell_y'], row['ell_w'], row['ell_h'], row['ell_a']
    ell = ellipse_interpolation(x, y, w, h, a, n_points=100)
    plt.plot(ell[0], ell[1], 'k-')
for _, row in s2.iterrows():
    x, y, w, h, a = row['ell_x'], row['ell_y'], row['ell_w'], row['ell_h'], row['ell_a']
    ell = ellipse_interpolation(x, y, w, h, a, n_points=100)
    plt.plot(ell[0], ell[1], 'b-')

# ==============================================================================================================

# visu n_berries = f(t) for each plant, to find abnormal snapshots
selec = df[(df['exp'] == exp)]
selec = selec[~selec['task'].isin([3797, 3798, 3804, 3810, 3811, 3819, 3827, 3829, 3831, 3843])]
gb = selec.groupby(['plantid', 'grapeid']).size().reset_index()
for _, row in gb.iterrows():
    s = selec[(selec['plantid'] == row['plantid']) & (selec['grapeid'] == row['grapeid'])]
    gb2 = s.groupby('task')[['timestamp', 'ell_y']].agg(['mean', 'size']).reset_index().sort_values(('timestamp', 'mean'))
    plt.plot(gb2['timestamp']['mean'], gb2['ell_y']['mean'], '.-', label='{}_{}'.format(plantid, grapeid))
# plt.legend()

# ==== quantify camera shift on DYN2020-05-15 ==================================================================

# selec = df[(df['exp'] == 'DYN2020-05-15')]
#
# shift = []
# for angle in [k * 30 for k in range(12)]:
#     shift_a = []
#     for plantid in selec['plantid'].unique():
#         s = selec[(selec['plantid'] == plantid) & (selec['angle'] == angle)]
#         # angle = df[(df['exp'] == exp) & (df['plantid'] == plantid)].groupby('angle').size().sort_values().index[-1]
#         # s = s[s['angle'] == angle]
#         s1 = s[s['task'] == 2566]
#         s2 = s[s['task'] == 2559]
#
#         if len(s1) > 10 and len(s2) > 10:
#             dx, dy = topo_registration(s1, s2)
#             shift.append([angle, plantid, dx, dy])
#         else:
#             print(plantid, angle)
# shift = pd.DataFrame(shift, columns=['angle', 'plantid', 'dx', 'dy'])
# gb = shift.groupby('angle')[['dx', 'dy']].median()
# print(gb.reset_index().median())


# ==== CPD tree ================================================================================================

"""
IDEES
-premiere passe de l'algo, puis affinement : on recale chaque image vs la ref grace aux couples de baies, et on 
regarde si on a des nouveaux matchs. Ou bien pas forcemment avec la ref, on itère des "rematch" entre deux sets i,j
-distance + complexe que linéaire (détail)
"""

# plantid, grapeid, angle = 7794, 0, 180

for df_exp in [df21, df20]:

    for plantid in df_exp['plantid'].unique():

        selec = df_exp[df_exp['plantid'] == plantid]
        angles = list(selec.groupby('angle').size().sort_values(ascending=False).index)

        print('=====', plantid, '=====')

        for angle in [k * 30 for k in range(12)]:

            # print(plantid, grapeid, angle)

            s = selec[selec['angle'] == angle]

            # TODO CACHE filter tasks here
            tasks = list(s.groupby('task')['timestamp'].mean().sort_values().reset_index()['task'])
            # ellipses_sets = [s[s['task'] == task][['ell_x', 'ell_y', 'ell_w', 'ell_h', 'ell_a']] for task in tasks]
            ellipses_sets = [s[s['task'] == task] for task in tasks]
            # TODO don't forget this later in the code!
            ellipses_sets = [s for s in ellipses_sets if len(s) > 1]

            # TODO CACHE camera shift here
            points_sets = [np.array(s[['ell_x', 'ell_y']]) for s in ellipses_sets]

            # TODO : put matrix computation inside points_set_alignment()
            matrix_path = 'data/grapevine/temporal/distance_matrix/{}_{}_{}.npy'.format(
                s.iloc[0]['exp'], plantid, angle)
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
            final_res = pd.concat(ellipses_sets)

            final_res.to_csv('data/grapevine/temporal/results/{}_{}_{}.csv'.format(
                s.iloc[0]['exp'], plantid, angle))


# ===== explore tracking results ====================================================================================

fd = 'data/grapevine/temporal/results/'

res = []
for file in os.listdir(fd):
    print(file)
    res.append(pd.read_csv(fd + file))
res = pd.concat(res)

exp = 'DYN2020-05-15'  # 'ARCH2021-05-27'
selec = res[res['exp'] == exp]
for plantid in selec['plantid'].unique():
    s0 = selec[selec['plantid'] == plantid]
    for grapeid in s0['grapeid'].unique():
        plt.figure('{}_{}'.format(plantid, grapeid))
        s = s0[s0['grapeid'] == grapeid]

        for angle in [k * 30 for k in range(12)]:
        # angle = s.groupby('angle').size().sort_values().index[-1]

            s_angle = s[s['angle'] == angle]
            gb = s_angle.groupby('t')['berryid'].agg(['nunique', 'size']).reset_index()
            plt.ylim((-2, 102))
            plt.xlabel('Time (days)')
            plt.ylabel('Percentage of berries tracked (%)')
            plt.plot(gb['t'], 100 * gb['nunique'] / gb['size'], '.-',
                     label='{}° ; n = {}'.format(angle, round(len(s_angle) / len(s_angle['task'].unique()), 1)))

        plt.legend()

# details of what appends for one

s_angle = res[(res['plantid'] == 7245) & (res['angle'] == 60)]

tasks = list(s_angle.groupby('task').mean().sort_values('t').reset_index()['task'])

da_all = []
for i in range(len(tasks) - 1):
    s = s_angle[s_angle['task'].isin([tasks[i], tasks[i + 1]])]
    da = []
    for k in [k for k in s['berryid'] if k != -1]:
        s_k = s[s['berryid'] == k]
        if len(s_k) == 2:
            da.append(max(s_k['area']) / min(s_k['area']))
    da_all.append(da)
plt.plot(np.arange(len(da_all)), [np.median(da) for da in da_all], 'r.-')
# plt.plot(np.arange(len(da_all)), [len(da) for da in da_all], 'k.')


# ===== graph algorithm for distance matrix =========================================================================

import networkx as nx

M_int = M.copy()
# M_int[M_int > 10] *= 10
M_int[M_int == float('inf')] = 99999
M_int = (100 * (M_int ** 1.)).astype(int)

G = nx.Graph()
for i1 in range(len(M) - 1):
    for i2 in np.arange(i1 + 1, len(M)):
        G.add_edge(i1, i2, weight=np.min(M_int[i1, i2]), capacity=99999)

i_start = 100
print(tasks[i_start])
paths = []
pairs = []
v_max = []
for i in range(len(M)):
    res = nx.shortest_path(G, source=i_start, target=i, weight='weight')
    for pair in zip(res[:-1], res[1:]):
        if pair not in pairs:
            pairs.append(list(pair))

    paths.append(res)
    vals = [np.min(M[i1, i2]) for i1, i2 in zip(res[:-1], res[1:])]
    if vals:
        v_max.append(np.max(vals))
    print(tasks[i], [round(k, 1) for k in vals])

for [i1, i2] in pairs:
    print(np.argmin(M[i1, i2]), round(np.min(M[i1, i2]), 1))

# comparison with other method
pairs2 = pairs_order(M, threshold=8, i_start=50)




