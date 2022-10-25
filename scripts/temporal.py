import pandas as pd
import numpy as np
import shutil
import matplotlib.pyplot as plt
import cv2
import os

from scipy.optimize import minimize, basinhopping, linear_sum_assignment
from scipy.spatial.distance import directed_hausdorff

from deepberry.src.openalea.deepberry.utils import ellipse_interpolation

from functools import partial
from pycpd import AffineRegistration, RigidRegistration, DeformableRegistration

import time

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
t_camera = 1592571441  # np.mean(df20.groupby('task')['timestamp'].mean().sort_values()[[2566, 2559]].values)
df20.loc[df20['timestamp'] > t_camera, ['ell_x', 'ell_y']] += np.array([-4.2, -437.8])

# 2021
df21 = df[df['exp'] == 'ARCH2021-05-27']
df21 = df21[df21['genotype'].isin(['V.vinifera/V6863_H10-PL3', 'V.vinifera/V6860_DO4-PL4'])]

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


# ==================================================================================================================


def scaled_cpd(X, Y, X_add=None, Y_add=None, transformation='affine'):
    """
    Coherent Point Drift algorithm, for point set registration, with inputs scaling.

    X: array of shape (n1, 2) containing the 2d coordinates of the reference/fixed point cloud
    Y: array of shape (n2, 2) containing the 2d coordinates of the moving point cloud
    X_add: array of shape (n1) containing an additional feature describing the points in X
    Y_add: array of shape (n2) containing an additional feature describing the points in Y
    transformation: type of registration, among 'rigid', 'affine', 'deformable'
    """

    if (len(X) <= 2) or (len(Y) <= 2):
        return Y

    # scaling of X and Y
    q, m = 0.5 * np.max(np.max(X, axis=0) - np.min(X, axis=0)), np.mean(X, axis=0)
    X_scaled, Y_scaled = (X - m) / q, (Y - m) / q

    # (optional) scaling of X_add and Y_add. They are then added to X and Y as third dimensions.
    if X_add is not None and Y_add is not None:
        X_add_scaled = (X_add - np.mean(X_add)) / np.std(X_add)
        Y_add_scaled = (Y_add - np.mean(Y_add)) / np.std(Y_add)
        X_scaled = np.concatenate((X_scaled, np.array([X_add_scaled]).T), axis=1)
        Y_scaled = np.concatenate((Y_scaled, np.array([Y_add_scaled]).T), axis=1)

    # point set registration (Coherent Point Drift)
    reg_functions = {'rigid': RigidRegistration, 'affine': AffineRegistration, 'deformable': DeformableRegistration}
    Y_scaled_reg = reg_functions[transformation](**{'X': X_scaled, 'Y': Y_scaled}).register()[0]

    # reverse Y scaling
    Y_scaled_reg = Y_scaled_reg[:, :2]
    Y_reg = (Y_scaled_reg * q) + m

    return Y_reg


def distance_matrix(sets):
    """
    sets = list of ellipses sets, where each "set" is a dataframe with x,y,w,h,a parameters for each ellipse.
    """

    M = np.zeros((len(sets), len(sets), 3))
    M[np.arange(len(M)), np.arange(len(M))] = float('inf')  # diagonal

    for i1 in range(len(M) - 1):
        for i2 in np.arange(i1 + 1, len(M)):
            print(i1, i2)

            set1, set2 = sets[i1], sets[i2]

            centers1, centers2 = np.array(set1[['ell_x', 'ell_y']]), np.array(set2[['ell_x', 'ell_y']])

            # reg = AffineRegistration(**{'X': centers1, 'Y': centers2})
            # centers2_reg = reg.register()[0]
            # reg = AffineRegistration(**{'X': centers2, 'Y': centers1})
            # centers1_reg = reg.register()[0]
            # reg_topo = topo_registration(set1, set2)
            # centers2_reg_topo = centers2 + reg_topo

            centers2_reg = scaled_cpd(centers1, centers2, transformation='affine')
            centers1_reg = scaled_cpd(centers2, centers1, transformation='affine')
            # centers2_reg_d = scaled_cpd(centers1, centers2, transformation='deformable')
            # centers1_reg_d = scaled_cpd(centers2, centers1, transformation='deformable')
            # centers2_reg_a = scaled_cpd(centers1, centers2, set1['area'], set2['area'], transformation='affine')
            # centers1_reg_a = scaled_cpd(centers2, centers1, set2['area'], set1['area'], transformation='affine')

            # ===== distance matrix ==============
            mats, scores = [], []
            for ctr1, ctr2 in [[centers1, centers2],
                               [centers1, centers2_reg], [centers2, centers1_reg]]:
                D = np.zeros((len(ctr1), len(ctr2)))
                for k, c1 in enumerate(ctr1):
                    D[k] = np.sqrt(np.sum((c1 - ctr2) ** 2, axis=1))
                d = 0.5 * (np.median(np.min(D, axis=0)) + np.median(np.min(D, axis=1)))
                scores.append(d)

            M[[i1, i2], [i2, i1]] = scores
            print(scores)

    return M


def pairs_order(M, threshold=8, i_start=0):

#for i_start in range(len(M)):

    scores = []
    # for threshold in np.linspace(1, 20, 50):

    M_min = np.min(M, axis=2)
    i_done = [i_start]
    l_previous = 0
    n_steps = [0, 0]
    pairs = []

    while len(i_done) != len(M):

        while l_previous != len(i_done) and len(i_done) != len(M):
            l_previous = len(i_done)
            # i_matches = sorted(set(np.where(M[i_done, :] < threshold)[1]))
            # a, b = np.where(M[i_done, :] < threshold)

            # select all possible matches under threshold
            i_match = np.where(np.min(M_min[i_done, :], axis=0) < threshold)[0]
            m = M_min[i_done, :][:, i_match]
            i_match_sources = np.array(i_done)[np.argmin(m, axis=0)]

            for i_ms, i_m in zip(i_match_sources, i_match):
                pairs.append([i_ms, i_m])

            i_done = sorted(i_done + list(i_match))
            M_min[np.ix_(i_done, i_done)] = 999
            n_steps[0] += 1
            # print(len(i_done))

        if len(i_done) != len(M):
            # if no match score under threshold, add only one match with the minimum score
            i1, i2 = np.unravel_index(M_min[i_done, :].argmin(), M_min[i_done, :].shape)
            i_match_source, i_match = np.array(i_done)[i1], i2
            score = M_min[i_match_source, i_match]
            i_done += [i_match]
            pairs.append([i_match_source, i_match])
            M_min[np.ix_(i_done, i_done)] = 999
            n_steps[1] += 1
            # print(len(i_done), '(add match above threshold: s = {}, i = {})'.format(round(score, 1), i_match))
            scores.append(score)

    return pairs


# def topo_features(ellipses, nvecs=3):
#
#     centers = np.array(ellipses[['ell_x', 'ell_y']])
#
#     # distance matrix
#     D = np.zeros((len(ellipses), len(ellipses)))
#     for i, c in enumerate(centers):
#         D[i] = np.sqrt(np.sum((c - centers) ** 2, axis=1))
#         D[i, i] = float('inf')
#
#     vecs = []
#     for i_berry in range(len(centers)):
#         vec = []
#         ng = D[i_berry, :].argsort()[:nvecs]
#         x, y = centers[i_berry]
#         for i_ng in ng:
#             x_ng, y_ng = centers[i_ng]
#             vec += [x - x_ng, y - y_ng]
#         vecs.append(vec)
#
#     return np.array(vecs)
#
#
# def topo_registration(s1, s2, nvecs=3):
#     """ s1, s2 = two sets (dataframes) of ellipses """
#
#     # TODO : vec[d1, a1, d2, a2, d3, a3] for rigid transfo (consistent with rotation+translation+reflection)
#
#     # TODO vec[[d, a]i], imax = 5, 10 ? faire un matching entre les deux vec et prendre la d mediane
#
#     # topologic features extraction
#     n = min(nvecs, min(len(s1), len(s2)))
#     ft1 = topo_features(s1, nvecs=n)
#     ft2 = topo_features(s2, nvecs=n)
#
#     # distance matrix
#     F = np.zeros((len(ft1), len(ft2)))
#     for i, vec in enumerate(ft1):
#         F[i] = np.sqrt(np.sum(np.abs(vec - ft2), axis=1))
#
#     # feature matching (greedy algorithm)
#     matches, _ = matching(F)
#     good_matches = matches[:(int(len(matches) / 4))]
#
#     centers1 = np.array(s1[['ell_x', 'ell_y']])
#     centers2 = np.array(s2[['ell_x', 'ell_y']])
#
#     # plt.figure()
#     # plt.gca().invert_yaxis()
#     # plt.plot(centers1[:, 0], centers1[:, 1], 'bo', alpha=0.5)
#     # plt.plot(centers2[:, 0], centers2[:, 1], 'ro', alpha=0.5)
#     # for i1, i2 in good_matches:
#     #     (x1, y1), (x2, y2) = centers1[i1], centers2[i2]
#     #     plt.plot([x1, x2], [y1, y2], 'g-')
#
#     # translation registration
#     dx, dy = np.mean(centers1[good_matches[:, 0]] - centers2[good_matches[:, 1]], axis=0)
#
#     # plt.figure()
#     # plt.plot(centers1[:, 0], centers1[:, 1], 'bo', alpha=0.5)
#     # centers2_reg = centers2 + np.array([dx, dy])
#     # plt.plot(centers2_reg[:, 0], centers2_reg[:, 1], 'ro', alpha=0.5)
#
#     return np.array([dx, dy])


# def get_features2(ellipses):
#
#     centers = np.array(ellipses[['ell_x', 'ell_y']])
#
#     # distance matrix
#     D = np.zeros((len(ellipses), len(ellipses)))
#     for i, c in enumerate(centers):
#         D[i] = np.sqrt(np.sum((c - centers) ** 2, axis=1))
#         D[i, i] = float('inf')
#
#     vecs = []
#     for i_berry in range(len(centers)):
#         vec = []
#         ng = D[i_berry, :].argsort()[:5]
#         x, y = centers[i_berry]
#         for i_ng in ng:
#             x_ng, y_ng = centers[i_ng]
#             distance = np.sqrt((x - x_ng) ** 2 + (y - y_ng) ** 2)
#             area = ellipses.iloc[i_ng]['area']
#             vec += [[distance, area]]
#         vecs.append(vec)
#
#     vecs = np.array(vecs)
#
#     return vecs


def matching(M, threshold=float('inf')):
    """ greedy algorithm """
    M2 = M.copy()
    matches = []
    dists = []
    for k in range(min(M2.shape)):
        i_min, j_min = np.unravel_index(M2.argmin(), M2.shape)
        d = M2[i_min, j_min]
        if d < threshold:  # TODO while loop would be faster
            matches.append([i_min, j_min])
            dists.append(d)
        M2[i_min, :] = float('inf')
        M2[:, j_min] = float('inf')
    return np.array(matches), dists


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

# ===== clustering method ======================================================================================

tasks = np.array(selec.groupby('task')['timestamp'].mean().sort_values().reset_index()['task'])

i1, i2, = 2, 121

s1 = selec[selec['task'] == tasks[i1]]
s2 = selec[selec['task'] == tasks[i2]]

centers1 = np.array(s1[['ell_x', 'ell_y']])
centers2 = np.array(s2[['ell_x', 'ell_y']])

plt.figure()
plt.plot(centers1[:, 0], centers1[:, 1], 'bo', alpha=0.5)
plt.plot(centers2[:, 0], centers2[:, 1], 'ro', alpha=0.5)

plt.gca().set_aspect('equal', adjustable='box')
plt.plot(selec['ell_x'], selec['ell_y'], 'k.', alpha=0.05)


def get_features3(ellipses):

    centers = np.array(ellipses[['ell_x', 'ell_y']])

    # distance matrix
    D = np.zeros((len(ellipses), len(ellipses)))
    for i, c in enumerate(centers):
        D[i] = np.sqrt(np.sum((c - centers) ** 2, axis=1))
        D[i, i] = float('inf')

    vecs = []
    for i_berry in range(len(centers)):
        vec = []
        ng = D[i_berry, :].argsort()[:5]
        x, y = centers[i_berry]
        for i_ng in ng:
            x_ng, y_ng = centers[i_ng]
            distance = np.sqrt((x - x_ng) ** 2 + (y - y_ng) ** 2)
            area = ellipses.iloc[i_ng]['area']
            vec += [[distance, area]]
        vecs.append(vec)

    vecs = np.array(vecs)

    return vecs

# ===== just some visu # TODO to remove =======================================================================

for set in np.array(ellipses_sets)[[10, 90]]:
    plt.figure()
    plt.gca().set_aspect('equal', adjustable='box')
    for _, row in set.iterrows():
        x, y, w, h, a = row['ell_x'], row['ell_y'], row['ell_w'], row['ell_h'], row['ell_a']
        ex, ey = ellipse_interpolation(x, y, w, h, a, n_points=100)
        # s = ((row['score'] - 0.95) * 20) ** 15
        s = row['roundness'] ** 7
        plt.plot(ex, ey, '-', color=[1 - s, s, 0.])

s = pd.concat(ellipses_sets)['roundness']
plt.hist(s ** 7, 100)

# ===== test vec # TODO to remove =======================================================================

i1, i2 = 10, 90
set1, set2 = ellipses_sets[i1], ellipses_sets[i2]
centers1, centers2 = np.array(set1[['ell_x', 'ell_y']]), np.array(set2[['ell_x', 'ell_y']])

D = np.zeros((len(centers1), len(centers2)))
for k, c1 in enumerate(centers1):
    D[k] = np.sqrt(np.sum((c1 - centers2) ** 2, axis=1))

# ==== CPD tree ================================================================================================

"""
IDEES
-premiere passe de l'algo, puis affinement : on recale chaque image vs la ref grace aux couples de baies, et on 
regarde si on a des nouveaux matchs. Ou bien pas forcemment avec la ref, on itère des "rematch" entre deux sets i,j
-distance + complexe que linéaire (détail)
"""

# plantid, grapeid, angle = 7794, 0, 180

for df_exp in [df21, df20]:

    gb = df_exp.groupby(['plantid', 'grapeid']).size().reset_index()

    for _, row in gb.iterrows():

        # row = gb.iloc[-3]

        plantid, grapeid = row[['plantid', 'grapeid']]

        selec = df_exp[(df_exp['plantid'] == plantid) & (df_exp['grapeid'] == grapeid)]
        angles = list(selec.groupby('angle').size().sort_values(ascending=False).index)

        print('=====', plantid, grapeid, '=====')

        for angle in [k * 30 for k in range(12)]:

            # print(plantid, grapeid, angle)

            s = selec[selec['angle'] == angle]

            tasks = list(s.groupby('task')['timestamp'].mean().sort_values().reset_index()['task'])
            # ellipses_sets = [s[s['task'] == task][['ell_x', 'ell_y', 'ell_w', 'ell_h', 'ell_a']] for task in tasks]
            ellipses_sets = [s[s['task'] == task] for task in tasks]

            # TODO don't forget this later in the code!
            ellipses_sets = [s for s in ellipses_sets if len(s) > 1]

            # n_berry = round(len(s) / len(s['task'].unique()), 1)
            # print(s.iloc[0]['exp'], plantid, grapeid, angle, n_berry, len(ellipses_sets))

            matrix_path = 'data/grapevine/temporal/distance_matrix/{}_{}_{}_{}.npy'.format(
                s.iloc[0]['exp'], plantid, grapeid, angle)

            if not os.path.isfile(matrix_path):
                print('computing distance matrix...')
                M = distance_matrix(ellipses_sets)
                np.save(matrix_path, M)

            M = np.load(matrix_path)

            if len(ellipses_sets) != len(M):
                print(plantid, grapeid)

            if M.shape[2] > 3:
                M[:, :, [3, 4, 5, 6]] = float('inf')
            # print(len(M), [round(len(m[m == k]) / len(m), 2) for k in range(3)])

            set_threshold = 8

            # selecting optimal value for i_start
            i_best, k_max = None, float('-inf')
            for i_start in range(len(ellipses_sets)):
                set_pairs = pairs_order(M, i_start=i_start, threshold=set_threshold)
                k_above = [k for k, (i, j) in enumerate(set_pairs) if np.min(M[i, j]) > set_threshold]
                k_fail = len(ellipses_sets) - 1 if not k_above else k_above[0]
                if k_fail > k_max:
                    i_best, k_max = i_start, k_fail

            i_start = i_best
            set_pairs = pairs_order(M, i_start=i_start, threshold=set_threshold)

            # ===== match berry ids =====
            # /!\ don't forget that some dates may be removed if not enough ellipse detections

            # init
            ellipses_sets[i_start]['berryid'] = np.arange(len(ellipses_sets[i_start]))

            for i1, i2 in set_pairs:

                s1, s2 = ellipses_sets[i1], ellipses_sets[i2]
                centers1 = np.array(s1[['ell_x', 'ell_y']])
                centers2 = np.array(s2[['ell_x', 'ell_y']])

                # TODO REFACTOR function
                if np.argmin(M[i1, i2]) == 1:
                    centers2 = scaled_cpd(centers1, centers2, transformation='affine')
                elif np.argmin(M[i1, i2]) == 2:
                    centers1 = scaled_cpd(centers2, centers1, transformation='affine')

                # berry distance matrix
                D = np.zeros((len(centers1), len(centers2)))
                for i, c1 in enumerate(centers1):
                    D[i] = np.sqrt(np.sum((c1 - centers2) ** 2, axis=1))
                # print(i1, i2, round(np.median(np.min(D, axis=0)), 1))

                berry_threshold = 16

                berry_pairs, dists = matching(D, threshold=berry_threshold)

                new_indexes = np.zeros(len(s2)) - 1
                # existing id for berries that matched with the reference
                for b1, b2 in berry_pairs:
                    new_indexes[b2] = s1.iloc[b1]['berryid']
                ellipses_sets[i2]['berryid'] = new_indexes
                # s.at[s2.index, 'berryid'] = new_indexes

            # ===== final res =====

            final_res = pd.concat(ellipses_sets)

            # print(len(final_res[final_res['berryid'] != -1]) / len(final_res) * 100)
            # gb = final_res.groupby('t')['berryid'].agg(['nunique', 'size']).reset_index()
            # plt.plot(gb['t'], 100 * gb['nunique'] / gb['size'], '.-', label=str(out))

            final_res.to_csv('data/grapevine/temporal/results/{}_{}_{}_{}.csv'.format(
                s.iloc[0]['exp'], plantid, grapeid, angle))

# ===== effect of t0 choice ==========================================================

# exp = 'DYN2020-05-15'  # 'ARCH2021-05-27'
selec = df20
for plantid in selec['plantid'].unique():
    s0 = selec[selec['plantid'] == plantid]
    for grapeid in s0['grapeid'].unique():
        s = s0[s0['grapeid'] == grapeid]
        plt.ylim((0, 1))
        gb = s.groupby('task')['t'].agg(['mean', 'size']).reset_index().sort_values('mean')
        plt.plot(gb['mean'], gb['size'] / np.max(gb['size']), 'k-')
        # for angle in [k * 30 for k in range(12)]:
        #     s_angle = s[s['angle'] == angle]
        #     gb = s_angle.groupby('t')['area'].agg(['mean', 'size']).reset_index().sort_values('t')



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

# ==================================================================================================================

import glob
from PIL import Image
fps = 8
paths = ['data/grapevine/temporal/time-series/2612_{}.0.png'.format(k) for k in [k * 30 for k in range(12)]]
imgs = [cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB) for path in paths]
imgs_gif = [cv2.resize(img, tuple((np.array([2048, 2448]) / 4).astype(int))) for img in imgs]

imgs_gif = [Image.fromarray(np.uint8(img)) for img in imgs_gif]
# imgs = (Image.open(f) for f in paths)
img = imgs_gif[0]  # extract first image from iterator
img = next(imgs)
img.save(fp='data/videos/berry_tracking2_small_{}fps.gif'.format(fps), format='GIF', append_images=imgs,
         save_all=True, duration=1000/fps, loop=0)

    # print(threshold, i, len(i_done), n_steps)
    # scores.append(n_steps)
# plt.plot(np.linspace(1, 20, 50), np.array(scores)[:, 0], 'k.-')
# plt.plot(np.linspace(1, 20, 50), np.array(scores)[:, 1], 'r.-')
# plt.plot(np.linspace(1, 20, 50), np.sum(np.array(scores), axis=1), 'g.-')



i1, i2 = 2, 121

s1 = selec[selec['task'] == tasks[i1]]
s2 = selec[selec['task'] == tasks[i2]]
centers1 = np.array(s1[['ell_x', 'ell_y']])
centers2 = np.array(s2[['ell_x', 'ell_y']])
dx, dy = topo_registration(s1, s2)
centers2_reg = centers2 + np.array([dx, dy])
plt.figure()
plt.plot(centers1[:, 0], centers1[:, 1], 'bo', alpha=0.5)
plt.plot(centers2[:, 0], centers2[:, 1], 'ro', alpha=0.5)
plt.figure()
plt.plot(centers1[:, 0], centers1[:, 1], 'bo', alpha=0.5)
plt.plot(centers2_reg[:, 0], centers2_reg[:, 1], 'ro', alpha=0.5)

# ===== tracking ===============================================================================================

for i_task, task in enumerate(tasks[:-1]):

    # i = 0
    # i_task, task = i, tasks[i]

    s1 = selec[selec['task'] == tasks[i_task]]
    s2 = selec[selec['task'] == tasks[i_task + 1]]

    # init
    if i_task == 0:
        ref = np.array(s1[['ell_x', 'ell_y']])

    # centers1 = np.array(s1[['ell_x', 'ell_y']])
    centers2 = np.array(s2[['ell_x', 'ell_y']])

    # point set registration
    reg = AffineRegistration(**{'X': centers2, 'Y': ref})
    ref_reg = reg.register()[0]

    # ===== distance matrix ==============

    # a) euclidian distance between centers
    mats, scores = [], []
    for centers1 in [ref, ref_reg]:
        D = np.zeros((len(centers1), len(centers2)))
        for i, c1 in enumerate(centers1):
            D[i] = np.sqrt(np.sum((c1 - centers2) ** 2, axis=1))
        mats.append(D)
        scores.append(np.median(np.min(D, axis=0)))
    D = mats[np.argmin(scores)]  # chose matrix with lowest score

    ref = [ref, ref_reg][np.argmin(scores)]

    # b) if needed, more accurate distance measurement
    # threshold = 200
    # h_d = max(directed_hausdorff(ell1.T, ell2.T)[0], directed_hausdorff(ell2.T, ell1.T)[0])
    # D[i, j] = h_d

    # ===== pairwise matching ===================

    # greedy algorithm
    threshold = 30
    matches = []
    dists = []
    # h_dists = []
    for k in range(min(D.shape)):
        i_min, j_min = np.unravel_index(D.argmin(), D.shape)
        d = D[i_min, j_min]
        if d < threshold:  # TODO while loop would be faster
            matches.append([i_min, j_min])
            dists.append(d)
        D[i_min, :] = float('inf')
        D[:, j_min] = float('inf')
    matches = np.array(matches)

    # # alternative option : hungarian algorithm
    # matches = np.array(linear_sum_assignment(D)).T

    # ===== save matches (tracking) =================

    if i_task == 0:
        s1['berryid'] = np.arange(len(s1))
        selec.at[s1.index, 'berryid'] = s1['berryid']

    new_indexes = np.zeros(len(s2)) - 1
    # existing id for berries that matched with the reference
    for (i, j), d in zip(matches, dists):
        new_indexes[j] = i  # because berries sorted by id in "refs"
    # new id for berries that didn't match
    # new_indexes[new_indexes == -1] = np.arange(len(ref), len(ref) + len(new_indexes[new_indexes == -1]))
    # save
    selec.at[s2.index, 'berryid'] = new_indexes

    # ref = []
    # for id in range(int(max(selec['berryid']))):
    #     s = selec[selec['berryid'] == id].sort_values('t')
    #     ref.append(np.array(s[['ell_x', 'ell_y']].iloc[-1]))
    # ref = np.array(ref)

    # ===== visu ==================================

    name = '{}) {} vs {}'.format(i_task, tasks[i_task], tasks[i_task + 1])
    print(name, round(np.median(dists), 1), round(scores[0] / scores[1], 1), len(ref))
    all_d.append(np.median(dists))

    # plot
    ellipses1 = [ellipse_interpolation(e['ell_x'], e['ell_y'], e['ell_w'], e['ell_h'], e['ell_a'], n_points=100)
     for _, e in s1.iterrows()]
    ellipses2 = [ellipse_interpolation(e['ell_x'], e['ell_y'], e['ell_w'], e['ell_h'], e['ell_a'], n_points=100)
                 for _, e in s2.iterrows()]
    plt.figure(name)
    plt.gca().set_aspect('equal', adjustable='box')
    # dists = []
    for i, j in matches:
        ell1 = ellipses1[i]
        ell2 = ellipses2[j]
        plt.plot(ell1[0], 2448 - ell1[1], 'b-', linewidth=0.8)
        plt.plot(ell2[0], 2448 - ell2[1], 'r-', linewidth=0.8)
        (x1, y1), (x2, y2) = s1.iloc[i][['ell_x', 'ell_y']], s2.iloc[j][['ell_x', 'ell_y']]
        plt.plot([x1, x2], [2448 - y1, 2448 - y2], 'k-')
        dists.append(np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2))

gb = selec.groupby('task')['berryid'].nunique()
gb = gb[tasks]  # correct order
plt.plot(np.arange(len(gb)), gb, 'k.-')
plt.ylim((0, 5 + np.max(gb)))

plt.figure()
#ids = list(selec.groupby('berryid').size().sort_values(ascending=False).reset_index()['berryid'])
# ids = np.random.choice([id for id in selec['berryid'].unique() if id != -1], 10)
s0 = selec[selec['task'] == tasks[0]]
sn = selec[selec['task'] == tasks[-1]]
ids = [id for id in sn['berryid'] if id in list(s0['berryid'])]
ids = np.random.choice(ids, 10, replace=False)
for id in ids:
#for id in np.unique([id for id in selec['berryid'] if id != -1])[::5]:
    s = selec[selec['berryid'] == id].sort_values('t')
    plt.plot(s['t'], s['area'], '.-')

plt.figure()
for task in tasks:
    s = selec[selec['task'] == task]
    plt.plot(s['t'], s['berryid'], 'k.')




# =======================================================================================================

    # ===== Point Set Registration ======

    X = np.array(s1[['ell_x', 'ell_y']])
    Y = np.array(s2[['ell_x', 'ell_y']])

    def visualize(iteration, error, X, Y, ax):
        plt.cla()
        ax.scatter(X[:, 0],  X[:, 1], color='red', label='Target')
        ax.scatter(Y[:, 0],  Y[:, 1], color='blue', label='Source')
        plt.text(0.87, 0.92, 'Iteration: {:d}\nQ: {:06.4f}'.format(
            iteration, error), horizontalalignment='center', verticalalignment='center', transform=ax.transAxes, fontsize='x-large')
        ax.legend(loc='upper left', fontsize='x-large')
        plt.draw()
        plt.pause(0.1)

    fig = plt.figure()
    fig.add_axes([0, 0, 1, 1])
    callback = partial(visualize, ax=fig.axes[0])

    # reg = AffineRegistration(**{'X': X, 'Y': Y})
    reg = AffineRegistration(**{'X': X, 'Y': Y})
    # reg = DeformableRegistration(**{'X': X, 'Y': Y})
    reg.register(callback)
    plt.show()

    plt.plot(X[:, 0], X[:, 1], 'ro')
    plt.plot(Y[:, 0], Y[:, 1], 'bo')
    plt.plot(reg.TY[:, 0], reg.TY[:, 1], 'b*')


# =================================================================================================================

path_visu = 'data/grapevine/temporal/{}/'.format(int(plantid))
if not os.path.isdir(path_visu):
    os.mkdir(path_visu)
for d in list(res.keys()):
    pred = res[d]['pred']
    pred = pred[pred['score'] > 0.95]
    img = res[d]['img']
    plt.figure()
    plt.imshow(img)
    for _, (x, y, w, h, score) in pred.iterrows():
        plt.plot([x, x, x + w, x + w, x], [y, y + h, y + h, y, y], 'r-')
    plt.savefig(path_visu + '{}.png'.format(d))
plt.close('all')

all_segs = []
for d in list(res.keys()):
    pred = res[d]['pred']
    pred = pred[pred['score'] > 0.95]
    print(d, len(pred))
    segs = []
    for _, (x, y, w, h, score) in pred.iterrows():
        segs.append([[x, x + w], [y, y + h]])
    all_segs.append(segs)
all_segs = np.array(all_segs)

# ==================================================================================================================


def rescale_new_cost(x, seg_pairs):

    dx, dy = x
    seg_pairs2 = seg_pairs.copy()
    seg_pairs2[:, 1, :, :] += np.array([[dx, dx], [dy, dy]])
    dists = [np.sum(np.sqrt(np.sum(np.square(s1 - s2), axis=0))) for (s1, s2) in seg_pairs2]
    cost = np.sum(dists)
    return cost


def fit_rescale(seg_pairs):

    x0 = np.array([0., 0.])
    bounds = ((-10, 10), (-10., 10.))
    opti_local = minimize(fun=rescale_new_cost,
                          x0=x0,
                          bounds=bounds,
                          args=seg_pairs)
    # opti_global = basinhopping(func=fit_length_eval,
    #                            x0=x0,
    #                            niter=100,
    #                            minimizer_kwargs={'args': (n, length_dict, method),
    #                                              'bounds': bounds})
    params = opti_local.x
    return params

# ==================================================================================================================


t1, t2 = 2, 3

# segs1 = target (blue) ; segs2 = moving (red)
segs1, segs2 = np.array(all_segs[t1]), np.array(all_segs[t2])

update_res = {'dx': 0, 'dy': 0}
for k_iter in range(10):

    # distance matrix
    D = np.zeros((len(segs1), len(segs2)))
    for i, s1 in enumerate(segs1):
        for j, s2 in enumerate(segs2):
            d = np.sum(np.sqrt(np.sum(np.square(s1 - s2), axis=0)))
            D[i, j] = d

    # pairwise matches
    seg_pairs = []
    for k in range(5):
        i_min, j_min = np.unravel_index(D.argmin(), D.shape)
        seg_pairs.append([segs1[i_min], segs2[j_min]])
        D[i_min, :] = float('inf')
        D[:, j_min] = float('inf')
    seg_pairs = np.array(seg_pairs)

    score0 = rescale_new_cost([0, 0], seg_pairs)

    # search parameters dx, dy that minimize d(segs1, segs2) by transforming segs2
    dx, dy = fit_rescale(seg_pairs)

    score = rescale_new_cost([dx, dy], seg_pairs)
    print('{}) fit. dx = {}, dy = {}, score : {} -> {}'.format(k_iter, round(dx, 2), round(dy, 2), round(score0, 3), round(score, 3)))

    # update segs2 with the parameters found
    segs2 += np.array([[dx, dx], [dy, dy]])

    # save the update in a global update
    update_res['dx'] += dx
    update_res['dy'] += dy

print(update_res)

# ========================================================================

# plot all segs + pairs of matched segs
plt.figure()
plt.gca().set_aspect('equal', adjustable='box')  # same x y sce
target = all_segs[t1]
source = all_segs[t2]
moved_source = source + np.array([2*[update_res['dx']], 2*[update_res['dy']]])
for col, segs in zip(['b-', 'r-', 'r--'], [target, source, moved_source]):
    for ((xa, xb), (ya, yb)) in segs:
        plt.plot([xa, xb], [ya, yb], col)





