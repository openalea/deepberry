import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

DIR_OUTPUT = 'data/grapevine/paper/'

# ===== accuracy ====================================================================================================

df = []
for exp in ['DYN2020-05-15', 'ARCH2021-05-27']:

    res = pd.read_csv('X:/phenoarch_cache/cache_{0}/full_results_temporal_{0}.csv'.format(exp))

    # remove plantids included in the training set
    if exp == 'DYN2020-05-15':
        res = res[res['plantid'] != 7243]
    if exp == 'ARCH2021-05-27':
        res = res[~res['plantid'].isin([7764, 7781])]

    for plantid in res['plantid'].unique():
        s0 = res[res['plantid'] == plantid]
        best_angle = s0.groupby('angle').size().sort_values().index[-1]
        for angle in [k * 30 for k in range(12)]:
            s = s0[s0['angle'] == angle]
            acc = np.sum(s['berryid'] != -1) / len(s)
            df.append([exp, plantid, angle, angle == best_angle, acc])
df = pd.DataFrame(df, columns=['exp', 'plantid', 'angle', 'is_best_angle', 'acc'])

plt.xlabel('plant')
plt.ylabel('% of tracked berries')
for k, plantid in enumerate(df['plantid'].unique()):
    s = df[df['plantid'] == plantid]
    plt.plot([k] * 2, [min(s['acc'] * 100), max(s['acc'] * 100)], 'k-')
    for _, row in s.iterrows():
        plt.plot(k, 100 * row['acc'], 'o', color='r' if row['is_best_angle'] else 'k')

# mean cluster_acc / exp
df.groupby(['exp', 'plantid', 'angle'])['acc'].mean().groupby('exp').mean()
# mean cluster_acc(best_angle) / exp
df[df['is_best_angle']].groupby(['exp', 'plantid', 'angle'])['acc'].mean().groupby('exp').mean()

# ===== ruptures =====================================================================================================

# i means rupture from t=i-1 and t=i
ruptures = {'DYN2020-05-15': {7240: [7, 87],  # pot rotation / camera y_shift
                              7238: [],
                              7232: [9, 41, 71],  # small pot rotation / grape rotation / pot rotation
                              7245: [39, 65, 116],  # pot rotation / pot rotation / pot rotation
                              7235: [18, 57, 83],  # ? / pot rotation / y_shift leading to leaf occlusion
                              7233: [70, 88, 122],  # pot rotation / y_shift leading to leaf occlusion and small rot /
                              # pot rotation
                              7236: [33, 61, 85],  # pot rotation / pot rotation / y_shift changes berry visibles
                              7243: [60, 91],  # new berries due to branch fall / rotation
                              7244: [40, 89, 123]  # pot rotation / grape rotation / pot rotation
                              },
            'ARCH2021-05-27': {7791: [28, 47],  # pot rotation / pot rotation
                               7760: [39],  # pot rotation
                               7781: [25],  # pot rotation
                               7764: [48],  # small non-linear grape deformation
                               7788: [47],  # non-linear grape deformation
                               7794: [],
                               7772: [51],  # small non-linear grape deformation
                               7763: [],
                               7783: [47]  # pot rotation
                               }
            }

# => 21 rotations (70%), 1 unknown, 1 branch fall, 4 y_shift (13%),  3 deformations (10%)

# ===== details =====================================================================================================


def plot_acc(df):
    df['identified'] = [1 if k != -1 else 0 for k in df['berryid']]
    gb = df.groupby('task').mean().sort_values('timestamp')
    plt.ylim((-0.02, 1.02))
    plt.gca().set_box_aspect(1)
    # plt.plot(np.argmax(gb['identified']), np.max(gb['identified']), 'bo')
    plt.plot(2 * [np.argmax(gb['identified'])], [-1, 2], 'b--', alpha=0.5)
    plt.plot(np.arange(len(gb)), gb['identified'], 'k-')


def plot_matrix(m, ruptures_list=[]):
    m2 = m[:, :, 0].copy()
    for i in range(len(m2)):
        for j in range(len(m2)):
            if i < j:
                m2[i, j] = np.min(M[i, j][[0, 2]])
            elif i > j:
                m2[i, j] = np.min(M[i, j][[1, 2]])
    all_mat.append(m2)
    plt.imshow(np.minimum(m2, 50))
    plt.rcParams.update({'font.size': 10})
    # set_threshold = 8
    # plt.plot(np.where(M2 < set_threshold)[0], np.where(M2 < set_threshold)[1], 'r.', markersize=1)
    # cb = plt.colorbar()
    for t in ruptures_list:
        plt.plot([t - 0.5] * 2, [0, len(m2) - 1], 'r-', linewidth=2)


exp = 'DYN2020-05-15'
# exp = 'ARCH2021-05-27'

res = pd.read_csv('X:/phenoarch_cache/cache_{0}/full_results_temporal_{0}.csv'.format(exp))

plt.figure()
for k_plantid, plantid in enumerate(res['plantid'].unique()[:6]):

    best_angle = res[res['plantid'] == plantid].groupby('angle').size().sort_values().index[-1]

    all_mat = []
    for angle in [best_angle]:

        plt.subplot(2, 6, k_plantid + 1)

        M = np.load('X:/phenoarch_cache/cache_{}/distance_matrix/{}/{}.npy'.format(exp, plantid, angle))
        ruptures_list = ruptures[exp][plantid]
        plot_matrix(M, ruptures_list)

        plt.subplot(2, 6, 6 + k_plantid + 1)

        selec = res[(res['plantid'] == plantid) & (res['angle'] == angle)]
        plot_acc(selec)

plt.savefig(DIR_OUTPUT + 'matrix_multi_plant.png', bbox_inches='tight')


plt.figure()
plantid = 7245
for k_angle, angle in enumerate([0, 60, 120, 180, 240, 300]):

    plt.subplot(2, 6, k_angle + 1)

    M = np.load('X:/phenoarch_cache/cache_{}/distance_matrix/{}/{}.npy'.format(exp, plantid, angle))
    plot_matrix(M)

    plt.subplot(2, 6, 6 + k_angle + 1)

    selec = res[(res['plantid'] == plantid) & (res['angle'] == angle)]
    plot_acc(selec)


plt.savefig(DIR_OUTPUT + 'matrix_multi_angle.png', bbox_inches='tight')


# lst = [len(m) for m in all_mat]
# all_mat = [m for m in all_mat if len(m) == max(set(lst), key=lst.count)]
# print(len(all_mat))
#
# M2 = np.mean(np.array(all_mat), axis=0)
# plt.subplot(3, 5, k_plantid + 3)
# plt.imshow(np.minimum(M2, 50))
# plt.rcParams.update({'font.size': 10})
# cb = plt.colorbar()
#

# ===== visualize a rupture to classify it

import cv2

index = pd.read_csv('data/grapevine/image_index.csv')

res_dict = {exp: pd.read_csv('X:/phenoarch_cache/cache_{0}/full_results_temporal_{0}.csv'.format(exp))
            for exp in ['DYN2020-05-15', 'ARCH2021-05-27']}

exp = 'ARCH2021-05-27'

k_plantid = 8

res = res_dict[exp]
plantid = res_dict[exp]['plantid'].unique()[k_plantid]
print(plantid)
print(ruptures[exp][plantid])
angle = res[res['plantid'] == plantid].groupby('angle').size().sort_values().index[-1]
s = res[(res['plantid'] == plantid) & (res['angle'] == angle)]
tasks = list(s.groupby('task')['timestamp'].mean().sort_values().reset_index()['task'])

# just to be sure
M = np.load('X:/phenoarch_cache/cache_{}/distance_matrix/{}/{}.npy'.format(exp, plantid, angle))
assert len(M) == len(tasks)

selec_index = index[(index['exp'] == exp) & (index['plantid'] == plantid) & (index['imgangle'] == angle)]


timing = 39
plt.close('all')
for t in range(timing - 2, timing + 2):
    row_img = selec_index[selec_index['taskid'] == tasks[t]].iloc[0]
    img = cv2.cvtColor(cv2.imread('Z:/{}/{}/{}.png'.format(exp, row_img['taskid'], row_img['imgguid'])),
                       cv2.COLOR_BGR2RGB)
    plt.figure(t)
    plt.imshow(img)


# ===== run algo inside each stable time-windows

res_ruptures = []
for exp in ruptures.keys():
    res = pd.read_csv('X:/phenoarch_cache/cache_{0}/full_results_temporal_{0}.csv'.format(exp))
    for plantid in ruptures[exp].keys():
        print(exp, plantid)

        angle = res[res['plantid'] == plantid].groupby('angle').size().sort_values().index[-1]

        M = np.load('X:/phenoarch_cache/cache_{}/distance_matrix/{}/{}.npy'.format(exp, plantid, angle))

        s = res[(res['plantid'] == plantid) & (res['angle'] == angle)]
        tasks = list(s.groupby('task')['timestamp'].mean().sort_values().reset_index()['task'])
        ellipses_sets = [s[s['task'] == task] for task in tasks]
        points_sets = [np.array(s[['ell_x', 'ell_y']]) for s in ellipses_sets]

        assert len(M) == len(tasks)

        acc_raw = len(s[s['berryid'] != -1]) / len(s)
        print('raw acc : {:.2}'.format(acc_raw))

        timings = [0] + ruptures[exp][plantid] + [len(tasks)]
        n, n_tot = 0, 0
        for k in range(len(timings) - 1):
            t0, tn = timings[k], timings[k + 1]
            berry_ids = points_sets_alignment(points_sets=points_sets[t0:tn], dist_mat=M[t0:tn, t0:tn, :])
            x = [k for bi in berry_ids for k in bi]
            n += len([xi for xi in x if xi != -1])
            n_tot += len(x)

        acc_new = n / n_tot
        print('new acc : {:.2}'.format(acc_new))

        res_ruptures.append([exp, plantid, acc_raw, acc_new])

res_ruptures = pd.DataFrame(res_ruptures, columns=['exp', 'plantid', 'acc_raw', 'acc_new'])

# ====================================================================================================================

accuracies = []
for angle in [k * 30 for k in range(12)]:
    s = res[(res['plantid'] == plantid) & (res['angle'] == angle)]
    accuracies.append(len(s[s['berryid'] != -1]) / len(s))












