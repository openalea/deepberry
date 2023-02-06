import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ===== accuracy ====================================================================================================

index = pd.read_csv('data/grapevine/image_index.csv')

df = []
for exp in ['DYN2020-05-15', 'ARCH2021-05-27']:

    res = pd.read_csv('X:/phenoarch_cache/cache_{0}_NEW/full_results_temporal_{0}.csv'.format(exp))

    # remove plantids included in the training set
    if exp == 'DYN2020-05-15':
        res = res[res['plantid'] != 7243]
    if exp == 'ARCH2021-05-27':
        res = res[~res['plantid'].isin([7764, 7781])]

    for plantid in res['plantid'].unique():
        for angle in [k * 30 for k in range(12)]:
            s = res[(res['plantid'] == plantid) & (res['angle'] == angle)]
            acc = np.sum(s['berryid'] != -1) / len(s)
            df.append([exp, plantid, angle, acc])
df = pd.DataFrame(df, columns=['exp', 'plantid', 'angle', 'acc'])

for k, plantid in enumerate(df['plantid'].unique()):
    s = df[df['plantid'] == plantid]
    col = 'r' if plantid in [7781, 7791, 7245, 7236] else 'k'
    plt.plot([k] * len(s), s['acc'], '.', color=col)
    plt.plot(k, np.median(s['acc']), 'o', color=col)

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


# ===== details =====================================================================================================

from deepberry.src.openalea.deepberry.temporal import points_sets_alignment

exp = 'DYN2020-05-15'

res = pd.read_csv('X:/phenoarch_cache/cache_{0}/full_results_temporal_{0}.csv'.format(exp))

# for k_plantid, plantid in enumerate(res['plantid'].unique()[:8]):
#     angle = res[res['plantid'] == plantid].groupby('angle').size().sort_values().index[-1]

for plantid in res['plantid'].unique():
    plt.figure(plantid)

    all_mat = []
    for k_angle, angle in enumerate([k * 30 for k in range(12)]):

        M = np.load('X:/phenoarch_cache/cache_{}/distance_matrix/{}/{}.npy'.format(exp, plantid, angle))

        M2 = M[:, :, 0].copy()
        for i in range(len(M)):
            for j in range(len(M)):
                if i < j:
                    M2[i, j] = np.min(M[i, j][[0, 2]])
                elif i > j:
                    M2[i, j] = np.min(M[i, j][[1, 2]])

        all_mat.append(M2)

        # plt.subplot(2, 4, k_plantid + 1)
        # plt.title('plant {}'.format(k_plantid + 1), fontsize=20)

        plt.subplot(3, 5, k_angle + 1)
        plt.title('{}°'.format(angle), fontsize=20)

        plt.imshow(np.minimum(M2, 50))
        plt.rcParams.update({'font.size': 10})
        # set_threshold = 8
        # plt.plot(np.where(M2 < set_threshold)[0], np.where(M2 < set_threshold)[1], 'r.', markersize=1)
        cb = plt.colorbar()

    lst = [len(m) for m in all_mat]
    all_mat = [m for m in all_mat if len(m) == max(set(lst), key=lst.count)]
    print(len(all_mat))

    M2 = np.mean(np.array(all_mat), axis=0)
    plt.subplot(3, 5, k_angle + 3)
    plt.imshow(np.minimum(M2, 50))
    plt.rcParams.update({'font.size': 10})
    cb = plt.colorbar()

accuracies = []
for angle in [k * 30 for k in range(12)]:
    s = res[(res['plantid'] == plantid) & (res['angle'] == angle)]
    accuracies.append(len(s[s['berryid'] != -1]) / len(s))

accuracies = []
t0, tn = 116, 139 + 1
for angle in [k * 30 for k in range(12)]:
    M = np.load('X:/phenoarch_cache/cache_{}/distance_matrix/{}/{}.npy'.format(exp, plantid, angle))

    s = res[(res['plantid'] == plantid) & (res['angle'] == angle)]
    tasks = list(s.groupby('task')['timestamp'].mean().sort_values().reset_index()['task'])
    ellipses_sets = [s[s['task'] == task] for task in tasks]
    points_sets = [np.array(s[['ell_x', 'ell_y']]) for s in ellipses_sets]

    berry_ids = points_sets_alignment(points_sets=points_sets[t0:tn], dist_mat=M[t0:tn, t0:tn, :])

    x = [k for bi in berry_ids for k in bi]
    acc = len([xi for xi in x if xi != -1]) / len(x)
    accuracies.append(acc)
    print(acc)
print(np.mean(accuracies))












