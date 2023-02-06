import pandas as pd
import numpy as np
import os

from deepberry.src.openalea.deepberry.temporal import points_sets_alignment

index = pd.read_csv('data/grapevine/image_index.csv')

# exp = 'ARCH2021-05-27'
exp = 'DYN2020-05-15'
res = pd.read_csv('X:/phenoarch_cache/cache_{0}/full_results_temporal_{0}.csv'.format(exp))

plantid = 7245
selec = res[res['plantid'] == plantid]
angle = selec.groupby('angle').size().sort_values().index[-1]
s = selec[selec['angle'] == angle]

# =====================================================================================================================

tasks = list(s.groupby('task')['timestamp'].mean().sort_values().reset_index()['task'])

ellipses_sets = [s[s['task'] == task] for task in tasks]
points_sets = [np.array(s[['ell_x', 'ell_y']]) for s in ellipses_sets]

matrix_path = 'X:/phenoarch_cache/cache_{}/distance_matrix/{}/{}.npy'.format(exp, plantid, angle)
M = np.load(matrix_path)

# import matplotlib.pyplot as plt
# plt.gca().set_aspect('equal', adjustable='box')
# s1, s2 = ellipses_sets[70], ellipses_sets[80]
# s1, s2 = s1[s1['berryid'] != -1], s2[s2['berryid'] != -1]
# for _, row1 in s1.iterrows():
#     for _, row2 in s2.iterrows():
#         if row1['berryid'] == row2['berryid']:
#             plt.plot([row1['ell_x'], row2['ell_x']], [row1['ell_y'], row2['ell_y']], 'ko-')

res = []
for t_start in range(len(M)):
    print(t_start)
    berry_ids = points_sets_alignment(points_sets=points_sets, dist_mat=M, t_start=t_start)
    f = [len([k for k in x if k != -1]) / len(x) for x in berry_ids]
    res.append(f)
    # x = [k for l in berry_ids for k in l]
    # print((len(x) - len(np.where(np.array(x) == -1)[0])) / len(x))

from scipy.ndimage import uniform_filter1d
plt.figure()
sig = np.mean([np.abs(np.diff(r)) for r in res], axis=0)
plt.plot(sig)
plt.plot(uniform_filter1d(sig, size=5))


for t_start, r in enumerate(res):
    plt.plot(np.abs(np.diff(r)))

for k in range(len(ellipses_sets)):
    ellipses_sets[k].loc[:, 'berryid'] = berry_ids[k]
    ellipses_sets[k].loc[:, 'task'] = tasks[k]
final_res = pd.concat(ellipses_sets)

# final_res.to_csv(plantid_t_path + '{}_{}.csv'.format(angle, time_period), index=False)
final_res.to_csv(plantid_t_path + '{}.csv'.format(angle), index=False)
