import pandas as pd
import numpy as np
import os

from deepberry.src.openalea.deepberry.temporal import tracking

df_benchmark = []

for exp in ['DYN2020-05-15', 'ARCH2021-05-27']:

    res = pd.read_csv('X:/phenoarch_cache/cache_{0}/full_results_temporal_{0}.csv'.format(exp))

    for plantid in res['plantid'].unique():

        selec = res[res['plantid'] == plantid]
        angle = selec.groupby('angle').size().sort_values().index[-1]
        s = selec[selec['angle'] == angle]

        tasks = list(s.groupby('task')['timestamp'].mean().sort_values().reset_index()['task'])
        ellipses_sets = [s[s['task'] == task] for task in tasks]
        points_sets = [np.array(s[['ell_x', 'ell_y']]) for s in ellipses_sets]

        # TODO camera shift 2020 !

        matrix_path = 'X:/phenoarch_cache/cache_{}/distance_matrix/{}/{}.npy'.format(exp, plantid, angle)
        M = np.load(matrix_path)

        assert len(M) == len(points_sets)

        M_noreg = M.copy()
        M_noreg[:, :, [1, 2]] = 999999999

        for tree_order, registration in [[True, True], [True, False], [False, True], [False, False]]:

            berry_ids = tracking(points_sets, M if registration else M_noreg, tree_order=tree_order)

            for k in range(len(ellipses_sets)):
                ellipses_sets[k].loc[:, 'berryid'] = berry_ids[k]
                ellipses_sets[k].loc[:, 'task'] = tasks[k]
            final_res = pd.concat(ellipses_sets)

            acc = np.sum(final_res['berryid'] > -1) / len(final_res)

            print(exp, plantid, tree_order, registration, round(acc, 3))

            df_benchmark.append([exp, plantid, angle, registration, tree_order, acc])

df_benchmark = pd.DataFrame(df_benchmark, columns=['exp', 'plantid', 'angle', 'registration', 'tree_order', 'acc'])


# df_benchmark.groupby(['exp', 'registration', 'tree_order']).mean()
df_benchmark.groupby(['registration', 'tree_order']).mean()













