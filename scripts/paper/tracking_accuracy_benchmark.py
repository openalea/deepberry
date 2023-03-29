import pandas as pd
import numpy as np

from deepberry.scripts.paper.ruptures import RUPTURES
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

        # camera shift
        if exp == 'DYN2020-05-15':
            print('camera shift')
            t_camera = 1592571441  # between tasks 2566 and 2559
            for i_task, task in enumerate(tasks):
                timestamp = s[s['task'] == task]['timestamp'].mean()
                if timestamp > t_camera:
                    points_sets[i_task] += np.array([-4.2, -437.8])

        matrix_path = 'X:/phenoarch_cache/cache_{}/distance_matrix/{}/{}.npy'.format(exp, plantid, angle)
        M = np.load(matrix_path)

        assert len(M) == len(points_sets)

        M_noreg = M.copy()
        M_noreg[:, :, [1, 2]] = 999999999

        for tree_order, registration, subperiods in [[True, True, False],
                                                     [True, False, False],
                                                     [False, True, False],
                                                     [False, False, False],
                                                     [False, False, True],
                                                     [True, True, True]]:

            dist_mat = M if registration else M_noreg

            if subperiods:
                timings = [0] + RUPTURES[exp][plantid] + [len(tasks)]
                sum_tot, len_tot = 0, 0
                for k in range(len(timings) - 1):
                    t0, tn = timings[k], timings[k + 1]
                    berry_ids = tracking(points_sets=points_sets[t0:tn], dist_mat=dist_mat[t0:tn, t0:tn, :],
                                         do_tree_order=tree_order)
                    has_id = [b != -1 for ids in berry_ids for b in ids]
                    sum_tot += sum(has_id)
                    len_tot += len(has_id)
                acc = sum_tot / len_tot

            else:
                berry_ids = tracking(points_sets, dist_mat=dist_mat, do_tree_order=tree_order)
                has_id = [b != -1 for ids in berry_ids for b in ids]
                acc = sum(has_id) / len(has_id)

            print(exp, plantid, tree_order, registration, subperiods, round(acc, 3))

            df_benchmark.append([exp, plantid, angle, registration, tree_order, subperiods, acc])

df_benchmark = pd.DataFrame(df_benchmark, columns=['exp', 'plantid', 'angle', 'registration',
                                                   'tree_order', 'subperiods', 'acc'])


df_benchmark.groupby(['registration', 'tree_order', 'subperiods', 'exp']).mean()

df_benchmark.groupby(['registration', 'tree_order', 'subperiods']).mean()













