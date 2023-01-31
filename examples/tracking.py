import numpy as np
from deepberry.src.openalea.deepberry.temporal import distance_matrix, points_sets_alignment

"""
data (PhenoArch 2021 - plantid 7794 - angle 120 - half of tasks)
"""

points_sets = [np.array(s[['ell_x', 'ell_y']]) for s in ellipses_sets]

M = distance_matrix(points_sets)

berry_ids = points_sets_alignment(points_sets=points_sets, dist_mat=M)
for k in range(len(ellipses_sets)):
    ellipses_sets[k].loc[:, 'berryid'] = berry_ids[k]
    ellipses_sets[k].loc[:, 'task'] = tasks[k]
final_res = pd.concat(ellipses_sets)
