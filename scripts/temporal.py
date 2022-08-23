import pandas as pd
import numpy as np
import shutil
import matplotlib.pyplot as plt
import cv2
from scipy.optimize import minimize, basinhopping
from scipy.spatial.distance import directed_hausdorff
import os


from deepberry.src.openalea.deepberry.prediction import ellipse_interpolation

from functools import partial

from pycpd import AffineRegistration, RigidRegistration, DeformableRegistration

from shapely.geometry import box, Polygon


# TODO use it in DL codes if it's faster ? (nms)
def IOU(pol1_xy, pol2_xy):
    """
    https://stackoverflow.com/questions/58435218/intersection-over-union-on-non-rectangular-quadrilaterals
    """
    # Define each polygon
    polygon1_shape = Polygon(pol1_xy)
    polygon2_shape = Polygon(pol2_xy)

    # Calculate intersection and union, and the IOU
    polygon_intersection = polygon1_shape.intersection(polygon2_shape).area
    polygon_union = polygon1_shape.area + polygon2_shape.area - polygon_intersection
    return polygon_intersection / polygon_union


PATH = 'data/grapevine/results/'

#plant_index = pd.read_csv(PATH + 'image_index_old.csv')

df = pd.read_csv(PATH + 'full_results.csv')
index = pd.read_csv('data/grapevine/image_index.csv')

# plantid, angle = 7236, 120
plantid, angle = 7243, 120
exp = 'DYN2020-05-15'

selec = df[(df['exp'] == exp) & (df['plantid'] == plantid) & (df['angle'] == angle)]

# for loading images
selec_index = index[(index['exp'] == exp) & (index['plantid'] == plantid) & (index['imgangle'] == angle)]

# # load images
# for k, (_, row) in enumerate(selec_index.sort_values('timestamp').iterrows()):
#     path1 = 'V:/{}/{}/{}.png'.format(exp, row['taskid'], row['imgguid'])
#     path2 = 'data/grapevine/temporal/time-series/{}_{}.png'.format(k, row['taskid'])
#     img = cv2.cvtColor(cv2.imread(path1), cv2.COLOR_BGR2RGB)
#     ellipses = selec[selec['task'] == row['taskid']]
#     for _, ell in ellipses.iterrows():
#         x, y, w, h, a = ell[['ell_x', 'ell_y', 'ell_w', 'ell_h', 'ell_a']]
#         lsp_x, lsp_y = ellipse_interpolation(x=x, y=y, w=w, h=h, a=a, n_points=100)
#         img = cv2.polylines(img, [np.array([lsp_x, lsp_y]).T.astype('int32')], True, (255, 0, 0), 5)
#     plt.imsave(path2, img)

all_d = []

selec = selec[~selec['task'].isin([2377, 2379, 2494, 2554])]

selec['berry_id'] = -1

# pb 2380, 2382, 2384, 2385
selec = selec[~selec['task'].isin([2380, 2382, 2384])]

# 2520 -> 2527 legere rotation, donc des baies disparaissent (raparaissent 3-4 frames plus tard)


tasks = list(selec.groupby('task')['timestamp'].mean().sort_values().reset_index()['task'])
# tasks = tasks[7:40]

for i_task, task in enumerate(tasks[:-1]):

    i = 56
    i_task, task = i, tasks[i]

    s1 = selec[selec['task'] == tasks[i_task]]
    s2 = selec[selec['task'] == tasks[i_task + 1]]

    centers1, centers2 = np.array(s1[['ell_x', 'ell_y']]), np.array(s2[['ell_x', 'ell_y']])

    # point set registration
    reg = AffineRegistration(**{'X': centers1, 'Y': centers2})
    centers2_reg = reg.register()[0]

    # DISTANCE MATRIX
    # a) euclidian distance between centers
    mats, dists = [], []
    for c2 in [centers2, centers2_reg]:
        D = np.zeros((len(s1), len(s2)))
        for i, c1 in enumerate(centers1):
            D[i] = np.sqrt(np.sum((c1 - c2) ** 2, axis=1))
        mats.append(D)
        dists.append(np.median(np.min(D, axis=0)))
    D = mats[np.argmin(dists)]  # chose matrix with lowest score

    # b) if needed, more accurate distance measurement
    # threshold = 200
    # h_d = max(directed_hausdorff(ell1.T, ell2.T)[0], directed_hausdorff(ell2.T, ell1.T)[0])
    # D[i, j] = h_d

    # pairwise matches
    matches = []
    dists = []
    # h_dists = []
    for k in range(min(D.shape)):
        i_min, j_min = np.unravel_index(D.argmin(), D.shape)
        matches.append([i_min, j_min])
        dists.append(D[i_min, j_min])
        D[i_min, :] = float('inf')
        D[:, j_min] = float('inf')
    matches = np.array(matches)

    # save matches (tracking)
    if i_task == 0:
        s1['berry_id'] = np.arange(len(s1))
        selec.at[s1.index, 'berry_id'] = s1['berry_id']
    threshold = 30
    new_indexes = np.zeros(len(s2)) - 1
    for (i, j), d in zip(matches, dists):
        if d < threshold:
            new_indexes[j] = s1.iloc[i]['berry_id']
    selec.at[s2.index, 'berry_id'] = new_indexes

    name = '{}) {} vs {}'.format(i_task, tasks[i_task], tasks[i_task + 1])
    print(name, round(np.median(dists), 1))
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

gb = selec.groupby('task')['berry_id'].nunique()
gb = gb[tasks]  # correct order
plt.plot(np.arange(len(gb)), gb, 'k.-')

plt.figure()
ids = list(selec.groupby('berry_id').size().sort_values(ascending=False).reset_index()['berry_id'][1:30])
for id in ids:
#for id in np.unique([id for id in selec['berry_id'] if id != -1])[::5]:
    s = selec[selec['berry_id'] == id].sort_values('t')
    plt.plot(s['t'], s['area'], '.-')

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
plt.gca().set_aspect('equal', adjustable='box') # same x y sce
target = all_segs[t1]
source = all_segs[t2]
moved_source = source + np.array([2*[update_res['dx']], 2*[update_res['dy']]])
for col, segs in zip(['b-', 'r-', 'r--'], [target, source, moved_source]):
    for ((xa, xb), (ya, yb)) in segs:
        plt.plot([xa, xb], [ya, yb], col)





