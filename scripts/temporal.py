import pandas as pd
import numpy as np
import shutil
import matplotlib.pyplot as plt
import cv2
from scipy.optimize import minimize, basinhopping
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

plantid = 7236
selec = df[df['plantid'] == plantid]

# angle with most berries
angle = selec.groupby('angle').size().sort_values().reset_index()['angle'].iloc[-1]
selec = selec[selec['angle'] == angle]

tasks = list(selec.groupby('task')['timestamp'].mean().reset_index().sort_values('timestamp')['task'])
for i in range(len(tasks) - 1):

    # i = 140
    i = 139

    s1 = selec[selec['task'] == tasks[i]]
    s2 = selec[selec['task'] == tasks[i + 1]]

    plt.figure()
    plt.gca().set_aspect('equal', adjustable='box')
    for s, color in zip([s1, s2], ['r', 'b']):
        for _, ell in s.iterrows():
            x, y, w, h, a = ell[['ell_x', 'ell_y', 'ell_w', 'ell_h', 'ell_a']]
            lsp_x, lsp_y = ellipse_interpolation(x=x, y=y, w=w, h=h, a=a, n_points=20)
            plt.plot(lsp_x, lsp_y, color + '-')

    ellipses1 = [ellipse_interpolation(e['ell_x'], e['ell_y'], e['ell_w'], e['ell_h'], e['ell_a'], n_points=20)
     for _, e in s1.iterrows()]
    ellipses2 = [ellipse_interpolation(e['ell_x'], e['ell_y'], e['ell_w'], e['ell_h'], e['ell_a'], n_points=20)
                 for _, e in s2.iterrows()]

    ell1 = ellipses1[0]
    ell2 = ellipses2[5]
    plt.plot(ell1[0], ell1[1], 'b-')
    plt.plot(ell2[0], ell2[1], 'r-')
    d = IOU(ell1.T, ell2.T)

    # distance matrix
    D = np.zeros((len(ellipses1), len(ellipses2)))
    for i, ell1 in enumerate(ellipses1):
        for j, ell2 in enumerate(ellipses2):
            d = IOU(ell1.T, ell2.T)
            d2 = ell1[['ell_x', 'ell_y']] - ell2[['ell_x', 'ell_y']]
            d = d if d != 0 else
            D[i, j] = d

    # pairwise matches
    seg_pairs = []
    for k in range(5):
        i_min, j_min = np.unravel_index(D.argmin(), D.shape)
        seg_pairs.append([segs1[i_min], segs2[j_min]])
        D[i_min, :] = float('inf')
        D[:, j_min] = float('inf')
    seg_pairs = np.array(seg_pairs)

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





