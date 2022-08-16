import pandas as pd
import numpy as np
import shutil
import matplotlib.pyplot as plt
import cv2
from scipy.optimize import minimize, basinhopping
import os

from functools import partial
from pycpd import AffineRegistration, RigidRegistration, DeformableRegistration
import numpy as np


PATH = 'data/grapevine/results/'
ellipse_path = PATH + 'ellipse_old/'
plant_index = pd.read_csv(PATH + 'image_index_old.csv')

plantid = int(os.listdir(ellipse_path)[0])

df = []
for f in os.listdir(ellipse_path + str(plantid)):
    dfi = pd.read_csv(ellipse_path + '{}/{}'.format(plantid, f))
    dfi[['task', 'angle']] = [int(k) for k in f[:-4].split('_')]
    df.append(dfi)
df = pd.concat(df)

# angle with most berries
angle = df.groupby('angle').size().sort_values().reset_index()['angle'].iloc[-1]

selec = df[df['angle'] == angle]
tasks = sorted(selec['task'].unique())
for i in range(len(tasks) - 1):
    s1 = selec[selec['task'] == tasks[i]]
    s2 = selec[selec['task'] == tasks[i + 1]]

    plt.plot(s1['ell_x'], s1['ell_y'], 'r*')
    plt.plot(s2['ell_x'], s2['ell_y'], 'b*')


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

# ===== Point Set Registration ====================================================================================

t1 = 0
t2 = 1

plt.figure()
plt.cla()
ellipses = res[list(res.keys())[t1]]['pred']
print(len(ellipses))
plt.plot(ellipses['ell_x'], ellipses['ell_y'], 'ro')
X = np.array(ellipses[['ell_x', 'ell_y']])

ellipses = res[list(res.keys())[t2]]['pred']
print(len(ellipses))
plt.plot(ellipses['ell_x'], ellipses['ell_y'], 'bo')
Y = np.array(ellipses[['ell_x', 'ell_y']])


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
reg = RigidRegistration(**{'X': X, 'Y': Y})
# reg = DeformableRegistration(**{'X': X, 'Y': Y})
reg.register(callback)
plt.show()

plt.plot(X[:, 0], X[:, 1], 'ro')
plt.plot(Y[:, 0], Y[:, 1], 'bo')
plt.plot(reg.TY[:, 0], reg.TY[:, 1], 'b*')

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





