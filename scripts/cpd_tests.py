import time
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os

from deepberry.src.openalea.deepberry.utils import ellipse_interpolation

from pycpd import AffineRegistration, RigidRegistration, DeformableRegistration
# from probreg import cpd, filterreg, l2dist_regs

index = pd.read_csv('data/grapevine/image_index.csv')

fd = 'data/grapevine/temporal/results/'

res = []
for file in os.listdir(fd):
    print(file)
    res.append(pd.read_csv(fd + file))
res = pd.concat(res)


# ===== test different Point-Set Registration algorithms =======================================================

exp = 'ARCH2021-05-27'
res_exp = res[res['exp'] == exp]

plantid, grapeid, angle = 7794, 0, 180
# plantid, grapeid, angle = np.random.choice(res_exp['plantid']), 0, np.random.choice(res_exp['angle'])
selec = res_exp[(res_exp['plantid'] == plantid) & (res_exp['grapeid'] == grapeid) & (res_exp['angle'] == angle)]
tasks = np.array(selec.groupby('task')['timestamp'].mean().sort_values().reset_index()['task'])

i1, i2 = 0, 50
s1, s2 = selec[selec['task'] == tasks[i1]], selec[selec['task'] == tasks[i2]]
centers1 = np.array(s1[['ell_x', 'ell_y']])
centers2 = np.array(s2[['ell_x', 'ell_y']])

# a) no reg

plt.figure()
plt.gca().set_aspect('equal', adjustable='box')
plt.plot(centers1[:, 0], centers1[:, 1], 'ro')
plt.plot(centers2[:, 0], centers2[:, 1], 'bo', fillstyle='none')

plt.figure()
plt.gca().set_aspect('equal', adjustable='box')
for _, row in s1.iterrows():
    x, y, w, h, a = row['ell_x'], row['ell_y'], row['ell_w'], row['ell_h'], row['ell_a']
    ell = ellipse_interpolation(x, y, w, h, a, n_points=100)
    plt.plot(ell[0], ell[1], 'r-')
for _, row in s2.iterrows():
    x, y, w, h, a = row['ell_x'], row['ell_y'], row['ell_w'], row['ell_h'], row['ell_a']
    ell = ellipse_interpolation(x, y, w, h, a, n_points=100)
    plt.plot(ell[0], ell[1], 'b-')


# b) Affine reg

centers2_reg = scaled_cpd(centers1, centers2)

plt.figure()
plt.gca().set_aspect('equal', adjustable='box')
plt.plot(centers1[:, 0], centers1[:, 1], 'ro')
plt.plot(centers2_reg[:, 0], centers2_reg[:, 1], 'bo', fillstyle='none')

# c) Affine reg 3D

centers2_reg_3d = scaled_cpd(centers1, centers2, X_add=np.array(s1['area']), Y_add=np.array(s2['area']),
                             transformation='affine')

plt.figure()
plt.gca().set_aspect('equal', adjustable='box')
plt.plot(centers1[:, 0], centers1[:, 1], 'ro')
plt.plot(centers2_reg_3d[:, 0], centers2_reg_3d[:, 1], 'bo', fillstyle='none')


# berry distance matrix
D = np.zeros((len(centers1), len(centers2_reg)))
for i, c1 in enumerate(centers1):
    D[i] = np.sqrt(np.sum((c1 - centers2_reg) ** 2, axis=1))
# print(i1, i2, round(np.median(np.min(D, axis=0)), 1))


# # classify matches vs no matches
# berry_pairs, dists = matching(D, threshold=16)
# i1, i2 = berry_pairs[:, 0], berry_pairs[:, 1]  # match
# j1, j2 = np.delete(np.arange(len(centers1)), i1), np.delete(np.arange(len(centers2)), i2)  # no match
# plt.figure()
# plt.gca().set_aspect('equal', adjustable='box')
# plt.plot(centers1[i1, 0], centers1[i1, 1], 'ro')
# plt.plot(centers2_reg[i2, 0], centers2_reg[i2, 1], 'bo', fillstyle='none')
# plt.plot(centers1[j1, 0], centers1[j1, 1], 'ko')
# plt.plot(centers2_reg[j2, 0], centers2_reg[j2, 1], 'ko', fillstyle='none')
#
# # repeat cpd for no match points
# centers1_bis, centers2_bis = centers1[j1], centers2_reg[j2]
# centers2_bis_reg = scaled_cpd(centers1_bis, centers2_bis)
# plt.figure()
# plt.gca().set_aspect('equal', adjustable='box')
# plt.plot(centers1_bis[:, 0], centers1_bis[:, 1], 'ro')
# plt.plot(centers2_bis_reg[:, 0], centers2_bis_reg[:, 1], 'bo', fillstyle='none')

# # c) Deformable reg
#
# t0 = time.time()
# centers2_reg = DeformableRegistration(**{'X': centers1, 'Y': centers2}, alpha=2).register()[0]
# print(time.time() - t0)
#
# plt.figure()
# plt.gca().set_aspect('equal', adjustable='box')
# plt.plot(centers1[:, 0], centers1[:, 1], 'ro')
# plt.plot(centers2_reg[:, 0], centers2_reg[:, 1], 'bo', fillstyle='none')
