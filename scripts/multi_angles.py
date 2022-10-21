import os
import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt

from pycpd import AffineRegistration, RigidRegistration, DeformableRegistration

from PIL import Image

from deepberry.src.openalea.deepberry.utils import ellipse_interpolation

index = pd.read_csv('data/grapevine/image_index.csv')

fd = 'data/grapevine/temporal/results/'

res = []
for file in os.listdir(fd):
    print(file)
    res.append(pd.read_csv(fd + file))
res = pd.concat(res)

# ===================================================================================================================

exp = 'DYN2020-05-15'
res_exp = res[res['exp'] == exp]

plantid = res_exp['plantid'].unique()[0]
selec = res_exp[res_exp['plantid'] == plantid]

tasks = list(selec.groupby('task')['t'].mean().sort_values().index)
task = tasks[30]

s = selec[selec['task'] == task]

a1, a2 = 0, 30
s1, s2 = s[s['angle'] == a1], s[s['angle'] == a2]

# plt.figure()
# plt.gca().set_aspect('equal', adjustable='box')
# for _, row in s1.iterrows():
#     x, y, w, h, a = row['ell_x'], row['ell_y'], row['ell_w'], row['ell_h'], row['ell_a']
#     ell = ellipse_interpolation(x, y, w, h, a, n_points=100)
#     plt.plot(ell[0], ell[1], 'r-')
# for _, row in s2.iterrows():
#     x, y, w, h, a = row['ell_x'], row['ell_y'], row['ell_w'], row['ell_h'], row['ell_a']
#     ell = ellipse_interpolation(x, y, w, h, a, n_points=100)
#     plt.plot(ell[0], ell[1], 'b-')

centers1 = np.array(s1[['ell_x', 'ell_y']])
centers2 = np.array(s2[['ell_x', 'ell_y']])

q, m = np.max(np.max(centers1, axis=0) - np.min(centers1, axis=0)), np.mean(centers1)
centers1, centers2 = (centers1 - m) / q * 2, (centers2 - m) / q * 2

a_mean, a_std = np.mean(s['area']), np.std(s['area'])
centers1 = np.concatenate((centers1, np.array([(s1['area'] - a_mean) / a_std]).T), axis=1)
centers2 = np.concatenate((centers2, np.array([(s2['area'] - a_mean) / a_std]).T), axis=1)

# import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# ax.plot(centers1[:, 0], centers1[:, 1], centers1[:, 2], 'ro')
# ax.plot(centers2[:, 0], centers2[:, 1], centers2[:, 2], 'bo', fillstyle='none')

plt.figure()
plt.gca().set_aspect('equal', adjustable='box')
plt.plot(centers1[:, 0], centers1[:, 1], 'ro')
plt.plot(centers2[:, 0], centers2[:, 1], 'bo', fillstyle='none')

# b) Affine reg

centers2_reg = AffineRegistration(**{'X': centers1, 'Y': centers2}).register()[0]

plt.figure()
plt.gca().set_aspect('equal', adjustable='box')
plt.plot(centers1[:, 0], centers1[:, 1], 'ro')
plt.plot(centers2_reg[:, 0], centers2_reg[:, 1], 'bo', fillstyle='none')

# c) Deformable reg

centers2_reg = DeformableRegistration(**{'X': centers1, 'Y': centers2}).register()[0]

plt.figure()
plt.gca().set_aspect('equal', adjustable='box')
plt.plot(centers1[:, 0], centers1[:, 1], 'ro')
plt.plot(centers2_reg[:, 0], centers2_reg[:, 1], 'bo', fillstyle='none')













