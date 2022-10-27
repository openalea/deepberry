import os
import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt
from skimage import io

from scipy.ndimage import uniform_filter1d
from scipy.stats import circmean

import openalea.maizetrack.phenomenal_display as phm_display

from PIL import Image
from deepberry.src.openalea.deepberry.utils import ellipse_interpolation

index = pd.read_csv('data/grapevine/image_index.csv')

fd = 'data/grapevine/temporal/results/'

res = []
for file in [f for f in os.listdir(fd) if f[-4:] == '.csv']:
    print(file)
    res.append(pd.read_csv(fd + file))
res = pd.concat(res)

# ===================================================================================================================

exp = 'DYN2020-05-15'
res_exp = res[res['exp'] == exp]

plantid = 7240
selec0 = res_exp[res_exp['plantid'] == plantid]

# last task from 7240 / 2020 is too late !
selec0 = selec0[selec0['task'] != 2656]

# angle = 270
# selec = selec0[selec0['angle'] == angle]
selec = selec0

# selec['area'] = (4/3) * np.pi * ((np.sqrt(selec['area'] / np.pi)) ** 3) # conversion area --> volume

# ==== filter best berries ===========================================================================================

df_berry = []
for angle in [k * 30 for k in range(12)]:
    s_angle = selec[selec['angle'] == angle]
    for k in [k for k in s_angle['berryid'].unique() if k != -1]:
        s = s_angle[s_angle['berryid'] == k].sort_values('t')
        a, b = s['area'][1:], s['area'][:-1]
        q1 = np.median(np.max((a, b), axis=0) / np.min((a, b), axis=0))
        q2 = np.sum(np.abs(np.diff(s['area']))) / (np.max(s['area']) - np.min(s['area']))
        y_averaged = uniform_filter1d(s['area'], size=30, mode='nearest')  # TODO what if dt not constant ?
        mape = 100 * np.mean(np.abs((y_averaged - s['area']) / y_averaged))
        df_berry.append([angle, k, q1, q2, mape, len(s)])  # TODO median or mean ?
df_berry = pd.DataFrame(df_berry, columns=['angle', 'id', 'q1', 'q2', 'mape', 'n'])

# ===== area = f(t) ==================================================================================================

#fig, axs = plt.subplots(2, 5)
plt.figure()
K = 6
# ids = list(df_berry[df_berry['n'] > 40].sort_values('dif')['id'].iloc[:25])
df_berry_selec = df_berry.sort_values('n', ascending=False).iloc[:(K ** 2)].sort_values('q1')
# df_berry_selec = df_berry[df_berry['n'] > 100].sort_values('mape', ascending=True)[::1].iloc[:(K ** 2)]
for i, (_, row) in enumerate(df_berry_selec.iterrows()):
    angle, id, q1, q2 = row['angle'], row['id'], (row['q1'] - 1) * 100, row['q2']
    print(i, id)
    s = selec[(selec['angle'] == angle) & (selec['berryid'] == id)].sort_values('t')
    # x, y = np.array(s['t']), np.array(s['area'])
    x, y = np.array(s['t']), np.array(s['hue'])
    # f = savgol_smoothing_function(x, y, dw=3, polyorder=2, repet=3, monotony=False)
    y_averaged = uniform_filter1d(y, size=30, mode='nearest')

    mape = 100 * np.mean(np.abs((y_averaged - y) / y_averaged))

    # # one graph per berry
    # ax = plt.subplot(K, K, i + 1)
    # # col = phm_display.PALETTE[i]/255.
    # # plt.plot(x, y, '.', color='k')
    # plt.scatter(x, y, marker='.', c=np.array([cv2.cvtColor(np.uint8([[[h, 255, 150]]]),
    #                                                        cv2.COLOR_HSV2RGB)[0][0] / 255. for h in s['hue']]))
    # plt.plot(x, y_averaged, '-', color='r', linewidth=2)
    # plt.text(0.03, 0.97, 'q1={}\nq2={}'.format(round(q1, 2), round(q2, 2)),
    #          horizontalalignment='left', verticalalignment='top', transform=ax.transAxes)
    # plt.text(0.99, 0.03, 'MAPE={}%\na{} id{}'.format(round(mape, 2), int(angle), int(id)),
    #          horizontalalignment='right', verticalalignment='bottom', transform=ax.transAxes)
    # plt.xlim(0, max(selec['t']) + 2)

    # all berries in the same graph
    i_black = next(i for i, val in enumerate(uniform_filter1d(np.array(s['hue']), size=30, mode='nearest'))
                   if val >= 60)
    y_scaled = y_averaged / np.max(y_averaged)
    plt.plot(x[:(i_black + 1)], y_scaled[:(i_black + 1)], '-', color='limegreen', linewidth=0.4)
    plt.plot(x[i_black:], y_scaled[i_black:], '-', color='darkblue', linewidth=0.4)
    # plt.scatter(x, y_scaled, marker='.',
    #             c=np.array(['darkblue' if c == 100 else 'limegreen' for c in s['black']]))
    plt.plot(x[i_black], y_scaled[i_black], color='orange', marker='.', markersize=10)
    gb = selec.groupby('task')[['area', 't']].mean().reset_index().sort_values('t')
    plt.plot(gb['t'], gb['area'] / np.max(gb['area']), 'r-')
    # a_min = np.min(gb['area'] / np.max(gb['area']))

    # plt.plot(x, y, 'k.-')
    # plt.plot(x, y / np.median(sorted(f(x))[-10:]), 'k.-')
    # plt.plot(x, f(x) / np.median(sorted(f(x))[-10:]), 'k-')

    plt.xlabel('time (days)')
    plt.ylabel('Berry_area / max_area (smoothed)')

    # plt.plot(x, f(x), 'r-')

    # ymin = np.median(f(x)[:5])
    # ymax = np.median(sorted(f(x))[-5:])
    # y2 = (f(x) - ymin) / (ymax - ymin)
    # y2 = f(x) / ymin
    # plt.plot(x, f(x) / ymin, 'k-')

# ===== color = f(t) ==================================================================================================

#fig, axs = plt.subplots(2, 5)
plt.figure()
K = 3
df_berry_selec = df_berry.sort_values('n', ascending=False).iloc[(1 * K ** 2):(2 * K ** 2)]
# df_berry_selec = df_berry.sample(K ** 2)
# df_berry_selec = df_berry[df_berry['n'] > 100].sort_values('mape', ascending=True)[::1].iloc[:(K ** 2)]
for i, (_, row) in enumerate(df_berry_selec.iterrows()):
    s = selec[(selec['angle'] == row['angle']) & (selec['berryid'] == row['id'])].sort_values('t')
    x, y = np.array(s['t']), np.array(s['hue'])
    y_averaged = uniform_filter1d(y, size=5, mode='nearest')

    mape = 100 * np.mean(np.abs((y_averaged - y) / y_averaged))

    # one graph per berry (COLOR)
    ax = plt.subplot(K, K, i + 1)
    plt.xlim((0, np.max(selec['t'])))
    plt.ylim((25, 150))
    # y_averaged = uniform_filter1d(y, size=5, mode='nearest')
    # plt.plot(x, y_averaged, '-', color='r', linewidth=2)
    # y = (y - np.median(y[:10])) / (np.median(y[-10:]) - np.median(y[:10]))
    # x = x - x[next(k for k, val in enumerate(y) if val > 0.5)]
    plt.plot(x, y, 'k.', linewidth=1)
    plt.text(0.99, 0.03, 'a{} id{}'.format(int(row['angle']), int(row['id'])),
             horizontalalignment='right', verticalalignment='bottom', transform=ax.transAxes)

# ===== visu one berry over time ======================================================================================

berryid, angle = 4, 270
selec_index = index[(index['exp'] == exp) & (index['plantid'] == plantid) & (index['imgangle'] == angle)]

s = selec[(selec['angle'] == angle) & (selec['berryid'] == berryid)].sort_values('timestamp')
s.loc[s['timestamp'] > 1592571441, ['ell_x', 'ell_y']] -= np.array([-4.2, -437.8])

plt.plot(np.arange(len(s)), s['hue'], 'k.-')


px_list = []
imgs = []
for k, (_, row) in enumerate(s.iterrows()):
# for k, (_, row) in enumerate(s.iloc[[43, 61, 67, 78, 100]].iterrows()):

    print(k)

    # plt.figure('{}_{}'.format(row['task'], round(row['hue'], 2)))

    # plt.gca().set_aspect('equal', adjustable='box')
    # plt.ylim((np.min(selec['ell_y']), np.max(selec['ell_y'])))
    task = row['task']
    path = 'Z:/{}/{}/{}.png'.format(exp, task, selec_index[selec_index['taskid'] == task].iloc[0]['imgguid'])
    img = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)

    x, y, w, h, a = row['ell_x'], row['ell_y'], row['ell_w'], row['ell_h'], row['ell_a']
    ell = ellipse_interpolation(x, y, w, h, a, n_points=100)
    # plt.plot(ell[0], ell[1], '-', color=('red' if row['berryid'] == berryid else 'blue'))

    mask = cv2.ellipse(np.float32(img[:, :, 0] * 0), (round(x), round(y)),
                       (round((w * 0.85) / 2), round((h * 0.85) / 2)), a, 0., 360, (1), -1)
    px = img[mask == 1]  # r, g, b
    px_hsv = cv2.cvtColor(np.array([px]), cv2.COLOR_RGB2HSV)[0]
    px_lab = cv2.cvtColor(np.array([px]).astype(np.uint8), cv2.COLOR_RGB2LAB)[0].astype(np.float)
    px_lab[:, 0] = px_lab[:, 0] / 255 * 100
    px_lab[:, 1] = px_lab[:, 1] - 128
    px_lab[:, 2] = px_lab[:, 2] - 128

    hue = circmean(px_hsv[:, 0], low=0, high=180)
    cols = [hue] + list(np.mean(np.concatenate((px_hsv[:, 1:], px_lab), axis=1), axis=0))
    px_list.append(cols)

    # hue2 = hue.astype(np.float)  # hue was in uint8 i.e. can't go outside [0, 255]

    img = cv2.ellipse(img, (round(x), round(y)), (round(w / 2), round(h / 2)), a, 0., 360, [255, 0, 0], 1)
    img = img[(round(y) - 45):(round(y) + 45), (round(x) - 45):(round(x) + 45)]
    img = cv2.resize(img, (360, 360))
    img2 = img.copy()
    img2[325:, 250:] *= 0
    img2 = cv2.putText(img2, 't={}'.format(round(row['t'], 1)),
                       (260, 350), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 1, cv2.LINE_AA)

    # io.imsave('{}.png'.format(k+5), img2)
    # plt.figure()
    # plt.imshow(img2)

    imgs.append(img2)

    # plt.xlim((x - 45, x + 45))
    # plt.ylim((y - 45, y + 45))
    # plt.plot(ell[0], ell[1], '-', color='red')


px = np.array(px_list)
plt.plot(s['t'], ((180 - px[:, 0]) - 100) % 180, 'k.', label='((180 - h) + 100) % 180')


CIRG = (180-H) / (L*+C)
# CIRG2 : CIRG but H (hue) was calculated considering the hue values included between 360° and 270° as negative

L, a, b = [px[:, k] for k in [3, 4, 5]]
h = np.degrees(np.arctan(b / a))
C = np.sqrt((a ** 2) + (b ** 2))
CIRWG = np.radians(h) / (L * b) * 100

plt.plot(s['t'], px[:, 0], 'k.-')
plt.plot(s['t'], h, 'b.-')

plt.plot(s['t'], L, 'r.', label='L*')
plt.plot(s['t'], a, 'g.', label='a*')
plt.plot(s['t'], b, 'b.', label='b*')
plt.legend()

plt.xlabel('Time (days)')
plt.ylabel('Mean pixel value')
plt.legend()

plt.plot(np.arange(len(px)), px[:, 5] - px[:, 2], '.-', color='black', label='hue')
plt.plot(np.arange(len(px)), px[:, 3], '.-', color='black', label='hue')

# ===== color = f(t) =================================================================================================

# selection of one grape
selec_index = index[(index['exp'] == exp) & (index['plantid'] == plantid)]

df_berry_selec = df_berry[df_berry['n'] > 100].sort_values('mape', ascending=True)[:100]


pixels = {}
# angle loop
for angle in [k * 30 for k in range(12)]:

    pixels[angle] = {}
    s_index = selec_index[selec_index['imgangle'] == angle]
    s_berries = df_berry_selec[df_berry_selec['angle'] == angle]

    # task loop
    for k, (_, row_index) in enumerate(s_index.iterrows()):

        pixels[angle][row_index['taskid']] = {}

        img = cv2.cvtColor(cv2.imread('Z:/{}/{}/{}.png'.format(exp, row_index['taskid'], row_index['imgguid'])),
                           cv2.COLOR_BGR2RGB)

        s_ell = selec[(selec['angle'] == angle) & (selec['task'] == row_index['taskid']) &
                      (selec['berryid'].isin(s_berries['id']))]

        s_ell.loc[s_ell['timestamp'] > 1592571441, ['ell_x', 'ell_y']] -= np.array([-4.2, -437.8])

        # berry loop
        for _, row_ell in s_ell.iterrows():

            print(angle, k, int(row_ell['berryid']))
            x, y, w, h, a = row_ell[['ell_x', 'ell_y', 'ell_w', 'ell_h', 'ell_a']]
            mask = cv2.ellipse(np.float32(img[:, :, 0] * 0), (round(x), round(y)), (round(w / 2), round(h / 2)),
                               a, 0., 360, (1), -1)
            px = img[mask == 1]  # r, g, b
            px_hsv = cv2.cvtColor(np.array([px]), cv2.COLOR_RGB2HSV)[0]
            pixels[angle][row_index['taskid']][row_ell['berryid']] = np.concatenate((px, px_hsv), axis=1)

df_pixels = []
for angle in pixels.keys():
    for task in pixels[angle].keys():
        t = np.mean(selec[selec['task'] == task]['t'])
        print(task, t)
        for berryid in pixels[angle][task].keys():
            px = pixels[angle][task][berryid]
            df_pixels.append([angle, t, berryid] + list(np.mean(px, axis=0)))
df_pixels = pd.DataFrame(df_pixels, columns=['angle', 't', 'id', 'r', 'g', 'b', 'h', 's', 'v'])

plt.figure()
for angle in df_pixels['angle'].unique():
    s0 = df_pixels[df_pixels['angle'] == angle]
    for berryid in s0['id'].unique()[::1]:
        s = s0[s0['id'] == berryid].sort_values('t')
        plt.plot(s['t'], s['h'], 'b.-')


for i, (_, row) in enumerate(df_berry_selec.iterrows()):

    # selection of a berry + angle
    angle, id, q1, q2 = row['angle'], row['id'], (row['q1'] - 1) * 100, row['q2']
    s = selec[(selec['angle'] == angle) & (selec['berryid'] == id)]

    df_color = []

    for _, row_ell in s.iterrows():

        # task = row_ell['task']
        # path = 'Z:/{}/{}/{}.png'.format(exp, task, s_index[s_index['taskid'] == task].iloc[0]['imgguid'])
        # img = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)
        #
        # x, y, w, h, a = row_ell[['ell_x', 'ell_y', 'ell_w', 'ell_h', 'ell_a']]
        #
        # mask = cv2.ellipse(np.float32(img[:, :, 0] * 0), (round(x), round(y)), (round(w / 2), round(h / 2)),
        #                    a, 0., 360, (1), -1)
        # pixels = img[mask == 1]  # r, g, b
        # pixels_hsv = cv2.cvtColor(np.array([pixels]), cv2.COLOR_RGB2HSV)[0]

        df_color.append([row_ell['t']] + list(np.mean(pixels, axis=0)) + list(np.mean(pixels_hsv, axis=0)))

    df_color = pd.DataFrame(df_color, columns=['t', 'r', 'g', 'b', 'h', 's', 'v'])

    for i, col in enumerate(['r', 'g', 'b', 'orange', 'black', 'grey']):
        plt.plot(df_color['t'], df_color.iloc[:, i + 1], '-')















