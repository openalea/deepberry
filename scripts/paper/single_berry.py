import os
import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt
from skimage import io

from scipy.ndimage import uniform_filter1d
from scipy.stats import circmean
from scipy import interpolate
from sklearn.metrics import r2_score

import openalea.maizetrack.phenomenal_display as phm_display

from PIL import Image
from deepberry.src.openalea.deepberry.utils import ellipse_interpolation

index = pd.read_csv('data/grapevine/image_index.csv')

exp = 'DYN2020-05-15'
res = pd.read_csv('X:/phenoarch_cache/cache_{0}/full_results_{0}.csv'.format(exp))
res['hue_scaled'] = ((180 - np.array(res['hue'])) - 100) % 180
res = res[res['plantid'] != 7243]
if exp == 'DYN2020-05-15':
    res = res[~(res['task'] == 2656)]  # remove task too late

# ===== filtering berries ============================================================================================

df_berry = []

for plantid in res['plantid'].unique():

    print(plantid)
    selec = res[(res['plantid'] == plantid) & (res['berryid'] != -1)]

    median_volume = np.median(selec.groupby(['angle', 'berryid'])['volume'].median())
    n_last = max(1, round(len(selec['task'].unique()) / 15))

    for angle in [k * 30 for k in range(12)]:
        s_angle = selec[selec['angle'] == angle]
        for k in s_angle['berryid'].unique():

            # one berry trajectory
            s = s_angle[s_angle['berryid'] == k].sort_values('t')
            v_averaged = uniform_filter1d(s['volume'], size=15, mode='nearest')  # TODO value tested only for dt=8h
            mape = 100 * np.mean(np.abs((v_averaged - s['volume']) / v_averaged))
            if len(s) > 1:
                v_scaled = (v_averaged - np.min(v_averaged)) / (np.max(v_averaged) - np.min(v_averaged))
                # TODO generalise more to different t distributions
                normal_end_volume = np.median(v_scaled[-n_last:]) > 0.3
                normal_end_hue = np.median(s['hue_scaled'][-n_last:]) > 120
                enough_points = len(s) > int(0.90 * len(selec['task'].unique()))
                not_small = np.median(s['volume']) > 0.2 * median_volume
                df_berry.append([plantid, angle, k, mape, not_small, enough_points, normal_end_volume, normal_end_hue])

df_berry = pd.DataFrame(df_berry, columns=['plantid', 'angle', 'id', 'mape', 'not_small',
                                               'enough_points', 'normal_end_volume', 'normal_end_hue'])

# filter berries tracked during >90% of the experiment
df_berry_filter = df_berry[df_berry['enough_points']]
# filter berries with MAPE(V) > 2.5%
df_berry_filter = df_berry_filter[df_berry_filter['mape'] < 2.5]
# filter berries with no abnormalities
df_berry_filter = df_berry_filter[(df_berry_filter['normal_end_volume']) & (df_berry_filter['normal_end_hue'])
                                & (df_berry_filter['not_small'])]

# ===== 1 graph, several single berry curves =========================================================================

plantid = 7238

selec = res[res['plantid'] == plantid]
df_berry_selec = df_berry_filter[df_berry_filter['plantid'] == plantid]


for var in ['volume', 'hue_scaled']:

    plt.figure()
    if var == 'volume':
        plt.title('Berry Volume (V)')
        plt.ylabel(r'$V / V_{max}$')
    elif var == 'hue_scaled':
        plt.title('Berry Hue (H)')
        plt.ylabel(r'$(H - H_0) / (H_n - H_0)$')
        plt.ylim((-0.05, 1.03))
    plt.xlabel('Time (days)')

    xy_list = []
    for i, (_, row) in enumerate(df_berry_selec.iterrows()):
        angle, berryid = row['angle'], row['id']
        s = selec[(selec['angle'] == angle) & (selec['berryid'] == berryid)].sort_values('t')

        x, y = np.array(s['t']), np.array(s[var])

        if var == 'volume':
            y_averaged = uniform_filter1d(y, size=15, mode='nearest')
            y_scaled = y_averaged / np.max(y_averaged)
        elif var == 'hue_scaled':
            y_scaled = (y - np.min(y[:10])) / (np.max(y[-10:]) - np.min(y[:10]))

        xy_list.append([s['task'], x, y_scaled])

        plt.plot(x, y_scaled, '-', color='grey', linewidth=0.6, alpha=0.6)

    xy = pd.DataFrame(np.concatenate(xy_list, axis=1).T, columns=['task', 'x', 'y'])

    for metrics, linestyle in zip(['mean', 'median'], ['-', '--']):
        # mean berry
        xy_gb = xy.groupby('task').agg(metrics).reset_index().sort_values('x')
        plt.plot(xy_gb['x'], xy_gb['y'], 'k' + linestyle)
        # single berry
        gb = selec.groupby('task')[[var, 't']].agg(metrics).reset_index().sort_values('t')
        if var == 'volume':
            mean_scaled = gb[var] / np.max(gb[var])
        elif var == 'hue_scaled':
            mean_scaled = (gb[var] - np.min(gb[var])) / (np.max(gb[var]) - np.min(gb[var]))
        plt.plot(gb['t'], mean_scaled, 'r' + linestyle)


# ====================================================================================================================

for angle in [k * 30 for k in range(12)]:
    s = selec[selec['angle'] == angle]
    plt.figure(angle)
    plt.plot(s['t'], s['hue_scaled'], 'k.')

# ===== area = f(t) ==================================================================================================

df_res = []

#fig, axs = plt.subplots(2, 5)

# K = 10
# df_berry_selec = df_berry.sort_values('n', ascending=False).iloc[:(K ** 2)].sort_values('mape')
# df_berry_selec = df_berry[(df_berry['n'] > 120) & (df_berry['mape'] < 2.5)]
# df_berry_selec = df_berry[df_berry['n'] > 120].sort_values('mape').iloc[(K ** 2):(2 * (K ** 2))]
n0 = len(df_berry_selec)

plt.figure(plantid)
for i, (_, row) in enumerate(df_berry_selec.iterrows()):
    angle, berryid = row['angle'], row['id']

    s = selec[(selec['angle'] == angle) & (selec['berryid'] == berryid)].sort_values('t')
    x, y = np.array(s['t']), np.array(s['volume'])
    y_averaged = uniform_filter1d(y, size=15, mode='nearest')

    # # ===== one graph per berry =====================================================================================
    # mape = 100 * np.mean(np.abs((y_averaged - y) / y_averaged))
    # ax = plt.subplot(K, K, i + 1)
    # # col = phm_display.PALETTE[i]/255.
    # # plt.plot(x, y, '.', color='k')
    # plt.scatter(x, y, marker='.', c=np.array([cv2.cvtColor(np.uint8([[[h, 255, 140]]]),
    #                                                        cv2.COLOR_HSV2RGB)[0][0] / 255. for h in s['hue']]))
    # plt.plot(x, y_averaged, '-', color='r', linewidth=2)
    # # plt.text(0.03, 0.97, 'q1={}\nq2={}'.format(round(q1, 2), round(q2, 2)),
    # #          horizontalalignment='left', verticalalignment='top', transform=ax.transAxes)
    # txt = 'MAPE={}%\na{} id{}'.format(round(mape, 2), int(angle), int(id))
    # # txt = 'MAPE={}%'.format(round(mape, 2))
    # plt.text(0.99, 0.03, txt, horizontalalignment='right', verticalalignment='bottom', transform=ax.transAxes)
    # plt.xlim(0, max(selec['t']) + 2)

    # ===== all berries in the same graph =============================================================================

    # color
    t, h = np.array(s['t']), np.array(s['hue'])
    h_scaled = (h - np.median(h[:10])) / (np.median(h[-10:]) - np.median(h[:10]))
    t_color = {}
    for q in [0.1, 0.5, 0.9]:
        k = next(k for k, val in enumerate(h_scaled) if val > q)
        t_color[q] = t[k - 1] + (t[k] - t[k - 1]) * ((q - h_scaled[k - 1]) / (h_scaled[k] - h_scaled[k - 1]))

    x0 = t_color[0.1]  # t0.1
    y0 = y[np.argmin(np.abs(x0 - x))]  # V(t0.1)

    y_scaled = y_averaged / np.max(y_averaged)
    # ymax = np.max(y_averaged)
    # y_scaled = (y_averaged - y0) / (ymax - y0)
    # y_scaled = y_averaged / np.max(y_averaged)
    # x_scaled = x - x0

    plt.ylabel('V / Vmax')
    plt.xlabel('Time (days)')
    plt.plot(x, y_scaled, 'k-', linewidth=0.7, alpha=0.5)

    # x0 = df_res[(df_res['plantid'] == plantid) & (df_res['berryid'] == berryid)].iloc[0]['t_h0.1']
    # print(x0)
    # plt.plot(x - x0, y_scaled, 'k-', linewidth=0.7, alpha=0.5)

    q = 0.8  # 0.5
    k = next(k for k, val in enumerate(y_scaled) if val > q and k >= np.argmin(y_scaled))
    t_v05 = x[k - 1] + (x[k] - x[k - 1]) * ((q - y_scaled[k - 1]) / (y_scaled[k] - y_scaled[k - 1]))
    df_res.append([plantid, row['angle'], row['id'], t_v05] + list(t_color.values()))


    # plt.figure()
    # plt.plot(x, y_scaled, 'k-', linewidth=1., alpha=1.)
    # plt.plot(t_v05, 0.5, 'ko', markersize=10)
    # plt.xlabel('t')
    # plt.ylabel('(V - Vmin) / (Vmax - Vmin)')
    # plt.title('Volume (V)')
    # plt.figure()
    # plt.plot(t, h_scaled, 'k-', linewidth=1., alpha=1.)
    # plt.plot(t_color[0.1], 0.1, 'o', color='green', markersize=10)
    # plt.plot(t_color[0.5], 0.5, 'o', color='darkred', markersize=10)
    # plt.plot(t_color[0.9], 0.9, 'o', color='darkblue', markersize=10)
    # plt.xlabel('t')
    # plt.ylabel('(H - Hmin) / (Hmax - Hmin)')
    # plt.title('Hue (H)')

    # def cubic_spline(x, y):
    #     x2 = (x - np.min(x)) / (np.max(x) - np.min(x))
    #     y2 = (y - np.min(y)) / (np.max(y) - np.min(y))
    #     tck = interpolate.splrep(x2, y2, s=0.0002*(len(x) - np.sqrt(2 * len(x))))
    #     y_smooth = interpolate.splev(x2, tck)
    #     return (y_smooth * (np.max(y) - np.min(y))) + np.min(y)
    #
    # plt.plot(x, cubic_spline(x, y_scaled), 'r-')

    # =================================================================================================================


df_res = pd.DataFrame(df_res, columns=['plantid', 'angle', 'berryid', 't_v0.5', 't_h0.1', 't_h0.5', 't_h0.9'])


for q, col in zip([0.1, 0.5, 0.9], ['green', 'darkred', 'darkblue']):
    _, ax = plt.subplots()
    _.gca().set_aspect('equal', adjustable='box')
    t1, t2 = np.array(df_res['t_v0.5']), np.array(df_res['t_h{}'.format(q)])
    plt.plot([min(t1), max(t1)], [min(t2), max(t2)], '-', color='grey')
    # non_outlier = np.max((t1, t2), axis=0) / np.min((t1, t2), axis=0) < 2
    # print(len(non_outlier) - np.sum(non_outlier))
    # t1, t2 = t1[non_outlier], t2[non_outlier]
    rmse = np.sqrt(np.sum((t1 - t2) ** 2) / len(t1))
    r2 = r2_score(t1, t2)
    bias = np.mean(t2 - t1)
    a, b = np.polyfit(t1, t2, 1)
    plt.plot([min(t1), max(t1)], a * np.array([min(t1), max(t1)]) + b, '--', color=col, label=f'y = {a:.{2}f}x {b:+.{2}f}')
    print(q, [round(k, 2) for k in [rmse, r2, bias]])
    plt.plot(t1, t2, 'o', color=col, alpha=0.3)
    plt.xlabel('t(V=0.5) (days)')
    plt.ylabel('t(Hue={}) (days)'.format(q))
    plt.text(0.05, 0.95, 'R² = {}'.format(round(r2, 2)),
         fontsize=15, horizontalalignment='left', verticalalignment='top', transform=ax.transAxes)

# a_min = np.min(gb['area'] / np.max(gb['area']))
#
# plt.plot(x, y, 'k.-')
# plt.plot(x, y / np.median(sorted(f(x))[-10:]), 'k.-')
# plt.plot(x, f(x) / np.median(sorted(f(x))[-10:]), 'k-')
#
# plt.plot(x, f(x), 'r-')
#
# ymin = np.median(f(x)[:5])
# ymax = np.median(sorted(f(x))[-5:])
# y2 = (f(x) - ymin) / (ymax - ymin)
# y2 = f(x) / ymin
# plt.plot(x, f(x) / ymin, 'k-')

# ===== color = f(t) ==================================================================================================

#fig, axs = plt.subplots(2, 5)
plt.figure()
K = 10
# df_berry_selec = df_berry.sort_values('n', ascending=False).iloc[(1 * K ** 2):(2 * K ** 2)]
# df_berry_selec = df_berry.sample(K ** 2)
# df_berry_selec = df_berry[df_berry['n'] > 40].sort_values('mape', ascending=True)[::1].iloc[:(K ** 2)]
# df_berry_selec = df_berry.sort_values('n', ascending=False).iloc[:(K ** 2)].sort_values('mape')

for i, (_, row) in enumerate(df_berry_selec.iterrows()):
    s = selec[(selec['angle'] == row['angle']) & (selec['berryid'] == row['id'])].sort_values('t')

    x, y = np.array(s['t']), np.array(s['hue'])
    y_averaged = uniform_filter1d(y, size=5, mode='nearest')

    mape = 100 * np.mean(np.abs((y_averaged - y) / y_averaged))

    # # one graph per berry (COLOR)
    # ax = plt.subplot(K, K, i + 1)
    # plt.xlim((0, 2 + np.max(selec['t'])))
    # plt.ylim((25, 160))
    # plt.plot(x, y, 'k.', linewidth=1)
    # # plt.text(0.99, 0.03, 'a{} id{}'.format(int(row['angle']), int(row['id'])),
    # #          horizontalalignment='right', verticalalignment='bottom', transform=ax.transAxes)

    # all beries
    y = (y - np.median(y[:10])) / (np.median(y[-10:]) - np.median(y[:10]))
    # q = 0.1
    # k = next(k for k, val in enumerate(y) if val > q)
    # x01 = x[k - 1] + (x[k] - x[k - 1]) * ((q - y[k - 1]) / (y[k] - y[k - 1]))
    # x = x - x01
    plt.plot(x, y, 'k-', linewidth=0.7, alpha=0.5)
    plt.xlabel('Time (days)')
    plt.ylabel('Centered hue')

    gb = selec.groupby('task')[['hue', 't']].agg(['median', 'mean']).reset_index().sort_values(('t', 'mean'))
    for metrics, col in zip(['mean', 'median'], ['blue', 'red']):
        xm = np.array(gb['t']['mean'])
        ym = np.array(gb['hue'][metrics])
        ym = (ym - np.median(ym[:10])) / (np.median(ym[-10:]) - np.median(ym[:10]))
        # q = 0.1
        # k = next(k for k, val in enumerate(ym) if val > q)
        # x01 = xm[k - 1] + (xm[k] - xm[k - 1]) * ((q - ym[k - 1]) / (ym[k] - ym[k - 1]))
        # xm = xm - x01
        plt.plot(xm, ym, '-', color=col)



    selec2 = selec.copy()
    selec2['hue'] = ((180 - np.array(selec2['hue'])) - 100) % 180
    selec2['black'] = (selec2['hue'] > 50)
    gb1 = selec2.groupby('task')[['hue', 't']].median().reset_index().sort_values('t')
    plt.plot(gb1['t'], gb1['hue'], 'r-', linewidth=2)
    gb2 = selec2.groupby('task')[['hue', 't']].mean().reset_index().sort_values('t')
    plt.plot(gb2['t'], gb2['hue'], 'b-', linewidth=2)
    gb3 = selec2.groupby('task')[['black', 't']].mean().reset_index().sort_values('t')
    black = (gb3['black'] * (np.max(gb2['hue'] - np.min(gb2['hue'])))) + np.min(gb2['hue'])
    plt.plot(gb3['t'], black, '-', color='lightgreen', linewidth=3)

# ===== visu one berry over time ======================================================================================

# berryid, angle = 4, 270
# berryid, angle = 26, 30
# berryid, angle = 38, 330
berryid, angle = 40, 330
selec_index = index[(index['exp'] == exp) & (index['plantid'] == plantid) & (index['imgangle'] == angle)]

s = selec[(selec['angle'] == angle) & (selec['berryid'] == berryid)].sort_values('timestamp')
# s.loc[s['timestamp'] > 1592571441, ['ell_x', 'ell_y']] -= np.array([-4.2, -437.8])

plt.figure()
y = ((180 - s['hue']) - 100) % 180
plt.scatter(s['t'], y, marker='o', c=np.array([cv2.cvtColor(np.uint8([[[h, 255, 140]]]),
                                                       cv2.COLOR_HSV2RGB)[0][0] / 255. for h in s['hue']]))
# y_averaged = uniform_filter1d(y, size=15, mode='nearest')
# mape = 100 * np.median(np.abs((y_averaged - y) / y_averaged))
# print(mape)
# plt.plot(x, y_averaged, '-', color='r', linewidth=2)
plt.xlabel('Time (days)')
plt.ylabel('Centered Hue (°)')

plt.figure()
x, y = s['t'], s['volume'] / 100000
plt.scatter(x, y, marker='o', c=np.array([cv2.cvtColor(np.uint8([[[h, 255, 140]]]),
                                                       cv2.COLOR_HSV2RGB)[0][0] / 255. for h in s['hue']]))
y_averaged = uniform_filter1d(y, size=15, mode='nearest')
mape = 100 * np.median(np.abs((y_averaged - y) / y_averaged))
print(mape)
plt.plot(x, y_averaged, '-', color='r', linewidth=2)
plt.xlim(0, max(selec['t']) + 2)
plt.xlabel('Time (days)')
plt.ylabel('Volumne (10^5 px³)')

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

imgs_gif = (Image.fromarray(img) for img in imgs)
fps = 6
img = next(imgs_gif)
img.save(fp='data/videos/berry_individual/{}_{}_{}_{}fps.gif'.format(plantid, angle, berryid, fps),
         format='GIF', append_images=imgs_gif, save_all=True, duration=1000 / fps, loop=0)

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















