"""
TODO Remove border pixels (different color) !
TODO Same simple threshold for mean & individual berry: Hue=80
TODO Use mean, not median
"""

import pandas as pd
import numpy as np
import os
import cv2
import matplotlib.pyplot as plt

import shutil

from skimage import io

from deepberry.src.openalea.deepberry.utils import ellipse_interpolation


def is_black(green, hue):
    if green < 35:
        return True
    elif green > 80:
        return False
    else:
        if hue > 80:
            return True
        elif hue > 41:
            return False
        else:
            # /!\ green in [35, 80] and hue in [0, 41] is ambiguous!
            if hue > 25:
                return False
            else:
                return True


index = pd.read_csv('data/grapevine/image_index.csv')

df = pd.read_csv('data/grapevine/results/full_results.csv')

# 2020
df20 = df[(df['exp'] == 'DYN2020-05-15') & ~(df['task'] < 2380)]
t_camera = 1592571441  # np.mean(df20.groupby('task')['timestamp'].mean().sort_values()[[2566, 2559]].values)
df20.loc[df20['timestamp'] > t_camera, ['ell_x', 'ell_y']] += np.array([-4.2, -437.8])

# 2021
df21 = df[df['exp'] == 'ARCH2021-05-27']
df21 = df21[~(df21['task'].isin([3797, 3798, 3804, 3810, 3811, 3819, 3827, 3829, 3831, 3843, 3686, 3687, 3685]))]
df21 = df21[df21['genotype'].isin(['V.vinifera/V6863_H10-PL3', 'V.vinifera/V6860_DO4-PL4'])]

# 2022
df22 = df[df['exp'] == 'ARCH2022-05-18']
df22 = df22[~df22['task'].isin([5742, 5744, 5876, 5877])]

df = pd.concat((df20, df21, df22))

# =====================================================================================================================

black_genotypes = np.unique([g.split('_')[0] for g in os.listdir('data/grapevine/color/grape_classif/black')])

# ===== visualise (t0, tn) for each genotype

df_pixel = []

exp = 'ARCH2022-05-18'
df_exp = df[df['exp'] == exp]
for genotype in df_exp['genotype'].unique():

    selec = df_exp[df_exp['genotype'] == genotype]

    # select plantid with the biggest color difference between t0 and tn
    dcol_max, best_plantid = float('-inf'), None
    for plantid in selec['plantid'].unique():
        s = selec[selec['plantid'] == plantid]
        col = s.groupby('task')['black'].mean()
        dcol = np.max(col) - np.min(col)
        if dcol > dcol_max:
            dcol_max, best_plantid = dcol, plantid
    plantid = best_plantid
    print(genotype, dcol_max)

    selec1 = selec[selec['plantid'] == plantid]

    for task, t in zip(selec1['task'].unique()[[0, -1]], ['t0', 'tn']):

        # select best angle
        selec2 = selec1[selec1['task'] == task]
        angle = selec2.groupby('angle').size().sort_values().index[-1]

        s_ell = selec2[selec2['angle'] == angle]
        row_img = index[(index['exp'] == exp) & (index['plantid'] == plantid) & (index['taskid'] == task) &
                          (index['imgangle'] == angle)].iloc[0]

        img_path = 'Z:/{}/{}/{}.png'.format(row_img['exp'], task, row_img['imgguid'])
        img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)

        # plt.figure('{}_{}_{}'.format(genotype, task, dcol_max))
        # plt.imshow(img)
        # plt.plot(s_ell[s_ell['black'] == 0]['ell_x'], s_ell[s_ell['black'] == 0]['ell_y'], 'go')
        # plt.plot(s_ell[s_ell['black'] != 0]['ell_x'], s_ell[s_ell['black'] != 0]['ell_y'], 'ro')

        # berry loop
        for _, row_ell in s_ell.iterrows():
            x, y, w, h, a = row_ell[['ell_x', 'ell_y', 'ell_w', 'ell_h', 'ell_a']]
            w, h = w * 0.85, h * 0.85  # remove border pixels
            mask = cv2.ellipse(np.float32(img[:, :, 0] * 0), (round(x), round(y)), (round(w / 2), round(h / 2)),
                               a, 0., 360, (1), -1)
            px = img[mask == 1]  # r, g, b
            px_hsv = cv2.cvtColor(np.array([px]), cv2.COLOR_RGB2HSV)[0]
            px6 = np.concatenate((px, px_hsv), axis=1)
            res_px = list(np.mean(px6, axis=0)) + list(np.median(px6, axis=0))

            df_pixel.append([genotype, plantid, task, t, angle, genotype in black_genotypes] + res_px)

df_pixel = pd.DataFrame(df_pixel, columns=['genotype', 'plantid', 'task', 't', 'angle', 'genotype_col'] +
                                          [k + '_mean' for k in 'rgbhsv'] + [k + '_median' for k in 'rgbhsv'])
df_pixel.to_csv('data/grapevine/color/grape_classif/color.csv', index=False)

        # fd = 'black' if dcol_max > 50 else 'green'
        # io.imsave('data/grapevine/color/grape_classif/{}/{}_{}_{}.png'.format(fd, genotype, t, round(dcol_max, 1)), img)

# ===== find optimal threshold (green t0, green tn, black t0, black tn)

df = pd.read_csv('data/grapevine/color/grape_classif/color.csv')
df['obs'] = [(1 if row['t'] == 'tn' and row['genotype_col'] else 0) for _, row in df.iterrows()]
df.groupby(['genotype_col', 't'])[['g_mean', 'h_mean']].median()

df['pred'] = [(1 if row['h_mean'] > 80 else 0) for _, row in df.iterrows()]

boxplots = {}
for k, (t, black) in enumerate([['t0', False], ['tn', False], ['t0', True], ['tn', True]]):
    selec = df[(df['t'] == t) & (df['genotype_col'] == black)]
    boxplot = []
    for genotype in selec['genotype'].unique():
        s = selec[selec['genotype'] == genotype]
        m = np.median(s['h_mean'])
        boxplot.append(m)
        if m > 80:
            print(genotype, m)
    boxplots['{}{}'.format(t, ' ' if black else '')] = boxplot
    # plt.plot([k] * len(boxplot), boxplot, 'o')

fig, ax = plt.subplots()
ax.boxplot(boxplots.values(), patch_artist=True)
ax.set_xticklabels(boxplots.keys())
plt.ylabel('median hue')

# df['pred'] = [(1 if is_black(row['g_median'], row['h_median']) else 0) for _, row in df.iterrows()]
#
# tp = len(df[(df['obs'] == 1) & (df['pred'] == 1)]) / len(df) * 100
# tn = len(df[(df['obs'] == 0) & (df['pred'] == 0)]) / len(df) * 100
# fp = len(df[(df['obs'] == 0) & (df['pred'] == 1)]) / len(df) * 100
# fn = len(df[(df['obs'] == 1) & (df['pred'] == 0)]) / len(df) * 100
# print(f'TP: {tp:.1f}%  TN: {tn:.1f}%  FP: {fp:.2f}%  FN: {fn:.2f}%')




# ===== filter grapes that exhibit a full green --> full black transition

pixels = {}

gb = df.groupby(['exp', 'plantid', 'grapeid']).size().reset_index()
for _, row in gb[gb['exp'] == 'ARCH2022-05-18'].iterrows():
    exp, plantid, grapeid = np.array(row[['exp', 'plantid', 'grapeid']])
    selec = df[(df['exp'] == exp) & (df['plantid'] == plantid) & (df['grapeid'] == grapeid)]
    gb2 = selec.groupby('task')[['timestamp', 'black']].mean().reset_index().sort_values('timestamp')
    col_min, col_max = np.array(gb2.iloc[[0, -1]]['black'])
    s_index = index[(index['exp'] == exp) & (index['plantid'] == plantid) & (index['grapeid'] == grapeid)]

    print(plantid)
    #if col_min < 10 and col_max > 90:
    if True:

        print((col_min, col_max))
        pixels[plantid] = {}
        for task in selec.sort_values('timestamp')['task'].unique()[[-1]]:

            angle = np.random.choice([k * 30 for k in range(12)])
            s_ell = selec[(selec['task'] == task) & (selec['angle'] == angle)]
            row_img = s_index[(s_index['taskid'] == task) & (s_index['imgangle'] == angle)].iloc[0]
            img_path = 'Z:/{}/{}/{}.png'.format(row_img['exp'], task, row_img['imgguid'])
            img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)

            plt.figure()
            plt.imshow(img)
            plt.plot(s_ell[s_ell['black'] == 0]['ell_x'], s_ell[s_ell['black'] == 0]['ell_y'], 'go')
            plt.plot(s_ell[s_ell['black'] != 0]['ell_x'], s_ell[s_ell['black'] != 0]['ell_y'], 'ro')

            # for _, row_ell in s_ell.iterrows():
            #     x, y, w, h, a = row_ell[['ell_x', 'ell_y', 'ell_w', 'ell_h', 'ell_a']]
            #     print(w, h)
            #     ell = ellipse_interpolation(x, y, w, h, a, n_points=100)
            #     plt.plot(ell[0], ell[1], 'r-', linewidth=0.7)
            #     x, y, w, h, a = row_ell[['ell_x', 'ell_y', 'ell_w', 'ell_h', 'ell_a']]
            #     ell = ellipse_interpolation(x, y, w * 0.8, h * 0.8, a, n_points=100)
            #     plt.plot(ell[0], ell[1], 'g-', linewidth=0.7)

            # berry loop
            ellipses = []
            for _, row_ell in s_ell.iterrows():
                x, y, w, h, a = row_ell[['ell_x', 'ell_y', 'ell_w', 'ell_h', 'ell_a']]
                w, h = w * 0.85, h * 0.85  # remove border pixels
                mask = cv2.ellipse(np.float32(img[:, :, 0] * 0), (round(x), round(y)), (round(w / 2), round(h / 2)),
                                   a, 0., 360, (1), -1)
                px = img[mask == 1]  # r, g, b
                px_hsv = cv2.cvtColor(np.array([px]), cv2.COLOR_RGB2HSV)[0]
                ellipses.append(np.concatenate((px, px_hsv), axis=1))

            t = np.mean(s_ell['timestamp'])
            pixels[plantid][t] = {'ellipses': ellipses, 'colors': s_ell['black']}


plt.figure()
for k, plantid in enumerate(list(pixels.keys())[::1]):
    # plt.figure()
    # plt.ylim((0, 200))
    timestamps = np.array([k for k in pixels[plantid].keys() if not np.isnan(k)])
    for t, col in zip(timestamps[[0, -1]], ['g', 'b']):
        ellipses = pixels[plantid][t]['ellipses']
        # colors = pixels[plantid][t]['colors']
        # ellipses = [e for (e, c) in zip(ellipses, colors) if c == 100]
        print(len(ellipses))

        # plt.plot([t] * len(ellipses), [np.mean(e[:, 3]) for e in ellipses], 'ko', alpha=0.2)
        plt.plot([k] * len(ellipses), [np.mean(e[:, 1]) for e in ellipses], 'o', color=col, alpha=0.2)
        plt.plot(k, np.median([np.mean(e[:, 1]) for e in ellipses]), 'r*')

# ===== extract various (t0, tn) images to see the color difference

exp = 'ARCH2022-05-18'  # 'DYN2020-05-15', 'ARCH2021-05-27', 'ARCH2022-05-18'

index_exp = index[index['exp'] == exp]
df_exp = df[df['exp'] == exp]

for plantid in np.random.choice(index_exp['plantid'].unique(), 300):
    s_ell = df_exp[df_exp['plantid'] == plantid]
    if len(s_ell) > 0:
        angle = s_ell.groupby('angle').size().sort_values().index[-1]
        s_index = index_exp[(index_exp['plantid'] == plantid) & (index_exp['imgangle'] == angle)].sort_values('timestamp')
        s_ell = s_ell[s_ell['angle'] == angle]

        for _, row in s_index.iloc[[0, -1]].iterrows():
            path1 = 'Z:/{}/{}/{}.png'.format(exp, row['taskid'], row['imgguid'])
            path2 = 'data/grapevine/color/t0_tn/{}_{}_{}_{}.png'.format(exp, int(plantid), int(row['grapeid']),
                                                                     row['daydate'])
            shutil.copyfile(path1, path2)














