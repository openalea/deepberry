import os
import matplotlib.pyplot as plt
import numpy as np
import cv2
import pandas as pd

import shutil

from deepberry.src.openalea.deepberry.utils import ellipse_interpolation, get_image_path


PATH = 'data/grapevine/results/'

colors = {'WW': 'dodgerblue', 'WD1': 'orange', 'WD2': 'red', 'normal': 'black', 'odium': 'green'}
symbols = {'A02-PL6': '.:', 'BARESA': '*-', 'PRIMITIV': 'o--'}

index = pd.read_csv('data/grapevine/image_index.csv')

PALETTE = np.array(
    [[255, 0, 0], [0, 255, 0], [0, 0, 255], [255, 255, 0], [204, 121, 167], [0, 158, 115],
     [0, 114, 178], [230, 159, 0], [140, 86, 75], [0, 255, 255], [255, 0, 100], [0, 77, 0], [100, 0, 255],
     [100, 0, 0], [0, 0, 100], [100,100, 0], [0, 100,100], [100, 0, 100], [255, 100, 100]])
PALETTE = np.array(20 * list(PALETTE) + [[0, 0, 0]])

# ===================================================================================================================

df = []
for exp in index['exp'].unique():
    # exp_path = PATH + exp + '/'
    # exp_path = PATH + 'cache_' + exp + '/'
    exp_path = 'X:/phenoarch_cache/cache_{}/'.format(exp)
    s = index[index['exp'] == exp]
    if os.path.isdir(exp_path):
        for fd in os.listdir(exp_path):
            plantid, grapeid = fd.split('_') if '_' in fd else (fd, 0)
            s2 = s[s['plantid'] == int(plantid)]
            genotype, scenario = s2[['genotype', 'scenario']].iloc[0]
            plantid_path = exp_path + fd + '/'
            for f in os.listdir(plantid_path):
                print(plantid_path + f)
                df_f = pd.read_csv(plantid_path + f)
                # TODO : info specific to plant = not in this df ? only in index ?
                task, angle = [int(k) for k in f[:-4].split('_')]
                timestamp = s2[(s2['taskid'] == task) & (s2['imgangle'] == angle)]['timestamp'].iloc[0]
                df_f[['exp', 'plantid', 'grapeid', 'task', 'timestamp', 'angle', 'genotype', 'scenario']] = \
                    exp, int(plantid), int(grapeid), task, timestamp, angle, genotype, scenario
                df.append(df_f)
df = pd.concat(df)

df['area'] = (df['ell_w'] / 2) * (df['ell_h'] / 2) * np.pi
df['volume'] = (4 / 3) * np.pi * ((np.sqrt(df['area'] / np.pi)) ** 3)
df['roundness'] = df['ell_w'] / df['ell_h']  # always h > w
# df['black'] = df['black'].astype(int) * 100

# TODO should be in image_index.py
df = df[~((df['exp'] == 'DYN2020-05-15') & (df['task'] < 2380))]
df = df[~((df['exp'] == 'ARCH2021-05-27') & (df['task'].isin([3797, 3798, 3804, 3810, 3811, 3819, 3827, 3829, 3831, 3843, 368])))]
df = df[~((df['exp'] == 'ARCH2022-05-18') & (df['task'].isin([5742, 5744, 5876, 5877])))]

tmin_dic = {row['exp']: row['timestamp'] for _, row in df.groupby('exp')['timestamp'].min().reset_index().iterrows()}
df['t'] = df.apply(lambda row: (row['timestamp'] - tmin_dic[row['exp']]) / 3600 / 24, axis=1)

df.to_csv(PATH + 'full_results.csv', index=False)

# gb = df.groupby('genotype')['plantid'].nunique().sort_values().reset_index()
# genotypes = list(gb[gb['plantid'] >= 6]['genotype'])

# ===== image + ellipses ============================================================================================

# plantid = 415
# task = 5944
# angle = 240
# s = image_index[(image_index['plantid'] == plantid) & (image_index['task'] == task) & (image_index['angle'] == angle)]

# 'V:/ARCH2022-05-18/5928/dd3debb4-5e13-4430-b83b-dd25c71f0b61.png'

s = df.sample().iloc[0]  # random ellipse

img_path = get_image_path(index, s['plantid'], s['task'], s['angle'], disk='Z:/')
img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
ellipses = pd.read_csv('X:/phenoarch_cache/cache_{}/{}/{}_{}.csv'.format(s['exp'], s['plantid'], s['task'], s['angle']))

plt.figure()
plt.imshow(img)
for _, ell in ellipses.iterrows():
    lsp_x, lsp_y = ellipse_interpolation(x=ell['ell_x'], y=ell['ell_y'], w=ell['ell_w'],
                                         h=ell['ell_h'], a=ell['ell_a'], n_points=100)
    plt.plot(lsp_x, lsp_y, 'red', linewidth=1)
    # if ell['black']:
    #     plt.plot(ell['ell_x'], ell['ell_y'], 'wx')

# ===== save some images ===========================================================================================

s_img = index[(index['exp'] == exp) & (index['taskid'] == 3754)]
for plantid in df21['plantid'].unique():
    selec = df21[df21['plantid'] == plantid]
    for grapeid in selec['grapeid'].unique():
        angle = selec[selec['grapeid'] == grapeid].groupby('angle').size().sort_values().index[-1]  # select best angle
        row_img = s_img[(s_img['plantid'] == plantid) & (s_img['grapeid'] == grapeid) & (s_img['imgangle'] == angle)].iloc[0]
        path1 = 'Z:/{}/{}/{}.png'.format(exp, row_img['taskid'], row_img['imgguid'])
        path2 = 'data/grapevine/rgb/{}/{}_{}.png'.format(exp, plantid, grapeid)
        shutil.copyfile(path1, path2)

# ===== 2021 =======================================================================================================

exp = 'ARCH2021-05-27'
df = pd.read_csv(PATH + 'full_results.csv')
df21 = df[df['exp'] == exp]

var = 'black'
for genotype, col in zip(df21['genotype'].unique(), ['black', 'red', 'blue', 'green', 'orange']):
    selec = df21[df21['genotype'] == genotype]
    for plantid in selec['plantid'].unique():
        selec2 = selec[selec['plantid'] == plantid]
        for grapeid in selec2['grapeid'].unique():
            s = selec2[selec2['grapeid'] == grapeid]
            gb = s.groupby('task')[['t', var]].mean().reset_index().sort_values('t')
            plt.plot(gb['t'], gb[var], '.-', color=col)



var = 'black'
plt.ylabel('Proportion of black berries')
plt.xlabel('Time (days)')
for col, genotype, threshold in zip(['green', 'red'],
                         ['V.vinifera/V6863_H10-PL3', 'V.vinifera/V6860_DO4-PL4'],
                         [2500, 1900]):

    selec = df21[df21['genotype'] == genotype]
    selec = selec[selec['area'] > threshold]

    # plt.figure(genotype)
    # plt.hist(selec['area'], 500)

    for plantid in selec['plantid'].unique():
        selec2 = selec[selec['plantid'] == plantid]
        for grapeid in selec2['grapeid'].unique():
            s = selec2[selec2['grapeid'] == grapeid]
            gb = s.groupby('task')[[var, 't']].mean().reset_index().sort_values('t')
            # plt.plot(gb['t'], gb[var] / np.median(gb[var].iloc[:3]), '.-', color=col)
            plt.plot(gb['t'], gb[var], '.-', color=col)


plt.subplot(2, 1, 2)
var = 'black'
plt.xlabel('Time (days)')
plt.ylabel('Proportion of black berries (%)')
plt.ylim((-2, 102))
for plantid in selec['plantid'].unique():
    s = selec[selec['plantid'] == plantid]
    gb = s.groupby('task')[[var, 't']].mean().reset_index().sort_values('t')
    plt.plot(gb['t'], gb[var], '.-', color=colors[s.iloc[0]['scenario']])


# ===== 2022 =======================================================================================================

df = pd.read_csv(PATH + 'full_results.csv')
df22 = df[df['exp'] == 'ARCH2022-05-18']

genotypes = list(df22.groupby('genotype')['plantid'].nunique().sort_values(ascending=False).reset_index()['genotype'])

# sort all genotypes by a trait
var = 'area'
selec = df22[(df22['t'] > 32) & (df22['t'] < 37)]  # small temporal window to make comparison possible

gb = selec.groupby(['genotype', 'plantid', 'scenario'])[var].mean().reset_index()
gb2 = gb.groupby('genotype')[var].mean().sort_values().reset_index()
plt.figure()
plt.ylabel('Mean berry area (px²)')
plt.xlabel('Genotype')
for i, genotype in enumerate(gb2['genotype']):
    s = gb[gb['genotype'] == genotype]
    plt.plot([i] * len(s), s[var], 'k-')
    for _, row in s.iterrows():
        plt.plot(i, row[var], 'o', color=colors[row['scenario']])

# when photos were taken
genotypes_t = df22.groupby('genotype')['t'].mean().sort_values().index
for i, genotype in enumerate(genotypes_t):
    print(genotype)
    selec = df22[df22['genotype'] == genotype]
    gb = selec.groupby('task')['t'].mean().reset_index()
    plt.plot(gb['t'], [i] * len(gb), 'k.-')

# which genotypes are black
gb = df22.groupby(['genotype', 'plantid', 'task'])['black'].mean().reset_index()
gb = gb.groupby('genotype')['black'].max().reset_index()
genotypes_black = list(gb[gb['black'] > 30]['genotype'])
genotypes_black = list(df22[df22['genotype'].isin(genotypes_black)].groupby('genotype')['plantid'].nunique().sort_values(ascending=False).reset_index()['genotype'])

# ==== G comparison =====

var = 'black'
k = 0
for genotype in genotypes_black[:10]:
    selec = df22[(df22['genotype'] == genotype)]  # & (df22['scenario'] == 'WW')]
    for plantid in selec['plantid'].unique():
        s = selec[selec['plantid'] == plantid]
        gb = s.groupby('task')[[var, 't']].mean().sort_values('t')
        plt.plot(gb['t'], gb[var], '.-', color=PALETTE[k]/255.)
    k += 1

# ===== one genotype =====

for i_g, genotype in enumerate([g for g in genotypes if g not in genotypes_black][:9]):
# for genotype in genotypes_black[:5]:

    selec = df22[df22['genotype'] == genotype]

    # remove task if few data (= image pb)
    gb = selec.groupby(['task', 'plantid']).size().reset_index().groupby('task').mean().reset_index()
    tasks_to_keep = gb['task'][[k > 0.5 * np.median(gb[0]) for k in gb[0]]]
    print('nb of tasks to remove:', len(selec['task'].unique()) - len(tasks_to_keep))
    selec = selec[selec['task'].isin(tasks_to_keep)]

    # plt.figure(genotype)
    # plt.suptitle(genotype)
    plt.subplot(3, 3, i_g + 1)
    var = 'area'
    plt.ylabel('Mean berry area (px²)')
    for plantid in selec['plantid'].unique():
        s = selec[selec['plantid'] == plantid]
        gb = s.groupby('task')[[var, 't']].mean().reset_index().sort_values('t')
        plt.plot(gb['t'], gb[var], '.-', color=colors[s.iloc[0]['scenario']])

    # plt.subplot(3, 1, 2)
    # var = 'roundness'
    # plt.ylabel('Mean berry roundness')
    # for plantid in selec['plantid'].unique():
    #     s = selec[selec['plantid'] == plantid]
    #     gb = s.groupby('task')[[var, 't']].mean().reset_index().sort_values('t')
    #     plt.plot(gb['t'], gb[var], '.-', color=colors[s.iloc[0]['scenario']])

    plt.subplot(2, 2, 3)
    var = 'black'
    plt.xlabel('Time (days)')
    plt.ylabel('Proportion of black berries (%)')
    plt.ylim((-2, 102))
    for plantid in selec['plantid'].unique():
        s = selec[selec['plantid'] == plantid]
        gb = s.groupby('task')[[var, 't']].mean().reset_index().sort_values('t')
        plt.plot(gb['t'], gb[var], '.-', color=colors[s.iloc[0]['scenario']])

# ===== high frequency 2020 =========================================================================================

plantid = 7243

selec = df[(df['exp'] == 'DYN2020-05-15') & (df['plantid'] == plantid)]
selec = selec.sort_values('timestamp', ascending=False)

from pylab import *
from scipy.optimize import curve_fit
from scipy.stats import norm
def gauss(x, mu, sigma, A):
    return A * exp(-(x - mu) ** 2 / 2 / sigma ** 2)
def bimodal(x, mu1, sigma1, A1, mu2, sigma2, A2):
    return gauss(x, mu1, sigma1, A1) + gauss(x, mu2, sigma2, A2)

ar = selec['area']
y, x = np.histogram(ar, 500, weights=np.ones_like(ar)/float(len(ar)))
x = (x[1:] + x[:-1]) / 2  # for len(x)==len(y)
y *= len(ar)
plt.plot(x, y, 'k-')
expected = (np.mean(ar) / 7, np.std(ar) / 7, len(ar) / 30,
            np.mean(ar), np.std(ar) / 2, len(ar) / 30)
params, cov = curve_fit(bimodal, x, y, expected)
sigma = sqrt(diag(cov))
threshold = norm.ppf(.999) * params[1] + params[0]
plt.plot(x, bimodal(x, *params), color='red', lw=3)
plt.plot([threshold]*2, [0, max(bimodal(x, *params))], color='green', lw=3)
selec['small'] = selec['area'] < threshold

# berry histogram
plt.figure()
plt.xlabel('berry area (px²)')
plt.xlim((0, 8000))
plt.hist(selec['area'], bins=500)

# small vs large berry, for each task separately
selec['area_quantile'] = None
selec.index = np.arange(len(selec))
for task in np.unique(selec['task']):
    s = selec[selec['task'] == task]
    n = 20  # how many subdivisions
    quantiles = np.quantile(s['area'], [k * (1 / n) for k in range(1, n)])
    quantiles = [float('-inf')] + list(quantiles) + [float('+inf')]
    for k in range(n):
        s2 = s[(quantiles[k] <= s['area']) & (s['area'] < quantiles[k + 1])]
        selec.at[s2.index, 'area_quantile'] = k

#
s = selec[selec['task'] == 2527]
print(s.iloc[0]['t'])
gb = s.groupby('area_quantile')[['area', 'black']].mean()
hist = plt.hist(s['area'], 50)
plt.ylim((0, max(hist[0])))
for area in gb['area']:
    plt.plot([area, area], [0, 2 * max(hist[0])], 'r-')
plt.xlabel('berry area (px²)')
plt.figure()
plt.plot(gb['area'], gb['black'], 'k.-')
plt.xlabel('Mean berry area in the quantile group (px²)')
plt.ylabel('Proportion of black berries (%)')

plt.subplot(3, 1, 1)
var = 'area'
plt.ylabel('Mean berry area (px²)')
gb = selec.groupby('task')[[var, 't']].mean().reset_index().sort_values('t')
plt.plot(gb['t'], gb[var], 'k.-')
gb = selec[selec['small']].groupby('task')[[var, 't']].mean().reset_index().sort_values('t')
plt.plot(gb['t'], gb[var], '-', color='green')
gb = selec[~selec['small']].groupby('task')[[var, 't']].mean().reset_index().sort_values('t')
plt.plot(gb['t'], gb[var], '-', color='orange')
plt.subplot(3, 1, 2)
var = 'roundness'
plt.ylabel('Mean berry roundness')
gb = selec.groupby('task')[[var, 't']].mean().reset_index().sort_values('t')
plt.plot(gb['t'], gb[var], 'k.-')
gb = selec[selec['small']].groupby('task')[[var, 't']].mean().reset_index().sort_values('t')
plt.plot(gb['t'], gb[var], '-', color='green')
gb = selec[~selec['small']].groupby('task')[[var, 't']].mean().reset_index().sort_values('t')
plt.plot(gb['t'], gb[var], '-', color='orange')
plt.subplot(3, 1, 3)
var = 'black'
plt.xlabel('Time (days)')
plt.ylabel('Proportion of black berries (%)')
gb = selec.groupby('task')[[var, 't']].mean().reset_index().sort_values('t')
plt.plot(gb['t'], gb[var], 'k.-')
gb = selec[selec['small']].groupby('task')[[var, 't']].mean().reset_index().sort_values('t')
plt.plot(gb['t'], gb[var], '-', color='green')
gb = selec[~selec['small']].groupby('task')[[var, 't']].mean().reset_index().sort_values('t')
plt.plot(gb['t'], gb[var], '-', color='orange')

tasks = np.unique(selec.sort_values('timestamp')['task'])
tasks = tasks[[0, int(len(tasks) / 3.5), -1]]
for k in range(3):
    plt.subplot(3, 1, k + 1)
    plt.xlim((0, 7000))
    s = selec[selec['task'] == tasks[k]]
    plt.hist(s['area'], bins=50)
    print('t = {} days, mean = {} px²'.format(round(s.iloc[0]['t'], 1), round(np.mean(s['area']), 1)))
plt.xlabel('Berry area (px²)')

var = 'roundness'
secondary_var = 'plantid'
plt.figure()
for q in sorted(selec[secondary_var].unique()):
    s = selec[selec[secondary_var] == q]
    gb = s.groupby('task')[[var, 't']].mean().reset_index().sort_values('t')
    plt.plot(gb['t'], gb[var], '-', label=q)
gb = selec.groupby('task')[[var, 't']].mean().reset_index().sort_values('t')
plt.plot(gb['t'], gb[var], '-r.')
plt.legend()


var = 'roundness'
gb = selec.groupby('angle')[[var]].mean().reset_index().sort_values('angle')
plt.plot(gb['angle'], gb[var])

angles = [0, 60, 120, 180, 240, 300]
gb = selec.groupby(['angle', 'timestamp']).size().reset_index().sort_values('timestamp')
for angle in angles:
    s = gb[gb['angle'] == angle]
    plt.plot(s['timestamp'], s[0], label=angle)
plt.legend()

# ===== 2020; all plants ============================================================================================

selec = df[df['exp'] == 'DYN2020-05-15']

plt.figure()
#plt.ylabel('Mean berry area (px²)')
#plt.ylabel('Mean berry roundness')
plt.ylabel('Proportion of black berries (%)')
plt.xlabel('Time (days)')

for plantid in selec['plantid'].unique():
    s = df[df['plantid'] == plantid]
    s = s.sort_values('timestamp', ascending=False)
    s = s[s['t'] > 2.3]  # remove first tasks

    s = s[s['area'] > 1400]

    area0 = np.mean(s[s['t'] < 6])['area']
    #selec['area'] /= area0

    # 50% black timing
    gb = s.groupby('task')[['black', 't']].mean().reset_index().sort_values('t')
    timing = gb[gb['black'] > 50].sort_values('t').iloc[0]['t']
    timing = 0

    var = 'area'
    gb = s.groupby('task')[[var, 't']].mean().reset_index().sort_values('t')
    plt.plot(gb['t'] - timing, gb[var], '-')

# histograms
threshold = 1400
for k, plantid in enumerate(selec['plantid'].unique()):
    plt.subplot(3, 3, k + 1)
    s = selec[selec['plantid'] == plantid]
    hist = plt.hist(s['area'], 500)
    plt.xlim((0, 9000))
    plt.plot([threshold] * 2, [0, np.max(hist[0])], 'r-')


# ===== GxE =========================================================================================================

import seaborn as sns

plt.figure()
selec = df[df['genotype'].isin(genotypes[:10])]
sns.boxplot(x='genotype', y='area', hue='scenario', data=selec, palette=colors)

for genotype in genotypes[:5]:
    s = df[df['genotype'] == genotype]
    print(genotype)
    print(s.groupby(['plantid', 'scenario'])['area'].agg(['size', 'median']).sort_values('scenario'))

# ===== color = f(t) all plants ======================================================================================

for scenario in df['scenario'].unique():
    s = df[df['scenario'] == scenario]
    gb = s.groupby('task')['black'].agg(['size', 'sum']).reset_index()
    gb = gb[gb['size'] > 300]
    plt.plot(gb['task'], gb['sum'] / gb['size'], '.-', color=colors[s.iloc[0]['scenario']])

# ===== area = f(t) all plants ======================================================================================

for scenario in df['scenario'].unique():
    s = df[df['scenario'] == scenario]
    gb = s.groupby('task')['area'].agg(['size', 'median']).reset_index()
    gb = gb[gb['size'] > 300]
    plt.plot(gb['median'], '.-', color=colors[s.iloc[0]['scenario']])

# ===== color ========================================================================================================

for genotype in genotypes:
    plt.figure()
    plt.title(genotype)
    plt.ylim((-0.02, 1.02))
    selec = df[df['genotype'] == genotype]
    for plantid in selec['plantid'].unique():
        s = selec[selec['plantid'] == plantid]
        gb = s.groupby('task')['black'].agg(['size', 'sum']).reset_index()
        plt.plot(gb['task'], gb['sum'] / gb['size'], '.-', color=colors[s.iloc[0]['scenario']])

# ===== plant : angle effect =========================================================================================

for plantid in df['plantid'].unique():

    plt.figure(plantid)
    s = df[df['plantid'] == plantid]
    plt.boxplot([list(s[s['angle'] == a]['area']) for a in s['angle'].unique()], labels=s['angle'].unique())

# ===== trait histogram, per plant ===================================================================================

for genotype in df['genotype'].unique():
    selec = df[df['genotype'] == genotype]

    plt.figure()
    plt.title(genotype)
    for plantid in selec['plantid'].unique():
        s = selec[selec['plantid'] == plantid]
        plt.hist(s['area'], bins=100, histtype='step', fill=False, stacked=True, color=colors[s.iloc[0]['scenario']])
        plt.xlabel('berry area')

# ===== trait histogram = f(t), one plant ============================================================================

for plantid in df['plantid'].unique():

    plt.figure()
    plt.title('plantid' + str(plantid))
    selec = df[df['plantid'] == plantid]
    for task in selec['task'].unique():
        s = selec[selec['task'] == task]
        plt.hist(s['area'], bins=30, histtype='step', fill=False, stacked=True, label=task)
    plt.legend()

# ===== trait median = f(t), per plant ===============================================================================

selec = df

df_trait = []
for plantid in selec['plantid'].unique():
    s = selec[selec['plantid'] == plantid]
    for task in s['task'].unique():
        # TODO: sometimes different values
        t = image_index[(image_index['plantid'] == int(plantid)) & (image_index['taskid'] == task)]['timestamp'].iloc[0]
        s2 = s[s['task'] == task]
        area = s2['area']
        black = 100 * (np.sum(s2['black']) / len(s2))
        #roundness = np.max(s2[['ell_w', 'ell_h']], axis=1) / np.min(s2[['ell_w', 'ell_h']], axis=1)
        df_trait.append([s.iloc[0]['genotype'], s.iloc[0]['scenario'], plantid, task, t,
                         len(area), np.median(area), np.std(area), np.median(roundness), black])
df_trait = pd.DataFrame(df_trait, columns=['genotype', 'scenario', 'plantid', 'task', 'timestamp', 'n', 'area_med', 'area_std', 'roundness_med', 'black'])

# AREA
for genotype in genotypes:
    selec = df_trait[df_trait['genotype'] == genotype]
    plt.figure()
    plt.title(genotype)
    plt.xlabel('Timestamp')
    plt.ylabel('Median berry area (px²)')
    for plantid in selec['plantid'].unique():
        s = selec[selec['plantid'] == plantid]
        x, y = list(s['timestamp']), list(s['area_med'])
        plt.plot(x, y, '.-', color=colors[s.iloc[0]['scenario']])
        plt.text(x[-1], y[-1], plantid)

# COLOR
for genotype in genotypes:
    selec = df_trait[df_trait['genotype'] == genotype]
    plt.figure()
    plt.ylim((-2, 102))
    plt.title(genotype)
    plt.xlabel('Task')
    plt.ylabel('Black berries (%)')
    for plantid in selec['plantid'].unique():
        s = selec[selec['plantid'] == plantid]
        x, y = list(s['task']), list(s['black'])
        plt.plot(x, y, '.-', color=colors[s.iloc[0]['scenario']])
        plt.text(x[-1], y[-1], plantid)

plt.figure()
plt.title('ratio')
for plantid in df_trait['plantid'].unique():
    s = df_trait[df_trait['plantid'] == plantid]
    plt.plot(s['task'], s['ratio_med'], symbols[s.iloc[0]['genotype']], color=colors[s.iloc[0]['scenario']])

# ===== exploration ==============================================================

genotype = 'PRIMITIV'
selec = df[(df['genotype'] == genotype)]
gb_tasks = selec.groupby('task')['plantid'].nunique().reset_index()
task = np.max(gb_tasks[gb_tasks['plantid'] == max(gb_tasks['plantid'])]['task'])
selec = selec[selec['task'] == task]

for plantid in selec['plantid'].unique():
    s = selec[selec['plantid'] == plantid]
    best_angle = s.groupby('angle').size().sort_values().reset_index()['angle'].iloc[-1]

    median_area = np.median(s[s['angle'] == angle]['area'])

    img_path = image_index[(image_index['plantid'] == plantid) &
                      (image_index['task'] == task) &
                      (image_index['angle'] == best_angle)].iloc[0]['image_path']
    img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
    plt.figure()
    plt.title('{}_{}'.format(plantid, round(median_area, 1)))
    plt.imshow(img)


for plantid in selec['plantid'].unique():
    s = selec[selec['plantid'] == plantid]
    #plt.figure(plantid)
    plt.hist(s['area'], bins=30, histtype='step', fill=False, stacked=True, color=colors[s.iloc[0]['scenario']])

for scenario in colors.keys():
    plt.plot(0, 0, 'wo', color=colors[scenario], label=scenario)
plt.legend(prop={'size': 50}, markerscale=5)












