import os
import matplotlib.pyplot as plt
import numpy as np
import cv2
import pandas as pd

from grapevine.utils import ellipse_interpolation, get_image_path


PATH = 'data/grapevine/results/'

colors = {'WW': 'dodgerblue', 'WD1': 'orange', 'WD2': 'red'}
symbols = {'A02-PL6': '.:', 'BARESA': '*-', 'PRIMITIV': 'o--'}

index = pd.read_csv('data/grapevine/image_index.csv')

# ===================================================================================================================

df = []
for exp in index['exp'].unique():
    s = index[index['exp'] == exp]
    if os.path.isdir(PATH + exp):
        for plantid in os.listdir(PATH + exp):
            s2 = s[s['plantid'] == int(plantid)]
            genotype, scenario = s2[['genotype', 'scenario']].iloc[0]
            for f in os.listdir(PATH + exp + '/' + plantid):
                df_f = pd.read_csv(PATH + exp + '/' + plantid + '/' + f)
                # TODO : info specific to plant = not in this df ? only in index ?
                task, angle = [int(k) for k in f[:-4].split('_')]
                timestamp = s2[(s2['taskid'] == task) & (s2['imgangle'] == angle)]['timestamp'].iloc[0]
                df_f[['exp', 'plantid', 'task', 'timestamp', 'angle', 'genotype', 'scenario']] = \
                    exp, int(plantid), task, timestamp, angle, genotype, scenario
                df.append(df_f)
df = pd.concat(df)

df['area'] = (df['ell_w'] / 2) * (df['ell_h'] / 2) * np.pi
df['ratio'] = np.max(df[['ell_w', 'ell_h']], axis=1) / np.min(df[['ell_w', 'ell_h']], axis=1)

# gb = df.groupby('genotype')['plantid'].nunique().sort_values().reset_index()
# genotypes = list(gb[gb['plantid'] >= 6]['genotype'])

# ===== image + ellipses ============================================================================================

# plantid = 415
# task = 5944
# angle = 240
# s = image_index[(image_index['plantid'] == plantid) & (image_index['task'] == task) & (image_index['angle'] == angle)]

# 'V:/ARCH2022-05-18/5928/dd3debb4-5e13-4430-b83b-dd25c71f0b61.png'

s = df.sample().iloc[0]  # random ellipse

img_path = get_image_path(index, s['plantid'], s['task'], s['angle'])
img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
ellipses = pd.read_csv(PATH + '{}/{}/{}_{}.csv'.format(s['exp'], s['plantid'], s['task'], s['angle']))

plt.figure()
plt.imshow(img)
for _, ell in ellipses.iterrows():
    lsp_x, lsp_y = ellipse_interpolation(x=ell['ell_x'], y=ell['ell_y'], w=ell['ell_w'],
                                         h=ell['ell_h'], a=ell['ell_a'], n_points=100)
    plt.plot(lsp_x, lsp_y, 'red', linewidth=0.6)
    if ell['black']:
        plt.plot(ell['ell_x'], ell['ell_y'], 'wx')


# ===== high frequency 2020 =========================================================================================

selec = df[(df['exp'] == 'DYN2020-05-15') & (df['plantid'] == 7243)]
selec = selec.sort_values('timestamp', ascending=False)
selec = selec[~(selec['task'].isin([2377, 2379]))]

gb = selec.groupby('task')[['area', 'timestamp']].mean().reset_index().sort_values('timestamp')
plt.figure()
plt.title('Area')
plt.plot(gb['timestamp'], gb['area'], '-k.')

gb = selec.groupby('task')[['ratio', 'timestamp']].mean().reset_index().sort_values('timestamp')
plt.figure()
plt.title('Ratio')
plt.plot(gb['timestamp'], gb['ratio'], '-k.')

gb = selec.groupby('task')[['timestamp', 'black']].agg(['sum', 'size'])
gb['t_mean'] = gb['timestamp']['sum'] / gb['timestamp']['size']
gb['black_mean'] = gb['black']['sum'] / gb['black']['size']
gb = gb.sort_values('t_mean')
plt.figure()
plt.title('% Black')
plt.plot(gb['t_mean'], gb['black_mean'], '-k.')

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
        ratio = np.max(s2[['ell_w', 'ell_h']], axis=1) / np.min(s2[['ell_w', 'ell_h']], axis=1)
        df_trait.append([s.iloc[0]['genotype'], s.iloc[0]['scenario'], plantid, task, t,
                         len(area), np.median(area), np.std(area), np.median(ratio), black])
df_trait = pd.DataFrame(df_trait, columns=['genotype', 'scenario', 'plantid', 'task', 'timestamp', 'n', 'area_med', 'area_std', 'ratio_med', 'black'])

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












