import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt

from scipy.stats import circmean, linregress

from scipy.ndimage import uniform_filter1d, median_filter

DIR_OUTPUT = 'data/grapevine/paper/'

ML_PER_PX3 = (0.158 ** 3) * 0.001  # px3 --> mm3 --> cm3 = mL

exp = 'DYN2020-05-15'
res = pd.read_csv('X:/phenoarch_cache/cache_{0}/full_results_temporal_{0}.csv'.format(exp))
res['t'] = (res['timestamp'] - min(res['timestamp'])) / 3600 / 24

# remove plantid used during training
res = res[res['plantid'] != 7243]

# remove task too late
if exp == 'DYN2020-05-15':
    res = res[~(res['task'] == 2656)]

# generated with berry_filtering.py
df_berry_classif = pd.read_csv('data/grapevine/berry_filter.csv')

df_berry = df_berry_classif[(df_berry_classif['enough_points']) &
                            (df_berry_classif['normal_end_volume']) &
                            (df_berry_classif['normal_end_hue']) &
                            (df_berry_classif['not_small'])]

# ===== one berry as example ==========================================================================================

plantid, angle, berryid = 7240, 270, 7
# plantid, angle, berryid, mape = df_berry.sample()[['plantid', 'angle', 'id', 'mape']].iloc[0]
# print(k, plantid, angle, berryid, mape)

s = res[(res['plantid'] == plantid) & (res['angle'] == angle) & (res['berryid'] == berryid)].sort_values('t')
hue_cv2 = (180 - 100 - np.array(s['hue_scaled'])) % 180
colors = np.array([cv2.cvtColor(np.uint8([[[h, 255, 140]]]), cv2.COLOR_HSV2RGB)[0][0] / 255. for h in hue_cv2])

plt.figure(figsize=(10, 10), dpi=100)
plt.subplot(2, 1, 1)
# fig, ax = plt.subplots(figsize=(10, 10), dpi=100)
plt.gca().tick_params(axis='both', which='major', direction='in', labelsize=20)
plt.gca().yaxis.set_ticks_position('both')
plt.gca().xaxis.set_ticks_position('both')
x, y = np.array(s['t']), np.array(s['volume']) * ML_PER_PX3
plt.scatter(x, y, marker='o', c=colors, s=50)
# plt.title('Single berry size dynamics', fontsize=35)
plt.ylabel('Volume (mL)', fontsize=20)
# plt.xlabel('Time (days)', fontsize=10)
y_averaged = median_filter(y, size=25, mode='reflect')

plt.plot(x, y_averaged, '-', color='r', linewidth=3)
mape = 100 * np.mean(np.abs((y_averaged - y) / y_averaged))
txt = 'MAPE={}%'.format(round(mape, 2), int(angle), int(berryid))
# plt.text(0.99, 0.03, txt, horizontalalignment='right', verticalalignment='bottom', transform=ax.transAxes,
#          size=30, color='red')

# plt.savefig(DIR_OUTPUT + 'single_volume', bbox_inches='tight')
# plt.close()

plt.subplot(2, 1, 2)
# fig, ax = plt.subplots(figsize=(10, 10), dpi=100)
plt.gca().tick_params(axis='both', which='major', labelsize=20)
plt.gca().yaxis.set_ticks_position('both')
plt.gca().xaxis.set_ticks_position('both')
plt.gca().tick_params(axis='both', which='major', direction='in', labelsize=20)  # axis number size
x, y = np.array(s['t']), np.array(s['hue_scaled'])
plt.fill_between(x, y - np.array(s['hue_scaled_std']) / 2, y + np.array(s['hue_scaled_std']) / 2,
                 color='grey', alpha=0.3)
# for xi, yi, yi_std in zip(x, y, np.array(s['hue_scaled_std'])):
#     plt.plot([xi, xi], [yi - yi_std / 2, yi + yi_std / 2], 'k-', alpha=0.3)
plt.scatter(x, y, marker='o', c=colors, s=50)
# plt.title('Single berry color dynamics', fontsize=35)
plt.ylabel('Hue (deg)', fontsize=20)
plt.xlabel('Time (days)', fontsize=20)

plt.savefig(DIR_OUTPUT + 'one_berry', bbox_inches='tight')
plt.close()

# ===== many individual graphs (supplementary material) ==================================================)============

df_berry_selec = df_berry[df_berry['plantid'] == 7232]
df_berry_selec = df_berry_selec[df_berry_selec['mape'] <= np.quantile(df_berry_selec['mape'], 0.9)]

K = 7
df_berry_sample = df_berry_selec.sample(K ** 2)
# df_berry_selec = df_berry.sort_values('n', ascending=False).iloc[:(K ** 2)].sort_values('mape')
# df_berry_selec = df_berry[(df_berry['n'] > 120) & (df_berry['mape'] < 2.5)]
# df_berry_selec = df_berry[df_berry['n'] > 120].sort_values('mape').iloc[(K ** 2):(2 * (K ** 2))]

for var in ['volume', 'hue_scaled']:

    plt.figure(var)

    for i, (_, row) in enumerate(df_berry_sample.sort_values('mape').iterrows()):
        plantid, angle, berryid = row['plantid'], row['angle'], row['id']

        s = res[(res['plantid'] == plantid) & (res['angle'] == angle) & (res['berryid'] == berryid)].sort_values('t')

        if var == 'volume':
            x, y = np.array(s['t']), np.array(s['volume']) * ML_PER_PX3
        elif var == 'hue_scaled':
            x, y = np.array(s['t']), np.array(s['hue_scaled'])

        ax = plt.subplot(K, K, i + 1)
        ax.yaxis.get_major_locator().set_params(integer=True)  # y axis = int

        hue_cv2 = (180 - 100 - np.array(s['hue_scaled'])) % 180
        plt.scatter(x, y, marker='.', c=np.array([cv2.cvtColor(np.uint8([[[h, 255, 140]]]),
                                                               cv2.COLOR_HSV2RGB)[0][0] / 255. for h in hue_cv2]))

        if var == 'volume':
            y_averaged = median_filter(y, size=25, mode='reflect')
            plt.plot(x, y_averaged, '-', color='r', linewidth=2)
            ymin, ymax = min(y_averaged), max(y_averaged)
            plt.ylim((ymin - 0.15 * (ymax - ymin), ymax + 0.15 * (ymax - ymin)))
        elif var == 'hue_scaled':
            plt.ylim((25, 155))
            plt.fill_between(x, y - np.array(s['hue_scaled_std']) / 2, y + np.array(s['hue_scaled_std']) / 2,
                             color='grey', alpha=0.3)

        plt.gca().yaxis.set_ticks_position('both')
        plt.gca().xaxis.set_ticks_position('both')
        plt.gca().tick_params(axis='both', which='major', direction='in', labelsize=10)  # axis number size

plt.savefig(DIR_OUTPUT + 'multi_{}.png'.format(var))  # full screen !

# ===== multiple berries in the same graph ============================================================================

plantid = 7232

berries = df_berry_classif[df_berry_classif['plantid'] == plantid]
# filter berries tracked during >90% of the experiment
berries = berries[berries['enough_points']]
# filter berries with no abnormalities
berries = berries[(berries['normal_end_volume']) & (berries['normal_end_hue']) & (berries['not_small'])]
# filter 10% berries with highest V_MAPE
berries = berries[berries['mape'] < 3.8]

# berries = berries[berries['angle'].isin([60, 150, 240, 330])]
berries = berries[berries['angle'].isin([30, 150, 270])]

# berries = berries.sample(20, replace=False)

selec = res[res['plantid'] == plantid]
df_single = []
for _, row in berries.iterrows():
    s = selec[(selec['angle'] == row['angle']) & (selec['berryid'] == row['id'])].sort_values('t')
    s['angle_id'] = f'{row["angle"]}_{row["id"]}'
    s = s.rename(columns={'hue_scaled': 'h', 'volume': 'v'})
    s['v'] = s['v'] * ML_PER_PX3
    s['v_smooth'] = median_filter(np.array(s['v']), size=25, mode='reflect')
    s['mape'] = row['mape']
    s = s[['v', 'v_smooth', 'h', 'angle_id', 'task', 't', 'mape']]
    df_single.append(s)
df_single = pd.concat(df_single)
# df_single.to_csv('data/grapevine/paper/single_berries.csv', index=False)


# ===== final plot ==================================================================================================


def y_to_x(x, y, y_target):
    """ x must be monotonous """
    k_next = next(k for k, val in enumerate(y) if val > y_target and k >= np.argmin(y))
    x1, x2 = x[k_next - 1], x[k_next]  # x1 < x2
    y1, y2 = y[k_next - 1], y[k_next]
    x_target = x1 + (x2 - x1) * ((y_target - y1) / (y2 - y1))
    return x_target


trajs = [df_single[df_single['angle_id'] == i].sort_values('t') for i in df_single['angle_id'].unique()]
n = len(trajs)

# _____ volume _____

qa, qb = 0.15, 0.85

# single berry
t, v = [np.array(s['t']) for s in trajs], [np.array(s['v_smooth']) for s in trajs]
v_0 = np.array([np.mean(vi[:1]) for vi in v])
v_max = np.array([max(vi) for vi in v])
vr = [(vi - vi_0) / vi_0 for vi, vi_0 in zip(v, v_0)]
vs = [(vi - vi_0) / (vi_max - vi_0) for vi, vi_0, vi_max in zip(v, v_0, v_max)]
t_va = np.array([y_to_x(ti, vsi, qa) for ti, vsi in zip(t, vs)])
t_vb = np.array([y_to_x(ti, vsi, qb) for ti, vsi in zip(t, vs)])
t_v05 = np.array([y_to_x(ti, vsi, 0.5) for ti, vsi in zip(t, vs)])
vra = np.array([np.interp(ti_va, ti, vri) for ti_va, ti, vri in zip(t_va, t, vr)])
vrb = np.array([np.interp(ti_vb, ti, vri) for ti_vb, ti, vri in zip(t_vb, t, vr)])

# mean berry
t_mean, v_mean = np.array(df_single.groupby('task').mean().sort_values('t')[['t', 'v_smooth']]).T
# v_mean = uniform_filter1d(v_mean, size=3, mode='nearest')
v_mean_0 = v_mean[0]
v_mean_max = max(v_mean)
vr_mean = (v_mean - v_mean_0) / v_mean_0
vs_mean = (v_mean - v_mean_0) / (v_mean_max - v_mean_0)
t_mean_va = y_to_x(t_mean, vs_mean, qa)
t_mean_vb = y_to_x(t_mean, vs_mean, qb)
# t_mean_v05 = y_to_x(t_mean, vs_mean, 0.5)
t_mean_vra = np.interp(t_mean_va, t_mean, vr_mean)
t_mean_vrb = np.interp(t_mean_vb, t_mean, vr_mean)

# _____ plot volume main _____

fig = plt.figure(figsize=(16, 8), dpi=100)
plt.gca().tick_params(axis='both', which='major', labelsize=20)
for k in range(n):
    plt.plot(t[k], vr[k], '-', color='grey', linewidth=1., alpha=0.7)
plt.plot(t_mean, vr_mean, 'r--', linewidth=2., alpha=0.9)
plt.xlabel('Time (days)', fontsize=20)
plt.ylabel(r'$V_{r} = (V - V_0) / V_0$', fontsize=20)
plt.gca().tick_params(axis='both', which='major', labelsize=15)
plt.gca().yaxis.set_ticks_position('both')
plt.gca().xaxis.set_ticks_position('both')
plt.gca().tick_params(axis='both', which='major', direction='in')  # axis number size
plt.gca().annotate(f'n = {n}', xy=(0.98, 0.02), color='blue',
                   xycoords='axes fraction', fontsize=25, horizontalalignment='right', verticalalignment='bottom')

plt.savefig('data/grapevine/paper/final_a.png')

# _____ plot volume scatters _____

x = v_max / v_0
fig = plt.figure(figsize=(16, 6), dpi=100)

def f_subplot(x, y):
    plt.plot(x, y, 'o', color='grey', alpha=0.8)
    plt.xlabel(r'$V_{max}$' + ' / ' + r'$V_0$', fontsize=20)
    plt.gca().tick_params(axis='both', which='major', labelsize=15)
    plt.gca().yaxis.set_ticks_position('both')
    plt.gca().xaxis.set_ticks_position('both')
    plt.gca().tick_params(axis='both', which='major', direction='in')  # axis number size
    r2 = linregress(x, y)[2] ** 2
    plt.gca().annotate(f'R² = {r2:.3f}', xy=(0.98, 0.02), color='blue',
                       xycoords='axes fraction', fontsize=15, horizontalalignment='right', verticalalignment='bottom')


plt.subplot(1, 3, 1)
y = t_va  # timing
f_subplot(x, y)
plt.plot(v_mean_max / v_mean_0, t_mean_va, 'r*', markersize=15)
# plt.ylabel(r'$t(V_s = 15$%$)$', fontsize=15)
plt.ylabel('Ripening timing (days)', fontsize=20)

plt.subplot(1, 3, 2)
y = t_vb - t_va  # duration
f_subplot(x, y)
plt.plot(v_mean_max / v_mean_0, t_mean_vb - t_mean_va, 'r*', markersize=15)
# plt.ylabel(r'D = $t(V_s = 85$%$) - t(V_s = 15$%$)$', fontsize=15)
plt.ylabel('Ripening duration (days)', fontsize=20)

plt.subplot(1, 3, 3)
y = (vrb - vra) / (t_vb - t_va)  # speed
f_subplot(x, y)
plt.plot(v_mean_max / v_mean_0, (t_mean_vrb - t_mean_vra) / (t_mean_vb - t_mean_va), 'r*', markersize=15)
# plt.ylabel(r'$(Vr(t(Vs=85$%$)) - Vr(t(Vs=15$%$))) / D$', fontsize=15)
plt.ylabel('Ripening relative speed ($days^{-1}$)', fontsize=20)
a, b = np.polyfit(x, y, 1)
plt.plot([min(x), max(x)], a * np.array([min(x), max(x)]) + b, '--', color='b', label=f'y = {a:.{2}f}x {b:+.{2}f}')
plt.legend(prop={'size': 15})

fig.tight_layout()

plt.savefig('data/grapevine/paper/final_b.png')


# _____ plot volume hists _____


fig = plt.figure(figsize=(6, 6), dpi=100)

plt.subplot(1, 2, 1)
plt.ylim((min(v_0) * 0.95, max(v_max) * 1.05))
plt.hist(v_0, 15, orientation='horizontal', color='grey')
plt.gca().tick_params(axis='both', which='major', labelsize=25)
plt.gca().axes.xaxis.set_ticklabels([])
plt.gca().axes.get_xaxis().set_visible(False)
ax.tick_params(axis='both', which='major', labelsize=30)
plt.title(r'$V_0$' + ' (mL)', fontsize=25)


plt.subplot(1, 2, 2)
plt.ylim((min(v_0) * 0.95, max(v_max) * 1.05))
plt.hist(v_max, 25, orientation='horizontal', color='grey')
plt.gca().tick_params(axis='both', which='major', labelsize=25)
plt.gca().axes.xaxis.set_ticklabels([])
plt.gca().axes.get_xaxis().set_visible(False)
ax.tick_params(axis='both', which='major', labelsize=30)
plt.title(r'$V_{max}$' + ' (mL)', fontsize=25)

fig.tight_layout()

plt.savefig('data/grapevine/paper/final_c.png')


# _____ color ______

# single berry
t, h = [np.array(s['t']) for s in trajs], [np.array(s['h']) for s in trajs]
h_0 = np.array([np.median(hi[:25]) for hi in h])
h_n = np.array([np.median(hi[-25:]) for hi in h])
hs = [(hi - hi_0) / (hi_n - hi_0) for hi, hi_0, hi_n in zip(h, h_0, h_n)]
hs = [median_filter(hsi, size=3, mode='reflect') for hsi in hs]
t_h005 = np.array([y_to_x(ti, hsi, 0.05) for ti, hsi in zip(t, hs)])
t_h05 = np.array([y_to_x(ti, hsi, 0.5) for ti, hsi in zip(t, hs)])

# _____ plot color main _____

fig = plt.figure(figsize=(13, 6), dpi=100)
plt.gca().tick_params(axis='both', which='major', labelsize=25)
for k in range(n):
    plt.plot(t[k], hs[k], '-', color='grey', linewidth=1., alpha=0.7)
plt.ylim((-0.07, 1.07))
plt.plot([min(t_h005), max(t_h005)], [0.05, 0.05], '--', color='forestgreen')
plt.xlabel('Time (days)', fontsize=25)
plt.ylabel(r'$H_{s} = (H - H_0) / (H_n - H_0)$', fontsize=25)
plt.gca().tick_params(axis='both', which='major', labelsize=20)
plt.gca().yaxis.set_ticks_position('both')
plt.gca().xaxis.set_ticks_position('both')
plt.gca().tick_params(axis='both', which='major', direction='in')  # axis number size
plt.gca().annotate(f'n = {n}', xy=(0.98, 0.02), color='blue',
                   xycoords='axes fraction', fontsize=25, horizontalalignment='right', verticalalignment='bottom')
fig.tight_layout()

plt.savefig('data/grapevine/paper/final_d.png')

# _____ scatter _____

fig = plt.figure(figsize=(6, 6), dpi=100)
x, y = t_v05, t_h005
plt.plot(x, y, 'o', color='grey')
r2 = linregress(x, y)[2] ** 2
a, b = np.polyfit(x, y, 1)
plt.plot([min(x), max(x)], a * np.array([min(x), max(x)]) + b, '--', color='b', label=f'y = {a:.{2}f}x {b:+.{2}f}')
plt.legend(prop={'size': 20})
plt.xlabel(r'$t(V_s = 0.5)$' + ' (days)', fontsize=25)
plt.ylabel(r'$t(H_s = 0.05)$' + ' (days)', fontsize=25)
plt.gca().tick_params(axis='both', which='major', labelsize=20)
plt.gca().yaxis.set_ticks_position('both')
plt.gca().xaxis.set_ticks_position('both')
plt.gca().tick_params(axis='both', which='major', direction='in')  # axis number size
plt.gca().annotate(f'R² = {r2:.2f}', xy=(0.98, 0.02), color='blue',
                   xycoords='axes fraction', fontsize=25, horizontalalignment='right', verticalalignment='bottom')

fig.tight_layout()

plt.savefig('data/grapevine/paper/final_e.png')

# _____ hist t _____

fig = plt.figure(figsize=(6, 6), dpi=100)
plt.hist(t_h005, 15, color='blue')
plt.gca().tick_params(axis='both', which='major', labelsize=25)
plt.gca().axes.yaxis.set_ticklabels([])
plt.gca().axes.get_yaxis().set_visible(False)
plt.gca().tick_params(axis='both', which='major', labelsize=30)
plt.title(r'$t(H_s = 0.05)$' + ' (days)', fontsize=25)

# _____ hist h _____

fig = plt.figure(figsize=(6, 6), dpi=100)
plt.ylim(30, 50)
plt.hist(h_0, 15, orientation='horizontal', color='grey')
plt.gca().tick_params(axis='both', which='major', labelsize=25)
plt.gca().axes.xaxis.set_ticklabels([])
plt.gca().axes.get_xaxis().set_visible(False)
plt.gca().tick_params(axis='both', which='major', labelsize=30)
plt.title(r'$H_0$', fontsize=25)

fig = plt.figure(figsize=(6, 6), dpi=100)
plt.ylim(127.5, 147.5)
plt.hist(h_n, 15, orientation='horizontal', color='grey')
plt.gca().tick_params(axis='both', which='major', labelsize=25)
plt.gca().axes.xaxis.set_ticklabels([])
plt.gca().axes.get_xaxis().set_visible(False)
plt.gca().tick_params(axis='both', which='major', labelsize=30)
plt.title(r'$H_n$', fontsize=25)

fig.tight_layout()








