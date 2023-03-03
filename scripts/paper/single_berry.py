import os
import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt
from skimage import io

from scipy.stats import circmean, linregress
from scipy.interpolate import interp1d

from scipy.ndimage import uniform_filter1d, median_filter

from PIL import Image
from deepberry.src.openalea.deepberry.utils import ellipse_interpolation


DIR_OUTPUT = 'data/grapevine/paper/'


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

# ===== one berry as example ==========================================================================================

plantid, angle, berryid = 7232, 0, 28
s = res[(res['plantid'] == plantid) & (res['angle'] == angle) & (res['berryid'] == berryid)].sort_values('t')
hue_cv2 = (180 - 100 - np.array(s['hue_scaled'])) % 180
colors = np.array([cv2.cvtColor(np.uint8([[[h, 255, 140]]]), cv2.COLOR_HSV2RGB)[0][0] / 255. for h in hue_cv2])


plt.subplot(3, 1, 1)
# fig, ax = plt.subplots(figsize=(10, 10), dpi=100)
plt.gca().tick_params(axis='both', which='major', labelsize=20)
x, y = np.array(s['t']), np.array(s['volume']) / 1e4
plt.scatter(x, y, marker='o', c=colors, s=50)
# plt.title('Single berry size dynamics', fontsize=35)
plt.ylabel('Volume ($10^4$ px³)', fontsize=20)
# plt.xlabel('Time (days)', fontsize=10)
y_averaged = uniform_filter1d(y, size=int(len(s) / 10), mode='nearest')
plt.plot(x, y_averaged, '-', color='r', linewidth=3)
mape = 100 * np.mean(np.abs((y_averaged - y) / y_averaged))
txt = 'MAPE={}%'.format(round(mape, 2), int(angle), int(berryid))
# plt.text(0.99, 0.03, txt, horizontalalignment='right', verticalalignment='bottom', transform=ax.transAxes,
#          size=30, color='red')

# plt.savefig(DIR_OUTPUT + 'single_volume', bbox_inches='tight')
# plt.close()

plt.subplot(3, 1, 2)
# fig, ax = plt.subplots(figsize=(10, 10), dpi=100)
plt.gca().tick_params(axis='both', which='major', labelsize=20)
x, y = np.array(s['t']), np.array(s['hue_scaled'])
plt.fill_between(x, y - np.array(s['hue_scaled_std']) / 2, y + np.array(s['hue_scaled_std']) / 2,
                 color='grey', alpha=0.3)
for xi, yi, yi_std in zip(x, y, np.array(s['hue_scaled_std'])):
    plt.plot([xi, xi], [yi - yi_std / 2, yi + yi_std / 2], 'k-', alpha=0.3)
plt.scatter(x, y, marker='o', c=colors, s=50)
plt.gca().tick_params(axis='both', which='major', labelsize=20)  # axis number size
# plt.title('Single berry color dynamics', fontsize=35)
plt.ylabel('Scaled hue (°)', fontsize=20)
# plt.xlabel('Time (days)', fontsize=20)

# plt.savefig(DIR_OUTPUT + 'single_hue', bbox_inches='tight')
# plt.close()

plt.subplot(3, 1, 3)
# fig, ax = plt.subplots(figsize=(10, 10), dpi=100)
plt.gca().tick_params(axis='both', which='major', labelsize=20)
x, y = np.array(s['t']), np.array(s['hue_scaled_above50']) * 100
plt.scatter(x, y, marker='o', c=colors, s=50)
plt.gca().tick_params(axis='both', which='major', labelsize=20)  # axis number size
# plt.title('Single berry color dynamics', fontsize=35)
plt.ylabel('px(Hue > 50) (%)', fontsize=20)
plt.xlabel('Time (days)', fontsize=20)

# plt.savefig(DIR_OUTPUT + 'single_hue_50', bbox_inches='tight')
# plt.close()

# ===== berry movie

index = pd.read_csv('data/grapevine/image_index.csv')
index = index[index['imgangle'].notnull()]
s_index = index[(index['exp'] == exp) & (index['plantid'] == plantid) & (index['imgangle'] == angle)]

plt.figure()
tasks = s[(s['t'] > 22) & (s['t'] < 29)]['task'].unique()
for k_task, task in enumerate(tasks):

    row_index = s_index[s_index['taskid'] == task].iloc[0]
    path = 'Z:/{}/{}/{}.png'.format(exp, task, row_index['imgguid'])
    img = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)

    row = s[s['task'] == task].iloc[0]
    x, y, w, h, a = row['ell_x'], row['ell_y'], row['ell_w'], row['ell_h'], row['ell_a']
    ell = ellipse_interpolation(x, y, w, h, a, n_points=100)

    # # with cv2
    # mask = cv2.ellipse(np.float32(image[:, :, 0] * 0), (round(x), round(y)),
    #                    (round((w * 0.85) / 2), round((h * 0.85) / 2)), a, 0., 360, (1), -1)
    # image = cv2.ellipse(image, (round(x), round(y)), (round(w / 2), round(h / 2)), a, 0., 360, [255, 0, 0], 1)
    # image = image[(round(y) - 35):(round(y) + 35), (round(x) - 35):(round(x) + 35)]
    # image = cv2.resize(image, (360, 360))
    # img2 = image.copy()
    # img2[325:, 250:] *= 0
    # img2 = cv2.putText(img2, 't={}'.format(round(row['t'], 1)),
    #                    (260, 350), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 1, cv2.LINE_AA)

    # with matplotlib
    ell_x, ell_y = ellipse_interpolation(x, y, w, h, a, n_points=100)

    plt.subplot(1, len(tasks), k_task + 1)
    plt.imshow(img)
    plt.axis('off')
    plt.title(round(row['t'], 1))
    plt.plot(ell_x, ell_y, 'r-')
    plt.xlim((x - 40, x + 40))
    plt.ylim((y - 40, y + 40))


# ===== many individual graphs (supplementary material) ==================================================)============

K = 7
df_berry_s = df_berry.sample(K ** 2)
# df_berry_selec = df_berry.sort_values('n', ascending=False).iloc[:(K ** 2)].sort_values('mape')
# df_berry_selec = df_berry[(df_berry['n'] > 120) & (df_berry['mape'] < 2.5)]
# df_berry_selec = df_berry[df_berry['n'] > 120].sort_values('mape').iloc[(K ** 2):(2 * (K ** 2))]

for var in ['volume', 'hue_scaled']:

    plt.figure(var)

    for i, (_, row) in enumerate(df_berry_s.iterrows()):
        plantid, angle, berryid = row['plantid'], row['angle'], row['id']

        s = res[(res['plantid'] == plantid) & (res['angle'] == angle) & (res['berryid'] == berryid)].sort_values('t')

        if var == 'volume':
            x, y = np.array(s['t']), np.array(s['volume']) / 10_000
        elif var == 'hue_scaled':
            x, y = np.array(s['t']), np.array(s['hue_scaled'])

        ax = plt.subplot(K, K, i + 1)
        ax.yaxis.get_major_locator().set_params(integer=True)  # y axis = int

        hue_cv2 = (180 - 100 - np.array(s['hue_scaled'])) % 180
        plt.scatter(x, y, marker='.', c=np.array([cv2.cvtColor(np.uint8([[[h, 255, 140]]]),
                                                               cv2.COLOR_HSV2RGB)[0][0] / 255. for h in hue_cv2]))

        if var == 'volume':
            y_averaged = uniform_filter1d(y, size=15, mode='nearest')
            plt.plot(x, y_averaged, '-', color='r', linewidth=2)
        elif var == 'hue_scaled':
            plt.ylim((25, 155))

# ===== multiple berries in the same graph ============================================================================

plantid = 7232

berries = df_berry_classif[df_berry_classif['plantid'] == plantid]
# filter berries tracked during >90% of the experiment
berries = berries[berries['enough_points']]
# filter berries with no abnormalities
berries = berries[(berries['normal_end_volume']) & (berries['normal_end_hue']) & (berries['not_small'])]
# filter 10% berries with highest V_MAPE
berries = berries[berries['mape'] < 3.8]

# berries = berries.sample(20, replace=False)

selec = res[res['plantid'] == plantid]
df_single = []
for _, row in berries.iterrows():
    s = selec[(selec['angle'] == row['angle']) & (selec['berryid'] == row['id'])].sort_values('t')
    s['angle_id'] = f'{row["angle"]}_{row["id"]}'
    s = s.rename(columns={'hue_scaled': 'h', 'volume': 'v'})
    s['v'] = s['v'] * (0.158 ** 3) * 0.001  # px3 --> mm3 --> cm3 = mL
    s['v_smooth'] = median_filter(np.array(s['v']), size=25, mode='reflect')
    s['mape'] = row['mape']
    s = s[['v', 'v_smooth', 'h', 'angle_id', 'task', 't', 'mape']]
    df_single.append(s)
df_single = pd.concat(df_single)
# df_single.to_csv('data/grapevine/paper/single_berries.csv', index=False)

# ===== focus to understand volume =====


def y_to_x(x, y, y_target):
    """ x must be monotonous """
    k_next = next(k for k, val in enumerate(y) if val > y_target and k >= np.argmin(y))
    x1, x2 = x[k_next - 1], x[k_next]  # x1 < x2
    y1, y2 = y[k_next - 1], y[k_next]
    x_target = x1 + (x2 - x1) * ((y_target - y1) / (y2 - y1))
    return x_target


plt.figure()
df_vars = []
for angle_id in df_single['angle_id'].unique()[::1]:
    s = df_single[df_single['angle_id'] == angle_id].sort_values('t')
    t, v = np.array(s['t']), np.array(s['v_smooth'])
    v0, vmax = np.median(v[:20]), max(v)
    vr = (v - v0) / v0
    vs = (v - v0) / (vmax - v0)

    t25 = y_to_x(t, vs, 0.25)
    t75 = y_to_x(t, vs, 0.75)

    vr25 = np.interp(t25, t, vr)
    vr75 = np.interp(t75, t, vr)

    plt.plot(t, vr, '-', color='grey', alpha=0.6, linewidth=0.6)
    # plt.plot(t25, 0.25, 'go')
    # plt.plot(t75, 0.75, 'ro')
    plt.plot(t25, vr25, 'go')
    plt.plot(t75, vr75, 'ro')
    plt.plot([t25, t75], [vr25, vr75], 'b-')

    df_vars.append([v0, vmax, t25, t75, vr25, vr75])
df_vars = pd.DataFrame(df_vars, columns=['v0', 'vmax', 't25', 't75', 'vr25', 'vr75'])

duration = df_vars['t75'] - df_vars['t25']
vgain = (df_vars['vr75'] - df_vars['vr25'])  # proportionnal to vmax/v0
speed = vgain / duration
r2 = linregress(np.array(df_vars['vmax'] / df_vars['v0']), np.array(speed))[2] ** 2
print(r2)
plt.figure()
plt.plot(df_vars['vmax'] / df_vars['v0'], speed, 'ko')
r2 = linregress(np.array(df_vars['vmax'] / df_vars['v0']), np.array(duration))[2] ** 2
print(r2)
plt.figure()
plt.plot(df_vars['vmax'] / df_vars['v0'], duration, 'ko')

plt.plot(df_vars['v0'], speed, 'ko')
r2 = linregress(np.array(df_vars['v0']), np.array(speed))[2] ** 2
print(r2)

# ===== and the mean berry ? =====
gb = df_single.groupby('task').mean().reset_index().sort_values('t')
t, v = np.array(gb['t']), np.array(gb['v_smooth'])
v0, vmax = np.median(v[:20]), max(v)
vr = (v - v0) / v0
vs = (v - v0) / (vmax - v0)
t25 = y_to_x(t, vs, 0.25)
t75 = y_to_x(t, vs, 0.75)
vr25 = np.interp(t25, t, vr)
vr75 = np.interp(t75, t, vr)
duration = t75 - t25
vgain = vr75 - vr25  # proportionnal to vmax/v0
speed = vgain / duration
print(duration, speed, vmax/v0)

plt.plot(t, vr, 'k-')

# ======================================

do_x_scaling = False

hists = {'v0': [], 'vmax': [], 't_v': [], 'kin_v': [],
         'h0': [], 'hn': [], 't_h': [], 't_h01': [], 'kin_h': []}

for var in ['volume', 'hue_scaled', 'both']:

    fig, ax = plt.subplots()
    fontsize = 16
    ax.tick_params(axis='both', which='major', labelsize=10)
    if var == 'volume':
        # plt.ylim((-0.1, 1.1))
        # plt.ylabel(r'$V_s = (V - V_0) / (V_{n} - V_0)$', fontsize=fontsize)
        # plt.ylabel(r'$V_{s2} = (V - V_0) / V_0$', fontsize=fontsize)
        plt.ylabel(r'$V_{s3} = V / V_{max}$', fontsize=fontsize)
        if do_x_scaling:
            plt.xlabel(r'$t - t(V_s = 0.5)$ (days)', fontsize=fontsize)
        else:
            plt.xlabel(r'$t$ (days)', fontsize=fontsize)
    elif var == 'hue_scaled':
        # plt.ylabel(r'$H_s = (H - H_0) / (H_n - H_0)$', fontsize=fontsize)
        plt.ylabel('PC = Pixels with >10% color change (%)', fontsize=fontsize)
        plt.ylim((-0.1, 1.1))
        if do_x_scaling:
            plt.xlabel(r'$t - t(H_s = 0.5)$ (days)', fontsize=fontsize)
        else:
            plt.xlabel(r'$t$ (days)', fontsize=fontsize)

    elif var == 'both':
        # plt.xlim((-0.1, 1.1))
        # plt.ylim((-0.1, 1.1))
        plt.xlabel(r'$V_s$', fontsize=fontsize)
        # plt.xlabel(r'$(V - V_0) / V_0$', fontsize=fontsize)
        # plt.ylabel(r'$H_s$', fontsize=fontsize)
        plt.ylabel('PC (%)', fontsize=fontsize)

    # if var in ['volume', 'hue_scaled']:
    #     if do_x_scaling:
    #         plt.axhline(y=0.25, color='g', linestyle='--', linewidth=1., alpha=0.6)
    #         plt.axhline(y=0.75, color='g', linestyle='--', linewidth=1., alpha=0.6)
    #     else:
    #         plt.axhline(y=0.5, color='g', linestyle='--', linewidth=1., alpha=0.6)

    xy_list = []
    for angle_id in df_single['angle_id'].unique():
        s = df_single[df_single['angle_id'] == angle_id].sort_values('t')

        x_scaled, y_scaled = {}, {}

        # ===== Volume ================================================================================================

        x, y = np.array(s['t']), np.array(s['volume'])
        # y_averaged = uniform_filter1d(y, size=15, mode='nearest')
        y_averaged = median_filter(y, size=25, mode='reflect')

        y0, yn = np.median(y_averaged[:20]), max(y_averaged)
        # ys = (y_averaged - y0) / (yn - y0)
        ys = (y_averaged - y0) / y0
        # ys = y_averaged / yn

        if var == 'volume':
            hists['v0'].append(y0)
            hists['vmax'].append(yn)

            # # kinetics
            # k25 = next(k for k, val in enumerate(ys) if val > 0.25 and k >= np.argmin(ys))
            # k75 = next(k for k, val in enumerate(ys) if val > 0.75 and k >= np.argmin(ys))
            # hists['kin_v'].append(x[k75] - x[k25])

        if do_x_scaling:
            k_scaling = next(k for k, val in enumerate(ys) if val > 0.5 and k >= np.argmin(ys))
            xs = x - x[k_scaling]
            if var == 'volume':
                hists['t_v'].append(x[k_scaling])
        else:
            xs = x
        x_scaled['volume'], y_scaled['volume'] = xs, ys

        # ===== Color =================================================================================================

        x, y = np.array(s['t']), np.array(s['hue_scaled'])
        y0, yn = np.median(y[:20]), np.median(y[-20:])
        ys = (y - y0) / (yn - y0)

        if var == 'hue_scaled':
            hists['h0'].append(y0)
            hists['hn'].append(yn)

            # kinetics
            k25 = next(k for k, val in enumerate(ys) if val > 0.25 and k >= np.argmin(ys))
            k75 = next(k for k, val in enumerate(ys) if val > 0.75 and k >= np.argmin(ys))
            hists['kin_h'].append(x[k75] - x[k25])

        if do_x_scaling:
            k_scaling = next(k for k, val in enumerate(ys) if val > 0.5 and k >= np.argmin(ys))
            xs = x - x[k_scaling]
            if var == 'hue_scaled':
                hists['t_h'].append(x[k_scaling])
                k_scaling_01 = next(k for k, val in enumerate(ys) if val > 0.1 and k >= np.argmin(ys))
                hists['t_h01'].append(x[k_scaling_01])
        else:
            xs = x

        x_scaled['hue_scaled'], y_scaled['hue_scaled'] = xs, ys

        # =============================================================================================================

        if var == 'both':
            x_plot, y_plot = y_scaled['volume'], y_scaled['hue_scaled']
        else:
            x_plot, y_plot = x_scaled[var], y_scaled[var]

        plt.plot(x_plot, y_plot, '-', color='grey', linewidth=0.6, alpha=0.6)

        # x-scaling means that measurements are not at the same timings anymore across berries. So interpolation is
        # necessary. (Here, it's chosen to interpolate through all x-int values across the berry range)
        f = interp1d(x_plot, y_plot, fill_value='extrapolate')
        x_lnp = np.arange(int(min(x_plot)), int(max(x_plot)) + 1)
        xy_list.append([x_lnp, f(x_lnp)])

    xy = pd.DataFrame(np.concatenate(xy_list, axis=1).T, columns=['x', 'y'])

    # for metrics, linestyle in zip(['mean'], ['-']):
    #     # single berry
    #     if do_x_scaling:
    #         xy_gb = xy.groupby('x').agg(metrics).reset_index().sort_values('x')
    #         plt.plot(xy_gb['x'], xy_gb['y'], 'k' + linestyle)

        # # mean berry (using the same berries as the one successfully tracked, to remove shift bias). Boths methods
        # # still differ by the way data are standardised (per berry or globally)
        # selec_same_berries = pd.concat([selec[(selec['angle'] == r['angle']) & (selec['berryid'] == r['id'])]
        #                                 for _, r in df_berry.iterrows()])
        # gb = selec_same_berries.groupby('task')[[var, 't']].agg(metrics).reset_index().sort_values('t')
        # x_mean, y_mean = np.array(gb['t']), np.array(gb[var])
        # if var == 'volume':
        #     y_mean_scaled = y_mean / max(y_mean)
        #     k_scaling = next(k for k, val in enumerate(y_mean_scaled) if val > 0.85 and k >= np.argmin(y_mean_scaled))
        # elif var == 'hue_scaled':
        #     y_mean_scaled = (y_mean - min(y_mean)) / (max(y_mean) - min(y_mean))
        #     k_scaling = next(k for k, val in enumerate(y_mean_scaled) if val > 0.5 and k >= np.argmin(y_mean_scaled))
        # if do_x_scaling:
        #     x_mean_scaled = x_mean - x_mean[k_scaling]
        # else:
        #     x_mean_scaled = x_mean
        # plt.plot(x_mean_scaled, y_mean_scaled, 'r' + linestyle)

    plt.savefig(DIR_OUTPUT + 'multi_berry_{}_x{}.png'.format(var, do_x_scaling), bbox_inches='tight')


def fig_histo():
    fig, ax = plt.subplots(figsize=(5, 1.5), dpi=100)
    plt.gca().axes.yaxis.set_ticklabels([])
    plt.gca().axes.get_yaxis().set_visible(False)
    ax.tick_params(axis='both', which='major', labelsize=20)


# V0
fig_histo()
plt.xlabel(r'$V_0$ ($10^4$ px³)', fontsize=30)
plt.hist(np.array(hists['v0']) / 1e4, 15, color='grey')
plt.savefig(DIR_OUTPUT + 'multi_berry_hist_v0.png', bbox_inches='tight')

# Vmax
fig_histo()
plt.xlabel(r'$V_{max}$ ($10^4$ px³)', fontsize=30)
plt.hist(np.array(hists['vmax']) / 1e4, 15, color='grey')
plt.savefig(DIR_OUTPUT + 'multi_berry_hist_vmax.png', bbox_inches='tight')

# H0
fig_histo()
plt.xlabel(r'$H_0$', fontsize=30)
plt.hist(np.array(hists['h0']), 15, color='grey')
plt.savefig(DIR_OUTPUT + 'multi_berry_hist_h0.png', bbox_inches='tight')

# Hn
fig_histo()
plt.xlabel(r'$H_n$', fontsize=30)
plt.hist(np.array(hists['hn']), 15, color='grey')
plt.savefig(DIR_OUTPUT + 'multi_berry_hist_hn.png', bbox_inches='tight')

# V_kin
fig_histo()
plt.xlim((-1, 16))
plt.xlabel(r'$V_{kin} = t2 - t1$ (days)', fontsize=30)
plt.hist(np.array(hists['kin_v']), 15, color='green', alpha=0.6)
plt.savefig(DIR_OUTPUT + 'multi_berry_hist_vkin.png', bbox_inches='tight')

# H_kin
fig_histo()
plt.xlim((-1, 16))
plt.xlabel(r'$H_{kin} = t2 - t1$ (days)', fontsize=30)
plt.hist(np.array(hists['kin_h']), 15, color='green', alpha=0.6)
plt.savefig(DIR_OUTPUT + 'multi_berry_hist_hkin.png', bbox_inches='tight')

# t(Hs = 0.5)
fig_histo()
plt.xlim((9, 36))
plt.xlabel(r'$t(H_s = 0.5)$ (days)', fontsize=30)
plt.hist(np.array(hists['t_h']), 15, color='green', alpha=0.6)
plt.savefig(DIR_OUTPUT + 'multi_berry_hist_th.png', bbox_inches='tight')

# t(Vs = 0.5)
fig_histo()
plt.xlim((9, 36))
plt.xlabel(r'$t(V_s = 0.5)$ (days)', fontsize=30)
plt.hist(np.array(hists['t_v']), 15, color='green', alpha=0.6)
plt.savefig(DIR_OUTPUT + 'multi_berry_hist_tv.png', bbox_inches='tight')

# t(Vs = 0.5) vs t(Hs = 0.5)
fig, ax = plt.subplots(figsize=(3, 5), dpi=100)
ax.tick_params(axis='both', which='major', labelsize=20)
x, y = hists['t_v'], hists['t_h']
ax.axline((np.mean(x), np.mean(x)), slope=1, color='k', linestyle='-', label='x = y')
plt.plot(x, y, '.', color='grey')
r2 = linregress(x, y)[2] ** 2
plt.text(0.52, 0.03, f'R² = {r2:.2f}', color='red',
         horizontalalignment='left', verticalalignment='bottom', transform=ax.transAxes, fontsize=15)
a, b = np.polyfit(x, y, 1)
plt.plot([min(x), max(x)], a * np.array([min(x), max(x)]) + b, '--', color='r', label=f'y = {a:.{2}f}x {b:+.{2}f}')
plt.legend()
plt.xlabel(r'$t(V_s = 0.5)$ (days)', fontsize=20)
plt.ylabel(r'$t(H_s = 0.5)$ (days)', fontsize=20)
plt.savefig(DIR_OUTPUT + 'multi_berry_hist_cor1.png', bbox_inches='tight')

# t(Vs = 0.5) vs t(Hs = 0.5)
fig, ax = plt.subplots(figsize=(3, 5), dpi=100)
ax.tick_params(axis='both', which='major', labelsize=20)
x, y = hists['kin_v'], hists['kin_h']
ax.axline((np.mean(x), np.mean(x)), slope=1, color='k', linestyle='-', label='x = y')
plt.plot(x, y, '.', color='grey')
r2 = linregress(x, y)[2] ** 2
plt.text(0.52, 0.03, f'R² = {r2:.2f}', color='red',
         horizontalalignment='left', verticalalignment='bottom', transform=ax.transAxes, fontsize=15)
a, b = np.polyfit(x, y, 1)
plt.plot([min(x), max(x)], a * np.array([min(x), max(x)]) + b, '--', color='r', label=f'y = {a:.{2}f}x {b:+.{2}f}')
plt.legend()
plt.xlabel(r'$V_{kin}$ (days)', fontsize=20)
plt.ylabel(r'$H_{kin}$ (days)', fontsize=20)
plt.savefig(DIR_OUTPUT + 'multi_berry_hist_cor2.png', bbox_inches='tight')

# additional graphs for Charles

v0, vmax = np.array(hists['v0']), np.array(hists['vmax'])
vkin = np.array(hists['kin_v'])

plt.figure()
plt.xlabel('$V_0$', fontsize=20)
plt.ylabel('$V_{max}$', fontsize=20)
plt.plot(v0, vmax, 'ko')
a, b = np.polyfit(v0, vmax, 1)
plt.plot([0, max(v0)], a * np.array([0, max(v0)]) + b, '--', color='r', label=f'$V_{{max}}$ = {a:.{2}f}$V_0$ {b:+.{2}f}')
plt.xlim((0, max(v0) * 1.05))
plt.ylim((0, max(vmax) * 1.05))
plt.legend(prop={'size': 15})
plt.gca().set_aspect('equal', adjustable='box')

plt.figure()
plt.xlabel('$V_0$', fontsize=20)
plt.ylabel('$V_{max} / V_0$', fontsize=20)
plt.plot(v0, vmax/v0, 'ko')

plt.figure()
plt.xlabel('$V_0$', fontsize=20)
plt.ylabel('$(1 / V_{kin}) . (V_{max} / V_0$)', fontsize=20)
plt.plot(v0, (1/vkin) * (vmax/v0), 'ko')

plt.figure()
plt.xlabel(r'$t(V_s = 0.5)$', fontsize=20)
plt.ylabel('$V_{max} / V_0$', fontsize=20)
plt.plot(np.array(hists['t_v']), vmax/v0, 'ko')

plt.figure()
plt.xlabel(r'$t(V_s = 0.5)$', fontsize=20)
plt.ylabel('$V_{kin}$', fontsize=20)
plt.plot(np.array(hists['t_v']), vkin, 'ko')

# new graphs for Charles


plt.figure()
plt.xlabel('$V_0$', fontsize=20)
plt.ylabel('t(Vs=0.5)', fontsize=20)
plt.plot(v0, np.array(hists['t_v']), 'ko')

plt.figure()
plt.xlabel('$V_0$', fontsize=20)
plt.ylabel('$Vkin', fontsize=20)
plt.plot(v0, vkin, 'ko')


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
        # image = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)
        #
        # x, y, w, h, a = row_ell[['ell_x', 'ell_y', 'ell_w', 'ell_h', 'ell_a']]
        #
        # mask = cv2.ellipse(np.float32(image[:, :, 0] * 0), (round(x), round(y)), (round(w / 2), round(h / 2)),
        #                    a, 0., 360, (1), -1)
        # pixels = image[mask == 1]  # r, g, b
        # pixels_hsv = cv2.cvtColor(np.array([pixels]), cv2.COLOR_RGB2HSV)[0]

        df_color.append([row_ell['t']] + list(np.mean(pixels, axis=0)) + list(np.mean(pixels_hsv, axis=0)))

    df_color = pd.DataFrame(df_color, columns=['t', 'r', 'g', 'b', 'h', 's', 'v'])

    for i, col in enumerate(['r', 'g', 'b', 'orange', 'black', 'grey']):
        plt.plot(df_color['t'], df_color.iloc[:, i + 1], '-')















