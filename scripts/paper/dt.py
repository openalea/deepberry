import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt

TIME_PERIODS = [1, 2, 3, 6, 9, 12, 15]

# ===== what is the mean & median dt per experiment ===================================================================

for exp in ['DYN2020-05-15', 'ARCH2021-05-27']:
    res = pd.read_csv('X:/phenoarch_cache/cache_{0}/full_results_temporal_{0}.csv'.format(exp))
    dt = np.diff(sorted(res.groupby('task')['timestamp'].mean())) / 3600
    print(exp, np.mean(dt), np.median(dt))

# ===== tracking performance = f(dt) ==================================================================================
"""
- Only on best angle
- Mean of plant means
"""

df = []
for exp in ['DYN2020-05-15', 'ARCH2021-05-27']:
    for period in TIME_PERIODS:
        period_str = '' if period == 1 else '_{}'.format(period)
        res = pd.read_csv('X:/phenoarch_cache/cache_{0}/full_results_temporal_{0}{1}.csv'.format(exp, period_str))
        print(period, len(res))

        for plantid in res['plantid'].unique():
            s0 = res[res['plantid'] == plantid]
            angle = s0.groupby('angle').size().sort_values().index[-1]
            s = s0[s0['angle'] == angle]
            dt = np.mean(np.diff(sorted(s.groupby('task')['timestamp'].mean())) / 3600)
            acc = np.sum(s['berryid'] != -1) / len(s)
            df.append([period, dt, exp, plantid, angle, acc])

df = pd.DataFrame(df, columns=['period', 'dt', 'exp', 'plantid', 'angle', 'acc'])

# for the paper
print(df.groupby(['exp', 'plantid', 'period']).mean().reset_index().groupby(['exp', 'period']).mean())

boxplots = {k * 8: list(df[df['period'] == k]['acc'] * 100) for k in TIME_PERIODS}

# Plot the dataframe
plt.figure()
pd.DataFrame(boxplots).plot(kind='box', title='')
plt.ylim((0, 100))
plt.ylabel('Ellipses tracked (%)')
plt.xlabel('Δt (h)')

# ===== effect of dt on t(H=0.1) from single berry dynamics ===========================================================

exp = 'DYN2020-05-15'
res = pd.read_csv('X:/phenoarch_cache/cache_{0}/full_results_temporal_{0}.csv'.format(exp))
res['t'] = (res['timestamp'] - min(res['timestamp'])) / 3600 / 24
# remove plantid used during training
res = res[res['plantid'] != 7243]
# remove task too late
if exp == 'DYN2020-05-15':
    res = res[~(res['task'] == 2656)]

# generated with berry_filtering.py
df_berry = pd.read_csv('data/grapevine/berry_filter.csv')

plantid = 7232
selec = res[res['plantid'] == plantid]
df_berry_selec = df_berry[df_berry['plantid'] == plantid]

df = []
# angle, berryid = df_berry_selec.iloc[1][['angle', 'id']]
for k_berry, (_, row) in enumerate(df_berry_selec.iterrows()):
    angle, berryid = row[['angle', 'id']]

    # plt.figure()
    for k, period in enumerate(TIME_PERIODS):
        s = selec[(selec['angle'] == angle) & (selec['berryid'] == berryid)].sort_values('t')
        s = s[::period]
        x = np.array(s['t'])
        y = np.array(s['hue_scaled'])
        y_scaled = (y - np.median(y[:3])) / (np.median(y[-3:]) - np.median(y[:3]))

        # ax = plt.subplot(len(periods), 1, k + 1)
        # plt.xlim((min(selec['t']) - 2, max(selec['t']) + 2))
        # plt.plot(x, y_scaled, 'k.-')

        q = 0.1  # 0.5
        k = next(k for k, val in enumerate(y_scaled) if val > q and k >= np.argmin(y_scaled))
        t_01 = x[k - 1] + (x[k] - x[k - 1]) * ((q - y_scaled[k - 1]) / (y_scaled[k] - y_scaled[k - 1]))

        # plt.plot(t_01, 0.1, 'go')
        # plt.text(0.9, 0.1, 't(H=0.1)={}d'.format(round(t_01, 1)), ha='right', va='bottom', transform=ax.transAxes, color='green')
        df.append([k_berry, period, t_01])

df = pd.DataFrame(df, columns=['berry', 'period', 't01'])

boxplots = {k * 8: [] for k in TIME_PERIODS[1:]}
for berry in df['berry'].unique():
    row1 = df[(df['berry'] == berry) & (df['period'] == 1)].iloc[0]
    for period in TIME_PERIODS[1:]:
        row2 = df[(df['berry'] == berry) & (df['period'] == period)].iloc[0]
        boxplots[period * 8].append(abs(row2['t01'] - row1['t01']))

# Plot the dataframe
pd.DataFrame(boxplots).plot(kind='box', title='')
plt.ylabel('Absolute difference with the reference value (Δt=8h)')
plt.xlabel('Δt (h)')








