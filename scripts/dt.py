import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt

for exp in ['DYN2020-05-15', 'ARCH2021-05-27']:
    res = pd.read_csv('X:/phenoarch_cache/cache_{0}/full_results_{0}.csv'.format(exp))
    dt = np.diff(sorted(res.groupby('task')['timestamp'].mean())) / 3600
    print(exp, np.mean(dt), np.median(dt))

exp = 'DYN2020-05-15'
# exp = 'ARCH2021-05-27'

df = []
for period in [1, 2, 3, 6, 9, 12, 15]:
    period_str = '' if period == 1 else '_period{}'.format(period)
    res = pd.read_csv('X:/phenoarch_cache/cache_{0}/full_results{1}_{0}.csv'.format(exp, period_str))
    print(period, len(res))

    for plantid in res['plantid'].unique():
        for angle in [k * 30 for k in range(12)]:
            s = res[(res['plantid'] == plantid) & (res['angle'] == angle)]
            dt = np.mean(np.diff(sorted(s.groupby('task')['timestamp'].mean())) / 3600)
            acc = np.sum(s['berryid'] != -1) / len(s)
            df.append([period, dt, exp, plantid, angle, acc])

df = pd.DataFrame(df, columns=['period', 'dt', 'exp', 'plantid', 'angle', 'acc'])

plt.plot(df['dt'], df['acc'], 'k.')

gb = df.groupby(['plantid', 'period'])[['acc', 'dt']].mean().reset_index()
for plantid in gb['plantid'].unique():
    s = gb[gb['plantid'] == plantid]
    # plt.plot(s['dt'], s['acc'] / s['acc'].iloc[0], '.-', label=plantid)
    plt.plot(s['dt'], s['acc'] / s['acc'].iloc[0], '.-', label=plantid)
plt.legend()
plt.ylabel('Ellipses tracked (%)')
plt.xlabel('Mean Δt between successive images (h)')







