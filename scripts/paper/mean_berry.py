"""
PhenoArch 2020
8 plants (excludes plantid 7243 which was used for model training)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

exp = 'DYN2020-05-15'
res = pd.read_csv('X:/phenoarch_cache/cache_{0}/full_results_segmentation_{0}.csv'.format(exp))
res['t'] = (res['timestamp'] - min(res['timestamp'])) / 3600 / 24

selec = res[res['plantid'] != 7243]

# ===== mean berry dynamics =========================================================================================

var = 'volume'
plt.figure()
plt.title('Mean berry volume (V)')
plt.ylabel('V / Vmax')
plt.xlabel('Time (days)')
for plantid in selec['plantid'].unique():
    s = selec[selec['plantid'] == plantid]
    gb = s.groupby('task')[[var, 't']].mean().reset_index().sort_values('t')
    y = gb[var] / max(gb[var])
    plt.plot(gb['t'], y, 'k.-', linewidth=0.7, markersize=3)

var = 'roundness'
plt.figure()
plt.title('Mean berry roundness (R)')
plt.ylabel('R')
plt.xlabel('Time (days)')
for plantid in selec['plantid'].unique():
    s = selec[selec['plantid'] == plantid]
    gb = s.groupby('task')[[var, 't']].mean().reset_index().sort_values('t')
    y = gb[var]
    plt.plot(gb['t'], y, 'k.-', linewidth=0.7, markersize=3)

var = 'hue_scaled'
plt.figure()
plt.title('Mean berry hue (H)')
plt.ylabel('(H - Hmin) / (Hmax - Hmin)')
plt.xlabel('Time (days)')
for plantid in selec['plantid'].unique():
    s = selec[selec['plantid'] == plantid]
    gb = s.groupby('task')[[var, 't']].mean().reset_index().sort_values('t')
    y = (gb[var] - min(gb[var])) / (max(gb[var]) - min(gb[var]))
    plt.plot(gb['t'], y, 'k.-', linewidth=0.7, markersize=3)

# ===== histograms ===============================================================================================

plt.figure()
plantids = [p for p in selec['plantid'].unique() if p != 7235]
for k, plantid in enumerate(plantids):
    s = selec[selec['plantid'] == plantid]
    s = s[s['task'].isin((2643, 2638, 2641))]  # last day with 3 snapshots
    print(plantid, len(s['task'].unique()), len(s))

    ax = plt.subplot(len(plantids), 1, k + 1)
    plt.hist(s['volume'], 70)

    ax.axes.yaxis.set_ticklabels([])
    ax.axes.yaxis.set_ticks_position('none')
    if k + 1 != len(plantids):
        ax.axes.xaxis.set_ticklabels([])
    else:
        ax.set_xlabel('Berry volume (px³)')

