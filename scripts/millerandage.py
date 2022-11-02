"""
area = more visual than volume
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv('data/grapevine/results/full_results.csv')

# exp = 'DYN2020-05-15'
exp = 'ARCH2021-05-27'
# genotype = 'V.vinifera/V6863_H10-PL3'  # V_threshold = 95000
genotype = 'V.vinifera/V6860_DO4-PL4'

selec = df[df['exp'] == exp]
selec = selec[selec['genotype'] == genotype]

plt.figure()
plantids = selec['plantid'].unique()
for k, plantid in enumerate(plantids):
    s = selec[selec['plantid'] == plantid]

    tasks = list(s.groupby('task').mean().sort_values('t').reset_index()['task'])

    ax = plt.subplot(len(plantids), 1, k + 1)
    ax.xaxis.grid(True)
    xmax = 10000
    plt.xlim((0, xmax))
    w = xmax / 200
    plt.hist(s[s['task'].isin(tasks[:5])]['area'], bins=np.arange(0, xmax + w, w), alpha=0.5, color='green')
    plt.hist(s[s['task'].isin(tasks[-5:])]['area'], bins=np.arange(0, xmax + w, w), alpha=0.5, color='darkblue')

    ax.axes.yaxis.set_ticklabels([])
    ax.axes.yaxis.set_ticks_position('none')
    if k + 1 != len(plantids):
        ax.axes.xaxis.set_ticklabels([])
    else:
        ax.set_xlabel('Berry area (px²)')

# ====================================================================================================================


















