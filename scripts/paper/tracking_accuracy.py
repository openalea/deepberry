import pandas as pd
import numpy as np
import matplotlib.pyplot as plt



# ===== accuracy ====================================================================================================

index = pd.read_csv('data/grapevine/image_index.csv')

df = []
for exp in ['DYN2020-05-15', 'ARCH2021-05-27']:

    res = pd.read_csv('X:/phenoarch_cache/cache_{0}_NEW/full_results_temporal_{0}.csv'.format(exp))

    # remove plantids included in the training set
    if exp == 'DYN2020-05-15':
        res = res[res['plantid'] != 7243]
    if exp == 'ARCH2021-05-27':
        res = res[~res['plantid'].isin([7764, 7781])]

    for plantid in res['plantid'].unique():
        for angle in [k * 30 for k in range(12)]:
            s = res[(res['plantid'] == plantid) & (res['angle'] == angle)]
            acc = np.sum(s['berryid'] != -1) / len(s)
            df.append([exp, plantid, angle, acc])
df = pd.DataFrame(df, columns=['exp', 'plantid', 'angle', 'acc'])

for k, plantid in enumerate(df['plantid'].unique()):
    s = df[df['plantid'] == plantid]
    col = 'r' if plantid in [7781, 7791, 7245, 7236] else 'k'
    plt.plot([k] * len(s), s['acc'], '.', color=col)
    plt.plot(k, np.median(s['acc']), 'o', color=col)

n = 0
plt.ylim((-2, 102))
plt.ylabel('Accuracy (%)')
for exp in df['exp'].unique():
    s0 = df[df['exp'] == exp]
    for plantid in s0['plantid'].unique():
        s = s0[s0['plantid'] == plantid]
        plt.plot([n] * len(s), s['acc'] * 100, 'ko')
        plt.plot([n] * 2, [min(s['acc'] * 100), max(s['acc'] * 100)], 'k-')
        n += 1
    n += 5

