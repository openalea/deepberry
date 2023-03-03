"""
Only done for 2020 exp, for the paper
"""

import numpy as np
import pandas as pd

from scipy.ndimage import uniform_filter1d

index = pd.read_csv('data/grapevine/image_index.csv')

exp = 'DYN2020-05-15'
res = pd.read_csv('X:/phenoarch_cache/cache_{0}/full_results_temporal_{0}.csv'.format(exp))
res['t'] = (res['timestamp'] - min(res['timestamp'])) / 3600 / 24

# remove plantid used during training
res = res[res['plantid'] != 7243]

# remove task too late
if exp == 'DYN2020-05-15':
    res = res[~(res['task'] == 2656)]

# ===== filtering berries ============================================================================================

"""
=== exp2020 plantid 7232 ===
# raw: 901 berryids. 
len > 1 : 847.
> 90% points: 293
not small: 293
normal end volume (or color): 281
bot normal volume color: 279

df_berry[(df_berry['mape'] < 2.5) & (df_berry['enough_points'])]
df_berry[(df_berry['mape'] < 2.5) & (df_berry['enough_points']) & (df_berry['normal_end_volume'])]

df_berry[(df_berry['enough_points']) & (df_berry['normal_end_volume'])]
df_berry[(df_berry['enough_points']) & (df_berry['not_small'])]
s = df_berry[(df_berry['enough_points']) & (df_berry['normal_end_volume']) & (df_berry['normal_end_hue'])]

"""
df_berry = []

for plantid in res['plantid'].unique():

    print(plantid)
    selec = res[(res['plantid'] == plantid) & (res['berryid'] != -1)]

    median_volume = np.median(selec.groupby(['angle', 'berryid'])['volume'].median())
    n_last = max(1, round(len(selec['task'].unique()) / 15))

    for angle in [k * 30 for k in range(12)]:
        s_angle = selec[selec['angle'] == angle]
        for k in s_angle['berryid'].unique():

            # one berry trajectory
            s = s_angle[s_angle['berryid'] == k].sort_values('t')
            v_averaged = uniform_filter1d(s['volume'], size=15, mode='nearest')  # TODO value tested only for dt=8h
            mape = 100 * np.mean(np.abs((v_averaged - s['volume']) / v_averaged))
            if len(s) > 1:
                v_scaled = (v_averaged - np.min(v_averaged)) / (np.max(v_averaged) - np.min(v_averaged))
                # TODO generalise more to different t distributions
                normal_end_volume = np.median(v_scaled[-n_last:]) > 0.3
                normal_end_hue = np.median(s['hue_scaled'][-n_last:]) > 120
                enough_points = len(s) > int(0.90 * len(selec['task'].unique()))
                not_small = np.median(s['volume']) > 0.2 * median_volume
                df_berry.append([plantid, angle, k, mape, not_small, enough_points, normal_end_volume, normal_end_hue])

df_berry_classif = pd.DataFrame(df_berry, columns=['plantid', 'angle', 'id', 'mape', 'not_small',
                                               'enough_points', 'normal_end_volume', 'normal_end_hue'])

df_berry_classif.to_csv('data/grapevine/berry_filter.csv', index=False)
