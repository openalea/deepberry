import pandas as pd

# ===== complete annotation dataset with metadata =====================================================================

df = pd.read_csv('data/grapevine/dataset/grapevine_annotation.csv')
index = pd.read_csv('data/grapevine/image_index.csv')

df['exp'] = [n.split('_')[0] for n in df['image_name']]
df['plantid'] = [int(n.split('_')[1]) for n in df['image_name']]

genotypes = []
for _, row in df.iterrows():
    s = index[(index['exp'] == row['exp']) & (index['plantid'] == row['plantid'])]
    genotypes.append(s.iloc[0]['genotype'])
df['genotype'] = genotypes

# TODO remove test vs valid  # ['ARCH2021-05-27', 'ARCH2022-05-18', 'DYN2020-05-15']
s = df[df['dataset'] == 'valid']
names = ['ARCH2021-05-27_7772_3793_30.png', 'ARCH2021-05-27_7760_3986_240.png',
        'DYN2020-05-15_7238_2388_330.png', 'DYN2020-05-15_7244_2636_330.png',
        'DYN2020-05-15_7240_2551_330.png', 'ARCH2021-05-27_7783_3942_0.png',
        'ARCH2021-05-27_7791_3835_300.png']
s_test = s[(s['exp'] == 'ARCH2022-05-18') | (s['image_name'].isin(names))]

import os
from shutil import move
for name in s_test['image_name'].unique():
    # path1 = 'data/grapevine/dataset/dataset_raw/image_valid/' + name
    # path2 = 'data/grapevine/dataset/dataset_raw/image_test/' + name
    # move(path1, path2)
    path1 = 'data/grapevine/dataset/dataset_raw/label_valid/' + name.replace('.png', '.json')
    path2 = 'data/grapevine/dataset/dataset_raw/label_test/' + name.replace('.png', '.json')
    move(path1, path2)

# ===== show info ====================================================================================================

df.groupby('exp')[['plantid', 'genotype', 'image_name']].nunique()

df[df['dataset'] == 'valid'].groupby('exp')[['plantid', 'genotype', 'image_name']].nunique()

df[df['dataset'] == 'valid'].groupby('exp').size()

s = index[index['exp'] == 'ARCH2021-05-27']
s.groupby('genotype')['plantid'].nunique()









