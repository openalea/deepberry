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

# ===== show info ====================================================================================================

df.groupby('exp')[['plantid', 'genotype', 'image_name']].nunique()

df[df['dataset'] == 'valid'].groupby('exp')[['plantid', 'genotype', 'image_name']].nunique()

df[df['dataset'] == 'valid'].groupby('exp').size()

s = index[index['exp'] == 'ARCH2021-05-27']
s.groupby('genotype')['plantid'].nunique()









