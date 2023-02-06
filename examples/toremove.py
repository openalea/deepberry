# TODO remove file

import cv2

import pandas as pd

"""
data (PhenoArch 2021 - plantid 7794 - task 3786 - angle 120)
"""

# plantid = 7794

exp = 'ARCH2021-05-27'
res = pd.read_csv('X:/phenoarch_cache/cache_{0}/full_results_temporal_{0}.csv'.format(exp))

selec = res[(res['plantid'] == 7794) & (res['angle'] == 120)]
for task in selec.sort_values('timestamp')['task'].unique()[::2]:
    s = selec[selec['task'] == task]
    s2 = s[['ell_x', 'ell_y', 'ell_w', 'ell_h', 'ell_a', 'score', 'hue_scaled', 'area', 'volume', 'roundness']]
    s.to_csv('deepberry/examples/data/temporal/{}.csv'.format(s['timestamp'].iloc[0]))


index = pd.read_csv('data/grapevine/image_index.csv')
index = index[index['imgangle'].notnull()]

s_index = index[(index['exp'] == exp) & (index['plantid'] == 7794) & (index['imgangle'] == 120)]

img = cv2.cvtColor(cv2.imread('Z:/{}/{}/{}.png'.format(exp, row_img['taskid'], row_img['imgguid'])),
                   cv2.COLOR_BGR2RGB)

cv2.imwrite('deepberry/examples/data/image.png', cv2.cvtColor(img, cv2.COLOR_RGB2BGR))



