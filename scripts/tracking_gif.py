import os
import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt

from PIL import Image

from deepberry.src.openalea.deepberry.utils import ellipse_interpolation

index = pd.read_csv('data/grapevine/image_index.csv')

# fd = 'data/grapevine/temporal/results/'

exp = 'DYN2020-05-15'
# exp = 'ARCH2021-05-27'

res = pd.read_csv('X:/phenoarch_cache/cache_{0}/full_results_temporal_{0}.csv'.format(exp))

# =================================================================================================================

n_frames = np.max(res.groupby('plantid')['task'].nunique())

# for plantid in res_selec['plantid'].unique():

plantid = 7243

# for plantid in res['plantid'].unique():

selec0 = res[res['plantid'] == plantid]

# angle = selec0.groupby('angle').size().sort_values().index[-1]
for angle in selec0['angle'].unique():

    selec = selec0[selec0['angle'] == angle]

    print(len(selec['task'].unique()))

    # ===== Generate gif ======

    colors = [[int(k) for k in np.random.randint(0, 256, 3)] for k in range(1000)]

    exp, plantid, angle = selec.iloc[0][['exp', 'plantid', 'angle']]
    selec_index = index[(index['exp'] == exp) & (index['plantid'] == int(plantid)) & (index['imgangle'] == int(angle))]

    imgs = []
    tasks = list(selec.groupby('task')['timestamp'].mean().sort_values().index)
    for i_task, task in enumerate(tasks[::1]):
        print(i_task)
        s = selec[selec['task'] == task]
        row_img = selec_index[selec_index['taskid'] == task].iloc[0]
        img = cv2.cvtColor(cv2.imread('Z:/{}/{}/{}.png'.format(exp, row_img['taskid'], row_img['imgguid'])),
                           cv2.COLOR_BGR2RGB)
        s = s.sort_values('berryid')
        for _, row in s.iterrows():
            x, y, w, h, a = row['ell_x'], row['ell_y'], row['ell_w'], row['ell_h'], row['ell_a']

            # # hack to correct camera position shift
            # if row['exp'] == 'DYN2020-05-15' and row['timestamp'] > 1592571441:
            #     x, y = x + 4.2, y + 437.8

            ell = ellipse_interpolation(x, y, w, h, a, n_points=100)
            # if row['berryid'] in ids:
            if row['berryid'] != -1:
                # col = [int(k) for k in PALETTE[ids.index(int(row['berryid']))]]
                #col = [int(k) for k in PALETTE[int(row['berryid'])]]
                col = colors[int(row['berryid'])]

                img = cv2.ellipse(img, (round(x), round(y)), (round(w / 2), round(h / 2)), a, 0., 360,
                                  col, -1)

                img = cv2.ellipse(img, (round(x), round(y)), (round(w / 2), round(h / 2)), a, 0., 360,
                                  [0, 0, 0], 3)
            else:
                img = cv2.ellipse(img, (round(x), round(y)), (round(w / 2), round(h / 2)), a, 0., 360,
                                  [255, 255, 255], 5)

        # plt.figure('{}_{}'.format(i_task, task))
        # plt.imshow(img2)

        # img_save = cv2.resize(img, tuple((np.array([2048, 2448]) / 4).astype(int)))
        # plt.imsave('data/videos/gif_imgs4/{}.png'.format(str(i_task).zfill(3)), img_save)

        img = cv2.resize(img, tuple((np.array([2048, 2448]) / 6).astype(int)))

        s1, s2, s3 = img.shape
        img2 = np.zeros((int(s1 * 1.06), s2, s3)).astype(np.uint8)
        img2[:s1, :, :] = img

        t = str(i_task + 1).zfill(len(str(len(tasks))))

        img2 = cv2.putText(img2, '{}_{}_{} | t={}/{}'.format(exp, plantid, angle, t, len(tasks)),
                           (10, int(s1 * 1.04)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
        # plt.figure()
        # plt.imshow(img2)

        imgs.append(img2)

    # same number of frames per plant
    imgs = imgs + [imgs[-1]] * (n_frames - len(imgs))

    imgs_gif = (Image.fromarray(img) for img in imgs)
    # imgs = (Image.open(f) for f in paths)
    fps = 6
    # img = imgs_gif[0]  # extract first image from iterator
    img = next(imgs_gif)
    img.save(fp='data/videos/berry_tracking/7233_multi_angle/{}_{}_{}fps.gif'.format(plantid, angle, fps),
             format='GIF', append_images=imgs_gif, save_all=True, duration=1000/fps, loop=0)
