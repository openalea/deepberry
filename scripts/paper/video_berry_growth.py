import os
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from openalea.deepberry.utils import ellipse_interpolation

DIR_OUTPUT = 'data/grapevine/paper/'

index = pd.read_csv('data/grapevine/image_index.csv')
index = index[index['imgangle'].notnull()]

# ===== berry movie

exp, plantid, angle = 'DYN2020-05-15', 7232, 0
s_index = index[(index['exp'] == exp) & (index['plantid'] == plantid) & (index['imgangle'] == angle)]

res = pd.read_csv('X:/phenoarch_cache/cache_{0}/full_results_temporal_{0}.csv'.format(exp))
res['t'] = (res['timestamp'] - min(res['timestamp'])) / 3600 / 24
selec = res[(res['plantid'] == plantid) & (res['angle'] == angle)]

tasks = list(s_index.sort_values('timestamp')['taskid'])

gb = selec.groupby('berryid').size().sort_values().reset_index()

# berryids = np.random.choice(gb[(gb[0] > 120) & (gb['berryid'] != -1)]['berryid'], 4 * 7, replace=False)
berryids = [70,  27,  25,  47,  31, 102,  30,  71,  32,  67,  84,  62,  28, 23,  88,  96,  76,  54,  15,
            49,  89,  38,  42,  22,  43,  33, 78,  56]

for k_task, task in enumerate(tasks[1:]):

    plt.figure(figsize=(17, 10), dpi=100)

    row_index = s_index[s_index['taskid'] == task].iloc[0]
    path = 'Z:/{}/{}/{}.png'.format(exp, task, row_index['imgguid'])
    img = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)

    for k_berryid, berryid in enumerate(berryids):

        selec_id = selec[(selec['task'] == task) & (selec['berryid'] == berryid)]

        if len(selec_id) == 1:
            row = selec_id.iloc[0]
            x, y, w, h, a = row['ell_x'], row['ell_y'], row['ell_w'], row['ell_h'], row['ell_a']
            ell_x, ell_y = ellipse_interpolation(x, y, w, h, a, n_points=100)
            #
            # # with cv2
            # mask = cv2.ellipse(np.float32(img[:, :, 0] * 0), (round(x), round(y)),
            #                    (round((w * 0.85) / 2), round((h * 0.85) / 2)), a, 0., 360, (1), -1)
            # img = cv2.ellipse(img, (round(x), round(y)), (round(w / 2), round(h / 2)), a, 0., 360, [255, 0, 0], 1)
            # img = img[(round(y) - 35):(round(y) + 35), (round(x) - 35):(round(x) + 35)]
            # img = cv2.resize(img, (360, 360))
            # img2 = img.copy()
            # img2[325:, 250:] *= 0
            # img2 = cv2.putText(img2, 't={}'.format(round(row['t'], 1)),
            #                    (260, 350), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 1, cv2.LINE_AA)

            plt.subplot(4, 7, k_berryid + 1)
            plt.imshow(img)
            plt.gca().get_xaxis().set_visible(False)
            plt.gca().get_yaxis().set_visible(False)
            plt.plot(ell_x, ell_y, 'r-')
            plt.xlim((x - 50, x + 50))
            plt.ylim((y - 50, y + 50))

    plt.suptitle('t = {} days'.format(round(selec[selec['task'] == task]['t'].mean(), 1)), fontsize=20)
    plt.subplots_adjust(left=None, bottom=0.19, right=None, top=0.94, wspace=0.05, hspace=0.03)

    plt.savefig(DIR_OUTPUT + 'video_berry_growth/{}.png'.format(str(k_task).zfill(3)))
    plt.close('all')

# ____________________________________________________________________________________________________________________

image_folder = DIR_OUTPUT + 'video_berry_growth'
video_name = 'video.avi'
fps = 5

images = [img for img in os.listdir(image_folder) if img.endswith(".png")]
frame = cv2.imread(os.path.join(image_folder, images[0]))
height, width, layers = frame.shape

video = cv2.VideoWriter(video_name, 0, fps, (width, height))

for image in images:
    video.write(cv2.imread(os.path.join(image_folder, image)))

cv2.destroyAllWindows()
video.release()






