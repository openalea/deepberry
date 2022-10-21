import pandas as pd
import cv2
import matplotlib.pyplot as plt
import numpy as np

from deepberry.src.openalea.deepberry.utils import ellipse_interpolation

PATH = 'data/grapevine/results/'

df = pd.read_csv(PATH + 'full_results.csv')
index = pd.read_csv('data/grapevine/image_index.csv')

plantid, angle = 7236, 120
# plantid, angle = 7243, 120
exp = 'DYN2020-05-15'

selec = df[(df['exp'] == exp) & (df['plantid'] == plantid) & (df['angle'] == angle)]

# for loading images
selec_index = index[(index['exp'] == exp) & (index['plantid'] == plantid) & (index['imgangle'] == angle)]

tasks = list(selec.groupby('task')['timestamp'].mean().sort_values().reset_index()['task'])

# ===== plot images of berry+ellipse without background ============================================================

for i_task, task in enumerate(tasks):

    row_img = selec_index[selec_index['taskid'] == task].iloc[0]
    path = 'Z:/{}/{}/{}.png'.format(exp, row_img['taskid'], row_img['imgguid'])
    img = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)

    ellipses = selec[selec['task'] == task]

    mask_ell = img * 0
    for _, row in ellipses.iterrows():
        mask_ell = cv2.ellipse(mask_ell,
                            (round(row['ell_x']), round(row['ell_y'])),
                            (round(row['ell_w'] / 2), round(row['ell_h'] / 2)),
                            row['ell_a'], 0., 360,
                            (1, 1, 1), -1)

    img_ell = img * mask_ell  # black background
    img_ell[img_ell == [0, 0, 0]] += 255  # white background

    # red ellipses
    for _, row in ellipses.iterrows():
        x, y, w, h, a = row[['ell_x', 'ell_y', 'ell_w', 'ell_h', 'ell_a']]
        lsp_x, lsp_y = ellipse_interpolation(x=x, y=y, w=w, h=h, a=a, n_points=30)
        img_ell = cv2.polylines(img_ell, [np.array([lsp_x, lsp_y]).T.astype('int32')], True, (255, 0, 0), 2)

    plt.imsave('data/grapevine/temporal/ell/{}.png'.format(i_task), img_ell)

# ===== test registration ===========================================================================================

akaze = cv2.AKAZE_create()

task_ref = selec.groupby('task').size().sort_values().index[-1]

imgs = []
for i_task, task in enumerate([task_ref, tasks[80]]):

    row_img = selec_index[selec_index['taskid'] == task].iloc[0]
    path = 'Z:/{}/{}/{}.png'.format(exp, row_img['taskid'], row_img['imgguid'])
    img = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)

    ellipses = selec[selec['task'] == task]

    mask_ell = img * 0
    for _, row in ellipses.iterrows():
        mask_ell = cv2.ellipse(mask_ell,
                            (round(row['ell_x']), round(row['ell_y'])),
                            (round(row['ell_w'] / 2), round(row['ell_h'] / 2)),
                            row['ell_a'], 0., 360,
                            (1, 1, 1), -1)

    img_ell = img * mask_ell  # black background
    # img_ell[img_ell == [0, 0, 0]] += 255  # white background

    imgs.append(img_ell)

    # # feature extraction
    # gray = cv2.cvtColor(img_ell, cv2.COLOR_RGB2GRAY)
    # kp, descriptor = akaze.detectAndCompute(gray, None)
    # img_ell = cv2.drawKeypoints(gray, kp, img_ell)

    # plt.figure(i_task)
    # plt.imshow(img_ell)


img1 = cv2.cvtColor(imgs[0], cv2.COLOR_RGB2GRAY)  # referenceImage
img2 = cv2.cvtColor(imgs[1], cv2.COLOR_RGB2GRAY)  # sensedImage

plt.figure('img2')
plt.imshow(img2)
plt.figure('ref')
plt.imshow(img1)


# Find the keypoints and descriptors with SIFT
kp1, des1 = akaze.detectAndCompute(img1, None)
kp2, des2 = akaze.detectAndCompute(img2, None)

# BFMatcher with default params
bf = cv2.BFMatcher()
matches = bf.knnMatch(des1, des2, k=2)

# Apply ratio test
good_matches = []
for m, n in matches:
    if m.distance < 0.75 * n.distance:
        good_matches.append([m])
print(len(good_matches))

# # Draw matches
# img3 = cv2.drawMatchesKnn(img1, kp1, img2, kp2, good_matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
# plt.imshow(img3)

# Select good matched keypoints
ref_matched_kpts = np.float32([kp1[m[0].queryIdx].pt for m in good_matches])
sensed_matched_kpts = np.float32([kp2[m[0].trainIdx].pt for m in good_matches])

# Compute homography
H, status = cv2.findHomography(sensed_matched_kpts, ref_matched_kpts, cv2.RANSAC, 5.0)

# Warp image
warped_image = cv2.warpPerspective(img2, H, (img2.shape[1], img2.shape[0]))

plt.figure('img2 bis')
plt.imshow(warped_image)

# ===== try registration with only ellipses parameters ===================================

akaze = cv2.AKAZE_create()

task_ref = selec.groupby('task').size().sort_values().index[-1]

imgs = []
ellipses = []
for i_task, task in enumerate([task_ref, tasks[80]]):

    row_img = selec_index[selec_index['taskid'] == task].iloc[0]
    path = 'Z:/{}/{}/{}.png'.format(exp, row_img['taskid'], row_img['imgguid'])
    img = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)

    ellipses.append(selec[selec['task'] == task])

    imgs.append(img)

img1 = imgs[0]
img2 = imgs[1]

plt.figure('img2')
plt.imshow(img2)
plt.figure('ref')
plt.imshow(img1)

# # Find the keypoints and descriptors with SIFT
# kp1, des1 = akaze.detectAndCompute(img1, None)
# kp2, des2 = akaze.detectAndCompute(img2, None)

pts1 = np.array(ellipses[0][['ell_x', 'ell_y']])
pts2 = np.array(ellipses[1][['ell_x', 'ell_y']])

des1_bis = (np.array(ellipses[0][['area']]) / 8000 * 255).astype(np.uint8)
des2_bis = (np.array(ellipses[1][['area']]) / 8000 * 255).astype(np.uint8)

# BFMatcher with default params
bf = cv2.BFMatcher()
matches = bf.knnMatch(des1_bis, des2_bis, k=2)

# Apply ratio test
good_matches = []
for m, n in matches:
    if m.distance < 0.75 * n.distance:
        good_matches.append([m])
print(len(good_matches))

# # Draw matches
# img3 = cv2.drawMatchesKnn(img1, kp1, img2, kp2, good_matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
# plt.imshow(img3)

# Select good matched keypoints
ref_matched_kpts = np.float32([pts1[m[0].queryIdx] for m in good_matches])
sensed_matched_kpts = np.float32([pts2[m[0].trainIdx] for m in good_matches])

# Compute homography
H, status = cv2.findHomography(sensed_matched_kpts, ref_matched_kpts, cv2.RANSAC, 5.0)

# Warp image
warped_image = cv2.warpPerspective(img2, H, (img2.shape[1], img2.shape[0]))

plt.figure('img2 bis')
plt.imshow(warped_image)
