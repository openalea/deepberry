"""
Run the detection and segmentation models on the test dataset
https://stats.stackexchange.com/questions/531132/on-which-set-train-val-test-do-people-calculate-f1-score-precision-and-recall
"""

import cv2
import numpy as np
import pandas as pd

from deepberry.src.openalea.deepberry.ellipse_segmentation import \
    berry_detection, berry_segmentation, load_berry_models

MODEL_DET, MODEL_SEG = load_berry_models('Y:/lepseBinaries/Trained_model/deepberry/')

# dir containing image_test folder
DIR_DATASET = 'data/grapevine/dataset/dataset_raw/'

# dataframe generated with the script generate_training_datasets.py
df_annot = pd.read_csv(DIR_DATASET + 'grapevine_annotation.csv')

# where the files created in this script are saved
DIR_OUTPUT = 'data/grapevine/paper/test_set_predictions/'

# this parameter is kept fixed for the computation of all metrics
IOU_THRESHOLD = 0.5  # match between box1 and box2 if iou(box1, box2) > IOU_THRESHOLD

# ===== run detection on validation set (no score_threshold) ==========================================================

det = []
for k_image, image_name in enumerate(df_annot[df_annot['dataset'] == 'test']['image_name'].unique()):
    print(image_name)

    obs = df_annot[df_annot['image_name'] == image_name]
    img = cv2.cvtColor(cv2.imread(DIR_DATASET + 'image_test/' + image_name), cv2.COLOR_BGR2RGB)

    # score_threshold=0 bc it's necessary to save all predictions to compute AP metric (and select optimal threshold)
    pred = berry_detection(image=img, model=MODEL_DET, score_threshold=0.)

    for _, row in obs.iterrows():
        score = 1
        x, y, w, h = row[['box_x', 'box_y', 'box_w', 'box_h']]
        x, y = x - w / 2, y - h / 2
        det.append(['obs', x, y, x + w, y + h, score, image_name])

    for _, row in pred.iterrows():
        x, y, w, h, score = row
        det.append(['pred', x, y, x + w, y + h, score, image_name])

det = pd.DataFrame(det, columns=['type', 'x1', 'y1', 'x2', 'y2', 'score', 'image_name'])

det.to_csv(DIR_OUTPUT + 'detection.csv', index=False)

# ===== run segmentation on test set (with the selected score_threshold) ==============================================

score_threshold = 0.89  # deduced from the analysis of val_det

seg = []
for k_image, image_name in enumerate(df_annot[df_annot['dataset'] == 'test']['image_name'].unique()):
    print(image_name)

    obs = df_annot[df_annot['image_name'] == image_name]
    img = cv2.cvtColor(cv2.imread(DIR_DATASET + 'image_test/' + image_name), cv2.COLOR_BGR2RGB)

    res_det = berry_detection(image=img, model=MODEL_DET, score_threshold=score_threshold)
    pred = berry_segmentation(image=img, model=MODEL_SEG, boxes=res_det)

    for _, row in obs.iterrows():
        score = 1
        x, y, w, h = row[['box_x', 'box_y', 'box_w', 'box_h']]
        x, y = x - w / 2, y - h / 2
        seg.append(['obs',
                    x, y, x + w, y + h,
                    row['ell_x'], row['ell_y'], row['ell_w'], row['ell_h'], row['ell_a'],
                    score, image_name])

    # len(res_det) >= len(pred) (not always equal) ,so we use score as berry identifiers. It's a hack and it's not very
    # safe, but most of the time all berries have different scores for a grape. if not the case, take the closest one
    for _, row_seg in pred.iterrows():
        s_det = res_det[res_det['score'] == row_seg['score']]
        if len(s_det) > 1:
            d = np.sum(np.abs(np.array(s_det[['x', 'y']]) - np.array(row_seg[['ell_x', 'ell_y']])), axis=1)
            row_det = s_det.iloc[np.argmin(d)]
        else:
            row_det = s_det.iloc[0]

        seg.append(['pred',
                    row_det['x'], row_det['y'], row_det['x'] + row_det['w'], row_det['y'] + row_det['h'],
                    row_seg['ell_x'], row_seg['ell_y'], row_seg['ell_w'], row_seg['ell_h'], row_seg['ell_a'],
                    row_det['score'], image_name])

seg = pd.DataFrame(seg, columns=['type',
                                       'x1', 'y1', 'x2', 'y2',
                                       'ell_x', 'ell_y', 'ell_w', 'ell_h', 'ell_a',
                                       'score', 'image_name'])

seg.to_csv(DIR_OUTPUT + 'segmentation.csv')
