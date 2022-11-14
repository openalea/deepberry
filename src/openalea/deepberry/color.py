import cv2
import numpy as np
from scipy.stats import circmean


def mean_hue_berry(image, ellipses, edge_spacing=3, remove_overlap=True):
    """
    edge_spacing : number of pixels removed along ellipse edges
    """

    hues = []
    masks = []
    res = ellipses.copy()
    for _, row in ellipses.iterrows():
        x, y, w, h, a = list(row[['ell_x', 'ell_y', 'ell_w', 'ell_h', 'ell_a']])
        e = min(int(w / 4), edge_spacing)  # to not divide w, h by more than 2. (w is used because w <= h)
        w2, h2 = w - (2 * e), h - (2 * e)
        mask = cv2.ellipse(np.float32(image[:, :, 0] * 0), (round(x), round(y)), (round(w2 / 2), round(h2 / 2)),
                           a, 0., 360, (1), -1)
        masks.append(mask)

    mask_sum = np.sum(masks, axis=0)
    for mask in masks:
        pixels = image[mask == 1]
        if remove_overlap:
            px_nonoverlap = image[(mask == 1) & (mask_sum == 1)]
            pixels = px_nonoverlap if len(px_nonoverlap) / len(pixels) > 0.2 else pixels

        pixels_hsv = cv2.cvtColor(np.array([pixels]), cv2.COLOR_RGB2HSV)[0]
        hue = circmean(pixels_hsv[:, 0], low=0, high=180)
        hues.append(hue)
    res['hue'] = hues

    return res