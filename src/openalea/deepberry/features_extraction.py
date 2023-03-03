""" Extract morphology features from the segmented berries """

import cv2
import numpy as np
from scipy.stats import circmean, circstd


def round_up_to_odd(f):
    return int(np.ceil(f) // 2 * 2 + 1)


def berry_features_extraction(image, ellipses, edge_spacing=3, remove_overlap=True, hue_threshold=50):
    """
    Parameters
    ----------
    image : 3D array
        image of a grapevine cluster.
    ellipses : pandas.core.frame.DataFrame
        Each row includes the 5 parameters of one ellipse : 'ell_x', 'ell_y', 'ell_w', 'ell_h', 'ell_a'.
    edge_spacing : int
        Size in pixels of the ellipse's edge which are excluded, when computing color features.
    remove_overlap : bool
        Whether pixels shared by several ellipses are excluded or not, when computing color features.
    hue_threshold : int
        threshold used to compute the color feature "hue_scaled_above". in [0, 180]. The default value (50) was only
        tested for one grape and probably needs to be reassessed. Specifically, it would probably be better to set
        this threshold after a temporal analysis of the other color features.

    Returns
    -------
    pandas.core.frame.DataFrame
        A copy of "ellipses" input, with new feature columns describing berry morphology:
        "hue_scaled" : mean value of ((180 - hue_opencv) - 100) % 180
        "hue_scaled_std" : std value of ((180 - hue_opencv) - 100) % 180
        "hue_scaled_above" : percentage above "hue_threshold" in ((180 - hue_opencv) - 100) % 180
        "area" : ellipse projection area, in px2
        "volume" : berry ellipsoid volume, computed as a spherical equivalent of the projection area, in px3
        "roundness" : ratio between ellipse width and height, in [0, 1].
    """

    res = ellipses.copy()

    # ==== color features =============================================================================================

    hues_mean, hues_std, hues_above = [], [], []

    # this image-array will count the number of ellipses overlapping at each pixel
    ellipses_overlap = np.zeros((image.shape[0], image.shape[1]))

    vignettes_data = []
    for _, row in ellipses.iterrows():
        x, y, w, h, a = list(row[['ell_x', 'ell_y', 'ell_w', 'ell_h', 'ell_a']])
        e = min(int(w / 4), edge_spacing)  # to not divide w and h by more than 2. (w is used because w <= h)
        w2, h2 = w - (2 * e), h - (2 * e)
        vig_size = round_up_to_odd(h2)

        # border conditions: reduce vig_size value if the corresponding vignette goes beyond the image edges
        vig_size = min(vig_size, int(2 * min((y, image.shape[0] - y))), int(2 * min((x, image.shape[1] - x))))

        vig_x, vig_y = int(x - int(vig_size / 2)), int(y - int(vig_size / 2))  # top left vignette corner
        vignette = image[vig_y:(vig_y + vig_size), vig_x:(vig_x + vig_size)]

        # it's much faster to run cv2.ellipse() on a small subset of the image
        vignette_mask = cv2.ellipse(np.float32(vignette[:, :, 0] * 0),
                                    (round(x - vig_x), round(y - vig_y)),
                                    (round(w2 / 2), round(h2 / 2)),
                                    a, 0., 360, (1), -1)

        ellipses_overlap[vig_y:(vig_y + vig_size), vig_x:(vig_x + vig_size)][vignette_mask == 1] += 1

        vignettes_data.append([vig_x, vig_y, vig_size, vignette, vignette_mask])

    for (vig_x, vig_y, vig_size, vignette, vignette_mask) in vignettes_data:
        pixels = vignette[vignette_mask == 1]
        if remove_overlap:
            vignette_overlap = ellipses_overlap[vig_y:(vig_y + vig_size), vig_x:(vig_x + vig_size)]
            px_nonoverlap = vignette[(vignette_mask == 1) & (vignette_overlap == 1)]
            pixels = px_nonoverlap if len(px_nonoverlap) / len(pixels) > 0.2 else pixels

        pixels_hsv = cv2.cvtColor(np.array([pixels]), cv2.COLOR_RGB2HSV)[0]
        pixels_hue = np.array(pixels_hsv[:, 0]).astype(int)  # np.uint8 -> int to avoid strange behavior after
        pixels_hue_rescaled = ((180 - pixels_hue) - 100) % 180

        hues_mean.append(circmean(pixels_hue_rescaled, low=0, high=180))
        hues_std.append(circstd(pixels_hue_rescaled, low=0, high=180))
        hues_above.append(np.sum(pixels_hue_rescaled > hue_threshold) / len(pixels_hue_rescaled))

    res['hue_scaled'] = hues_mean
    res['hue_scaled_std'] = hues_std
    res['hue_scaled_above'] = hues_above

    # ===== size / shape features =====================================================================================

    res['area'] = (res['ell_w'] / 2) * (res['ell_h'] / 2) * np.pi
    res['volume'] = (4 / 3) * np.pi * ((np.sqrt(res['area'] / np.pi)) ** 3)
    res['roundness'] = res['ell_w'] / res['ell_h']  # w <= h

    return res
