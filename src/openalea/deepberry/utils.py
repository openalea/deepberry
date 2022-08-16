import cv2
import numpy as np
import pandas as pd

# ===== cache (benoit) ==============================================================================================


def get_image_path(index, plantid, task, angle, disk='V:/'):
    row = index[(index['plantid'] == plantid) & (index['taskid'] == task) & (index['imgangle'] == angle)].iloc[0]
    path = disk + '{}/{}/{}.png'.format(row['exp'], row['taskid'], row['imgguid'])
    return path

# ===== image data augmentation =====================================================================================


def hsv_variation(img, variation=[20, 20, 50]):
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)

    for i, var in enumerate(variation):
        value = np.random.randint(-var, var)
        if value >= 0:
            lim = 255 - value
            hsv[:, :, i][hsv[:, :, i] > lim] = 255
            hsv[:, :, i][hsv[:, :, i] <= lim] += value
        else:
            lim = 0 - value
            hsv[:, :, i][hsv[:, :, i] < lim] = 0
            hsv[:, :, i][hsv[:, :, i] >= lim] -= - value

    img = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
    return img


def adjust_contrast_brightness(img, contrast:float=1.0, brightness:int=0):
    """
    Adjusts contrast and brightness of an uint8 image.
    contrast:   (0.0,  inf) with 1.0 leaving the contrast as is
    brightness: [-255, 255] with 0 leaving the brightness as is
    """
    brightness += int(round(255*(1-contrast)/2))
    return cv2.addWeighted(img, contrast, img, 0, brightness)

# ===== ellipse stuff ================================================================================================


def ellipse_interpolation(x, y, w, h, a, n_points=100):
    """
    Return a number n_points  of coordinates x,y for the ellipse of parameters x,y,w,h,a
    (a is in degrees)
    """
    lsp = np.linspace(0, 2 * np.pi, n_points)
    ell = np.array([w / 2 * np.cos(lsp), h / 2 * np.sin(lsp)])
    a_rad = a * np.pi / 180
    r_rot = np.array([[np.cos(a_rad), -np.sin(a_rad)], [np.sin(a_rad), np.cos(a_rad)]])
    rot = np.zeros((2, ell.shape[1]))
    for i in range(ell.shape[1]):
        rot[:, i] = np.dot(r_rot, ell[:, i])
    points_x, points_y = x + rot[0, :], y + rot[1, :]
    return points_x, points_y

# ===== bounding box stuff ===========================================================================================


# TODO : copy from deepcollar. import it differently ?
def get_iou(bb1, bb2):
    """
    Generic function to calculate the Intersection over Union (IoU) of two bounding boxes.

    Parameters
    ----------
    bb1 : dict
        Keys: {'x1', 'x2', 'y1', 'y2'}
        The (x1, y1) position is at the top left corner,
        the (x2, y2) position is at the bottom right corner
    bb2 : dict
        Keys: {'x1', 'x2', 'y1', 'y2'}
        The (x, y) position is at the top left corner,
        the (x2, y2) position is at the bottom right corner

    Returns
    -------
    float
        in [0, 1]
    """
    assert bb1['x1'] < bb1['x2']
    assert bb1['y1'] < bb1['y2']
    assert bb2['x1'] < bb2['x2']
    assert bb2['y1'] < bb2['y2']

    # determine the coordinates of the intersection rectangle
    x_left = max(bb1['x1'], bb2['x1'])
    y_top = max(bb1['y1'], bb2['y1'])
    x_right = min(bb1['x2'], bb2['x2'])
    y_bottom = min(bb1['y2'], bb2['y2'])

    if x_right < x_left or y_bottom < y_top:
        return 0.0

    # The intersection of two axis-aligned bounding boxes is always an
    # axis-aligned bounding box
    intersection_area = (x_right - x_left) * (y_bottom - y_top)

    # compute the area of both AABBs
    bb1_area = (bb1['x2'] - bb1['x1']) * (bb1['y2'] - bb1['y1'])
    bb2_area = (bb2['x2'] - bb2['x1']) * (bb2['y2'] - bb2['y1'])

    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = intersection_area / float(bb1_area + bb2_area - intersection_area)
    assert iou >= 0.0
    assert iou <= 1.0
    return iou


# TODO : copy from deepcollar. import it differently ?
def nms(df, nms_threshold):
    """
    Apply nms algorithm to remove redundant predictions.
    /!\ might be slow if there are lots of rows...
    df : dataframe with at least columns ['x', 'y', 'w', 'h', 'score']
    """

    res = df.copy()

    res_nms = pd.DataFrame(columns=res.columns)
    while len(res) != 0:

        res = res.sort_values('score', ascending=False)
        row1 = res.iloc[0]  # prediction with best score
        res_nms = pd.concat((res_nms, pd.DataFrame([row1])))  # add this row in the new df (without .append bc deprecated)
        res = res.drop(res.index[0])  # remove this row in the old df

        box1 = {'x1': row1['x'], 'x2': row1['x'] + row1['w'], 'y1': row1['y'], 'y2': row1['y'] + row1['h']}
        to_remove = []
        for row_index, row2 in res.iterrows():
            box2 = {'x1': row2['x'], 'x2': row2['x'] + row2['w'], 'y1': row2['y'], 'y2': row2['y'] + row2['h']}
            iou = get_iou(box1, box2)
            if iou > nms_threshold:
                to_remove.append(row_index)
        res = res.drop(to_remove)

    return res_nms

