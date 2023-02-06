import numpy as np
import pandas as pd

from shapely.geometry import Polygon

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
    return np.array([points_x, points_y])

# ===== bounding box stuff ===========================================================================================


def generic_iou(polygon1, polygon2):
    """
    https://stackoverflow.com/questions/58435218/intersection-over-union-on-non-rectangular-quadrilaterals
    """

    # Define each polygon
    polygon1_shape = Polygon(polygon1)
    polygon2_shape = Polygon(polygon2)

    # Calculate intersection and union, and the IOU
    polygon_intersection = polygon1_shape.intersection(polygon2_shape).area
    polygon_union = polygon1_shape.area + polygon2_shape.area - polygon_intersection
    return polygon_intersection / polygon_union


def nms(df, nms_threshold, ellipse=False):
    """
    # TODO
    Apply nms algorithm to remove redundant predictions.
    /!\ might be slow if there are lots of rows...
    df : dataframe with at least columns ['x', 'y', 'w', 'h', 'score']
    """

    res = df.copy()
    res['id'] = np.arange(len(res))

    polygons = {}
    for _, row in res.iterrows():
        if ellipse:
            xe, ye, we, he, ae = row[['ell_x', 'ell_y', 'ell_w', 'ell_h', 'ell_a']]
            polygons[row['id']] = ellipse_interpolation(x=xe, y=ye, w=we, h=he, a=ae, n_points=30).T
        else:
            x, y, w, h = row[['x', 'y', 'w', 'h']]
            polygons[row['id']] = np.array([[x, x, x + w, x + w, x], [y, y + h, y + h, y, y]]).T

    res_nms = pd.DataFrame(columns=res.columns)
    while len(res) != 0:

        res = res.sort_values('score', ascending=False)
        row1 = res.iloc[0]  # prediction with best score
        res_nms = pd.concat((res_nms, pd.DataFrame([row1])))  # add row to new df (without .append bc deprecated)
        res = res.drop(res.index[0])  # remove this row in the old df

        to_remove = []
        for row_index, row2 in res.iterrows():

            iou = generic_iou(polygons[row1['id']], polygons[row2['id']])
            if iou > nms_threshold:
                to_remove.append(row_index)

        res = res.drop(to_remove)

    del res_nms['id']
    return res_nms
