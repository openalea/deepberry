import numpy as np

from shapely.geometry import Polygon


def ellipse_interpolation(x, y, w, h, a, n_points=100):
    """
    Return a number n_points of coordinates x',y' for the ellipse of parameters x,y,w,h,a (a is in degrees).
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


def iou_between_polygons(polygon1, polygon2):
    """
    Computes the intersection over union (IoU) between two polygons
    https://stackoverflow.com/questions/58435218/intersection-over-union-on-non-rectangular-quadrilaterals
    """

    # Defines each polygon
    polygon1_shape = Polygon(polygon1)
    polygon2_shape = Polygon(polygon2)

    # Calculates intersection and union, and the IoU
    polygon_intersection = polygon1_shape.intersection(polygon2_shape).area
    polygon_union = polygon1_shape.area + polygon2_shape.area - polygon_intersection
    return polygon_intersection / polygon_union


def nms(polygons, scores, threshold):
    """
    Non-maximum suppression algorithm

    Parameters
    ----------
    polygons : array
        array of polygons
    scores : array
        array containing the confidence score associated to each polygon
    threshold : float
        The algorithm aims to avoid having polygons with an IoU above this threshold. in [0,1].

    Returns
    -------
    list
        indexes of the remaining polygons after applying the algorithm
    """

    to_keep = []
    order = np.argsort(scores)[::-1]

    while len(order) > 0:
        i_max = order[0]
        to_keep.append(i_max)
        ious = np.array([iou_between_polygons(polygons[i_max], polygons[i]) for i in order])
        order = order[np.where(ious < threshold)[0]]

    return to_keep
