import numpy as np

from openalea.deepberry.utils import ellipse_interpolation, iou_between_polygons, nms


def test_ellipse_interpolation():

    n = 20
    ell = ellipse_interpolation(50.5, 49.5, 5.2, 6.3, 150.2, n_points=n)

    assert ell.shape == (2, n)


def test_iou_box():

    p1 = np.array([[10, 10, 20, 20, 10], [30, 40, 40, 30, 30]]).T
    p2 = np.array([[10, 10, 20, 20, 10], [30, 50, 50, 30, 30]]).T
    p3 = np.array([[10, 10, 20, 20, 10], [40, 50, 50, 40, 40]]).T

    iou0 = iou_between_polygons(p1, p3)
    iou05 = iou_between_polygons(p1, p2)
    iou1 = iou_between_polygons(p1, p1)

    assert round(iou0, 3) == 0.
    assert round(iou05, 3) == 0.5
    assert round(iou1, 3) == 1.


def test_iou_ellipse():

    ell1 = ellipse_interpolation(5, 5, 2, 2, 90)
    ell2 = ellipse_interpolation(50, 50, 2, 2, 90)

    p1 = ell1.T
    p2 = ell2.T

    iou0 = iou_between_polygons(p1, p2)
    iou1 = iou_between_polygons(p1, p1)

    assert round(iou0, 3) == 0.
    assert round(iou1, 3) == 1.


def test_nms():

    n = 50
    ellipses = [ellipse_interpolation(x, y, w, h, 90, n_points=10).T for x, y, w, h in np.random.random((n, 4))]
    scores = np.random.random(n)

    nms0 = nms(polygons=ellipses, scores=scores, threshold=0.)
    nms05 = nms(polygons=ellipses, scores=scores, threshold=0.5)
    nms1 = nms(polygons=ellipses, scores=scores, threshold=1.)

    assert len(nms0) <= len(nms05) <= len(nms1)
    assert len(nms1) == n
