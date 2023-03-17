import os
import cv2
import pytest
import numpy as np
import pandas as pd

from openalea.deepberry.training.training_dataset import generate_detection_instance, \
    generate_segmentation_instance
from openalea.deepberry.ellipse_segmentation import VIGNETTE_SIZE_DET, VIGNETTE_SIZE_SEG


@pytest.fixture
def image():
    return np.random.randint(0, 255, (700, 700, 3), dtype=np.uint8)


def test_det_instance(image):

    n = 5
    labels = pd.DataFrame({'box_x': np.random.uniform(100, 400, n),
                           'box_y': np.random.uniform(100, 400, n),
                           'box_w': np.random.uniform(20, 60, n),
                           'box_h': np.random.uniform(20, 60, n)})

    filename = 'tmp'
    _ = generate_detection_instance(image=image, labels=labels, dirsave=filename, target_center=None, shift_center=10)

    vignette = cv2.cvtColor(cv2.imread(filename + '.png'), cv2.COLOR_BGR2RGB)

    assert vignette.shape == (VIGNETTE_SIZE_DET, VIGNETTE_SIZE_DET, 3)

    # remove temporary files
    os.remove(filename + '.png')
    os.remove(filename + '.txt')


def test_seg_instance(image):

    # box and ell parameters are not coherent here, but it's not a problem for the test
    labels = pd.Series({'box_x': np.random.uniform(100, 400),
                        'box_y': np.random.uniform(100, 400),
                        'box_w': np.random.uniform(20, 60),
                        'box_h': np.random.uniform(20, 60),
                        'ell_x': np.random.uniform(100, 400),
                        'ell_y': np.random.uniform(100, 400),
                        'ell_w': np.random.uniform(20, 60),
                        'ell_h': np.random.uniform(20, 60),
                        'ell_a': np.random.uniform(0, 180)})

    filename = 'tmp'
    _ = generate_segmentation_instance(image=image, label=labels, dirsave=filename, center_noise=0.1, scale_noise=0.1)

    x = cv2.cvtColor(cv2.imread(filename + 'x.png'), cv2.COLOR_BGR2RGB)

    assert x.shape == (VIGNETTE_SIZE_SEG, VIGNETTE_SIZE_SEG, 3)

    y = cv2.imread(filename + 'y.png', 0)

    assert y.shape == (VIGNETTE_SIZE_SEG, VIGNETTE_SIZE_SEG)
    assert set(np.unique(y)) == {0, 255}

    # remove temporary files
    os.remove(filename + 'x.png')
    os.remove(filename + 'y.png')









