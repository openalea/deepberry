import os
import cv2
import pytest
import numpy as np

from openalea.deepberry.ellipse_segmentation import load_berry_models, VIGNETTE_SIZE_DET, \
    berry_detection, VIGNETTE_SIZE_SEG, berry_segmentation

pkg_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
datadir = os.path.join(pkg_dir, 'examples', 'data')


@pytest.fixture
def models():
    model_detection, model_segmentation = load_berry_models(datadir + '/model')
    return model_detection, model_segmentation


@pytest.fixture
def image():
    image = cv2.cvtColor(cv2.imread(datadir + '/image/image.png'), cv2.COLOR_BGR2RGB)
    return image


def test_load_model(models):

    model_detection, model_segmentation = models

    assert type(model_detection).__name__ in ['DetectionModel', 'dnn_DetectionModel']  # may differ depending on
    # cv2 version
    assert type(model_segmentation).__name__ == 'Functional'  # keras


def test_run_detection_random_vignette(models):

    model_detection, _ = models
    model_detection.setInputParams(size=(VIGNETTE_SIZE_DET, VIGNETTE_SIZE_DET), scale=1 / 255, swapRB=False)

    vignette = np.random.randint(0, 255, (VIGNETTE_SIZE_DET, VIGNETTE_SIZE_DET, 3), dtype=np.uint8)

    _, _, _ = model_detection.detect(vignette, confThreshold=0.5, nmsThreshold=0.5)


def test_run_segmentation_random_vignette(models):

    vignette = np.random.randint(0, 255, (VIGNETTE_SIZE_SEG, VIGNETTE_SIZE_SEG, 3))

    _, model_segmentation = models

    res = model_segmentation.predict(np.array([vignette]), verbose=0)

    assert res.shape == (1, VIGNETTE_SIZE_SEG, VIGNETTE_SIZE_SEG, 2)
    assert np.all(res >= 0.)
    assert np.all(res <= 1.)


def test_det_and_seg_real_image(models, image):

    model_detection, model_segmentation = models

    image_small = image[:1000, :1000, :]

    det = berry_detection(image=image_small, model=model_detection)

    assert type(det).__name__ == 'DataFrame'
    assert set(det.columns) == {'x', 'y', 'w', 'h', 'score'}

    seg = berry_segmentation(image=image_small, model=model_segmentation, boxes=det)

    assert type(seg).__name__ == 'DataFrame'
    assert set(seg.columns) == {'ell_x', 'ell_y', 'ell_w', 'ell_h', 'ell_a', 'score'}
    assert np.all(seg['ell_w'] <= seg['ell_h'])

    assert len(seg) <= len(det)
