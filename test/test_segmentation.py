from deepberry.src.openalea.deepberry.ellipse_segmentation import load_berry_models


def test_load_model():
    model_detection, model_segmentation = load_berry_models('deepberry/examples/data/model')