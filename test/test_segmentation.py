from openalea.deepberry.ellipse_segmentation import load_berry_models

from ..examples.datadir import datadir
print(datadir)

def test_load_model():
    model_detection, model_segmentation = load_berry_models(datadir + '/model')
