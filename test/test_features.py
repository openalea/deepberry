import numpy as np
import pandas as pd

from openalea.deepberry.features_extraction import berry_features_extraction


def test_features():

    n = 100
    h = np.random.uniform(20, 80, n)
    ellipses = pd.DataFrame({'ell_x': np.random.uniform(100, 400, n),
                             'ell_y': np.random.uniform(100, 400, n),
                             'ell_w': h * np.random.uniform(0.5, 1, n),
                             'ell_h': h,
                             'ell_a': np.random.random(n) * 180})

    image = np.random.randint(0, 255, (500, 500, 3), dtype=np.uint8)

    res = berry_features_extraction(image=image, ellipses=ellipses)

    assert set(res.columns) == {'ell_x', 'ell_y', 'ell_w', 'ell_h', 'ell_a', 'hue_scaled',
                                'hue_scaled_std', 'hue_scaled_above', 'area', 'volume', 'roundness'}
