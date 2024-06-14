from MinDet.Process import mcmc, measure_funcs,  shape_calc, texture
from skimage.draw import ellipse
import matplotlib.pyplot as plt 
import numpy as np

def test_measure_funcs():
    
    rr, cc = ellipse(10, 20, 10, 15)

    img = np.zeros((30, 60), dtype=np.uint8)

    img[rr, cc] = 1

    res1 = measure_funcs.gen_texture_data(img, True, False)
    res2 = measure_funcs.gen_texture_data(img, True, True)
    res3 = measure_funcs.gen_texture_data(img, False, False)
    res4 = measure_funcs.gen_texture_data(img, False, True)

    assert np.around(res1[0],2) == 21.33 and np.around(res1[1],2) == 1.51 and np.around(res2[0],2) == 22.45 and np.around(res2[1],2) == 1.56 and np.around(res3[0],2) == 29.55 and np.around(res3[1],2) == 1.51 and  res4[0] == 28.0 and np.around(res4[1], 2) == 1.56
