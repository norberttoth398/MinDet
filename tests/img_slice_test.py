from MinDet import run, slicing
import numpy as np

def test_img_slice():
    img = np.zeros((1000,1000,3), dtype="uint8")
    img_side = 200

    n,m = slicing.img_slice(img, ".", img_side)
    assert n == 10, m == 10