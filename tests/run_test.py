from MinDet import run
import numpy as np

def test_run_pad_img():
    img = np.zeros((100,100,3), dtype="uint8")
    img_side = 200

    imgr = run.pad_img(img, img_side)

    assert imgr.shape[0] == 200 and imgr.shape[1] == 200