from mmdet.apis import init_detector, inference_detector
import mmcv
import torch
import matplotlib.pyplot as plt
import numpy as np
import glob
import cv2
import os
from PIL import Image
Image.MAX_IMAGE_PIXELS = 100000000000

def img_slice(img, dir, img_size = 2000, overlap = 100, slicing = True):
    """Function used to slice the main panorama image into the desired tiles. 
       Images are saved to be used for inference later.

    Args:
        img (ndarray): Panorama image
        dir (str): Directory to save files in.
        img_size (int, optional): Size of tile images in pixels - square tiles assumed. Defaults to 2000.
        overlap (int, optional): Overlap between tiles. Defaults to 100.
        slicing (bool, optional): Set False if you don't want to slice image, but just figure out th (n,m) values. Defaults to True.

    Returns:
        n, m: grid sizes for tiling post-inference.
    """

    n = int((img.shape[0]-overlap)/(img_size - overlap))+1
    m = int((img.shape[1]-overlap)/(img_size - overlap))+1

    step_size = img_size - overlap
    if slicing == True:
        for i in range(n):
            for j in range(m):
                x_start = i*step_size
                y_start = j*step_size
                if (x_start+img_size) > img.shape[0]:
                    x_start = img.shape[0]-img_size
                if (y_start+img_size) > img.shape[1]:
                    y_start = img.shape[1]-img_size
                else:
                    pass

                temp_slice = img[x_start:x_start+img_size,
                                y_start:y_start +img_size,
                                :]
                if os.path.exists(dir + "/imgs") == True:
                    pass
                else:
                    os.makedirs(dir + "/imgs")
                cv2.imwrite( dir + "/imgs/image_" + str(i) + "_" + str(j) + ".jpg", temp_slice, [int(cv2.IMWRITE_JPEG_QUALITY), 100])
    else:
        pass
    return n,m