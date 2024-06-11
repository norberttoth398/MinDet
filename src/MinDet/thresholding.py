import matplotlib.pyplot as plt
import numpy as np
import glob
import cv2
import os
from PIL import Image
Image.MAX_IMAGE_PIXELS = 100000000000


def threshold_labs(pred, thresh = 0, save = False):
    """Quick and easy function to filter out results based on the saved detection scores for each proposed object.

    Args:
        pred (ndarray): name of the segmentation output data - assumed to be output of this software
        thresh (int, optional): Threshold for filtering low scores; between 0 and 1. Defaults to 0.
        save (bool, optional): Set to True if image is to be saved. Defaults to False.

    Returns:
        _type_: _description_
    """
    #assume here the output of this software is used (the .npz file)
    pred_data = np.load(pred, allow_pickle=True)
    img = pred_data["img"]
    scores = pred_data["scores"].item()
    #find label values with score below threshold
    filter_out = {k:v for k, v in scores.item() if v < thresh}
    #remove those label values
    for item in filter_out.keys():
        img[img == item] = 0

    if save == True:
        plt.imsave("pred_img.png", img, dpi = 300)
    else:
        pass
    
    return img

def build_img(orig_img, pred, thresh = 0):
    """Function to plot image with its predicted labels as a translucent mask on top.

    Args:
        orig_img (ndarrau): Image used for inference.
        pred (ndarray): Predicted label image.
        thresh (int, optional): Threshold for filtering low scores; between 0 and 1. Defaults to 0.

    Returns:
        _type_: _description_
    """
    #load original img
    original_img = plt.imread(orig_img)
    pred_labels = threshold_labs(pred, thresh)
    fig, ax = plt.subplots((1,1), figsize = (12,12))
    ax.imshow(original_img)
    mask = pred_labels > 0
    ax.imshow(pred_labels, alpha = mask*0.5)

    fig.savefig(orig_img + "_predicted_thresh" + str(thresh))
    return None