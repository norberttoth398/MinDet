from mmdet.apis import init_detector, inference_detector
import mmcv
import torch
import matplotlib.pyplot as plt
import numpy as np
import glob
import cv2
import os

def nms_remove(result, nms_res, crit = 0.5):
    post_nms_bb = np.asarray(result[0].copy())
    post_nms_mask = np.asarray(result[1].copy())
    #remove unwanted entries in results array
    del_list = np.where(np.asarray(nms_res) > crit)
    del_list = np.asarray(del_list).astype("int64")
    for item in sorted(del_list, reverse=True):
        post_nms_mask = np.delete(post_nms_mask, item, 1)
        post_nms_bb = np.delete(post_nms_bb, item, 1)

    return (post_nms_bb, post_nms_mask)

def mask_nms(seg_masks, cate_labels, sum_masks = None):
    """Mask NMS for multi-class masks.
    Args:
        seg_masks (Tensor): shape (n, h, w)
        cate_labels (Tensor): shape (n), mask labels in descending order
        cate_scores (Tensor): shape (n), mask scores in descending order
        kernel (str):  'linear' or 'gauss' 
        sigma (float): std in gaussian method
        sum_masks (Tensor): The sum of seg_masks
    Returns:
        Tensor: cate_scores_update, tensors of shape (n)
    """
    n_samples = len(cate_labels)
    if n_samples == 0:
        return []
    if sum_masks is None:
        sum_masks = seg_masks.sum((1, 2)).float()
    seg_masks = seg_masks.reshape(n_samples, -1).float()
    # inter.
    inter_matrix = torch.mm(seg_masks, seg_masks.transpose(1, 0))
    # union.
    sum_masks_x = sum_masks.expand(n_samples, n_samples)
    # iou.
    iou_matrix = (inter_matrix / (sum_masks_x + sum_masks_x.transpose(1, 0) - inter_matrix)).triu(diagonal=1)
    # label_specific matrix.
    cate_labels_x = cate_labels.expand(n_samples, n_samples)
    label_matrix = (cate_labels_x == cate_labels_x.transpose(1, 0)).float().triu(diagonal=1)

    # IoU compensation
    compensate_iou, _ = (iou_matrix * label_matrix).max(0)
        
    return compensate_iou
