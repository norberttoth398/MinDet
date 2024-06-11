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

def create_label_image(result,img_side, score_thresh = 0):
    """Creates label image from list of results

    Args:
        result (list/ndarray): direct result of inference by model
        img_side (int): size of square image.
        score_thresh (int, optional): Threshold for filtering low scores; between 0 and 1. Defaults to 0.

    Returns:
        label_img: image showing position of segmented crystals
        used_scores: list of scores used in label_img
    """
    #create label image from result list (pre or post NMS)
    label_img = np.zeros((img_side,img_side))
    used_scores = []
    for i in range(len(result[1][0])):
        if result[0][0][i,4] < score_thresh:
            pass
        else:
            pred_mask = result[1][0][i]
            label_img_mask = label_img > 0
            #label_img_mask = label_img_mask.astype("int")

            new_lblImg = label_img_mask.astype("int") + pred_mask.astype("int")*2
            label_img[new_lblImg == 2] = i+1
            #scores[i] is for label_number = i+1 annoyingly as 0 is unclassified
            used_scores.append(result[0][0][i,4])

    return label_img, used_scores

def load_imgs(path, img_side ,grid = (6,5), score_thresh = 0):
    """define relations between each tile for loading and post-processing
    """
    

    n_tiles = grid[0]*grid[1]
    tile_list = []
    score_list = []

    for n in range(n_tiles):
        data = np.load(path + "/full_image_" + str(int(n/grid[1])) + "_" + str(int(n%grid[1])) + ".jpg.npz")
        res = [data["bb"], data["mask"]]
        lImage, scores = create_label_image(res, img_side, score_thresh)
        tile_list.append(lImage)
        score_list.append(scores)

    return tile_list, score_list


def replace_vals(img1, img2, overlapped, corner_x, corner_y):
    """ Performs the stitching procedure where images are overlapping.
    """
    coords = np.where(overlapped == 2)
    used_vals = []

    for i in range(len(coords[0])):
        coord = [coords[0][i], coords[1][i]]

        if img1[corner_y + coord[0], coord[1]+corner_x] == img2[coord[0], coord[1]]:
            new_val = img1[corner_y + coord[0], coord[1]+corner_x]
        else:
            if img2[coord[0], coord[1]] in used_vals:
                new_val = img2[coord[0], coord[1]]
                old_val = img1[coord[0] + corner_y, coord[1]+corner_x]
                img1[img1 == img1[coord[0] + corner_y, coord[1]+corner_x]] = new_val
                img2[img2 == old_val] = new_val
            else:
                new_val = img1[coord[0] + corner_y, coord[1]+corner_x]
                img2[img2 == img2[coord[0], coord[1]]] = new_val

        used_vals.append(new_val)

    return img1, img2


def left_overlap(i1, i2, corner_y, corner_x, over_n = 100, img_side = 2000):
    """ Wrapper for stitching function when building row of tile images.
    """
    o1 = i1[corner_y:corner_y+img_side, corner_x:corner_x+over_n] #tile image
    o2 = i2[:,:over_n]

    mask1 = o1 != 0
    mask2 = o2 != 0
    overlapped = mask1.astype("int") + mask2.astype("int")

    i1, i2 = replace_vals(i1,i2,overlapped, corner_x, corner_y)

    overlapped[overlapped == 0] = 1
    i1[corner_y:corner_y + img_side, corner_x:corner_x+img_side] += i2
    i1[corner_y:corner_y + img_side, corner_x:corner_x+over_n] = i1[corner_y:corner_y + img_side, corner_x:corner_x+over_n]/overlapped

    return i1, i2

def top_overlap(i1, i2, corner_y, corner_x, over_n = 100, img_side = 2000, row_len = 9600):
    """ Wrapper for stitching function when building columns (may be deprecated)
    """
    o1 = i1[corner_y:corner_y+over_n, corner_x:corner_x+img_side] #tile image
    o2 = i2[:over_n,:]

    mask1 = o1 != 0
    mask2 = o2 != 0
    overlapped = mask1.astype("int") + mask2.astype("int")

    i1, i2 = replace_vals(i1,i2,overlapped, corner_x, corner_y)

    overlapped[overlapped == 0] = 1
    i1[corner_y:corner_y + img_side, corner_x:corner_x+img_side] += i2
    i1[corner_y:corner_y + over_n, corner_x:corner_x+img_side] = i1[corner_y:corner_y + over_n, corner_x:corner_x+img_side]/overlapped

    return i1, i2

def top_overlap_row(i1, i2, corner_y, corner_x, over_n = 100, img_side = 2000, row_len = 9600):
    """ Wrapper for stitching function when building full image from rows.
    """
    o1 = i1[corner_y:corner_y+over_n, corner_x:corner_x+row_len] #tile image
    o2 = i2[:over_n,:]

    mask1 = o1 != 0
    mask2 = o2 != 0
    overlapped = mask1.astype("int") + mask2.astype("int")
    i1, i2 = replace_vals(i1,i2,overlapped, corner_x, corner_y)
 
    overlapped[overlapped == 0] = 1
    i1[corner_y:corner_y + img_side, corner_x:corner_x+row_len] += i2
    i1[corner_y:corner_y + over_n, corner_x:corner_x+row_len] = i1[corner_y:corner_y + over_n, corner_x:corner_x+row_len]/overlapped

    return i1, i2

def score_update(scores, vals, score_dict, filler = 0):
    """Updates score dictionary prior to left_overlap functions.
    """
    for i in range(len(vals)):
        score_dict[vals[i]] = scores[i]

    return score_dict

def post_tile_score_update(old_img, new_img, dictionary):
    """Update score dictionary post-tiling.
    """
    label_change = old_img - new_img
    changed_mask = label_change != 0
    old_labels = old_img[changed_mask]
    new_labels = old_img[changed_mask]
    for l in np.unique(new_labels):
        l_mask = new_labels == l
        replaced = np.unique(old_labels[l_mask])
        s = []
        s.append(dictionary)
        for val in replaced:
            s.append(dictionary[val])
        max_index = s.index(max(s))
        if max_index != 0:
            dictionary[l] = max(s)
        else:
            pass

    return dictionary

def row_score_update(row_dict,overall_dict, row_filler):
    """Update score dictionary for overall scores based on the separate row_dictionary produced
    using the left_overlap phase.
    """
    keys = row_dict.keys()
    for k in keys:
        overall_dict[k+row_filler] = row_dict[k]

    return overall_dict

def tile_run(path,grid, orig_shape, img_side, over_n, score_thresh = 0):
    """Main wrapper function to start the tile stitching process for the overall image from individual tiles post-inference.

    Args:
        path (str): path with all the data in it - should be same as that used previously for inference.
        grid ((int, int)): (n,m) values from function that sliced images
        orig_shape ((int, int)): Original shape of the image inference was performed on.
        img_side (int): Size of square tile image used - length of side.
        over_n (int): Overlap between tiles in pixels

    Returns:
        tiled_img (ndarray): Final tiled image showing all the labels.
        final_scores (dict): Final dictionary of all the detection scores.
    """
    tiled_img = np.zeros(orig_shape[:2])
    row_len = orig_shape[1]
    #scores[0] is for label == 1 for future reference
    overall_scores = {}
    tiled, scores = load_imgs(path, img_side, grid, score_thresh)
    for j in range(grid[0]):
        
        tiled_row = np.zeros((img_side, orig_shape[1]))
        row_scores = {}
        for i in range(grid[1]):
            
            current_tile_end = (i+1)*(img_side - over_n) + over_n
            if current_tile_end > orig_shape[1]:
                excess = current_tile_end - orig_shape[1]
            else:
                excess = 0

            #print(i)
            x = i#horizontal        

            #set top left corner at which they overlap
            corner_x = x*(img_side-over_n) - excess

            if i == 0:
                tiled_row[0:img_side, corner_x:corner_x + img_side] += tiled[j*grid[1]]
                row_scores = score_update(scores[j*grid[1]], np.unique(tiled[j*grid[1]])[1:],row_scores)
            else:
                #make sure no labels are the same across the two images
                filler_label = np.unique(tiled_row)[-1]
                new_img = tiled[(grid[1]*j)+i] + filler_label
                new_img[new_img == filler_label] = 0
                row_scores = score_update(scores[(grid[1]*j)+i], np.unique(new_img)[1:], row_scores, filler=filler_label)

                tiled_row, n_img = left_overlap(tiled_row, new_img, 0, corner_x, over_n=over_n + excess ,img_side=img_side)
                #update score dictionary
                row_scores = post_tile_score_update(new_img, n_img, row_scores)
        
        
        current_row_end = (j+1)*(img_side - over_n) + over_n
        if current_row_end > orig_shape[0]:
            row_excess = current_row_end - orig_shape[0]
        else:
            row_excess = 0

        y = j#vertical
        corner_y = y*(img_side - over_n) - row_excess

        if j == 0:
            tiled_img[corner_y:corner_y+img_side, 0:row_len] += tiled_row
            #just copy the existing dictionary
            overall_scores = row_scores.copy()
        else:
            row_filler_label = np.unique(tiled_img)[-1]
            tiled_row = tiled_row + row_filler_label
            tiled_row[tiled_row == row_filler_label] = 0
            #add row score in to overall dict
            overall_scores = row_score_update(row_scores, overall_scores, row_filler_label)
            #do tiling
            tiled_img, n_img = top_overlap_row(tiled_img, tiled_row, corner_y, 0, over_n=over_n+row_excess ,img_side = img_side, row_len=row_len)
            #update score dictionary
            overall_scores = post_tile_score_update(tiled_row, n_img, overall_scores)

    #optimizing label assignment
    n_unique = np.unique(tiled_img)
    print(n_unique.size)
    print(len(overall_scores.keys()))
    final_scores = {}
    for i in range(n_unique.size):
        tiled_img[tiled_img == n_unique[i]] = i
        if i == 0:
            pass
        else:
            final_scores[i] = overall_scores[n_unique[i]]

    return tiled_img, final_scores