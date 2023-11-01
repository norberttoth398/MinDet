import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from skimage import measure


#file with the functions used for producing conventional textural plots

#Aspect ratio distributions with size (easy)
#CSD's (need lots of options - eg. geometric/linear binning and different ways data can be treated)
#Zingg plots (relatively easy - just use Mangler type plots probably?)

def remove_zero_elements(counts, bins, bin_centres = None):
    """_summary_

    Args:
        counts (_type_): _description_
        bins (_type_): _description_
        bin_centres (_type_, optional): _description_. Defaults to None.

    Returns:
        _type_: _description_
    """
    bins = np.asarray(bins)
    counts = np.asarray(counts)
    new_bins = bins[:bins.shape[0]-1]
    index = counts >0

    new_bins = new_bins[index]
    counts = counts[index]
    if bin_centres is not None:
        bin_centres = np.asarray(bin_centres)
        bin_centres = bin_centres[index]

        return counts, new_bins, bin_centres
    else:
        return counts, new_bins


def remove_all_zero_elements(counts, bins, bin_centres, bw, counts_bwNorm):
    """_summary_

    Args:
        counts (_type_): _description_
        bins (_type_): _description_
        bin_centres (_type_): _description_
        bw (_type_): _description_
        counts_bwNorm (_type_): _description_

    Returns:
        _type_: _description_
    """
    bins = np.asarray(bins)
    counts = np.asarray(counts)
    new_bins = bins[:bins.shape[0]-1]
    index = counts >0

    new_bins = new_bins[index]
    counts = counts[index]
    bin_centres = np.asarray(bin_centres)
    bin_centres = bin_centres[index]
    bw = np.asarray(bw)
    bw = bw[index]
    counts_bwNorm = np.asarray(counts_bwNorm)
    counts_bwNorm = counts_bwNorm[index]

    return counts, new_bins, bin_centres, bw, counts_bwNorm

def gen_aspect_ratio_data(size, aspect_ratio, bin_edges):
    """Generator for aspect ratio data with given bin edges.

    Args:
        size (ndarray): _description_
        aspect_ratio (ndarray): _description_
        bin_edges (ndarray): _description_

    Returns:
        aspect_ratio_binned
        aspect_ratio_sigma
        bin_centres
    """

    aspect_ratio_binned = []
    aspect_ratio_sigma = []
    bin_centres = []

    for i in range(len(bin_edges)-1):
        n = i+1

        above_lower_edge = size >= bin_edges[i]
        new_size = size[above_lower_edge]
        new_aspect = np.asarray(aspect_ratio)[above_lower_edge]

        below_upper = new_size < bin_edges[n]
        binned_aspect = new_aspect[below_upper]

        aspect_ratio_binned.append(np.mean(binned_aspect))
        aspect_ratio_sigma.append(np.std(binned_aspect))
        bin_centres.append(0.5*(bin_edges[i]+bin_edges[n]))

    return aspect_ratio_binned, aspect_ratio_sigma, bin_centres


def AR_plot(aspect_ratio, size,auto_bins = True, bins = 10, geometric = False, min_max = None, manual_lims = False, x_lims = None, y_lims = None, ax = None):
    """_summary_

    Args:
        aspect_ratio (_type_): _description_
        size (_type_): _description_
        auto_bins (bool, optional): _description_. Defaults to True.
        bins (int, optional): _description_. Defaults to 10.
        geometric (bool, optional): _description_. Defaults to False.
        min_max (_type_, optional): _description_. Defaults to None.
        manual_lims (bool, optional): _description_. Defaults to False.
        x_lims (_type_, optional): _description_. Defaults to None.
        y_lims (_type_, optional): _description_. Defaults to None.

    Returns:
        _type_: _description_
    """

    #if auto_bins is false, bins should be a set of bins, otherwise just an integer; geometric used to define how bins are generated in automatic
    
    #manual_lims allows user to define x and y limits themselves, otherwise will be done automatically by matplotlib

    if min_max is None:
        max_val = np.max(size)
        min_val = np.min(size)
    else:
        max_val = min_max[1]
        min_val = min_max[0]


    if auto_bins == True:
        #generate bins
        if geometric == True:
            #generate them geometrically
            b = np.logspace(min_val, max_val, num = bins)
        else:
            #generate linear bins
            b = np.linspace(min_val, max_val, num = bins)
    else:
        b = bins

    aspect_ratio_binned, aspect_ratio_sigma, centres = gen_aspect_ratio_data(size, aspect_ratio, b)
    if ax is not None:
        ax.scatter(bin_centres, aspect_ratio_binned)
        ax.errorbar(bin_centres, aspect_ratio_binned, yerr=aspect_ratio_sigma, fmt = 'o')
        return None
    else:
        fig, ax = plt.subplots(1,1,figsize = (8,8))
        ax.scatter(bin_centres, aspect_ratio_binned)
        ax.errorbar(bin_centres, aspect_ratio_binned, yerr=aspect_ratio_sigma, fmt = 'o')
        return fig, ax

def CSD_plot(size,auto_bins = True, bins = 10, geometric = False, min_max = None, manual_lims = False, x_lims = None, y_lims = None, ax = None):
    """_summary_

    Args:
        size (_type_): _description_
        auto_bins (bool, optional): _description_. Defaults to True.
        bins (int, optional): _description_. Defaults to 10.
        geometric (bool, optional): _description_. Defaults to False.
        min_max (_type_, optional): _description_. Defaults to None.
        manual_lims (bool, optional): _description_. Defaults to False.
        x_lims (_type_, optional): _description_. Defaults to None.
        y_lims (_type_, optional): _description_. Defaults to None.

    Returns:
        _type_: _description_
    """
    if min_max is None:
        max_val = np.max(size)
        min_val = np.min(size)
    else:
        max_val = min_max[1]
        min_val = min_max[0]


    if auto_bins == True:
        #generate bins
        if geometric == True:
            #generate them geometrically
            b = np.logspace(min_val, max_val, num = bins)
        else:
            #generate linear bins
            b = np.linspace(min_val, max_val, num = bins)
    else:
        b = bins

    counts, bin_edges = np.histogram(size, bins)
    bw = [bin_edges[i+1] - bin_edges[i] for i in range(len(bin_edges)-1)]
    counts_bwNorm = [counts[i]/bw[i] for i in range(len(counts))]

    centres = [0.5*(bin_edges[i]+bin_edges[i+1]) for i in range(len(bin_edges)-1)]

    counts, bin_edges, centres, bw, counts_bwNorm = remove_all_zero_elements(counts, bin_edges, centres, bw, counts_bwNorm)

    if ax is not None:
        ax.scatter(centres,np.log(counts_bwNorm/roi_size))
        return None
    else:
        fig, ax = plt.subplots(1,1,figsize = (8,8))
        ax.scatter(centres,np.log(counts_bwNorm/roi_size))
        ax.set_xlabel(r"Area$^{0.5}$ ($\mu$m)")
        ax.set_ylabel("ln(N / bw)")
        return fig, ax

