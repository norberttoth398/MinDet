import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from skimage import measure

#will need to upload csv file to github and use the url to load it in when using shape calc
# use url of

def GOF(o, s):
    #from Martin Mangler's 2022 paper - supplementary file 2 (note their syntax is not the clearest)
    numerator = np.sum((s - o)**2)
    denominator = np.sum(s - (s/25))**2
    return 1 - numerator/denominator

def split_at(string, char, n):
    """
    Splits string into two at the nth occurence of the character specified.
    
    Input
    ------------------------------------
    string - string to be split into two
    char - character to split at
    n - the occurrence of character at which splitting should occur.
    
    Returns
    -------------------------------------
    The two ends of the string split at the specified position.
    """
    words = string.split(char)
    return char.join(words[:n]), char.join(words[n:])

def split_s_i_l(word):
    s, rest = split_at(word, ":", 1)
    i,l = split_at(rest, ":", 1)

    #checks to make sure numbers make sense
    s = split_at(s, '.', 2)[0]
    i = split_at(i, '.', 2)[0]
    l = split_at(l, '.', 2)[0]

    #print(word)
    return float(s), float(i), float(l)

def zingg_values(word):
    s, i, l = split_s_i_l(word)

    return s/i, i/l

def calc_zingg(feature, models, mean = False, n = 30):

    #calculate scores for each model distribution
    scores = []
    for i in range(models.shape[1]):
        scores.append(GOF(models.iloc[:,i], feature))

    #sort scores in descending order
    scores = np.asarray(scores)
    ind = np.argpartition(scores, -n)[-n:]
    ind = ind[np.argsort(-scores[ind])]

    #sort zingg values in descending order of fit
    zingg = []
    for i in ind:
        zingg.append(zingg_values(models.columns[i]))
    zingg = np.asarray(zingg)

    #assign desired zingg values
    if mean == True:
        zingg_val = np.asarray([np.mean(zingg[:, 0]), np.mean(zingg[:,1])])
    else:
        zingg_val = zingg[0]

    #calculate errors as standard deviations
    errors = np.asarray([np.std(zingg[:,0]), np.std(zingg[:,1])])

    return zingg_val, errors

def ZinggPlot(feature, models, mean = False, n = 30, ax = None, marker = None):
    """_summary_

    Args:
        feature (_type_): _description_
        models (_type_): _description_
        mean (bool, optional): _description_. Defaults to False.
        n (int, optional): _description_. Defaults to 30.
        ax (_type_, optional): _description_. Defaults to None.
        marker (_type_, optional): _description_. Defaults to None.

    Returns:
        _type_: _description_
    """

    if marker is not None:
        m = marker
    else:
        m = 'k.'

    z, err = calc_zingg(feature, models)

    if ax is not None:
        ax.plot(z[0], z[1], m)
        ax.errorbar(z[0], z[1], xerr = err[0], yerr = err[1], fmt = m)
        return None
    else:
        fig, ax = plt.subplots(1,1,figsize = (8,8))
        ax.plot(z[0], z[1], m)
        ax.errorbar(z[0], z[1], xerr = err[0], yerr = err[1], fmt = m)
        ax.set_ylim([0,1])
        ax.set_xlim([0,1])
        ax.set_ylabel("I/L")
        ax.set_xlabel("S/I")
        return fig, ax

    