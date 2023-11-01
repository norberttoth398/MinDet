import numpy as np
import bilby

def mcmc_predict(x, sample_m, sample_c, return_full = False):
    """Function to perform mcmc prediction when given samples for both m and c.

    Args:
        x (ndarray): Data to perform prediction over.
        sample_m (ndarray): Sample of m parameters
        sample_c (ndarray): Sample of c parameters.
        return_full (bool, optional): Set to True if full sample of results is to be returned. Defaults to False,
                                        meaning only mean and SD are returned.

    Returns:
        results : If return_full is True - all samples are returned, otherwise just mean and SD
    """
    x = np.asarray(x)
    y = x*sample_m.reshape(-1,1) + sample_c.reshape(-1,1)
    if return_full is True:
        return y
    else:
        return np.mean(y, axis = 0), np.std(y, axis = 0)


def mcmc_predict_from_file(x, file, return_full = False):
    """_summary_

    Args:
        x (ndarray): Data to perform prediction over.
        file (str): Filename for MCMC results.
        return_full (bool, optional): Set to True if full sample of results is to be returned. Defaults to False,
                                        meaning only mean and SD are returned.

    Returns:
        results : If return_full is True - all samples are returned, otherwise just mean and SD
    """
    result = bilby.result.read_in_result(filename=file)

    m = result.posterior["m"].to_numpy()
    c = result.posterior["c"].to_numpy()

    return mcmc_predict(x, m, c, return_full)