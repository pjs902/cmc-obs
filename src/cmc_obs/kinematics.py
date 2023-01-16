import numpy as np
from scipy.optimize import fsolve
from tqdm import trange


def comp_veldisp(vi, ei):
    """
    (from Rebecca)
    Compute velocity dispersion and uncertainty using maximum-likelihood method
    outlined in https://articles.adsabs.harvard.edu/pdf/1993sdgc.proc..357P
    Assumes a gaussian velocity dispersion.
    """

    def func(x):
        # x[0] is velocity dispersion, x[1] is mean (vbar)
        eq1_1 = np.sum(vi / (x[0] ** 2 + ei**2))
        eq1_2 = x[1] * np.sum(1.0 / (x[0] ** 2 + ei**2))
        eq2_1 = np.sum((vi - x[1]) ** 2 / (x[0] ** 2 + ei**2) ** 2)
        eq2_2 = np.sum(1.0 / (x[0] ** 2 + ei**2))

        return [eq1_1 - eq1_2, eq2_1 - eq2_2]

    # use sample standard deviation as initial guess for solution
    guess_sigma_c = np.std(vi)
    guess_vbar = np.mean(vi)  # use sample mean as initial guess for solution

    root = fsolve(func, [guess_sigma_c, guess_vbar], factor=0.1)
    # print("root = ", root)

    # assert np.isclose(func(root), [0.0, 0.0])  # func(root) should be almost 0.0.

    sigma_c = root[0]
    vbar = root[1]

    I_11 = np.sum(1.0 / (sigma_c**2 + ei**2))

    I_22 = np.sum(
        1.0 / (sigma_c**2 + ei**2)
        - ((vi - vbar) ** 2 + 2.0 * sigma_c**2) / (sigma_c**2 + ei**2) ** 2
        + 4.0 * sigma_c**2 * (vi - vbar) ** 2 / (sigma_c**2 + ei**2) ** 3
    )

    I_12 = np.sum(2.0 * sigma_c * (vi - vbar) / (sigma_c**2 + ei**2) ** 2)

    var_vbar = I_22 / (I_11 * I_22 - I_12**2)
    error_vbar = np.sqrt(var_vbar)

    var_sigma_c = I_11 / (I_11 * I_22 - I_12**2)
    error_sigma_c = np.sqrt(var_sigma_c)

    # print("vbar = ", vbar, " +/- ", error_vbar)
    # print("sigma_c = ", sigma_c, " +/- ", error_sigma_c)

    return ((sigma_c), (error_sigma_c))


def veldisp_profile(x, vi, ei, stars_per_bin=15):
    """
    Compute velocity dispersion profile from velocity dispersion measurements and their uncertainties.

    Parameters
    ----------
    x : array_like
        Array of x values (projected radial distances).
    vi : array_like
        Array of velocities.
    ei : array_like
        Array of velocity uncertainties.
    stars_per_bin : int
        Number of stars per bin. Default is 15.

    Returns
    -------
    bin_centers : array_like
        Array of bin centers.
    sigma : array_like
        Array of velocity dispersions.
    delta_sigma : array_like
        Array of velocity dispersion uncertainties.
    """

    # calculate number of bins
    bins = int(np.ceil(len(x) / stars_per_bin))
    bin_edges = np.histogram_bin_edges(x, bins=bins)

    # initialize arrays
    sigma = np.zeros(bins)
    delta_sigma = np.zeros(bins)

    # loop over bins, tqdm
    for i in trange(len(bin_edges) - 1):
        # select stars in bin
        idx = (x >= bin_edges[i]) & (x < bin_edges[i + 1])
        # calculate velocity dispersion
        sigma[i], delta_sigma[i] = comp_veldisp(vi[idx], ei[idx])

        # put this here for now, this shouldn't happen anymore?
        if np.abs(sigma[i]) > 100:
            raise RuntimeError("sigma is too large, solver failed")

    # return bin centers and velocity dispersion
    return (bin_edges[:-1] + bin_edges[1:]) / 2, sigma, delta_sigma



def hubble_PMs():
    pass

def gaia_PMs():
    pass

def LOS_dispersion():
    pass