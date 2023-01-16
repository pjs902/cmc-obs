import numpy as np
from scipy.optimize import fsolve
from tqdm import trange
from fitter.util import angular_width
import astropy.units as u
import cmctoolkit as ck


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

    root = fsolve(func, [guess_sigma_c, guess_vbar], factor=0.1, maxfev=1000)
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
            raise RuntimeError("solver failed, try more stars per bin")

    # return bin centers and velocity dispersion
    return (bin_edges[:-1] + bin_edges[1:]) / 2, sigma, delta_sigma


class Kinematics:
    def __init__(self, snapshot):
        self.snapshot = snapshot
        self.snapshot.make_2d_projection()
        self.dist = snapshot.dist
        u.set_enabled_equivalencies(angular_width(D=self.dist * u.kpc))

        filttable = ck.load_filtertable(
            "/home/peter/research/cmctoolkit/filt_index.txt"
        )
        self.snapshot.add_photometry(filttable)

    def hubble_PMs(self, stars_per_bin=15):

        # inner 100 arcsec only
        rad_lim = (100 * u.arcsec).to(u.pc).value

        # 16 < V < 17.5

        # uncertainty of 0.1 mas/yr
        err = (0.1 * u.Unit("mas/yr")).to(u.km / u.s)
        errs = np.ones(len(self.snapshot.data)) * err

        # build profile

        pass

    def gaia_PMs(self, stars_per_bin=15):
        # select MS stars

        ms = self.snapshot.data[
            self.snapshot.data["startype"] == [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
        ]

        # proper motion uncertainties are
        # 0.02-0.03 mas/yr for G<15,
        #  0.07 mas/yr at G=17,
        #  0.5 mas/yr at G=20,
        #  and 1.4 mas/yr at G=21 mag."

        # select each magnitude bin

        bright = ms[ms["obsMag_GaiaG"] < 15]
        err_bright = (0.03 * u.Unit("mas/yr")).to(u.km / u.s)

        med = ms[(ms["obsMag_GaiaG"] >= 15) & (ms["obsMag_GaiaG"] < 17)]
        err_med = (0.07 * u.Unit("mas/yr")).to(u.km / u.s)

        faint = ms[(ms["obsMag_GaiaG"] >= 17) & (ms["obsMag_GaiaG"] < 20)]
        err_faint = (0.5 * u.Unit("mas/yr")).to(u.km / u.s)

        # build each profile

        # concatenate the profiles

        pass

    def LOS_dispersion(self, stars_per_bin=15):

        # select only red giants
        giants = self.snapshot.data[self.snapshot.data["startype"] == 3]
        # select only stars with V < 15
        # giants = giants[giants["obsMag_V"] < 15]

        # uncertainty of 1 km/s
        err = np.ones(len(giants)) * 1

        # build profile
        bin_centers, sigma, delta_sigma = veldisp_profile(
            x=giants["d[PC]"],
            vi=giants["vz[KM/S]"],
            ei=err,
            stars_per_bin=stars_per_bin,
        )
        return bin_centers, sigma, delta_sigma
