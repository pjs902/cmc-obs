import numpy as np
from scipy.optimize import fsolve
from tqdm import trange
from fitter.util import angular_width
import astropy.units as u
import cmctoolkit as ck
import logging


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
    Creates bins with equal number of stars in each bin.

    Parameters
    ----------
    x : array_like
        Array of x values (projected radial distances).
    vi : array_like
        Array of velocities.
    ei : array_like
        Array of velocity uncertainties.
    stars_per_bin : int
        Number of stars per bin. Default is 15 (more is generally better).

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
    # bin_edges = np.histogram_bin_edges(x, bins=bins)

    # initialize arrays
    sigma = np.zeros(bins)
    delta_sigma = np.zeros(bins)

    bin_centers = np.zeros(bins)

    # loop over bins, tqdm
    start = 0
    for i in trange(bins):
        # set end of bin
        end = np.min([start + stars_per_bin, len(x)])
        # calculate velocity dispersion
        sigma[i], delta_sigma[i] = comp_veldisp(vi[start:end], ei[start:end])
        bin_centers[i] = np.mean(x[start:end])

        start += stars_per_bin

        # put this here for now, this shouldn't happen anymore?
        if np.abs(sigma[i]) > 100:
            logging.warning("solver failed, try more stars per bin")
            sigma[i] = np.nan
            delta_sigma[i] = np.nan
    # return bin centers and velocity dispersion
    # return (bin_edges[:-1] + bin_edges[1:]) / 2, sigma, delta_sigma
    return bin_centers, sigma, delta_sigma


class Observations:
    """
    A class for computing observations from a snapshot.
    """

    def __init__(
        self, snapshot, filtindex="/home/peter/research/cmctoolkit/filt_index.txt"
    ):
        """ """
        self.snapshot = snapshot
        self.snapshot.make_2d_projection()
        self.dist = snapshot.dist
        u.set_enabled_equivalencies(angular_width(D=self.dist * u.kpc))

        filttable = ck.load_filtertable(filtindex)
        self.snapshot.add_photometry(filttable)

        # sort by projected distance
        self.snapshot.data = self.snapshot.data.sort_values(by="d[PC]")

    def hubble_PMs(self, stars_per_bin=120):
        """
        Simulate proper motion measurements with HST-like performance.
        (performance details from VHB2019)

        Parameters
        ----------
        stars_per_bin : int
            Number of stars per bin. Default is 120 (more is generally better).

        Returns
        -------
        bin_centers : array_like
            Array of bin centers, units of pc.
        sigma_r : array_like
            Array of velocity dispersions in the radial direction.
        delta_sigma_r : array_like
            Array of velocity dispersion uncertainties in the radial direction.
        sigma_t : array_like
            Array of velocity dispersions in the tangential direction.
        delta_sigma_t : array_like
            Array of velocity dispersion uncertainties in the tangential direction.
        """

        ms = self.snapshot.data[
            self.snapshot.data["startype"].isin([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
        ]
        print("number of stars = ", len(ms))

        # inner 100 arcsec only
        rad_lim = (100 * u.arcsec).to(u.pc).value

        # select only stars with 16 < V < 17.5 # VHB2019
        ms = ms[ms["obsMag_V"] < 17.5]
        ms = ms[ms["obsMag_V"] > 16.0]
        ms = ms[ms["d[PC]"] < rad_lim]
        print("number of stars = ", len(ms))

        # uncertainty of 0.1 mas/yr
        err = (0.1 * u.Unit("mas/yr")).to(u.km / u.s)
        errs = (np.ones(len(self.snapshot.data)) * err).value

        # build profiles
        bin_centers, sigma_r, delta_sigma_r = veldisp_profile(
            x=ms["d[PC]"].values,
            vi=ms["vd[KM/S]"].values,
            ei=errs,
            stars_per_bin=stars_per_bin,
        )

        bin_centers, sigma_t, delta_sigma_t = veldisp_profile(
            x=ms["d[PC]"].values,
            vi=ms["va[KM/S]"].values,
            ei=errs,
            stars_per_bin=stars_per_bin,
        )
        # get mean mass for these profiles
        mean_mass = np.mean(ms["m[MSUN]"])

        return bin_centers, sigma_r, delta_sigma_r, sigma_t, delta_sigma_t, mean_mass

    def gaia_PMs(self, stars_per_bin=120):
        """
        Simulate proper motion measurements with Gaia-like performance.
        (performance details from VHB2019 and https://www.cosmos.esa.int/web/gaia/earlydr3)

        Parameters
        ----------
        stars_per_bin : int
            Number of stars per bin. Default is 120 (more is generally better).

        Returns
        -------
        bin_centers : array_like
            Array of bin centers, units of pc.
        sigma_r : array_like
            Array of velocity dispersions in the radial direction, units of mas/yr.
        delta_sigma_r : array_like
            Array of velocity dispersion uncertainties in the radial direction, units of mas/yr.
        sigma_t : array_like
            Array of velocity dispersions in the tangential direction, units of mas/yr.
        delta_sigma_t : array_like
            Array of velocity dispersion uncertainties in the tangential direction, units of mas/yr.
        """
        # select MS stars

        ms = self.snapshot.data[
            self.snapshot.data["startype"].isin([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
        ]
        print("number of stars = ", len(ms))

        # select based on G mag, 17 from VHB+2019
        ms = ms[ms["obsMag_GaiaG"] < 17]
        ms = ms[ms["obsMag_GaiaG"] > 3]
        print("number of stars = ", len(ms))

        # proper motion uncertainties are
        # 0.02-0.03 mas/yr for G<15,
        #  0.07 mas/yr at G=17,
        #  0.5 mas/yr at G=20,
        #  and 1.4 mas/yr at G=21 mag."

        # function to calculate error based on G mag
        def err_func(G):
            # TODO: make this smooth?
            if G < 15:
                return 0.02
            elif G < 17:
                return 0.07
            elif G < 20:
                return 0.5
            else:
                return 1.4

        err = np.array([err_func(G) for G in ms["obsMag_GaiaG"]])

        # convert to km/s
        errs = (err * u.Unit("mas/yr")).to(u.km / u.s).value

        # build profiles
        bin_centers, sigma_r, delta_sigma_r = veldisp_profile(
            x=ms["d[PC]"].values,
            vi=ms["vd[KM/S]"].values,
            ei=errs,
            stars_per_bin=stars_per_bin,
        )

        bin_centers, sigma_t, delta_sigma_t = veldisp_profile(
            x=ms["d[PC]"].values,
            vi=ms["va[KM/S]"].values,
            ei=errs,
            stars_per_bin=stars_per_bin,
        )

        # get mean mass for this profile
        mean_mass = np.mean(ms["m[MSUN]"])

        # convert to mas/yr
        sigma_r = (sigma_r * u.km / u.s).to(u.mas / u.yr).value
        delta_sigma_r = (delta_sigma_r * u.km / u.s).to(u.mas / u.yr).value
        sigma_t = (sigma_t * u.km / u.s).to(u.mas / u.yr).value
        delta_sigma_t = (delta_sigma_t * u.km / u.s).to(u.mas / u.yr).value

        return bin_centers, sigma_r, delta_sigma_r, sigma_t, delta_sigma_t, mean_mass

    def LOS_dispersion(self, stars_per_bin=25):
        """
        Simulate LOS velocity dispersion measurements with performance based on VHB+2019.

        Parameters
        ----------
        stars_per_bin : int
            Number of stars per bin. Default is 25 (more is generally better).

        Returns
        -------
        bin_centers : array_like
            Array of bin centers, units of pc.
        sigma : array_like
            Array of velocity dispersions, units of km/s.
        delta_sigma : array_like
            Array of velocity dispersion uncertainties, units of km/s.
        """

        # select only red giants
        giants = self.snapshot.data[self.snapshot.data["startype"] == 3]

        print("number of giants", len(giants))

        # select only stars with V < 15 VHB+2019
        giants = giants[giants["obsMag_V"] < 15]

        print("number of giants", len(giants))

        # uncertainty of 1 km/s
        err = np.ones(len(giants)) * 1

        # build profile
        bin_centers, sigma, delta_sigma = veldisp_profile(
            x=giants["d[PC]"].values,
            vi=giants["vz[KM/S]"].values,
            ei=err,
            stars_per_bin=stars_per_bin,
        )

        # get mean mass for this profile
        mean_mass = np.mean(giants["m[MSUN]"])
        return bin_centers, sigma, delta_sigma, mean_mass

    def number_density(self, Nbins=50):
        """
        Simulate number density measurements with performance based on de Boer+2019.

        Parameters
        ----------
        Nbins : int
            Number of bins. Default is 50 (less may be prefferable for undersampled clusters)

        Returns
        -------
        bin_centers : array_like
            Array of bin centers, units of pc.
        number_density : array_like
            Array of number densities, units of arcmin^-2.
        delta_number_density : array_like
            Array of number density uncertainties, units of arcmin^-2.
        """

        # select main sequence stars
        ms = self.snapshot.data[
            self.snapshot.data["startype"].isin([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
        ]
        print("number of stars = ", len(ms))

        # select stars brighter than G 20, de Boer+2019
        ms = ms[ms["obsMag_GaiaG"] < 20]
        print("number of stars = ", len(ms))

        # calculate number of stars per bin given total bins
        stars_per_bin = len(ms) / Nbins

        # initialize arrays
        bin_centers = np.zeros(Nbins)
        number_density = np.zeros(Nbins)
        delta_number_density = np.zeros(Nbins)

        # loop over bins
        for i in range(Nbins):
            # select stars in bin
            bin = ms[int(i * stars_per_bin) : int((i + 1) * stars_per_bin)]

            # get edges of bin
            bin_min = np.min(bin["d[PC]"])
            bin_max = np.max(bin["d[PC]"])

            # get center of bin
            bin_centers[i] = (bin_max + bin_min) / 2
            bin_centers[i] = np.mean(bin["d[PC]"])

            # calculate surface number density, in current annulus
            number_density[i] = len(bin) / (np.pi * (bin_max**2 - bin_min**2))

            # calculate error
            delta_number_density[i] = np.sqrt(len(bin)) / (
                np.pi * (bin_max**2 - bin_min**2)
            )

        # get mean mass for this profile
        mean_mass = np.mean(ms["m[MSUN]"])

        # convert from linear units to angular units
        number_density *= u.Unit("pc^-2")
        delta_number_density *= u.Unit("pc^-2")

        number_density = number_density.to(u.Unit("arcmin^-2")).value
        delta_number_density = delta_number_density.to(u.Unit("arcmin^-2")).value

        return bin_centers, number_density, delta_number_density, mean_mass
