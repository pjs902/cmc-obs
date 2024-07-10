import json
import logging
import pathlib
from functools import partial
import os

import astropy.units as u
import blackjax
import cmctoolkit as ck
import h5py
import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
import scipy as sp
from gcfit.util import angular_width
from gcfit.util.data import ClusterFile, Dataset
from scipy.optimize import fsolve
from tqdm import trange

# jax setup
jax.config.update("jax_enable_x64", True)

rng_key = jax.random.PRNGKey(137)


def comp_veldisp_MLE(vi, ei, guess_sigma_c=None, guess_vbar=None):
    """
    (from Rebecca)
    Compute velocity dispersion and uncertainty using maximum-likelihood method
    outlined in https://articles.adsabs.harvard.edu/pdf/1993sdgc.proc..357P
    Assumes a gaussian velocity dispersion.

    Parameters
    ----------
    vi : array_like
        Array of velocities.
    ei : array_like
        Array of velocity uncertainties.

    Returns
    -------
    sigma_c : float
        Velocity dispersion.
    error_sigma_c : float
        Velocity dispersion uncertainty.
    """

    def func(x):
        # x[0] is velocity dispersion, x[1] is mean (vbar)
        eq1_1 = np.sum(vi / (x[0] ** 2 + ei**2))
        eq1_2 = x[1] * np.sum(1.0 / (x[0] ** 2 + ei**2))
        eq2_1 = np.sum((vi - x[1]) ** 2 / (x[0] ** 2 + ei**2) ** 2)
        eq2_2 = np.sum(1.0 / (x[0] ** 2 + ei**2))

        return [eq1_1 - eq1_2, eq2_1 - eq2_2]

    root = fsolve(func, [guess_sigma_c, guess_vbar], factor=0.1, maxfev=10000)

    # assert np.isclose(func(root), [0.0, 0.0])  # func(root) should be almost 0.0.

    sigma_c = root[0]
    vbar = root[1]

    I_11 = np.sum(1.0 / (sigma_c**2 + ei**2))

    I_22 = np.sum(
        1.0 / (sigma_c**2 + ei**2)
        - ((vi - vbar) ** 2 + (2.0 * (sigma_c**2))) / ((sigma_c**2 + ei**2) ** 2)
        + 4.0 * (sigma_c**2) * ((vi - vbar) ** 2) / (sigma_c**2 + ei**2) ** 3
    )

    I_12 = np.sum((2.0 * sigma_c * (vi - vbar)) / ((sigma_c**2 + ei**2) ** 2))

    var_vbar = I_22 / (I_11 * I_22 - I_12**2)
    error_vbar = np.sqrt(var_vbar)

    var_sigma_c = I_11 / (I_11 * I_22 - I_12**2)
    error_sigma_c = np.sqrt(var_sigma_c)

    # print("vbar = ", vbar, " +/- ", error_vbar)
    # print("sigma_c = ", sigma_c, " +/- ", error_sigma_c)

    return ((sigma_c), (error_sigma_c))


# heres all the jax stuff for the full MCMC dispersion profile


# likelihood function for points with gaussian errors drawn from a gaussian (eq 5.64 in astroML book)
def logp(mu, sigma, data, errs):
    return -0.5 * jnp.sum(jnp.log(sigma**2 + errs**2) + (((data - mu) ** 2) / (sigma**2 + errs**2)))


# log posterior
@jax.jit
def logL(theta, data, errs):
    mu, sigma = theta["mu"], theta["sigma"]

    # get mean and std of data to set priors
    mu_data = jnp.mean(data)
    sigma_data = jnp.std(data)

    # uniform priors around the data mean and std
    prior = jax.scipy.stats.uniform.logpdf(loc=mu_data - 10, scale=20, x=mu) + jax.scipy.stats.uniform.logpdf(
        loc=0, scale=20, x=sigma
    )

    # prior = jax.scipy.stats.uniform.logpdf(loc=-500, scale=1000, x=mu) + jax.scipy.stats.uniform.logpdf(loc=0, scale=50, x=sigma)

    return logp(mu, sigma, data, errs) + prior


def inference_loop(rng_key, kernel, initial_state, num_samples):
    @jax.jit
    def one_step(state, rng_key):
        state, _ = kernel(rng_key, state)
        return state, state

    keys = jax.random.split(rng_key, num_samples)
    _, states = jax.lax.scan(one_step, initial_state, keys)

    return states


def comp_veldisp(vi, ei):
    # sampler parameters
    inv_mass_matrix = np.array([0.5, 0.01])
    step_size = 1e-3

    # log likelihood with data frozen in
    like = partial(logL, data=vi, errs=ei)

    # initial pos
    initial_position = {"mu": jnp.mean(vi), "sigma": jnp.std(vi)}

    # do warm up
    warmup = blackjax.window_adaptation(blackjax.nuts, like)
    (state, parameters), _ = warmup.run(rng_key, initial_position, num_steps=1000)

    # set up kernel
    kernel = blackjax.nuts(like, **parameters).step

    # do inference
    states = inference_loop(rng_key, kernel, state, 1_000)

    # get samples
    mu_samples = states.position["mu"].block_until_ready()
    sigma_samples = states.position["sigma"]

    return np.mean(sigma_samples), np.std(sigma_samples)


def veldisp_profile(x, vi, ei, stars_per_bin=15, show_progress=False):
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
    stars_per_bin : Union[int, array_like]
        Number of stars per bin. Can be an integer for uniform binning or an array of bin sizes. Default is 15.
    show_progress : bool
        Whether to show progress with tqdm bar. Default is False.

    Returns
    -------
    bin_centers : array_like
        Array of bin centers.
    sigma : array_like
        Array of velocity dispersions.
    delta_sigma : array_like
        Array of velocity dispersion uncertainties.

    See Also
    --------
    create_binning : Create binning for dispersion profiles.
    """

    # check if stars_per_bin is an array or list
    if isinstance(stars_per_bin, list | np.ndarray):
        bins = len(stars_per_bin)
    elif isinstance(stars_per_bin, int):
        bins = int(np.ceil(len(x) / stars_per_bin))
        stars_per_bin = np.ones(bins) * stars_per_bin
    else:
        msg = "stars_per_bin must be an integer, list or numpy array"
        raise ValueError(msg)

    # initialize arrays
    sigma = np.zeros(bins)
    delta_sigma = np.zeros(bins)
    bin_centers = np.zeros(bins)

    # loop over bins
    start = 0
    for i in trange(bins, disable=not show_progress):
        # set end of bin
        end = np.min([start + int(stars_per_bin[i]), len(x)])
        # calculate velocity dispersion
        sigma[i], delta_sigma[i] = comp_veldisp(vi[start:end], ei[start:end])
        bin_centers[i] = np.mean(x[start:end])

        start += int(stars_per_bin[i])

        # put this here for now, this shouldn't happen anymore?
        if np.abs(sigma[i]) > 100:
            logging.warning("solver failed, try more stars per bin")
            sigma[i] = np.nan
            delta_sigma[i] = np.nan

    return bin_centers, sigma, delta_sigma


def create_binning(xs, prepend_bins, append_bins, middle_bin_size=None, N_middle_bins=None):
    """
    Create binning for things like dispersion profiles. Allows for you to specify some starting and
    ending bins with custom bin sizes, filling in the middle with bins of uniform size. The middle bins
    can be either a fixed number of bins or a fixed bin size.

    Parameters
    ----------
    xs : array-like
        The values to bin.
    prepend_bins : array-like
        The bins to prepend to the start of the array.
    append_bins : array-like
        The bins to append to the end of the array.
    middle_bin_size : int, optional
        The size of the middle bins. If not provided, N_middle_bins must be provided, but not both.
    N_middle_bins : int, optional
        The number of middle bins. If not provided, middle_bin_size must be provided, but not both.
    Returns
    -------
    bins : array-like
        The bin sizes.
    """

    N = len(xs)
    to_bin = N - (sum(prepend_bins) + sum(append_bins))

    if middle_bin_size is not None and N_middle_bins is not None:
        msg = "Only one of middle_bin_size or N_middle_bins can be provided."
        raise ValueError(msg)

    if middle_bin_size is None and N_middle_bins is None:
        msg = "Either middle_bin_size or N_middle_bins must be provided."
        raise ValueError(msg)

    if middle_bin_size is not None:
        stars_per_bin = middle_bin_size
        N_middle_bins = to_bin // middle_bin_size
        N_extra = to_bin % middle_bin_size
    else:
        N_middle_bins = N_middle_bins
        stars_per_bin = to_bin // N_middle_bins
        N_extra = to_bin % N_middle_bins

    append_bins[-1] += N_extra

    bins = np.concatenate([prepend_bins, np.repeat(stars_per_bin, N_middle_bins), append_bins])

    assert sum(bins) == N

    return bins


class Observations:
    """
    A class for computing observations from a snapshot.
    """

    def __init__(
        self,
        snapshot,
        filtindex=None,
        cluster_name="CMC",
        add_photometry=True,
    ):
        """
        Initialize an Observations object.

        Parameters
        ----------
        snapshot : Snapshot
            Snapshot object.
        filtindex : str
            Path to filter index file, used for adding photometry. Default is None.
            If None, default filters are used which include UVBRI as well as Gaia G and HST F814W.
        cluster_name : str
            Name of cluster.
        add_photometry : bool
            Whether to add photometry to the snapshot. Default is True.
        """

        self.rng = np.random.default_rng()

        # load snapshot
        self.snapshot = snapshot

        # make 2d projection
        self.snapshot.make_2d_projection()

        # set distance
        self.dist = snapshot.dist
        u.set_enabled_equivalencies(angular_width(D=self.dist * u.kpc))

        # add photometry
        if add_photometry:
            filttable = ck.load_default_filters() if filtindex is None else ck.load_filtertable(filtindex)
            self.snapshot.add_photometry(filttable)

        self.cluster_name = cluster_name

        # get main-sequence stars
        ms_mask = (
            (self.snapshot.data["startype"].isin([0, 1]))
            | (self.snapshot.data["bin_startype0"].isin([0, 1]))
            | (self.snapshot.data["bin_startype1"].isin([0, 1]))
        )
        self.ms_mask = ms_mask

        # startypes for main-sequence stars and giants
        self.startypes = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

        # sort by projected distance
        self.snapshot.data = self.snapshot.data.sort_values(by="d[PC]")

        # BH info
        self.snapshot.bh_radii_proj = pd.concat(
            [
                self.snapshot.data.loc[(self.snapshot.data["startype"] == 14)]["d[PC]"],
                self.snapshot.data.loc[(self.snapshot.data["bin_startype0"] == 14)]["d[PC]"],
                self.snapshot.data.loc[(self.snapshot.data["bin_startype1"] == 14)]["d[PC]"],
            ],
            axis=0,
        ).to_list()

        self.snapshot.bh_radii_3d = pd.concat(
            [
                self.snapshot.data.loc[(self.snapshot.data["startype"] == 14)]["r[PC]"],
                self.snapshot.data.loc[(self.snapshot.data["bin_startype0"] == 14)]["r[PC]"],
                self.snapshot.data.loc[(self.snapshot.data["bin_startype1"] == 14)]["r[PC]"],
            ],
            axis=0,
        ).to_list()
        self.snapshot.bh_masses = pd.concat(
            [
                self.snapshot.data.loc[(self.snapshot.data["startype"] == 14)]["m_MSUN"],
                self.snapshot.data.loc[(self.snapshot.data["bin_startype0"] == 14)]["m0_MSUN"],
                self.snapshot.data.loc[(self.snapshot.data["bin_startype1"] == 14)]["m1_MSUN"],
            ],
            axis=0,
        ).to_list()
        self.snapshot.M_BH = np.sum(self.snapshot.bh_masses)
        self.snapshot.N_BH = len(self.snapshot.bh_masses)

    def hubble_PMs(self, stars_per_bin=120, r_outer=100, mag_lim_bright=15, mag_lim_faint=18, per_star_err=0.1):
        """
        Simulate proper motion measurements with HST-like performance.
        (performance details from VHB2019)

        Parameters
        ----------
        stars_per_bin : int
            Number of stars per bin. Default is 120 (more is generally better).
        r_outer : float
            Outer radius for HST measurements, units of arcseconds. Default is 100.
        mag_lim_bright : float
            Bright magnitude limit for HST measurements. Default is 15.
        mag_lim_faint : float
            Faint magnitude limit for HST measurements. Default is 18.
        per_star_err : float
            Per-star proper motion error, units of mas/yr. Default is 0.1.

        Returns
        -------
        bin_centers : array_like
            Array of bin centers, units of arcseconds.
        sigma_r : array_like
            Array of velocity dispersions in the radial direction, units are mas/yr.
        delta_sigma_r : array_like
            Array of velocity dispersion uncertainties in the radial direction.
        sigma_t : array_like
            Array of velocity dispersions in the tangential direction, units are mas/yr.
        delta_sigma_t : array_like
            Array of velocity dispersion uncertainties in the tangential direction.
        """

        stars = self.snapshot.data.loc[
            (self.snapshot.data["startype"].isin(self.startypes))
            | (self.snapshot.data["bin_startype0"].isin(self.startypes))
            | (self.snapshot.data["bin_startype1"].isin(self.startypes))
        ]
        logging.info(f"HSTPM: number of stars, prefilter = {len(stars)}")

        # inner 100 arcsec only
        rad_lim = (r_outer * u.arcsec).to(u.pc).value

        # select only stars with V 15 to 18 based on looking at some of the HACKS photometry tables

        stars = stars.loc[
            (stars["tot_obsMag_V"] < mag_lim_faint)
            & (stars["tot_obsMag_V"] > mag_lim_bright)
            & (stars["d[PC]"] < rad_lim)
        ]
        logging.info(f"HSTPM: number of stars, postfilter = {len(stars)}")

        # calculate how many stars per bin to use
        # we want to target 5 bins but require at least 120 stars per bin
        stars_per_bin = int(np.ceil(len(stars) / 5))
        stars_per_bin = np.max([stars_per_bin, 120])
        logging.info(f"HSTPM: stars per bin = {stars_per_bin}")

        # uncertainty of 0.1 mas/yr
        err = (per_star_err * u.Unit("mas/yr")).to(u.km / u.s).value
        errs = np.ones(len(stars)) * err

        # resample based on errors
        kms_r = self.rng.normal(loc=stars["vd[KM/S]"].values, scale=errs)
        kms_t = self.rng.normal(loc=stars["va[KM/S]"].values, scale=errs)

        # build profiles
        bin_centers, sigma_r, delta_sigma_r = veldisp_profile(
            x=stars["d[PC]"].values,
            vi=kms_r,
            ei=errs,
            stars_per_bin=stars_per_bin,
        )

        bin_centers, sigma_t, delta_sigma_t = veldisp_profile(
            x=stars["d[PC]"].values,
            vi=kms_t,
            ei=errs,
            stars_per_bin=stars_per_bin,
        )
        # get mean mass for these profiles
        mean_mass = np.mean(stars["m[MSUN]"])

        # convert to mas/yr
        sigma_r = (sigma_r * u.km / u.s).to(u.mas / u.yr).value
        delta_sigma_r = (delta_sigma_r * u.km / u.s).to(u.mas / u.yr).value
        sigma_t = (sigma_t * u.km / u.s).to(u.mas / u.yr).value
        delta_sigma_t = (delta_sigma_t * u.km / u.s).to(u.mas / u.yr).value

        # convert bin centers to arcsec
        bin_centers = (bin_centers * u.pc).to(u.arcsec).value

        return bin_centers, sigma_r, delta_sigma_r, sigma_t, delta_sigma_t, mean_mass

    def gaia_PMs(self, stars_per_bin=120, r_inner=100, mag_lim_bright=13, mag_lim_faint=19):
        """
        Simulate proper motion measurements with Gaia-like performance.
        (performance details from VHB2019 and https://www.cosmos.esa.int/web/gaia/earlydr3)

        Parameters
        ----------
        stars_per_bin : int
            Number of stars per bin. Default is 120 (more is generally better).
        r_inner : float
            Inner radius for Gaia measurements, units of arcseconds. Default is 100.
        mag_lim_bright : float
            Bright magnitude limit for Gaia measurements. Default is 13.
        mag_lim_faint : float
            Faint magnitude limit for Gaia measurements. Default is 19.

        Returns
        -------
        bin_centers : array_like
            Array of bin centers, units of arcseconds.
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
        stars = self.snapshot.data.loc[
            (self.snapshot.data["startype"].isin(self.startypes))
            | (self.snapshot.data["bin_startype0"].isin(self.startypes))
            | (self.snapshot.data["bin_startype1"].isin(self.startypes))
        ]
        logging.info(f"GaiaPM: number of stars, prefilter = {len(stars)}")

        # select based on G mag
        # also only want stars further than 100 arcsec so we dont overlap with HST
        # using the 21>G>13 mag limit from Vasiliev+Baumgardt+2021 https://arxiv.org/pdf/2102.09568.pdf
        rad_lim = (r_inner * u.arcsec).to(u.pc).value
        stars = stars.loc[
            (stars["tot_obsMag_GaiaG"] < mag_lim_faint)
            & (stars["tot_obsMag_GaiaG"] > mag_lim_bright)
            & (stars["d[PC]"] > rad_lim)
        ]
        logging.info(f"GaiaPM: number of stars, postfilter = {len(stars)}")

        # calculate how many stars per bin to use
        # require at least 120 stars per bin, target 5 bins
        stars_per_bin = int(np.ceil(len(stars) / 5))
        stars_per_bin = np.max([stars_per_bin, 120])

        logging.info(f"GaiaPM: stars per bin = {stars_per_bin}")

        # Gaia error function
        err = np.array([gaia_err_func(G) for G in stars["tot_obsMag_GaiaG"]])

        # convert to km/s
        errs = (err * u.Unit("mas/yr")).to(u.km / u.s).value

        # resample based on errors
        kms_r = self.rng.normal(loc=stars["vd[KM/S]"].values, scale=errs)
        kms_t = self.rng.normal(loc=stars["va[KM/S]"].values, scale=errs)

        # build profiles
        bin_centers, sigma_r, delta_sigma_r = veldisp_profile(
            x=stars["d[PC]"].values,
            vi=kms_r,
            ei=errs,
            stars_per_bin=stars_per_bin,
        )

        bin_centers, sigma_t, delta_sigma_t = veldisp_profile(
            x=stars["d[PC]"].values,
            vi=kms_t,
            ei=errs,
            stars_per_bin=stars_per_bin,
        )

        # get mean mass for this profile
        mean_mass = np.mean(stars["m[MSUN]"])

        # convert to mas/yr
        sigma_r = (sigma_r * u.km / u.s).to(u.mas / u.yr).value
        delta_sigma_r = (delta_sigma_r * u.km / u.s).to(u.mas / u.yr).value
        sigma_t = (sigma_t * u.km / u.s).to(u.mas / u.yr).value
        delta_sigma_t = (delta_sigma_t * u.km / u.s).to(u.mas / u.yr).value

        # convert bin centers to arcsec
        bin_centers = (bin_centers * u.pc).to(u.arcsec).value

        return bin_centers, sigma_r, delta_sigma_r, sigma_t, delta_sigma_t, mean_mass

    def LOS_dispersion(self, stars_per_bin=70, mag_lim_faint=17, per_star_err=1.0):
        """
        Simulate LOS velocity dispersion measurements with performance based on VHB+2019.

        Parameters
        ----------
        stars_per_bin : int
            Number of stars per bin. Default is 70 (more is generally better).
        mag_lim_faint : float
            Faint magnitude limit for LOS measurements. Default is 17.
        per_star_err : float
            Per-star velocity error, units of km/s. Default is 1.0.

        Returns
        -------
        bin_centers : array_like
            Array of bin centers, units of arcseconds.
        sigma : array_like
            Array of velocity dispersions, units of km/s.
        delta_sigma : array_like
            Array of velocity dispersion uncertainties, units of km/s.
        """

        # select only giants (No Binaries, there is some filtering in the real data, not sure we want to get into that)
        giants = self.snapshot.data.loc[(self.snapshot.data["startype"].isin([3, 4, 5, 6, 7, 8]))]

        logging.info(f"LOS: number of giants, prefilter = {len(giants)}")

        # select only stars with G mag <17 based on Holger's compilations
        giants = giants.loc[giants["obsMag_GaiaG"] < 17]

        logging.info(f"LOS: number of giants, postfilter = {len(giants)}")

        # generate errors drawn from a gaussian with sigma = 1 km/s
        errs = np.ones(len(giants)) * 1.0

        # resample based on errors
        kms = self.rng.normal(loc=giants["vz[KM/S]"].values, scale=errs)

        # calculate how many stars per bin to use
        # we want to target 10 bins but require at least 70 stars per bin
        stars_per_bin = int(np.ceil(len(giants) / 10))
        stars_per_bin = np.max([stars_per_bin, 70])
        logging.info(f"LOS: stars per bin = {stars_per_bin}")

        # build profile
        bin_centers, sigma, delta_sigma = veldisp_profile(
            x=giants["d[PC]"].values,
            vi=kms,
            ei=errs,
            stars_per_bin=stars_per_bin,
        )

        # convert bin centers to arcsec
        bin_centers = (bin_centers * u.pc).to(u.arcsec).value

        # get mean mass for this profile
        mean_mass = np.mean(giants["m[MSUN]"])
        return bin_centers, sigma, delta_sigma, mean_mass

    def number_density(self, Nbins=50, mag_lim_faint=20):
        """
        Simulate number density measurements with performance based on de Boer+2019.

        Parameters
        ----------
        Nbins : int
            Number of bins. Default is 50.
        mag_lim_faint : float
            Faint magnitude limit for number density measurements. Default is 20.

        Returns
        -------
        bin_centers : array_like
            Array of bin centers, units of arcminutes.
        number_density : array_like
            Array of number densities, units of arcmin^-2.
        delta_number_density : array_like
            Array of number density uncertainties, units of arcmin^-2.
        mean_mass : float
            Mean mass of stars in the profile, units of solar masses.
        """

        # select main sequence stars
        stars = self.snapshot.data.loc[
            (self.snapshot.data["startype"].isin(self.startypes))
            | (self.snapshot.data["bin_startype0"].isin(self.startypes))
            | (self.snapshot.data["bin_startype1"].isin(self.startypes))
        ]
        logging.info(f"ND: number of stars, prefilter = {len(stars)}")

        # select stars brighter than G 20, de Boer+2019
        stars = stars.loc[stars["tot_obsMag_GaiaG"] < mag_lim_faint]
        logging.info(f"ND: number of stars, postfilter = {len(stars)}")

        # calculate number of stars per bin given total bins
        stars_per_bin = len(stars) / Nbins

        # initialize arrays
        bin_centers = np.zeros(Nbins)
        number_density = np.zeros(Nbins)
        delta_number_density = np.zeros(Nbins)

        # loop over bins
        for i in range(Nbins):
            # select stars in bin
            sel = stars[int(i * stars_per_bin) : int((i + 1) * stars_per_bin)]

            # get edges of bin
            bin_min = np.min(sel["d[PC]"])
            bin_max = np.max(sel["d[PC]"])

            # get center of bin
            bin_centers[i] = (bin_max + bin_min) / 2
            bin_centers[i] = np.mean(sel["d[PC]"])

            # calculate surface number density, in current annulus
            number_density[i] = len(sel) / (np.pi * (bin_max**2 - bin_min**2))

            # calculate error
            delta_number_density[i] = np.sqrt(len(sel)) / (np.pi * (bin_max**2 - bin_min**2))

        # get mean mass for this profile
        mean_mass = np.mean(stars["m[MSUN]"])

        # convert from linear units to angular units
        number_density *= u.Unit("pc^-2")
        delta_number_density *= u.Unit("pc^-2")

        number_density = number_density.to(u.Unit("arcmin^-2")).value
        delta_number_density = delta_number_density.to(u.Unit("arcmin^-2")).value

        # convert bin centers to arcmin
        bin_centers = (bin_centers * u.pc).to(u.arcmin).value

        return bin_centers, number_density, delta_number_density, mean_mass

    def mass_function(self, r_in, r_out, bins=10, extra_scatter=3.0):
        """
        Extract the mass function in a given annulus.

        Parameters
        ----------
        r_in : float
            Inner radius of annulus, units of arcmin.
        r_out : float
            Outer radius of annulus, units of arcmin.
        bins : int (optional)
            Number of bins to use for mass function. Default is 10.
        extra_scatter : float (optional)
            Factor by which to increase the scatter in the mass function. Default is 3.0.

        Returns
        -------
        bin_edges : array_like
            Array of bin edges, units of Msun.
        mass_function : array_like
            Array of mass function, units of Msun^-1.
        delta_mass_function : array_like
            Array of mass function uncertainties, units of Msun^-1.
        """

        # convert radii to pc
        r_in_pc = (r_in * u.arcmin).to(u.pc).value
        r_out_pc = (r_out * u.arcmin).to(u.pc).value

        # select main sequence stars, using ms mask
        sel = self.snapshot.data.loc[self.ms_mask]

        # only do the mass function for main sequence masses
        lower = np.min(sel["m[MSUN]"])
        upper = ck.find_MS_TO(t=self.snapshot.age, z=self.snapshot.z)

        # select stars in annulus
        sel = sel.loc[(sel["d[PC]"] > r_in_pc) & (sel["d[PC]"] < r_out_pc)]

        # Here we can just tack on the limiting mass logic

        # first we need the number density profile
        (
            bin_centers,
            number_density,
            delta_number_density,
            mean_mass,
        ) = self.number_density()

        # set up interpolation function
        nd_interp = sp.interpolate.interp1d(
            bin_centers,
            number_density,
            kind="cubic",
            bounds_error=False,
            fill_value=(8000, 0),
        )

        # get the average number density in the annulus, get the limiting mass
        ND = np.mean(nd_interp([r_in, r_out]))

        # limiting mass should never be more than the MSTO,
        limiting_mass = np.min([ND_limiting_mass(ND), upper])

        # update sel to only include stars above the limiting mass
        sel = sel.loc[sel["m[MSUN]"] > (limiting_mass - 0.1)]

        # update lower mass limits for histogram
        lower = np.min(sel["m[MSUN]"])

        # need to update number of bins based on new mass limits
        bins = int(np.ceil((upper - lower) / 0.1))

        heights, edges = np.histogram(a=sel["m[MSUN]"], bins=bins, range=(lower, upper))
        err = np.sqrt(heights)

        # need to add some scatter for realism
        # adopt F=3
        F = extra_scatter
        new_uncertainties = err * F
        new_heights = self.rng.normal(loc=heights, scale=new_uncertainties)

        # print a bunch of debug info
        logging.info(
            f"MF: inner radius: {r_in:.2f} arcmin, outer radius: {r_out:.2f} arcmin, ND: {ND:.2f} arcmin^-2, limiting mass: {limiting_mass:.2f} Msun"
        )

        return edges, new_heights, err

    def write_obs(self):
        """
        Write the simulated observations to a file.
        """

        # check that ./raw_data/ exists, create if not
        if not os.path.exists("./raw_data"):
            os.makedirs("./raw_data")

        # metadata dict to hold thins like masses
        metadata = {}

        # Hubble PM
        (
            bin_centers,
            sigma_r,
            delta_sigma_r,
            sigma_t,
            delta_sigma_t,
            mean_mass,
        ) = self.hubble_PMs()

        # round to 3 decimal places
        bin_centers = np.round(bin_centers, 3)
        sigma_r = np.round(sigma_r, 3)
        delta_sigma_r = np.round(delta_sigma_r, 3)
        sigma_t = np.round(sigma_t, 3)
        delta_sigma_t = np.round(delta_sigma_t, 3)
        mean_mass = np.round(mean_mass, 3)

        # create dataframe
        df = pd.DataFrame(
            {
                "r": bin_centers,
                "σ_R": sigma_r,
                "Δσ_R": delta_sigma_r,
                "σ_T": sigma_t,
                "Δσ_T": delta_sigma_t,
            }
        )
        # drop rows with NaNs
        df = df.dropna()

        # write to file
        df.to_csv(f"./raw_data/{self.cluster_name}_hubble_pm.csv", index=False, header=True)
        metadata["hubble_mean_mass"] = mean_mass

        # Gaia PM
        (
            bin_centers,
            sigma_r,
            delta_sigma_r,
            sigma_t,
            delta_sigma_t,
            mean_mass,
        ) = self.gaia_PMs()

        # round to 3 decimal places
        bin_centers = np.round(bin_centers, 3)
        sigma_r = np.round(sigma_r, 3)
        delta_sigma_r = np.round(delta_sigma_r, 3)
        sigma_t = np.round(sigma_t, 3)
        delta_sigma_t = np.round(delta_sigma_t, 3)
        mean_mass = np.round(mean_mass, 3)

        # create dataframe
        df = pd.DataFrame(
            {
                "r": bin_centers,
                "σ_R": sigma_r,
                "Δσ_R": delta_sigma_r,
                "σ_T": sigma_t,
                "Δσ_T": delta_sigma_t,
            }
        )
        # drop rows with NaNs
        df = df.dropna()

        # write to file
        df.to_csv(f"./raw_data/{self.cluster_name}_gaia_pm.csv", index=False, header=True)
        metadata["gaia_mean_mass"] = mean_mass

        # LOS Dispersion
        bin_centers, sigmas, delta_sigmas, mean_mass = self.LOS_dispersion()

        # round to 3 decimal places
        bin_centers = np.round(bin_centers, 3)
        sigmas = np.round(sigmas, 3)
        delta_sigmas = np.round(delta_sigmas, 3)
        mean_mass = np.round(mean_mass, 3)

        # create dataframe
        df = pd.DataFrame(
            {
                "r": bin_centers,
                "σ": sigmas,
                "Δσ": delta_sigmas,
            }
        )
        # drop rows with NaNs
        df = df.dropna()

        # write to file
        df.to_csv(
            f"./raw_data/{self.cluster_name}_los_dispersion.csv",
            index=False,
            header=True,
        )
        metadata["los_mean_mass"] = mean_mass

        # Number Density
        (
            bin_centers,
            number_density,
            delta_number_density,
            mean_mass,
        ) = self.number_density()

        # round to 3 decimal places
        bin_centers = np.round(bin_centers, 3)
        number_density = np.round(number_density, 3)
        delta_number_density = np.round(delta_number_density, 3)
        mean_mass = np.round(mean_mass, 3)

        df = pd.DataFrame(
            {
                "rad": bin_centers,
                "density": number_density,
                "density_err": delta_number_density,
            }
        )
        # drop rows with NaNs
        df = df.dropna()

        # write to file
        df.to_csv(
            f"./raw_data/{self.cluster_name}_number_density.csv",
            index=False,
            header=True,
        )
        metadata["number_mean_mass"] = mean_mass

        # Mass Function

        # area of innermost annulus
        # for only the outer 2 annuli:
        # calculate the inner and outer radii of each annulus, based on the
        # centers of the annuli matching the desired area
        # For now lets normalize to the 47 Tuc Heinke exposure, eg 0-1.5' annulus
        inner_annuli = [
            (0, 0.4),
            (0.4, 0.8),
            (0.8, 1.2),
            (1.2, 1.6),
        ]
        outer_annuli_centers = [2.5, 5] << u.arcmin

        # target area of outer annuli
        area = np.pi * 1.5 * u.arcmin**2

        # build the outer annuli
        outer_annuli = []
        for i in range(len(outer_annuli_centers)):
            w = area / (4 * np.pi * outer_annuli_centers[i])
            outer_annuli.append(
                (
                    outer_annuli_centers[i].value - w.value,
                    outer_annuli_centers[i].value + w.value,
                )
            )

        # concat the inner and outer annuli
        annuli = inner_annuli + outer_annuli
        logging.info(f"MF: Annuli: {annuli}")

        r_ins = []
        r_outs = []
        m1s = []
        m2s = []
        mass_functions = []
        delta_mass_functions = []

        for i in range(len(annuli)):
            mass_edges, mass_function, delta_mass_function = self.mass_function(r_in=annuli[i][0], r_out=annuli[i][1])
            for _ in range(len(mass_function)):
                r_ins.append(annuli[i][0])
                r_outs.append(annuli[i][1])
            m1 = mass_edges[:-1]
            m2 = mass_edges[1:]
            m1s.append(m1.flatten())
            m2s.append(m2.flatten())
            mass_functions.append(mass_function.flatten())
            delta_mass_functions.append(delta_mass_function.flatten())

        # need to flatten lists before converting to numpy arrays, not sure why this broke things,
        # somehow related to making arrays from ragged lists and its recent deprecation
        m1s = [item for elem in m1s for item in elem]
        m2s = [item for elem in m2s for item in elem]
        mass_functions = [item for elem in mass_functions for item in elem]
        delta_mass_functions = [item for elem in delta_mass_functions for item in elem]

        # convert to numpy arrays
        r_ins = np.array(r_ins).flatten()
        r_outs = np.array(r_outs).flatten()
        m1s = np.array(m1s).flatten()
        m2s = np.array(m2s).flatten()
        mass_functions = np.array(mass_functions).flatten()
        delta_mass_functions = np.array(delta_mass_functions).flatten()

        # round to 3 decimal places
        r_ins = np.round(r_ins, 3)
        r_outs = np.round(r_outs, 3)
        m1s = np.round(m1s, 3)
        m2s = np.round(m2s, 3)
        mass_functions = np.round(mass_functions, 3)
        delta_mass_functions = np.round(delta_mass_functions, 3)

        # create dataframe
        df = pd.DataFrame(
            {
                "r1": r_ins,
                "r2": r_outs,
                "m1": m1s,
                "m2": m2s,
                "N": mass_functions,
                "ΔN": delta_mass_functions,
            }
        )

        # drop rows with NaNs
        df = df.dropna()

        # write to file
        df.to_csv(
            f"./raw_data/{self.cluster_name}_mass_function.csv",
            index=False,
            header=True,
        )

        # save metadata
        with open(f"{self.cluster_name}_metadata.json", "w", encoding="utf8") as f:
            json.dump(metadata, f, indent=4)

    def create_datafile(self, cluster_name, include_masses=False):
        """
        Create a GCfit datafile from the synthetic data.

        Parameters
        ----------
        cluster_name : str
            Name of the cluster.
        include_masses : bool
            Whether to include the mean masses of each dataset in the Observations file. Default is False.
            We wouldn't know this in the real world, so we don't include it by default but it can be useful for testing.
        """

        # first, write out the data just in case it hasn't been done yet
        self.write_obs()

        # read metadata (unneeded for now, was used to set mean masses of datasets)
        with open(f"{self.cluster_name}_metadata.json", encoding="utf8") as f:
            metadata = json.load(f)

        # initialize datafile
        cf = ClusterFile(cluster_name, force_new=True)

        # add metadata
        FeH = np.log10(self.snapshot.z / 0.02)
        cf.add_metadata("FeH", FeH)
        cf.add_metadata("age", self.snapshot.age)
        cf.add_metadata("distance", self.snapshot.dist)
        cf.add_metadata("l", 0.0)
        cf.add_metadata("b", 0.0)
        cf.add_metadata("RA", 0.0)
        cf.add_metadata("DEC", 0.0)
        cf.add_metadata("μ", 0.0)
        cf.add_metadata("Ndot", 0.0)
        cf.add_metadata("vesc", self.snapshot.vesc_initial)

        # start with radial velocity data

        LOS_fn = pathlib.Path(f"./raw_data/{self.cluster_name}_los_dispersion.csv")

        err = {"Δσ": "σ"}
        units = {"r": "arcsec", "σ": "km/s", "Δσ": "km/s"}
        keys = "r", "σ", "Δσ"

        LOS = Dataset("velocity_dispersion/LOS")

        LOS.read_data(LOS_fn, delim=r",", keys=keys, units=units, errors=err)
        if include_masses:
            LOS.add_metadata("m", float(metadata["los_mean_mass"]))

        LOS.add_metadata("source", "LOS Data")
        cf.add_dataset(LOS)

        # Now the proper motion data

        # start with the Hubble data

        Hubble_fn = pathlib.Path(f"./raw_data/{self.cluster_name}_hubble_pm.csv")

        # keys = 'r', 'σ_tot', 'Δσ_tot', 'σ_R', 'Δσ_R', 'σ_T', 'Δσ_T'
        keys = "r", "σ_R", "Δσ_R", "σ_T", "Δσ_T"

        names = {"σ_R": "PM_R", "σ_T": "PM_T"}
        err = {"Δσ_R": "PM_R", "Δσ_T": "PM_T"}
        units = {
            "r": "arcsec",
            "σ_R": "mas/yr",
            "σ_T": "mas/yr",
            "Δσ_R": "mas/yr",
            "Δσ_T": "mas/yr",
        }
        PM = Dataset("proper_motion/Hubble")

        PM.read_data(Hubble_fn, delim=r",", keys=keys, units=units, errors=err, names=names)
        if include_masses:
            PM.add_metadata("m", float(metadata["hubble_mean_mass"]))

        PM.add_metadata("source", "HST")

        cf.add_dataset(PM)

        # Now the Gaia data
        Gaia_fn = pathlib.Path(f"./raw_data/{self.cluster_name}_gaia_pm.csv")

        keys = "r", "σ_R", "Δσ_R", "σ_T", "Δσ_T"

        names = {"σ_R": "PM_R", "σ_T": "PM_T"}
        err = {"Δσ_R": "PM_R", "Δσ_T": "PM_T"}
        units = {
            "r": "arcsec",
            "σ_R": "mas/yr",
            "σ_T": "mas/yr",
            "Δσ_R": "mas/yr",
            "Δσ_T": "mas/yr",
        }
        PM = Dataset("proper_motion/Gaia")

        PM.read_data(Gaia_fn, delim=r",", keys=keys, units=units, errors=err, names=names)

        if include_masses:
            PM.add_metadata("m", float(metadata["gaia_mean_mass"]))

        PM.add_metadata("source", "Gaia")
        cf.add_dataset(PM)

        # now the number density data

        ND_fn = pathlib.Path(f"./raw_data/{self.cluster_name}_number_density.csv")

        units = {"rad": "arcmin", "density": "1/arcmin2", "density_err": "1/arcmin2"}
        names = {"rad": "r", "density": "Σ"}
        err = {"density_err": "Σ"}

        ND = Dataset("number_density")

        ND.read_data(ND_fn, delim=r",", units=units, errors=err, names=names)

        if include_masses:
            ND.add_metadata("m", float(metadata["number_mean_mass"]))

        # Set the background level, zero in this case
        bg = 0.0
        ND.add_metadata("background", bg)

        ND.add_metadata("source", "Gaia + HST")
        cf.add_dataset(ND)

        # now the mass function data

        keys = ("r1", "r2", "m1", "m2", "N", "ΔN")
        units = {"r1": "arcmin", "r2": "arcmin", "m1": "Msun", "m2": "Msun"}
        err = {"ΔN": "N"}
        fields = {}
        # just a huge square
        fields["CMC"] = {"a": np.array([[2.0, 2.0], [2.0, -2.0], [-2.0, -2.0], [-2.0, 2.0]], dtype="f")}
        MF = Dataset("mass_function/CMC")
        mf_df = pd.read_csv(f"./raw_data/{self.cluster_name}_mass_function.csv")

        MF.read_data(mf_df, keys=keys, units=units, errors=err)

        logging.info(f"MF: {fields = }")
        fld = fields["CMC"]
        logging.info(f"MF: {fld = }")
        MF.add_variable("fields", h5py.Empty("f"), "deg", fld)
        MF.add_metadata("field_unit", "deg")

        MF.add_metadata("source", "HST")
        MF.add_metadata("proposal", "CMC-obs")
        cf.add_dataset(MF)

        # write the datafile
        cf.save(force=True)

    def write_cluster_data(self):
        """
        Write a bunch of random data to a file.
        Includes cluster mass, half-mass radius, distance,
        FeH, Age and a whole bunch of info about the BH population.
        """
        info = {}
        info["name"] = self.snapshot.name
        info["mass"] = self.snapshot.mass
        info["initial_mass"] = self.snapshot.initial_mass
        info["r_h"] = self.snapshot.rh
        info["r_c"] = self.snapshot.rcore
        info["r_t"] = self.snapshot.rtidal
        info["distance"] = self.snapshot.dist
        info["FeH"] = self.snapshot.FeH
        info["age"] = self.snapshot.age
        info["BH_total_mass"] = self.snapshot.M_BH
        info["BH_total_number"] = self.snapshot.N_BH
        info["BH_masses"] = self.snapshot.bh_masses
        info["BH_radii_proj"] = self.snapshot.bh_radii_proj
        info["BH_radii_3d"] = self.snapshot.bh_radii_3d
        info["vesc_initial"] = self.snapshot.vesc_initial
        info["vesc_final"] = self.snapshot.vesc_final
        info["rho_0"] = self.snapshot.rho0_MSUN_pc3
        info["Trh"] = self.snapshot.Trh

        # write the data
        with open(f"{self.cluster_name}_cluster_info.json", "w", encoding="utf8") as f:
            json.dump(info, f, indent=4)


def gaia_err_func(G):
    """
    Get approximate errors for Gaia DR3 astrometric measurements.
    Uses average of RA and dec uncertainties from Table 4 of
    # https://www.aanda.org/articles/aa/abs/2021/05/aa39709-20/aa39709-20.html

    Parameters
    ----------
    G : float
        Gaia G-band magnitude.

    Returns
    -------
    err : float
        Error in mas/yr.
    """

    mags = np.linspace(12, 21, 10)
    errs = [0.017, 0.015, 0.018, 0.026, 0.041, 0.067, 0.117, 0.2185, 0.4575, 1.423]
    # return interpolated error, filling in left and right edges with 0.017 and 1.423 respectively.
    return np.interp(G, mags, errs)


def ND_limiting_mass(ND):
    """
    Get a rough estimate of the limiting mass for a given number density for the mass
    function data. Uses 47 Tuc as the reference cluster because its large and nearby
    with lots of data.

    Parameters
    ----------
    ND : float
        Number density in stars/arcmin^2

    Returns
    -------
    m_lim : float
        Limiting mass in Msun
    """

    # these values are extracted from the mass function and number density data for 47 Tuc
    # did tweak that innermost density value and outermost mass value to be monotonic
    nds = [
        8555.4066905877835,
        8449.380572042252,
        3756.8946538476357,
        2344.9320491979393,
        1456.5664897698775,
        1051.6472865401154,
        721.4176477673034,
        336.27092110511785,
        235.53654529775912,
        173.15865839699967,
        129.4437097007724,
        97.66022445255905,
        61.69620878087059,
        24.41689517414916,
        0.5781700910928782,
    ]
    ms = [
        0.715,
        0.634,
        0.589,
        0.53,
        0.38,
        0.33,
        0.33,
        0.33,
        0.28,
        0.13,
        0.13,
        0.13,
        0.13,
        0.13,
        0.13,
    ]

    # interpolate the data
    lim_spl = sp.interpolate.interp1d(x=nds, y=ms, kind="linear", bounds_error=False, fill_value=(0.13, 0.8))

    # return the interpolated value
    return lim_spl(ND)
