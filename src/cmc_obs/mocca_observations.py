import pathlib

import numpy as np
import pandas as pd
import cmctoolkit as ck
from astropy import units as u


# Full list of columns in MOCCA snapshots.
_MOCCA_colnames = (
    "im",
    "x",
    "y",
    "z",
    "vx",
    "vy",
    "vz",
    "idd1",
    "idd2",
    "ikb",
    "ik1",
    "ik2",
    "sm1",
    "sm2",
    "slum1",
    "slum2",
    "rad1",
    "rad2",
    "spin1",
    "spin2",
    "hist1",
    "hist2",
    "a",
    "ecc",
    "mv1",
    "mv2",
    "mu1",
    "mu2",
    "mb1",
    "mb2",
    "mi1",
    "mi2",
    "mtr1",
    "mtr2",
    "popid1",
    "popid2",
    "r",
    "vr",
    "vt",
    "u_potential"
)


# Map MOCCA column names to CMC column names.
_MOCCA_CMC_MAPPING = {
    "ikb": "binflag",
    "ik1": "bin_startype0",
    "ik2": "bin_startype1",
    "r": "r[PC]",
    "sm1": "m0_MSUN",
    "sm2": "m1_MSUN",
    "x": "x[PC]",
    "y": "y[PC]",
    "z": "z[PC]",
    "vx": "vx[KM/S]",
    "vy": "vy[KM/S]",
    "vz": "vz[KM/S]",
    "slum1": "bin_star_lum0_LSUN",
    "slum2": "bin_star_lum1_LSUN",
    "rad1": "bin_star_radius0_RSUN",
    "rad2": "bin_star_radius1_RSUN",
}


class MoccaSnapshot:
    '''Snapshot class recreating `cmctoolkit.Snapshot` for MOCCA snapshots.

    This class spoofs and maps everything necessary to recreate enough
    of `cmctoolkit.Snapshot` in order to create mock observations using the
    same `.observations.Observations` class, for a MOCCA model snapshot, instead
    of CMC. It maps all necessary data, column names and attributes to those
    used for the CMC snapshots, and computes the relevant projections and
    photometry.

    This class should be used to create mock observations in the exact same
    way as normal snapshots, with the note that it is not required for
    photometry to be computed again.

    Note that some attributes are spoofed as NaN or empty, as they are not
    directly used in creating observations, so this class won't recreate the
    _entire_ functionality of `cmctoolkit`.

    Parameters
    ----------
    fname : str
        Path to the MOCCA snapshot file.

    dist : float
        Distance to the cluster in kiloparsecs (kpc).

    z : float
        Absolute metallicity of the cluster.

    age : float, optional
        Age of the cluster in Gyr. Defaults to 12 Gyr.

    system_fname : str, optional
        Path to the `system.dat` file, which contains the evolution of
        various cluster properties and is necessary to compute some initial
        quantities.
        If not provided, it is assumed to be in the same directory as the
        snapshot file, with the name "system.dat".

    Examples
    --------
    >>> snap = MoccaSnapshot('./snap12.dat', dist=5, z=0.0005)
    >>> obs = cmc_obs.Observations(snap, cluster_name='mocca1',
                                   add_photometry=False)
    '''

    # Steal these from `cmctoolkit` without inheriting anything else.
    add_photometry = ck.Snapshot.add_photometry
    calc_Teff = ck.Snapshot.calc_Teff
    coldict = ck.coldict_h5

    def __init__(self, fname, dist, z, age=12, system_fname=None):

        # Try to read in system.dat file
        if system_fname is None:
            system_fname = pathlib.Path(fname).parent / "system.dat"

        system = np.loadtxt(system_fname).T

        self.dist = dist  # [kpc]
        self.age = age
        self.z = z

        # Columns we might actually care about
        colnames = [
            "x", "y", "z", "vx", "vy", "vz", "ikb", "ik1", "ik2",
            "sm1", "sm2", "slum1", "slum2", "rad1", "rad2", "mv1", "mv2",
            "mu1", "mu2", "mb1", "mb2", "mi1", "mi2", "r"
        ]

        # Read in snapshot data

        self.data = pd.read_table(fname, sep=r"\s+",
                                  names=_MOCCA_colnames, usecols=colnames)

        # Map names to match CMC

        self.data.rename(_MOCCA_CMC_MAPPING, inplace=True, axis='columns')

        # Recreate all single stars

        self.data.loc[self.data['binflag'] == 2, 'binflag'] = 1
        sing = self.data['binflag'] == 0

        # Mass is sum of both stars when binary
        self.data['m_MSUN'] = self.data['m0_MSUN'] + self.data['m1_MSUN']

        # Some quantities are set to -100 in total column when binary
        self.data['startype'] = -1 * np.ones_like(self.data['bin_startype0'])
        self.data.loc[sing, 'startype'] = self.data.loc[sing, 'bin_startype0']

        self.data['luminosity_LSUN'] = -1 * np.ones_like(self.data['bin_star_lum0_LSUN'])
        self.data.loc[sing, 'luminosity_LSUN'] = self.data.loc[sing, 'bin_star_lum0_LSUN']

        self.data['radius_RSUN'] = -1 * np.ones_like(self.data['bin_star_radius0_RSUN'])
        self.data.loc[sing, 'radius_RSUN'] = self.data.loc[sing, 'bin_star_radius0_RSUN']

        # This compatibility layer is needed in some `Observations` methods
        self.data["m[MSUN]"] = self.data["m_MSUN"]
        self.data["m0[MSUN]"] = self.data["m0_MSUN"]
        self.data["m1[MSUN]"] = self.data["m1_MSUN"]
        self.data["luminosity[LSUN]"] = self.data["luminosity_LSUN"]

        # Recreate the final projections needed (not all in MOCCA projections)

        theta = np.arccos(self.data['z[PC]'] / self.data['r[PC]'])
        d = self.data['r[PC]'] * np.sin(theta)
        phi = np.arccos(self.data['x[PC]'] / d)

        vd = (self.data['vx[KM/S]'] * np.cos(phi)
              + self.data['vy[KM/S]'] * np.sin(phi))
        va = (-self.data['vx[KM/S]'] * np.sin(phi)
              + self.data['vy[KM/S]'] * np.cos(phi))

        self.data["d[PC]"] = d
        self.data["vd[KM/S]"] = vd
        self.data["va[KM/S]"] = va


        # ATTRIBUTES
        # Note that many of these are not from cmctoolkit but added to the
        # snapshot by `cmc-browser`. Many of these only seem to exist for
        # `write_cluster_data` which isn't used to make mock obs, so I'll just
        # be lazy and spoof them as empty for now (but could read from system
        # if really needed)

        # age (fixed to 12 Gyr in the given MOCCA snapshots)
        # self.age = 12

        # z (Set to 0.0005 in mocca.ini)
        # self.z = 0.0005

        # vesc_initial
        M0 = system[2][0]
        rh0 = system[16][0]  # this is "actual r_h", 18 is 2D r_h
        rhoh0 = 3 * M0 / (8 * np.pi * rh0 ** 3) # [Msun / pc^3]
        vesc0 = 50 * (M0 / 1e5) ** (1./3) * (rhoh0 / 1e5) ** (1./6)
        self.vesc_initial = vesc0

        # name (TODO if this is important, should make match cmc better)
        self.name = fname

        # mass
        self.mass = self.data['m0_MSUN'].sum() + self.data['m1_MSUN'].sum()

        # Spoofing these assuming they aren't really important
        self.FeH = np.log10(self.z / 0.02)
        self.initial_mass = M0
        self.rh = np.nan
        self.rcore = np.nan
        self.rtidal = np.nan
        self.M_BH = np.nan
        self.N_BH = np.nan
        self.bh_masses = np.nan
        self.bh_radii_proj = np.nan
        self.bh_radii_3d = np.nan
        self.vesc_final = np.nan
        self.rho0_MSUN_pc3 = np.nan
        self.Trh = np.nan

        # Recreate the photometry needed
        # TODO if add_photometry=True in Observations then this is repeated

        self.filtertable = pd.DataFrame({"filtname": [], "path": [],
                                         "zp_spectralflux[JY]": []})

        self.add_photometry(ck.load_default_filters())

    def convert_units(self, quantity, unit_in, unit_out):
        '''Spoof Snapshot.convert_units to just use astropy instead.'''

        # TODO won't work if called with "code" units"
        cmc_name_mapping = {
            'rsun': 'Rsun',
            'lsun': 'Lsun',
            'jy': 'Jy',
            'gyr': 'Gyr',
        }
        unit_in = u.Unit(cmc_name_mapping.get(unit_in, unit_in))
        unit_out = u.Unit(cmc_name_mapping.get(unit_out, unit_out))
        return (quantity << unit_in).to_value(unit_out)

    def make_2d_projection(self):
        '''Spoof, as projections already created or read in above'''
        return
