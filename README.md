## cmc-obs

This library takes in a `Snapshot` object loaded by `cmc-browser` and computed various mock
observations. In particular, this library will extract number density profiles, line-of-sight and
both radial and tangential proper motion dispersion profiles and stellar mass functions.

This all starts with the `Observations` object:

```python
import cmc_obs
k = cmc_obs.observations.Observations(snap)
```

which then handles the extraction of simulated observations which in most cases is a simple as
calling a method:

```python
# compute simpulated Gaia proper motions
bin_centers, sigma_r, delta_sigma_r, sigma_t, delta_sigma_t, mean_mass = k.gaia_PMs()
```

Internally, filtering based on stellar types and magnitudes are done for each dataset to match
real-world performance. For example, line-of-sight velocity dispersions are limited to bright giants
while the Gaia proper motions cover a range of $13 < G< 19$.

To compute the dispersion profiles use Hamiltonian Monte Carlo, implemented in
[blackjax](https://github.com/blackjax-devs/blackjax) to sample from a Gaussian likelihood that is
implemented in [JAX](https://github.com/google/jax) which means that we don't lose all that much
speed compared to the MLE approach (code for this
[here](https://github.com/pjs902/cmc-obs/blob/63c59aed95b2015b086173ff3f4b925f84bb8986/cmc-obs/src/cmc_obs/observations.py#L126)).

The HST and LOS dispersion profiles and number density profiles are computed in a similar way.

```python
bin_centers, sigma_r, delta_sigma_r, sigma_t, delta_sigma_t, mean_mass = k.hubble_PMs()
bin_centers, sigmas, delta_sigmas, mean_mass = k.LOS_dispersion()
bin_centers, number_density, delta_number_density, mean_mass = k.number_density()
```

The extraction of mass function data happens in a similar way, though it only operates in a single
annulus at a time:

```python
mass_edges, mass_function, delta_mass_function = k.mass_function(r_in=0, r_out=0.4)
```

This method will additionally compute the number density within the annulus and use that to
calculate a reasonable limiting mass for the annulus. This replicates the crowding-based effects we
see in the real data and is based on the performance of the mass function data available for 47 Tuc.

Finally, this library will handle wrangling the data into the format that `GCfit` expects for its
`ClusterFile` object and will output the data in a fully formed GCfit datafile, populated with all
the needed metadata to be used directly in fitting.

## Snapshot matching and analysis

Aside from those two libraries, this repo also contains scripts for matching snapshots to real
clusters, generating mock data from those snapshots, fitting to the mock data and analyzing the
fits.

### Snapshot matching

The matching process is located
[here](https://github.com/pjs902/CMC-validation/blob/main/snapshot-search/search.py) and is essentially a
direct port of the code presented in Rui et al. 2021, with a few modifications to save all the
snapshots that match a given cluster and use updated data sources for velocity dispersions and
orbital parameters of GCs.

This script outputs a list of all matching snapshots for each GC in the VDP, SBP and orbital tables.
This output is then fed to the datafile generation script (located
[here](https://github.com/pjs902/CMC-validation/blob/main/snapshot-search/create_datafiles.py)). This
script loops through each GC and attempts to load the best matching snapshot. This will occasionally
fail because there are more snapshots per model in the pre-extracted SBPs and VDPs that we match
against than exist in the public grid. In these cases we simply go down the list of best matching
snapshots until we either find an existing snapshot or exhaust the list of snapshots. This script
output all the GCfit datafiles for the selected snapshots into the usual GCfit directory, ready for
fitting.

There are a few other convenience/infrastructure scripts for fitting, that do things like create a
neat directory layout for fitting, create SLURM job scripts and priors for each snapshot and move
metadata files into the corresponding directories.  

### Analysis

Finally, there are a couple of plotting scripts which create the CI model fit plots for every
snapshot and generate useful plots from the CMC models like, the number of BHs, half-mass radius,
core radius and escape velocity over time, histograms of remnant populations and cumulative mass
profiles with the GCfit fits over plotted.

Analysis of population-level performance is
[this](https://github.com/pjs902/CMC-validation/blob/main/snapshot-search/snapshot_checks.ipynb) notebook
where the results of all the fits are collected and looked at as a whole.
