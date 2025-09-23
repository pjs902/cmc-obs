## cmc-obs

This library takes in a `Snapshot` object loaded by [`cmc-browser`](https://github.com/pjs902/cmc-browser) and computes various mock
observations. In particular, this library will extract number density profiles, line-of-sight and
both radial and tangential proper motion dispersion profiles and stellar mass functions.

This all starts with the `Observations` object:

```python
import cmc_obs
k = cmc_obs.observations.Observations(snap)
```

which then handles the extraction of simulated observations which in most cases is as simple as
calling a method:

```python
# compute simulated Gaia proper motions
bin_centers, sigma_r, delta_sigma_r, sigma_t, delta_sigma_t, mean_mass = k.gaia_PMs()
```

## Controlling Observations Generation

By default, `cmc-obs` generates all observations except ERIS proper motions. You can control which observations are generated using the `observations` parameter in the `write_obs()` method:

```python
# Default behavior (excludes ERIS)
k.write_obs()

# Include ERIS observations explicitly
k.write_obs(observations=['hubble', 'gaia', 'eris', 'los', 'nd', 'mf'])

# Generate only proper motion observations
k.write_obs(observations=['hubble', 'gaia', 'eris'])

# Generate only ERIS observations
k.write_obs(observations=['eris'])
```

Available observation types:
- `'hubble'`: Hubble Space Telescope proper motions
- `'gaia'`: Gaia proper motions  
- `'eris'`: ERIS proper motions (excluded by default)
- `'los'`: Line-of-sight velocity dispersions
- `'nd'`: Number density profiles
- `'mf'`: Stellar mass functions

**Note**: ERIS observations are excluded by default since they are not typically available for real observations and should only be generated when explicitly requested for specialized studies.

Internally, filtering based on stellar types and magnitudes is done for each dataset to match
real-world performance. For example, line-of-sight velocity dispersions are limited to bright giants
while the Gaia proper motions cover a range of $13 < G< 19$.

To compute the dispersion profiles use Hamiltonian Monte Carlo, implemented in
[blackjax](https://github.com/blackjax-devs/blackjax) to sample from a Gaussian likelihood that is
implemented in [JAX](https://github.com/google/jax) which means that we don't lose all that much
speed compared to the MLE approach.

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
the needed metadata to be used directly in the fitting.

## See also
[`cmc-browser`](https://github.com/pjs902/cmc-browser): A small library for managing a local grid of CMC models and loading the models as `Snapshot` objects which are needed for `cmc-obs`.
