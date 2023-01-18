from cmc_obs import kinematics
from cmcbrowser import CMCBrowser
import pytest


@pytest.fixture
def snapshot():
    browser = CMCBrowser()

    browser.load_snapshot(
        model_name="N4e5_rv1_rg8_Z0.02", ss_name="initial.snap0147.dat.gz", distance=5.0
    )
    return browser.loaded_snapshots["N4e5_rv1_rg8_Z0.02/initial.snap0147.dat.gz"]


@pytest.fixture
def kinematics_object(snapshot):
    return kinematics.Kinematics(snapshot=snapshot)


def test_LOS(kinematics_object):
    bin_centers, sigmas, delta_sigmas, mean_mass = kinematics_object.LOS_dispersion()

    assert all(sigmas > 0)
    assert all(sigmas < 30)
    assert all(delta_sigmas > 0)
    assert mean_mass > 0
    assert len(bin_centers) == len(sigmas)



def test_HubblePM(kinematics_object):
    (
        bin_centers,
        sigma_r,
        delta_sigma_r,
        sigma_t,
        delta_sigma_t,
        mean_mass,
    ) = kinematics_object.hubble_PMs()

    assert all(sigma_r > 0)
    assert all(sigma_r < 30)
    assert all(sigma_t > 0)
    assert all(sigma_t < 30)
    assert all(delta_sigma_r > 0)
    assert all(delta_sigma_t > 0)
    assert mean_mass > 0

    assert len(bin_centers) == len(sigma_r)


def test_GaiaPM(kinematics_object):
    (
        bin_centers,
        sigma_r,
        delta_sigma_r,
        sigma_t,
        delta_sigma_t,
        mean_mass,
    ) = kinematics_object.gaia_PMs()

    assert all(sigma_r > 0)
    assert all(sigma_r < 30)
    assert all(sigma_t > 0)
    assert all(sigma_t < 30)
    assert all(delta_sigma_r > 0)
    assert all(delta_sigma_t > 0)
    assert mean_mass > 0

    assert len(bin_centers) == len(sigma_r)


def test_number_density(kinematics_object):

    (
        bin_centers,
        number_density,
        delta_number_density,
    ) = kinematics_object.number_density()

    assert all(number_density > 0)
    assert all(delta_number_density > 0)
    assert all(bin_centers > 0)

    assert len(bin_centers) == len(number_density)
