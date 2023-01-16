from cmc_obs import kinematics
from cmcbrowser import CMCBrowser
import pytest


@pytest.fixture
def snapshot():
    browser = CMCBrowser()

    browser.load_snapshot(
        model_name="N4e5_rv1_rg8_Z0.02", ss_name="initial.snap0158.dat.gz", distance=3.5
    )
    return browser.loaded_snapshots["N4e5_rv1_rg8_Z0.02/initial.snap0158.dat.gz"]


@pytest.fixture
def kinematics_object(snapshot):
    return kinematics.Kinematics(snapshot=snapshot)


def test_LOS(kinematics_object):
    kinematics_object.LOS_dispersion(stars_per_bin=30)
