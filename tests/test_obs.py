from cmc_obs import *
from cmcbrowser import CMCBrowser
import pytest


@pytest.fixture
def snapshot():
    browser = CMCBrowser()

    browser.load_snapshot(
        model_name="N4e5_rv1_rg8_Z0.02", ss_name="initial.snap0158.dat.gz"
    )
    return browser.loaded_snapshots["N4e5_rv1_rg8_Z0.02"]["initial.snap0158.dat.gz"]

