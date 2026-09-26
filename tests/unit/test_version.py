from importlib.metadata import version

import dfwb


def test_version_matches_distribution_metadata():
    assert dfwb.__version__ == version("deepfake-workbench") == "0.1.0b3.dev0"
