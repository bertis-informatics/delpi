import pytest


def pytest_addoption(parser):
    parser.addoption(
        "--mzml", action="store", default=None, help="Path to an mzML file"
    )
    parser.addoption(
        "--raw", action="store", default=None, help="Path to a Thermo RAW file"
    )


@pytest.fixture
def mzml_path(request):
    path = request.config.getoption("--mzml")
    if path is None:
        pytest.skip("--mzml not provided")
    return path


@pytest.fixture
def raw_path(request):
    path = request.config.getoption("--raw")
    if path is None:
        pytest.skip("--raw not provided")
    return path
