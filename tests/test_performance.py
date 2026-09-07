import os
from pathlib import Path

import numpy as np
import pytest

from saltfit import SALTModel, Spectrum

pytestmark = pytest.mark.integration


@pytest.fixture
def salt_model() -> SALTModel:
    salt_path = os.environ.get("SALTFIT_SALT_PATH")
    if salt_path is None:
        pytest.skip("Set SALTFIT_SALT_PATH to run SALT performance benchmarks.")

    root = Path(salt_path)
    if not (root / "python" / "run_salt.py").is_file() or not (root / "libsalt.so").is_file():
        pytest.skip("SALTFIT_SALT_PATH must point to a built Linux SALT checkout.")

    initial_values = {
        "alpha": np.pi / 4.0,
        "psi": 0.0,
        "gamma": 1.0,
        "tau": 0.1,
        "v_0": 25.0,
        "v_w": 500.0,
        "f_c": 1.0,
        "k": 0.0,
        "delta": 1.5,
    }
    bounds = {
        "alpha": (0.001, np.pi / 2.0),
        "psi": (0.0, np.pi / 2.0),
        "gamma": (0.01, 3.0),
        "tau": (0.0, 10.0),
        "v_0": (1.0, 200.0),
        "v_w": (50.0, 1_500.0),
        "f_c": (0.0, 1.0),
        "k": (0.0, 3.0),
        "delta": (0.01, 3.0),
    }
    profile_parameters = {
        "absorption_parameters": {"abs_waves": [2343.49], "abs_osc_strs": [0.114]},
        "emission_parameters": {
            "em_waves": [2343.49, 2343.49, 2343.49],
            "emitted_waves": [2343.49, 2364.83, 2380.76],
            "em_osc_strs": [0.114, 0.114, 0.114],
            "res": [True, False, False],
            "fluor": [False, True, True],
            "p_r": [0.6578, 0.6578, 0.6578],
            "p_f": [0.3422, 0.2243, 0.1179],
            "line_num": [3],
        },
        "observing_parameters": {"APERTURE": True, "v_ap": 400.0},
        "miscellaneous_parameters": {"OCCULTATION": True},
    }
    return SALTModel(
        lam_ref=2343.49,
        profile_type="pcygni",
        profile_parameters=profile_parameters,
        initial_values=initial_values,
        bounds=bounds,
        flow="inflow",
        salt_path=root,
    )


@pytest.fixture
def benchmark_inputs(salt_model: SALTModel) -> tuple[np.ndarray, Spectrum]:
    velocity = np.linspace(-1000.0, 5500.0, 400)
    spectrum = Spectrum(velocity, np.ones_like(velocity), np.full_like(velocity, 0.02))
    return velocity, spectrum


def test_benchmark_compute_spectrum(benchmark, salt_model, benchmark_inputs):
    velocity, _ = benchmark_inputs

    benchmark(salt_model.compute_spectrum, velocity, salt_model.initial_parameters)


def test_benchmark_log_probability(benchmark, salt_model, benchmark_inputs):
    _, spectrum = benchmark_inputs

    benchmark(salt_model.log_probability, spectrum, salt_model.initial_parameters)
