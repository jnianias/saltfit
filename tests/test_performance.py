import os
from pathlib import Path

import numpy as np
import pytest

from saltfit import SALTModel, Spectrum

pytestmark = pytest.mark.integration


@pytest.fixture(params=("inflow", "outflow"), ids=("inflow", "outflow"))
def salt_model(request: pytest.FixtureRequest) -> SALTModel:
    flow = request.param
    salt_path = os.environ.get("SALTFIT_SALT_PATH")
    if salt_path is None:
        pytest.skip("Set SALTFIT_SALT_PATH to run SALT integration tests.")

    root = Path(salt_path)
    library_exists = (root / "libsalt.so").is_file() or (root / "libsalt.dylib").is_file()
    if not (root / "python" / "run_salt.py").is_file() or not library_exists:
        pytest.skip("SALTFIT_SALT_PATH must point to a built SALT checkout.")

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
    if flow == "outflow":
        initial_values["v_b"] = 10.0
        bounds["v_b"] = (0.0, 100.0)
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
    if flow == "outflow":
        profile_parameters["absorption_parameters"]["abs_ein"] = [1.73e8]
        profile_parameters["emission_parameters"].update({
            "em_ein": [1.73e8, 1.73e8, 1.73e8],
        })
        profile_parameters["blending_parameters"] = {
            "blended_waves": [[0.0], [0.0], [0.0]],
            "blended_osc_strs": [[0.0], [0.0], [0.0]],
            "blended_abs_ein": [[0.0], [0.0], [0.0]],
            "blended_fluor": [[False], [False], [False]],
            "blended_p_r": [[0.0], [0.0], [0.0]],
            "blended_p_f": [[[0.0]], [[0.0]], [[0.0]]],
            "blended_flour_waves": [[[0.0]], [[0.0]], [[0.0]]],
            "blending": [False, False, False],
        }
        profile_parameters["miscellaneous_parameters"].update({"Sobolev": True, "SW": 100.0})
    return SALTModel(
        lam_ref=2343.49,
        profile_type="pcygni",
        profile_parameters=profile_parameters,
        initial_values=initial_values,
        bounds=bounds,
        flow=flow,
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


def test_official_salt_returns_finite_spectrum(salt_model, benchmark_inputs):
    velocity, _ = benchmark_inputs

    output = salt_model.compute_spectrum(velocity, salt_model.initial_parameters)

    assert output.shape == velocity.shape
    assert np.isfinite(output).all()


def test_official_salt_evaluation_is_deterministic(salt_model, benchmark_inputs):
    velocity, _ = benchmark_inputs

    first = salt_model.compute_spectrum(velocity, salt_model.initial_parameters)
    second = salt_model.compute_spectrum(velocity, salt_model.initial_parameters)

    np.testing.assert_array_equal(first, second)


def test_official_salt_responds_to_parameter_change(salt_model, benchmark_inputs):
    velocity, _ = benchmark_inputs
    initial_flux = salt_model.compute_spectrum(velocity, salt_model.initial_parameters)
    changed_parameters = salt_model.initial_parameters
    changed_parameters["tau"] *= 1.5

    changed_flux = salt_model.compute_spectrum(velocity, changed_parameters)

    assert not np.array_equal(initial_flux, changed_flux)


def test_official_salt_log_probability_is_finite(salt_model, benchmark_inputs):
    _, spectrum = benchmark_inputs

    assert np.isfinite(salt_model.log_probability(spectrum, salt_model.initial_parameters))


def test_official_salt_mcmc_smoke(salt_model, benchmark_inputs):
    _, spectrum = benchmark_inputs

    result = salt_model.fit_to(spectrum, n_steps=2, n_walkers=2 * len(salt_model.parameter_names), random_seed=7)

    assert result.chain.shape == (2, 2 * len(salt_model.parameter_names), len(salt_model.parameter_names))
    assert np.isfinite(result.log_probability).all()
