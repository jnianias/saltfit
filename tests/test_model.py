import math
from pathlib import Path

import numpy as np
import pytest

from saltfit import SALTConfigurationError, SALTModel, Spectrum


def _make_salt_installation(root: Path) -> Path:
    (root / "python").mkdir(parents=True)
    (root / "libsalt_inflows.so").touch()
    (root / "python" / "run_salt.py").write_text(
        "def Line_Profile(**kwargs):\n    return kwargs['background'] + kwargs['flow_parameters']['tau']\n"
    )
    return root


def _profile_parameters() -> dict[str, object]:
    return {"abs_waves": [2382.04]}


def _initial_values() -> dict[str, float]:
    return {
        "alpha": 1.0,
        "psi": 0.0,
        "gamma": 1.0,
        "tau": 0.2,
        "v_0": 25.0,
        "v_w": 500.0,
        "v_ap": 50.0,
        "f_c": 1.0,
        "k": 0.0,
        "delta": 1.5,
    }


def _bounds() -> dict[str, tuple[float, float]]:
    return {
        "alpha": (0.0, 3.0),
        "psi": (-1.0, 1.0),
        "gamma": (0.0, 3.0),
        "tau": (0.0, 1.0),
        "v_0": (0.0, 100.0),
        "v_w": (100.0, 1_000.0),
        "v_ap": (0.0, 100.0),
        "f_c": (0.0, 1.0),
        "k": (-1.0, 1.0),
        "delta": (0.0, 3.0),
    }


@pytest.fixture
def model(tmp_path) -> SALTModel:
    return SALTModel(
        salt_path=str(_make_salt_installation(tmp_path / "salt")),
        lam_ref=2382.04,
        profile_type="pcygni",
        profile_parameters=_profile_parameters(),
        initial_values=_initial_values(),
        bounds=_bounds(),
    )


@pytest.fixture
def spectrum() -> Spectrum:
    return Spectrum(velocity=[-1.0, 0.0, 1.0], flux=[1.2, 1.2, 1.2], uncertainty=[0.1, 0.1, 0.1])


def test_model_loads_local_salt_and_predicts(model):
    np.testing.assert_allclose(model.predict([-1.0, 0.0, 1.0], model.initial_parameters), [1.2, 1.2, 1.2])
    assert model.parameter_names == tuple(_initial_values())


def test_model_calculates_normalized_log_probability(model, spectrum):
    expected = -3.0 * math.log(0.1 * math.sqrt(2.0 * math.pi))

    assert model.log_probability(spectrum, model.initial_parameters) == pytest.approx(expected)


def test_model_fits_spectrum_with_mcmc(model, spectrum):
    result = model.fit_to(spectrum, n_steps=3, n_walkers=20, random_seed=42)

    assert result.chain.shape == (3, 20, 10)
    assert result.log_probability.shape == (3, 20)
    assert set(result.median_parameters()) == set(model.parameter_names)


def test_model_reports_missing_salt_configuration(monkeypatch):
    monkeypatch.delenv("SALTFIT_SALT_PATH", raising=False)

    with pytest.raises(SALTConfigurationError, match="SALTFIT_SALT_PATH"):
        SALTModel(
            lam_ref=2382.04,
            profile_type="emission",
            profile_parameters=_profile_parameters(),
            initial_values=_initial_values(),
            bounds=_bounds(),
        )


def test_model_rejects_unsupported_flow(tmp_path):
    with pytest.raises(ValueError, match="Only flow='inflow'"):
        SALTModel(
            salt_path=str(_make_salt_installation(tmp_path / "salt")),
            lam_ref=2382.04,
            profile_type="emission",
            profile_parameters=_profile_parameters(),
            initial_values=_initial_values(),
            bounds=_bounds(),
            flow="outflow",  # type: ignore[arg-type]
        )


def test_model_rejects_incomplete_parameter_configuration(tmp_path):
    initial_values = _initial_values()
    initial_values.pop("tau")

    with pytest.raises(ValueError, match="initial_values must contain exactly"):
        SALTModel(
            salt_path=str(_make_salt_installation(tmp_path / "salt")),
            lam_ref=2382.04,
            profile_type="emission",
            profile_parameters=_profile_parameters(),
            initial_values=initial_values,
            bounds=_bounds(),
        )


def test_model_rejects_invalid_aperture_velocity(model, spectrum):
    parameters = model.initial_parameters
    parameters["v_ap"] = parameters["v_w"]

    assert model.log_probability(spectrum, parameters) == -math.inf
