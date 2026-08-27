import math
from pathlib import Path

import numpy as np
import pytest

from saltfit import Parameter, SALTConfigurationError, SALTInflowModel, Spectrum, UniformPrior


def _make_salt_installation(root: Path) -> Path:
    (root / "python").mkdir(parents=True)
    (root / "libsalt_inflows.so").touch()
    (root / "python" / "run_salt.py").write_text(
        "def Line_Profile(**kwargs):\n    return kwargs['background'] + kwargs['flow_parameters']['tau']\n"
    )
    return root


def _profile_parameters() -> dict[str, object]:
    return {"abs_waves": [2382.04]}


def _parameters() -> tuple[Parameter, ...]:
    return (
        Parameter("alpha", 1.0, UniformPrior(0.0, 3.0)),
        Parameter("psi", 0.0, UniformPrior(-1.0, 1.0)),
        Parameter("gamma", 1.0, UniformPrior(0.0, 3.0)),
        Parameter("tau", 0.2, UniformPrior(0.0, 1.0)),
        Parameter("v_0", 25.0, UniformPrior(0.0, 100.0)),
        Parameter("v_w", 500.0, UniformPrior(100.0, 1_000.0)),
        Parameter("v_ap", 50.0, UniformPrior(0.0, 100.0)),
        Parameter("f_c", 1.0, UniformPrior(0.0, 1.0)),
        Parameter("k", 0.0, UniformPrior(-1.0, 1.0)),
        Parameter("delta", 1.5, UniformPrior(0.0, 3.0)),
    )


@pytest.fixture
def model(tmp_path) -> SALTInflowModel:
    return SALTInflowModel(
        salt_path=str(_make_salt_installation(tmp_path / "salt")),
        lam_ref=2382.04,
        profile_type="pcygni",
        profile_parameters=_profile_parameters(),
        parameters=_parameters(),
    )


@pytest.fixture
def spectrum() -> Spectrum:
    return Spectrum(velocity=[-1.0, 0.0, 1.0], flux=[1.2, 1.2, 1.2], uncertainty=[0.1, 0.1, 0.1])


def test_model_loads_local_salt_and_predicts(model):
    np.testing.assert_allclose(model.predict([-1.0, 0.0, 1.0], model.initial_parameters), [1.2, 1.2, 1.2])
    assert model.parameter_names == tuple(parameter.name for parameter in _parameters())


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
        SALTInflowModel(
            lam_ref=2382.04,
            profile_type="emission",
            profile_parameters=_profile_parameters(),
            parameters=_parameters(),
        )
