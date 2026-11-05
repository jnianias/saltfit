import math
from pathlib import Path

import numpy as np
import pytest

from saltfit import (
    FitResult,
    SALTConfigurationError,
    SALTModel,
    Spectrum,
    SystemicAbsorptionModel,
    calculate_information_criteria,
    load_fit,
    load_salt_model,
)


def _make_salt_installation(root: Path) -> Path:
    (root / "python").mkdir(parents=True)
    (root / "libsalt.so").touch()
    (root / "python" / "run_salt.py").write_text(
        "LAST_KWARGS = None\n"
        "def salt(**kwargs):\n"
        "    global LAST_KWARGS\n"
        "    LAST_KWARGS = kwargs\n"
        "    if kwargs['profile_parameters']['miscellaneous_parameters'].get('return_nan'):\n"
        "        return kwargs['background'] * float('nan')\n"
        "    if kwargs['model_type'] == 'outflow':\n"
        "        return kwargs['background'] + kwargs['flow_parameters']['v_b']\n"
        "    return kwargs['background'] + kwargs['flow_parameters']['tau']\n"
    )
    return root


def _profile_parameters() -> dict[str, object]:
    return {
        "absorption_parameters": {"abs_waves": [2382.04], "abs_osc_strs": [0.32]},
        "emission_parameters": {"em_waves": [2382.04], "emitted_waves": [2382.04], "em_osc_strs": [0.32]},
        "observing_parameters": {"v_ap": 50.0, "APERTURE": True},
        "miscellaneous_parameters": {"OCCULTATION": True},
    }


def _initial_values() -> dict[str, float]:
    return {
        "alpha": 1.0,
        "psi": 0.0,
        "gamma": 1.0,
        "tau": 0.2,
        "v_0": 25.0,
        "v_w": 500.0,
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
        "f_c": (0.0, 1.0),
        "k": (-1.0, 1.0),
        "delta": (0.0, 3.0),
    }


def _systemic_absorption_initial_values() -> dict[str, float]:
    return _initial_values() | {"systemic_absorption_sigma": 2.0, "systemic_absorption_depth": 0.3}


def _systemic_absorption_bounds() -> dict[str, tuple[float, float]]:
    return _bounds() | {"systemic_absorption_sigma": (0.1, 10.0), "systemic_absorption_depth": (0.0, 1.0)}


@pytest.fixture
def model(tmp_path) -> SALTModel:
    return SALTModel(
        salt_path=str(_make_salt_installation(tmp_path / "salt")),
        lam_ref=2382.04,
        profile_type="pcygni",
        profile_parameters=_profile_parameters(),
        initial_values=_initial_values(),
        bounds=_bounds(),
        flow="inflow",
    )


@pytest.fixture
def spectrum() -> Spectrum:
    return Spectrum(velocity=[-1.0, 0.0, 1.0], flux=[1.2, 1.2, 1.2], uncertainty=[0.1, 0.1, 0.1])


def test_model_loads_local_salt_and_computes_spectrum(model):
    np.testing.assert_allclose(model.compute_spectrum([-1.0, 0.0, 1.0], model.initial_parameters), [1.2, 1.2, 1.2])
    assert model.parameter_names == tuple(_initial_values())


def test_model_adds_fitted_systemic_gaussian_absorption(tmp_path):
    model = SALTModel(
        salt_path=str(_make_salt_installation(tmp_path / "salt")),
        lam_ref=2382.04,
        profile_type="pcygni",
        profile_parameters=_profile_parameters(),
        initial_values=_systemic_absorption_initial_values(),
        bounds=_systemic_absorption_bounds(),
        flow="inflow",
        systemic_absorption=True,
    )

    np.testing.assert_allclose(
        model.compute_spectrum([-2.0, 0.0, 2.0], model.initial_parameters),
        [1.2 - 0.3 * math.exp(-0.5), 0.9, 1.2 - 0.3 * math.exp(-0.5)],
    )
    assert model.parameter_names[-2:] == ("systemic_absorption_sigma", "systemic_absorption_depth")


@pytest.mark.parametrize(
    ("name", "value"),
    [("systemic_absorption_sigma", 0.0), ("systemic_absorption_depth", -0.1), ("systemic_absorption_depth", 1.1)],
)
def test_model_rejects_invalid_systemic_absorption_parameters(tmp_path, spectrum, name, value):
    model = SALTModel(
        salt_path=str(_make_salt_installation(tmp_path / "salt")),
        lam_ref=2382.04,
        profile_type="pcygni",
        profile_parameters=_profile_parameters(),
        initial_values=_systemic_absorption_initial_values(),
        bounds=_systemic_absorption_bounds(),
        flow="inflow",
        systemic_absorption=True,
    )
    parameters = model.initial_parameters
    parameters[name] = value

    assert model.log_probability(spectrum, parameters) == -math.inf


def test_model_omits_legacy_dynamic_inputs_from_wrapper_configuration(model):
    construction_velocity = np.linspace(-1_000.0, 5_500.0, 1_000)
    profile_parameters = _profile_parameters()
    profile_parameters["observing_parameters"]["v_obs"] = construction_velocity
    profile_parameters["miscellaneous_parameters"]["lam_ref"] = 2382.04
    model = SALTModel(
        salt_path=str(model.salt_path),
        lam_ref=model.lam_ref,
        profile_type=model.profile_type,
        profile_parameters=profile_parameters,
        initial_values=model.initial_parameters,
        bounds=model.bounds,
        flow=model.flow,
    )
    evaluation_velocity = np.array([-500.0, 0.0, 300.0])

    model.compute_spectrum(evaluation_velocity, model.initial_parameters)

    wrapper_kwargs = model._evaluator._salt.__globals__["LAST_KWARGS"]
    forwarded_profile = wrapper_kwargs["profile_parameters"]
    np.testing.assert_array_equal(wrapper_kwargs["v_obs"], evaluation_velocity)
    assert "v_obs" not in forwarded_profile["observing_parameters"]
    assert "lam_ref" not in forwarded_profile["miscellaneous_parameters"]
    np.testing.assert_array_equal(profile_parameters["observing_parameters"]["v_obs"], construction_velocity)
    assert profile_parameters["miscellaneous_parameters"]["lam_ref"] == model.lam_ref


def test_model_calculates_normalized_log_probability(model, spectrum):
    log_likelihood = -3.0 * math.log(0.1 * math.sqrt(2.0 * math.pi))
    log_prior = sum(-math.log(upper - lower) for lower, upper in model.bounds.values())

    assert model.log_probability(spectrum, model.initial_parameters) == pytest.approx(log_likelihood + log_prior)


def test_model_fits_spectrum_with_mcmc(model, spectrum):
    result = model.fit_to(spectrum, n_steps=3, n_walkers=18, random_seed=42)

    assert result.chain.shape == (3, 18, 9)
    assert result.log_probability.shape == (3, 18)
    assert set(result.median_parameters()) == set(model.parameter_names)
    assert result.spectrum is not None
    assert result.acceptance_fraction.shape == (18,)
    assert result.fit_settings == {
        "n_steps": 3,
        "n_walkers": 18,
        "initial_spread": 1e-4,
        "random_seed": 42,
        "progress": False,
    }


def test_fit_archive_round_trip(tmp_path, model, spectrum):
    result = model.fit_to(
        spectrum,
        n_steps=3,
        n_walkers=18,
        random_seed=42,
        metadata={"source": "SP8826.A2744", "line": "CII1334"},
    )
    filepath = tmp_path / "previous-fit.h5"

    result.write_hdf5(filepath)
    loaded = load_fit(filepath)

    assert loaded.parameter_names == result.parameter_names
    np.testing.assert_array_equal(loaded.chain, result.chain)
    np.testing.assert_array_equal(loaded.log_probability, result.log_probability)
    np.testing.assert_array_equal(loaded.acceptance_fraction, result.acceptance_fraction)
    np.testing.assert_array_equal(loaded.spectrum.velocity, spectrum.velocity)
    np.testing.assert_array_equal(loaded.spectrum.flux, spectrum.flux)
    np.testing.assert_array_equal(loaded.spectrum.uncertainty, spectrum.uncertainty)
    assert loaded.model_configuration["profile_parameters"]["absorption_parameters"]["abs_waves"] == [2382.04]
    assert loaded.fit_settings == result.fit_settings
    assert loaded.metadata["saltfit_version"]
    assert loaded.metadata["source"] == "SP8826.A2744"

    restored_model = loaded.build_model()
    assert restored_model.parameter_names == model.parameter_names
    np.testing.assert_array_equal(
        restored_model.compute_spectrum(spectrum.velocity, restored_model.initial_parameters),
        model.compute_spectrum(spectrum.velocity, model.initial_parameters),
    )
    assert load_salt_model(filepath).parameter_names == model.parameter_names


def test_result_without_fit_context_cannot_write_archive(tmp_path):
    result = FitResult(
        parameter_names=("x",),
        chain=np.zeros((1, 1, 1)),
        log_probability=np.zeros((1, 1)),
    )

    with pytest.raises(ValueError, match="returned by SALTModel.fit_to"):
        result.write_hdf5(tmp_path / "incomplete.h5")


def test_systemic_absorption_model_fits_and_reconstructs_without_salt(tmp_path, spectrum):
    model = SystemicAbsorptionModel(
        initial_values={"sigma": 1.0, "depth": 0.2},
        bounds={"sigma": (0.1, 10.0), "depth": (0.0, 1.0)},
    )
    np.testing.assert_allclose(
        model.compute_spectrum([-1.0, 0.0, 1.0], model.initial_parameters),
        [1.0 - 0.2 * math.exp(-0.5), 0.8, 1.0 - 0.2 * math.exp(-0.5)],
    )

    result = model.fit_to(spectrum, n_steps=3, n_walkers=4, random_seed=42)
    filepath = tmp_path / "systemic-only.h5"
    result.write_hdf5(filepath)
    loaded = load_fit(filepath)
    restored_model = loaded.build_model()

    assert isinstance(restored_model, SystemicAbsorptionModel)
    np.testing.assert_array_equal(
        restored_model.compute_spectrum(spectrum.velocity, restored_model.initial_parameters),
        model.compute_spectrum(spectrum.velocity, model.initial_parameters),
    )
    assert np.isfinite(loaded.information_criteria(restored_model).aic)


def test_information_criteria_use_standard_formulas():
    criteria = calculate_information_criteria(log_likelihood=-10.0, n_parameters=3, n_observations=20)

    assert criteria.aic == pytest.approx(26.0)
    assert criteria.bic == pytest.approx(20.0 + 3.0 * math.log(20.0))


def test_fit_result_calculates_information_criteria(model, spectrum):
    result = model.fit_to(spectrum, n_steps=3, n_walkers=18, random_seed=42)

    criteria = result.information_criteria(model)

    assert criteria.n_parameters == len(model.parameter_names)
    assert criteria.n_observations == spectrum.size
    assert np.isfinite(criteria.log_likelihood)
    assert np.isfinite(criteria.aic)
    assert np.isfinite(criteria.bic)


def test_fit_result_reports_emcee_convergence_diagnostics(model, spectrum):
    result = model.fit_to(spectrum, n_steps=4, n_walkers=18, random_seed=42)

    diagnostics = result.convergence_diagnostics()

    assert diagnostics.parameter_names == model.parameter_names
    assert diagnostics.autocorrelation_time.shape == (len(model.parameter_names),)
    assert not diagnostics.converged


def test_model_reproduces_seeded_mcmc_runs(model, spectrum):
    first = model.fit_to(spectrum, n_steps=3, n_walkers=18, random_seed=42)
    second = model.fit_to(spectrum, n_steps=3, n_walkers=18, random_seed=42)

    np.testing.assert_array_equal(first.chain, second.chain)
    np.testing.assert_array_equal(first.log_probability, second.log_probability)


def test_model_reports_missing_salt_configuration(monkeypatch):
    monkeypatch.delenv("SALTFIT_SALT_PATH", raising=False)

    with pytest.raises(SALTConfigurationError, match="SALTFIT_SALT_PATH"):
        SALTModel(
            lam_ref=2382.04,
            profile_type="emission",
            profile_parameters=_profile_parameters(),
            initial_values=_initial_values(),
            bounds=_bounds(),
            flow="inflow",
        )


def test_outflow_model_requires_and_uses_doppler_parameter(tmp_path):
    initial_values = _initial_values() | {"v_b": 10.0}
    bounds = _bounds() | {"v_b": (0.0, 50.0)}
    model = SALTModel(
        salt_path=str(_make_salt_installation(tmp_path / "salt")),
        lam_ref=2382.04,
        profile_type="emission",
        profile_parameters=_profile_parameters(),
        initial_values=initial_values,
        bounds=bounds,
        flow="outflow",
    )

    np.testing.assert_allclose(model.compute_spectrum([-1.0, 0.0, 1.0], model.initial_parameters), [11.0, 11.0, 11.0])


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
    profile_parameters = _profile_parameters()
    profile_parameters["observing_parameters"]["v_ap"] = parameters["v_w"]
    model = SALTModel(
        salt_path=str(model.salt_path),
        lam_ref=model.lam_ref,
        profile_type=model.profile_type,
        profile_parameters=profile_parameters,
        initial_values=parameters,
        bounds=model.bounds,
        flow=model.flow,
    )

    assert model.log_probability(spectrum, parameters) == -math.inf


@pytest.mark.parametrize(
    ("name", "value"),
    [("alpha", -0.1), ("v_0", 600.0), ("f_c", 1.1), ("delta", 0.0)],
)
def test_model_rejects_invalid_physical_parameters_before_evaluation(model, spectrum, name, value):
    parameters = model.initial_parameters
    parameters[name] = value

    assert model.log_probability(spectrum, parameters) == -math.inf


def test_model_conditions_non_finite_salt_output(tmp_path):
    profile_parameters = _profile_parameters()
    profile_parameters["miscellaneous_parameters"]["return_nan"] = True
    model = SALTModel(
        salt_path=str(_make_salt_installation(tmp_path / "salt")),
        lam_ref=2382.04,
        profile_type="emission",
        profile_parameters=profile_parameters,
        initial_values=_initial_values(),
        bounds=_bounds(),
        flow="inflow",
    )

    with pytest.warns(RuntimeWarning, match="non-finite flux"):
        flux = model.compute_spectrum([-1.0, 0.0, 1.0], model.initial_parameters)

    np.testing.assert_array_equal(flux, [1.0, 1.0, 1.0])
