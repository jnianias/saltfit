"""Results returned by MCMC fitting."""

from __future__ import annotations

import json
import platform
from dataclasses import dataclass, field
from datetime import datetime, timezone
from importlib.metadata import version
from typing import TYPE_CHECKING, Mapping

import numpy as np
from numpy.typing import NDArray

if TYPE_CHECKING:
    from pathlib import Path

    from saltfit.model import SALTModel, SystemicAbsorptionModel
    from saltfit.spectrum import Spectrum

_ARCHIVE_FORMAT = "saltfit-fit-archive"
_ARCHIVE_SCHEMA_VERSION = 1


@dataclass(frozen=True, slots=True)
class FitResult:
    """Posterior samples and log probabilities produced by an MCMC fit."""

    parameter_names: tuple[str, ...]
    chain: NDArray[np.float64]
    log_probability: NDArray[np.float64]
    spectrum: Spectrum | None = field(default=None, repr=False, compare=False)
    model_configuration: Mapping[str, object] | None = field(default=None, repr=False, compare=False)
    fit_settings: Mapping[str, object] | None = field(default=None, repr=False, compare=False)
    metadata: Mapping[str, object] | None = field(default=None, repr=False, compare=False)
    acceptance_fraction: NDArray[np.float64] | None = field(default=None, repr=False, compare=False)

    def __post_init__(self) -> None:
        """Validate sampler output shapes."""
        chain = np.asarray(self.chain, dtype=float)
        log_probability = np.asarray(self.log_probability, dtype=float)
        if chain.ndim != 3:
            raise ValueError("chain must have shape (steps, walkers, parameters).")
        if chain.shape[2] != len(self.parameter_names):
            raise ValueError("chain parameter dimension must match parameter_names.")
        if log_probability.shape != chain.shape[:2]:
            raise ValueError("log_probability must have shape (steps, walkers).")
        acceptance_fraction = self.acceptance_fraction
        if acceptance_fraction is not None:
            acceptance_fraction = np.asarray(acceptance_fraction, dtype=float)
            if acceptance_fraction.shape != (chain.shape[1],):
                raise ValueError("acceptance_fraction must have shape (walkers,).")

        object.__setattr__(self, "chain", chain)
        object.__setattr__(self, "log_probability", log_probability)
        object.__setattr__(self, "acceptance_fraction", acceptance_fraction)

    @property
    def n_steps(self) -> int:
        """Number of MCMC steps retained in the chain."""
        return self.chain.shape[0]

    @property
    def n_walkers(self) -> int:
        """Number of MCMC walkers in the chain."""
        return self.chain.shape[1]

    @property
    def samples(self) -> NDArray[np.float64]:
        """Flattened posterior samples with shape ``(steps * walkers, parameters)``."""
        return self.chain.reshape(-1, self.chain.shape[-1])

    def median_parameters(self) -> dict[str, float]:
        """Return the posterior median for every parameter."""
        medians = np.median(self.samples, axis=0)
        return dict(zip(self.parameter_names, medians, strict=True))

    def ml_parameters(
        self,
        model: SALTModel | SystemicAbsorptionModel | None = None,
        *,
        burn_in: int = 0,
    ) -> dict[str, float]:
        """Return the chain sample with the highest likelihood, excluding optional burn-in."""
        parameters, _ = self._maximum_likelihood_sample(model, burn_in=burn_in)
        return parameters

    def build_model(self, *, salt_path: str | Path | None = None) -> SALTModel | SystemicAbsorptionModel:
        """Recreate the fitted model, using a local SALT installation when required."""
        if self.model_configuration is None:
            raise ValueError("This fit result does not contain a model configuration.")

        from saltfit.model import SALTModel, SystemicAbsorptionModel

        configuration = dict(self.model_configuration)
        model_type = configuration.pop("model_type", "salt")
        if model_type == "systemic_absorption":
            model = SystemicAbsorptionModel(**configuration)
        elif model_type == "salt":
            if salt_path is not None:
                configuration["salt_path"] = str(salt_path)
            model = SALTModel(**configuration)
        else:
            raise ValueError(f"Unsupported archived model type: {model_type!r}.")
        if model.parameter_names != self.parameter_names:
            raise ValueError("The archived model parameters do not match the saved fit.")
        return model

    def information_criteria(
        self,
        model: SALTModel | SystemicAbsorptionModel | None = None,
        *,
        burn_in: int = 0,
    ) -> InformationCriteria:
        """Calculate AIC and BIC from the highest likelihood among retained chain samples.

        This uses the best likelihood sampled by the chain as an estimate of the
        maximum likelihood; it does not perform a separate maximum-likelihood
        optimization.
        """
        _, maximum_log_likelihood = self._maximum_likelihood_sample(model, burn_in=burn_in)
        return calculate_information_criteria(
            log_likelihood=maximum_log_likelihood,
            n_parameters=len(self.parameter_names),
            n_observations=self.spectrum.size,
        )

    def _maximum_likelihood_sample(
        self,
        model: SALTModel | SystemicAbsorptionModel | None,
        *,
        burn_in: int,
    ) -> tuple[dict[str, float], float]:
        if self.spectrum is None:
            raise ValueError("The input spectrum is required to calculate a maximum likelihood estimate.")
        if not isinstance(burn_in, int) or isinstance(burn_in, bool) or not 0 <= burn_in < self.n_steps:
            raise ValueError("burn_in must be an integer in [0, n_steps).")

        model = model or self.build_model()
        if model.parameter_names != self.parameter_names:
            raise ValueError("model parameter names must match the fit result.")

        chain = self.chain[burn_in:].reshape(-1, len(self.parameter_names))
        log_probability = self.log_probability[burn_in:].reshape(-1)
        best_parameters = None
        maximum_log_likelihood = -np.inf
        for parameter_values, log_posterior in zip(chain, log_probability, strict=True):
            parameters = dict(zip(self.parameter_names, parameter_values, strict=True))
            log_prior = model.log_prior(parameters)
            if np.isfinite(log_posterior) and np.isfinite(log_prior):
                log_likelihood = float(log_posterior - log_prior)
                if log_likelihood > maximum_log_likelihood:
                    best_parameters = parameters
                    maximum_log_likelihood = log_likelihood
        if best_parameters is None:
            raise ValueError("No finite likelihood values were found in the selected chain samples.")
        return best_parameters, maximum_log_likelihood

    def convergence_diagnostics(
        self,
        *,
        minimum_steps_per_tau: float = 50.0,
        tau_relative_tolerance: float = 0.01,
    ) -> ConvergenceDiagnostics:
        """Estimate integrated autocorrelation times and check standard emcee heuristics."""
        if self.n_steps < 4:
            raise ValueError("At least four MCMC steps are needed for convergence diagnostics.")
        if minimum_steps_per_tau <= 0 or tau_relative_tolerance <= 0:
            raise ValueError("Diagnostic thresholds must be positive.")

        import emcee

        with np.errstate(divide="ignore", invalid="ignore"):
            full_tau = np.asarray(emcee.autocorr.integrated_time(self.chain, tol=0), dtype=float)
            half_tau = np.asarray(emcee.autocorr.integrated_time(self.chain[: self.n_steps // 2], tol=0), dtype=float)
        valid = np.isfinite(full_tau) & (full_tau > 0) & np.isfinite(half_tau) & (half_tau > 0)
        relative_change = np.full(full_tau.shape, np.inf)
        relative_change[valid] = np.abs(full_tau[valid] - half_tau[valid]) / full_tau[valid]
        steps_per_tau = self.n_steps / full_tau
        length_sufficient = valid & (steps_per_tau >= minimum_steps_per_tau)
        half_chain_length_sufficient = (self.n_steps // 2) >= minimum_steps_per_tau * half_tau
        estimate_stable = valid & half_chain_length_sufficient & (relative_change <= tau_relative_tolerance)

        return ConvergenceDiagnostics(
            parameter_names=self.parameter_names,
            autocorrelation_time=full_tau,
            steps_per_autocorrelation_time=steps_per_tau,
            tau_relative_change=relative_change,
            length_sufficient=length_sufficient,
            estimate_stable=estimate_stable,
            converged=bool(np.all(length_sufficient & estimate_stable)),
            acceptance_fraction=self.acceptance_fraction,
        )

    def write_hdf5(self, filepath: Path | str) -> None:
        """Write this fit, its model configuration, and its input spectrum to HDF5."""
        import h5py

        if self.spectrum is None or self.model_configuration is None or self.fit_settings is None:
            raise ValueError("Only fit results returned by SALTModel.fit_to() can be written as complete archives.")

        with h5py.File(filepath, "w") as archive:
            archive.attrs["format"] = _ARCHIVE_FORMAT
            archive.attrs["schema_version"] = _ARCHIVE_SCHEMA_VERSION

            metadata = archive.create_group("metadata")
            metadata_values = dict(self.metadata or {})
            metadata_values.setdefault("saved_utc", datetime.now(timezone.utc).isoformat())
            metadata_values.setdefault("saltfit_version", version("saltfit"))
            metadata_values.setdefault("python_version", platform.python_version())
            metadata_values.setdefault("numpy_version", version("numpy"))
            metadata_values.setdefault("emcee_version", version("emcee"))
            metadata_values.setdefault("h5py_version", version("h5py"))
            for name, value in metadata_values.items():
                metadata.attrs[name] = _json_value(value)

            model = archive.create_group("model")
            model.create_dataset(
                "configuration",
                data=json.dumps(_json_value(self.model_configuration), allow_nan=False),
                dtype=h5py.string_dtype("utf-8"),
            )

            data = archive.create_group("data")
            spectrum = data.create_group("spectrum")
            spectrum.create_dataset("velocity", data=self.spectrum.velocity)
            spectrum.create_dataset("flux", data=self.spectrum.flux)
            spectrum.create_dataset("uncertainty", data=self.spectrum.uncertainty)

            fit = archive.create_group("fit")
            fit.create_dataset(
                "parameter_names",
                data=self.parameter_names,
                dtype=h5py.string_dtype("utf-8"),
            )
            fit.create_dataset("chain", data=self.chain)
            fit.create_dataset("log_probability", data=self.log_probability)
            if self.acceptance_fraction is not None:
                fit.create_dataset("acceptance_fraction", data=self.acceptance_fraction)
            fit.create_dataset(
                "settings",
                data=json.dumps(_json_value(self.fit_settings), allow_nan=False),
                dtype=h5py.string_dtype("utf-8"),
            )


def _json_value(value: object) -> object:
    """Convert supported scientific configuration values into JSON-compatible values."""
    from collections.abc import Mapping as MappingABC

    if isinstance(value, MappingABC):
        if not all(isinstance(key, str) for key in value):
            raise TypeError("HDF5 archive configuration mappings must have string keys.")
        return {key: _json_value(item) for key, item in value.items()}
    if isinstance(value, np.ndarray):
        return _json_value(value.tolist())
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    if isinstance(value, np.generic):
        return _json_value(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        raise ValueError("HDF5 archive configuration cannot contain non-finite numbers.")
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    raise TypeError(f"Unsupported HDF5 archive configuration value: {type(value).__name__}.")


@dataclass(frozen=True, slots=True)
class InformationCriteria:
    """Likelihood-based information criteria for a fitted model."""

    log_likelihood: float
    n_parameters: int
    n_observations: int
    aic: float
    bic: float


def calculate_information_criteria(
    *,
    log_likelihood: float,
    n_parameters: int,
    n_observations: int,
) -> InformationCriteria:
    """Calculate Akaike and Bayesian information criteria from a maximum log likelihood."""
    if not np.isfinite(log_likelihood):
        raise ValueError("log_likelihood must be finite.")
    if not isinstance(n_parameters, int) or isinstance(n_parameters, bool) or n_parameters < 1:
        raise ValueError("n_parameters must be a positive integer.")
    if not isinstance(n_observations, int) or isinstance(n_observations, bool) or n_observations < 1:
        raise ValueError("n_observations must be a positive integer.")

    log_likelihood = float(log_likelihood)
    aic = 2 * n_parameters - 2 * log_likelihood
    bic = n_parameters * np.log(n_observations) - 2 * log_likelihood
    return InformationCriteria(
        log_likelihood=log_likelihood,
        n_parameters=n_parameters,
        n_observations=n_observations,
        aic=float(aic),
        bic=float(bic),
    )


@dataclass(frozen=True, slots=True)
class ConvergenceDiagnostics:
    """Integrated autocorrelation estimates and emcee convergence heuristics."""

    parameter_names: tuple[str, ...]
    autocorrelation_time: NDArray[np.float64]
    steps_per_autocorrelation_time: NDArray[np.float64]
    tau_relative_change: NDArray[np.float64]
    length_sufficient: NDArray[np.bool_]
    estimate_stable: NDArray[np.bool_]
    converged: bool
    acceptance_fraction: NDArray[np.float64] | None
