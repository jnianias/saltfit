"""High-level models and posterior-probability calculations."""

from __future__ import annotations

from dataclasses import dataclass
from dataclasses import field as dataclass_field
from math import inf
from typing import Mapping, Sequence

import emcee
import numpy as np
from numpy.typing import ArrayLike, NDArray

from saltfit.backend import _SALTInflowEvaluator
from saltfit.parameters import Parameter
from saltfit.result import FitResult
from saltfit.spectrum import Spectrum


@dataclass(frozen=True, slots=True)
class SALTInflowModel:
    """A SALT inflow line-profile model that can be fit to an observed spectrum.

    Parameters
    ----------
    lam_ref
        Rest wavelength of the reference transition.
    profile_type
        SALT profile type: ``"absorption"``, ``"emission"``, or ``"pcygni"``.
    profile_parameters
        Fixed atomic and profile configuration accepted by the SALT wrapper.
    parameters
        Flow parameters to sample and their priors.
    salt_path
        Optional local SALT project root. When omitted, SALTFit uses the
        ``SALTFIT_SALT_PATH`` environment variable.
    background
        Constant normalized background flux supplied to the SALT model.
    """

    lam_ref: float
    profile_type: str
    profile_parameters: Mapping[str, object]
    parameters: Sequence[Parameter]
    salt_path: str | None = None
    background: float = 1.0
    _evaluator: _SALTInflowEvaluator = dataclass_field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        """Validate the model parameter definition."""
        parameters = tuple(self.parameters)
        if not parameters:
            raise ValueError("parameters must contain at least one Parameter.")
        names = [parameter.name for parameter in parameters]
        if len(set(names)) != len(names):
            raise ValueError("parameter names must be unique.")
        object.__setattr__(self, "parameters", parameters)
        object.__setattr__(
            self,
            "_evaluator",
            _SALTInflowEvaluator(
                lam_ref=self.lam_ref,
                profile_type=self.profile_type,
                profile_parameters=self.profile_parameters,
                salt_path=self.salt_path,
                background=self.background,
            ),
        )

    @property
    def parameter_names(self) -> tuple[str, ...]:
        """Names of the free model parameters in sampler order."""
        return tuple(parameter.name for parameter in self.parameters)

    @property
    def initial_parameters(self) -> dict[str, float]:
        """Initial values keyed by parameter name."""
        return {parameter.name: parameter.initial_value for parameter in self.parameters}

    def predict(self, velocity: ArrayLike, parameters: Mapping[str, float]) -> NDArray[np.float64]:
        """Evaluate model flux at each supplied velocity sample."""
        velocity_array = np.asarray(velocity, dtype=float)
        if velocity_array.ndim != 1:
            raise ValueError("velocity must be one-dimensional.")
        if not np.isfinite(velocity_array).all():
            raise ValueError("velocity must contain only finite values.")
        self._validate_parameter_values(parameters)

        flux = np.asarray(self._evaluator.evaluate(velocity_array, parameters), dtype=float)
        if flux.shape != velocity_array.shape:
            raise ValueError("backend output must have the same shape as velocity.")
        if not np.isfinite(flux).all():
            raise ValueError("backend output must contain only finite values.")
        return flux

    def log_prior(self, parameters: Mapping[str, float]) -> float:
        """Return the total log prior density, or negative infinity if invalid."""
        try:
            self._validate_parameter_values(parameters)
        except ValueError:
            return -inf

        log_prior = sum(parameter.prior.logpdf(float(parameters[parameter.name])) for parameter in self.parameters)
        return float(log_prior) if np.isfinite(log_prior) else -inf

    def log_likelihood(self, spectrum: Spectrum, parameters: Mapping[str, float]) -> float:
        """Return the normalized independent-Gaussian log likelihood."""
        predicted_flux = self.predict(spectrum.velocity, parameters)
        residual = spectrum.flux - predicted_flux
        variance = spectrum.uncertainty**2
        return float(-0.5 * np.sum(residual**2 / variance + np.log(2.0 * np.pi * variance)))

    def log_probability(self, spectrum: Spectrum, parameters: Mapping[str, float]) -> float:
        """Return the log posterior probability for ``parameters`` and ``spectrum``."""
        log_prior = self.log_prior(parameters)
        if not np.isfinite(log_prior):
            return -inf

        try:
            return log_prior + self.log_likelihood(spectrum, parameters)
        except ValueError:
            return -inf

    def fit_to(
        self,
        spectrum: Spectrum,
        *,
        n_steps: int,
        n_walkers: int | None = None,
        initial_spread: float = 1e-4,
        random_seed: int | None = None,
        progress: bool = False,
    ) -> FitResult:
        """Sample the posterior distribution for ``spectrum`` using ``emcee``.

        Parameters
        ----------
        spectrum
            Observed spectrum to fit.
        n_steps
            Number of MCMC steps to run.
        n_walkers
            Number of ensemble walkers. Defaults to two times the parameter
            count plus two.
        initial_spread
            Relative Gaussian displacement used to initialise each walker.
        random_seed
            Seed for reproducible walker initialisation.
        progress
            Show emcee's progress bar while sampling.
        """
        if n_steps < 1:
            raise ValueError("n_steps must be positive.")
        if initial_spread <= 0:
            raise ValueError("initial_spread must be positive.")

        n_dimensions = len(self.parameters)
        n_walkers = n_walkers if n_walkers is not None else 2 * n_dimensions + 2
        if n_walkers < 2 * n_dimensions:
            raise ValueError("n_walkers must be at least twice the parameter count.")

        random_generator = np.random.default_rng(random_seed)
        initial_state = self._initial_walker_positions(n_walkers, initial_spread, random_generator)

        def log_probability(parameter_values: NDArray[np.float64]) -> float:
            parameters = dict(zip(self.parameter_names, parameter_values, strict=True))
            return self.log_probability(spectrum, parameters)

        sampler = emcee.EnsembleSampler(n_walkers, n_dimensions, log_probability)
        sampler.run_mcmc(initial_state, n_steps, progress=progress)
        return FitResult(
            parameter_names=self.parameter_names,
            chain=sampler.get_chain(),
            log_probability=sampler.get_log_prob(),
        )

    def _initial_walker_positions(
        self,
        n_walkers: int,
        initial_spread: float,
        random_generator: np.random.Generator,
    ) -> NDArray[np.float64]:
        initial_values = np.array([parameter.initial_value for parameter in self.parameters])
        scales = initial_spread * np.maximum(np.abs(initial_values), 1.0)
        positions = np.empty((n_walkers, len(self.parameters)))

        for walker_index in range(n_walkers):
            for _ in range(1_000):
                position = initial_values + random_generator.normal(scale=scales)
                parameters = dict(zip(self.parameter_names, position, strict=True))
                if np.isfinite(self.log_prior(parameters)):
                    positions[walker_index] = position
                    break
            else:
                raise ValueError("could not initialise walkers within the configured parameter priors.")

        return positions

    def _validate_parameter_values(self, parameters: Mapping[str, float]) -> None:
        expected_names = set(self.parameter_names)
        if set(parameters) != expected_names:
            raise ValueError("parameters must contain exactly the model parameter names.")
        if not all(np.isfinite(float(value)) for value in parameters.values()):
            raise ValueError("parameter values must be finite.")
