"""High-level models and posterior-probability calculations."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from dataclasses import field as dataclass_field
from math import inf
from types import MappingProxyType
from typing import Literal, Mapping
from warnings import warn

import emcee
import numpy as np
from numpy.typing import ArrayLike, NDArray

from saltfit.backend import _SALTModelEvaluator
from saltfit.parameters import Parameter, UniformPrior
from saltfit.result import FitResult
from saltfit.spectrum import Spectrum

_FLOW_PARAMETER_NAMES = {
    "inflow": (
        "alpha",
        "psi",
        "gamma",
        "tau",
        "v_0",
        "v_w",
        "f_c",
        "k",
        "delta",
    ),
    "outflow": (
        "alpha",
        "psi",
        "gamma",
        "tau",
        "v_0",
        "v_w",
        "v_b",
        "f_c",
        "k",
        "delta",
    ),
}

_SYSTEMIC_ABSORPTION_PARAMETER_NAMES = (
    "systemic_absorption_sigma",
    "systemic_absorption_depth",
)


@dataclass(frozen=True, slots=True)
class SALTModel:
    """Fit a SALT line profile to a continuum-normalized observed spectrum.

    Parameters
    ----------
    lam_ref
        Rest wavelength of the reference transition. Use the same wavelength
        unit as the atomic wavelengths in ``profile_parameters``.
    profile_type
        Component of the line profile to evaluate: ``"absorption"``,
        ``"emission"``, or ``"pcygni"``.
    profile_parameters
        Fixed atomic-transition configuration for the SALT implementation.
        SALT configuration in its documented nested layout: an
        ``absorption_parameters`` mapping, an ``emission_parameters`` mapping,
        an ``observing_parameters`` mapping containing ``v_ap`` and optionally
        ``APERTURE``, and a ``miscellaneous_parameters`` mapping. Outflow
        models additionally require ``blending_parameters`` and atomic
        Einstein A coefficients. See the official SALT documentation for the
        complete flow-specific configuration.
    initial_values
        Initial values for the parameters sampled by MCMC, supplied as a
        mapping from parameter name to value. The current implementation
        requires ``alpha``, ``psi``, ``gamma``, ``tau``, ``v_0``, ``v_w``,
        ``f_c``, ``k``, and ``delta``. Outflow models additionally require
        ``v_b``, the Doppler parameter for thermal and microturbulent
        broadening. ``alpha`` and ``psi`` are angles in radians; ``v_0``,
        ``v_w``, and ``v_b`` are velocities in km/s; the other parameters are
        dimensionless.
    bounds
        Uniform prior bounds for each sampled parameter, as a mapping from
        parameter name to a ``(lower, upper)`` pair. Each initial value must
        lie within its corresponding bounds.
    flow
        Direction of the modelled gas flow: ``"inflow"`` or ``"outflow"``.
    salt_path
        Optional local SALT project root. When omitted, SALTFit uses the
        ``SALTFIT_SALT_PATH`` environment variable.
    background
        Constant continuum-normalized background flux supplied to the SALT
        model. The default of one is appropriate for ``Spectrum`` objects.
    systemic_absorption
        Include a Gaussian absorption component centered at systemic velocity
        (zero km/s). When enabled, ``initial_values`` and ``bounds`` must also
        define ``systemic_absorption_sigma`` in km/s and the dimensionless
        ``systemic_absorption_depth``.

    Notes
    -----
    With ``APERTURE=True``, SALTFit assigns parameter combinations with an
    aperture velocity outside the model flow to a log prior of negative
    infinity.
    """

    lam_ref: float
    profile_type: str
    profile_parameters: Mapping[str, object]
    initial_values: Mapping[str, float]
    bounds: Mapping[str, tuple[float, float]]
    flow: Literal["inflow", "outflow"] = "outflow"
    salt_path: str | None = None
    background: float = 1.0
    systemic_absorption: bool = False
    _parameters: tuple[Parameter, ...] = dataclass_field(init=False, repr=False, compare=False)
    _evaluator: _SALTModelEvaluator = dataclass_field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        """Validate the model parameter definition."""
        if self.flow not in _FLOW_PARAMETER_NAMES:
            raise ValueError("flow must be 'inflow' or 'outflow'.")
        object.__setattr__(self, "profile_parameters", _freeze_mapping(self.profile_parameters))
        self._validate_parameter_configuration()
        parameters = tuple(
            Parameter(name, float(self.initial_values[name]), UniformPrior(*self.bounds[name]))
            for name in self.parameter_names
        )
        object.__setattr__(
            self, "initial_values", {parameter.name: parameter.initial_value for parameter in parameters}
        )
        object.__setattr__(
            self, "bounds", {name: tuple(map(float, self.bounds[name])) for name in self.parameter_names}
        )
        object.__setattr__(self, "_parameters", parameters)
        object.__setattr__(
            self,
            "_evaluator",
            _SALTModelEvaluator(
                lam_ref=self.lam_ref,
                profile_type=self.profile_type,
                profile_parameters=self.profile_parameters,
                flow=self.flow,
                salt_path=self.salt_path,
                background=self.background,
            ),
        )

    @property
    def parameter_names(self) -> tuple[str, ...]:
        """Names of the free model parameters in sampler order."""
        parameter_names = _FLOW_PARAMETER_NAMES[self.flow]
        if self.systemic_absorption:
            return parameter_names + _SYSTEMIC_ABSORPTION_PARAMETER_NAMES
        return parameter_names

    @property
    def initial_parameters(self) -> dict[str, float]:
        """Initial values keyed by parameter name."""
        return dict(self.initial_values)

    def compute_spectrum(self, velocity: ArrayLike, parameters: Mapping[str, float]) -> NDArray[np.float64]:
        """Evaluate model flux at each supplied velocity sample."""
        velocity_array = np.asarray(velocity, dtype=float)
        if velocity_array.ndim != 1:
            raise ValueError("velocity must be one-dimensional.")
        if not np.isfinite(velocity_array).all():
            raise ValueError("velocity must contain only finite values.")
        self._validate_parameter_values(parameters)

        salt_parameters = {name: parameters[name] for name in _FLOW_PARAMETER_NAMES[self.flow]}
        flux = np.asarray(self._evaluator.evaluate(velocity_array, salt_parameters), dtype=float)
        if flux.shape != velocity_array.shape:
            warn(
                "SALT returned a spectrum with an unexpected shape; using the normalized continuum instead.",
                RuntimeWarning,
                stacklevel=2,
            )
            return np.full_like(velocity_array, self.background)
        if not np.isfinite(flux).all():
            warn(
                "SALT returned non-finite flux values; replacing them with the normalized continuum.",
                RuntimeWarning,
                stacklevel=2,
            )
            return np.where(np.isfinite(flux), flux, self.background)
        if self.systemic_absorption:
            flux -= parameters["systemic_absorption_depth"] * np.exp(
                -0.5 * (velocity_array / parameters["systemic_absorption_sigma"]) ** 2
            )
        return flux

    def log_prior(self, parameters: Mapping[str, float]) -> float:
        """Return the total log prior density, or negative infinity if invalid."""
        try:
            self._validate_parameter_values(parameters)
        except ValueError:
            return -inf

        if not self._has_valid_physical_parameters(parameters):
            return -inf

        log_prior = sum(parameter.prior.logpdf(float(parameters[parameter.name])) for parameter in self._parameters)
        return float(log_prior) if np.isfinite(log_prior) else -inf

    def log_likelihood(self, spectrum: Spectrum, parameters: Mapping[str, float]) -> float:
        """Return the normalized independent-Gaussian log likelihood."""
        predicted_flux = self.compute_spectrum(spectrum.velocity, parameters)
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

        n_dimensions = len(self._parameters)
        n_walkers = n_walkers if n_walkers is not None else 2 * n_dimensions + 2
        if n_walkers < 2 * n_dimensions:
            raise ValueError("n_walkers must be at least twice the parameter count.")

        random_generator = np.random.default_rng(random_seed)
        initial_state = self._initial_walker_positions(n_walkers, initial_spread, random_generator)

        def log_probability(parameter_values: NDArray[np.float64]) -> float:
            parameters = dict(zip(self.parameter_names, parameter_values, strict=True))
            return self.log_probability(spectrum, parameters)

        sampler = emcee.EnsembleSampler(n_walkers, n_dimensions, log_probability)
        if random_seed is not None:
            sampler.random_state = np.random.RandomState(random_seed).get_state()
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
        initial_values = np.array([parameter.initial_value for parameter in self._parameters])
        scales = initial_spread * np.maximum(np.abs(initial_values), 1.0)
        positions = np.empty((n_walkers, len(self._parameters)))

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

    def _validate_parameter_configuration(self) -> None:
        expected_names = set(self.parameter_names)
        if set(self.initial_values) != expected_names:
            raise ValueError(f"initial_values must contain exactly the required {self.flow} SALT parameter names.")
        if set(self.bounds) != expected_names:
            raise ValueError(f"bounds must contain exactly the required {self.flow} SALT parameter names.")

        try:
            observing = self.profile_parameters["observing_parameters"]
            v_ap = float(observing["v_ap"])
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError("profile_parameters['observing_parameters'] must contain a finite v_ap.") from error
        if not np.isfinite(v_ap) or v_ap < 0:
            raise ValueError("profile_parameters['observing_parameters']['v_ap'] must be finite and nonnegative.")

        for name in self.parameter_names:
            try:
                lower, upper = self.bounds[name]
                initial_value = float(self.initial_values[name])
                lower = float(lower)
                upper = float(upper)
            except (TypeError, ValueError) as error:
                raise ValueError(f"bounds[{name!r}] must be a pair of finite numbers.") from error
            if not np.isfinite((initial_value, lower, upper)).all():
                raise ValueError(f"initial_values[{name!r}] and bounds[{name!r}] must be finite.")
            if lower >= upper:
                raise ValueError(f"bounds[{name!r}] must have lower < upper.")
            if not lower <= initial_value <= upper:
                raise ValueError(f"initial_values[{name!r}] must lie within bounds[{name!r}].")

    def _valid_aperture_velocity(self, v_ap: object, v_w: float) -> bool:
        aperture_velocity = float(v_ap)
        if self.flow == "inflow":
            return aperture_velocity < v_w
        return 0 < aperture_velocity <= v_w

    def _has_valid_physical_parameters(self, parameters: Mapping[str, float]) -> bool:
        """Return whether parameters satisfy SALT's shared physical domain."""
        if not 0.0 <= parameters["alpha"] <= np.pi / 2.0:
            return False
        if not 0.0 <= parameters["psi"] <= np.pi / 2.0:
            return False
        if parameters["gamma"] <= 0.0 or parameters["tau"] < 0.0:
            return False
        if parameters["v_0"] <= 0.0 or parameters["v_w"] <= parameters["v_0"]:
            return False
        if not 0.0 <= parameters["f_c"] <= 1.0 or parameters["k"] < 0.0 or parameters["delta"] <= 0.0:
            return False
        if self.flow == "outflow" and parameters["v_b"] < 0.0:
            return False
        if self.systemic_absorption:
            if parameters["systemic_absorption_sigma"] <= 0.0:
                return False
            if not 0.0 <= parameters["systemic_absorption_depth"] <= 1.0:
                return False

        try:
            observing = self.profile_parameters["observing_parameters"]
            aperture_enabled = observing.get("APERTURE", True)
            return not aperture_enabled or self._valid_aperture_velocity(observing["v_ap"], parameters["v_w"])
        except (KeyError, TypeError, ValueError):
            return False


def _freeze_mapping(mapping: Mapping[str, object]) -> Mapping[str, object]:
    """Return an immutable deep copy of nested SALT configuration mappings."""
    return MappingProxyType({key: _freeze_value(value) for key, value in deepcopy(dict(mapping)).items()})


def _freeze_value(value: object) -> object:
    """Recursively freeze nested configuration values."""
    if isinstance(value, Mapping):
        return _freeze_mapping(value)
    if isinstance(value, list):
        return tuple(_freeze_value(item) for item in value)
    return value
