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

    from saltfit.model import SALTModel
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

        object.__setattr__(self, "chain", chain)
        object.__setattr__(self, "log_probability", log_probability)

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

    def ml_parameters(self) -> dict[str, float]:
        """Return the maximum likelihood estimate for every parameter."""
        max_idx = np.argmax(self.log_probability)
        ml_values = self.samples[max_idx]
        return dict(zip(self.parameter_names, ml_values, strict=True))

    def build_model(self, *, salt_path: str | Path | None = None) -> SALTModel:
        """Recreate the fitted SALT model using a local SALT installation."""
        if self.model_configuration is None:
            raise ValueError("This fit result does not contain a model configuration.")

        from saltfit.model import SALTModel

        configuration = dict(self.model_configuration)
        if salt_path is not None:
            configuration["salt_path"] = str(salt_path)
        model = SALTModel(**configuration)
        if model.parameter_names != self.parameter_names:
            raise ValueError("The archived model parameters do not match the saved fit.")
        return model

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
