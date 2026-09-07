"""Interfaces for evaluating line-profile models."""

from __future__ import annotations

from dataclasses import dataclass, field
from importlib.util import module_from_spec, spec_from_file_location
from os import environ
from pathlib import Path
from platform import system
from types import ModuleType
from typing import Callable, Mapping

import numpy as np
from numpy.typing import NDArray


class SALTConfigurationError(RuntimeError):
    """Raised when a local SALT installation cannot be used."""


_LIBRARY_NAMES = {"Darwin": "libsalt.dylib", "Linux": "libsalt.so"}


@dataclass(frozen=True, slots=True)
class _SALTModelEvaluator:
    """Evaluate the current local SALT implementation through its Python wrapper.

    This is an implementation detail of :class:`saltfit.SALTModel`.
    """

    lam_ref: float
    profile_type: str
    profile_parameters: Mapping[str, object]
    flow: str
    salt_path: Path | str | None = None
    background: float = 1.0
    _salt: Callable[..., NDArray[np.float64]] = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        """Resolve, validate, and load the locally installed SALT wrapper."""
        if self.profile_type not in {"absorption", "emission", "pcygni"}:
            raise ValueError("profile_type must be 'absorption', 'emission', or 'pcygni'.")
        if not np.isfinite(self.lam_ref) or self.lam_ref <= 0:
            raise ValueError("lam_ref must be a positive finite wavelength.")
        if not np.isfinite(self.background):
            raise ValueError("background must be finite.")

        salt_path = self._resolve_salt_path()
        wrapper_path = salt_path / "python" / "run_salt.py"
        library_name = _library_name_for_current_platform()
        library_path = salt_path / library_name
        if not wrapper_path.is_file():
            raise SALTConfigurationError(f"SALT wrapper not found: {wrapper_path}")
        if not library_path.is_file():
            raise SALTConfigurationError(
                f"SALT shared library not found: {library_path}. Build it by running 'make -C {salt_path} -j'."
            )

        wrapper = _load_wrapper(wrapper_path)
        try:
            salt = wrapper.salt
        except AttributeError as error:
            raise SALTConfigurationError(f"SALT wrapper does not define salt: {wrapper_path}") from error

        object.__setattr__(self, "salt_path", salt_path)
        object.__setattr__(self, "_salt", salt)

    def evaluate(self, velocity: NDArray, parameters: Mapping[str, float]) -> NDArray[np.float64]:
        """Return a SALT flux profile at the supplied velocity samples."""
        flow_parameters = {name: float(value) for name, value in parameters.items()}
        profile_parameters = self._profile_parameters_for_evaluation()
        return np.asarray(
            self._salt(
                v_obs=velocity,
                lam_ref=self.lam_ref,
                background=np.full_like(velocity, self.background, dtype=float),
                flow_parameters=flow_parameters,
                profile_parameters=profile_parameters,
                profile_type=self.profile_type,
                model_type=self.flow,
            ),
            dtype=float,
        )

    def _profile_parameters_for_evaluation(self) -> dict[str, object]:
        """Return configuration without legacy duplicates of dynamic inputs."""
        profile_parameters = dict(self.profile_parameters)
        for section_name, legacy_key in (
            ("observing_parameters", "v_obs"),
            ("miscellaneous_parameters", "lam_ref"),
        ):
            section = profile_parameters.get(section_name)
            if isinstance(section, Mapping):
                section_copy = dict(section)
                section_copy.pop(legacy_key, None)
                profile_parameters[section_name] = section_copy
        return profile_parameters

    def _resolve_salt_path(self) -> Path:
        configured_path = self.salt_path or environ.get("SALTFIT_SALT_PATH")
        if configured_path is None:
            raise SALTConfigurationError(
                "SALT location is not configured. Pass salt_path or set SALTFIT_SALT_PATH to the SALT project root."
            )

        salt_path = Path(configured_path).expanduser()
        if not salt_path.is_dir():
            raise SALTConfigurationError(f"SALT project root is not a directory: {salt_path}")
        return salt_path.resolve()


def _library_name_for_current_platform() -> str:
    try:
        return _LIBRARY_NAMES[system()]
    except KeyError as error:
        raise SALTConfigurationError(
            f"The local SALT implementation supports only macOS and Linux, not {system()!r}."
        ) from error


def _load_wrapper(wrapper_path: Path) -> ModuleType:
    module_name = f"saltfit_salt_wrapper_{abs(hash(wrapper_path))}"
    spec = spec_from_file_location(module_name, wrapper_path)
    if spec is None or spec.loader is None:
        raise SALTConfigurationError(f"Could not load the SALT wrapper: {wrapper_path}")

    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    return module
