"""Input/output utilities for SaltFit."""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from saltfit.result import _ARCHIVE_FORMAT, _ARCHIVE_SCHEMA_VERSION, FitResult
from saltfit.spectrum import Spectrum

if TYPE_CHECKING:
    from saltfit.model import SALTModel


def load_fit(filepath: Path | str) -> FitResult:
    """Load a saved fit archive without requiring a local SALT installation."""
    import h5py

    with h5py.File(filepath, "r") as archive:
        if archive.attrs.get("format") != _ARCHIVE_FORMAT:
            raise ValueError(f"{filepath!s} is not a SALTFit fit archive.")
        schema_version = archive.attrs.get("schema_version")
        if schema_version != _ARCHIVE_SCHEMA_VERSION:
            raise ValueError(f"Unsupported SALTFit archive schema version: {schema_version!r}.")

        try:
            model_configuration = json.loads(_read_text(archive["model/configuration"][()]))
            fit_settings = json.loads(_read_text(archive["fit/settings"][()]))
            names = tuple(_read_text(value) for value in archive["fit/parameter_names"][...])
            spectrum = Spectrum(
                velocity=archive["data/spectrum/velocity"][...],
                flux=archive["data/spectrum/flux"][...],
                uncertainty=archive["data/spectrum/uncertainty"][...],
            )
            chain = archive["fit/chain"][...]
            log_probability = archive["fit/log_probability"][...]
            metadata = {name: _decode_attribute(value) for name, value in archive["metadata"].attrs.items()}
        except KeyError as error:
            raise ValueError(f"SALTFit archive is missing required content: {error}.") from error
        except (json.JSONDecodeError, TypeError) as error:
            raise ValueError(f"SALTFit archive contains invalid serialized configuration: {error}.") from error

    return FitResult(
        parameter_names=names,
        chain=np.asarray(chain, dtype=float),
        log_probability=np.asarray(log_probability, dtype=float),
        spectrum=spectrum,
        model_configuration=model_configuration,
        fit_settings=fit_settings,
        metadata=metadata,
    )


def load_salt_model(filepath: Path | str, *, salt_path: str | Path | None = None) -> SALTModel:
    """Recreate the SALT model configuration stored in a fit archive."""
    return load_fit(filepath).build_model(salt_path=salt_path)


def _read_text(value: object) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return str(value)


def _decode_attribute(value: object) -> object:
    if isinstance(value, bytes):
        return value.decode("utf-8")
    if isinstance(value, np.generic):
        return value.item()
    return value
