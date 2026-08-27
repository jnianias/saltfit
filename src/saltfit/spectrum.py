"""Observed spectral data containers."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray


@dataclass(frozen=True, slots=True)
class Spectrum:
    """One-dimensional observed spectrum sampled on a velocity axis.

    Parameters
    ----------
    velocity
        Observed velocity samples in km/s.
    flux
        Flux measured at each velocity sample.
    uncertainty
        One-sigma flux uncertainties at each velocity sample.
    """

    velocity: ArrayLike
    flux: ArrayLike
    uncertainty: ArrayLike

    def __post_init__(self) -> None:
        """Normalise input data and validate the spectrum contract."""
        velocity = np.asarray(self.velocity, dtype=float)
        flux = np.asarray(self.flux, dtype=float)
        uncertainty = np.asarray(self.uncertainty, dtype=float)

        arrays = {"velocity": velocity, "flux": flux, "uncertainty": uncertainty}
        for name, values in arrays.items():
            if values.ndim != 1:
                raise ValueError(f"{name} must be one-dimensional.")
            if values.size == 0:
                raise ValueError(f"{name} must not be empty.")
            if not np.isfinite(values).all():
                raise ValueError(f"{name} must contain only finite values.")

        if velocity.shape != flux.shape or velocity.shape != uncertainty.shape:
            raise ValueError("velocity, flux, and uncertainty must have the same shape.")
        if not np.all(np.diff(velocity) > 0):
            raise ValueError("velocity must be strictly increasing.")
        if not np.all(uncertainty > 0):
            raise ValueError("uncertainty values must be positive.")

        object.__setattr__(self, "velocity", velocity)
        object.__setattr__(self, "flux", flux)
        object.__setattr__(self, "uncertainty", uncertainty)

    @property
    def size(self) -> int:
        """Number of spectral samples."""
        return self.velocity.size

    @property
    def weights(self) -> NDArray[np.float64]:
        """Inverse-variance weights for likelihood calculations."""
        return 1.0 / self.uncertainty**2

    def masked(self, mask: ArrayLike) -> Spectrum:
        """Return the samples selected by a one-dimensional Boolean mask."""
        mask_array = np.asarray(mask, dtype=bool)
        if mask_array.ndim != 1 or mask_array.shape != self.velocity.shape:
            raise ValueError("mask must be one-dimensional and match the spectrum shape.")
        if not np.any(mask_array):
            raise ValueError("mask must select at least one sample.")

        return Spectrum(
            velocity=self.velocity[mask_array],
            flux=self.flux[mask_array],
            uncertainty=self.uncertainty[mask_array],
        )
