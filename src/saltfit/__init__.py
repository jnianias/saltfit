"""Public API for SALTFit."""

from saltfit.backend import SALTConfigurationError
from saltfit.model import SALTModel
from saltfit.result import FitResult
from saltfit.spectrum import Spectrum

__all__ = [
    "FitResult",
    "SALTConfigurationError",
    "SALTModel",
    "Spectrum",
]
