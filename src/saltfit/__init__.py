"""Public API for SALTFit."""

from saltfit.backend import SALTConfigurationError
from saltfit.model import SALTInflowModel
from saltfit.parameters import Parameter, UniformPrior
from saltfit.result import FitResult
from saltfit.spectrum import Spectrum

__all__ = [
    "FitResult",
    "Parameter",
    "SALTConfigurationError",
    "SALTInflowModel",
    "Spectrum",
    "UniformPrior",
]
