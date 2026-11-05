"""Public API for SALTFit."""

from saltfit.backend import SALTConfigurationError
from saltfit.io import load_fit, load_salt_model
from saltfit.model import SALTModel, SystemicAbsorptionModel
from saltfit.result import (
    ConvergenceDiagnostics,
    FitResult,
    InformationCriteria,
    calculate_information_criteria,
)
from saltfit.spectrum import Spectrum

__all__ = [
    "FitResult",
    "InformationCriteria",
    "ConvergenceDiagnostics",
    "SALTConfigurationError",
    "SALTModel",
    "SystemicAbsorptionModel",
    "Spectrum",
    "calculate_information_criteria",
    "load_fit",
    "load_salt_model",
]
