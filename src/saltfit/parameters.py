"""Model parameters and probability priors."""

from __future__ import annotations

from dataclasses import dataclass
from math import inf
from typing import Protocol


class Prior(Protocol):
    """Probability distribution used to constrain one model parameter."""

    def logpdf(self, value: float) -> float:
        """Return the natural logarithm of the prior density at ``value``."""


@dataclass(frozen=True, slots=True)
class UniformPrior:
    """Uniform prior over a finite, closed interval."""

    lower: float
    upper: float

    def __post_init__(self) -> None:
        """Validate the interval bounds."""
        if not self.lower < self.upper:
            raise ValueError("lower must be less than upper.")

    def logpdf(self, value: float) -> float:
        """Return zero within the interval and negative infinity outside it."""
        return 0.0 if self.lower <= value <= self.upper else -inf


@dataclass(frozen=True, slots=True)
class Parameter:
    """A named scalar model parameter with an initial value and prior."""

    name: str
    initial_value: float
    prior: Prior

    def __post_init__(self) -> None:
        """Validate the parameter name and initial value."""
        if not self.name or not self.name.isidentifier():
            raise ValueError("name must be a valid Python identifier.")
        if self.prior.logpdf(self.initial_value) == -inf:
            raise ValueError("initial_value must be within the parameter prior.")
