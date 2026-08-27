"""Results returned by MCMC fitting."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True, slots=True)
class FitResult:
    """Posterior samples and log probabilities produced by an MCMC fit."""

    parameter_names: tuple[str, ...]
    chain: NDArray[np.float64]
    log_probability: NDArray[np.float64]

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
