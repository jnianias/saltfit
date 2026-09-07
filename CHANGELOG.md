# Changelog

All notable changes to SALTFit are documented here.

## [Unreleased]

SALTFit is currently preparing its first alpha release. The public API remains
subject to change.

### Added

- Added `Spectrum`, a validated container for continuum-normalized spectra
  sampled on a strictly increasing velocity axis.
- Added FITS and ASCII table loading through `Spectrum.from_file()` and
  `Spectrum.from_table()` when Astropy support is installed.
- Added `SALTModel` for fitting SALT line profiles with MCMC through `emcee`.
- Added support for both SALT inflow and outflow model selection through the
  `flow` argument.
- Added local integration with the official SALT implementation through
  `SALTFIT_SALT_PATH` or an explicit `salt_path`.
- Added user-friendly `initial_values` and `bounds` mappings for configuring
  uniform parameter priors.
- Added reproducible MCMC sampling through the `random_seed` argument.
- Added `FitResult` for posterior chains, log probabilities, flattened samples,
  and posterior medians.
- Added opt-in SALT performance benchmarks for model evaluation and complete
  log-probability evaluation.
- Added pre-commit hooks, Ruff formatting and linting, and CI checks across
  supported Python versions.

### Changed

- SALT configuration now follows the official released wrapper interface and
  its nested profile-parameter layout.
- Invalid SALT model output is reported with a warning and conditioned to the
  normalized continuum so an upstream numerical result does not crash a fit.
- The standard test suite excludes SALT-dependent integration benchmarks unless
  they are explicitly selected.

### Dependencies

- Added `tqdm` as a runtime dependency for MCMC progress reporting.
- Added optional Astropy support for FITS and ASCII table ingestion.
- Added `pytest-benchmark` for development performance testing.

### Known requirements

- A separately installed and built official SALT checkout is required to run
  `SALTModel` against the SALT implementation.
- SALTFit does not vendor or redistribute SALT source code or compiled
  libraries.
- Input spectra must be continuum-normalized before constructing a `Spectrum`.
- Integration benchmarks require `SALTFIT_SALT_PATH` and are not part of the
  default CI test run.
