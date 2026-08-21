# SALTFit

SALTFit is an independent Python package for fitting Semi-Analytical Line Transfer (SALT -- see [Carr et al. 2023](https://iopscience.iop.org/article/10.3847/1538-4357/acd331)) models to astronomical spectra with an intuitive, object-oriented API and Markov chain
Monte Carlo (MCMC) methods.

## Project status

SALTFit is at an early stage of development and should be considered unstable.
The public API and SALT integration are not available yet. The current
repository contains the project scaffold, development tooling, documentation,
and initial tests.

## Development

SALTFit is developed with [uv](https://docs.astral.sh/uv/). Python 3.13 is the
current development version, and Python 3.11 to 3.14 are currently
targeted for support.

Create the development environment and install the development dependencies:

```bash
uv sync --all-groups
```

Run the checks used by continuous integration:

```bash
uv run ruff format --check .
uv run ruff check .
uv run pytest
uv build
```

To apply Ruff's formatting locally:

```bash
uv run ruff format .
```

## Project documentation

Development notes and project decisions are kept in [`docs/development/`](docs/development/),
including:

- [Project brief](docs/development/project-brief.md)
- [Provenance](docs/development/provenance.md)

## License

SALTFit is licensed under the MIT License. See [LICENSE](LICENSE) for the
full license text.
