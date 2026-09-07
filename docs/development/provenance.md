# SALTFit Provenance

This document records the origin of external scientific software and data used
or evaluated by SALTFit.

## SALT model

### Current status

SALT is officially released as C SALT under the BSD 3-Clause License. SALTFit
loads a user-supplied, separately built SALT checkout through its public Python
interface. SALTFit does not vendor or distribute SALT source code or binaries.

### Upstream project

- Project: SALT
- Repository: https://github.com/CodyCarr/SALT
- Author or maintainer: Cody Carr
- Version or commit evaluated: main at commit `71d7341e37de3b261e82bca1bbcd9fb61af00538`
- Date evaluated: 2026-08-31

### License and permission

- License: BSD 3-Clause License
- License file: https://github.com/CodyCarr/SALT/blob/main/LICENSE.md
- SALTFit integration: dynamically loads the public `python/run_salt.py`
  interface from a local SALT checkout selected by `SALTFIT_SALT_PATH` or
  `SALTModel(salt_path=...)`.

### Planned integration

SALTFit provides a high-level, object-oriented fitting API over SALT's public
Python dispatcher. The integration boundary remains narrow: SALTFit supplies
validated fit parameters and a normalized velocity-space spectrum, while the
official wrapper evaluates the selected inflow or outflow model.

Before each SALTFit release, verify the upstream SALT revision used for
integration and retain the required SALT citation guidance in user documentation.

## Development policy

- Do not vendor or publish SALT source code or compiled binaries in SALTFit.
- Do not represent SALTFit as affiliated with or endorsed by the SALT authors.
- Keep the SALT integration limited to its documented public Python interface.
- Record the tested SALT revision whenever the integration contract changes.
