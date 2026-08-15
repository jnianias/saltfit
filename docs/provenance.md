# SALTFit Provenance

This document records the origin of external scientific software and data used
or evaluated by SALTFit. It should be updated when the upstream SALT release
and licensing information become available.

## SALT model

### Current status

The current SALT repository is being used as a reference for the scientific
model and fitting workflow. Its source code has not been copied into SALTFit
and is not distributed by this private development repository.

### Upstream project

- Project: SALT
- Repository: https://github.com/CodyCarr/SALT
- Author or maintainer: Cody Carr
- Version or commit evaluated: not yet recorded
- Date evaluated: not yet recorded

### License and permission

The current upstream repository does not appear to contain an explicit license.
No permission to copy, modify, or redistribute its source code has been
recorded for SALTFit.

The planned approach is to wait for the author's new version, then record the
license and the scope of any permission before incorporating or depending on
the implementation.

When the release is available, record:

- Released version or commit:
- Release date:
- License:
- License file or repository URL:
- Permission to modify and redistribute:
- Date permission was received:
- Person or organization granting permission:
- Files or components covered:

### Planned integration

SALTFit is intended to provide a high-level, object-oriented fitting API. The
integration boundary should be kept narrow so that SALTFit can use the
officially released SALT Python interface without modifying the underlying C
implementation or its wrapper.

Before public distribution, verify and document:

- the license of the C implementation,
- the license of the Python wrapper,
- the licenses of bundled dependencies and data,
- the upstream version used,
- any required attribution or notices,
- whether the upstream package can be declared as a normal dependency.

## Development policy

Until the licensing and release status are clear:

- keep the SALTFit repository private,
- do not copy the current SALT source into this repository,
- do not publish wheels or source distributions containing SALT code,
- use the upstream repository and documentation only as references,
- record relevant correspondence and release information separately from
  source code when appropriate.
