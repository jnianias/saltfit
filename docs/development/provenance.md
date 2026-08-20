# SALTFit Provenance

This document records the origin of external scientific software and data used
or evaluated by SALTFit. It should be updated when the upstream SALT release and
licensing information become available.

## SALT model

### Current status

The current SALT repository is being used as a reference for the scientific
model and fitting workflow. Its source code, data, and documentation have not
been copied into SALTFit and are not distributed by this repository.

### Upstream project

- Project: SALT
- Repository: https://github.com/CodyCarr/SALT
- Author or maintainer: Cody Carr
- Version or commit evaluated: not yet recorded
- Date evaluated: not yet recorded

### License and permission

The current upstream repository does not appear to contain an explicit license.
No permission to copy, modify, or redistribute its source code has been recorded
for SALTFit.

The planned approach is to develop and publish SALTFit as an independent package
while waiting for the author's new version. SALTFit will not implement an
upstream integration or distribute upstream material until the release, license,
and scope of any permission have been recorded.

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
integration boundary will be kept narrow through a model-evaluation backend
contract. A future adapter may use an officially released SALT Python interface,
or a permitted vendored implementation, without changing the public SALTFit API.
The decision on the integration mechanism is deferred until the new release can
be evaluated.

Before public distribution, verify and document:

- the license of the C implementation,
- the license of the Python wrapper,
- the licenses of bundled dependencies and data,
- the upstream version used,
- any required attribution or notices,
- whether the upstream package can be declared as a normal dependency.

## Development policy

Until the licensing and release status are clear:

- SALTFit may be developed and published as a public, independent repository,
- do not copy the current SALT source, data, or documentation into this
  repository,
- do not publish wheels or source distributions containing SALT material,
- do not represent SALTFit as affiliated with or endorsed by the SALT authors,
- use the upstream repository and documentation only as references,
- record relevant correspondence and release information separately from source
  code when appropriate.
