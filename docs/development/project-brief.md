# SALTFit Project Brief

## What does SALTFit do?

SALTFit provides a flexible, intuitive API for fitting SALT models to data using
 MCMC and visualising results.

## Project relationship to SALT

SALTFit is an independent package. Until an upstream SALT release is available
under terms that permit integration, this repository will contain only original
SALTFit code and documentation. It will not include, redistribute, or modify the
SALT source code, data, or documentation.

SALTFit will define its own high-level fitting workflow independently of any
particular SALT implementation. The model-evaluation layer will sit behind a
narrow backend interface so that a future licensed SALT Python package or
vendored implementation can be integrated without changing the public SALTFit
API.

## What input does SALTFit accept?

At its core, SALTFit needs three arrays: (1) velocity, (2) flux, (3) errors.
Users should be able to provide data directly as a tuple of arrays, or as a
dictionary containing arrays, or as a table (astropy/pandas) or as a filepath
pointing to either a FITS or ASCII table.

Users should be able to provide either rest-frame velocity axes or rest
wavelengths along with systemic redshift (in which case SALTFit will generate
the velocity axis). in all cases, the rest wavelength and name of the line needs
to be specified -- probably better to force the user to provide these rather
than use an in-built dictionary of wavelengths.

## What can users fit?

Any SALT model they like to either singlets or doublet lines. So emission,
absorption, and P Cygni.

## What kind of fitting is used?

SALTFit will use MCMC only -- deterministic optimisation will not be of much use
except possibly during development. Even then it should only be added if the use
 case becomes clear.

## What will the front end look like?

We want to use an object-oriented approach similar to sklearn. Users will
initialise their model and then use a .fit_to() method to fit it to data. This
is just a rough idea that may need refinement depending on what is available in
SALT itself. Supposing we use this approach, we would have something like a
SALTModel or LineModel class which would contain all the parameters, priors,
model choice etc. Could be initialised simply with model = SALTModel(line_name,
rest_wavelength, profile_type). regardless of exactly how it's implemented, OOP
should be kept at the heart of the design philosophy as far as possible.

The public fitting API must depend only on a documented model-evaluation backend
contract, rather than on a specific SALT source layout or private upstream
functions. An adapter for a licensed SALT implementation can be added later
behind that contract.
