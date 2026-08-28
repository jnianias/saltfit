import numpy as np
import pytest

from saltfit import Spectrum


def test_spectrum_normalises_inputs_and_exposes_likelihood_helpers():
    spectrum = Spectrum(
        velocity=[-100.0, 0.0, 100.0],
        flux=[1.0, 0.8, 1.1],
        uncertainty=[0.1, 0.2, 0.1],
    )

    assert spectrum.size == 3
    assert isinstance(spectrum.velocity, np.ndarray)
    np.testing.assert_allclose(spectrum.weights, [100.0, 25.0, 100.0])


@pytest.mark.parametrize(
    ("velocity", "flux", "uncertainty", "message"),
    [
        ([0.0, 1.0], [1.0], [0.1, 0.1], "same shape"),
        ([0.0, 0.0], [1.0, 1.0], [0.1, 0.1], "strictly increasing"),
        ([0.0, 1.0], [1.0, 1.0], [0.1, 0.0], "positive"),
        ([0.0, np.nan], [1.0, 1.0], [0.1, 0.1], "finite"),
    ],
)
def test_spectrum_rejects_invalid_data(velocity, flux, uncertainty, message):
    with pytest.raises(ValueError, match=message):
        Spectrum(velocity=velocity, flux=flux, uncertainty=uncertainty)


def test_masked_returns_selected_samples():
    spectrum = Spectrum(
        velocity=[-100.0, 0.0, 100.0],
        flux=[1.0, 0.8, 1.1],
        uncertainty=[0.1, 0.2, 0.1],
    )

    masked = spectrum.masked([True, False, True])

    np.testing.assert_array_equal(masked.velocity, [-100.0, 100.0])
    np.testing.assert_array_equal(masked.flux, [1.0, 1.1])


def test_spectrum_loads_an_ascii_table(tmp_path):
    pytest.importorskip("astropy.table")
    path = tmp_path / "spectrum.txt"
    path.write_text("velocity flux uncertainty\n-100.0 1.0 0.1\n0.0 0.8 0.2\n100.0 1.1 0.1\n")

    spectrum = Spectrum.from_file(
        path,
        velocity_column="velocity",
        flux_column="flux",
        uncertainty_column="uncertainty",
        format="ascii.basic",
    )

    np.testing.assert_allclose(spectrum.velocity, [-100.0, 0.0, 100.0])
    np.testing.assert_allclose(spectrum.flux, [1.0, 0.8, 1.1])


def test_spectrum_loads_a_fits_table(tmp_path):
    astropy_table = pytest.importorskip("astropy.table")
    table = astropy_table.Table({
        "velocity": [-100.0, 0.0, 100.0],
        "flux": [1.0, 0.8, 1.1],
        "uncertainty": [0.1, 0.2, 0.1],
    })

    path = tmp_path / "spectrum.fits"
    table.write(path, format="fits")

    spectrum = Spectrum.from_file(
        path,
        velocity_column="velocity",
        flux_column="flux",
        uncertainty_column="uncertainty",
    )

    np.testing.assert_allclose(spectrum.velocity, [-100.0, 0.0, 100.0])
    np.testing.assert_allclose(spectrum.flux, [1.0, 0.8, 1.1])
    np.testing.assert_allclose(spectrum.uncertainty, [0.1, 0.2, 0.1])
