"""SDSS spectra example: reconstruct galaxy spectra streamed in place from spPlate FITS.

Mirrors astroML's SDSS spectral-PCA reconstruction (``compute_sdss_pca`` /
``fetch_sdss_corrected_spectra``), but where astroML first downloads the raw archive and
resamples every spectrum onto a common grid into a single ``spec4000.npz`` file, this streams
per-plate ``spPlate`` frames -- already on a common log-wavelength grid -- straight from the
SDSS archive as virtual references, with no reshard. See :mod:`examples.sdss.data`.
"""
