from calibration_steps.do_deconvolution import fit_pixels_adam, derotate
from astropy.io import fits
from scipy.ndimage import rotate, gaussian_filter
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import PowerNorm


dirty_im = fits.open(
    # "/Users/jwisbell/Documents/lbti/lbti_scripting/reduction/mrk1239_test/Mrk 1239 M-band corotated stacked psf.fits"
    "/Users/jwisbell/Documents/lbti/2025A/processed_data/ngc1068_m/calibrated/flux_calibration/sci_NGC 1068_with_cal_HD19305_flux_calibrated.fits"
)[0].data

dirty_im -= np.nanmean(dirty_im[:30, :30])
dirty_im[dirty_im < 0] = 0
# dirty_im -= np.nanmin(dirty_im)


psf_estimate = fits.open(
    # "/Users/jwisbell/Documents/lbti/lbti_scripting/reduction/mrk1239_test/TYC M-band corotated stacked psf.fits"
    "/Users/jwisbell/Documents/lbti/2025A/processed_data/ngc1068_m/plots/estimate_final_psf/psf_estimate_HD19305_for_NGC 1068_250116.fits"
)[0].data

psf_estimate -= np.nanmean(psf_estimate[:20, :20])
# psf_estimate[psf_estimate < 0] = 0
# psf_estimate -= np.nanmin(psf_estimate)
psf_estimate /= np.max(psf_estimate)
psf_estimate /= np.sum(psf_estimate)

forwardmodel = fit_pixels_adam(
    dirty_im / np.max(dirty_im),
    psf_estimate,
    lr=0.05,
    alpha=1e-7,
    mode="l1",
    iterations=900,
)
major = 4.7e-6 / 8.4 / 2 * 206265 * 1000 / 11
minor = 4.7e-6 / 8.4 / 2 * 206265 * 1000 / 11

fm_conv = derotate(
    gaussian_filter(
        derotate(forwardmodel * 1, 0),
        (
            major / (2 * np.sqrt(2 * np.log(2))),
            minor / (2 * np.sqrt(2 * np.log(2))),
        ),
        mode="mirror",
    ),
    0,
)  # + residual_im

fig, (ax, bx, cx, dx) = plt.subplots(1, 4, sharex=True, sharey=True)
ax.imshow(
    forwardmodel,
    origin="lower",
    norm=PowerNorm(gamma=0.5, vmin=0),
    cmap="Spectral_r",
)

bx.imshow(
    fm_conv,
    origin="lower",
    norm=PowerNorm(gamma=0.5, vmin=0),
    cmap="Spectral_r",
)
fm_conv = np.abs(
    np.fft.fftshift(
        np.fft.ifft2(
            np.fft.fft2(forwardmodel) * np.fft.fft2(psf_estimate / np.sum(psf_estimate))
        )
    )
)
cx.imshow(
    fm_conv,
    origin="lower",
    norm=PowerNorm(gamma=0.5, vmin=0),
    cmap="Spectral_r",
)

dx.imshow(
    dirty_im / np.nanmax(dirty_im),
    origin="lower",
    norm=PowerNorm(gamma=0.5, vmin=0),
    cmap="Spectral_r",
)
plt.show()
