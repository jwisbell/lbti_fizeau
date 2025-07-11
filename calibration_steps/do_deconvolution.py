"""
do_deconvolution -- LIZARD Pipeline
Author: Jacob Isbell

Deconvolution functions for Fizeau images. Requires a science target and a calibrator to previously have been reduced.

Produces the final images and plots to diagnose quality.
Called by lizard_calibrate
"""

from typing import Dict
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import PowerNorm
from reduction_steps.do_image_corotation import recenter
from scipy.ndimage import rotate
from scipy.ndimage import gaussian_filter, median_filter, shift
from skimage import restoration
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Ellipse
from scipy.optimize import least_squares
import pickle
import json
from astropy.io import fits

from utils.util_logger import Logger
from utils.utils import argmax2d, gauss, imshift, find_max_loc, write_to_fits
from calibration_steps.bad_pixel_correction import identify_bad_pixels as bp_corr

PROCESS_NAME = "deconvolution"
is_flux_cal = False

PIXEL_SCALE = 0.018  # arcsec/pixel, NOMIC
w = 50
# NOMIC (default)
xticks = np.array(
    [
        -0.75 / PIXEL_SCALE + w,
        -0.5 / PIXEL_SCALE + w,
        -0.25 / PIXEL_SCALE + w,
        w,
        0.25 / PIXEL_SCALE + w,
        0.5 / PIXEL_SCALE + w,
        0.75 / PIXEL_SCALE + w,
    ]
)
xticklabels = ["", "-0.50", "", "0.0", "", "0.50", ""]
yticklabels = xticklabels[::1]


mpl.rcParams["font.family"] = "serif"
mpl.rcParams["xtick.direction"] = "in"
mpl.rcParams["xtick.top"] = True
mpl.rcParams["ytick.direction"] = "in"
mpl.rcParams["ytick.right"] = True


def do_convolution(im, psf):
    fft_im = np.fft.fft2(im)
    fft_psf = np.fft.fft2(psf)
    return np.abs(np.fft.fftshift(np.fft.ifft2(fft_im * fft_psf)))


def derotate(frame, rot):
    new_frame = rotate(frame, rot, reshape=False)
    return new_frame


def _write_fits_file(data, relerr, header_info, fname):
    hdr = fits.Header()
    for key, value in header_info.items():
        if key in ["target_config", "calib_config", "output_dir"]:
            continue
        try:
            if isinstance(value, dict):
                for k, v in value.items():
                    hdr[k] = v
            else:
                hdr[key] = value
        except ValueError:
            continue

    hdu = fits.PrimaryHDU(data=data, header=hdr)
    hdu2 = fits.ImageHDU(data=np.mean(relerr, 0) * data, name="IMERRS")
    hdul = fits.HDUList([hdu, hdu2])
    hdul.writeto(
        fname,
        overwrite=True,
    )


def fit_func(params, y):
    psf_est = y
    xv, yv = np.meshgrid(
        np.arange(-psf_est.shape[1] // 2, psf_est.shape[1] // 2),
        np.arange(-psf_est.shape[0] // 2, psf_est.shape[0] // 2),
    )
    model = gauss(
        xv,
        yv,
        0,
        0,
        params[0] / (2 * np.sqrt(2 * np.log(2))),
        params[1] / (2 * np.sqrt(2 * np.log(2))),
        params[2],
        np.mean(
            psf_est[
                psf_est.shape[0] // 2 - 3 : psf_est.shape[0] // 2 + 3,
                psf_est.shape[1] // 2 - 3 : psf_est.shape[1] // 2 + 3,
            ]
        ),
    )
    residuals = psf_est.flatten() - model.flatten()
    return residuals


def fit_gauss(psf_est, level=0.0):
    # return the major, minor, and pa of a gaussian
    # data = median_filter(psf_est, 3)
    data = np.copy(psf_est)
    cut = level * np.max(data)
    data -= cut
    data[data < 0] = 0

    x0 = [5, 5, 0]
    res = least_squares(fit_func, x0, args=([data]))
    xv, yv = np.meshgrid(
        np.arange(-psf_est.shape[1] // 2, psf_est.shape[1] // 2),
        np.arange(-psf_est.shape[0] // 2, psf_est.shape[0] // 2),
    )
    model = gauss(
        xv,
        yv,
        0,
        0,
        res.x[0] / (2 * np.sqrt(2 * np.log(2))),
        res.x[1] / (2 * np.sqrt(2 * np.log(2))),
        res.x[2],
        np.mean(
            psf_est[
                psf_est.shape[0] // 2 - 3 : psf_est.shape[0] // 2 + 3,
                psf_est.shape[1] // 2 - 3 : psf_est.shape[1] // 2 + 3,
            ]
        ),
    )
    """fig, (ax, bx, cx) = plt.subplots(1, 3)
    ax.imshow(psf_est, origin="lower", vmax=np.max(psf_est))
    bx.imshow(model, origin="lower", vmax=np.max(psf_est))
    cx.imshow(psf_est - model, origin="lower", vmax=np.max(psf_est))
    plt.show()
    plt.close()"""
    return [
        res.x[0] / (2 * np.sqrt(2 * np.log(2))),
        res.x[1] / (2 * np.sqrt(2 * np.log(2))),
        res.x[2],
    ], model


def _do_psf_subtraction(targ_image, psf_image, configdata, targname, calibname):
    # 1. first scale the psf to the target image flux
    psf_scaled = (psf_image / np.nanmax(psf_image)) * np.nanmax(targ_image)

    # 2. Recenter
    xc, yc = find_max_loc(targ_image, do_median=True)
    # psf_scaled = imshift(psf_scaled, yc, xc)

    # 3. Do the subtraction
    delta = targ_image - psf_scaled

    # 4. plot
    fig, ax = plt.subplots(figsize=(4.25, 4.0))
    im = ax.imshow(delta, origin="lower", cmap="magma", norm=PowerNorm(0.5, vmin=0))
    plt.colorbar(im, ax=ax, shrink=0.7, label="Flux [mJy/px]")
    outdir = configdata["output_dir"]
    plt.tight_layout()
    plt.savefig(
        f"{outdir}/plots/deconvolution/{targname}_{calibname}_psf_subtraction.png"
    )
    plt.close()
    np.save(
        f"{outdir}/calibrated/deconvolution/{targname}_{calibname}_psf_subtraction.npy",
        delta,
    )

    hdu = fits.PrimaryHDU(data=delta)
    hdul = fits.HDUList([hdu])
    hdul.writeto(
        f"{outdir}/calibrated/deconvolution/{targname}_{calibname}_psf_subtraction.fits",
        overwrite=True,
    )


def do_clean(
    dirty_im,
    psf_estimate,
    n_iter,
    gain=1e-4,
    threshold=1e-6,
    negstop=False,
    phat=0.0,
    resulting_im=None,
):
    k = 0
    im_0 = np.copy(dirty_im)
    im_i = np.copy(im_0)
    if resulting_im is None:
        resulting_im = np.zeros(im_0.shape)

    beam = np.copy(psf_estimate)
    beam /= np.max(beam)  # renormalize the psf estimate
    beam[beam.shape[0] // 2, beam.shape[1] // 2] += phat
    beam /= np.sum(beam)  # renormalize the psf estimate

    iterations = []
    delta = np.inf
    reason = f"niter: {n_iter}"
    new_im = np.copy(dirty_im)
    if threshold is not None:
        if threshold > 0:
            n_iter = 1e7  # something crazy long
    else:
        threshold = -1
    while k < n_iter:  # and delta > 1e-8:
        pk_y, pk_x = find_max_loc(im_i)

        if abs(im_i[pk_y, pk_x]) < threshold:
            reason = f"threshold ({k} iter)"
            break
        shifted_beam = np.copy(beam)
        shifted_beam = imshift(
            shifted_beam, -pk_y, -pk_x
        )  # move the psf to the "peak" location

        # scale shifted beam to flux * gain
        shifted_beam *= im_i[pk_y, pk_x] * gain

        new_im = im_i - shifted_beam
        delta = im_i[pk_y, pk_x] * gain  # np.sum(shifted_beam) #maybe?
        resulting_im[pk_y, pk_x] += (
            delta  # im_i[pk_x, pk_y] #* gain #np.max([im_i[pk_x, pk_y] * gain ,0])
        )
        # iterations.append([new_im, np.copy(resulting_im), delta]) #keep track of how things evolve
        assert np.abs(im_i[pk_y, pk_x]) == np.max(np.abs(im_i)), "what???"
        if negstop:
            if delta < 0:
                break
        if k % 100 == 0:
            iterations.append(delta / gain)
            # how much has the image changed
            if len(iterations) > 2:
                im_diff = np.abs(iterations[-2] - delta / gain)
                if im_diff < 5e-9:
                    reason = f"change too small ({k} iter)"
                    # break
        im_i = np.copy(new_im)  # update the image
        k += 1

    return resulting_im, new_im, iterations, reason


def _plot_beamsize(
    psf_estimate,
    minor,
    major,
    angle,
    output_dir,
    targname,
    mode="interactive",
):
    """
    Plots the FWHM of the psf calibration in comparison to the supplied ellipse parameters
    """
    fig1 = plt.figure()
    xv, yv = np.meshgrid(
        np.arange(psf_estimate.shape[1]), np.arange(psf_estimate.shape[0])
    )

    fitted_gauss, psf_model = fit_gauss(psf_estimate, level=0.25)
    # psf_model = imshift(psf_model, *find_max_loc(psf_model))
    print(fitted_gauss, "test")

    plt.contour(
        xv,
        yv,
        psf_estimate,
        levels=np.array([0.25, 0.50, 0.75]) * np.max(psf_estimate),
        cmap="viridis",
    )
    plt.contour(
        xv,
        yv,
        psf_model,
        levels=np.array([0.25, 0.50, 0.75]) * np.max(psf_model),
        cmap="Reds",
    )
    ell = Ellipse(
        xy=(psf_estimate.shape[1] // 2, psf_estimate.shape[0] // 2),
        width=minor,
        height=major,
        angle=angle,
        edgecolor="r",
        fc="None",
        lw=5,
        ls="-",
    )
    ax = plt.gca()
    ax.add_patch(ell)
    ax.set_aspect("equal")

    ell2 = Ellipse(
        xy=(psf_estimate.shape[1] // 2, psf_estimate.shape[0] // 2),
        width=fitted_gauss[1],
        height=fitted_gauss[0],
        angle=fitted_gauss[2],
        edgecolor="k",
        fc="None",
        lw=3,
        ls="--",
    )
    ax.add_patch(ell2)
    plt.title(
        f"Restoring Beam: approx {minor} x {major} px ({minor*PIXEL_SCALE*1000:0.1f} x {major*PIXEL_SCALE*1000:0.1f} mas)\n Fitted: {fitted_gauss[1]:0.1f}x{fitted_gauss[0]:0.1f}px, PA:{fitted_gauss[2]:0.1f}deg"
    )
    plt.savefig(f"{output_dir}/plots/{PROCESS_NAME}/psf_fwhm_{targname}.png")

    if mode == "interactive":
        fig1.show()
    else:
        plt.close()

    xv, yv = np.meshgrid(
        np.arange(-psf_estimate.shape[1] // 2, psf_estimate.shape[1] // 2),
        np.arange(-psf_estimate.shape[0] // 2, psf_estimate.shape[0] // 2),
    )
    model = gauss(
        xv,
        yv,
        0,
        -1,
        major,
        minor,
        angle,
        np.mean(
            psf_estimate[
                psf_estimate.shape[0] // 2 - 3 : psf_estimate.shape[0] // 2 + 3,
                psf_estimate.shape[1] // 2 - 3 : psf_estimate.shape[1] // 2 + 3,
            ]
        ),
    )

    return model


def wrap_clean(
    dirty_im, psf_estimate, configdata, target, configfile: str, mode="interactive"
):
    """
    Wraps the do_clean function, extracting the relevant parameters from the config files and plotting the results
    """

    try:
        n_iter = float(configdata["clean_niter"])  # 1e5
        gain = float(configdata["clean_gain"])  # 1e-3
        minor = float(configdata["clean_beam"]["minor"])
        major = float(configdata["clean_beam"]["major"])
        angle = float(configdata["clean_beam"]["rot"])
        phat = float(configdata["clean_phat"])
        threshold = float(configdata["clean_threshold"])
    except KeyError as e:
        logger.error(
            PROCESS_NAME, f"One or more config parameters for CLEAN is invalid: {e}"
        )
        return None, None, None

    if threshold == -1:
        threshold = None

    while True:
        fitted_gauss = _plot_beamsize(
            psf_estimate,
            minor,
            major,
            angle,
            configdata["output_dir"],
            target,
            mode=mode,
        )

        if mode != "interactive":
            plt.close("all")
            break
        else:
            logger.info(
                PROCESS_NAME,
                f"Current values -- major:{major:.1f}px\tminor:{minor:.1f}px\tpa:{angle:.2f}deg ",
            )
            command = input(
                "Modify the psf ('major XX', 'minor XX', 'pa XX') or 'okay':\t"
            )
            if command == "okay":
                # TODO: save the updated settings
                plt.close("all")
                break
            elif "major" in command:
                try:
                    tmp = float(command.split()[-1])
                    major = tmp
                except (SyntaxError, ValueError) as e:
                    logger.warn(PROCESS_NAME, f"Could not parse major axis value: {e}")
            elif "minor" in command:
                try:
                    tmp = float(command.split()[-1])
                    minor = tmp
                except (SyntaxError, ValueError) as e:
                    logger.warn(PROCESS_NAME, f"Could not parse minor axis value: {e}")
            elif "pa" in command:
                try:
                    tmp = float(command.split()[-1])
                    angle = tmp
                except (SyntaxError, ValueError) as e:
                    logger.warn(
                        PROCESS_NAME, f"Could not parse position angle value: {e}"
                    )
            else:
                print(f"Command '{command}' not recognized...")

            plt.close("all")
    logger.info(PROCESS_NAME, "Beam selection completed. Continuing ... ")

    resulting_im = None
    im_to_clean = np.copy(dirty_im)
    while True:
        logger.info(PROCESS_NAME, "Starting  CLEAN...")
        resulting_im, residual_im, iterations, _ = do_clean(
            im_to_clean,
            np.copy(psf_estimate),
            n_iter=n_iter,
            gain=gain,
            threshold=threshold,
            negstop=False,
            phat=phat,
            resulting_im=resulting_im,
        )

        xv, yv = np.meshgrid(
            np.arange(psf_estimate.shape[1]), np.arange(psf_estimate.shape[0])
        )
        mygauss = gauss(
            xv,
            yv,
            resulting_im.shape[1] // 2,
            resulting_im.shape[0] // 2,
            major / (2 * np.sqrt(2 * np.log(2))),
            minor / (2 * np.sqrt(2 * np.log(2))),
            angle,
            1,
        )
        mygauss /= np.sum(mygauss)

        lower_resolution = np.array(
            do_convolution(resulting_im, psf_estimate / np.sum(psf_estimate))
        )
        scaling_factor = np.max(dirty_im) / np.max(lower_resolution)
        # print(scaling_factor, np.sum(dirty_im) / np.sum(resulting_im))

        convim = derotate(
            gaussian_filter(
                derotate(resulting_im * scaling_factor, angle),
                (
                    major / (2 * np.sqrt(2 * np.log(2))),
                    minor / (2 * np.sqrt(2 * np.log(2))),
                ),
                mode="mirror",
            ),
            -angle,
        )  # + residual_im

        # convim2 = do_convolution(resulting_im * scaling_factor, mygauss)  # + residual_im

        lower_resolution *= scaling_factor
        print(np.sum(convim), np.sum(dirty_im), np.sum(convim) / np.sum(dirty_im))

        fig1, axarr = plt.subplots(2, 3, figsize=(12, 6))
        ax, bx, dx, cx, ex, fx = axarr.flatten()
        # ex.axis("off")
        gamma = 0.5

        cbar1 = ax.imshow(
            convim + residual_im * 0,
            origin="lower",
            cmap="Spectral_r",
            interpolation="none",
            norm=PowerNorm(vmin=0, vmax=0.95 * np.max(convim), gamma=gamma),
        )
        # fix
        cbar2 = bx.imshow(
            lower_resolution,
            origin="lower",
            cmap="magma",
            interpolation="none",
            norm=PowerNorm(vmin=0, gamma=gamma, vmax=None),
        )
        im = cx.imshow(
            residual_im,
            origin="lower",
            cmap="Spectral_r",
            interpolation="none",
            norm=PowerNorm(vmin=None, vmax=None, gamma=gamma),
        )

        cbar3 = dx.imshow(
            dirty_im,
            origin="lower",
            cmap="magma",
            interpolation="none",
            norm=PowerNorm(vmin=0, gamma=gamma, vmax=None),
        )
        cbar4 = fx.imshow(
            psf_estimate,
            origin="lower",
            cmap="magma",
            interpolation="none",
            norm=PowerNorm(vmin=0, gamma=gamma, vmax=1),
        )

        _ = ex.imshow(
            resulting_im,
            origin="lower",
            cmap="Greys",
            interpolation="none",
            norm=PowerNorm(vmin=0, gamma=gamma, vmax=None),
        )
        spacing = np.array(
            [0.9 / 512, 0.9 / 256, 0.9 / 128, 0.9 / 32, 0.9 / 8, 0.9 / 2]
        )
        ax.contour(
            xv,
            yv,
            convim + 0 * residual_im,
            levels=spacing * np.max(convim),
            colors="white",
            alpha=0.5,
        )

        ell = Ellipse(
            (10, 10),
            minor,
            major,
            edgecolor="w",
            fc="None",
            lw=2,
            angle=angle,
            hatch="/////",
        )
        ax.add_patch(ell)

        dx.set_title("Original Target Image")
        bx.set_title("Target Estimate (convolved)")
        cx.set_title("Residual map in last iteration")
        ax.set_title("CLEANed Image")
        fx.set_title("PSF Estimate (corotated)")

        flux_label = "Flux density [cts/px]"
        if is_flux_cal:
            flux_label = "Flux density [mJy/px]"

        plt.colorbar(im, ax=cx, shrink=0.75, label=flux_label)

        plt.colorbar(cbar1, ax=ax, shrink=0.75, label=flux_label)
        plt.colorbar(cbar2, ax=bx, shrink=0.75, label=flux_label)
        plt.colorbar(cbar3, ax=dx, shrink=0.75, label=flux_label)
        plt.colorbar(cbar4, ax=fx, shrink=0.75, label="Flux density [relative/px]")

        # ell = Ellipse((10, 10), minor, major, edgecolor="w", fc="None", lw=1, angle=angle)
        # ax.add_patch(ell)
        ell = Ellipse(
            (10, 10), minor, major, edgecolor="w", fc="None", lw=1, angle=angle
        )
        fx.add_patch(ell)

        w = convim.shape[0]

        # xticks = np.array([-w//2, -5/9*50 ,0, 5/9*50, w//2-1]) + w//2
        # xticklabels = [f'-{w//2*0.018:.1}', '-0.5','0', '0.5' ,f'{w//2*0.018:.1}' ]
        ax.set_xticks(xticks[::-1])
        ax.set_xticklabels(xticklabels)
        bx.set_xticks(xticks[::-1])
        bx.set_xticklabels(xticklabels)
        cx.set_xticks(xticks[::-1])
        cx.set_xticklabels(xticklabels)
        dx.set_xticks(xticks[::-1])
        dx.set_xticklabels(xticklabels)

        ax.set_yticks(xticks)
        ax.set_yticklabels(xticklabels)
        bx.set_yticks(xticks)
        bx.set_yticklabels(xticklabels)
        cx.set_yticks(xticks)
        cx.set_yticklabels(xticklabels)
        dx.set_yticks(xticks)
        dx.set_yticklabels(xticklabels)

        fx.set_xticks(xticks[::-1])
        fx.set_xticklabels(xticklabels)
        fx.set_yticks(xticks)
        fx.set_yticklabels(xticklabels)

        ax.set_xlabel(r"$\delta$ RA [arcsec]")
        ax.set_ylabel(r"$\delta$ DEC [arcsec]")
        fig1.subplots_adjust(wspace=0.1, hspace=0.1)
        fig1.tight_layout()
        fig1.savefig(
            f"{configdata['output_dir']}/plots/{PROCESS_NAME}/{target}_clean_deconvolution.pdf"
        )

        fig2, ax2 = plt.subplots()
        ax2.plot(np.arange(len(iterations)) * 100, iterations)
        ax2.set_yscale("log")
        fig2.savefig(
            f"{configdata['output_dir']}/plots/{PROCESS_NAME}/{target}_clean_iterations.png"
        )

        fig3, ax3 = plt.subplots()
        g = np.copy(mygauss)
        g /= np.max(g)
        g *= np.max(convim)
        cbar1 = ax3.imshow(
            g,
            origin="lower",
            cmap="Spectral_r",
            interpolation="none",
            norm=PowerNorm(vmin=0, vmax=0.95 * np.max(convim), gamma=gamma),
        )
        ax3.set_xlabel(r"$\delta$ RA [arcsec]")
        ax3.set_ylabel(r"$\delta$ DEC [arcsec]")
        ax3.set_xticks(xticks[::-1])
        ax3.set_xticklabels(xticklabels)
        ax3.set_yticks(xticks)
        ax3.set_yticklabels(xticklabels)
        ax3.set_aspect("equal")

        fig1.show()
        fig2.show()
        fig3.show()

        if mode != "interactive":
            plt.close("all")
            break
        else:
            logger.info(
                PROCESS_NAME,
                f"Current values-- niter:{n_iter:0f}\tgain:{gain:.7f}\tphat:{phat:.2f}",
            )
            command = input(
                "Change clean parameters ('niter XX', 'gain XX', 'phat XX', 'continue XXX', 'reset', 'automatic XX'), 'help' or 'okay':\t"
            )
            if command == "okay":
                do_save = input("Save your parameters? (y | n): ")
                if do_save == "y" or do_save == "yes":
                    with open(configfile) as cfg:
                        old_configdata = json.load(cfg)
                    new_configdata = {k: v for k, v in old_configdata.items()}
                    new_configdata["clean_niter"] = n_iter
                    new_configdata["clean_gain"] = gain
                    new_configdata["clean_phat"] = phat
                    new_configdata["clean_threshold"] = threshold
                    new_configdata["clean_beam"] = {
                        "major": major,
                        "minor": minor,
                        "rot": angle,
                    }
                    with open(configfile, "w") as file:
                        json.dump(new_configdata, file, indent=4)
                    logger.info(PROCESS_NAME, "New parameters saved!")

                plt.close("all")
                break
            if command == "help":
                print("help info todo")
            elif "niter" in command:
                try:
                    tmp = int(float(command.split()[-1]))
                    n_iter = tmp
                    im_to_clean = np.copy(dirty_im)
                    resulting_im = None
                    threshold = -1
                except SyntaxError as e:
                    logger.warn(PROCESS_NAME, f"Could not parse niter value: {e}")
            elif "continue" in command:
                try:
                    tmp = int(float(command.split()[-1]))
                    n_iter = tmp
                    im_to_clean = residual_im
                except SyntaxError as e:
                    logger.warn(PROCESS_NAME, f"Could not parse continue value: {e}")
            elif "reset" in command:
                try:
                    n_iter = float(configdata["clean_niter"])  # 1e5
                    gain = float(configdata["clean_gain"])  # 1e-3
                    phat = float(configdata["clean_phat"])
                    im_to_clean = np.copy(dirty_im)
                    resulting_im = None
                except SyntaxError as e:
                    logger.warn(PROCESS_NAME, f"Could not parse reset : {e}")
            elif "gain" in command:
                try:
                    tmp = float(command.split()[-1])
                    gain = tmp
                    im_to_clean = np.copy(dirty_im)
                    resulting_im = None
                    threshold = -1
                except SyntaxError as e:
                    logger.warn(PROCESS_NAME, f"Could not parse gain value: {e}")
            elif "automatic" in command:
                try:
                    tmp = float(command.split()[-1])
                    threshold = tmp
                    n_iter = float(configdata["clean_niter"])  # 1e5
                    gain = float(configdata["clean_gain"])  # 1e-3
                    phat = float(configdata["clean_phat"])
                    im_to_clean = np.copy(dirty_im)
                    resulting_im = None
                except SyntaxError as e:
                    logger.warn(
                        PROCESS_NAME, f"Could not parse automatic threshold value: {e}"
                    )
            elif "phat" in command:
                try:
                    tmp = float(command.split()[-1])
                    phat = tmp
                    im_to_clean = np.copy(dirty_im)
                    resulting_im = None
                except SyntaxError as e:
                    logger.warn(PROCESS_NAME, f"Could not parse phat value: {e}")
            else:
                print(f"Command not recognized: '{command}'")

            plt.close("all")

    return convim, residual_im, resulting_im


def wrap_rl(dirty_im, psf_estimate, configdata, target):
    """
    Wraps the richardson_lucy deconvolution, extracting relevant config parameters and plotting the results
    """

    try:
        niter = int(configdata["rl_niter"])
        eps = float(configdata["rl_eps"])
    except KeyError as e:
        logger.error(
            PROCESS_NAME,
            f"One or more config parameters for R-L deconvolution is invalid: {e}",
        )
        return None

    gamma = 0.5

    normalized_psf = psf_estimate / np.sum(psf_estimate)

    deconvolved_RL = restoration.richardson_lucy(
        dirty_im, normalized_psf, num_iter=niter, filter_epsilon=eps, clip=False
    )

    xv, yv = np.meshgrid(
        np.arange(psf_estimate.shape[1]), np.arange(psf_estimate.shape[0])
    )

    # start the plotting
    fig = plt.figure(figsize=(8.5, 4), layout="constrained")
    gs = GridSpec(1, 3, figure=fig, width_ratios=[0.495, 0.02, 0.495])
    ax = fig.add_subplot(gs[0])
    bx = fig.add_subplot(gs[2])
    cx = fig.add_subplot(gs[1])

    cbar_im = ax.imshow(
        deconvolved_RL,
        origin="lower",
        norm=PowerNorm(gamma, vmin=0, vmax=np.max(deconvolved_RL)),
        interpolation="gaussian",
        cmap="Spectral_r",
    )

    levels = np.array(
        [0.9 / 512, 0.9 / 256, 0.9 / 32, 0.9 / 16, 0.9 / 8, 0.9 / 4, 0.9 / 2, 0.9]
    ) * np.max(deconvolved_RL)
    bx.contour(
        xv,
        yv,
        deconvolved_RL,
        origin="lower",
        levels=levels,  # np.array([2,3,4,5,10,20,50])*noise,
        colors="k",
        norm=PowerNorm(gamma, vmin=0, vmax=np.max(deconvolved_RL)),
    )
    bx.set_aspect("equal")
    ax.set_aspect("equal")
    ax.set_xticks(xticks[::-1] - 1)
    ax.set_xticklabels(xticklabels, fontsize="small")
    ax.set_yticks(xticks - 4)
    ax.set_yticklabels(yticklabels, fontsize="small")
    bx.set_xticks(xticks[::-1] - 1)
    bx.set_xticklabels(xticklabels, fontsize="small")
    bx.set_yticks(xticks - 4)
    bx.set_yticklabels(yticklabels, fontsize="small")
    bx.grid()
    ax.text(11, 90, r"N", color="white")
    ax.text(1, 78, r"E", color="white")
    ax.plot([14, 14], [80, 88], "white")
    ax.plot([7, 14], [80, 80], "white")

    bx.text(11, 90, r"N", color="k")
    bx.text(1, 78, r"E", color="k")
    bx.plot([14, 14], [80, 88], "k")
    bx.plot([7, 14], [80, 80], "k")

    ax.set_xlabel(r"$\Delta\alpha$ [arcseconds]", fontsize="small")
    ax.set_ylabel(r"$\Delta\delta$ [arcseconds]", fontsize="small")
    cx.axis("off")

    flux_label = "Flux density [cts/px]"
    if is_flux_cal:
        flux_label = "Flux density [mJy/px]"
    plt.colorbar(
        cbar_im,
        ax=cx,
        label=flux_label,
        fraction=1.2,
        shrink=1,
        location="right",
        extend="max",
    )
    # plt.suptitle(f'N_iter = {n}, eps={eps}')
    # plt.savefig(f'../plots/fig3.pdf')#deconv_rl_n{n}_eps1e-2_pa{PA}.png')
    plt.savefig(
        f"{configdata['output_dir']}/plots/{PROCESS_NAME}/{target}_rl_deconvolution.png"
    )
    plt.close()

    # save the results
    # np.save(f"{configdata['output_dir']}/calibrated/{PROCESS_NAME}/{target}_RL_deconvolved_n{niter}_eps{eps_str}.npy", deconvolved_RL)

    return deconvolved_RL


def clean_test(debug=False):
    # 1. Make a fake dirty image with one point at x,y
    s = 10
    dirty_im = np.zeros((s, s))
    y, x = np.random.randint(0, s, 2)
    dirty_im[y, x] = 1
    # 2. Find the max val and assert they are the same
    max_y, max_x = find_max_loc(dirty_im, False)
    assert x == max_x and y == max_y, f"Expected {x,y}, got {max_x, max_y}"
    # 3. Shift a mock psf to that location
    psf = np.zeros((s, s))
    psf[psf.shape[0] // 2, psf.shape[1] // 2] = 1
    temp = imshift(psf, -max_y, -max_x)
    py, px = find_max_loc(temp, False)
    if debug:
        fig, (ax, bx) = plt.subplots(1, 2)
        ax.imshow(dirty_im, origin="lower")
        bx.imshow(temp, origin="lower")
        plt.show()
    # 4. assert that the max location of the psf is at the original location
    assert px == x and py == y, f"Expected {x,y}, got {px,py}"

    return 1


def do_deconvolution(
    configdata: dict,
    target_configdata: dict,
    calib_configdata: dict,
    mylogger: Logger,
    configfile: str,
    interactive=True,
) -> bool:
    """
    Wraps the two deconvolution methods, extracting relevant config parameters and saving the results.
    Calls wrap_clean and wrap_rl
    returns true if successful, false if there are errors
    """

    global logger
    logger = mylogger

    try:
        targ_output_dir = target_configdata["output_dir"]
        targname = target_configdata["target"]
        obsdate = target_configdata["obsdate"]
        output_dir = configdata["output_dir"]

        try:
            do_bpm = bool(configdata["estimate_bad_pixels"])
        except KeyError:
            logger.warn(PROCESS_NAME, "`estimate_bad_pixels` not set, assuming False")
            do_bpm = False

        # Load the dirty image
        dirty_im = np.load(
            f"{targ_output_dir}/intermediate/corotate/{targname}_corotated_stacked_im.npy"
        )
        dirty_im -= np.min(dirty_im)
        dirty_im -= np.mean(
            dirty_im[: dirty_im.shape[0] // 4, : dirty_im.shape[1] // 4]
        )

        dirty_im = imshift(
            dirty_im, *find_max_loc(dirty_im, do_median=True)
        )  # recenter
        if do_bpm:
            dirty_im, _ = bp_corr(dirty_im)

        # Load the psf image
        calibname = calib_configdata["target"]
        psf_estimate = np.load(
            f"{output_dir}/calibrated/estimate_final_psf/{calibname}_for_{targname}_{obsdate}.npy"
        )

        psf_estimate -= np.mean(psf_estimate[:20, :20])
        psf_estimate /= np.max(psf_estimate)
        psf_estimate = imshift(
            psf_estimate, *find_max_loc(psf_estimate, do_median=True)
        )  # recenter
        if do_bpm:
            psf_estimate, _ = bp_corr(psf_estimate)

        if target_configdata["instrument"] != "NOMIC":
            global PIXEL_SCALE
            global xticks
            global xticklabels
            global yticklabels
            PIXEL_SCALE = 0.0107  # lmircam arcsec/pixel
            w = dirty_im.shape[0] // 2

            xticks = np.array(
                [
                    -0.5 / PIXEL_SCALE + w,
                    -0.25 / PIXEL_SCALE + w,
                    w,
                    0.25 / PIXEL_SCALE + w,
                    0.5 / PIXEL_SCALE + w,
                ]
            )
            xticklabels = ["-0.50", "-0.25", "0.0", "0.25", "0.50"]
            yticklabels = xticklabels[::1]
        else:
            w = dirty_im.shape[0] // 2
            xticks = np.array(
                [
                    -0.75 / PIXEL_SCALE + w,
                    -0.5 / PIXEL_SCALE + w,
                    -0.25 / PIXEL_SCALE + w,
                    w,
                    0.25 / PIXEL_SCALE + w,
                    0.5 / PIXEL_SCALE + w,
                    0.75 / PIXEL_SCALE + w,
                ]
            )
            xticklabels = ["", "-0.50", "", "0.0", "", "0.50", ""]
            yticklabels = xticklabels[::1]
    except KeyError as e:
        logger.error(PROCESS_NAME, f"One or more config entries is incorrect: {e}")
        return False

    # 1. Attempt flux calibration, if files are not available, set a global flag for plot labels
    global is_flux_cal
    try:
        flux_fname = f"{output_dir}/calibrated/flux_calibration/sci_{targname}_with_cal_{calibname}_flux_percentiles.npy"
        flux_percentiles = np.load(flux_fname)
        is_flux_cal = True
        dirty_im /= np.sum(dirty_im)
        dirty_im *= flux_percentiles[1] * 1000
        # dirty_im_errs = np.array(
        #     [flux_percentiles[-2] * dirty_im, flux_percentiles[-1] * dirty_im]
        # )
        logger.info(PROCESS_NAME, "Flux calibration successfully loaded!")
        np.save(
            f"{output_dir}/calibrated/flux_calibration/sci_{targname}_with_cal_{calibname}_flux_calibrated_im.npy",
            dirty_im,
        )

        write_to_fits(
            dirty_im,
            f"{output_dir}/calibrated/flux_calibration/sci_{targname}_with_cal_{calibname}_flux_calibrated.fits",
        )

        relerrs = [flux_percentiles[-2], flux_percentiles[-1]]

    except FileNotFoundError:
        # flux files not present
        is_flux_cal = False
        logger.warn(PROCESS_NAME, "Proceeding without proper flux calibration")
        relerrs = [0, 0]

    _do_psf_subtraction(dirty_im, psf_estimate, configdata, targname, calibname)

    clean_restored, clean_residual, clean_pt_src = wrap_clean(
        dirty_im,
        psf_estimate,
        configdata,
        targname,
        configfile,
        mode="interactive",
    )
    if clean_restored is None:
        return False

    deconvolved_RL = wrap_rl(dirty_im, psf_estimate, configdata, targname)
    if deconvolved_RL is None:
        return False

    # save the results in a pkl file...
    datadict = {
        "clean_restored": clean_restored,
        "clean_residual": clean_residual,
        "psf": psf_estimate,
        "clean_pt_src": clean_pt_src,
        "dirty_im": dirty_im,
        "rl": deconvolved_RL,
        "relative_flux_errs": relerrs,
    }
    res = "matched"
    with open(
        f"{output_dir}/calibrated/{PROCESS_NAME}/{targname}_convolved_cleaned.pkl",
        "wb",
    ) as handle:
        pickle.dump(datadict, handle, protocol=pickle.HIGHEST_PROTOCOL)

    # save the images as fits files for sharing
    _write_fits_file(
        deconvolved_RL,
        relerrs,
        configdata,
        f"{output_dir}/calibrated/{PROCESS_NAME}/{targname}_deconvolved_rl.fits",
    )

    _write_fits_file(
        clean_restored,
        relerrs,
        configdata,
        f"{output_dir}/calibrated/{PROCESS_NAME}/{targname}_deconvolved_clean.fits",
    )

    return True
