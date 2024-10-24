"""
do_deconvolution -- LIZARD Pipeline
Author: Jacob Isbell

Deconvolution functions for Fizeau images. Requires a science target and a calibrator to previously have been reduced.

Produces the final images and plots to diagnose quality.
Called by lizard_calibrate
"""

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import PowerNorm
from scipy.ndimage import rotate
from scipy.ndimage import gaussian_filter
from scipy.signal import convolve2d
from skimage import restoration
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Ellipse
import pickle

from utils.util_logger import Logger
from utils.utils import gauss

PROCESS_NAME = "deconvolution"
is_flux_cal = False

# TODO: handle this
PIXEL_SCALE = 0.018  # arcsec/pixel, NOMIC
# PIXEL_SCALE = 0.0107 #lmircam arcsec/pixel
w = 49
# NOMIC
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

# LMIRCAM
# xticks = np.array([-0.5/PIXEL_SCALE+w,-0.25/PIXEL_SCALE+w,w,0.25/PIXEL_SCALE+w,0.5/PIXEL_SCALE+w])
# xticklabels = ['-0.50','-0.25','0.0','0.25','0.50']

# both
yticklabels = xticklabels[::1]


mpl.rcParams["font.family"] = "serif"
mpl.rcParams["xtick.direction"] = "in"
mpl.rcParams["xtick.top"] = True
mpl.rcParams["ytick.direction"] = "in"
mpl.rcParams["ytick.right"] = True


def find_max_loc(im):
    # find the x,y coords of peak of 2d array
    # w = im.shape[0]
    # temp_im = np.copy(im)[w//2-w//4:w//2+w//4,w//2-w//4:w//2+w//4]
    idx = np.argmax(np.abs(im))
    # idx  = np.argmax(im )
    x, y = np.unravel_index(idx, im.shape)

    return x, y


# unit test for find_max_loc
# print(find_max_loc(psf_estimate))


def imshift(im, x, y):
    # roll the im along each axis so that peak is at x,y
    w = im.shape[0] // 2
    temp_im = np.roll(im, x - w, axis=1)
    return np.roll(temp_im, y - w, axis=0)


def do_convolution(im, psf):
    fft_im = np.fft.fft2(im)
    fft_psf = np.fft.fft2(psf)
    return np.abs(np.fft.fftshift(np.fft.ifft2(fft_im * fft_psf)))


def derotate(frame, rot):
    new_frame = rotate(frame, rot, reshape=False)
    return new_frame


def do_clean(
    dirty_im, psf_estimate, n_iter, gain=1e-4, threshold=1e-6, negstop=False, phat=0.0
):
    k = 0
    im_0 = np.copy(dirty_im)
    im_i = np.copy(im_0)
    resulting_im = np.zeros(im_0.shape)

    beam = np.copy(psf_estimate)
    beam /= np.max(beam)  # renormalize the psf estimate
    beam[50, 50] += phat
    beam /= np.max(beam)  # renormalize the psf estimate

    iterations = []
    delta = np.inf
    reason = f"niter: {n_iter}"
    new_im = np.copy(dirty_im)
    while k < n_iter:  # and delta > 1e-8:
        pk_x, pk_y = find_max_loc(im_i)
        if im_i[pk_x, pk_y] < threshold:
            reason = f"threshold ({k} iter)"
            # break
        shifted_beam = np.copy(beam)
        shifted_beam = imshift(
            shifted_beam, pk_y, pk_x
        )  # move the psf to the "peak" location
        # scale shifted beam to flux * gain
        shifted_beam *= im_i[pk_x, pk_y] * gain

        new_im = im_i - shifted_beam
        delta = im_i[pk_x, pk_y] * gain  # np.sum(shifted_beam) #maybe?
        resulting_im[pk_x, pk_y] += (
            delta  # im_i[pk_x, pk_y] #* gain #np.max([im_i[pk_x, pk_y] * gain ,0])
        )
        # iterations.append([new_im, np.copy(resulting_im), delta]) #keep track of how things evolve
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


def _plot_beamsize(psf_estimate, minor, major, angle, output_dir, targname):
    """
    Plots the FWHM of the psf calibration in comparison to the supplied ellipse parameters
    """
    _ = plt.figure()
    xv, yv = np.meshgrid(
        np.arange(psf_estimate.shape[0]), np.arange(psf_estimate.shape[0])
    )

    w = psf_estimate.shape[0] // 2
    plt.contour(xv, yv, psf_estimate, levels=[0.50 * np.max(psf_estimate)])
    ell = Ellipse(
        xy=(w, w),
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
    plt.title(
        f"Restoring Beam: approx {minor} x {major} px ({minor*PIXEL_SCALE*1000:0.1f} x {major*PIXEL_SCALE*1000:0.1f} mas)"
    )
    plt.savefig(f"{output_dir}/plots/{PROCESS_NAME}/psf_fwhm_{targname}.png")
    plt.close()


def wrap_clean(dirty_im, psf_estimate, configdata, target):
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
    except KeyError as e:
        logger.error(
            PROCESS_NAME, f"One or more config parameters for CLEAN is invalid: {e}"
        )
        return None, None, None

    _plot_beamsize(psf_estimate, minor, major, angle, configdata["output_dir"], target)

    resulting_im, residual_im, iterations, _ = do_clean(
        dirty_im,
        psf_estimate,
        n_iter=n_iter,
        gain=gain,
        threshold=10,
        negstop=False,
        phat=phat,
    )

    xv, yv = np.meshgrid(
        np.arange(resulting_im.shape[0]), np.arange(resulting_im.shape[0])
    )
    mygauss = gauss(
        xv,
        yv,
        resulting_im.shape[0] // 2,
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
    print(scaling_factor, np.sum(dirty_im) / np.sum(resulting_im))

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

    convim2 = do_convolution(resulting_im * scaling_factor, mygauss)  # + residual_im

    lower_resolution *= scaling_factor

    _, axarr = plt.subplots(2, 3, figsize=(12, 6))
    ax, bx, dx, cx, ex, fx = axarr.flatten()
    ex.axis("off")
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

    spacing = np.array([0.9 / 512, 0.9 / 256, 0.9 / 128, 0.9 / 32, 0.9 / 8, 0.9 / 2])
    ax.contour(xv, yv, convim, levels=spacing * np.max(convim), colors="white")

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
    ell = Ellipse((10, 10), minor, major, edgecolor="w", fc="None", lw=1, angle=angle)
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
    plt.subplots_adjust(wspace=0.1, hspace=0.1)
    plt.tight_layout()
    plt.savefig(
        f"{configdata['output_dir']}/plots/{PROCESS_NAME}/{target}_clean_deconvolution.png"
    )
    plt.close()

    _ = plt.figure()
    plt.plot(np.arange(len(iterations)) * 100, iterations)
    plt.yscale("log")
    plt.savefig(
        f"{configdata['output_dir']}/plots/{PROCESS_NAME}/{target}_clean_iterations.png"
    )
    plt.close()

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

    xv, yv = np.meshgrid(range(deconvolved_RL.shape[0]), range(deconvolved_RL.shape[1]))

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
        [0.9 / 256, 0.9 / 32, 0.9 / 16, 0.9 / 8, 0.9 / 4, 0.9 / 2, 0.9]
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


def do_deconvolution(
    configdata: dict, target_configdata: dict, calib_configdata: dict, mylogger: Logger
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
        dirty_im = np.load(
            f"{targ_output_dir}/intermediate/corotate/{targname}_corotated_stacked_im.npy"
        )
        dirty_im -= np.min(dirty_im)

        output_dir = configdata["output_dir"]
        calibname = calib_configdata["target"]
        psf_estimate = np.load(
            f"{output_dir}/calibrated/estimate_final_psf/{calibname}_for_{targname}_{obsdate}.npy"
        )

        psf_estimate -= np.mean(psf_estimate[:20, :20])
        psf_estimate /= np.max(psf_estimate)
        psf_estimate = imshift(psf_estimate, *find_max_loc(psf_estimate))  # recenter

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
        logger.info(PROCESS_NAME, "Flux calibration successfully loaded!")

    except FileNotFoundError:
        # flux files not present
        is_flux_cal = False
        logger.warn(PROCESS_NAME, "Proceeding without proper flux calibration")

    clean_restored, clean_residual, clean_pt_src = wrap_clean(
        dirty_im, psf_estimate, configdata, targname
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
    }
    res = "matched"
    with open(
        f"{output_dir}/calibrated/{PROCESS_NAME}/{targname}_convolved_cleaned_{res}.pkl",
        "wb",
    ) as handle:
        pickle.dump(datadict, handle, protocol=pickle.HIGHEST_PROTOCOL)

    return True
