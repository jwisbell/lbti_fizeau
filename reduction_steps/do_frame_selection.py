"""
do_frame_selection -- LIZARD Pipeline
Author: Jacob Isbell

Functions to do lucky fringe frame selection using the cross-correlation between the observation frames and either
a model PSF or an emperical, calibrated PSF.
Called by lizard_reduce
"""

import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import correlate2d
from matplotlib.colors import PowerNorm
import pickle
from scipy.ndimage import zoom
from glob import glob
from utils.utils import argmax2d, gauss
from utils.util_logger import Logger


PROCESS_NAME = "frame_selection"
logger: Logger = Logger("./")


# Function Definitions
def _frame_selection_scores_cc(images, psf, keep_fraction=0.1, debug=False):
    """
    Function to select frames based on their cross correlation to the reference psf. A fraction `keep_fraction` is used to make up a final, stacked image.
    """
    correlation_vals = []
    corrected_ims = []
    shiftsx = []
    shiftsy = []
    for temp_im in images:
        im = np.copy(temp_im)

        # renormalize the images
        s = np.nansum(im)
        im -= np.min(im)
        im /= np.nanmax(im)
        psf -= np.min(psf)
        psf /= np.nanmax(psf)

        # do the cross correlation
        cross_correlation = correlate2d(
            im, np.copy(psf), mode="same", boundary="fill", fillvalue=0
        )
        max_row, max_col = argmax2d(cross_correlation)
        shift_x = max_row[0] - im.shape[0] // 2 + 1
        shift_y = max_col[0] - im.shape[1] // 2 + 1

        # shift the image accordingly
        new_im = np.roll(temp_im, -shift_x, axis=1)
        new_im = np.roll(new_im, -shift_y, axis=0)

        corrected_ims.append(new_im)
        shiftsx.append(shift_x)
        shiftsy.append(shift_y)

        test_im = np.copy(new_im)
        test_im -= np.mean(test_im)
        test_im /= np.max(test_im)

        correlation_vals.append(
            np.sum(np.square(test_im - psf)) / np.square(len(new_im))
        )  #

        if debug:
            _, (ax, bx) = plt.subplots(1, 2)
            t = new_im / np.sum(new_im)
            t -= np.mean(t)
            t /= np.max(t)
            ax.plot(t[14, :], label="shifted im")
            ax.plot(im[14, :], label="original im")
            ax.plot(psf[14, :], label="reference psf")
            bx.imshow(im, origin="lower")
            bx.scatter(14 + shift_x, 14 + shift_y, marker="x", color="r")
            ax.legend()
            plt.show()
            plt.close()

    s = np.argsort(correlation_vals)
    mask = np.zeros(len(correlation_vals))
    mask[s[-int(len(s) * keep_fraction) :]] = 1
    return (
        correlation_vals,
        corrected_ims,
        {"shiftsx": shiftsx, "shiftsy": shiftsy, "mask": mask.astype("bool")},
    )


def shift_images(im_arr, cc_info):
    """
    Function to shift an array of images by an array of (x,y) coordinates
    Used to co-align the selected frames
    """
    shiftsx = cc_info["shiftsx"]
    shiftsy = cc_info["shiftsy"]
    shifted_images = []
    for i, im in enumerate(im_arr):
        temp_im = np.copy(im)
        shift_x = shiftsx[i]
        shift_y = shiftsy[i]

        new_im = np.roll(temp_im, -shift_x, axis=1)
        new_im = np.roll(new_im, -shift_y, axis=0)
        shifted_images.append(new_im)
    return shifted_images


"""
def _frame_selection_qa(my_psf, correlation_vals, other_info, nod):
    fig1, axarr = plt.subplots(2, 2, sharex=False, sharey=False, figsize=(10, 10))
    ax = axarr[1, 0]
    bx = axarr[0, 0]
    cx = axarr[1, 1]
    axarr[0, 1].axis("off")
    ax.imshow(my_psf, origin="lower", norm=PowerNorm(0.5), cmap="Greys")
    slc = np.nanmax(my_psf, 0)
    bx.plot(slc)
    slc = np.nanmax(my_psf, 1)
    cx.plot(slc, range(len(slc)))
    plt.suptitle(nod)
    plt.show()

    fig = plt.figure()
    plt.plot(correlation_vals)
    plt.show()

    fig, (ax, bx) = plt.subplots(1, 2)
    ax.hist(other_info["shiftsx"])
    bx.hist(other_info["shiftsy"])
    plt.show()

    return fig1
"""


def _frame_selection_qa_many(axarr, my_psf, correlation_vals, other_info, nod, xoffset):
    # examine how the xoffset effects the resulting selected frames and psf
    ax = axarr[1, 0]
    bx = axarr[0, 0]
    cx = axarr[1, 1]
    axarr[0, 1].axis("off")
    if xoffset == 0:
        ax.imshow(my_psf, origin="lower", norm=PowerNorm(0.5), cmap="Greys")
    slc = np.nanmax(my_psf, 0)
    color = "firebrick"
    lw = 1
    alpha = 0.5
    zorder = 0
    if xoffset == 0:
        color = "k"
        lw = 2
        alpha = 1
        zorder = 1
    bx.plot(np.roll(slc, -xoffset), color=color, alpha=alpha, lw=lw, zorder=zorder)
    slc = np.nanmax(my_psf, 1)
    cx.plot(slc, range(len(slc)), color=color, alpha=alpha, lw=lw, zorder=zorder)
    plt.suptitle(nod)
    # plt.show()


# # Frame selection
# ## 1. Make the ideal psf estimate
# use an ideal psf as the frame selection criterion
# model 8.4m psf


def add_circle(im, xc, yc, r):
    new_im = np.copy(im)
    for x in range(im.shape[0]):
        for y in range(im.shape[1]):
            dist = np.sqrt(np.square(x - xc) + np.square(y - yc))
            if dist <= r + 0.5:
                new_im[y, x] = 1
    return new_im


def _mk_model_psf():
    # make a model of the fringe pattern
    # TODO: rescale based on the detector
    size = 120
    model = np.zeros((size, size))  # (extraction_size,extraction_size))
    xv, yv = np.meshgrid(np.arange(-size / 2, size / 2), np.arange(-size / 2, size / 2))

    model = add_circle(model, size // 2 - 9, size // 2, 6.5)
    model = add_circle(model, size // 2 + 9, size // 2, 6.5)

    min_scale = (
        1.22 * 8.6e-6 / (8.4) * 206265 / (2 * np.sqrt(2 * np.log(2)))
    )  # sigma, in arcsec

    fft = np.fft.fft2(model)
    fft = np.fft.fftshift(fft)

    # move the psf to the source location
    psf = np.abs(fft)

    model_psf = psf
    model_psf *= gauss(xv, yv, 0, 0, 6.5 * 3, 6.5 * 3, 0, 1)

    return model_psf


#
# Match to the ideal psf
#
# This also shifts all frames such that the peak flux is at `rough_loc` -- useful to aligning multiple nod pairs for later analysis
#
# **Note:** this is a slow process due to many fourier transforms
def _frame_centering_and_selection(
    nod,
    bg_subtracted_frames,
    centroid_positions,
    rotations,
    output_dir,
    target,
    do_save=True,
    cutoff_fraction=0.9,
    binning=[1, 1],
    usepsf=None,
    do_offset=True,
    custom_window_size=48,
):
    """
    the window size can be set
    this allows 1. for a speedup and 2. for focus on a specific region of the image
    window size should not go smaller than 30
    """
    frame_idx = 0
    use_full_window = False

    # for using the full image
    window_size = usepsf.shape[0]
    cutout_loc = [window_size // 2, window_size // 2]

    # this is for using a small window
    if not use_full_window:
        cutout_loc = np.copy(centroid_positions[nod])
        cutout_loc[0] = cutout_loc[0]
        window_size = custom_window_size

    ###############################################################################
    ################### nominal psf selection #####################################
    ###############################################################################

    # use a real frame
    good_frame = bg_subtracted_frames[nod][frame_idx]
    nominal_psf = good_frame[
        cutout_loc[1] - window_size // 2 : cutout_loc[1] + window_size // 2,
        cutout_loc[0] - window_size // 2 : cutout_loc[0] + window_size // 2,
    ]

    w = usepsf.shape[0]
    temp_ideal_psf = np.copy(
        usepsf[
            w // 2 - window_size // 2 : w // 2 + window_size // 2,
            w // 2 - window_size // 2 : w // 2 + window_size // 2,
        ]
    )

    if binning != [1, 1]:
        nominal_psf = zoom(
            nominal_psf,
            (1 / binning[1], 1 / binning[0]),
            prefilter=False,
            mode="grid-constant",
            grid_mode=True,
        )
        temp_ideal_psf = zoom(
            temp_ideal_psf,
            (1 / binning[1], 1 / binning[0]),
            prefilter=False,
            mode="grid-constant",
            grid_mode=True,
        )
    # preview
    _, (ax, bx, cx) = plt.subplots(1, 3, figsize=(8.5, 4))
    ax.imshow(
        bg_subtracted_frames[nod][frame_idx],
        origin="lower",
        cmap="Greys",
        norm=PowerNorm(0.5),
    )
    ax.scatter(cutout_loc[0], cutout_loc[1], marker="x", color="r")
    bx.set_title("Image frame region to be used")
    if binning != [1, 1]:
        bx.set_title(f"Image frame region to be used\n(binned by {binning})")
    bx.imshow(nominal_psf, origin="lower", cmap="Greys")

    cx.set_title("Ideal PSF\nAt same binning")

    cx.imshow(temp_ideal_psf, origin="lower", cmap="Greys")
    plt.tight_layout()
    plt.savefig(f"{output_dir}/plots/{PROCESS_NAME}/{target}_nod{nod}_extraction.png")
    plt.close()

    # or use the model psf
    if usepsf is not None:
        w = usepsf.shape[0]
        nominal_psf = np.copy(
            usepsf[
                w // 2 - window_size // 2 : w // 2 + window_size // 2,
                w // 2 - window_size // 2 : w // 2 + window_size // 2,
            ]
        )
        if binning != [1, 1]:
            nominal_psf = zoom(
                nominal_psf,
                (1 / binning[1], 1 / binning[0]),
                prefilter=False,
                mode="grid-constant",
                grid_mode=True,
            )
    else:
        logger.info(PROCESS_NAME, "Please supply a psf!!!")
        return

    nominal_psf /= np.sum(nominal_psf)
    nominal_psf -= np.mean(nominal_psf)
    nominal_psf /= np.max(nominal_psf)

    ###############################################################################
    ################### run the cross correlation #################################
    ###############################################################################
    results = []
    recovered_psfs = []
    _, axarr = plt.subplots(2, 2, sharex=False, sharey=False, figsize=(10, 10))
    for x_offset in [-3, -2, -1, 0, 1, 2, 3]:
        if not do_offset:
            x_offset = 0
        # given a nominal psf and an array of images, compute the cross-correlation values
        # after centering using cross-correlation, sort by the chi-squared comparison to the psf
        # TODO: use the Fourier phase information somehow? Or is that a different step (after centering/sorting)
        if binning != [1, 1]:
            correlation_vals, corrected_ims, other_info = _frame_selection_scores_cc(
                [
                    zoom(
                        x[
                            cutout_loc[1] - window_size // 2 : cutout_loc[1]
                            + window_size // 2,
                            cutout_loc[0] - window_size // 2 + x_offset : cutout_loc[0]
                            + x_offset
                            + window_size // 2,
                        ],
                        (1 / binning[1], 1 / binning[0]),
                        prefilter=False,
                        mode="grid-constant",
                        grid_mode=True,
                    )
                    for x in bg_subtracted_frames[nod]
                ],
                nominal_psf,
                keep_fraction=cutoff_fraction,
            )
            results.append([correlation_vals, corrected_ims, other_info])
        else:
            correlation_vals, corrected_ims, other_info = _frame_selection_scores_cc(
                [
                    x[
                        cutout_loc[1] - window_size // 2 : cutout_loc[1]
                        + window_size // 2,
                        cutout_loc[0] - window_size // 2 + x_offset : cutout_loc[0]
                        + x_offset
                        + window_size // 2,
                    ]
                    for x in bg_subtracted_frames[nod]
                ],
                nominal_psf,
                keep_fraction=cutoff_fraction,
            )
            results.append([correlation_vals, corrected_ims, other_info])

        mask = other_info["mask"]
        im_subset = np.array(shift_images(bg_subtracted_frames[nod], other_info))[~mask]
        # im_subset = np.array(bg_subtracted_frames[nod])[mask]
        recovered_psf = np.mean(im_subset, 0)
        recovered_psfs.append(recovered_psf)

        # plot the effects of shifting

        _frame_selection_qa_many(
            axarr, recovered_psf, correlation_vals, other_info, nod, x_offset
        )
        if not do_offset:
            break

    plt.savefig(f"{output_dir}/plots/{PROCESS_NAME}/{target}_nod{nod}_result.png")
    plt.close()

    # recover the nominal value
    if do_offset:
        recovered_psf = recovered_psfs[3]
        correlation_vals, corrected_ims, other_info = results[3]
    else:
        recovered_psf = recovered_psfs[0]
        correlation_vals, corrected_ims, other_info = results[0]

    # save the information
    if do_save:
        np.save(
            f"{output_dir}/intermediate/{PROCESS_NAME}/{target}_fs_imstack_cycle{nod}.npy",
            recovered_psf,
        )
        logger.info(
            PROCESS_NAME,
            f"Offset info for cycle {nod} saved to {output_dir}/intermediate/{PROCESS_NAME}/{target}_fs_imstack_cycle{nod}.npy",
        )
        with open(
            f"{output_dir}/intermediate/{PROCESS_NAME}/{target}_fs_info_cycle{nod}.pk",
            "wb",
        ) as handle:
            info_dict = {k: v for k, v in other_info.items()}
            info_dict["mean_pa"] = rotations[nod]
            info_dict["correlation_vals"] = np.copy(correlation_vals)
            info_dict["corrected_ims"] = np.copy(corrected_ims)
            pickle.dump(info_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

    return recovered_psf, correlation_vals, corrected_ims, other_info


def do_frame_selection(config: dict, mylogger: Logger) -> bool:
    global logger
    logger = mylogger

    try:
        target = config["target"]
        nod_info = config["nod_info"]
        instrument = config["instrument"]
        targ_type = config["targ_type"]
        obsdate = config["obsdate"]
        output_dir = config["output_dir"]
        skips = config["skips"]
        CUTOFF_FRACTION = config["cutoff_fraction"]
        psfname = config["psfname"]
    except KeyError as e:
        logger.error(PROCESS_NAME, f"Key missing from config: {e}")
        return False

    # load the psf
    if psfname != "model":
        try:
            empirical_psf = np.load(psfname)[0]
            print(empirical_psf.shape)
        except FileNotFoundError:
            logger.error(PROCESS_NAME, f"PSF {psfname} not found. Exiting...")
            return False

    centroid_positions = {}
    bg_subtracted_frames = {}  # key: window_background_subtraction(ims[key], backgrounds[nod_info[key]["subtract"]]["mean"], nod_info[key]["position"]) for key in ims.keys()}
    rotations = {}
    for name, _ in nod_info.items():
        if name in skips:
            continue
        cent = np.load(
            f"{output_dir}/intermediate/bkg_subtraction/{target}_centroid-positions_cycle{name}.npy"
        )
        bkgsubtracted_ims = np.load(
            f"{output_dir}/intermediate/bkg_subtraction/{target}_bkg-subtracted_cycle{name}.npy"
        )
        rotations[name] = np.load(
            f"{output_dir}/intermediate/bkg_subtraction/{target}_rotations_cycle{name}.npy"
        )
        bg_subtracted_frames[name] = bkgsubtracted_ims
        centroid_positions[name] = cent

    # do all frames for actual processing
    for key in bg_subtracted_frames:
        logger.info(PROCESS_NAME, f"Processing nod {key}...")
        mypsf = _mk_model_psf()
        if psfname != "model":
            mypsf = empirical_psf
        _frame_centering_and_selection(
            key,
            bg_subtracted_frames,
            centroid_positions,
            rotations,
            f"{output_dir}",
            target,
            do_save=True,
            cutoff_fraction=CUTOFF_FRACTION,
            usepsf=mypsf,
            do_offset=False,
        )
        logger.info(PROCESS_NAME, "#" * 128)

    return True


if __name__ == "__main__":
    configfilename = "./nod_config_ngc4151.json"
    do_frame_selection(configfilename)
