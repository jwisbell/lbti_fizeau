"""
do_image_corotation -- LIZARD Pipeline
Author: Jacob Isbell

Functions to align all images of an observation such that north is up and that the target is at the center of the frame.
Called by lizard_reduce
"""

from calibration_steps import bad_pixel_correction
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import PowerNorm
import pickle
from scipy.ndimage import rotate, median_filter
from glob import glob
from utils.util_logger import Logger
from utils.utils import argmax2d

PROCESS_NAME = "corotate"


def load_bkg_subtracted_files(nod_info: dict, output_dir: str, target: str, skips=[]):
    """
    Docstring
    """

    bg_subtracted_frames = {}  # key: window_background_subtraction(ims[key], backgrounds[nod_info[key]["subtract"]]["mean"], nod_info[key]["position"]) for key in ims.keys()}
    centroid_positions = {}
    rotations = {}

    # TODO: delete this
    lsum = 0
    lcount = 0
    msum = 0
    mcount = 0

    for name, _ in nod_info.items():
        if name in skips:  # ['6','7','11']:
            continue
        if "bkg" in name or "off" in name:
            continue
        bkgsubtracted_ims = np.load(
            f"{output_dir}/intermediate/bkg_subtraction/{target}_bkg-subtracted_cycle{name}.npy"
        )
        bg_subtracted_frames[name] = bkgsubtracted_ims
        if "m" in name:
            msum += np.sum(bkgsubtracted_ims)
            mcount += len(bkgsubtracted_ims)
        else:
            lsum += np.sum(bkgsubtracted_ims)
            lcount += len(bkgsubtracted_ims)
        cent = np.load(
            f"{output_dir}/intermediate/bkg_subtraction/{target}_centroid-positions_cycle{name}.npy"
        )
        centroid_positions[name] = cent

        rotations[name] = np.load(
            f"{output_dir}/intermediate/bkg_subtraction/{target}_rotations_cycle{name}.npy"
        )

    """
    ratio = 2.227264280094873  # (msum / mcount) / (lsum / lcount)
    for name, v in bg_subtracted_frames.items():
        if "m" in name:
            v /= ratio
    """
    return bg_subtracted_frames, centroid_positions, rotations


def _plot_cycles(
    imdict: dict, stacked_im, unrotated_psf_percentiles, target: str, output_dir: str
):
    """
    Docstring
    """

    _, axarr = plt.subplots(len(imdict) // 6 + 1, 6)  # figsize=(8.5, 11))
    for ax in axarr.flatten():
        ax.axis("off")

    for k, key in enumerate(imdict.keys()):
        ims = imdict[key]["ims"]
        rotim, _, _ = recenter(np.mean(ims, 0))
        # rotim = np.mean(ims, 0)

        ax = axarr.flatten()[k]

        ax.imshow(rotim, origin="lower", norm=PowerNorm(0.5))
        # ax.scatter(rotim.shape[0] // 2, rotim.shape[1] // 2, marker="x", color="r")
        ax.set_title(f"Cycle {k+1}")
    plt.tight_layout()
    plt.savefig(f"{output_dir}/plots/{PROCESS_NAME}/{target}_all_cycles_rotated.png")
    plt.close()

    _, ((ax, bx), (cx, dx)) = plt.subplots(2, 2, figsize=(6, 6))
    ax.plot(np.max(stacked_im, 0))
    cx.imshow(stacked_im, origin="lower", norm=PowerNorm(0.5))
    dx.plot(np.max(stacked_im, 1), range(stacked_im.shape[0]))
    bx.axis("off")
    plt.savefig(
        f"{output_dir}/plots/{PROCESS_NAME}/{target}_all_cycles_stacked_psf.png"
    )
    plt.close()

    _, ((ax, bx), (cx, dx)) = plt.subplots(2, 2, figsize=(6, 6))

    w = stacked_im.shape[0]
    slcy = unrotated_psf_percentiles[0][:, w // 2]
    slcy_err = unrotated_psf_percentiles[1][:, w // 2]

    slcx = unrotated_psf_percentiles[0][w // 2, :]
    slcx_err = unrotated_psf_percentiles[1][w // 2, :]

    ax.errorbar(range(stacked_im.shape[1]), slcx, yerr=slcx_err)
    cx.imshow(unrotated_psf_percentiles[0], origin="lower", norm=PowerNorm(0.5))
    dx.errorbar(slcy, range(stacked_im.shape[0]), xerr=slcy_err)
    bx.axis("off")
    plt.savefig(
        f"{output_dir}/plots/{PROCESS_NAME}/{target}_all_cycles_median_unrotated_psf.png"
    )
    plt.close()


def _process_rotations(
    infos: dict,
    background_subtracted_frames: dict,
    centroid_positions: dict,
    all_rotations: dict,
    extraction_size: int,
    use_phase: bool = False,
):
    """
    Docstring
    """
    properly_rotated_ims = []
    w = extraction_size // 2
    proper_rotations = {}
    unrotated_ims = []
    for key, info in infos.items():
        logger.info(PROCESS_NAME, f"Processing key {key}")
        # cims = np.copy(info["corrected_ims"])
        cims = background_subtracted_frames[key]

        rotations = all_rotations[f"{key}"]
        # cc_vals = info["correlation_vals"]
        mask = info["mask"]
        if use_phase:
            mask = info["fourier"]["mask"]
        shiftsx = info["shiftsx"]
        shiftsy = info["shiftsy"]
        proper_rotations[f"{key}"] = {"ims": [], "rots": []}

        temp_imarr = []
        temp_rotarr = []
        temp_unrotarr = []

        for i, cim in enumerate(cims):
            if mask[i] == 1:
                # shift to center
                new_im = np.roll(cim, w - centroid_positions[f"{key}"][0], axis=1)
                new_im = np.roll(new_im, w - centroid_positions[f"{key}"][1], axis=0)

                new_im = np.roll(new_im, -shiftsx[i], axis=1)
                new_im = np.roll(new_im, -shiftsy[i], axis=0)

                # rotate to North
                pa = rotations[i]
                rotim = rotate(new_im, -pa, reshape=False, mode="nearest")
                temp_imarr.append(rotim)
                temp_rotarr.append(pa)
                temp_unrotarr.append(new_im)

                # unrotated_ims.append(new_im)
                # properly_rotated_ims.append(rotim)

        temp_imarr = np.array(temp_imarr)
        temp_rotarr = np.array(temp_rotarr)
        temp_unrotarr = np.array(temp_unrotarr)
        # temp_imarr = temp_imarr[mask]
        # temp_rotarr = temp_rotarr[mask]

        proper_rotations[f"{key}"]["ims"] = np.copy(temp_imarr)
        proper_rotations[f"{key}"]["rots"] = np.copy(temp_rotarr)
        proper_rotations[f"{key}"]["centered_unrot"] = np.copy(temp_unrotarr)
        del cims
    return proper_rotations


def recenter(im):
    x, y = argmax2d(median_filter(im, 3))

    new_im = np.roll(im, im.shape[1] // 2 - x, axis=1)
    new_im = np.roll(new_im, im.shape[0] // 2 - y, axis=0)

    return new_im, im.shape[1] // 2 - x, im.shape[0] // 2 - y


def do_image_corotation(config: dict, mylogger: Logger) -> bool:
    """
    Docstring!
    """
    global logger
    logger = mylogger

    # 0. Parse the config info
    try:
        nod_info = config["nod_info"]
        target = config["target"]
        output_dir = config["output_dir"]
        skips = config["skips"]
        extraction_size = config["sub_window"]
    except KeyError as e:
        logger.error(PROCESS_NAME, f"Key missing from config: {e}")
        return False

    try:
        use_phase = config["use_phase"]
    except KeyError:
        use_phase = False

    try:
        remove_bad_pixels = config["estimate_bad_pixels"]
    except KeyError:
        remove_bad_pixels = False
        logger.warn(
            PROCESS_NAME, "Bad pixel correction not set, assuming no correction."
        )

    # 1. Load the data
    # 1a - load the background subtracted frames
    background_subtracted_frames, centroid_positions, all_rotations = (
        load_bkg_subtracted_files(nod_info, output_dir, target, skips=skips)
    )

    # 1b - load the recentering info from prev step
    datadir = f"{output_dir}/intermediate/frame_selection/"
    num_files = len(glob(f"{datadir}/{target}*imstack*cycle*.npy"))

    info_files = np.sort(
        glob(f"{datadir}/{target}_fs_info_cycle*.pk")
    )  # [f"{datadir}/{target}_fs_info_cycle{i+1}.pk" for i in range(num_files)]

    infos = {}
    for i_f in info_files:
        try:
            with open(i_f, "rb") as handle:
                info = pickle.load(handle)
                cycle_num = i_f.split(".pk")[0].split("cycle")[-1]
                if cycle_num in skips:
                    continue
                infos[cycle_num] = info
        except FileNotFoundError:
            logger.warn(
                PROCESS_NAME,
                f"The file {i_f} was not found -- this nod position was likely skipped!",
            )

    # 2. for each individual image,
    # apply shifts
    # apply rotation
    rotation_dict = _process_rotations(
        infos,
        background_subtracted_frames,
        centroid_positions,
        all_rotations,
        extraction_size,
        use_phase=use_phase,
    )
    # print(len(properly_rotated_ims), len(properly_rotated_ims) / 58)
    # _ = input("continue?")

    # 3. Add all cycles together to get final observation PSF
    # compute the necessary images from the above rotation dict
    # 1. the summed, corotated image
    # 2. the summed, unrotated image
    sum_rotated = 0
    sum_unrotated = 0
    sum_std = 0
    count = 0
    all_rotated = []
    for _, entry in rotation_dict.items():
        # for some reason these need to be recentered again
        centered_rot, _, _ = recenter(np.sum(entry["ims"], 0))
        centered_unrot, x, y = recenter(np.sum(entry["centered_unrot"], 0))

        if remove_bad_pixels:
            centered_rot, _ = bad_pixel_correction.identify_bad_pixels(centered_rot)
            centered_unrot, _ = bad_pixel_correction.identify_bad_pixels(centered_unrot)

        sum_rotated += centered_rot
        sum_unrotated += centered_unrot
        sum_std += np.roll(
            np.roll(np.std(entry["centered_unrot"], 0), x, axis=1), y, axis=0
        )
        count += len(entry["ims"])
        for im in entry["ims"]:
            all_rotated.append(im)
    mean_rotated = sum_rotated / count
    mean_unrotated = sum_unrotated / count
    mean_std = sum_std / count

    # Stats for flux calibration
    temp = np.array([x - np.mean(x[:10, :10]) for x in all_rotated])

    # test = np.percentile(temp, [50 - 34, 50, 50 + 34], axis=0)
    # fig, (ax, bx, cx) = plt.subplots(1, 3)
    # ax.imshow(np.mean(test), origin="lower")
    # bx.imshow(np.mean([test[1] - test[0], test[2] - test[1]], 0), origin="lower")
    # cx.hist(np.array(temp)[:, 50, 50])
    # cx.hist(np.array(temp)[:, 10, 10])
    # plt.show()
    # plt.close()

    psf_unrotated_percentiles = np.array([mean_unrotated, mean_std])
    stacked_rotated_im, _, _ = recenter(mean_rotated)

    # also plot the individual/combined PSFs
    # stacked_rotated_im = np.mean(properly_rotated_ims, 0)
    _plot_cycles(
        rotation_dict,
        stacked_rotated_im,
        psf_unrotated_percentiles,
        target,
        output_dir,
    )

    # 4. Save to disk

    # this is multiple GB, default to not saving
    # df = pd.DataFrame.from_dict(proper_rotations)
    # df.to_pickle(f"{output_dir}/intermediate/{PROCESS_NAME}/{target}_rotated_ims.pkl")

    print([len(rotation_dict[nod]["rots"]) for nod in rotation_dict])

    kept_rots = []
    for _, entry in rotation_dict.items():
        for rot in entry["rots"]:
            kept_rots.append(rot)
    # kept_rots = np.array(
    # [rotation_dict[nod]["rots"] for nod in rotation_dict]
    # ).flatten()
    kept_rots = np.array(kept_rots)

    unrotated_per_nod = {
        nod: np.mean(rotation_dict[nod]["centered_unrot"], 0) for nod in rotation_dict
    }
    with open(
        f"{output_dir}/intermediate/{PROCESS_NAME}/{target}_unrotated_cycle_stacks.pkl",
        "wb",
    ) as pkl:
        pickle.dump(unrotated_per_nod, pkl, protocol=pickle.HIGHEST_PROTOCOL)

    rotated_per_nod = {
        nod: np.mean(rotation_dict[nod]["ims"], 0) for nod in rotation_dict
    }
    with open(
        f"{output_dir}/intermediate/{PROCESS_NAME}/{target}_rotated_cycle_stacks.pkl",
        "wb",
    ) as pkl:
        pickle.dump(rotated_per_nod, pkl, protocol=pickle.HIGHEST_PROTOCOL)

    rotations_per_nod = {
        nod: np.mean(rotation_dict[nod]["rots"]) for nod in rotation_dict
    }

    with open(
        f"{output_dir}/intermediate/{PROCESS_NAME}/{target}_rotations_per_cycle.pkl",
        "wb",
    ) as pkl:
        pickle.dump(rotations_per_nod, pkl, protocol=pickle.HIGHEST_PROTOCOL)

    np.save(
        f"{output_dir}/intermediate/{PROCESS_NAME}/{target}_included_rotations.npy",
        kept_rots,
    )

    np.save(
        f"{output_dir}/intermediate/{PROCESS_NAME}/{target}_corotated_stacked_im.npy",
        stacked_rotated_im,
    )

    np.save(
        f"{output_dir}/intermediate/{PROCESS_NAME}/{target}_unrotated_stacked_psf.npy",
        psf_unrotated_percentiles,
    )
    return True


def do_image_corotation_sd(config: dict, mylogger: Logger) -> bool:
    """
    Docstring for single dish image rotation
    """
    global logger
    logger = mylogger

    # 0. Parse the config info
    try:
        nod_info = config["nod_info"]
        target = config["target"]
        output_dir = config["output_dir"]
        skips = config["skips"]
        extraction_size = config["sub_window"]
    except KeyError as e:
        logger.error(PROCESS_NAME, f"Key missing from config: {e}")
        return False

    # 1. Load the data
    # 1a - load the background subtracted frames
    background_subtracted_frames, centroid_positions, all_rotations = (
        load_bkg_subtracted_files(nod_info, output_dir, target, skips=skips)
    )

    # TODO: replace this with new recentering step
    # 1b - load the recentering info from prev step
    datadir = f"{output_dir}/intermediate/bkg_subtraction/"
    num_files = len(glob(f"{datadir}/{target}*subtracted*cycle*.npy"))

    infos = {}
    for k, entry in background_subtracted_frames.items():
        if "off" in k or "bkg" in k:
            continue
        infos[k] = {}
        infos[k]["mask"] = np.array([True for _ in entry])
        infos[k]["shiftsx"] = np.array([0 for _ in entry])
        infos[k]["shiftsy"] = np.array([0 for _ in entry])

    # 2. for each individual image,
    # apply shifts
    # apply rotation
    rotation_dict = _process_rotations(
        infos,
        background_subtracted_frames,
        centroid_positions,
        all_rotations,
        extraction_size,
    )
    # print(len(properly_rotated_ims), len(properly_rotated_ims) / 58)
    # _ = input("continue?")

    # 3. Add all cycles together to get final observation PSF
    # compute the necessary images from the above rotation dict
    # 1. the summed, corotated image
    # 2. the summed, unrotated image
    sum_rotated = 0
    sum_unrotated = 0
    sum_std = 0
    count = 0
    all_rotated = []
    for _, entry in rotation_dict.items():
        # for some reason these need to be recentered again
        centered_rot, _, _ = recenter(np.sum(entry["ims"], 0))
        centered_unrot, x, y = recenter(np.sum(entry["centered_unrot"], 0))
        sum_rotated += centered_rot
        sum_unrotated += centered_unrot
        sum_std += np.roll(
            np.roll(np.std(entry["centered_unrot"], 0), x, axis=1), y, axis=0
        )
        count += len(entry["ims"])
        for im in entry["ims"]:
            all_rotated.append(im)
    mean_rotated = sum_rotated / count
    mean_unrotated = sum_unrotated / count
    mean_std = sum_std / count

    # Stats for flux calibration
    mean_flux, test_std = np.mean(all_rotated, 0), np.std(all_rotated, 0)
    fig, (ax, bx) = plt.subplots(1, 2)
    ax.imshow(mean_flux, origin="lower")
    bx.imshow(test_std, origin="lower")
    # plt.show()
    plt.close()

    psf_unrotated_percentiles = np.array([mean_unrotated, mean_std])
    stacked_rotated_im = mean_rotated

    # also plot the individual/combined PSFs
    # stacked_rotated_im = np.mean(properly_rotated_ims, 0)
    _plot_cycles(
        rotation_dict,
        stacked_rotated_im,
        psf_unrotated_percentiles,
        target,
        output_dir,
    )

    # 4. Save to disk

    # this is multiple GB, default to not saving
    # df = pd.DataFrame.from_dict(proper_rotations)
    # df.to_pickle(f"{output_dir}/intermediate/{PROCESS_NAME}/{target}_rotated_ims.pkl")

    print([len(rotation_dict[nod]["rots"]) for nod in rotation_dict])

    kept_rots = []
    for _, entry in rotation_dict.items():
        for rot in entry["rots"]:
            kept_rots.append(rot)
    # kept_rots = np.array(
    # [rotation_dict[nod]["rots"] for nod in rotation_dict]
    # ).flatten()
    kept_rots = np.array(kept_rots)

    unrotated_per_nod = {
        nod: np.mean(rotation_dict[nod]["centered_unrot"], 0) for nod in rotation_dict
    }
    with open(
        f"{output_dir}/intermediate/{PROCESS_NAME}/{target}_unrotated_cycle_stacks.pkl",
        "wb",
    ) as pkl:
        pickle.dump(unrotated_per_nod, pkl, protocol=pickle.HIGHEST_PROTOCOL)

    np.save(
        f"{output_dir}/intermediate/{PROCESS_NAME}/{target}_included_rotations.npy",
        kept_rots,
    )

    np.save(
        f"{output_dir}/intermediate/{PROCESS_NAME}/{target}_corotated_stacked_im.npy",
        stacked_rotated_im,
    )

    np.save(
        f"{output_dir}/intermediate/{PROCESS_NAME}/{target}_unrotated_stacked_psf.npy",
        psf_unrotated_percentiles,
    )
    return True
