"""
do_flux_calibration -- LIZARD Pipeline
Author: Jacob Isbell

Functions to use a calibrator to calculate the flux of the science target. Assumes that the calibrator is completely unresolved.
Called by lizard_calibrate
"""

import numpy as np
import pickle
from glob import glob
import matplotlib.pyplot as plt
from matplotlib.colors import PowerNorm

from utils.util_logger import Logger
from reduction_steps.do_image_corotation import load_bkg_subtracted_files


PROCESS_NAME = "flux_calibration"
logger = Logger("./")


def _load_images_and_compute_stats(
    path_dict: dict, skips: list, bkg_subtracted_frames: list
):
    all_kept_sums = []
    all_kept_peaks = []
    # TODO: modify to give per-pixel uncertainty????
    for key, path in path_dict.items():
        print(key, path)
        if key in skips:
            continue
        with open(
            path,
            "rb",
        ) as handle:
            info = pickle.load(handle)
            # images = bkg_subtracted_frames[key]

            mask = info["mask"]
            cims = info["corrected_ims"]
            # full_ims = images[mask]

            kept_ims = cims[mask]

            kept_ims = np.array([x - np.mean(x[:5, :5]) for x in kept_ims])
            kept_ims = np.array([x - np.min(x) for x in kept_ims])

            # fig = plt.figure()
            # plt.imshow(np.mean(kept_ims, 0), origin="lower")
            #
            # fig2 = plt.figure()
            # plt.imshow(np.std(kept_ims, 0), origin="lower")
            # plt.show()
            # plt.close("all")

            # all_kept_sums.append([np.sum(x) for x in kept_ims])
            for ki in kept_ims:
                all_kept_sums.append(np.sum(ki))
                all_kept_peaks.append(np.max(ki))
            # for fi in full_ims:
            #     all_images.append(fi)

    all_kept_sums = np.array(all_kept_sums).flatten()
    all_kept_peaks = np.array(all_kept_peaks).flatten()

    print(f"Using sums: {np.percentile(all_kept_sums, [50 - 34, 50, 50 + 34])}")
    print(f"Using peaks: {np.percentile(all_kept_peaks, [50 - 34, 50, 50 + 34])}")

    print(
        f"Relative errs: {np.std(all_kept_sums)/np.mean(all_kept_sums)} vs {np.std(all_kept_peaks)/np.mean(all_kept_peaks)}"
    )

    return {
        "mean": np.mean(all_kept_sums),
        "percentiles": np.percentile(all_kept_sums, [50 - 34, 50, 50 + 34]),
        "std": np.std(all_kept_sums),
    }


def _compute_ratios(target_stats, calib_stats):
    """
    Docstring
    """
    # mean_ratio = target_stats["mean"] / calib_stats["mean"]
    median_ratio = target_stats["percentiles"][1] / calib_stats["percentiles"][1]
    ratio_upperlim = target_stats["percentiles"][2] / calib_stats["percentiles"][0]
    ratio_lowerlim = target_stats["percentiles"][0] / calib_stats["percentiles"][2]

    uneven_relerr = [
        abs(median_ratio - ratio_lowerlim) / median_ratio,
        abs(ratio_upperlim - median_ratio) / median_ratio,
    ]

    return median_ratio, uneven_relerr


def _make_plot(lower, median, upper):
    fig, (ax, bx) = plt.subplots(1, 2)
    cbar1 = ax.imshow(median, origin="lower", norm=PowerNorm(0.5, vmin=0))
    std = np.mean([median - lower, upper - median], 0)
    cbar2 = bx.imshow(std, origin="lower")
    plt.colorbar(cbar1, ax=ax, scale=0.7, label="Flux [mJy/px]")
    plt.colorbar(cbar2, ax=bx, scale=0.7, label="Error [mJy/px]")
    plt.show()
    plt.close()


def do_flux_calibration(
    configdata: dict, target_configdata: dict, calib_configdata: dict, mylogger: Logger
) -> bool:
    """
    Docstring
    """
    global logger
    logger = mylogger

    # 0. Parse the config files
    try:
        # calibration properties
        output_dir = configdata["output_dir"]
        calib_flux = float(configdata["flux_cal"]["calib_flux_Jy"])
        calib_flux_err = float(configdata["flux_cal"]["calib_flux_err_Jy"])

        # calib properties
        calib_output_dir = calib_configdata["output_dir"]
        calib_name = calib_configdata["target"]
        calib_skips = calib_configdata["skips"]
        calib_nod_info = calib_configdata["nod_info"]

        # science target properties
        target_output_dir = target_configdata["output_dir"]
        target_name = target_configdata["target"]
        target_skips = target_configdata["skips"]
        target_nod_info = target_configdata["nod_info"]
    except KeyError as e:
        logger.error(
            PROCESS_NAME,
            f"One or more config keys was incorrect in do_flux_calibration: {e}",
        )
        return False

    # 1. Load the flux calibrator images and compute percentiles of the sum
    temp = glob(
        f"{calib_output_dir}/intermediate/frame_selection/{calib_name}*info*.pk"
    )
    calib_files = {(f.split(".pk")[0].split("_")[-1])[5:]: f for f in temp}

    calib_background_subtracted_frames, centroid_positions, all_rotations = (
        load_bkg_subtracted_files(
            calib_nod_info, output_dir, calib_name, skips=calib_skips
        )
    )

    logger.info(PROCESS_NAME, f"Loading the calibrator ({calib_name}) files...")
    try:
        calib_stats = _load_images_and_compute_stats(
            calib_files, calib_skips, calib_background_subtracted_frames
        )
    except FileNotFoundError as e:
        logger.error(
            PROCESS_NAME,
            f"One or more files not found for {calib_name} in the list {calib_files.values()}\n{e}",
        )
        return False
    except KeyError as e:
        logger.error(PROCESS_NAME, f"One or more keys not found for {calib_name}: {e}")
        return False

    # 2. Load the science target images and compute percentiles of the sum
    temp = glob(
        f"{target_output_dir}/intermediate/frame_selection/{target_name}*info*.pk"
    )
    target_files = {(f.split(".pk")[0].split("_")[-1])[5:]: f for f in temp}
    print(target_files)

    target_background_subtracted_frames, centroid_positions, all_rotations = (
        load_bkg_subtracted_files(
            target_nod_info, output_dir, target_name, skips=target_skips
        )
    )

    logger.info(PROCESS_NAME, f"Loading the target ({target_name}) files...")
    try:
        target_stats = _load_images_and_compute_stats(
            target_files, target_skips, target_background_subtracted_frames
        )
    except FileNotFoundError as e:
        logger.error(
            PROCESS_NAME,
            f"One or more files not found for {target_name} in the list {target_files.values()}\n{e}",
        )
        return False
    except KeyError as e:
        logger.error(PROCESS_NAME, f"One or more keys not found for {target_name}: {e}")
        return False

    # 3. Use the known calibrator flux and uncertainty to scale the target
    logger.info(PROCESS_NAME, "Computing image statistics ... ")
    ratio, relerr = _compute_ratios(target_stats, calib_stats)
    calib_flux_relerr = calib_flux_err / calib_flux

    final_relerr = np.array(
        [
            np.sqrt(relerr[0] ** 2 + calib_flux_relerr**2),
            np.sqrt(relerr[1] ** 2 + calib_flux_relerr**2),
        ]
    )

    final_flux = ratio * calib_flux
    final_flux_unc = calib_flux * final_relerr
    flux_range = np.array(
        [
            final_flux - final_flux_unc[0],
            final_flux,
            final_flux + final_flux_unc[1],
            final_relerr[0],
            final_relerr[1],
        ]
    )

    logger.info(
        PROCESS_NAME,
        f"The calibrated total flux is {final_flux:.2f}Jy -{final_flux_unc[0]:.2f} +{final_flux_unc[1]:.2f}",
    )

    # 4. Save the flux calibrated target images and the scaling information
    outname = f"{output_dir}/calibrated/{PROCESS_NAME}/sci_{target_name}_with_cal_{calib_name}_flux_percentiles.npy"
    np.save(
        outname,
        flux_range,
    )
    logger.info(PROCESS_NAME, f"Saved flux percentiles to {outname}")

    return True
