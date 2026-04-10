"""
do_bkg_subtraction -- LIZARD Pipeline
Author: Jacob Isbell

Functions to load the raw data files and then do background subtraction using the user-specified nod pairs.
This is the initial data reduction step, and is the most likely to fail if the config file is incorrectly prepared.

Called by lizard_reduce
"""

from multiprocessing import Pool
from itertools import repeat
import matplotlib.pyplot as plt
from astropy.io import fits
import numpy as np
from matplotlib import animation
from matplotlib.colors import PowerNorm
from scipy.ndimage import median_filter
from datetime import datetime
from utils.util_logger import Logger
import time
import polars as pl
import pickle
import os
from astropy.stats import sigma_clip
from utils.utils import create_filestructure
from typing import Tuple, Dict
import warnings

from calibration_steps.bad_pixel_correction import (
    correct_image_after_bpm,
    load_bpm,
    apply_bad_pixel_mask,
)

PROCESS_NAME = "bkg_subtraction"
extraction_size = 100
instrument = "NOMIC"
logger = None


def _extract_window_v2(im, center, size):
    xc, yc = center
    if size >= im.shape[0] or size <= 0:
        return im

    ylower = np.max([0, yc - size // 2])
    yupper = np.min([im.shape[0], yc + size // 2])
    xlower = np.max([0, xc - size // 2])
    xupper = np.min([im.shape[1], xc + size // 2])

    if size % 2 == 1:
        yupper += 1
        xupper += 1

    return im[ylower:yupper, xlower:xupper]


def _merge_headers_to_df(hdr_dicts, nod_name):
    # dfs = [polars.from_dict(h) for h in hdr_dicts]
    df = pl.from_dicts(hdr_dicts, strict=False)
    df = df.with_columns(pl.lit(nod_name).alias("nod_name"))
    return df


def _load_darks(filenames):
    mean_dark = 0.0
    for fn in filenames:
        hdu = fits.open(fn)
        im = hdu[0].data
        if len(hdu[0].data.shape) > 2:
            im = hdu[0].data[-1]
        mean_dark += im
    return mean_dark / len(filenames)


def _fast_fit_slope_np(imgcube):
    # Mask NaN and Inf values
    imgcube = np.ma.masked_invalid(imgcube)

    n_time, n_rows, n_cols = imgcube.shape

    # Create time vector (starting from 1)
    time = np.arange(n_time, dtype=np.float64) + 1

    # Build design matrix [time, 1] for slope + intercept
    A = np.column_stack([time, np.ones(n_time)])  # shape (n_time, 2)

    # Reshape imgcube from (n_time, n_rows, n_cols) → (n_time, n_rows*n_cols)
    b = imgcube.reshape(n_time, -1).astype(np.float64)  # shape (n_time, n_pixels)

    # Replace masked values with NaN so lstsq doesn't use them in the fill
    b = np.ma.filled(b, np.nan)

    # Find columns (pixels) with any NaN
    nan_mask = np.any(np.isnan(b), axis=0)  # shape (n_pixels,)
    valid_cols = ~nan_mask

    # Solve only on valid (non-NaN) pixels
    beta_flat = np.full(n_rows * n_cols, np.nan)
    alpha_flat = np.full(n_rows * n_cols, np.nan)

    if valid_cols.any():
        x, _, _, _ = np.linalg.lstsq(A, b[:, valid_cols], rcond=None)
        # x shape: (2, n_valid_pixels) — row 0 = slopes, row 1 = intercepts
        beta_flat[valid_cols] = x[0]
        alpha_flat[valid_cols] = x[1]

    # Reshape back to (n_rows, n_cols)
    beta = beta_flat.reshape(n_rows, n_cols)
    alpha = alpha_flat.reshape(n_rows, n_cols)

    return beta, alpha


def _fix_gain(image):
    # for each channel (spaced by XX pixels), find the median value of the top 100 and bottom 100 pixels and subtract out
    new_image = np.copy(image)
    channel_width = 64
    channel_region = 100
    skip_pixels = 5
    for xstart in np.arange(0, image.shape[0] + channel_width, channel_width):
        region_top = image[skip_pixels:channel_region, xstart : xstart + channel_width]
        region_bot = image[
            -channel_region:-skip_pixels, xstart : xstart + channel_width
        ]
        channel_val = np.nanmax(
            [
                np.nanmedian(sigma_clip(region_top)),
                np.nanmedian(sigma_clip(region_bot)),
            ]
        )
        new_image[:, xstart : xstart + channel_width] -= channel_val * 0

    return new_image


def _ramp_fitting(im_arr, spacing=1, do_plot=False, full_fit=False):
    # uses the exposure ramp to suppress read noise
    s = im_arr[0].shape[0]

    # no bias subtraction
    delta_ims = np.array([im - im_arr[0] for im in im_arr[::spacing]]).astype("float")

    # sigma clip each image
    for i in range(len(delta_ims)):
        delta_ims[i] = sigma_clip(delta_ims[i])

    std_vals = [np.nanstd(im) for im in delta_ims]
    try:
        std_vals[0] = std_vals[1]
    except IndexError:
        pass

    alpha = np.zeros(im_arr[0].shape)
    # beta, alpha = _fast_fit_slope(delta_ims)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        warnings.simplefilter("ignore", UserWarning)
        beta, alpha = _fast_fit_slope_np(delta_ims)

    if do_plot:
        fig, ax = plt.subplots()
        slope1 = beta[s // 2 + 1, s // 2 + 1]
        alpha1 = alpha[s // 2 + 1, s // 2 + 1] * 0
        upperleft = (512, 1550)
        slope2 = beta[upperleft[1], upperleft[0]]
        alpha2 = alpha[upperleft[1], upperleft[0]] * 0
        xvals = np.arange(len(im_arr))
        ax.plot(xvals, slope1 * xvals + alpha1)
        ax.errorbar(
            xvals,
            [di[s // 2 + 1, s // 2 + 1] for di in delta_ims],
            yerr=std_vals,
            ls="none",
            marker="s",
        )

        ax.plot(xvals, slope2 * xvals + alpha2)
        ax.errorbar(
            xvals,
            [di[upperleft[1], upperleft[0]] for di in delta_ims],
            yerr=0,
            ls="none",
            marker="s",
        )
        plt.tight_layout()

        fig2 = plt.figure()
        med = np.nanmedian(beta * len(im_arr))
        std = np.nanstd(beta * len(im_arr))
        plt.imshow(
            beta * len(im_arr) + alpha * 0,
            origin="lower",
            vmin=med - 3 * std,
            vmax=med + 3 * std,
            cmap="Spectral_r",
        )
        # TODO: save this
        plt.show()
        plt.close()

    # additionally subtract out the channel biases (due to gain)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        warnings.simplefilter("ignore", UserWarning)
        corrected_im = _fix_gain(beta * len(im_arr) + alpha * 0)
    return corrected_im


def _savefits(bkg_subbed_images, nod_key, headers, config, process_path):
    start_im = config["nod_info"][nod_key]["start"]
    end_im = config["nod_info"][nod_key]["end"]
    band = "lm"
    if config["instrument"] == "NOMIC":
        band = "n"
    # outname = f"lm_{date}_bkgsub_"
    output_dir = config["output_dir"]
    target = config["target"]
    obsdate = config["obsdate"]
    hdr = fits.Header()
    for key, value in headers[0].items():
        if key in ["SIMPLE", "BITPIX", "NAXIS", "NAXIS1", "NAXIS2", "NAXIS3", "EXTEND"]:
            continue
        hdr[key] = value
    hdu = fits.PrimaryHDU(data=bkg_subbed_images, header=hdr)
    hdul = fits.HDUList([hdu])
    print("Saving fits file")
    hdul.writeto(
        f"{output_dir}/{process_path}/{target}_bkgsub_nod{nod_key}_{band}_{obsdate}_{str(start_im).zfill(6)}-{str(end_im).zfill(6)}.fits",
        overwrite=True,
    )


def time_convert(date_string, time_string):
    year, month, day = date_string.split("-")
    hour, minute, second = time_string.split(":")
    try:
        return datetime_to_julian_date(
            int(year), int(month), int(day), int(hour), int(minute), int(float(second))
        )
    except ValueError:
        print("could not convert ", date_string, time_string)


def datetime_to_julian_date(year, month, day, hour, minute, second):
    """Converts a date and time to a Julian date."""
    dt = datetime(year, month, day, hour, minute, second)
    origin = datetime(1899, 12, 31, 12, 0, 0)
    time_delta = dt - origin
    return 2415020 + time_delta.total_seconds() / (24 * 60 * 60)


def _qa_plots(bg_subtracted_frames, centroid_positions, timestamps, output_dir, target):
    # ## (optional) Plot cycles to quickly assess quality
    # HTML(image_video(bg_subtracted_frames["1"][::2],"2").to_html5_video())
    for key in bg_subtracted_frames.keys():
        if "bkg" in key or "off" in key:
            continue
        _ = plt.figure()
        im = np.mean(bg_subtracted_frames[key][:], 0)
        plt.imshow(
            im,
            origin="lower",
            norm=PowerNorm(0.5, vmin=-0.05 * np.nanmax(im), vmax=0.90 * np.nanmax(im)),
            interpolation="gaussian",
        )
        plt.scatter(centroid_positions[key][0], centroid_positions[key][1])
        plt.title(key)
        plt.savefig(
            f"{output_dir}/plots/{PROCESS_NAME}/{target}_nod{key}_allframes.png"
        )
        plt.close()

    # HTML(image_video(bg_subtracted_frames["2"][::2],"2").to_html5_video())

    _ = plt.figure()
    plt.title("mean flux")
    for key in bg_subtracted_frames.keys():
        plt.plot(
            # [t for t in timestamps[key]],
            [np.nanmean(x) for x in bg_subtracted_frames[key]],
            label=key,
        )
    plt.legend()
    plt.savefig(f"{output_dir}/plots/{PROCESS_NAME}/{target}_allnods_meanflux.png")
    plt.close("all")


def _parse_config(config):
    # extract relevant info from config file
    target = config["target"]
    nod_info = config["nod_info"]
    instrument = config["instrument"]
    extraction_size = config["sub_window"]  # 100
    data_dir = config["data_dir"]
    obsdate = config["obsdate"]
    output_dir = config["output_dir"]
    skips = [str(x) for x in config["skips"]]
    batch_size = config["batch_size"]
    try:
        do_up_the_ramp = config["do_up_the_ramp"]
    except KeyError:
        do_up_the_ramp = False

    try:
        save_fits = config["save_fits"]
    except KeyError:
        save_fits = False

    try:
        ramp_params = config["ramp_params"]
    except KeyError:
        ramp_params = {"idx": -1, "subtract_min": True}
        logger.warn(
            PROCESS_NAME,
            f"No ramp params set, using default {ramp_params}",
        )

    try:
        skip_bpm = config["skip_bpm"]
    except KeyError:
        skip_bpm = False  # do bad pixel correction by default

    try:
        dark_file_range = config["dark_file_range"]
    except KeyError:
        dark_file_range = (0, 0)

    return (
        target,
        nod_info,
        instrument,
        extraction_size,
        data_dir,
        obsdate,
        output_dir,
        skips,
        batch_size,
        do_up_the_ramp,
        save_fits,
        ramp_params,
        skip_bpm,
        dark_file_range,
    )


def _get_filenames(data_dir, prefix, start_fn, end_fn, sub_start_fn, sub_end_fn):
    obj_files = [
        f"{data_dir}/{prefix}{str(i).zfill(6)}.fits"
        for i in range(start_fn, end_fn + 1)
    ]
    bkg_files = [
        f"{data_dir}/{prefix}{str(i).zfill(6)}.fits"
        for i in range(sub_start_fn, sub_end_fn + 1)
    ]

    # Check if we are dealing with fits or fits.gz
    test_fname = obj_files[np.random.randint(0, len(obj_files))]
    if os.path.exists(test_fname):
        print("Using uncompressed target files")
    else:
        print(
            "Using compressed target files (*fits.gz) -- NOTE: this takes longer to process"
        )
        obj_files = [
            f"{data_dir}/{prefix}{str(i).zfill(6)}.fits.gz"
            for i in range(start_fn, end_fn + 1)
        ]

    test_fname = bkg_files[np.random.randint(0, len(bkg_files))]
    if os.path.exists(test_fname):
        print("Using uncompressed background files")
    else:
        print(
            "Using compressed background files (*fits.gz) -- NOTE: this takes longer to process"
        )
        bkg_files = [
            f"{data_dir}/{prefix}{str(i).zfill(6)}.fits.gz"
            for i in range(sub_start_fn, sub_end_fn + 1)
        ]

    return obj_files, bkg_files


def _minimal_load_file(
    filename,
    ramp_params: dict = {"idx": -1, "subtract_min": False},
    do_up_the_ramp=False,
    mean_dark=0.0,
    mean_bkg=0.0,
    cutout_info={"size": -1, "pos": [0, 0]},
    skip_bpm=False,
    show_plot=False,
):
    try:
        with fits.open(filename) as x:
            im = np.copy(x[0].data) - mean_dark
            if len(x[0].data.shape) > 2:
                im = np.copy(x[0].data[ramp_params["idx"]])
                if instrument != "NOMIC":
                    im = np.copy(x[0].data[ramp_params["idx"]]) - mean_dark
                    # subtracting out the "zero" exposure to remove bad pixels
                    if ramp_params["subtract_min"]:
                        print("should get here!")
                        im -= x[0].data[0]
                    if do_up_the_ramp:
                        logger.info(PROCESS_NAME, f"\t\t Ramp fitting on {filename}")
                        im: np.ndarray = (
                            _ramp_fitting(x[0].data, do_plot=False) - mean_dark
                        )

            # temp.append(im)
            pa = float(x[0].header["LBT_PARA"])
            obstime = time_convert(x[0].header["date-obs"], x[0].header["time-obs"])
            hdr = {k: v for k, v in x[0].header.items()}

            # background subtraction (subtract 0 if background file)
            if type(mean_bkg) is type(0.0):
                return im, pa, obstime, hdr

            bkg_subbed = im - mean_bkg

            # apply bad pixel mask (ones if skip_bpm)
            bad_pixel_mask = load_bpm(hdr)
            if skip_bpm:
                bad_pixel_mask = np.ones(im.shape)

            masked = bkg_subbed * bad_pixel_mask

            # make the cutout
            cutout = _extract_window_v2(
                np.copy(masked), cutout_info["pos"], size=cutout_info["size"]
            )

            # correct bad pixels
            corrected = correct_image_after_bpm(cutout)
            if show_plot:
                fig, axarr = plt.subplots(2, 2)
                axarr[0, 0].imshow(im, origin="lower", norm=PowerNorm(0.5))
                axarr[0, 1].imshow(bkg_subbed, origin="lower", norm=PowerNorm(0.5))
                axarr[1, 0].imshow(masked, origin="lower", norm=PowerNorm(0.5))
                axarr[1, 1].imshow(corrected, origin="lower", norm=PowerNorm(0.5))
                plt.show()
                plt.close()

            return corrected, pa, obstime, hdr
    except FileNotFoundError as e:
        logger.warn(PROCESS_NAME, f"\t\t {filename} failed, {e}")
    except OSError as e:
        print(filename, e)


def _load_science_files(
    config,
    key,
    mean_bkg,
    ramp_params,
    fdir: str,
    prefix: str,
    mean_dark=0,
    do_up_the_ramp: bool = False,
    skip_bpm: bool = False,
):
    nod_info = config["nod_info"]

    logger.info(PROCESS_NAME, f"Loading nod {key}")
    filenames = [
        f"{fdir}{prefix}{str(i).zfill(6)}.fits"
        for i in range(nod_info[key]["start"], nod_info[key]["end"] + 1)
    ]
    logger.info(PROCESS_NAME, f"\t {len(filenames)} files")

    # Check if we are dealing with fits or fits.gz
    test_fname = filenames[np.random.randint(0, len(filenames))]
    if os.path.exists(test_fname):
        logger.info(PROCESS_NAME, "\t Using uncompressed target files")
    else:
        logger.info(
            PROCESS_NAME,
            "\t Using compressed target files (*fits.gz) -- NOTE: this takes longer to process",
        )
        filenames = [
            f"{fdir}{prefix}{str(i).zfill(6)}.fits.gz"
            for i in range(nod_info[key]["start"], nod_info[key]["end"] + 1)
        ]

    size = config["sub_window"]
    pos = nod_info[key]["position"]
    if type(pos) is type(""):
        pos = config["positions"][pos]

    with Pool() as pool:
        res = pool.starmap(
            _minimal_load_file,
            zip(
                filenames,
                repeat(ramp_params),
                repeat(do_up_the_ramp),
                repeat(mean_dark),
                repeat(mean_bkg),
                repeat({"size": size, "pos": pos}),
                repeat(skip_bpm),
            ),
        )
    images = np.array([x[0] for x in res if x is not None])
    rotations = np.array([x[1] for x in res if x is not None])
    obstime = np.array([x[2] for x in res if x is not None])
    headers = np.array([x[-1] for x in res if x is not None])
    return images, rotations, obstime, headers


def _load_background_files(
    config, key, fdir, prefix, ramp_params, do_up_the_ramp, mean_dark=0
):
    sum = 0.0
    n_ims = 0

    nod_info = config["nod_info"]

    logger.info(PROCESS_NAME, f"Loading nod {key} (as background)")
    filenames = [
        f"{fdir}{prefix}{str(i).zfill(6)}.fits"
        for i in range(nod_info[key]["start"], nod_info[key]["end"] + 1)
    ]
    logger.info(PROCESS_NAME, f"\t {len(filenames)} files")

    # Check if we are dealing with fits or fits.gz
    test_fname = filenames[np.random.randint(0, len(filenames))]
    if os.path.exists(test_fname):
        logger.info(PROCESS_NAME, "\t Using uncompressed target files")
    else:
        logger.info(
            PROCESS_NAME,
            "\t Using compressed target files (*fits.gz) -- NOTE: this takes longer to process",
        )
        filenames = [
            f"{fdir}{prefix}{str(i).zfill(6)}.fits.gz"
            for i in range(nod_info[key]["start"], nod_info[key]["end"] + 1)
        ]

    with Pool() as pool:
        res = pool.starmap(
            _minimal_load_file,
            zip(
                filenames,
                repeat(ramp_params),
                repeat(do_up_the_ramp),
                repeat(mean_dark),
                repeat(0.0),
                repeat({"size": -1, "pos": [0, 0]}),
                repeat(True),
            ),
        )
    images = np.array([x[0] for x in res if x is not None])

    return np.nanmedian(images, 0)


def new_improved_bkg_subtraction(config: dict, mylogger: Logger):
    # # Load the data
    global logger
    global extraction_size
    global instrument

    logger = mylogger

    (
        target,
        nod_info,
        instrument,
        extraction_size,
        data_dir,
        obsdate,
        output_dir,
        skips,
        batch_size,
        do_up_the_ramp,
        save_fits,
        ramp_params,
        skip_bpm,
        dark_file_range,
    ) = _parse_config(config)

    try:
        positions_dict = config["positions"]
    except KeyError as _:
        positions_dict = {}

    prefix = f"n_{obsdate}_"
    if instrument != "NOMIC":
        prefix = f"lm_{obsdate}_"

    process_path = f"intermediate/{PROCESS_NAME}/"

    # Load the (optional) darks for dark subtraction
    mean_dark = 0
    if dark_file_range != (0, 0):
        dark_files, _ = _get_filenames(
            data_dir, prefix, dark_file_range[0], dark_file_range[1], 0, 0
        )
        print(dark_files)
        mean_dark = _load_darks(dark_files)

    # Open the full BPM
    # bpm = load_bpm(None)

    centroid_positions = {}
    for key, value in nod_info.items():
        if "bkg" in key or "off" in key or "skip" in key or "bad" in key:
            continue

        bkg_key: str = value["subtract"]

        # load the background files and return the mean
        start = time.time()
        mean_bkg = _load_background_files(
            config, bkg_key, data_dir, prefix, ramp_params, do_up_the_ramp, mean_dark
        )
        end = time.time()
        print(
            f"Loading the background files (parallelized) took {end - start:.3f} seconds"
        )

        start = time.time()
        (
            bkg_subbed_images,
            rots,
            times,
            hdr_dicts,
        ) = _load_science_files(
            config,
            key,
            mean_bkg,
            ramp_params,
            data_dir,
            prefix,
            mean_dark=mean_dark,
            do_up_the_ramp=do_up_the_ramp,
            skip_bpm=skip_bpm,
        )
        end = time.time()
        print(
            f"Loading the science files (parallelized) took {end - start:.3f} seconds"
        )

        # do the stats and saving as before

        logger.info(PROCESS_NAME, f"Saving/plotting key {key}")

        im = np.sum(bkg_subbed_images, 0)
        im = median_filter(im, 5)
        centroid_positions[key] = [
            np.clip(np.argmax(np.nansum(im, 0)), 32, len(im) - 32),
            np.clip(np.argmax(np.nansum(im, 1)), 32, len(im) - 32),
        ]

        np.save(
            f"{output_dir}/{process_path}/{target}_centroid-positions_cycle{key}.npy",
            [np.argmax(np.nansum(im, 0)), np.argmax(np.nansum(im, 1))],
        )
        np.save(
            f"{output_dir}/{process_path}/{target}_rotations_cycle{key}.npy",
            np.array(rots),
        )
        # save the background-subtracted frames
        np.save(
            f"{output_dir}/{process_path}/{target}_bkg-subtracted_cycle{key}.npy",
            bkg_subbed_images,
        )

        np.save(
            f"{output_dir}/{process_path}/{target}_timestamps_cycle{key}.npy",
            np.array(times),
        )  # save the time stamps of the background-subtracted frames

        hdr_dicts = [
            {k: v for k, v in d.items() if not isinstance(v, np.ndarray)}
            for d in hdr_dicts
        ]
        polars_df = _merge_headers_to_df(hdr_dicts, key)
        # Save the DataFrame to a pickle file
        with open(
            f"{output_dir}/intermediate/headers/{target}_header_df_nod{key}.pkl",
            "wb",
        ) as f:
            pickle.dump(polars_df, f)

        # optionally save as fits files
        if save_fits:
            _savefits(bkg_subbed_images, key, hdr_dicts, config, process_path)

        # do the plotting
        try:
            _qa_plots(
                {key: bkg_subbed_images},
                centroid_positions,
                {key: np.array(times)},
                output_dir,
                target,
            )
        except Exception as e:
            logger.error(PROCESS_NAME, f"_qa_plots failed due to {e}")

    logger.info(PROCESS_NAME, "Background subtraction is done!")
    return True


if __name__ == "__main__":
    configfilename = "./nod_config_ngc4151.json"
    mylogger = Logger("../test/")
    do_bkg_subtraction(configfilename, mylogger)
