"""
do_bkg_subtraction -- LIZARD Pipeline
Author: Jacob Isbell

Functions to load the raw data files and then do background subtraction using the user-specified nod pairs.
This is the initial data reduction step, and is the most likely to fail if the config file is incorrectly prepared.

Called by lizard_reduce
"""

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
import numexpr as ne

from calibration_steps.bad_pixel_correction import (
    correct_image_after_bpm,
    load_bpm,
    apply_bad_pixel_mask,
)

PROCESS_NAME = "bkg_subtraction"
extraction_size = 100
instrument = "NOMIC"
logger = None


# below this should just run
#####################################################################################


# # Function Definitions
def _extract_window(im, center):
    """
    Extracts a window of a specified size from an image centered at a given location.

    Parameters:
        im (numpy array): The input image.
        center (tuple): The coordinates (x, y) of the center of the window.

    Returns:
        numpy array: The extracted window of the image.
    """
    xc, yc = center
    if extraction_size >= im.shape[0] or extraction_size <= 0:
        return im

    ylower = np.max([0, yc - extraction_size // 2])
    yupper = np.min([im.shape[0], yc + extraction_size // 2])
    xlower = np.max([0, xc - extraction_size // 2])
    xupper = np.min([im.shape[1], xc + extraction_size // 2])

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


def _fast_fit_slope(imgcube):
    # Mask NaN and Inf values
    imgcube = np.ma.masked_invalid(imgcube)

    # Create time vector (starting from 1)
    time = np.arange(imgcube.shape[0], dtype=np.float64) + 1

    # Prepare shape for broadcasting
    tshape = tuple(np.roll(imgcube.shape, -1))

    # Core precomputed quantities
    time_reshaped = np.transpose(np.resize(time, tshape), (2, 0, 1))
    time_sq_reshaped = np.transpose(np.resize(np.square(time), tshape), (2, 0, 1))

    Sx = np.ma.array(time_reshaped).sum(axis=0, dtype=np.float64)
    Sxx = np.ma.array(time_sq_reshaped).sum(axis=0, dtype=np.float64)
    Sy = np.ma.mean(imgcube, axis=0, dtype=np.float64)
    Sxsx = Sx * Sx
    Sxy = (imgcube * time[:, np.newaxis, np.newaxis]).sum(axis=0, dtype=np.float64)
    n = np.ma.count(imgcube, axis=0)

    # Apply regression formula (NaN-safe due to masked arrays)
    beta = ne.evaluate(
        "(((Sx / n) * Sy) - (Sxy / n)) / (((Sxsx / (n * n))) - (Sxx / n))"
    )
    alpha = ne.evaluate("Sy - (beta * (Sx / n))")

    # Fill masked values (if desired)
    beta = np.ma.filled(beta, np.nan)
    alpha = np.ma.filled(alpha, np.nan)

    return beta, alpha


def _fix_gain(image):
    # for each channel (spaced by XX pixels), find the median value of the top 100 and bottom 100 pixels and subtract out
    new_image = np.copy(image)
    channel_width = 64
    channel_region = 50
    for xstart in np.arange(0, image.shape[0] + channel_width, channel_width):
        region_top = image[0:channel_region, xstart : xstart + channel_width]
        region_bot = image[-channel_region:, xstart : xstart + channel_width]
        # fig, (ax, bx) = plt.subplots(1, 2)
        # ax.imshow(region_top)
        # bx.imshow(region_bot)
        # plt.show()
        # plt.close()
        channel_val = np.mean([np.median(region_top), np.median(region_bot)])
        new_image[:, xstart : xstart + channel_width] -= channel_val

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
    beta, alpha = _fast_fit_slope(delta_ims)

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


def _load_fits_files(
    fdir,
    nods,
    prefix,
    skipkeys=[],
    ramp_params: dict = {"idx": -1, "subtract_min": False},
    do_up_the_ramp=False,
    mean_dark=0.0,
):
    # for each nod position open the files
    # extract a box of size `aperture size` nod position in each file
    # extract background aperture in each file
    images = {}
    pas = {}
    fnames = {}
    timestamps = {}
    all_headers = {}
    # TODO: can this be sped up using multiprocess?
    for name, entry in nods.items():
        if name in skipkeys:
            # logger.info(PROCESS_NAME,"skipping!", name)
            continue
        temp = []
        temp_pas = []
        obstime = []
        headers = []
        logger.info(PROCESS_NAME, f"Loading nod {name}")
        filenames = [
            f"{fdir}{prefix}{str(i).zfill(6)}.fits"
            for i in range(entry["start"], entry["end"] + 1)
        ]
        logger.info(PROCESS_NAME, f"\t {len(filenames)} files")

        # Check if we are dealing with fits or fits.gz
        test_fname = filenames[np.random.randint(0, len(filenames))]
        if os.path.exists(test_fname):
            print("Using uncompressed target files")
        else:
            print(
                "Using compressed target files (*fits.gz) -- NOTE: this takes longer to process"
            )
            filenames = [
                f"{fdir}{prefix}{str(i).zfill(6)}.fits.gz"
                for i in range(entry["start"], entry["end"] + 1)
            ]

        start = time.time()
        for filename in filenames:
            try:
                with fits.open(filename) as x:
                    im = np.copy(x[0].data) - mean_dark
                    if len(x[0].data.shape) > 2:
                        im = np.copy(x[0].data[ramp_params["idx"]])
                        if instrument != "NOMIC":
                            im = np.copy(x[0].data[ramp_params["idx"]]) - mean_dark
                            # subtracting out the "zero" exposure to remove bad pixels
                            if ramp_params["subtract_min"]:
                                im -= x[0].data[0]
                            if do_up_the_ramp:
                                im = _ramp_fitting(x[0].data, do_plot=False)

                    temp.append(_extract_window(im, entry["position"]))
                    # temp.append(im)
                    pa = float(x[0].header["LBT_PARA"])
                    temp_pas.append(pa)
                    obstime.append(
                        time_convert(x[0].header["date-obs"], x[0].header["time-obs"])
                    )
                    headers.append({k: v for k, v in x[0].header.items()})
            except FileNotFoundError as e:
                logger.warn(PROCESS_NAME, f"\t\t {filename} failed, {e}")
                continue
            except OSError as e:
                print(filename)
                continue
        print(f"Took {time.time() - start:.1f} seconds")
        images[name] = temp
        pas[name] = temp_pas  # angle_mean(temp_pas)
        fnames[name] = filenames
        timestamps[name] = obstime
        all_headers[name] = headers

        logger.info(PROCESS_NAME, f"\t Done! Mean PA {np.mean(pas[name])}")
    return images, pas, fnames, timestamps, all_headers


def _window_background_subtraction(im_arr, background, window_center):
    # do the background subtraction inside a subwindow
    images = []
    for im in im_arr:
        test = im - background  # _extract_window(im - background, window_center)
        images.append(np.array(test))
    # logger.info(PROCESS_NAME,len(images))
    return images


def _image_video(img_list1, name):
    def init():
        img1.set_data(img_list1[0])
        return (img1,)

    def animate(i):
        img1.set_data(img_list1[i])
        return (img1,)

    fig, ax = plt.subplots()
    ax.set_title(name)
    img1 = ax.imshow(img_list1[0], cmap="Greys", origin="lower")
    anim = animation.FuncAnimation(
        fig, animate, init_func=init, frames=len(img_list1), interval=20, blit=True
    )
    return anim


def _qa_plots(bg_subtracted_frames, centroid_positions, timestamps, output_dir, target):
    # ## (optional) Plot cycles to quickly assess quality
    # HTML(image_video(bg_subtracted_frames["1"][::2],"2").to_html5_video())
    for key in bg_subtracted_frames.keys():
        if "bkg" in key or "off" in key:
            continue
        _ = plt.figure()
        plt.imshow(
            np.mean(bg_subtracted_frames[key][:], 0),
            origin="lower",
            norm=PowerNorm(0.5),
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
            [t for t in timestamps[key]],
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
        skip_bpm = config["skip_bpm_correction"]
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


def _old_bkg_subtraction(
    nod_info,
    data_dir,
    prefix,
    batch_size,
    skips,
    ramp_params,
    output_dir,
    process_path,
    target,
    save_fits=False,
    config=None,
    do_up_the_ramp=False,
    skip_bpm=False,
    mean_dark=0.0,
):
    list_keys = np.array(list(nod_info.keys()))
    num_entries = len(nod_info.keys())
    num_processed = 0

    while num_processed < num_entries:
        temp_skips = np.append(
            list_keys[
                np.append(
                    np.arange(num_processed + batch_size, num_entries, 1),
                    np.arange(0, num_processed, 1),
                )
            ],
            skips,
        )

        ims, rotations, _, timestamps, hdr_dicts = _load_fits_files(
            data_dir,
            nod_info,
            prefix,
            skipkeys=temp_skips,
            ramp_params=ramp_params,
            do_up_the_ramp=do_up_the_ramp,
            mean_dark=mean_dark,
        )

        # ## Do background subtraction and extract in a window
        backgrounds = {}
        for name, entry in ims.items():
            backgrounds[name] = {
                "mean": np.nanmean(entry, 0),
                "std": np.nanstd(entry, 0),
            }

        # 2.5 apply the bad pixel map
        # multiply the BPM with each image
        bg_subtracted_frames = {}
        for key in ims.keys():
            bpm = load_bpm(hdr_dicts[key][0])

            bkg_subbed = [
                im - backgrounds[nod_info[key]["subtract"]]["mean"] for im in ims[key]
            ]
            # 2.5.b. multiply the images by the bpm
            bpm_windowed = _extract_window(bpm, nod_info[key]["position"])
            masked_images = apply_bad_pixel_mask(
                bpm_windowed, bkg_subbed, skip=skip_bpm
            )

            # 3 Correct the bad pixels with the median of the neighbors
            corrected_ims = [
                correct_image_after_bpm(im, skip=skip_bpm) for im in masked_images
            ]
            bg_subtracted_frames[key] = np.array(corrected_ims)

            # optionally save as fits files
            if save_fits:
                _savefits(corrected_ims, key, hdr_dicts[key], config, process_path)

        # save the background-subtracted sub-windows in processed data folder
        centroid_positions = {}

        for key in bg_subtracted_frames.keys():
            x = bg_subtracted_frames[key]
            logger.info(PROCESS_NAME, f"Processing key {key}")
            # if "bkg" in key or "off" in key:
            # continue
            im = np.sum(bg_subtracted_frames[key], 0)
            im = median_filter(im, 5)
            centroid_positions[key] = [
                np.clip(np.argmax(np.nansum(im, 0)), 32, len(im) - 32),
                np.clip(np.argmax(np.nansum(im, 1)), 32, len(im) - 32),
            ]
            print(centroid_positions)
            # if extraction_size >= ims[key][0].shape[0]:
            #    centroid_positions[key] = nod_info[key]["position"]

            # TODO: put almost all of this in the dataframe
            np.save(
                f"{output_dir}/{process_path}/{target}_centroid-positions_cycle{key}.npy",
                [np.argmax(np.nansum(im, 0)), np.argmax(np.nansum(im, 1))],
            )
            np.save(
                f"{output_dir}/{process_path}/{target}_rotations_cycle{key}.npy",
                rotations[key],
            )
            np.save(
                f"{output_dir}/{process_path}/{target}_bkg-subtracted_cycle{key}.npy", x
            )  # save the background-subtracted frames

            np.save(
                f"{output_dir}/{process_path}/{target}_timestamps_cycle{key}.npy",
                timestamps[key],
            )  # save the time stamps of the background-subtracted frames

            if "bkg" in key or "off" in key:
                continue

            polars_df = _merge_headers_to_df(hdr_dicts[key], key)
            # Save the DataFrame to a pickle file
            with open(
                f"{output_dir}/intermediate/headers/{target}_header_df_nod{key}.pkl",
                "wb",
            ) as f:
                pickle.dump(polars_df, f)

        # do the plotting
        try:
            _qa_plots(
                bg_subtracted_frames,
                centroid_positions,
                timestamps,
                output_dir,
                target,
            )
        except Exception as e:
            logger.error(PROCESS_NAME, f"_qa_plots failed due to {e}")
            # return False

        num_processed += batch_size
        logger.info(
            PROCESS_NAME,
            f"Batch done! Processed {min(num_processed, num_entries)} of {num_entries}",
        )


def do_bkg_subtraction(config: dict, mylogger: Logger) -> bool:
    # # Load the data
    #
    # 1. Extracts a window from each frame
    # 2. Then does background subtraction using specified nod pairs

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

    prefix = f"n_{obsdate}_"
    if instrument != "NOMIC":
        prefix = f"lm_{obsdate}_"

    process_path = f"intermediate/{PROCESS_NAME}/"

    # 0. Load the (optional) darks for dark subtraction
    mean_dark = 0
    if dark_file_range != (0, 0):
        dark_files, _ = _get_filenames(
            data_dir, prefix, dark_file_range[0], dark_file_range[1], 0, 0
        )
        mean_dark = _load_darks(dark_files)

    try:
        from fits_lizard import subtract_mean_from_list

        if do_up_the_ramp:
            # force the script into python mode since up the ramp fitting
            # not yet working in the rust package
            raise AssertionError

        bg_subtracted_frames = {}
        rotations = {}
        timestamps = {}

        # 1. For each key collect the _filenames_ to be loaded for both the key and for the sub_key
        for key, value in nod_info.items():
            if "off" in key or "bkg" in key or "skip" in key:
                continue

            logger.info(PROCESS_NAME, f"Doing background subtraction for key {key}")
            start_fn = value["start"]
            end_fn = value["end"]
            sub_key = value["subtract"]

            sub_start_fn = nod_info[sub_key]["start"]
            sub_end_fn = nod_info[sub_key]["end"]
            # 1. Get the filenames to load
            obj_files, bkg_files = _get_filenames(
                data_dir, prefix, start_fn, end_fn, sub_start_fn, sub_end_fn
            )

            # 2. Calculate the bg_subtracted_frames for that key
            start = time.time()
            result = subtract_mean_from_list(
                obj_files, bkg_files, do_up_the_ramp, ramp_params["subtract_min"]
            )  # returns images, rotations, julian dates
            bkg_sub_ims = [x[0] - 2 * mean_dark for x in result[0]]
            rots = [x[1] for x in result[0]]
            times = [x[2] for x in result[0]]
            hdr_dicts = [x[3] for x in result[0]]

            polars_df = _merge_headers_to_df(hdr_dicts, key)
            print(f"Took {time.time() - start:.1f} seconds")

            # Save the DataFrame to a pickle file
            with open(
                f"{output_dir}/intermediate/headers/{target}_header_df_nod{key}.pkl",
                "wb",
            ) as f:
                pickle.dump(polars_df, f)

            # 2.5 apply the bad pixel map
            # 2.5.a. load the bad pixel map
            bpm = load_bpm(hdr_dicts[0])

            # 2.5.b. multiply the images by the bpm
            masked_images = apply_bad_pixel_mask(bpm, bkg_sub_ims, skip=skip_bpm)

            # 3. Crop each frame to the right size
            cropped_ims = [
                _extract_window(im, value["position"]) for im in masked_images
            ]

            # 3a Correct the bad pixels with the median of the neighbors
            corrected_ims = [
                correct_image_after_bpm(im, skip=skip_bpm) for im in cropped_ims
            ]

            # 4. Save the cropped frames in the bg_subtracted_frames dict
            bg_subtracted_frames[key] = np.array(corrected_ims)
            rotations[key] = np.array(rots)
            timestamps[key] = np.array(times)

            # optionally save as fits files
            if save_fits:
                _savefits(corrected_ims, key, hdr_dicts, config, process_path)

        # save the background-subtracted sub-windows in processed data folder
        centroid_positions = {}

        for key in bg_subtracted_frames.keys():
            x = bg_subtracted_frames[key]
            logger.info(PROCESS_NAME, f"Saving/plotting key {key}")
            # if "bkg" in key or "off" in key:
            # continue
            im = np.sum(bg_subtracted_frames[key], 0)
            im = median_filter(im, 5)
            centroid_positions[key] = [
                np.clip(np.argmax(np.nansum(im, 0)), 32, len(im) - 32),
                np.clip(np.argmax(np.nansum(im, 1)), 32, len(im) - 32),
            ]

            # if extraction_size >= ims[key][0].shape[0]:
            #    centroid_positions[key] = nod_info[key]["position"]
            # TODO: save almost all of this in the header dataframe

            np.save(
                f"{output_dir}/{process_path}/{target}_centroid-positions_cycle{key}.npy",
                [np.argmax(np.nansum(im, 0)), np.argmax(np.nansum(im, 1))],
            )
            np.save(
                f"{output_dir}/{process_path}/{target}_rotations_cycle{key}.npy",
                rotations[key],
            )
            # save the background-subtracted frames
            np.save(
                f"{output_dir}/{process_path}/{target}_bkg-subtracted_cycle{key}.npy", x
            )

            np.save(
                f"{output_dir}/{process_path}/{target}_timestamps_cycle{key}.npy",
                timestamps[key],
            )  # save the time stamps of the background-subtracted frames
        # do the plotting
        try:
            _qa_plots(
                bg_subtracted_frames,
                centroid_positions,
                timestamps,
                output_dir,
                target,
            )
        except Exception as e:
            logger.error(PROCESS_NAME, f"_qa_plots failed due to {e}")

    except (ModuleNotFoundError, ImportError, AssertionError) as e:
        logger.warn(
            PROCESS_NAME,
            f"fits_lizard package not found OR up-the-ramp selected, proceeding with the slower method: {e}",
        )
        # handle the file loading in batches to limit RAM usage
        # OLD WAY -- do only if rust version not available
        _old_bkg_subtraction(
            nod_info,
            data_dir,
            prefix,
            batch_size,
            skips,
            ramp_params,
            output_dir,
            process_path,
            target,
            do_up_the_ramp=do_up_the_ramp,
            save_fits=save_fits,
            config=config,
            skip_bpm=skip_bpm,
            mean_dark=mean_dark,
        )

    logger.info(PROCESS_NAME, "Background subtraction is done!")
    return True


if __name__ == "__main__":
    configfilename = "./nod_config_ngc4151.json"
    mylogger = Logger("../test/")
    do_bkg_subtraction(configfilename, mylogger)
