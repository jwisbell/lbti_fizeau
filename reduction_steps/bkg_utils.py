import matplotlib.pyplot as plt
from astropy.io import fits
import numpy as np
from matplotlib import animation
from datetime import datetime
from utils.util_logger import Logger
import time
import polars as pl
import os
from astropy.stats import sigma_clip
from utils.utils import create_filestructure
from typing import Tuple, Dict


PROCESS_NAME = "bkg_subtraction"
instrument = "NOMIC"
logger = None


def _extract_window(im, center, extraction_size=0):
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


# def _fast_fit_slope(imgcube):
#     # Mask NaN and Inf values
#     imgcube = np.ma.masked_invalid(imgcube)
#
#     # Create time vector (starting from 1)
#     time = np.arange(imgcube.shape[0], dtype=np.float64) + 1
#
#     # Prepare shape for broadcasting
#     tshape = tuple(np.roll(imgcube.shape, -1))
#
#     # Core precomputed quantities
#     time_reshaped = np.transpose(np.resize(time, tshape), (2, 0, 1))
#     time_sq_reshaped = np.transpose(np.resize(np.square(time), tshape), (2, 0, 1))
#
#     Sx = np.ma.array(time_reshaped).sum(axis=0, dtype=np.float64)
#     Sxx = np.ma.array(time_sq_reshaped).sum(axis=0, dtype=np.float64)
#     Sy = np.ma.mean(imgcube, axis=0, dtype=np.float64)
#     Sxsx = Sx * Sx
#     Sxy = (imgcube * time[:, np.newaxis, np.newaxis]).sum(axis=0, dtype=np.float64)
#     n = np.ma.count(imgcube, axis=0)
#
#     # Apply regression formula (NaN-safe due to masked arrays)
#     beta = ne.evaluate(
#         "(((Sx / n) * Sy) - (Sxy / n)) / (((Sxsx / (n * n))) - (Sxx / n))"
#     )
#     alpha = ne.evaluate("Sy - (beta * (Sx / n))")
#
#     # Fill masked values (if desired)
#     beta = np.ma.filled(beta, np.nan)
#     alpha = np.ma.filled(alpha, np.nan)
#
#     return beta, alpha


def _fix_gain(image):
    # for each channel (spaced by XX pixels), find the median value of the top 100 and bottom 100 pixels and subtract out
    new_image = np.copy(image)
    channel_width = 64
    channel_region = 50
    for xstart in np.arange(0, image.shape[0] + channel_width, channel_width):
        region_top = image[4:channel_region, xstart : xstart + channel_width]
        region_bot = image[-channel_region:4, xstart : xstart + channel_width]
        channel_val = np.nanmean([np.nanmedian(region_top), np.nanmedian(region_bot)])
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
    output_dir="",
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
    if do_up_the_ramp:
        # create the directory to save fits files in
        create_filestructure(
            output_dir, "ramp_fits", prefix="intermediate/bkg_subtraction"
        )

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
            logger.info(PROCESS_NAME, "\t Using uncompressed target files")
        else:
            logger.info(
                PROCESS_NAME,
                "\t Using compressed target files (*fits.gz) -- NOTE: this takes longer to process",
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
                                logger.info(
                                    PROCESS_NAME, f"\t\t Ramp fitting on {filename}"
                                )
                                im: np.ndarray = (
                                    _ramp_fitting(x[0].data, do_plot=False) - mean_dark
                                )
                                # TODO: save the ramp-fitted file in the format Jordan wants
                                lmir_im_fmt = np.zeros(shape=(2, 2048, 2048))
                                lmir_im_fmt[1] = im
                                new_hdu = fits.PrimaryHDU(
                                    data=lmir_im_fmt, header=x[0].header
                                )
                                hdul = fits.HDUList([new_hdu])
                                hdul.writeto(
                                    f"{output_dir}/intermediate/bkg_subtraction/ramp_fits/{filename.split('/')[-1].split('.fit')[0]}_ramp.fits",
                                    overwrite=True,
                                )

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
                print(filename, e)
                continue
        print(f"Took {time.time() - start:.1f} seconds")
        images[name] = temp
        pas[name] = temp_pas  # angle_mean(temp_pas)
        fnames[name] = filenames
        timestamps[name] = obstime
        all_headers[name] = headers

        logger.info(PROCESS_NAME, f"\t Done! Mean PA {np.mean(pas[name])}")
    return images, pas, fnames, timestamps, all_headers


def _load_fits_sum(
    fdir,
    entry,
    nod_key,
    prefix,
    output_dir="",
    ramp_params: dict = {"idx": -1, "subtract_min": False},
    do_up_the_ramp=False,
    mean_dark=0.0,
    mean_bkg: np.ndarray = np.zeros((1)),
    save_intermediate=False,
) -> Tuple[np.ndarray, list]:
    # for each nod position open the files
    # extract a box of size `aperture size` nod position in each file

    headers = []

    if do_up_the_ramp:
        # create the directory to save fits files in
        create_filestructure(
            output_dir, "ramp_fits", prefix="intermediate/bkg_subtraction"
        )
    elif save_intermediate:
        create_filestructure(
            output_dir, "bkg_subtracted_images", prefix="intermediate/bkg_subtraction"
        )

    logger.info(PROCESS_NAME, f"Loading nod {nod_key}")
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
    im_sum = 0.0
    for filename in filenames:
        try:
            with fits.open(filename) as x:
                im = np.copy(x[0].data) - mean_dark - mean_bkg
                if len(x[0].data.shape) > 2:
                    im = np.copy(x[0].data[ramp_params["idx"]])
                    if instrument != "NOMIC":
                        im = (
                            np.copy(x[0].data[ramp_params["idx"]])
                            - mean_dark
                            - mean_bkg
                        )
                        # subtracting out the "zero" exposure to remove bad pixels
                        if ramp_params["subtract_min"]:
                            im -= x[0].data[0]
                        if do_up_the_ramp:
                            im: np.ndarray = (
                                _ramp_fitting(x[0].data, do_plot=False)
                                - mean_dark
                                - mean_bkg
                            )
                            # TODO: save the ramp-fitted file in the format Jordan wants
                            lmir_im_fmt = np.zeros(shape=(2, 2048, 2048))
                            lmir_im_fmt[1] = im
                            new_hdu = fits.PrimaryHDU(
                                data=lmir_im_fmt, header=x[0].header
                            )
                            hdul = fits.HDUList([new_hdu])
                            hdul.writeto(
                                f"{output_dir}/intermediate/bkg_subtraction/ramp_fits/{filename.split('/')[-1].split('.fit')[0]}_ramp.fits",
                                overwrite=True,
                            )
                im_sum += im
                headers.append({k: v for k, v in x[0].header.items()})
                # optionall save the intermediate files individually
                # TODO: write wrapper function
                if save_intermediate:
                    new_hdu = fits.PrimaryHDU(data=im, header=x[0].header)
                    hdul = fits.HDUList([new_hdu])
                    hdul.writeto(
                        f"{output_dir}/intermediate/bkg_subtraction/bkg_subtracted_images/{filename.split('/')[-1].split('.fit')[0]}_bkgsub.fits"
                    )
        except FileNotFoundError as e:
            logger.warn(PROCESS_NAME, f"\t\t {filename} failed, {e}")
            continue
        except OSError as e:
            print(filename)
            continue
    print(f"Took {time.time() - start:.1f} seconds")
    return np.array(im_sum), headers


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
