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
    if extraction_size >= im.shape[0]:
        return im

    ylower = np.max([0, yc - extraction_size // 2])
    yupper = np.min([im.shape[0], yc + extraction_size // 2])
    xlower = np.max([0, xc - extraction_size // 2])
    xupper = np.min([im.shape[1], xc + extraction_size // 2])

    return im[ylower:yupper, xlower:xupper]


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
):
    # for each nod position open the files
    # extract a box of size `aperture size` nod position in each file
    # extract background aperture in each file
    images = {}
    pas = {}
    fnames = {}
    timestamps = {}
    # TODO: can this be sped up using multiprocess?
    for name, entry in nods.items():
        if name in skipkeys:
            # logger.info(PROCESS_NAME,"skipping!", name)
            continue
        temp = []
        temp_pas = []
        obstime = []
        logger.info(PROCESS_NAME, f"Loading nod {name}")
        filenames = [
            f"{fdir}{prefix}{str(i).zfill(6)}.fits"
            for i in range(entry["start"], entry["end"] + 1)
        ]
        logger.info(PROCESS_NAME, f"\t {len(filenames)} files")

        for filename in filenames:
            try:
                with fits.open(filename) as x:
                    im = np.copy(x[0].data)
                    if len(x[0].data.shape) > 2:
                        im = np.copy(x[0].data[ramp_params["idx"]])
                        if instrument != "NOMIC":
                            im = np.copy(x[0].data[ramp_params["idx"]])
                            # subtracting out the "zero" exposure to remove bad pixels
                            if ramp_params["subtract_min"]:
                                im -= x[0].data[0]

                    temp.append(_extract_window(im, entry["position"]))
                    pa = float(x[0].header["LBT_PARA"])
                    temp_pas.append(pa)
                    obstime.append(
                        time_convert(x[0].header["date-obs"], x[0].header["time-obs"])
                    )
                    # returna
            except FileNotFoundError as e:
                logger.warn(PROCESS_NAME, f"\t\t {filename} failed, {e}")
                continue
            except OSError as e:
                print(filename)
                continue
        images[name] = temp
        pas[name] = temp_pas  # angle_mean(temp_pas)
        fnames[name] = filenames
        timestamps[name] = obstime

        logger.info(PROCESS_NAME, f"\t Done! Mean PA {np.mean(pas[name])}")
    return images, pas, fnames, timestamps


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

        ims, rotations, _, timestamps = _load_fits_files(
            data_dir, nod_info, prefix, skipkeys=temp_skips, ramp_params=ramp_params
        )

        # ## Do background subtraction and extract in a window
        backgrounds = {}
        for name, entry in ims.items():
            backgrounds[name] = {
                "mean": np.nanmean(entry, 0),
                "std": np.nanstd(entry, 0),
            }
        print(list(backgrounds.keys()))
        bg_subtracted_frames = {
            key: _window_background_subtraction(
                ims[key],
                backgrounds[nod_info[key]["subtract"]]["mean"],
                nod_info[key]["position"],
            )
            for key in ims.keys()
        }

        # save the background-subtracted sub-windows in processed data folder
        centroid_positions = {}

        for key in bg_subtracted_frames.keys():
            x = bg_subtracted_frames[key]
            logger.info(PROCESS_NAME, f"Processing key {key}")
            # if "bkg" in key or "off" in key:
            # continue
            im = np.sum(bg_subtracted_frames[key], 0)
            im = median_filter(im, 7)
            centroid_positions[key] = [
                np.argmax(np.nansum(im, 0)),
                np.argmax(np.nansum(im, 1)),
            ]
            print(centroid_positions)
            # if extraction_size >= ims[key][0].shape[0]:
            #    centroid_positions[key] = nod_info[key]["position"]

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
            f"Batch done! Processed {min(num_processed,num_entries)} of {num_entries}",
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
        ramp_params = config["ramp_params"]
    except KeyError:
        logger.warn(
            PROCESS_NAME,
            "No ramp params set, using default {idx:-1, subtract_min:False}",
        )
        ramp_params = {"idx": -1, "subtract_min": False}

    prefix = f"n_{obsdate}_"
    if instrument != "NOMIC":
        prefix = f"lm_{obsdate}_"

    process_path = f"intermediate/{PROCESS_NAME}/"

    try:
        from fits_lizard import subtract_mean_from_list

        bg_subtracted_frames = {}
        rotations = {}
        timestamps = {}

        # 1. For each key collect the _filenames_ to be loaded for both the key and for the sub_key
        for key, value in nod_info.items():
            if "off" in key or "bkg" in key or "skip" in key:
                continue
            # TODO: there is only one image being saved...
            logger.info(PROCESS_NAME, f"Doing background subtraction for key {key}")
            start_fn = value["start"]
            end_fn = value["end"]
            sub_key = value["subtract"]

            sub_start_fn = nod_info[sub_key]["start"]
            sub_end_fn = nod_info[sub_key]["end"]

            obj_files = [
                f"{data_dir}/{prefix}{str(i).zfill(6)}.fits"
                for i in range(start_fn, end_fn + 1)
            ]
            bkg_files = [
                f"{data_dir}/{prefix}{str(i).zfill(6)}.fits"
                for i in range(sub_start_fn, sub_end_fn + 1)
            ]

            # 2. Calculate the bg_subtracted_frames for that key
            result = subtract_mean_from_list(
                obj_files, bkg_files
            )  # returns images, rotations, julian dates
            bkg_sub_ims = [x[0] for x in result]
            rots = [x[1] for x in result]
            times = [x[2] for x in result]

            # 3. Crop each frame to the right size
            cropped_ims = [_extract_window(im, value["position"]) for im in bkg_sub_ims]
            # TODO: check here

            # 4. Save the cropped frames in the bg_subtracted_frames dict
            bg_subtracted_frames[key] = np.array(cropped_ims)
            rotations[key] = np.array(rots)
            timestamps[key] = np.array(times)

        # save the background-subtracted sub-windows in processed data folder
        centroid_positions = {}

        for key in bg_subtracted_frames.keys():
            x = bg_subtracted_frames[key]
            logger.info(PROCESS_NAME, f"Saving/plotting key {key}")
            # if "bkg" in key or "off" in key:
            # continue
            im = np.sum(bg_subtracted_frames[key], 0)
            im = median_filter(im, 7)
            centroid_positions[key] = [
                np.argmax(np.nansum(im, 0)),
                np.argmax(np.nansum(im, 1)),
            ]

            # if extraction_size >= ims[key][0].shape[0]:
            #    centroid_positions[key] = nod_info[key]["position"]

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

    except ModuleNotFoundError as e:
        logger.warn(
            PROCESS_NAME,
            "fits_lizard package not found, proceeding with the slower method",
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
        )

    logger.info(PROCESS_NAME, "Background subtraction is done!")
    return True


if __name__ == "__main__":
    configfilename = "./nod_config_ngc4151.json"
    mylogger = Logger("../test/")
    do_bkg_subtraction(configfilename, mylogger)
