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


def _load_fits_files(fdir, nods, prefix, skipkeys=[]):
    # for each nod position open the files
    # extract a box of size `aperture size` nod position in each file
    # extract background aperture in each file
    images = {}
    pas = {}
    fnames = {}
    # TODO: can this be sped up using multiprocess?
    for name, entry in nods.items():
        if name in skipkeys:
            # logger.info(PROCESS_NAME,"skipping!", name)
            continue
        temp = []
        temp_pas = []
        logger.info(PROCESS_NAME, f"Loading nod {name}")
        filenames = [
            f"{fdir}{prefix}{str(i).zfill(6)}.fits"
            for i in range(entry["start"], entry["end"] + 1)
        ]
        logger.info(PROCESS_NAME, f"\t {len(filenames)} files")

        for filename in filenames:
            try:
                with fits.open(filename) as x:
                    im = np.copy(x[0].data[0])
                    if instrument != "NOMIC":
                        im = np.copy(x[0].data[-1])
                    temp.append(im)  # extract_window(im, entry['position']) )
                    pa = float(x[0].header["LBT_PARA"])
                    temp_pas.append(pa)
                    # return
            except FileNotFoundError:
                logger.warn(PROCESS_NAME, f"\t\t {filename} failed")
                continue
        images[name] = temp
        pas[name] = temp_pas  # angle_mean(temp_pas)
        fnames[name] = filenames

        logger.info(PROCESS_NAME, f"\t Done! Mean PA {np.mean(pas[name])}")
    return images, pas, fnames


def _window_background_subtraction(im_arr, background, window_center):
    # do the background subtraction inside a subwindow
    images = []
    for im in im_arr:
        test = _extract_window(im - background, window_center)
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


def _qa_plots(bg_subtracted_frames, ims, centroid_positions, output_dir, target):
    # ## (optional) Plot cycles to quickly assess quality
    # HTML(image_video(bg_subtracted_frames["1"][::2],"2").to_html5_video())
    for key in bg_subtracted_frames.keys():
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
    for key in ims.keys():
        plt.plot([np.nanmean(x) for x in bg_subtracted_frames[key]], label=key)
    plt.legend()
    plt.savefig(f"{output_dir}/plots/{PROCESS_NAME}/{target}_allnods_meanflux.png")
    plt.close("all")


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

    prefix = f"n_{obsdate}_"
    if instrument != "NOMIC":
        prefix = f"lm_{obsdate}_"

    process_path = f"intermediate/{PROCESS_NAME}/"

    list_keys = np.array(list(nod_info.keys()))
    num_entries = len(nod_info.keys())
    num_processed = 0

    # handle the file loading in batches to limit RAM usage
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

        ims, rotations, _ = _load_fits_files(
            data_dir, nod_info, prefix, skipkeys=temp_skips
        )

        # ## Do background subtraction and extract in a window
        backgrounds = {}
        for name, entry in ims.items():
            backgrounds[name] = {
                "mean": np.nanmean(entry, 0),
                "std": np.nanstd(entry, 0),
            }

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

            im = np.sum(bg_subtracted_frames[key], 0)
            im = median_filter(im, 3)
            centroid_positions[key] = [
                np.argmax(np.sum(im, 0)),
                np.argmax(np.sum(im, 1)),
            ]
            if extraction_size >= ims[key][0].shape[0]:
                centroid_positions[key] = nod_info[key]["position"]

            np.save(
                f"{output_dir}/{process_path}/{target}_centroid-positions_cycle{key}.npy",
                [np.argmax(np.sum(im, 0)), np.argmax(np.sum(im, 1))],
            )
            np.save(
                f"{output_dir}/{process_path}/{target}_rotations_cycle{key}.npy",
                rotations[key],
            )
            np.save(
                f"{output_dir}/{process_path}/{target}_bkg-subtracted_cycle{key}.npy", x
            )  # save the background-subtracted frames

        # do the plotting
        try:
            _qa_plots(bg_subtracted_frames, ims, centroid_positions, output_dir, target)
        except Exception as e:
            logger.error(PROCESS_NAME, f"_qa_plots failed due to {e}")
            return False

        num_processed += batch_size
        logger.info(
            PROCESS_NAME,
            f"Batch done! Processed {min(num_processed,num_entries)} of {num_entries}",
        )
    logger.info(PROCESS_NAME, "Background subtraction is done!")
    return True


if __name__ == "__main__":
    configfilename = "./nod_config_ngc4151.json"
    mylogger = Logger("../test/")
    do_bkg_subtraction(configfilename, mylogger)
