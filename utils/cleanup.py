import json
from sys import argv
from glob import glob
import numpy as np
import pickle
from astropy.io import fits
import matplotlib.pyplot as plt

from utils import create_filestructure, imshift, find_max_loc


def transfer_function(configfile):
    with open(configfile, "r") as inputfile:
        configdata = json.load(inputfile)

    target = configdata["target"]
    data_dir = configdata["data_dir"]
    output_dir = configdata["output_dir"]

    # collect times, rotations, and background subtracted images

    fdir = f"{output_dir}/intermediate/bkg_subtraction/"

    # first the times
    time_files = np.sort(glob(f"{fdir}/{target}*timestamps*.npy"))
    timestamps = [np.load(x) for x in time_files if "bkg.npy" not in x]

    # then the rotations
    rot_files = np.sort(glob(f"{fdir}/{target}*rotations*.npy"))
    rotations = [np.load(x) for x in rot_files if "bkg.npy" not in x]

    # then the images
    bkg_files = np.sort(glob(f"{fdir}/{target}*bkg-subtracted*.npy"))
    images = [np.load(x) for x in bkg_files if "bkg.npy" not in x]

    all_fluxes = []
    all_times = []
    all_rots = []

    fig = plt.figure()
    for i in range(len(timestamps)):
        # plt.scatter(timestamps[i], rotations[i])
        fluxes = np.array([np.mean(x) for x in images[i]])
        fluxes[0 : len(fluxes) + 200 : 200] = np.nan
        plt.scatter(timestamps[i], fluxes)
        all_fluxes += list(fluxes)
        all_times += list(timestamps[i].flatten())
        all_rots += list(rotations[i].flatten())
    plt.show()

    transfer_function_dict = {
        "times": all_times,
        "rotations": all_rots,
        "fluxes": all_fluxes,
    }

    with open(
        f"{output_dir}/intermediate/bkg_subtraction/{target}_transferfunction.pkl", "wb"
    ) as handle:
        pickle.dump(transfer_function_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)


def frame_selected_fits_files(configfile):
    """Turn the saved frame-selected images from each cycle as a fits file"""
    with open(configfile, "r") as inputfile:
        configdata = json.load(inputfile)

    target = configdata["target"]
    output_dir = configdata["output_dir"]
    skips = configdata["skips"]

    # collect the images
    fdir = f"{output_dir}/intermediate/frame_selection/"

    # load the images
    image_files = np.sort(glob(f"{fdir}/{target}*imstack*.npy"))
    images = [np.load(x) for x in image_files if "bkg.npy" not in x]

    # load the image info
    fs_info_files = np.sort(glob(f"{fdir}/{target}*info*.pk"))
    masks = []
    for fsi in fs_info_files:
        with open(fsi, "rb") as handle:
            i = pickle.load(handle)
            mask = i["mask"]
            masks.append(mask)

    # finally, load the rotations
    # then the rotations
    rot_files = np.sort(
        glob(f"{output_dir}/intermediate/bkg_subtraction/{target}*rotations*.npy")
    )
    rotations = [np.load(x) for x in rot_files if "bkg.npy" not in x]
    kept_rots = []
    for r, m in zip(rotations, masks):
        kept_rots.append(r[m == 1])

    create_filestructure(output_dir, "fits_files")
    for idx, im in enumerate(images):
        # recenter the image
        recentered_im = imshift(im, *find_max_loc(im, do_median=False))  # recenter

        hdu = fits.PrimaryHDU(data=recentered_im)
        cycle = image_files[idx].split("/")[-1].split("_")[-1].split(".")[0]
        hdu.header["cycle"] = cycle
        hdu.header["mean_rot"] = np.mean(kept_rots[idx])
        hdu.header["min_rot"] = np.min(kept_rots[idx])
        hdu.header["max_rot"] = np.max(kept_rots[idx])
        hdul = fits.HDUList([hdu])
        hdul.writeto(
            f"{output_dir}/intermediate/fits_files/{target}_selectedframes_{cycle}.fits",
            overwrite=True,
        )


if __name__ == "__main__":
    script, configfile = argv
    transfer_function(configfile)
    frame_selected_fits_files(configfile)
