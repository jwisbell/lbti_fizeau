"""
bad_pixel_correction -- LIZARD Pipeline
Author: Jacob Isbell

Functions to make an estimate of the bad pixels and remove them. This is
especially important for single dish observations or observations with a
small number of exposures.
"""

import numpy as np
from scipy.ndimage import rotate
import matplotlib.pyplot as plt
from matplotlib.colors import PowerNorm
from astropy.io import fits

from utils.util_logger import Logger
from utils.utils import imshift

PROCESS_NAME = "bad_pixel_correction"
logger = Logger("./")


def identify_bad_pixels(image, min_sigma=10, niter=1):
    """
    This is basically sigma clipping pixels that are very different from
    their neighbors and setting them to the value of their neighbors
    """
    corrected_image = np.copy(image)
    bad_pixel_map = np.zeros(image.shape)

    # 1. Naive approach is to scan over pixels and find any which
    # are > min_sigma*std different from the mean value of those pixels
    # also find pixels at 0 and above the central value (assuming that
    # the object has been found and centered in previous steps).

    kernel = np.zeros(image.shape)
    w = kernel.shape[0]
    kernel[w // 2 - 1 : w // 2 + 2, w // 2 - 1 : w // 2 + 2] = 1
    kernel[w // 2, w // 2] = 0
    kernel[kernel == 0] = np.nan

    for y in range(len(image)):
        for x in range(len(image[0])):
            # slide the kernel
            shifted_kernel = np.copy(kernel)
            shifted_kernel = imshift(
                shifted_kernel, y, x
            )  # move the psf to the "peak" location
            std = np.nanstd(shifted_kernel * image)
            mean = np.nanmean(shifted_kernel * image)
            if y == 30 and x == 40:
                print(std, mean, y, x, image[y, x])
            if abs(image[y, x] - mean) >= min_sigma * std:
                bad_pixel_map[y, x] = 1
                corrected_image[y, x] = mean

    return corrected_image, bad_pixel_map


def bpm_test():
    fake_image = np.zeros((50, 50)) + np.random.randn(50, 50) * 1e-6
    fake_image[10, 10] = 1
    fake_image[30, 40] = -1
    _, bpm = identify_bad_pixels(fake_image, 10)
    assert bpm[10, 10] == 1, "Bad pixel map failed at 10,10"

    assert bpm[30, 40] == 1, "Bad pixel map failed at 30,40"

    assert np.sum(bpm) == 2, f"Bad pixel map failed at sum. Got {np.sum(bpm)}, want {2}"

    return 1


def correct_image_after_bpm(masked_image, skip=False):
    # find each bad pixel and replace with the median of its neighbors
    # what do we do about the edges? (skip?)

    # zero_locations = []
    # for i in range(len(masked_image)):
    #     for j in range(len(masked_image[i])):
    #         if masked_image[i, j] == 0:
    #             zero_locations.append((i, j))
    if skip:
        return masked_image

    zero_locations = np.where(masked_image == 0)

    corrected_image = np.copy(masked_image)

    for i in range(len(zero_locations[0])):
        # p1 | p2 | p3
        # p4 | 0  | p6
        # p7 | p8 | p9
        loc = (zero_locations[0][i], zero_locations[1][i])

        # vals = []
        # for x in range(max(0, loc[0] - 1), min(loc[0] + 2, masked_image.shape[0])):
        #     for y in range(max(0, loc[1] - 1), min(loc[1] + 2, masked_image.shape[1])):
        #         if x == loc[0] and y == loc[1]:
        #             continue
        #
        #         if masked_image[x, y] == 0:
        #             continue
        #
        #         vals.append(masked_image[x, y])

        vals = masked_image[
            max(0, loc[0] - 1) : min(loc[0] + 2, masked_image.shape[0]),
            max(0, loc[1] - 1) : min(loc[1] + 2, masked_image.shape[0]),
        ]
        vals[vals == 0] = np.nan
        corrected_image[loc[0], loc[1]] = np.nanmedian(vals)
    return corrected_image


def apply_bad_pixel_mask(bpm, bkg_sub_ims, skip=False):
    # actually apply the bad pixel mask
    # input npm is a binary mask (True means good pixel, False means bad pixel)
    # ensure that all bad pixels are marked as np.nan for easier processing later
    if skip:
        return bkg_sub_ims

    masked_ims = [bpm * x for x in bkg_sub_ims]
    masked_ims_test = np.multiply(bpm, bkg_sub_ims)
    print(len(masked_ims), masked_ims_test.shape)
    return masked_ims


def load_bpm(im_hdr):
    # loads the bad pixel map
    # TODO: load this from a specified location
    hdu = fits.open("./bpm.fits")
    bpm = hdu[0].data[4].astype("bool")
    bpm = ~bpm

    # bpm[~bpm] = np.nan

    # print(bpm.shape)  # should be 2048x2048
    try:
        # get the proper readout region with desired shape
        x1 = int(im_hdr["SUBSECX1"]) - 1
        x2 = int(im_hdr["SUBSECX2"]) - 1
        y1 = int(im_hdr["SUBSECY1"]) - 1
        y2 = int(im_hdr["SUBSECY2"]) - 1

        bpm_window = bpm[y1 : y2 + 1, x1 : x2 + 1]
        return bpm_window
    except KeyError:
        return bpm


if __name__ == "__main__":
    fake_image = np.zeros((50, 50)) + np.random.randn(50, 50) * 1e-6
    fake_image[10, 10] = 1
    fake_image[30, 40] = -1
    cim, bpm = identify_bad_pixels(fake_image, 10)

    fig, (ax, bx, cx) = plt.subplots(1, 3)
    ax.imshow(bpm, origin="lower")
    bx.imshow(cim, origin="lower")
    cx.imshow(fake_image, origin="lower")
    plt.show()
