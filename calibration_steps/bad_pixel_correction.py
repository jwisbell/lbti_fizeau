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

from utils.util_logger import Logger
from utils.utils import imshift

PROCESS_NAME = "bad_pixel_correction"
logger = Logger("./")


def identify_bad_pixels(image, min_sigma=10):
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

    # TODO: attempt convolution
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
