"""
utils -- LIZARD pipeline

Contains various utility functions that are re-used by multiple scripts.

"""

import numpy as np
from scipy.ndimage import median_filter, vectorized_filter
from astropy.io import fits
from os import mkdir, path


def argmax2d(arr):
    # wrapper for numpy argmax to give x,y coordinates of the max in a 2d array
    m = np.nanmax(arr)
    s = np.where(arr == m)
    try:
        return s[1][0], s[0][0]
    except IndexError:
        # return the center of the image
        return arr.shape[0] // 2, arr.shape[1] // 2


def nanmedian_filter(arr, kernel_size=3):
    return vectorized_filter(arr, function=np.nanmedian, size=kernel_size)


def imshift(im, y, x):
    # roll the im along each axis so that peak is at x,y
    wx = im.shape[1] // 2
    wy = im.shape[0] // 2
    temp_im = np.roll(im, wx - x, axis=1)
    return np.roll(temp_im, wy - y, axis=0)


def find_max_loc(im, do_median=False):
    # find the x,y coords of peak of 2d array
    # w = im.shape[0]
    # temp_im = np.copy(im)[w//2-w//4:w//2+w//4,w//2-w//4:w//2+w//4]
    temp_im = np.copy(im)
    if do_median:
        temp_im = median_filter(im, 3)
    idx = np.argmax(np.abs(temp_im))
    # idx  = np.argmax(im )
    y, x = np.unravel_index(idx, im.shape)
    # x, y = argmax2d(np.abs(temp_im))
    return y, x


def angle_mean(arr):
    # take the mean of the angles, using complex numbers to account for wrapping
    vals = np.exp(1j * np.radians(arr))
    mean_angle = np.angle(np.mean(vals), deg=True)
    return mean_angle


def gauss(x, y, alpha, delta, major, minor, pa, f):
    # returns a 2d gaussian distribution
    phi = np.radians(pa)
    a = (x - alpha) * np.cos(phi) + (y - delta) * np.sin(phi)
    d = (x - alpha) * np.sin(phi) - (y - delta) * np.cos(phi)
    return f * np.exp(-4 * np.log(2) * (np.square(a / minor) + np.square(d / major)))


def create_filestructure(output_dir, process, prefix="intermediate"):
    # see if the intermediate products directory exists inside the output directory
    if not path.isdir(f"{output_dir}/{prefix}/"):
        mkdir(f"{output_dir}/{prefix}/")

    # see if the intermediate products directory exists inside the output directory
    if not path.isdir(f"{output_dir}/plots/"):
        mkdir(f"{output_dir}/plots/")

    # see if the intermediate products directory exists inside the output directory
    if not path.isdir(f"{output_dir}/{prefix}/{process}"):
        mkdir(f"{output_dir}/{prefix}/{process}")

    # see if the intermediate products directory exists inside the output directory
    if not path.isdir(f"{output_dir}/plots/{process}"):
        mkdir(f"{output_dir}/plots/{process}")


def write_to_fits(im, fname):
    hdu = fits.PrimaryHDU(im)
    hdu.header["fluxunit"] = "mJy/px"
    hdul = fits.HDUList([hdu])
    hdul.writeto(fname, overwrite=True)
