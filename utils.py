"""
utils -- LIZARD pipeline

Contains various utility functions that are re-used by multiple scripts.

"""

import numpy as np
from os import mkdir, path


def argmax2d(arr):
    # wrapper for numpy argmax to give x,y coordinates of the max in a 2d array
    m = np.nanmax(arr)
    s = np.where(arr == m)
    return s[1], s[0]


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
