import numpy as np


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
    phi = np.radians(pa)
    a = (x - alpha) * np.cos(phi) + (y - delta) * np.sin(phi)
    d = (x - alpha) * np.sin(phi) - (y - delta) * np.cos(phi)
    return f * np.exp(-4 * np.log(2) * (np.square(a / minor) + np.square(d / major)))
