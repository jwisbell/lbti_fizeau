"""
Docstring
"""

from util_logger import Logger

logger = Logger("./")


def do_flux_calibration(
    configdata: dict, target_configdata: dict, calib_configdata: dict, mylogger: Logger
) -> bool:
    """
    Docstring
    """
    global logger
    logger = mylogger

    # 0. Parse the config files

    # 1. Load the flux calibrator images and compute percentiles of the sum

    # 2. Load the science target images and compute percentiles of the sum

    # 3. Use the known calibrator flux and uncertainty to scale the target

    # 4. Save the flux calibrated target images and the scaling information

    return True
