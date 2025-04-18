"""
lizard_calibrate -- LIZARD Pipeline
Author: Jacob Isbell

Wrapper to do various calibration steps. Requires a calibration config file (see template).
Each step requires a science target AND a calibrator to previously have been reduced with lizard_reduce.py
"""

# package imports
from sys import argv
import json

# pipeline imports

from utils.utils import create_filestructure
from utils.util_logger import Logger
from calibration_steps.do_deconvolution import do_deconvolution

PROCESS_NAME = "calibration"


def wrap_do_convolution(
    configdata: dict,
    target_configdata: dict,
    calib_configdata: dict,
    output_dir: str,
    logger: Logger,
):
    create_filestructure(output_dir, "deconvolution", prefix="calibrated")
    logger.create_log_file("deconvolution")
    logger.info(PROCESS_NAME, "Starting process `do_deconvolution`")
    if not do_deconvolution(configdata, target_configdata, calib_configdata, logger):
        logger.error(PROCESS_NAME, "Process `do_deconvolution` failed")
        exit()
    logger.info(PROCESS_NAME, "Process `do_deconvolution` finished successfully")


def clean(configfile):
    with open(configfile, "r") as inputfile:
        configdata = json.load(inputfile)

    target_configfile = configdata["target_config"]
    calib_configfile = configdata["calib_config"]

    output_dir = configdata["output_dir"]

    if target_configfile is None:
        print("Target config filename is missing")
    with open(target_configfile) as targetcfg:
        target_configdata = json.load(targetcfg)

    if calib_configfile is None:
        print("Calib config filename is missing")
    with open(calib_configfile) as calibcfg:
        calib_configdata = json.load(calibcfg)

    logger = Logger(
        output_dir, f"{target_configdata['target']}_with_{calib_configdata['target']}"
    )
    logger.create_log_file(PROCESS_NAME)
    logger.info(PROCESS_NAME, "Config file loaded")
    logger.info(PROCESS_NAME, configdata)

    logger.info(
        PROCESS_NAME,
        f"Starting calibration of {target_configdata["target"]} with {calib_configdata["target"]}",
    )
    logger.info(PROCESS_NAME, f"Results will be put into directory {output_dir}")

    # TODO: rotation video within convolution script
    wrap_do_convolution(
        configdata, target_configdata, calib_configdata, output_dir, logger
    )


if __name__ == "__main__":
    script, configfile = argv

    if configfile is None:
        print("No config file specified. Please specify a config file")
        exit()

    clean(configfile)
