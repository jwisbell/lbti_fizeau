# package imports
from sys import argv
import json

# pipeline imports
from do_deconvolution import do_deconvolution
from utils import create_filestructure
from util_logger import Logger
from make_emperical_psf_estimate import do_estimate_final_psf

PROCESS_NAME = "calibration"


if __name__ == "__main__":
    script, configfile = argv
    # configfilename = "./nod_config_ngc4151.json"

    if configfile is None:
        print("No config file specified. Please specify a config file")
        exit()

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
    logger.info(PROCESS_NAME, "Config file loaded")
    logger.info(PROCESS_NAME, configdata)

    logger.info(
        PROCESS_NAME,
        f"Starting calibration of {target_configdata["target"]} with {calib_configdata["target"]}",
    )
    logger.info(PROCESS_NAME, f"Results will be put into directory {output_dir}")

    create_filestructure(output_dir, "estimate_final_psf", prefix="calibrated")
    logger.info(PROCESS_NAME, "Starting process `estimate_final_psf`")
    if not do_estimate_final_psf(
        configdata, target_configdata, calib_configdata, logger
    ):
        logger.error(PROCESS_NAME, "Process `estimate_final_psf` failed")
        exit()
    logger.info(PROCESS_NAME, "Process `estimate_final_psf` finished successfully")

    create_filestructure(output_dir, "deconvolution", prefix="calibrated")
    logger.info(PROCESS_NAME, "Starting process `do_deconvolution`")
    if not do_deconvolution(configdata, target_configdata, calib_configdata, logger):
        logger.error(PROCESS_NAME, "Process `do_deconvolution` failed")
        exit()
    logger.info(PROCESS_NAME, "Process `do_deconvolution` finished successfully")
