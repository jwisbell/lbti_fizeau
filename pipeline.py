# package imports
from sys import argv
import json

# pipeline imports
from utils import create_filestructure
from util_logger import Logger
from do_background_subtraction import do_bkg_subtraction
from do_frame_selection import do_frame_selection
from do_image_corotation import do_image_corotation


PROCESS_NAME = "pipeline"


if __name__ == "__main__":
    script, configfile = argv

    if configfile is None:
        print("No config file specified. Please specify a config file")
        exit()

    with open(configfile, "r") as inputfile:
        configdata = json.load(inputfile)

    target = configdata["target"]
    data_dir = configdata["data_dir"]
    output_dir = configdata["output_dir"]

    logger = Logger(output_dir, target)
    logger.info(PROCESS_NAME, "Config file loaded")
    logger.info(PROCESS_NAME, configdata)
    logger.info(
        PROCESS_NAME, f"Starting processing of {target} in directory {data_dir}"
    )
    logger.info(PROCESS_NAME, f"Results will be put into directory {output_dir}")

    create_filestructure(output_dir, "bkg_subtraction")
    logger.info(PROCESS_NAME, "Starting process `do_bkg_subtraction`")
    if not do_bkg_subtraction(configdata, logger):
        exit()
    logger.info(PROCESS_NAME, "Process `do_bkg_subtraction` finished successfully")

    create_filestructure(output_dir, "frame_selection")
    logger.info(PROCESS_NAME, "Starting process `do_frame_selection`")
    if not do_frame_selection(configdata, logger):
        exit()
    logger.info(PROCESS_NAME, "Process `do_frame_selection` finished successfully")

    create_filestructure(output_dir, "corotate")
    logger.info(PROCESS_NAME, "Starting process `do_image_corotation`")
    if not do_image_corotation(configdata, logger):
        exit()
    logger.info(PROCESS_NAME, "Process `do_image_corotation` finished successfully")
