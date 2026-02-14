#!/usr/bin/env python3

from sys import argv

from lizard_calibrate import calibrate
from lizard_deconvolve import deconvolve
from lizard_reduce import reduce, singledish
from interactive_deconv import clean
import argparse


def main():
    parser = argparse.ArgumentParser(
        description="LIZARD: An LBTI imaging (AO and Fizeau) processing pipeline."
    )

    # Create the 'subparser' object for commands
    subparsers = parser.add_subparsers(
        dest="command", required=True, help="Processing step to run"
    )

    # Command: reduce
    reduce_parser = subparsers.add_parser("reduce", help="Reduce the data")
    reduce_parser.add_argument("config", help="Path to the reduction config file")
    reduce_parser.add_argument(
        "--skip_bkg",
        action="store_true",
        help="Skip the background subtraction to use a previous result",
    )

    # Command: calibrate
    calibrate_parser = subparsers.add_parser(
        "calibrate", help="Run calibration on the dataset using a known calibrator"
    )
    calibrate_parser.add_argument("config", help="Path to the calibration config file")

    # Command: deconvolve
    deconv_parser = subparsers.add_parser("deconvolve", help="PSF deconvolution")
    deconv_parser.add_argument("config", help="Path to the calibration config file")
    # Optional Flags
    deconv_parser.add_argument(
        "--skip_clean", action="store_true", help="Skip the CLEANing phase"
    )
    deconv_parser.add_argument(
        "--skip_rl", action="store_true", help="Skip Richardson-Lucy deconvolution"
    )
    deconv_parser.add_argument(
        "--skip_pixelfit", action="store_true", help="Skip the pixel fitting routine"
    )

    # Command: clean
    clean_parser = subparsers.add_parser("clean", help="Interactive cleaning session")
    clean_parser.add_argument("config", help="Path to the calibration config file")

    # Command: singledish
    sd_parser = subparsers.add_parser("singledish", help="Single-dish reduction")
    sd_parser.add_argument("config", help="Path to the reduction config file")

    args = parser.parse_args()

    # Map commands to their respective functions
    tasks = {
        "reduce": reduce,
        "calibrate": calibrate,
        "deconvolve": deconvolve,
        "clean": clean,
        "singledish": singledish,
    }

    # Execute the function associated with the command
    if args.command == "deconvolve":
        deconvolve(
            args.config,
            skip_clean=args.skip_clean,
            skip_rl=args.skip_rl,
            skip_pixelfit=args.skip_pixelfit,
        )
    elif args.command == "reduce":
        reduce(args.config, skip_bkg=args.skip_bkg)
    else:
        tasks[args.command](args.config)


if __name__ == "__main__":
    main()
