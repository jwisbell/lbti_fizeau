#!/usr/bin/env python3

from sys import argv

from lizard_calibrate import calibrate
from lizard_reduce import reduce, singledish
from interactive_deconv import clean

if __name__ == "__main__":
    # TODO: allow multiple config files
    sys, process, config = argv

    err_str = "Help: correct usage is `lizard.py __process-name__ /path/to/config_file.json` \nProcess name must be one of (reduce|calibrate)"
    if process is None or config is None:
        print(err_str)

    match process:
        case "reduce":
            reduce(config)
        case "calibrate":
            calibrate(config)
        case "clean":
            clean(config)
        case "singledish":
            singledish(config)
        case _:
            print(err_str)
