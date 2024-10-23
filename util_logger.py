"""
Logger for LBTI Fizeau Pipeline
Author: Jacob Isbell (jwisbell.astro@gmail.com)
Version: 0.1.0
Date: Oct 17 2024

Contains the class which handles logging. To be loaded by whichever process calls individual pipeline steps.
"""

from datetime import datetime

pipelinename = "OCATILLO"


class Logger:
    def __init__(self, output_dir, targname="", level=0) -> None:
        self.output_dir = output_dir
        self.level = level
        self.target = targname

    def create_log_file(self, process):
        now = datetime.now()

        date_time = now.strftime("%m/%d/%Y, %H:%M:%S")

        message_string = (
            f"{pipelinename}\nRunning process: {process} at {date_time}\n" + "=" * 16
        )
        with open(
            f"{self.output_dir}/{process}{self.target}.log", "w"
        ) as logfile:  # open in append mode
            logfile.write(f"{message_string} \n")

    def info(self, process: str, message: str) -> None:
        message_string = f"INFO: {message}"
        self._do_log(message_string, process, 0)

    def warn(self, process: str, message: str) -> None:
        message_string = f"Warning: {message}"
        self._do_log(message_string, process, 1)

    def error(self, process: str, message: str) -> None:
        message_string = f"ERROR: {message}"

        self._do_log(message_string, process, 2)

    def _do_log(self, message_string, process, cutoff_level) -> None:
        now = datetime.now()

        date_time = now.strftime("%m/%d/%Y, %H:%M:%S")
        if self.level <= cutoff_level:
            print(f"[{date_time}]  {message_string}")

        with open(
            f"{self.output_dir}/{process}{self.target}.log", "a"
        ) as logfile:  # open in append mode
            logfile.write(f"[{date_time}]  {message_string} \n")


def do_test():
    logger = Logger("./")
    process = "testing"
    logger.create_log_file(process)

    logger.info(process, "testing")
    logger.warn(process, "test warning")
    logger.error(process, "test error")


if __name__ == "__main__":
    do_test()
