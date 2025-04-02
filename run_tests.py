from calibration_steps.do_deconvolution import clean_test
from calibration_steps.bad_pixel_correction import bpm_test

if __name__ == "__main__":
    total_passed = 0
    total_passed += clean_test()
    total_passed += bpm_test()
