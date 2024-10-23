import numpy as np
from scipy.ndimage import rotate
import matplotlib.pyplot as plt
from matplotlib.colors import PowerNorm

from util_logger import Logger

PROCESS_NAME = "estimate_final_psf"
logger = Logger("./")


def _mk_emperical_psf(
    targetname: str,
    obsdate: str,
    target_rotations_fname: str,
    psfname: str,
    psf_calib_fname: str,
    output_dir: str,
) -> bool:
    try:
        psf_percentiles = np.load(psf_calib_fname)
        psf_im = psf_percentiles  # [0]
        # psf_err = psf_percentiles[1]
    except FileNotFoundError:
        logger.info(PROCESS_NAME, f"File {psf_calib_fname} not found")
        return False

    psf_im -= np.mean(psf_im)
    psf_im /= np.max(psf_im)

    try:
        rotations = np.load(target_rotations_fname)
    except FileNotFoundError:
        logger.info(PROCESS_NAME, f"File {target_rotations_fname} not found")
        return False

    rotated_psfs = [
        rotate(psf_im, -pa, reshape=False, mode="nearest") for pa in rotations
    ]

    psf_estimate = np.mean(rotated_psfs, 0)
    # psf_estimate /= np.max(psf_estimate)

    output_path = f"{output_dir}/calibrated/{PROCESS_NAME}/{psfname}_for_{targetname}_{obsdate}.npy"
    logger.info(PROCESS_NAME, f"PSF estimate created and saved to {output_path}")

    np.save(output_path, psf_estimate)

    _, ((ax, bx), (cx, dx)) = plt.subplots(2, 2, figsize=(6, 6))

    ax.plot(np.max(psf_estimate, 0))
    cx.imshow(psf_estimate, origin="lower", norm=PowerNorm(0.5))
    dx.plot(np.max(psf_estimate, 1), range(psf_estimate.shape[0]))
    bx.axis("off")
    plt.savefig(
        f"{output_dir}/plots/{PROCESS_NAME}/psf_estimate_{psfname}_for_{targetname}_{obsdate}.png"
    )
    plt.close()

    return True


def do_estimate_final_psf(
    wrapperconfig: dict, targetconfig: dict, calibconfig: dict, mylogger: Logger
):
    global logger
    logger = mylogger

    try:
        output_dir = wrapperconfig["output_dir"]

        targetname = targetconfig["target"]
        psfname = calibconfig["target"]

        obsdate = targetconfig["obsdate"]

        target_outdir = targetconfig["output_dir"]
        calib_outdir = calibconfig["output_dir"]

    except KeyError as e:
        logger.error(
            PROCESS_NAME, f"One or more config files is improperly formatted {e}"
        )
        return False

    target_rotations_fname = (
        f"{target_outdir}/intermediate/corotate/{targetname}_included_rotations.npy"
    )
    psf_calib_fname = (
        f"{calib_outdir}/intermediate/corotate/{psfname}_unrotated_stacked_psf.npy"
    )

    return _mk_emperical_psf(
        targetname,
        obsdate,
        target_rotations_fname,
        psfname,
        psf_calib_fname,
        output_dir,
    )
