"""
lizard_from_corrected.py
========================
A condensed LIZARD-compatible pipeline that starts from pre-processed frames:
  - cropped
  - background-subtracted
  - bad-pixel corrected

Skips: raw I/O, background subtraction, bad pixel correction, cropping.

Runs:
  1. Frame selection  (lucky fringe selection by Strehl/peak proxy)
  2. Image stacking + corotation  (North-up, derotated stack)
  3. PSF estimation  (from calibrator stacked image)
  4. Flux calibration  (scale science image to physical units)
  5. CLEAN deconvolution  (Hogbom CLEAN)
  6. Richardson-Lucy deconvolution

Usage
-----
    python lizard_from_corrected.py config.json

Config file format (JSON)
-------------------------
{
    "target":           "NGC 5506",
    "science_frames_dir":  "/path/to/science/corrected_frames/",
    "calib_frames_dir":    "/path/to/calibrator/corrected_frames/",
    "output_dir":          "/path/to/output/",
    "obs_wavelength_um":   3.8,
    "pixel_scale_arcsec":  0.0107,
    "pa_keyword":          "ROTOFF",
    "cutoff_fraction":     0.5,
    "calib_flux_Jy":       1.23,
    "calib_flux_err_Jy":   0.05,
    "clean_niter":         100000,
    "clean_gain":          0.001,
    "clean_phat":          0.0,
    "rl_niter":            32,
    "rl_eps":              0.001
}

Frame file format
-----------------
Each corrected frame should be a single 2-D FITS file (or .npy array) named
consistently inside science_frames_dir / calib_frames_dir.
The parallactic angle used for corotation is read from the FITS header keyword
specified by "pa_keyword". If the files are .npy (no header), set
"pa_keyword": null and supply a separate "pa_file" (a .npy array of angles
in degrees, same length as the number of frames).

Dependencies: numpy, scipy, matplotlib
(Deliberately avoids astropy so it works in the same environment as your
bad-pixel detection scripts.)
"""

import os
import sys
import json
import glob
import struct
import logging
import numpy as np
from scipy.ndimage import rotate, shift
from scipy.signal import fftconvolve
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ─────────────────────────────────────────────────────────────────────────────
# Logging
# ─────────────────────────────────────────────────────────────────────────────

def make_logger(output_dir, name):
    os.makedirs(output_dir, exist_ok=True)
    log_path = os.path.join(output_dir, f"{name}.log")
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    fh = logging.FileHandler(log_path)
    fh.setLevel(logging.DEBUG)
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
    fh.setFormatter(fmt); ch.setFormatter(fmt)
    logger.addHandler(fh); logger.addHandler(ch)
    return logger


# ─────────────────────────────────────────────────────────────────────────────
# I/O helpers
# ─────────────────────────────────────────────────────────────────────────────

def _parse_fits_header(raw):
    """Return (header_dict, data_offset) from raw FITS bytes."""
    header = {}
    pos = 0
    while True:
        block = raw[pos:pos + 2880]
        pos += 2880
        for i in range(36):
            card = block[i * 80:(i + 1) * 80].decode("ascii", errors="replace")
            key = card[:8].strip()
            if key == "END":
                return header, pos
            if "=" in card:
                kw, val = card.split("=", 1)
                val = val.split("/")[0].strip().strip("'").strip()
                header[kw.strip()] = val
    return header, pos


def load_fits(path):
    """Load a 2-D FITS image. Returns (img_array, header_dict)."""
    with open(path, "rb") as f:
        raw = f.read()
    header, offset = _parse_fits_header(raw)
    naxis1 = int(header.get("NAXIS1", 0))
    naxis2 = int(header.get("NAXIS2", 0))
    bitpix = int(header.get("BITPIX", -32))
    bzero  = float(header.get("BZERO",  0))
    bscale = float(header.get("BSCALE", 1))
    dtype_map = {16: ">i2", 32: ">i4", -32: ">f4", -64: ">f8", 8: ">u1"}
    dt = np.dtype(dtype_map[bitpix])
    n_bytes = naxis1 * naxis2 * abs(bitpix) // 8
    img = np.frombuffer(raw[offset:offset + n_bytes], dtype=dt).reshape(naxis2, naxis1).astype(float)
    img = img * bscale + bzero
    return img, header


def save_fits(path, img):
    """Write a 2-D float64 array as a minimal FITS file."""
    ny, nx = img.shape
    arr = img.astype(">f8")
    # Build header (must be multiples of 2880 bytes, 80-char cards)
    cards = [
        f"SIMPLE  =                    T / Written by lizard_from_corrected",
        f"BITPIX  =                  -64 / 64-bit float",
        f"NAXIS   =                    2 / Number of axes",
        f"NAXIS1  = {nx:20d} / Width",
        f"NAXIS2  = {ny:20d} / Height",
        "END",
    ]
    header_str = "".join(f"{c:<80}" for c in cards)
    pad = (-len(header_str)) % 2880
    header_bytes = (header_str + " " * pad).encode("ascii")
    data_bytes = arr.tobytes()
    pad2 = (-len(data_bytes)) % 2880
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "wb") as f:
        f.write(header_bytes)
        f.write(data_bytes)
        f.write(b"\x00" * pad2)


def load_frame(path):
    """Load a single .fits or .npy frame as a 2-D float array."""
    if path.endswith(".npy"):
        img = np.load(path).astype(float)
    else:
        img, _ = load_fits(path)
    while img.ndim > 2:
        img = np.mean(img, axis=0)
    return img


def get_pa_from_header(path, pa_keyword):
    """Extract parallactic angle from FITS header."""
    if path.endswith(".npy"):
        return None
    with open(path, "rb") as f:
        raw = f.read(2880 * 10)  # first 10 blocks usually enough
    header, _ = _parse_fits_header(raw + b" " * 2880)
    val = header.get(pa_keyword, None)
    return float(val) if val is not None else 0.0


# ─────────────────────────────────────────────────────────────────────────────
# Step 1: Frame selection
# ─────────────────────────────────────────────────────────────────────────────

def frame_selection(frames_dir, cutoff_fraction, logger):
    """
    Lucky-fringe frame selection.
    Ranks frames by peak value (proxy for Strehl / fringe quality).
    Returns sorted list of (path, peak_value) keeping top cutoff_fraction.
    """
    logger.info(f"Frame selection: scanning {frames_dir}")
    patterns = ["*.fits", "*.fit", "*.npy"]
    paths = []
    for pat in patterns:
        paths.extend(glob.glob(os.path.join(frames_dir, pat)))
    paths.sort()

    if len(paths) == 0:
        raise FileNotFoundError(f"No FITS/npy frames found in {frames_dir}")

    logger.info(f"  Found {len(paths)} frames")

    scores = []
    for p in paths:
        try:
            img = load_frame(p)
            score = float(np.nanmax(img))
            scores.append((p, score))
        except Exception as e:
            logger.warning(f"  Skipping {p}: {e}")

    # Sort descending by peak
    scores.sort(key=lambda x: x[1], reverse=True)
    n_keep = max(1, int(np.ceil(len(scores) * cutoff_fraction)))
    selected = scores[:n_keep]

    logger.info(f"  Keeping top {cutoff_fraction*100:.0f}% = {n_keep}/{len(scores)} frames")
    logger.info(f"  Peak range kept: {selected[-1][1]:.2f} – {selected[0][1]:.2f}")
    return selected


# ─────────────────────────────────────────────────────────────────────────────
# Step 2: Stacking + corotation
# ─────────────────────────────────────────────────────────────────────────────

def centroid(img):
    """Intensity-weighted centroid."""
    img = np.clip(img, 0, None)
    total = img.sum()
    if total == 0:
        return img.shape[1] / 2, img.shape[0] / 2
    yy, xx = np.mgrid[:img.shape[0], :img.shape[1]]
    cx = (xx * img).sum() / total
    cy = (yy * img).sum() / total
    return cx, cy


def stack_and_corotate(selected_frames, pa_keyword, pa_file, logger):
    """
    Align frames to common centroid, apply parallactic angle rotation,
    and stack (mean combine).

    pa_keyword : FITS header keyword for parallactic angle (str or None)
    pa_file    : path to .npy array of PAs if pa_keyword is None
    """
    logger.info("Stacking and corotating frames")

    # Load PA array if provided
    pa_array = None
    if pa_file is not None and os.path.exists(pa_file):
        pa_array = np.load(pa_file)
        logger.info(f"  Loaded {len(pa_array)} PA values from {pa_file}")

    stack = []
    for i, (path, _) in enumerate(selected_frames):
        img = load_frame(path)

        # Centroid alignment
        cx, cy = centroid(img)
        ref_cx, ref_cy = img.shape[1] / 2.0, img.shape[0] / 2.0
        img = shift(img, [ref_cy - cy, ref_cx - cx], order=3, mode="constant", cval=0)

        # Parallactic angle rotation
        if pa_keyword is not None:
            pa = get_pa_from_header(path, pa_keyword)
        elif pa_array is not None and i < len(pa_array):
            pa = float(pa_array[i])
        else:
            pa = 0.0

        if pa != 0.0:
            img = rotate(img, -pa, reshape=False, order=3, mode="constant", cval=0)

        stack.append(img)

    stacked = np.mean(stack, axis=0)
    logger.info(f"  Stacked {len(stack)} frames → shape {stacked.shape}")
    return stacked


# ─────────────────────────────────────────────────────────────────────────────
# Step 3: PSF estimation
# ─────────────────────────────────────────────────────────────────────────────

def estimate_psf(calib_stacked, logger):
    """
    The stacked calibrator image IS the empirical PSF.
    Normalise to unit peak and centre it.
    """
    logger.info("Estimating PSF from calibrator stack")
    psf = calib_stacked.copy()
    psf = np.clip(psf, 0, None)
    psf /= psf.max()
    logger.info(f"  PSF shape: {psf.shape}, peak normalised to 1.0")
    return psf


# ─────────────────────────────────────────────────────────────────────────────
# Step 4: Flux calibration
# ─────────────────────────────────────────────────────────────────────────────

def flux_calibration(science_stacked, calib_stacked, calib_flux_Jy, calib_flux_err_Jy, logger):
    """
    Scale science image to physical flux units (Jy/pixel).
    Calibration factor = calib_flux_Jy / sum(calib_stacked)
    """
    logger.info("Flux calibration")
    calib_sum = np.nansum(calib_stacked)
    if calib_sum <= 0:
        raise ValueError("Calibrator image has non-positive total flux — check inputs")
    cal_factor = calib_flux_Jy / calib_sum
    cal_factor_err = calib_flux_err_Jy / calib_sum
    science_calibrated = science_stacked * cal_factor
    logger.info(f"  Calibrator total counts: {calib_sum:.4e}")
    logger.info(f"  Calibration factor: {cal_factor:.4e} Jy/count")
    logger.info(f"  Science peak flux: {np.nanmax(science_calibrated):.4e} Jy/pixel")
    return science_calibrated, cal_factor, cal_factor_err


# ─────────────────────────────────────────────────────────────────────────────
# Step 5: CLEAN deconvolution (Hogbom)
# ─────────────────────────────────────────────────────────────────────────────

def hogbom_clean(dirty, psf, niter, gain, phat, logger):
    """
    Hogbom CLEAN algorithm.
    dirty : 2-D science image (flux calibrated)
    psf   : 2-D PSF (normalised to peak=1)
    niter : number of CLEAN iterations
    gain  : loop gain (fraction of peak subtracted per iteration)
    phat  : pointy-hat parameter (boosts resolution for point-like sources)
    Returns (clean_image, residual_map, clean_components)
    """
    logger.info(f"CLEAN deconvolution: niter={niter}, gain={gain}, phat={phat}")

    ny, nx = dirty.shape
    residual = dirty.copy()
    components = np.zeros_like(dirty)

    # Centre PSF
    psf_cy = psf.shape[0] // 2
    psf_cx = psf.shape[1] // 2

    # Build "pointed" PSF: phat blends PSF with a delta function
    psf_use = (1 - phat) * psf + phat * (psf * 0)
    psf_use[psf_cy, psf_cx] += phat

    niter = int(niter)
    log_interval = max(1, niter // 10)

    for it in range(niter):
        # Find peak in residual
        peak_val = np.nanmax(np.abs(residual))
        if peak_val <= 0:
            logger.info(f"  CLEAN converged at iteration {it}")
            break
        py, px = np.unravel_index(np.argmax(np.abs(residual)), residual.shape)

        # Subtract scaled PSF centred on peak
        delta = gain * residual[py, px]
        components[py, px] += delta

        # Compute PSF region to subtract
        y0 = py - psf_cy; y1 = y0 + psf.shape[0]
        x0 = px - psf_cx; x1 = x0 + psf.shape[1]

        # Clip to image bounds
        iy0 = max(0, y0); iy1 = min(ny, y1)
        ix0 = max(0, x0); ix1 = min(nx, x1)
        py0 = iy0 - y0;   py1 = py0 + (iy1 - iy0)
        px0 = ix0 - x0;   px1 = px0 + (ix1 - ix0)

        residual[iy0:iy1, ix0:ix1] -= delta * psf_use[py0:py1, px0:px1]

        if (it + 1) % log_interval == 0:
            logger.info(f"  CLEAN iter {it+1}/{niter}, peak residual={peak_val:.4e}")

    # Restore: convolve components with PSF (acts as clean beam)
    clean_image = fftconvolve(components, psf, mode="same") + residual
    logger.info(f"  CLEAN complete. Residual peak: {np.nanmax(np.abs(residual)):.4e}")
    return clean_image, residual, components


# ─────────────────────────────────────────────────────────────────────────────
# Step 6: Richardson-Lucy deconvolution
# ─────────────────────────────────────────────────────────────────────────────

def richardson_lucy(dirty, psf, niter, eps, logger):
    """
    Richardson-Lucy deconvolution.
    eps : small regularisation constant to avoid division by zero
    """
    logger.info(f"Richardson-Lucy deconvolution: niter={niter}, eps={eps}")

    psf_norm = psf / psf.sum()
    psf_mirror = psf_norm[::-1, ::-1]

    # Initialise estimate as a copy of the (positive-clipped) dirty image
    estimate = np.clip(dirty.copy(), eps, None)

    for it in range(niter):
        conv = fftconvolve(estimate, psf_norm, mode="same")
        conv = np.clip(conv, eps, None)
        ratio = dirty / conv
        correction = fftconvolve(ratio, psf_mirror, mode="same")
        estimate = estimate * correction
        estimate = np.clip(estimate, 0, None)

        if (it + 1) % max(1, niter // 5) == 0:
            logger.info(f"  R-L iter {it+1}/{niter}")

    logger.info("  R-L complete")
    return estimate


# ─────────────────────────────────────────────────────────────────────────────
# Diagnostic plots
# ─────────────────────────────────────────────────────────────────────────────

def save_diagnostic_plot(output_dir, target, science_cal, psf,
                          clean_img, rl_img, pixel_scale):
    from matplotlib.colors import PowerNorm

    fig, axes = plt.subplots(1, 4, figsize=(14, 3.5))
    fig.suptitle(target, fontsize=10)

    titles = ["Science (calibrated)", "PSF", "CLEAN", "R-L"]
    imgs   = [science_cal, psf, clean_img, rl_img]

    for ax, title, img in zip(axes, titles, imgs):
        vmax = 0.25 * np.nanmax(img)
        vmin = max(0, np.nanmedian(img))
        ax.imshow(img, origin="lower", cmap="inferno",
                  norm=PowerNorm(gamma=0.5, vmin=vmin, vmax=vmax))
        ax.set_title(title, fontsize=8)
        ax.axis("off")

    outpath = os.path.join(output_dir, f"{target.replace(' ','')}_summary.png")
    fig.savefig(outpath, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return outpath


# ─────────────────────────────────────────────────────────────────────────────
# Main pipeline
# ─────────────────────────────────────────────────────────────────────────────

def run_pipeline(configfile):
    with open(configfile) as f:
        cfg = json.load(f)

    target         = cfg["target"]
    sci_dir        = cfg["science_frames_dir"]
    cal_dir        = cfg["calib_frames_dir"]
    output_dir     = cfg["output_dir"]
    cutoff         = float(cfg.get("cutoff_fraction", 0.5))
    pa_keyword     = cfg.get("pa_keyword", None)
    pa_file        = cfg.get("pa_file", None)
    calib_flux     = float(cfg["calib_flux_Jy"])
    calib_flux_err = float(cfg["calib_flux_err_Jy"])
    clean_niter    = int(float(cfg.get("clean_niter", 1e5)))
    clean_gain     = float(cfg.get("clean_gain", 1e-3))
    clean_phat     = float(cfg.get("clean_phat", 0.0))
    rl_niter       = int(cfg.get("rl_niter", 32))
    rl_eps         = float(cfg.get("rl_eps", 1e-3))

    os.makedirs(output_dir, exist_ok=True)
    logger = make_logger(output_dir, target.replace(" ", "_"))
    logger.info(f"=== lizard_from_corrected: {target} ===")
    logger.info(f"Config: {cfg}")

    # ── 1. Frame selection ───────────────────────────────────────────────────
    sci_selected  = frame_selection(sci_dir, cutoff, logger)
    cal_selected  = frame_selection(cal_dir, cutoff, logger)

    # ── 2. Stack + corotate ──────────────────────────────────────────────────
    science_stacked = stack_and_corotate(sci_selected, pa_keyword, pa_file, logger)
    calib_stacked   = stack_and_corotate(cal_selected, pa_keyword, pa_file, logger)

    np.save(os.path.join(output_dir, f"{target.replace(' ','')}_stacked.npy"), science_stacked)
    np.save(os.path.join(output_dir, f"{target.replace(' ','')}_calib_stacked.npy"), calib_stacked)
    logger.info("Saved stacked science and calibrator images")

    # ── 3. PSF estimation ────────────────────────────────────────────────────
    psf = estimate_psf(calib_stacked, logger)
    np.save(os.path.join(output_dir, f"{target.replace(' ','')}_psf.npy"), psf)

    # ── 4. Flux calibration ──────────────────────────────────────────────────
    science_cal, cal_factor, cal_factor_err = flux_calibration(
        science_stacked, calib_stacked, calib_flux, calib_flux_err, logger
    )
    np.save(os.path.join(output_dir, f"{target.replace(' ','')}_flux_calibrated.npy"), science_cal)
    save_fits(os.path.join(output_dir, f"{target.replace(' ','')}_flux_calibrated.fits"), science_cal)

    # ── 5. CLEAN deconvolution ───────────────────────────────────────────────
    clean_img, residual, components = hogbom_clean(
        science_cal, psf, clean_niter, clean_gain, clean_phat, logger
    )
    np.save(os.path.join(output_dir, f"{target.replace(' ','')}_clean.npy"), clean_img)
    np.save(os.path.join(output_dir, f"{target.replace(' ','')}_clean_residual.npy"), residual)
    save_fits(os.path.join(output_dir, f"{target.replace(' ','')}_clean.fits"), clean_img)

    # ── 6. R-L deconvolution ─────────────────────────────────────────────────
    rl_img = richardson_lucy(science_cal, psf, rl_niter, rl_eps, logger)
    np.save(os.path.join(output_dir, f"{target.replace(' ','')}_rl.npy"), rl_img)
    save_fits(os.path.join(output_dir, f"{target.replace(' ','')}_rl.fits"), rl_img)

    # ── Diagnostic plot ──────────────────────────────────────────────────────
    plot_path = save_diagnostic_plot(output_dir, target, science_cal, psf,
                                     clean_img, rl_img,
                                     cfg.get("pixel_scale_arcsec", 0.0107))
    logger.info(f"Diagnostic plot saved: {plot_path}")

    logger.info("=== Pipeline complete ===")
    logger.info(f"Outputs in: {output_dir}")


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python lizard_from_corrected.py config.json")
        sys.exit(1)
    run_pipeline(sys.argv[1])
