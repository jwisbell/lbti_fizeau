# Script to estimate flux in some of the LBTI filters based on the calibrator star's temperature and associated wise fluxes
import numpy as np
import matplotlib.pyplot as plt
from astropy.modeling.models import BlackBody
from astropy import units as u
from astroquery.vizier import Vizier
import re
from pprint import pprint


def spec_type_to_temp(spec_type):
    """
    Converts a spectral type string (e.g., 'G2V', 'M0III', 'A0')
    to an effective temperature in Kelvin.
    """
    spec_type = spec_type.upper().strip()

    # Mapping for Main Sequence (Class V) stars
    # Data source: Pecaut & Mamajek (2013)
    temp_map_v = {
        "O": 40000,
        "B0": 31400,
        "B5": 15700,
        "A0": 9600,
        "A5": 8190,
        "F0": 7170,
        "F5": 6510,
        "G0": 5920,
        "G2": 5770,
        "G5": 5660,
        "K0": 5280,
        "K5": 4410,
        "M0": 3850,
        "M5": 3120,
        "M9": 2400,
    }

    # Mapping for Giant (Class III) stars
    temp_map_iii = {
        "G0": 5580,
        "G5": 5050,
        "K0": 4850,
        "K5": 3950,
        "M0": 3800,
        "M5": 3400,
    }

    # 1. Determine Luminosity Class (default to V if not specified)
    is_giant = "III" in spec_type and "V" not in spec_type
    ref_map = temp_map_iii if is_giant else temp_map_v

    # 2. Extract Letter and Subtype Number
    match = re.match(r"([OBAFGKM])(\d\.?\d?)?", spec_type)
    if not match:
        return 5778.0  # Default to Solar temp if parsing fails

    letter = match.group(1)
    sub_type = float(match.group(2)) if match.group(2) else 0.0

    # 3. Interpolate within the spectral class
    # Get all anchors for the current letter (e.g., all 'G' anchors)
    anchors = sorted(
        [
            (float(k[1:] if len(k) > 1 else 0), v)
            for k, v in ref_map.items()
            if k.startswith(letter)
        ]
    )

    if not anchors:
        # Fallback to just the letter if specific subtype isn't mapped
        return ref_map.get(letter, 5778.0)

    x_points = [a[0] for a in anchors]
    y_points = [a[1] for a in anchors]

    # If only one anchor exists for that letter (like 'O'), return it
    if len(x_points) == 1:
        return float(y_points[0])

    # Interpolate for the specific subtype (e.g., 2.0 for G2)
    return float(np.interp(sub_type, x_points, y_points))


def mag_to_Jy(mag, mag_err=None, filter_name="w1"):
    """
    Converts magnitude (and optionally error) to Jansky.

    Formula: Flux = ZeroPoint * 10**(-mag / 2.5)
    Error: sigma_F = 0.921 * Flux * sigma_m
    """
    # Zero points in Janskys (Jy)
    zero_points = {
        "w1": 309.54,  # WISE 3.4um
        "w2": 171.787,  # WISE 4.6um
        "w3": 31.674,  # WISE 12um
        "2mass_j": 1594.0,
        "2mass_h": 1024.0,
        "2mass_k": 666.7,  # 2MASS Ks (2.15um)
    }

    if filter_name not in zero_points:
        raise ValueError(f"Filter {filter_name} not found.")

    f_nu = zero_points[filter_name] * 10 ** (-mag / 2.5)

    if mag_err is not None:
        # Propagation: dF = F * ln(10)/2.5 * dm
        f_nu_err = np.log(10) / 2.5 * f_nu * mag_err
        return f_nu * u.Jy, f_nu_err * u.Jy

    return f_nu * u.Jy, 0 * u.Jy


def get_2mass_mags(star_name):
    """Query Vizier for 2MASS (J, H, K) magnitudes and errors."""
    # II/246 is the 2MASS All-Sky Catalog of Point Sources
    # e_Xmag are the standard error columns
    v = Vizier(columns=["Jmag", "e_Jmag", "Hmag", "e_Hmag", "Kmag", "e_Kmag", "_r"])
    result = v.query_object(star_name, catalog="II/246/out")

    if not result or len(result) == 0:
        raise ValueError(f"No 2MASS data found for {star_name}")

    table = result[0]
    table.sort("_r")
    closest_star = table[0]

    mags = {
        "2mass_j": closest_star["Jmag"],
        "2mass_j_err": closest_star["e_Jmag"],
        "2mass_h": closest_star["Hmag"],
        "2mass_h_err": closest_star["e_Hmag"],
        "2mass_k": closest_star["Kmag"],
        "2mass_k_err": closest_star["e_Kmag"],
        "dist_arcsec": closest_star["_r"] * 3600,
    }
    return mags


def get_wise_mags(star_name):
    """Query Vizier for AllWISE magnitudes and errors."""
    # II/328 is the AllWISE Data Release
    v = Vizier(
        columns=["W1mag", "e_W1mag", "W2mag", "e_W2mag", "W3mag", "e_W3mag", "_r"]
    )
    result = v.query_object(star_name, catalog="II/328/allwise")

    if not result or len(result) == 0:
        raise ValueError(f"No WISE data found for {star_name}")

    table = result[0]
    table.sort("_r")
    closest_star = table[0]

    mags = {
        "w1": closest_star["W1mag"],
        "w1_err": closest_star["e_W1mag"],
        "w2": closest_star["W2mag"],
        "w2_err": closest_star["e_W2mag"],
        "w3": closest_star["W3mag"],
        "w3_err": closest_star["e_W3mag"],
        "dist_arcsec": closest_star["_r"] * 3600,
    }
    return mags


def main(star_name="Vega", spectral_type: str = "G"):
    temperature = spec_type_to_temp(spectral_type.upper())
    # 1. Look up star in wise catalog by name to get fluxes
    wise_mags = get_wise_mags(star_name)
    _2mass_mags = get_2mass_mags(star_name)

    # 2. Load Filter information
    # LBTI filter edges in micron
    lbti_filters = {
        "fe-ii": [1.63, 1.66],  # micron
        "std-l": [3.41, 3.99],  # micron
        "std-m": [4.60, 4.97],
        "W08699-9_122": [8.13, 9.35],
    }

    # WISE filter edges in micron
    wise_filters = {
        "w1": [2.8, 3.9],  # micron (Centered ~3.35)
        "w2": [4.1, 5.2],  # micron (Centered ~4.60)
        "w3": [7.5, 16.5],  # micron (Centered ~11.56)
    }

    # 2MASS K-short (Ks) filter edges
    _2mass_filters = {
        "2mass_k": [1.95, 2.36]  # micron (Centered ~2.15)
    }

    mapped_names = {
        "std-l": "w1",
        "std-m": "w2",
        "W08699-9_122": "w3",
        "fe-ii": "2mass_k",
    }

    wise_fluxes = {}
    wise_errs = {}
    for key in wise_filters:
        flux, err = mag_to_Jy(wise_mags[key], wise_mags[key + "_err"], key)
        wise_fluxes[key] = flux
        wise_errs[key] = err

    _2mass_fluxes = {}
    _2mass_errs = {}
    for key in _2mass_filters:
        flux, err = mag_to_Jy(_2mass_mags[key], _2mass_mags[key + "_err"], key)
        _2mass_fluxes[key] = flux
        _2mass_errs[key] = err

    all_fluxes = {k: v for k, v in wise_fluxes.items()}
    all_errs = {k: v for k, v in wise_errs.items()}
    for k, v in _2mass_fluxes.items():
        all_fluxes[k] = v
        all_errs[k] = _2mass_errs[k]

    all_filters = {k: wise_filters[k] for k in wise_fluxes.keys()}
    for k in _2mass_fluxes.keys():
        all_filters[k] = _2mass_filters[k]

    pprint(wise_mags)

    # 3. Make blackbody using known calibrator spectral type (ie temeprature)
    bb = BlackBody(temperature=temperature * u.K)

    wav_range = np.linspace(1, 12, 500) * u.micron

    plt.figure(figsize=(10, 6))

    print(f"\n--- Flux Estimates for {star_name} ---")

    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]
    results = {}
    result_errs = {}

    # 4. Scale the blackbody so that it matches the WISE flux
    for i, (name, bounds) in enumerate(lbti_filters.items()):
        mid_wav = np.mean(bounds) * u.micron

        mapped_name = mapped_names[name]  # pick the closest filter

        scaling_factor = all_fluxes[mapped_name] / bb(
            np.mean(all_filters[mapped_name]) * u.micron
        )
        relerr = all_errs[mapped_name] / all_fluxes[mapped_name]

        flux_curve = bb(wav_range) * scaling_factor

        est_flux = bb(mid_wav) * scaling_factor
        est_unc = relerr * est_flux
        results[name] = est_flux.to(u.Jy)
        result_errs[name] = est_unc.to(u.Jy)

        plt.plot(
            wav_range,
            flux_curve.value,
            color=colors[i],
            label=f"Scaled BB ({temperature}K, {mapped_name})",
        )
        # plt.fill_between(
        #     wav_range,
        #     flux_curve.value * (scaling_factor - scaling_factor * relerr).value,
        #     flux_curve.value * (scaling_factor + scaling_factor * relerr).value,
        #     color="gray",
        #     alpha=0.25,
        # )
        # Plot the boxcar filters
        plt.axvspan(bounds[0], bounds[1], alpha=0.3, color=colors[i], label=f"{name}")
        plt.errorbar(
            mid_wav.value,
            est_flux.value,
            yerr=est_unc.value,
            ls="none",
            capsize=0.25,
            color=colors[i],
            zorder=5,
            marker="s",
        )

        print(
            f"{name:15}: {results[name].value:.4f} +/- {result_errs[name].value:0.4f} Jy"
        )

    plt.xlabel("Wavelength (microns)")
    plt.ylabel("Flux Density (Jy)")
    plt.title(f"LBTI Filter Flux Estimation: {star_name}")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()

    # 5. Return the LBTI filter fluxes
    return results


if __name__ == "__main__":
    main("HD19305", "M0V")
