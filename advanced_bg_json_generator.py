#!/usr/bin/env python3
"""
Generate a four-profile Si-background JSON for 2d_map_fitting_2.py.

The output contains polarization-wise 1D spectra:
  isotropic_xx, isotropic_yx, angular_xx, angular_yx

The strong Si line near 520 cm-1 / 64.4 meV is fitted only to identify the
line position and a suitable exclusion window. The profile values inside that
window are replaced by a smooth continuum bridge so the JSON does not contain
another source for the 64.4 meV peak. 2d_map_fitting_2.py then fits that
B1g-like peak explicitly while using the cleaned profiles for the rest of the
background shape.
"""

import argparse
import json
import os

import numpy as np
import pandas as pd
from scipy.interpolate import splrep
from scipy.linalg import lstsq
from scipy.optimize import curve_fit, minimize_scalar
from scipy.signal import savgol_filter


CM1_PER_MEV = 8.065544


def _abs2(x):
    return np.abs(x) ** 2


def lorentzian(x, area, x0, gamma):
    g = np.abs(gamma) + 1e-12
    return area * (1.0 / np.pi) * (g / ((x - x0) ** 2 + g ** 2))


def b1g_basis(theta_deg, config, phi_deg):
    th = np.deg2rad(theta_deg - phi_deg)
    if config == "parallel":
        return _abs2(np.sin(2 * th))
    return _abs2(np.cos(2 * th))


def normalized_basis(theta_deg, config, phi_deg):
    raw = b1g_basis(theta_deg, config, phi_deg)
    mean = np.mean(raw)
    if np.abs(mean) < 1e-12:
        return raw
    return raw / mean


def load_map(path):
    df = pd.read_csv(path, index_col=0)
    angles = df.index.astype(float).to_numpy()
    shifts = df.columns.astype(float).to_numpy()
    values = df.to_numpy(dtype=float)

    order_x = np.argsort(shifts)
    shifts = shifts[order_x]
    values = values[:, order_x]

    order_a = np.argsort(angles)
    angles = angles[order_a]
    values = values[order_a, :]
    return angles, shifts, values


def common_grid(xx_path, yx_path):
    angles_xx, shifts_xx, xx = load_map(xx_path)
    angles_yx, shifts_yx, yx = load_map(yx_path)
    if len(angles_xx) != len(angles_yx) or np.max(np.abs(angles_xx - angles_yx)) > 1e-8:
        raise ValueError("XX and YX CSVs must have the same angle index.")
    if len(shifts_xx) != len(shifts_yx) or np.max(np.abs(shifts_xx - shifts_yx)) > 1e-8:
        raise ValueError("XX and YX CSVs must have the same shift columns.")
    return angles_xx, shifts_xx, xx, yx


def solve_pol_profiles(angles, values, config, phi_deg):
    basis = normalized_basis(angles, config, phi_deg)
    a = np.column_stack([np.ones_like(angles), basis])
    coeffs, _, _, _ = lstsq(a, values)
    return coeffs[0], coeffs[1]


def residual_for_phi(angles, xx, yx, phi_deg, weights):
    iso_xx, ang_xx = solve_pol_profiles(angles, xx, "parallel", phi_deg)
    iso_yx, ang_yx = solve_pol_profiles(angles, yx, "cross", phi_deg)
    rec_xx = np.ones((len(angles), 1)) @ iso_xx[None, :] + normalized_basis(angles, "parallel", phi_deg)[:, None] @ ang_xx[None, :]
    rec_yx = np.ones((len(angles), 1)) @ iso_yx[None, :] + normalized_basis(angles, "cross", phi_deg)[:, None] @ ang_yx[None, :]
    return np.sum((xx - rec_xx) ** 2 * weights[None, :]) + np.sum((yx - rec_yx) ** 2 * weights[None, :])


def fit_phi(angles, shifts, xx, yx, phi_initial, peak_center, peak_half_width, peak_weight):
    weights = np.ones_like(shifts, dtype=float)
    weights[np.abs(shifts - peak_center) <= peak_half_width] = peak_weight

    def objective(phi):
        return residual_for_phi(angles, xx, yx, phi, weights)

    lo = max(-180.0, phi_initial - 90.0)
    hi = min(180.0, phi_initial + 90.0)
    result = minimize_scalar(objective, bounds=(lo, hi), method="bounded", options={"xatol": 1e-4})
    return float(result.x) if result.success else float(phi_initial)


def smooth_profile(profile, window, polyorder):
    if window <= 1:
        return profile
    w = int(window)
    if w % 2 == 0:
        w += 1
    if w >= len(profile):
        w = len(profile) - 1 if len(profile) % 2 == 0 else len(profile)
    if w <= polyorder or w < 3:
        return profile
    return savgol_filter(profile, w, polyorder)


def fit_peak_for_window(x, y, center_guess, half_width, gamma_guess):
    fit_mask = np.abs(x - center_guess) <= half_width
    if np.count_nonzero(fit_mask) < 7:
        return None

    xw = x[fit_mask]
    yw = y[fit_mask]
    x_ref = center_guess

    def model(xv, offset, slope, area, x0, gamma):
        return offset + slope * (xv - x_ref) + lorentzian(xv, area, x0, gamma)

    area_guess = (np.nanmax(yw) - np.nanmin(yw)) * max(gamma_guess, 1e-6) * np.pi
    p0 = [np.nanmedian(yw), 0.0, area_guess, center_guess, gamma_guess]
    lower = [-np.inf, -np.inf, -np.inf, center_guess - half_width, max(gamma_guess * 0.05, 1e-6)]
    upper = [np.inf, np.inf, np.inf, center_guess + half_width, max(half_width * 2.0, gamma_guess * 10.0)]

    try:
        popt, _ = curve_fit(model, xw, yw, p0=p0, bounds=(lower, upper), maxfev=10000)
    except Exception as exc:
        return {"success": False, "message": str(exc)}

    fit_y = model(xw, *popt)
    rmse = float(np.sqrt(np.mean((yw - fit_y) ** 2)))
    dyn = float(np.nanmax(yw) - np.nanmin(yw))
    return {
        "success": True,
        "offset": float(popt[0]),
        "slope": float(popt[1]),
        "area": float(popt[2]),
        "center": float(popt[3]),
        "gamma": float(popt[4]),
        "rmse": rmse,
        "relative_rmse": rmse / max(dyn, 1e-12),
    }


def bridge_peak_region(x, y, center, half_width):
    remove_mask = np.abs(x - center) <= half_width
    if np.count_nonzero(remove_mask) == 0:
        return y.copy()

    left_mask = x < center - half_width
    right_mask = x > center + half_width
    if np.count_nonzero(left_mask) == 0 or np.count_nonzero(right_mask) == 0:
        return y.copy()

    left_edge = np.max(x[left_mask])
    right_edge = np.min(x[right_mask])

    band_width = max(0.25 * half_width, np.median(np.diff(x)) * 2.0)
    left_band = (x >= left_edge - band_width) & (x <= left_edge)
    right_band = (x >= right_edge) & (x <= right_edge + band_width)

    y_left = float(np.nanmedian(y[left_band])) if np.count_nonzero(left_band) else float(y[left_mask][-1])
    y_right = float(np.nanmedian(y[right_band])) if np.count_nonzero(right_band) else float(y[right_mask][0])

    cleaned = y.copy()
    cleaned[remove_mask] = np.interp(x[remove_mask], [left_edge, right_edge], [y_left, y_right])
    return cleaned


def profile_rmse(a, b, mask):
    if np.count_nonzero(mask) == 0:
        return np.nan
    return float(np.sqrt(np.mean((a[mask] - b[mask]) ** 2)))


def reconstruct_map(angles, iso_profile, angular_profile, config, phi):
    return (
        np.ones((len(angles), 1)) @ iso_profile[None, :]
        + normalized_basis(angles, config, phi)[:, None] @ angular_profile[None, :]
    )


def map_quality(raw, rec, mask):
    if np.count_nonzero(mask) == 0:
        return {"rmse": np.nan, "relative_rmse": np.nan}
    diff = raw[:, mask] - rec[:, mask]
    rmse = float(np.sqrt(np.mean(diff ** 2)))
    scale = float(np.nanmax(raw[:, mask]) - np.nanmin(raw[:, mask]))
    return {"rmse": rmse, "relative_rmse": rmse / max(scale, 1e-12)}


def spline_dict(x, y, smoothing):
    t, c, k = splrep(x, y, s=smoothing)
    return {"t": t.tolist(), "c": c.tolist(), "k": int(k)}


def default_peak_center(unit):
    return 520.0 if unit == "cm-1" else 64.4


def main():
    parser = argparse.ArgumentParser(description="Generate four-profile advanced Si BG JSON.")
    parser.add_argument("--xx", required=True, help="2D XX background CSV, angle index by shift columns.")
    parser.add_argument("--yx", required=True, help="2D YX background CSV, angle index by shift columns.")
    parser.add_argument("-o", "--output", required=True, help="Output JSON path.")
    parser.add_argument("--unit", choices=["cm-1", "meV"], default="meV")
    parser.add_argument("--phi-initial", type=float, default=0.0)
    parser.add_argument("--no-fit-phi", action="store_true")
    parser.add_argument("--smooth-window", type=int, default=11)
    parser.add_argument("--smooth-poly", type=int, default=3)
    parser.add_argument("--spline-smoothing", type=float, default=0.0)
    parser.add_argument("--peak-center", type=float, default=None)
    parser.add_argument("--peak-half-width", type=float, default=None)
    parser.add_argument("--remove-half-width", type=float, default=None)
    parser.add_argument("--peak-fit-weight", type=float, default=0.02)
    parser.add_argument("--gamma-guess", type=float, default=None)
    parser.add_argument("--export-profiles-csv", default=None, help="Optional CSV path for cleaned and raw extracted profiles.")
    args = parser.parse_args()

    angles, shifts, xx, yx = common_grid(args.xx, args.yx)
    peak_center = args.peak_center if args.peak_center is not None else default_peak_center(args.unit)
    peak_half_width = args.peak_half_width
    if peak_half_width is None:
        peak_half_width = 20.0 if args.unit == "cm-1" else 2.5
    gamma_guess = args.gamma_guess
    if gamma_guess is None:
        gamma_guess = 2.0 if args.unit == "cm-1" else 2.0 / CM1_PER_MEV

    phi = args.phi_initial if args.no_fit_phi else fit_phi(
        angles, shifts, xx, yx, args.phi_initial, peak_center, peak_half_width, args.peak_fit_weight
    )

    iso_xx, ang_xx = solve_pol_profiles(angles, xx, "parallel", phi)
    iso_yx, ang_yx = solve_pol_profiles(angles, yx, "cross", phi)

    raw_profiles = {
        "isotropic_xx": smooth_profile(iso_xx, args.smooth_window, args.smooth_poly),
        "isotropic_yx": smooth_profile(iso_yx, args.smooth_window, args.smooth_poly),
        "angular_xx": smooth_profile(ang_xx, args.smooth_window, args.smooth_poly),
        "angular_yx": smooth_profile(ang_yx, args.smooth_window, args.smooth_poly),
    }

    peak_fits = {}
    centers = []
    gammas = []
    for name, profile in raw_profiles.items():
        fit = fit_peak_for_window(shifts, profile, peak_center, peak_half_width, gamma_guess)
        peak_fits[name] = fit
        if fit and fit.get("success"):
            centers.append(fit["center"])
            gammas.append(abs(fit["gamma"]))

    if centers:
        shared_center = float(np.median(centers))
    else:
        shared_center = float(peak_center)

    if args.remove_half_width is not None:
        remove_half_width = float(args.remove_half_width)
    elif gammas:
        remove_half_width = float(max(peak_half_width, 6.0 * np.median(gammas)))
    else:
        remove_half_width = float(peak_half_width)

    profiles = {
        name: bridge_peak_region(shifts, profile, shared_center, remove_half_width)
        for name, profile in raw_profiles.items()
    }

    outside_peak = np.abs(shifts - shared_center) > remove_half_width
    inside_peak = ~outside_peak
    profile_quality = {}
    for name in profiles:
        profile_quality[name] = {
            "outside_peak_rmse_raw_vs_cleaned": profile_rmse(raw_profiles[name], profiles[name], outside_peak),
            "inside_peak_rmse_raw_vs_cleaned": profile_rmse(raw_profiles[name], profiles[name], inside_peak),
            "fit_success": bool(peak_fits[name] and peak_fits[name].get("success")),
        }

    rec_raw_xx = reconstruct_map(angles, raw_profiles["isotropic_xx"], raw_profiles["angular_xx"], "parallel", phi)
    rec_raw_yx = reconstruct_map(angles, raw_profiles["isotropic_yx"], raw_profiles["angular_yx"], "cross", phi)
    rec_clean_xx = reconstruct_map(angles, profiles["isotropic_xx"], profiles["angular_xx"], "parallel", phi)
    rec_clean_yx = reconstruct_map(angles, profiles["isotropic_yx"], profiles["angular_yx"], "cross", phi)
    reconstruction_quality = {
        "raw_profiles_vs_input_xx_all": map_quality(xx, rec_raw_xx, np.ones_like(shifts, dtype=bool)),
        "raw_profiles_vs_input_yx_all": map_quality(yx, rec_raw_yx, np.ones_like(shifts, dtype=bool)),
        "cleaned_profiles_vs_input_xx_outside_removed_peak": map_quality(xx, rec_clean_xx, outside_peak),
        "cleaned_profiles_vs_input_yx_outside_removed_peak": map_quality(yx, rec_clean_yx, outside_peak),
    }

    out = {
        "schema": "advanced_si_bg_v2",
        "unit": args.unit,
        "source": {"xx": os.path.abspath(args.xx), "yx": os.path.abspath(args.yx)},
        "b1g": {
            "phi_deg": float(phi),
            "basis_normalization": "mean_per_polarization",
            "fit_phi": not args.no_fit_phi,
        },
        "removed_peak": {
            "nominal_center": float(peak_center),
            "fit_half_width": float(peak_half_width),
            "shared_center": shared_center,
            "remove_half_width": remove_half_width,
            "gamma_guess": float(gamma_guess),
            "method": "lorentzian_center_then_bridge_interpolation",
            "profiles": peak_fits,
        },
        "quality": {
            "all_peak_fits_successful": all(v["fit_success"] for v in profile_quality.values()),
            "profile_checks": profile_quality,
            "reconstruction_checks": reconstruction_quality,
        },
        "components": {
            name: spline_dict(shifts, profile, args.spline_smoothing)
            for name, profile in profiles.items()
        },
    }

    with open(args.output, "w") as f:
        json.dump(out, f, indent=2)

    if args.export_profiles_csv:
        export = {"Shift": shifts, "removed_peak_mask": inside_peak.astype(int)}
        for name in ["isotropic_xx", "isotropic_yx", "angular_xx", "angular_yx"]:
            export[f"{name}_raw"] = raw_profiles[name]
            export[f"{name}_cleaned"] = profiles[name]
        pd.DataFrame(export).to_csv(args.export_profiles_csv, index=False)

    print(f"Wrote {args.output}")
    print(f"B1g phi = {phi:.4f} deg")
    print(f"Peak fit success = {out['quality']['all_peak_fits_successful']}")
    print(f"Removed peak by continuum bridge: center={shared_center:.4f} {args.unit}, half_width={remove_half_width:.4f}")
    print(
        "Cleaned outside-peak relative RMSE: "
        f"XX={reconstruction_quality['cleaned_profiles_vs_input_xx_outside_removed_peak']['relative_rmse']:.4g}, "
        f"YX={reconstruction_quality['cleaned_profiles_vs_input_yx_outside_removed_peak']['relative_rmse']:.4g}"
    )
    if args.export_profiles_csv:
        print(f"Wrote profile diagnostic CSV {args.export_profiles_csv}")


if __name__ == "__main__":
    main()
