from __future__ import annotations

import math
import os
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from matplotlib import colormaps
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.colors import to_hex
from matplotlib.figure import Figure
from matplotlib.lines import Line2D

from scipy.optimize import curve_fit, least_squares

import analysis
from data_structure import ExperimentSet, Run, RunType, cm1_to_unit, normalize_spectral_unit, unit_to_cm1


@dataclass
class RowPeakFit:
    target: float
    name: str
    rule: str
    angles: np.ndarray
    areas: np.ndarray
    centers: np.ndarray
    gammas: np.ndarray
    heights: np.ndarray
    config: str
    run_id: str
    run_label: str


@dataclass
class TensorFit:
    target: float
    name: str
    rule: str
    params: np.ndarray
    param_names: List[str]
    scale: float
    covariance: Optional[np.ndarray] = None
    param_std: Optional[np.ndarray] = None
    n_observations: int = 0
    degrees_of_freedom: int = 0
    residual_rmse: float = np.nan


@dataclass
class GlobalCenteredRatioBootstrap:
    global_ratio: float
    ratio_expression: str
    canonical_phi_deg: float
    reference_scale: float
    reference_phi_shift_deg: float
    reference_rmse: float
    ratio_std: float
    bootstrap_mean: float
    bootstrap_bias: float
    percentile_2_5: float
    percentile_97_5: float
    draws: np.ndarray
    attempted: int
    successful: int
    n_observations: int
    n_angle_clusters: int
    phase_boundary_hit: bool
    ratio_boundary_fraction: float


@dataclass
class _RowModelPeak:
    requested_target: float
    center: float
    name: str
    rule: str
    peak_state: Optional[Dict[str, Any]]
    gamma_guess: float
    gamma_lo: float
    gamma_hi: float
    output_index: Optional[int]


def oriented_2d(run: Run) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    return analysis.display_2d_from_run(run)


def infer_config(run: Run, fallback: str = "parallel") -> str:
    md = run.metadata or {}
    tokens = " ".join(str(md.get(k, "")) for k in ("pol", "nickname", "base", "stem")).lower()
    if any(t in tokens for t in ("yx", "xy", "cross", "rl", "lr")):
        return "cross"
    if any(t in tokens for t in ("xx", "yy", "parallel", "para", "rr", "ll")):
        return "parallel"
    return fallback


def find_fit_params_run(exp: ExperimentSet, run_ids: Sequence[str]) -> Optional[Run]:
    ids = set(run_ids)
    active_ids = []
    for run_id in run_ids:
        source = exp.get_run(str(run_id))
        active_id = (source.metadata or {}).get("active_fit_params_run_id") if source is not None else None
        if active_id:
            active_ids.append(str(active_id))
    if active_ids and len(set(active_ids)) == 1:
        active = exp.get_run(active_ids[0])
        if active is not None and active.run_type == RunType.FIT_PARAMS:
            source_ids = set((active.metadata or {}).get("source_run_ids", []))
            if not ids or ids.issubset(source_ids):
                return active
    for run in exp.runs.values():
        if run.run_type != RunType.FIT_PARAMS:
            continue
        source_ids = set(run.metadata.get("source_run_ids", []))
        if ids and ids.issubset(source_ids):
            return run
    for run in exp.runs.values():
        if run.run_type == RunType.FIT_PARAMS and run.metadata.get("fit_state"):
            source_ids = set(run.metadata.get("source_run_ids", []))
            if ids & source_ids:
                return run
    return None


def fit_state_for_runs(exp: ExperimentSet, runs: Sequence[Run]) -> Tuple[Optional[Dict[str, Any]], Optional[Run]]:
    params_run = find_fit_params_run(exp, [run.id for run in runs])
    if params_run is not None:
        return params_run.metadata.get("fit_state"), params_run
    for run in runs:
        state = (run.metadata or {}).get("fit_state") or (run.metadata or {}).get("map_fit_state")
        if state:
            return state, run
    return None, None


def runs_from_fit_selection(exp: ExperimentSet, run: Run) -> List[Run]:
    if run.run_type == RunType.FIT_PARAMS:
        source_ids = run.metadata.get("source_run_ids", [])
        out = [exp.get_run(rid) for rid in source_ids]
        return [r for r in out if r is not None and r.is_2d]
    return [run]


def match_peak_from_state(fit_state: Optional[Dict[str, Any]], target: float) -> Tuple[str, str, Optional[Dict[str, Any]]]:
    if not fit_state:
        return f"{target:.3g}", "D2h_Ag", None
    peaks = fit_state.get("peaks") or []
    best = None
    best_dist = math.inf
    for peak in peaks:
        try:
            center = _peak_center_cm1(peak, fit_state)
        except Exception:
            continue
        dist = abs(center - target)
        if dist < best_dist:
            best = peak
            best_dist = dist
    if best is None:
        return f"{target:.3g}", "D2h_Ag", None
    return str(best.get("name") or f"{target:.3g}"), str(best.get("rule") or "D2h_Ag"), best


def _fit_state_unit(fit_state: Optional[Dict[str, Any]]) -> str:
    if not fit_state:
        return "cm-1"
    return normalize_spectral_unit(fit_state.get("unit", "cm-1"), "cm-1")


def _state_value_to_cm1(value: float, fit_state: Optional[Dict[str, Any]]) -> float:
    return float(unit_to_cm1(np.asarray([float(value)], dtype=float), _fit_state_unit(fit_state))[0])


def _peak_center_cm1(peak: Dict[str, Any], fit_state: Optional[Dict[str, Any]]) -> float:
    value = float(peak.get("spec_params", {}).get("x0", [np.nan])[0])
    return _state_value_to_cm1(value, fit_state)


def _peak_gamma_cm1(peak: Optional[Dict[str, Any]], fit_state: Optional[Dict[str, Any]], x: np.ndarray, window: float) -> float:
    if peak:
        try:
            gamma = abs(float(peak.get("spec_params", {}).get("gamma", [np.nan])[0]))
            gamma = abs(_state_value_to_cm1(gamma, fit_state))
            if np.isfinite(gamma) and gamma > 0:
                return gamma
        except Exception:
            pass
    return _gamma_guess_from_peak(None, x, window)


def _state_fit_range_cm1(fit_state: Optional[Dict[str, Any]], shift: np.ndarray) -> Tuple[float, float]:
    finite_shift = np.asarray(shift, dtype=float)
    finite_shift = finite_shift[np.isfinite(finite_shift)]
    if finite_shift.size == 0:
        return -np.inf, np.inf
    default_lo = float(np.nanmin(finite_shift))
    default_hi = float(np.nanmax(finite_shift))
    if not fit_state:
        return default_lo, default_hi
    raw_lo = fit_state.get("x_min_limit", -np.inf)
    raw_hi = fit_state.get("x_max_limit", np.inf)
    try:
        lo = _state_value_to_cm1(float(raw_lo), fit_state) if np.isfinite(float(raw_lo)) else default_lo
    except Exception:
        lo = default_lo
    try:
        hi = _state_value_to_cm1(float(raw_hi), fit_state) if np.isfinite(float(raw_hi)) else default_hi
    except Exception:
        hi = default_hi
    lo, hi = sorted((lo, hi))
    if hi < default_lo or lo > default_hi:
        return default_lo, default_hi
    return max(lo, default_lo), min(hi, default_hi)


def _gamma_guess_from_peak(peak: Optional[Dict[str, Any]], x: np.ndarray, window: float) -> float:
    if peak:
        try:
            gamma = abs(float(peak.get("spec_params", {}).get("gamma", [np.nan])[0]))
            if np.isfinite(gamma) and gamma > 0:
                return gamma
        except Exception:
            pass
    if x.size > 1:
        step = float(np.nanmedian(np.abs(np.diff(np.sort(x)))))
        if np.isfinite(step) and step > 0:
            return max(step * 2.0, window / 10.0)
    return max(window / 10.0, 1e-3)


def _local_area_guess(x: np.ndarray, y: np.ndarray, center: float, gamma: float) -> float:
    finite = np.isfinite(x) & np.isfinite(y)
    if np.count_nonzero(finite) == 0:
        return 1e-12
    xf = x[finite]
    yf = y[finite]
    idx = int(np.argmin(np.abs(xf - center)))
    baseline = float(np.nanpercentile(yf, 20.0))
    height = max(float(yf[idx] - baseline), 0.0) if np.isfinite(yf[idx]) else 0.0
    return max(height * math.pi * max(abs(gamma), 1e-9), 1e-12)


def _area_guess_from_state(peak: _RowModelPeak, angle: float, config_name: str, x: np.ndarray, y: np.ndarray) -> float:
    state = peak.peak_state or {}
    rule_def = analysis.RULE_METADATA.get(peak.rule)
    if rule_def is not None:
        try:
            ang_p = [float(state.get("ang_params", {}).get(name, [np.nan])[0]) for name in rule_def["params"]]
            if all(np.isfinite(v) for v in ang_p):
                value = rule_def["func"](np.asarray([angle], dtype=float), config_name, *ang_p)[0]
                area = float(value) / (abs(peak.gamma_guess) + 1e-9)
                if np.isfinite(area) and area > 0:
                    return area
        except Exception:
            pass
    return _local_area_guess(x, y, peak.center, peak.gamma_guess)


def _build_row_model_peaks(
    shift: np.ndarray,
    targets: Sequence[float],
    fit_state: Optional[Dict[str, Any]],
    peak_window: float,
) -> Tuple[List[_RowModelPeak], List[int], np.ndarray]:
    targets = [float(t) for t in targets]
    if not targets:
        return [], [], np.zeros_like(shift, dtype=bool)

    model_peaks: List[_RowModelPeak] = []
    output_indices: List[int] = []
    fit_lo: float
    fit_hi: float

    if fit_state and fit_state.get("peaks"):
        fit_lo, fit_hi = _state_fit_range_cm1(fit_state, shift)
        state_peaks = []
        for peak in fit_state.get("peaks") or []:
            try:
                center = _peak_center_cm1(peak, fit_state)
            except Exception:
                continue
            if not np.isfinite(center):
                continue
            if fit_lo <= center <= fit_hi:
                state_peaks.append((center, peak))
        if not state_peaks:
            raise ValueError("Fit state contains no finite peaks inside the saved fitting range.")

        for center, peak in sorted(state_peaks, key=lambda item: item[0]):
            gamma = _peak_gamma_cm1(peak, fit_state, shift, peak_window)
            gamma_lo = max(gamma / 20.0, 1e-6)
            gamma_hi = max(gamma * 20.0, gamma_lo * 2.0)
            model_peaks.append(
                _RowModelPeak(
                    requested_target=center,
                    center=center,
                    name=str(peak.get("name") or f"{center:.3g}"),
                    rule=str(peak.get("rule") or "D2h_Ag"),
                    peak_state=peak,
                    gamma_guess=gamma,
                    gamma_lo=gamma_lo,
                    gamma_hi=gamma_hi,
                    output_index=None,
                )
            )

        centers = np.asarray([p.center for p in model_peaks], dtype=float)
        for out_idx, target in enumerate(targets):
            nearest = int(np.nanargmin(np.abs(centers - target)))
            model_peaks[nearest].output_index = out_idx
            output_indices.append(nearest)
        fit_mask = (shift >= fit_lo) & (shift <= fit_hi)
        return model_peaks, output_indices, fit_mask

    fit_mask = np.zeros_like(shift, dtype=bool)
    for out_idx, target in enumerate(targets):
        center = float(target)
        local_mask = (shift >= center - peak_window) & (shift <= center + peak_window)
        xw = shift[local_mask]
        gamma = _gamma_guess_from_peak(None, xw if xw.size else shift, peak_window)
        gamma_lo = max(gamma / 20.0, 1e-6)
        gamma_hi = max(gamma * 20.0, gamma_lo * 2.0)
        model_peaks.append(
            _RowModelPeak(
                requested_target=target,
                center=center,
                name=f"{target:.3g}",
                rule="D2h_Ag",
                peak_state=None,
                gamma_guess=gamma,
                gamma_lo=gamma_lo,
                gamma_hi=gamma_hi,
                output_index=out_idx,
            )
        )
        output_indices.append(len(model_peaks) - 1)
        fit_mask |= local_mask
    return model_peaks, output_indices, fit_mask


def fit_lorentzian_rows(
    run: Run,
    targets: Sequence[float],
    *,
    fit_state: Optional[Dict[str, Any]] = None,
    config: Optional[str] = None,
    peak_window: float = 8.0,
    center_window: float = 2.0,
) -> List[RowPeakFit]:
    shift, angles, intensity = oriented_2d(run)
    config_name = config or infer_config(run)
    targets = [float(t) for t in targets]
    peak_window = abs(float(peak_window))
    if peak_window <= 0:
        peak_window = 1e-6
    # ``center_window`` is kept in the public signature for CLI/GUI
    # compatibility. Centers are intentionally fixed in this simultaneous
    # row model so close peaks cannot trade position and area.
    _ = center_window

    model_peaks, output_indices, fit_mask = _build_row_model_peaks(shift, targets, fit_state, peak_window)
    if not model_peaks:
        return []
    x_fit_full = np.asarray(shift, dtype=float)[fit_mask]
    if x_fit_full.size < 5:
        raise ValueError("The simultaneous row-fit range has fewer than 5 x-points.")
    if x_fit_full.size < 2 + 2 * len(model_peaks):
        raise ValueError(
            f"The simultaneous row-fit range has {x_fit_full.size} x-points, "
            f"but {len(model_peaks)} peak(s) require at least {2 + 2 * len(model_peaks)}."
        )

    result_arrays = []
    for model_idx in output_indices:
        peak = model_peaks[model_idx]
        result_arrays.append(
            {
                "peak": peak,
                "areas": np.full(angles.shape, np.nan, dtype=float),
                "centers": np.full(angles.shape, peak.center, dtype=float),
                "gammas": np.full(angles.shape, np.nan, dtype=float),
                "heights": np.full(angles.shape, np.nan, dtype=float),
            }
        )

    output_for_model = {model_idx: out_idx for out_idx, model_idx in enumerate(output_indices)}
    x_ref = float(np.nanmedian(x_fit_full))
    x_centered_full = x_fit_full - x_ref

    def model(xc, offset, slope, *params):
        x_abs = xc + x_ref
        y = offset + slope * xc
        p_idx = 0
        for peak in model_peaks:
            area, gamma = params[p_idx], params[p_idx + 1]
            p_idx += 2
            y = y + area * analysis.lorentzian_normalized(x_abs, peak.center, gamma)
        return y

    for row_idx, angle in enumerate(angles):
        y_full = np.asarray(intensity[row_idx, :], dtype=float)[fit_mask]
        finite = np.isfinite(x_centered_full) & np.isfinite(y_full)
        if np.count_nonzero(finite) < max(5, 2 + 2 * len(model_peaks)):
            continue
        xf = x_centered_full[finite]
        yf = y_full[finite]
        offset0 = float(np.nanpercentile(yf, 20.0))
        slope0 = 0.0
        p0 = [offset0, slope0]
        lower = [-np.inf, -np.inf]
        upper = [np.inf, np.inf]
        for peak in model_peaks:
            p0.extend([_area_guess_from_state(peak, float(angle), config_name, x_fit_full[finite], yf), peak.gamma_guess])
            lower.extend([0.0, peak.gamma_lo])
            upper.extend([np.inf, peak.gamma_hi])
        try:
            popt, _ = curve_fit(model, xf, yf, p0=p0, bounds=(lower, upper), maxfev=10000)
        except Exception:
            continue

        p_idx = 2
        for model_idx, _peak in enumerate(model_peaks):
            area = float(popt[p_idx])
            gamma = abs(float(popt[p_idx + 1]))
            p_idx += 2
            out_idx = output_for_model.get(model_idx)
            if out_idx is None:
                continue
            result = result_arrays[out_idx]
            result["areas"][row_idx] = area
            result["gammas"][row_idx] = gamma
            result["heights"][row_idx] = area / (math.pi * gamma) if gamma > 0 else np.nan

    results: List[RowPeakFit] = []
    for result in result_arrays:
        peak = result["peak"]
        results.append(
            RowPeakFit(
                target=peak.requested_target,
                name=peak.name,
                rule=peak.rule,
                angles=np.asarray(angles, dtype=float),
                areas=result["areas"],
                centers=result["centers"],
                gammas=result["gammas"],
                heights=result["heights"],
                config=config_name,
                run_id=run.id,
                run_label=run.nickname,
            )
        )
    return results


def fit_lorentzian_rows_for_runs(
    runs: Sequence[Run],
    targets: Sequence[float],
    *,
    fit_state: Optional[Dict[str, Any]] = None,
    peak_window: float = 8.0,
    center_window: float = 2.0,
) -> List[List[RowPeakFit]]:
    """Fit every requested target once per run and group results by target."""
    targets = [float(t) for t in targets]
    row_fit_groups: List[List[RowPeakFit]] = [[] for _ in targets]
    for run_idx, run in enumerate(runs):
        fits = fit_lorentzian_rows(
            run,
            targets,
            fit_state=fit_state,
            config=infer_config(run, "parallel" if run_idx == 0 else "cross"),
            peak_window=peak_window,
            center_window=center_window,
        )
        for target_idx, rf in enumerate(fits):
            if target_idx < len(row_fit_groups):
                row_fit_groups[target_idx].append(rf)
    return row_fit_groups


def _initial_tensor_params(rule: str, peak_state: Optional[Dict[str, Any]], max_area: float) -> Tuple[List[str], List[float], List[float], List[float]]:
    meta = analysis.RULE_METADATA.get(rule) or analysis.RULE_METADATA["D2h_Ag"]
    names = list(meta["params"])
    p0: List[float] = []
    lo: List[float] = []
    hi: List[float] = []
    amp = math.sqrt(max(max_area, 1e-12))
    for name in names:
        val = None
        if peak_state:
            try:
                val = float(peak_state.get("ang_params", {}).get(name, [np.nan])[0])
            except Exception:
                val = None
        if val is None or not np.isfinite(val):
            val = 0.0 if name == "phi" else amp
        if name == "phi":
            val = float(np.clip(val, -180.0, 180.0))
            p0.append(val)
            lo.append(-180.0)
            hi.append(180.0)
        else:
            # The global map fit allows signed tensor coefficients. Keeping
            # those signs avoids scipy rejecting a perfectly valid saved fit
            # before optimization starts.
            p0.append(float(val))
            lo.append(-np.inf)
            hi.append(np.inf)
    return names, p0, lo, hi


def fit_tensor_for_peak(
    row_fits: Sequence[RowPeakFit],
    *,
    fit_state: Optional[Dict[str, Any]] = None,
) -> TensorFit:
    if not row_fits:
        raise ValueError("No row-fit data was provided.")
    target = float(row_fits[0].target)
    name, rule, peak_state = match_peak_from_state(fit_state, target)
    max_area = 0.0
    for rf in row_fits:
        valid = rf.areas[np.isfinite(rf.areas)]
        if valid.size:
            max_area = max(max_area, float(np.nanmax(valid)))
    param_names, p0, lo, hi = _initial_tensor_params(rule, peak_state, max_area)
    rule_func = (analysis.RULE_METADATA.get(rule) or analysis.RULE_METADATA["D2h_Ag"])["func"]

    theta_chunks = []
    config_chunks = []
    area_chunks = []
    for rf in row_fits:
        valid = np.isfinite(rf.angles) & np.isfinite(rf.areas)
        if valid.any():
            theta_chunks.append(rf.angles[valid])
            config_chunks.extend([rf.config] * int(valid.sum()))
            area_chunks.append(rf.areas[valid])

    if not theta_chunks:
        raise ValueError(f"No finite row-fit areas for target {target:g}.")

    theta = np.concatenate(theta_chunks)
    y = np.concatenate(area_chunks)
    configs = np.asarray(config_chunks, dtype=object)

    def model(_theta, *params):
        out = np.zeros_like(_theta, dtype=float)
        for cfg in ("parallel", "cross"):
            mask = configs == cfg
            if np.any(mask):
                out[mask] = rule_func(_theta[mask], cfg, *params)
        return out

    popt, pcov = curve_fit(model, theta, y, p0=p0, bounds=(lo, hi), maxfev=10000)
    fitted = np.asarray(model(theta, *popt), dtype=float)
    residual = np.asarray(y - fitted, dtype=float)
    residual_rmse = float(np.sqrt(np.mean(np.square(residual)))) if residual.size else np.nan
    covariance = np.asarray(pcov, dtype=float)
    diagonal = np.diag(covariance) if covariance.ndim == 2 else np.asarray([], dtype=float)
    param_std = np.full(len(popt), np.nan, dtype=float)
    count = min(len(param_std), len(diagonal))
    valid_variances = np.isfinite(diagonal[:count]) & (diagonal[:count] >= 0.0)
    param_std[:count] = np.where(valid_variances, np.sqrt(np.maximum(0.0, diagonal[:count])), np.nan)
    scale = float(np.nanmax(y)) if y.size else 1.0
    if not np.isfinite(scale) or scale <= 0:
        scale = 1.0
    return TensorFit(
        target=target,
        name=name,
        rule=rule,
        params=np.asarray(popt, dtype=float),
        param_names=param_names,
        scale=scale,
        covariance=covariance,
        param_std=param_std,
        n_observations=int(y.size),
        degrees_of_freedom=max(0, int(y.size - len(popt))),
        residual_rmse=residual_rmse,
    )


def canonical_d2h_ag_ratio(a: float, b: float, phi_deg: float) -> Tuple[float, str, float]:
    """Return the larger-magnitude tensor ratio and its equivalent local phase."""
    a = float(a)
    b = float(b)
    phi_deg = float(phi_deg)
    if not np.isfinite(a) or not np.isfinite(b) or not np.isfinite(phi_deg):
        raise ValueError("Global D2h Ag tensor parameters must be finite.")
    if abs(a) >= abs(b):
        if b == 0.0:
            raise ValueError("Cannot calculate a/b because global b is zero.")
        ratio = a / b
        expression = "a/b"
        canonical_phi = phi_deg
    else:
        if a == 0.0:
            raise ValueError("Cannot calculate b/a because global a is zero.")
        ratio = b / a
        expression = "b/a"
        canonical_phi = phi_deg + 90.0
    canonical_phi = float(((canonical_phi + 180.0) % 360.0) - 180.0)
    return float(ratio), expression, canonical_phi


def global_centered_ratio_bootstrap(
    row_fits: Sequence[RowPeakFit],
    *,
    global_a: float,
    global_b: float,
    global_phi_deg: float,
    replicates: int = 2000,
    seed: int = 20260723,
    phi_window_deg: float = 45.0,
) -> GlobalCenteredRatioBootstrap:
    """Estimate D2h Ag ratio spread around an unchanged global-fit center.

    The reference ratio is fixed to the canonical global value. A nuisance
    intensity scale and local phase are fitted to the row areas, after which a
    wild bootstrap applies one Rademacher weight per angle shared by all
    polarization observations at that angle.
    """
    if not row_fits:
        raise ValueError("No row-fit data was provided.")
    if int(replicates) < 2:
        raise ValueError("At least two bootstrap replicates are required.")
    if not np.isfinite(phi_window_deg) or phi_window_deg <= 0.0 or phi_window_deg > 90.0:
        raise ValueError("phi_window_deg must be in (0, 90].")

    theta_chunks: List[np.ndarray] = []
    area_chunks: List[np.ndarray] = []
    config_chunks: List[np.ndarray] = []
    for row_fit in row_fits:
        angles = np.asarray(row_fit.angles, dtype=float)
        areas = np.asarray(row_fit.areas, dtype=float)
        valid = np.isfinite(angles) & np.isfinite(areas)
        if not np.any(valid):
            continue
        theta_chunks.append(angles[valid])
        area_chunks.append(areas[valid])
        config_chunks.append(np.full(int(np.count_nonzero(valid)), str(row_fit.config), dtype=object))
    if not theta_chunks:
        raise ValueError("No finite row-fit angles and areas were available.")

    theta = np.concatenate(theta_chunks)
    areas = np.concatenate(area_chunks)
    configs = np.concatenate(config_chunks)
    global_ratio, expression, canonical_phi = canonical_d2h_ag_ratio(
        global_a,
        global_b,
        global_phi_deg,
    )
    if abs(global_ratio) < 1.0 - 1e-10:
        raise ValueError("Canonical global tensor ratio must have magnitude at least one.")

    angle_keys = np.round(theta, decimals=8)
    _unique_angles, cluster_index = np.unique(angle_keys, return_inverse=True)
    n_clusters = int(np.max(cluster_index)) + 1

    def model(params: Sequence[float]) -> np.ndarray:
        ratio, log_scale, phi_delta = map(float, params)
        scale = math.exp(log_scale)
        out = np.zeros_like(theta, dtype=float)
        for config_name in ("parallel", "cross"):
            mask = configs == config_name
            if np.any(mask):
                out[mask] = analysis.SelectionRules.D2h_Ag(
                    theta[mask],
                    config_name,
                    scale * ratio,
                    scale,
                    canonical_phi + phi_delta,
                )
        return out

    finite_areas = areas[np.isfinite(areas)]
    initial_scale = math.sqrt(max(float(np.nanmax(finite_areas)), 1e-12))
    initial_log_scale = float(np.log(max(initial_scale, 1e-12)))

    def reference_residual(nuisance: Sequence[float]) -> np.ndarray:
        return model((global_ratio, nuisance[0], nuisance[1])) - areas

    reference = least_squares(
        reference_residual,
        np.asarray([initial_log_scale, 0.0], dtype=float),
        bounds=(np.asarray([-30.0, -phi_window_deg]), np.asarray([30.0, phi_window_deg])),
        max_nfev=2000,
    )
    if not reference.success or not np.all(np.isfinite(reference.x)):
        raise ValueError(f"Global-centered reference fit failed: {reference.message}")

    reference_params = np.asarray([global_ratio, reference.x[0], reference.x[1]], dtype=float)
    reference_model = model(reference_params)
    residual = areas - reference_model
    reference_rmse = float(np.sqrt(np.mean(np.square(residual))))

    if global_ratio < 0.0:
        ratio_low, ratio_high = -20.0, -1.0
    else:
        ratio_low, ratio_high = 1.0, 20.0
    lower = np.asarray([ratio_low, -30.0, -phi_window_deg], dtype=float)
    upper = np.asarray([ratio_high, 30.0, phi_window_deg], dtype=float)
    rng = np.random.default_rng(int(seed))
    draws = np.full(int(replicates), np.nan, dtype=float)

    for replicate in range(int(replicates)):
        cluster_weights = rng.choice(np.asarray([-1.0, 1.0]), size=n_clusters)
        synthetic = reference_model + cluster_weights[cluster_index] * residual

        def bootstrap_residual(params: Sequence[float]) -> np.ndarray:
            return model(params) - synthetic

        try:
            fitted = least_squares(
                bootstrap_residual,
                reference_params,
                bounds=(lower, upper),
                max_nfev=1000,
            )
        except Exception:
            continue
        if fitted.success and np.all(np.isfinite(fitted.x)):
            draws[replicate] = float(fitted.x[0])

    finite_draws = draws[np.isfinite(draws)]
    successful = int(finite_draws.size)
    if successful < 2:
        raise ValueError("Fewer than two bootstrap replicates converged.")
    ratio_std = float(np.std(finite_draws, ddof=1))
    bootstrap_mean = float(np.mean(finite_draws))
    boundary = np.isclose(np.abs(finite_draws), 1.0, rtol=0.0, atol=1e-6)
    return GlobalCenteredRatioBootstrap(
        global_ratio=global_ratio,
        ratio_expression=expression,
        canonical_phi_deg=canonical_phi,
        reference_scale=float(math.exp(float(reference.x[0]))),
        reference_phi_shift_deg=float(reference.x[1]),
        reference_rmse=reference_rmse,
        ratio_std=ratio_std,
        bootstrap_mean=bootstrap_mean,
        bootstrap_bias=float(bootstrap_mean - global_ratio),
        percentile_2_5=float(np.percentile(finite_draws, 2.5)),
        percentile_97_5=float(np.percentile(finite_draws, 97.5)),
        draws=draws,
        attempted=int(replicates),
        successful=successful,
        n_observations=int(areas.size),
        n_angle_clusters=n_clusters,
        phase_boundary_hit=bool(abs(float(reference.x[1])) >= phi_window_deg - 1e-3),
        ratio_boundary_fraction=float(np.mean(boundary)),
    )


def colors_from_cmap(cmap_name: Optional[str], count: int, *, start: float = 0.15, stop: float = 0.85) -> List[str]:
    if count <= 0:
        return []
    try:
        cmap = colormaps.get_cmap(cmap_name or "viridis")
    except Exception:
        cmap = colormaps.get_cmap("viridis")
    if count == 1:
        return [to_hex(cmap(0.55))]
    return [to_hex(cmap(v)) for v in np.linspace(start, stop, count)]


def color_for_polarization(color_spec: Any, config_name: str) -> str:
    """Resolve a fit color while accepting legacy one-color-per-fit values."""
    if isinstance(color_spec, dict):
        aliases = (config_name, "xx" if config_name == "parallel" else "yx")
        for alias in aliases:
            value = color_spec.get(alias)
            if value:
                return str(value)
        for value in color_spec.values():
            if value:
                return str(value)
    elif isinstance(color_spec, (list, tuple)) and color_spec:
        index = 0 if config_name == "parallel" else min(1, len(color_spec) - 1)
        return str(color_spec[index])
    return str(color_spec)


def view_cmap_for_runs(exp: ExperimentSet, runs: Sequence[Run]) -> Optional[str]:
    run_ids = {run.id for run in runs}
    for view in exp.views.values():
        if run_ids & set(view.run_ids):
            view.seed_legacy_graph_configs()
            for idx, rid in enumerate(view.run_ids[:2], start=1):
                if rid in run_ids:
                    cfg = view.graph_configs.get(f"{idx}A")
                    if cfg and cfg.cmap:
                        return cfg.cmap
            return view.cmap
    return None


def write_row_fit_table(
    path: str,
    row_fits_by_target: Sequence[Sequence[RowPeakFit]],
    output_unit: str = "meV",
) -> None:
    import pandas as pd

    output_unit = normalize_spectral_unit(output_unit, "meV")
    unit_key = "meV" if output_unit == "meV" else "cm-1"
    rows = []
    for target_group in row_fits_by_target:
        for rf in target_group:
            for angle, area, center, gamma, height in zip(rf.angles, rf.areas, rf.centers, rf.gammas, rf.heights):
                rows.append(
                    {
                        "run_id": rf.run_id,
                        "run": rf.run_label,
                        "config": rf.config,
                        f"target_{unit_key}": float(cm1_to_unit(rf.target, output_unit)),
                        "peak_name": rf.name,
                        "angle_deg": angle,
                        "area": area,
                        f"center_{unit_key}": float(cm1_to_unit(center, output_unit)),
                        f"gamma_{unit_key}": float(cm1_to_unit(gamma, output_unit)),
                        "height": height,
                    }
                )
    pd.DataFrame(rows).to_csv(path, index=False)


def generate_polar_area_pdf(
    exp: ExperimentSet,
    runs: Sequence[Run],
    targets: Sequence[float],
    output_path: str,
    *,
    fit_state: Optional[Dict[str, Any]] = None,
    colors: Optional[Sequence[Any]] = None,
    cmap_name: Optional[str] = None,
    peak_window: float = 8.0,
    center_window: float = 2.0,
    normalize: bool = True,
    export_table: bool = True,
    row_fit_groups: Optional[Sequence[Sequence[RowPeakFit]]] = None,
    output_unit: str = "meV",
    panels_per_page: Optional[int] = None,
) -> Tuple[str, List[RowPeakFit], List[TensorFit]]:
    if not runs:
        raise ValueError("No 2D runs were provided.")
    targets = [float(t) for t in targets]
    if not targets:
        raise ValueError("At least one target peak is required.")
    output_unit = normalize_spectral_unit(output_unit, "meV")

    if row_fit_groups is None:
        row_fit_groups = fit_lorentzian_rows_for_runs(
            runs,
            targets,
            fit_state=fit_state,
            peak_window=peak_window,
            center_window=center_window,
        )
    else:
        row_fit_groups = [list(group) for group in row_fit_groups]
        if len(row_fit_groups) != len(targets):
            raise ValueError("Cached row-fit groups do not match the requested target peaks.")
        if any(not group for group in row_fit_groups):
            raise ValueError("At least one requested target has no cached row-fit series.")

    tensor_fits = [fit_tensor_for_peak(group, fit_state=fit_state) for group in row_fit_groups]

    if colors:
        plot_colors = list(colors)
    else:
        plot_colors = colors_from_cmap(cmap_name or view_cmap_for_runs(exp, runs), len(targets))
    if len(plot_colors) < len(targets):
        plot_colors.extend(colors_from_cmap(cmap_name or "viridis", len(targets) - len(plot_colors)))

    theta_dense = np.linspace(0.0, 360.0, 721)
    theta_rad = np.deg2rad(theta_dense)
    legend_handles = [
        Line2D([0], [0], marker="o", color="black", linestyle="-", markersize=5, linewidth=1.2, label="Parallel polarization (fit)"),
        Line2D([0], [0], marker="x", color="black", linestyle="-", markersize=6, linewidth=1.2, label="Cross polarization (fit)"),
    ]

    output_path = os.path.abspath(output_path)
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    page_size = len(targets) if panels_per_page is None else max(1, int(panels_per_page))
    with PdfPages(output_path) as pdf:
        for page_start in range(0, len(targets), page_size):
            page_end = min(len(targets), page_start + page_size)
            page_items = list(zip(
                row_fit_groups[page_start:page_end],
                tensor_fits[page_start:page_end],
                plot_colors[page_start:page_end],
            ))
            n = len(page_items)
            fig_w = max(3.0 * n, 3.4)
            fig = Figure(figsize=(fig_w, 4.0), constrained_layout=False)
            axes = [fig.add_subplot(1, n, i + 1, projection="polar") for i in range(n)]
            for ax, (group, tensor, color_spec) in zip(axes, page_items):
                scale = tensor.scale if normalize else 1.0
                for rf in group:
                    color = color_for_polarization(color_spec, rf.config)
                    valid = np.isfinite(rf.angles) & np.isfinite(rf.areas)
                    marker = "o" if rf.config == "parallel" else "x"
                    size = 18 if rf.config == "parallel" else 30
                    if valid.any():
                        ax.scatter(np.deg2rad(rf.angles[valid]), rf.areas[valid] / scale, color=color, marker=marker, s=size, alpha=0.9)
                    rule_func = (analysis.RULE_METADATA.get(tensor.rule) or analysis.RULE_METADATA["D2h_Ag"])["func"]
                    y_fit = rule_func(theta_dense, rf.config, *tensor.params) / scale
                    ax.plot(theta_rad, y_fit, color=color, linewidth=1.25, linestyle="-")

                ax.set_theta_zero_location("E")
                ax.set_theta_direction(-1)
                ax.set_thetagrids(np.arange(0, 360, 45), labels=[])
                ax.set_yticklabels([])
                ax.grid(True, color="#b8b8b8", linewidth=0.8, alpha=0.8)
                ax.spines["polar"].set_color("black")
                ax.spines["polar"].set_linewidth(1.1)
                display_target = float(cm1_to_unit(tensor.target, output_unit))
                unit_label = "meV" if output_unit == "meV" else r"cm$^{-1}$"
                ax.set_title(f"{tensor.name}\n({display_target:.2f} {unit_label})", y=-0.15, va="top", fontsize=11)

            fig.legend(handles=legend_handles, loc="lower center", bbox_to_anchor=(0.5, -0.01), ncol=2, frameon=False)
            fig.subplots_adjust(left=0.04, right=0.98, top=0.92, bottom=0.30, wspace=0.45)
            pdf.savefig(fig, bbox_inches="tight")
            fig.clear()

    if export_table:
        base, _ = os.path.splitext(output_path)
        write_row_fit_table(f"{base}_row_fit_areas.csv", row_fit_groups, output_unit=output_unit)

    return output_path, [rf for group in row_fit_groups for rf in group], tensor_fits
