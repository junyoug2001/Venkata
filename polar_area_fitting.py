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

from scipy.optimize import curve_fit

import analysis
from data_structure import EV_PER_CM1, ExperimentSet, Run, RunType


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


def oriented_2d(run: Run) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    if run.shift_cm1 is None or run.angle_values is None or run.intensity_2d is None:
        raise ValueError(f"Run '{run.nickname}' is not a complete 2D run.")
    shift = np.asarray(run.shift_cm1, dtype=float)
    angles = np.asarray(run.angle_values, dtype=float)
    intensity = np.asarray(run.intensity_2d, dtype=float)
    if intensity.shape != (angles.size, shift.size):
        if intensity.shape == (shift.size, angles.size):
            intensity = intensity.T
        else:
            raise ValueError(f"Run '{run.nickname}' has incompatible 2D shape {intensity.shape}.")
    return shift, angles, intensity


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
    for run in runs:
        state = (run.metadata or {}).get("fit_state") or (run.metadata or {}).get("map_fit_state")
        if state:
            return state, run
    params_run = find_fit_params_run(exp, [run.id for run in runs])
    if params_run is not None:
        return params_run.metadata.get("fit_state"), params_run
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
            center = float(peak.get("spec_params", {}).get("x0", [np.nan])[0])
        except Exception:
            continue
        dist = abs(center - target)
        if dist < best_dist:
            best = peak
            best_dist = dist
    if best is None:
        return f"{target:.3g}", "D2h_Ag", None
    return str(best.get("name") or f"{target:.3g}"), str(best.get("rule") or "D2h_Ag"), best


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
    results: List[RowPeakFit] = []
    peak_window = abs(float(peak_window))
    center_window = abs(float(center_window))
    if peak_window <= 0:
        peak_window = 1e-6
    if center_window <= 0:
        center_window = peak_window

    for target in targets:
        target = float(target)
        name, rule, peak_state = match_peak_from_state(fit_state, target)
        mask = (shift >= target - peak_window) & (shift <= target + peak_window)
        xw = shift[mask]
        if xw.size < 5:
            raise ValueError(f"Target {target:g} has fewer than 5 x-points inside the fitting window.")

        gamma_guess = _gamma_guess_from_peak(peak_state, xw, peak_window)
        gamma_lo = max(gamma_guess / 20.0, 1e-6)
        gamma_hi = max(gamma_guess * 20.0, gamma_lo * 2.0)
        areas = np.full(angles.shape, np.nan, dtype=float)
        centers = np.full(angles.shape, np.nan, dtype=float)
        gammas = np.full(angles.shape, np.nan, dtype=float)
        heights = np.full(angles.shape, np.nan, dtype=float)

        x_centered = xw - target

        def model(xc, offset, slope, area, x0_delta, gamma):
            x_abs = xc + target
            return offset + slope * xc + area * analysis.lorentzian_normalized(x_abs, target + x0_delta, gamma)

        for row_idx, _angle in enumerate(angles):
            yw = intensity[row_idx, mask]
            finite = np.isfinite(x_centered) & np.isfinite(yw)
            if finite.sum() < 5:
                continue
            xf = x_centered[finite]
            yf = yw[finite]
            offset0 = float(np.nanmedian(yf))
            amp0 = max(float(np.nanmax(yf) - offset0), 0.0)
            area0 = max(amp0 * math.pi * gamma_guess, 1e-12)
            p0 = [offset0, 0.0, area0, 0.0, gamma_guess]
            lower = [-np.inf, -np.inf, 0.0, -abs(center_window), gamma_lo]
            upper = [np.inf, np.inf, np.inf, abs(center_window), gamma_hi]
            try:
                popt, _ = curve_fit(model, xf, yf, p0=p0, bounds=(lower, upper), maxfev=4000)
            except Exception:
                continue
            area = float(popt[2])
            gamma = abs(float(popt[4]))
            areas[row_idx] = area
            centers[row_idx] = target + float(popt[3])
            gammas[row_idx] = gamma
            heights[row_idx] = area / (math.pi * gamma) if gamma > 0 else np.nan

        results.append(
            RowPeakFit(
                target=target,
                name=name,
                rule=rule,
                angles=np.asarray(angles, dtype=float),
                areas=areas,
                centers=centers,
                gammas=gammas,
                heights=heights,
                config=config_name,
                run_id=run.id,
                run_label=run.nickname,
            )
        )

    return results


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

    popt, _ = curve_fit(model, theta, y, p0=p0, bounds=(lo, hi), maxfev=10000)
    scale = float(np.nanmax(y)) if y.size else 1.0
    if not np.isfinite(scale) or scale <= 0:
        scale = 1.0
    return TensorFit(target=target, name=name, rule=rule, params=np.asarray(popt, dtype=float), param_names=param_names, scale=scale)


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


def write_row_fit_table(path: str, row_fits_by_target: Sequence[Sequence[RowPeakFit]]) -> None:
    import pandas as pd

    rows = []
    for target_group in row_fits_by_target:
        for rf in target_group:
            for angle, area, center, gamma, height in zip(rf.angles, rf.areas, rf.centers, rf.gammas, rf.heights):
                rows.append(
                    {
                        "run_id": rf.run_id,
                        "run": rf.run_label,
                        "config": rf.config,
                        "target_cm-1": rf.target,
                        "peak_name": rf.name,
                        "angle_deg": angle,
                        "area": area,
                        "center_cm-1": center,
                        "gamma_cm-1": gamma,
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
    colors: Optional[Sequence[str]] = None,
    cmap_name: Optional[str] = None,
    peak_window: float = 8.0,
    center_window: float = 2.0,
    normalize: bool = True,
    export_table: bool = True,
) -> Tuple[str, List[RowPeakFit], List[TensorFit]]:
    if not runs:
        raise ValueError("No 2D runs were provided.")
    targets = [float(t) for t in targets]
    if not targets:
        raise ValueError("At least one target peak is required.")

    row_fit_groups: List[List[RowPeakFit]] = []
    for target in targets:
        group = []
        for run in runs:
            group.extend(
                fit_lorentzian_rows(
                    run,
                    [target],
                    fit_state=fit_state,
                    config=infer_config(run, "parallel" if not group else "cross"),
                    peak_window=peak_window,
                    center_window=center_window,
                )
            )
        row_fit_groups.append(group)

    tensor_fits = [fit_tensor_for_peak(group, fit_state=fit_state) for group in row_fit_groups]

    if colors:
        plot_colors = list(colors)
    else:
        plot_colors = colors_from_cmap(cmap_name or view_cmap_for_runs(exp, runs), len(targets))
    if len(plot_colors) < len(targets):
        plot_colors.extend(colors_from_cmap(cmap_name or "viridis", len(targets) - len(plot_colors)))

    n = len(targets)
    fig_w = max(3.0 * n, 3.4)
    fig = Figure(figsize=(fig_w, 3.45), constrained_layout=False)
    axes = [fig.add_subplot(1, n, i + 1, projection="polar") for i in range(n)]
    theta_dense = np.linspace(0.0, 360.0, 721)
    theta_rad = np.deg2rad(theta_dense)

    for idx, (ax, group, tensor, color) in enumerate(zip(axes, row_fit_groups, tensor_fits, plot_colors)):
        scale = tensor.scale if normalize else 1.0
        for rf in group:
            valid = np.isfinite(rf.angles) & np.isfinite(rf.areas)
            marker = "o" if rf.config == "parallel" else "x"
            size = 18 if rf.config == "parallel" else 30
            if valid.any():
                ax.scatter(np.deg2rad(rf.angles[valid]), rf.areas[valid] / scale, color=color, marker=marker, s=size, alpha=0.9)
            rule_func = (analysis.RULE_METADATA.get(tensor.rule) or analysis.RULE_METADATA["D2h_Ag"])["func"]
            y_fit = rule_func(theta_dense, rf.config, *tensor.params) / scale
            linestyle = "-" if rf.config == "parallel" else "--"
            ax.plot(theta_rad, y_fit, color=color, linewidth=1.25, linestyle=linestyle)

        ax.set_theta_zero_location("E")
        ax.set_theta_direction(-1)
        ax.set_thetagrids(np.arange(0, 360, 45), labels=[])
        ax.set_yticklabels([])
        ax.grid(True, color="#b8b8b8", linewidth=0.8, alpha=0.8)
        ax.spines["polar"].set_color("black")
        ax.spines["polar"].set_linewidth(1.1)
        mev = tensor.target * EV_PER_CM1 * 1000.0
        ax.set_title(f"{tensor.name}\n({mev:.2f} meV)", y=-0.26, va="top", fontsize=11)

    legend_handles = [
        Line2D([0], [0], marker="o", color="black", linestyle="-", markersize=5, linewidth=1.2, label="Parallel polarization (fit)"),
        Line2D([0], [0], marker="x", color="black", linestyle="--", markersize=6, linewidth=1.2, label="Cross polarization (fit)"),
    ]
    fig.legend(handles=legend_handles, loc="lower center", bbox_to_anchor=(0.5, -0.02), ncol=2, frameon=False)
    fig.subplots_adjust(left=0.04, right=0.98, top=0.92, bottom=0.28, wspace=0.45)

    output_path = os.path.abspath(output_path)
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with PdfPages(output_path) as pdf:
        pdf.savefig(fig, bbox_inches="tight")
    fig.clear()

    if export_table:
        base, _ = os.path.splitext(output_path)
        write_row_fit_table(f"{base}_row_fit_areas.csv", row_fit_groups)

    return output_path, [rf for group in row_fit_groups for rf in group], tensor_fits
