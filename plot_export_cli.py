from __future__ import annotations

import argparse
import os
from typing import Optional, Sequence, Tuple

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

import polar_area_fitting
from config_manager import config
from data_structure import (
    ExperimentSet,
    Run,
    RunType,
    ViewState,
    cm1_to_unit,
    normalize_spectral_unit,
    spectral_axis_for_run,
    spectral_axis_label,
    unit_to_cm1,
)
from plotting import centers_to_edges


def _register_custom_colormaps() -> None:
    for name, nodes in config.get("custom_colormaps", {}).items():
        if len(nodes) < 2:
            continue
        sorted_nodes = sorted(nodes, key=lambda item: item[0])
        positions = [float(pos) for pos, _ in sorted_nodes]
        colors = [color for _, color in sorted_nodes]
        span = positions[-1] - positions[0]
        if span > 0:
            positions = [(pos - positions[0]) / span for pos in positions]
        positions[0] = 0.0
        positions[-1] = 1.0
        try:
            cmap = LinearSegmentedColormap.from_list(name, list(zip(positions, colors)))
            plt.colormaps.register(cmap=cmap, force=True)
        except Exception:
            pass


def _find_view(exp: ExperimentSet, token: Optional[str]) -> Optional[ViewState]:
    if not exp.views:
        return None
    if not token:
        return next(iter(exp.views.values()))
    if token in exp.views:
        return exp.views[token]
    matches = [view for view in exp.views.values() if view.title == token]
    if len(matches) == 1:
        return matches[0]
    lower = token.lower()
    matches = [view for view in exp.views.values() if lower in view.title.lower()]
    if len(matches) == 1:
        return matches[0]
    raise ValueError(f"Could not uniquely match view '{token}'.")


def _find_run(exp: ExperimentSet, token: Optional[str], view: Optional[ViewState] = None, slot: Optional[int] = None) -> Run:
    if view is not None:
        run_ids = [rid for rid in view.run_ids if exp.get_run(rid)]
        if slot is not None:
            if slot < 1 or slot > len(run_ids):
                raise ValueError(f"View '{view.title}' does not have run slot {slot}.")
            return exp.get_run(run_ids[slot - 1])
        if token is None and run_ids:
            return exp.get_run(run_ids[0])

    if token:
        if token in exp.runs:
            return exp.runs[token]
        matches = [run for run in exp.runs.values() if run.nickname == token]
        if len(matches) == 1:
            return matches[0]
        lower = token.lower()
        matches = [run for run in exp.runs.values() if lower in run.nickname.lower()]
        if len(matches) == 1:
            return matches[0]
        raise ValueError(f"Could not uniquely match run '{token}'.")

    if exp.runs:
        return next(iter(exp.runs.values()))
    raise ValueError("Experiment does not contain any runs.")


def _panel_key(panel: str, default_slot: int = 1) -> str:
    panel = (panel or "auto").upper()
    if panel in {"AUTO", "FULL"}:
        return panel
    if panel in {"A", "B", "C"}:
        return f"{default_slot}{panel}"
    if len(panel) == 2 and panel[0] in {"1", "2"} and panel[1] in {"A", "B", "C"}:
        return panel
    raise ValueError("Panel must be one of full, auto, A, B, C, 1A, 1B, 1C, 2A, 2B, or 2C.")


def _oriented_2d(run: Run) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
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


def _run_1d_xy(run: Run, unit: str) -> Tuple[np.ndarray, np.ndarray, str]:
    if run.intensity is None:
        raise ValueError(f"Run '{run.nickname}' does not contain stored 1D intensity data.")
    y = np.asarray(run.intensity, dtype=float)
    if run.shift_cm1 is not None and len(run.shift_cm1) == len(y):
        return spectral_axis_for_run(run, unit), y, spectral_axis_label(unit)
    if run.angle_values is not None and len(run.angle_values) == len(y):
        return np.asarray(run.angle_values, dtype=float), y, "Angle (deg)"
    if run.wl_nm is not None and len(run.wl_nm) == len(y):
        return np.asarray(run.wl_nm, dtype=float), y, "Wavelength (nm)"
    if run.energy_eV is not None and len(run.energy_eV) == len(y):
        return np.asarray(run.energy_eV, dtype=float), y, "Energy (eV)"
    return np.arange(y.size, dtype=float), y, "Index"


def _graph_config(view: Optional[ViewState], key: str):
    if view is None:
        return None
    view.seed_legacy_graph_configs()
    return view.graph_configs.get(key)


def _display_unit(view: Optional[ViewState], args) -> str:
    explicit = getattr(args, "unit", None)
    if explicit:
        return normalize_spectral_unit(explicit)
    if view is not None:
        return normalize_spectral_unit(getattr(view, "spectral_unit", None))
    return normalize_spectral_unit(config.get("default_spectral_unit", config.get("unit", "meV")))


def _apply_axis_config(ax, graph_cfg, args, *, spectral_x: bool = False, unit: str = "meV") -> None:
    xlim = args.xlim if args.xlim is not None else (graph_cfg.xlim if graph_cfg else None)
    if xlim is not None and spectral_x and args.xlim is None:
        xlim = tuple(float(v) for v in cm1_to_unit(np.asarray(xlim, dtype=float), unit))
    ylim = args.ylim if args.ylim is not None else (graph_cfg.ylim if graph_cfg else None)
    if xlim is not None:
        ax.set_xlim(xlim)
    if ylim is not None:
        ax.set_ylim(ylim)


def _apply_color_config(mesh, graph_cfg, args, data: np.ndarray) -> None:
    clim = args.clim if args.clim is not None else (graph_cfg.clim if graph_cfg else None)
    if clim is not None:
        mesh.set_clim(clim[0], clim[1])
        return

    pct = args.vlim_percent
    if pct is None and graph_cfg and graph_cfg.vmin is not None and graph_cfg.vmax is not None:
        pct = (graph_cfg.vmin, graph_cfg.vmax)
    if pct is not None:
        valid = data[np.isfinite(data)]
        if valid.size:
            lo, hi = np.percentile(valid, pct)
            if hi <= lo:
                hi = lo + 1e-9
            mesh.set_clim(lo, hi)


def _render_map(ax, run: Run, view: Optional[ViewState], panel_key: str, args):
    shift, angles, intensity = _oriented_2d(run)
    unit = _display_unit(view, args)
    x = cm1_to_unit(shift, unit)
    graph_cfg = _graph_config(view, panel_key)
    cmap = args.cmap or (
        graph_cfg.cmap if graph_cfg and graph_cfg.cmap else
        (view.cmap if view else config.get("default_colormap", config.get("colormap", "OrRd")))
    )
    mesh = ax.pcolormesh(
        centers_to_edges(x),
        centers_to_edges(angles),
        intensity,
        shading="auto",
        cmap=cmap,
    )
    ax.set_title(args.title or f"{run.nickname}: 2D map")
    ax.set_xlabel(spectral_axis_label(unit))
    ax.set_ylabel("Angle (deg)")
    _apply_axis_config(ax, graph_cfg, args, spectral_x=True, unit=unit)
    _apply_color_config(mesh, graph_cfg, args, intensity)
    return mesh


def _nearest_index(values: np.ndarray, target: Optional[float]) -> int:
    if target is None:
        return int(values.size // 2)
    return int(np.abs(values - target).argmin())


def _render_slice(ax, run: Run, view: Optional[ViewState], panel_key: str, args):
    shift, angles, intensity = _oriented_2d(run)
    unit = _display_unit(view, args)
    if panel_key.endswith("B"):
        x_value_cm1 = None if args.x_value is None else float(unit_to_cm1(args.x_value, unit))
        idx = _nearest_index(shift, x_value_cm1)
        x = angles
        y = intensity[:, idx]
        ax.set_xlabel("Angle (deg)")
        display_val = float(cm1_to_unit(shift[idx], unit))
        ax.set_title(args.title or f"Angular slice @ {display_val:.2f} {unit}")
        spectral_x = False
    else:
        idx = _nearest_index(angles, args.y_value)
        x = cm1_to_unit(shift, unit)
        y = intensity[idx, :]
        ax.set_xlabel(spectral_axis_label(unit))
        ax.set_title(args.title or f"Spectral slice @ {angles[idx]:.2f} deg")
        spectral_x = True
    ax.plot(x, y, color=args.line_color, linewidth=args.linewidth)
    ax.set_ylabel("Intensity (a.u.)")
    _apply_axis_config(ax, _graph_config(view, panel_key), args, spectral_x=spectral_x, unit=unit)


def _render_1d(ax, run: Run, view: Optional[ViewState], panel_key: str, args):
    unit = _display_unit(view, args)
    x, y, xlabel = _run_1d_xy(run, unit)
    ax.plot(x, y, color=args.line_color, linewidth=args.linewidth)
    ax.set_title(args.title or f"{run.nickname} (1D)")
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Intensity (a.u.)")
    raw_unit = str(run.metadata.get("raw_x_unit", "")).lower()
    spectral_x = run.shift_cm1 is not None and not ("angle" in raw_unit or "deg" in raw_unit)
    _apply_axis_config(ax, _graph_config(view, panel_key), args, spectral_x=spectral_x, unit=unit)


def _render_single(exp: ExperimentSet, view: Optional[ViewState], run: Run, panel_key: str, args):
    fig, ax = plt.subplots(figsize=(args.width, args.height), dpi=args.dpi)
    if panel_key.endswith("A") and run.is_2d:
        mesh = _render_map(ax, run, view, panel_key, args)
        fig.colorbar(mesh, ax=ax, label=args.colorbar_label)
    elif run.is_2d and panel_key.endswith(("B", "C")):
        _render_slice(ax, run, view, panel_key, args)
    else:
        _render_1d(ax, run, view, panel_key if panel_key != "AUTO" else "1C", args)
    fig.tight_layout()
    return fig


def _render_full_view(exp: ExperimentSet, view: ViewState, args):
    run_ids = [rid for rid in view.run_ids if exp.get_run(rid)][:2]
    if not run_ids:
        raise ValueError(f"View '{view.title}' has no runs.")

    fig = plt.figure(figsize=(args.width, args.height), dpi=args.dpi)
    rows = len(run_ids)
    gs = fig.add_gridspec(rows, 3, width_ratios=[2.0, 1.0, 1.0], hspace=0.55, wspace=0.45)

    for row, rid in enumerate(run_ids):
        run = exp.get_run(rid)
        slot = row + 1
        if run.is_2d:
            ax_map = fig.add_subplot(gs[row, 0])
            mesh = _render_map(ax_map, run, view, f"{slot}A", args)
            fig.colorbar(mesh, ax=ax_map, label=args.colorbar_label)
            _render_slice(fig.add_subplot(gs[row, 1]), run, view, f"{slot}B", args)
            _render_slice(fig.add_subplot(gs[row, 2]), run, view, f"{slot}C", args)
        else:
            ax = fig.add_subplot(gs[row, :])
            _render_1d(ax, run, view, f"{slot}C", args)

    fig.suptitle(args.title or view.title)
    fig.tight_layout()
    return fig


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Export plots from a saved Venkata wx GUI experiment without launching the GUI."
    )
    parser.add_argument("experiment", help="Path to a saved .h5/.hdf5 experiment.")
    parser.add_argument("-o", "--output", help="Output plot path. Extension selects png, pdf, svg, etc.")
    parser.add_argument("--view", help="View id, exact title, or unique title substring.")
    parser.add_argument("--run", help="Run id, exact nickname, or unique nickname substring.")
    parser.add_argument("--panel", default="auto", help="Panel to export: full, auto, A, B, C, 1A, 1B, 1C, 2A, 2B, 2C.")
    parser.add_argument("--unit", choices=["meV", "cm-1"], help="Spectral display unit for x-limits, x-value, and labels.")
    parser.add_argument("--xlim", nargs=2, type=float, metavar=("MIN", "MAX"))
    parser.add_argument("--ylim", nargs=2, type=float, metavar=("MIN", "MAX"))
    parser.add_argument("--clim", nargs=2, type=float, metavar=("MIN", "MAX"), help="Absolute colorbar limits.")
    parser.add_argument("--vlim-percent", nargs=2, type=float, metavar=("MIN", "MAX"), help="Percentile colorbar limits.")
    parser.add_argument("--cmap", help="Matplotlib or custom colormap name.")
    parser.add_argument("--x-value", type=float, help="Raman shift used for B-panel angular slices.")
    parser.add_argument("--y-value", type=float, help="Angle used for C-panel spectral slices.")
    parser.add_argument("--title")
    parser.add_argument("--colorbar-label", default="Intensity")
    parser.add_argument("--line-color", default="black")
    parser.add_argument("--linewidth", type=float, default=1.2)
    parser.add_argument("--width", type=float, default=7.0)
    parser.add_argument("--height", type=float, default=5.0)
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument("--list", action="store_true", help="List views and runs instead of exporting.")
    parser.add_argument("--polar-area-fit", action="store_true", help="Export polar peak-area tensor fits as vector PDF.")
    parser.add_argument("--target-peaks", nargs="+", type=float, metavar="CM1", help="Target Raman-shift peaks for --polar-area-fit.")
    parser.add_argument("--peak-window", type=float, default=8.0, help="Half-width around each target peak used for row fits.")
    parser.add_argument("--center-window", type=float, default=2.0, help="Allowed fitted-center deviation around each target peak.")
    parser.add_argument("--polar-colors", help="Comma-separated colors for polar plots. Defaults to the run/view colormap.")
    parser.add_argument("--no-polar-normalize", action="store_true", help="Plot raw peak areas instead of normalizing each polar subplot.")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    _register_custom_colormaps()
    exp = ExperimentSet.from_hdf5(args.experiment)

    if args.list:
        print("Views:")
        for view in exp.views.values():
            print(f"  {view.id}\t{view.title}\t{len(view.run_ids)} run(s)")
        print("Runs:")
        for run in exp.runs.values():
            print(f"  {run.id}\t{run.nickname}")
        return 0

    if args.polar_area_fit:
        if not args.output:
            parser.error("--output is required for --polar-area-fit.")
        if not args.target_peaks:
            parser.error("--target-peaks is required for --polar-area-fit.")

        view = _find_view(exp, args.view) if args.view else None
        selected_run = _find_run(exp, args.run, view=view)

        if selected_run.run_type == RunType.FIT_PARAMS:
            runs = polar_area_fitting.runs_from_fit_selection(exp, selected_run)
            fit_state = selected_run.metadata.get("fit_state")
        else:
            params_run = polar_area_fitting.find_fit_params_run(exp, [selected_run.id])
            if params_run is not None:
                runs = polar_area_fitting.runs_from_fit_selection(exp, params_run)
                fit_state = params_run.metadata.get("fit_state")
            else:
                runs = [selected_run]
                fit_state, _ = polar_area_fitting.fit_state_for_runs(exp, runs)

        if not runs:
            raise ValueError("No source 2D runs were found for polar-area fitting.")

        colors = None
        if args.polar_colors:
            colors = [c.strip() for c in args.polar_colors.split(",") if c.strip()]

        output = os.path.abspath(args.output)
        out_path, row_fits, tensor_fits = polar_area_fitting.generate_polar_area_pdf(
            exp,
            runs,
            args.target_peaks,
            output,
            fit_state=fit_state,
            colors=colors,
            cmap_name=args.cmap,
            peak_window=args.peak_window,
            center_window=args.center_window,
            normalize=not args.no_polar_normalize,
        )
        print(out_path)
        print(f"row_fit_series={len(row_fits)} tensor_fits={len(tensor_fits)}")
        return 0

    if not args.output:
        parser.error("--output is required unless --list is used.")

    view = _find_view(exp, args.view) if args.view or not args.run else None
    panel_key = _panel_key(args.panel)
    if panel_key == "FULL" and view is None:
        panel_key = "AUTO"

    if view is not None and panel_key == "FULL":
        fig = _render_full_view(exp, view, args)
    else:
        slot = int(panel_key[0]) if len(panel_key) == 2 and panel_key[0].isdigit() else None
        run = _find_run(exp, args.run, view=view, slot=slot)
        if panel_key == "AUTO":
            panel_key = "1A" if run.is_2d else "1C"
        fig = _render_single(exp, view, run, panel_key, args)

    output = os.path.abspath(args.output)
    os.makedirs(os.path.dirname(output) or ".", exist_ok=True)
    fig.savefig(output, dpi=args.dpi, bbox_inches="tight")
    plt.close(fig)
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
