from __future__ import annotations

import argparse
import json
import os
import re
from typing import List, Optional, Sequence

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from config_manager import config
from data_structure import (
    ExperimentSet,
    Run,
    RunType,
    ViewState,
    cm1_to_unit,
    new_experiment_id,
    new_view_id,
    normalize_spectral_unit,
    spectral_axis_label,
    unit_to_cm1,
)
from fit_overlay import normalize_fit_overlay_mode, overlay_data
from plotting import centers_to_edges

import plot_export_cli


def _normalize_colormap_name(name: Optional[str]) -> Optional[str]:
    if not name:
        return None
    value = str(name).strip()
    for prefix in ("Standard-", "standard-", "CMCrameri-", "cmcrameri-"):
        if value.startswith(prefix):
            return value[len(prefix):]
    return value


def _load_experiment(path: str) -> ExperimentSet:
    return ExperimentSet.from_hdf5(path)


def _tokenize(value: str) -> List[str]:
    return re.findall(r"[a-z0-9]+", value.lower())


def _run_search_text(run: Run) -> str:
    pieces = [run.id, run.nickname, run.source_path or ""]
    metadata = run.metadata or {}
    for key in ("nickname", "base", "stem", "sample", "pol", "category", "raw_x_unit", "fit_component"):
        pieces.append(str(metadata.get(key, "")))
    try:
        pieces.append(json.dumps(metadata, default=str, sort_keys=True))
    except TypeError:
        pieces.append(str(metadata))
    return " ".join(pieces).lower()


def _score_run(run: Run, query: str) -> int:
    text = _run_search_text(run)
    query_lower = query.lower().strip()
    tokens = _tokenize(query)
    if not tokens:
        return 0
    if not all(token in text for token in tokens):
        return 0
    score = len(tokens)
    if query_lower in text:
        score += 10
    if query_lower == run.nickname.lower():
        score += 50
    if query_lower == run.id.lower():
        score += 100
    if run.run_type == RunType.FIT_PARAMS and not any(token in {"fit", "params", "parameter", "parameters"} for token in tokens):
        score -= 20
    if not any(token in {"xx", "yx", "xy", "yy", "rl", "lr", "rr", "ll", "cross", "parallel"} for token in tokens):
        pol_text = " ".join(str((run.metadata or {}).get(k, "")) for k in ("pol", "nickname", "base", "stem")).lower()
        if any(token in pol_text for token in ("xx", "parallel", "para", "rr", "ll")):
            score += 1
    return score


def _find_run_by_query(exp: ExperimentSet, query: Optional[str]) -> Run:
    if not query:
        for run in exp.runs.values():
            return run
        raise ValueError("Experiment does not contain any runs.")

    scored = [(score, run) for run in exp.runs.values() if (score := _score_run(run, query)) > 0]
    if not scored:
        raise ValueError(f"No run matches query '{query}'.")
    scored.sort(key=lambda item: item[0], reverse=True)
    best_score = scored[0][0]
    matches = [run for score, run in scored if score == best_score]
    if len(matches) == 1:
        return matches[0]
    names = ", ".join(f"{run.nickname} ({run.id})" for run in matches[:8])
    raise ValueError(f"Run query '{query}' is ambiguous: {names}")


def _find_view(exp: ExperimentSet, token: Optional[str]) -> Optional[ViewState]:
    return plot_export_cli._find_view(exp, token) if token or exp.views else None


def _display_unit(view: Optional[ViewState], args) -> str:
    explicit = getattr(args, "unit", None)
    if explicit:
        return normalize_spectral_unit(explicit)
    if view is not None:
        return normalize_spectral_unit(getattr(view, "spectral_unit", None))
    return normalize_spectral_unit(config.get("default_spectral_unit", config.get("unit", "meV")))


def _oriented_2d(run: Run):
    return plot_export_cli._oriented_2d(run)


def _nearest_index(values: np.ndarray, target: Optional[float]) -> int:
    return plot_export_cli._nearest_index(values, target)


def _plot_map(ax, run: Run, view: Optional[ViewState], args):
    shift, angles, intensity = _oriented_2d(run)
    unit = _display_unit(view, args)
    x = cm1_to_unit(shift, unit)
    cmap = _normalize_colormap_name(args.cmap) or config.get("default_colormap", config.get("colormap", "OrRd"))
    mesh = ax.pcolormesh(centers_to_edges(x), centers_to_edges(angles), intensity, shading="auto", cmap=cmap)
    ax.set_xlabel(spectral_axis_label(unit))
    ax.set_ylabel("Angle (deg)")
    ax.set_title(args.title or f"{run.nickname}: 2D map")
    if args.xlim is not None:
        ax.set_xlim(args.xlim)
    if args.ylim is not None:
        ax.set_ylim(args.ylim)
    if args.clim is not None:
        mesh.set_clim(args.clim[0], args.clim[1])
    elif args.vlim_percent is not None:
        valid = intensity[np.isfinite(intensity)]
        if valid.size:
            lo, hi = np.percentile(valid, args.vlim_percent)
            if hi <= lo:
                hi = lo + 1e-9
            mesh.set_clim(lo, hi)
    return mesh


def _add_fit_markers(ax, exp: ExperimentSet, run: Run, unit: str, mode: str) -> None:
    data = overlay_data(exp, run, mode)
    if data is None or data.centers_cm1.size == 0:
        return
    for center in cm1_to_unit(data.centers_cm1, unit):
        if np.isfinite(center):
            ax.axvline(center, color="white", linestyle=":", linewidth=1.1, alpha=0.8)


def _plot_slice(ax, exp: ExperimentSet, run: Run, view: Optional[ViewState], args, kind: str):
    shift, angles, intensity = _oriented_2d(run)
    unit = _display_unit(view, args)
    mode = normalize_fit_overlay_mode(args.fit_overlay)
    fit_data = overlay_data(exp, run, mode)
    added_fit = False

    if kind == "slice-b":
        x_value_cm1 = None if args.x_value is None else float(unit_to_cm1(args.x_value, unit))
        idx = _nearest_index(shift, x_value_cm1)
        x = angles
        y = intensity[:, idx]
        display_val = float(cm1_to_unit(shift[idx], unit))
        if args.angle_slice == "polar":
            theta = np.deg2rad(x)
            ax.set_theta_zero_location("N")
            ax.set_theta_direction(-1)
            ax.plot(theta, y, color=args.line_color, linewidth=args.linewidth, label="Signal")
            if fit_data and fit_data.global_matrix is not None:
                ax.plot(theta, fit_data.global_matrix[:, idx], color="red", linewidth=args.linewidth, label="Global fit")
                added_fit = True
            if fit_data and fit_data.row_matrix is not None:
                ax.plot(theta, fit_data.row_matrix[:, idx], color="#1f77b4", linestyle="--", linewidth=args.linewidth, label="Row fit")
                added_fit = True
        else:
            ax.plot(x, y, color=args.line_color, linewidth=args.linewidth, label="Signal")
            if fit_data and fit_data.global_matrix is not None:
                ax.plot(x, fit_data.global_matrix[:, idx], color="red", linewidth=args.linewidth, label="Global fit")
                added_fit = True
            if fit_data and fit_data.row_matrix is not None:
                ax.plot(x, fit_data.row_matrix[:, idx], color="#1f77b4", linestyle="--", linewidth=args.linewidth, label="Row fit")
                added_fit = True
            ax.set_xlabel("Angle (deg)")
            ax.set_ylabel("Intensity (a.u.)")
        ax.set_title(args.title or f"Angular slice @ {display_val:.2f} {unit}")
    else:
        idx = _nearest_index(angles, args.y_value)
        x = cm1_to_unit(shift, unit)
        y = intensity[idx, :]
        ax.plot(x, y, color=args.line_color, linewidth=args.linewidth, label="Signal")
        if fit_data and fit_data.global_matrix is not None:
            ax.plot(x, fit_data.global_matrix[idx, :], color="red", linewidth=args.linewidth, label="Global fit")
            added_fit = True
        if fit_data and fit_data.row_matrix is not None:
            ax.plot(x, fit_data.row_matrix[idx, :], color="#1f77b4", linestyle="--", linewidth=args.linewidth, label="Row fit")
            added_fit = True
        ax.set_xlabel(spectral_axis_label(unit))
        ax.set_ylabel("Intensity (a.u.)")
        ax.set_title(args.title or f"Spectral slice @ {angles[idx]:.2f} deg")
        if args.xlim is not None:
            ax.set_xlim(args.xlim)

    if args.ylim is not None and kind == "slice-c":
        ax.set_ylim(args.ylim)
    if added_fit:
        ax.legend()


def _plot_1d(ax, run: Run, view: Optional[ViewState], args):
    x, y, xlabel = plot_export_cli._run_1d_xy(run, _display_unit(view, args))
    ax.plot(x, y, color=args.line_color, linewidth=args.linewidth)
    ax.set_title(args.title or f"{run.nickname} (1D)")
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Intensity (a.u.)")
    if args.xlim is not None:
        ax.set_xlim(args.xlim)
    if args.ylim is not None:
        ax.set_ylim(args.ylim)


def cmd_list(args) -> int:
    exp = _load_experiment(args.experiment)
    print("Views:")
    for view in exp.views.values():
        run_names = [exp.get_run(rid).nickname for rid in view.run_ids if exp.get_run(rid)]
        print(f"  {view.id}\t{view.title}\t{', '.join(run_names)}")
    print("Runs:")
    for run in exp.runs.values():
        kind = "2D" if run.is_2d else run.run_type.value
        print(f"  {run.id}\t{run.nickname}\t{kind}\t{run.source_path}")
    return 0


def cmd_plot(args) -> int:
    exp = _load_experiment(args.experiment)
    plot_export_cli._register_custom_colormaps()
    args.cmap = _normalize_colormap_name(args.cmap)
    view = _find_view(exp, args.view) if args.view else None
    kind = args.kind
    panel_key = plot_export_cli._panel_key(args.panel)
    slot = int(panel_key[0]) if len(panel_key) == 2 and panel_key[0].isdigit() else None
    if panel_key == "FULL":
        kind = "full"
    elif panel_key.endswith("A"):
        kind = "map"
    elif panel_key.endswith("B"):
        kind = "slice-b"
    elif panel_key.endswith("C"):
        kind = "slice-c"

    if view is not None and slot is not None and not args.run_query:
        run_ids = [rid for rid in view.run_ids if exp.get_run(rid)]
        if slot < 1 or slot > len(run_ids):
            raise ValueError(f"View '{view.title}' does not have run slot {slot}.")
        run = exp.get_run(run_ids[slot - 1])
    else:
        run = _find_run_by_query(exp, args.run_query)

    if kind == "auto":
        kind = "map" if run.is_2d else "trace"

    if kind == "full":
        if view is None:
            raise ValueError("--view is required for --kind full.")
        fig = plot_export_cli._render_full_view(exp, view, args)
    else:
        projection = "polar" if kind == "slice-b" and args.angle_slice == "polar" else None
        fig = plt.figure(figsize=(args.width, args.height), dpi=args.dpi)
        ax = fig.add_axes([0, 0, 1, 1], projection=projection) if args.data_only else fig.add_subplot(1, 1, 1, projection=projection)
        if kind == "map":
            mesh = _plot_map(ax, run, view, args)
            _add_fit_markers(ax, exp, run, _display_unit(view, args), args.fit_overlay)
            if args.data_only:
                ax.set_axis_off()
            else:
                fig.colorbar(mesh, ax=ax, label=args.colorbar_label)
        elif kind in {"slice-b", "slice-c"} and run.is_2d:
            _plot_slice(ax, exp, run, view, args, kind)
            if args.data_only:
                ax.set_axis_off()
        else:
            _plot_1d(ax, run, view, args)
            if args.data_only:
                ax.set_axis_off()
        if not args.data_only:
            fig.tight_layout()

    output = os.path.abspath(args.output)
    os.makedirs(os.path.dirname(output) or ".", exist_ok=True)
    save_kwargs = {"dpi": args.dpi}
    if args.data_only:
        save_kwargs.update({"bbox_inches": "tight", "pad_inches": 0})
    else:
        save_kwargs.update({"bbox_inches": "tight"})
    fig.savefig(output, **save_kwargs)
    plt.close(fig)
    print(output)
    return 0


def cmd_export_data(args) -> int:
    exp = _load_experiment(args.experiment)
    if not args.csv_dir and not args.igor:
        raise ValueError("Specify --csv-dir and/or --igor.")

    queries = args.run_query or []
    selected = [_find_run_by_query(exp, query) for query in queries] if queries else list(exp.runs.values())
    seen = set()
    runs = []
    for run in selected:
        if run.id not in seen:
            seen.add(run.id)
            runs.append(run)

    if args.csv_dir:
        os.makedirs(args.csv_dir, exist_ok=True)
        for run in runs:
            run.export_csv(output_dir=args.csv_dir)
        print(os.path.abspath(args.csv_dir))

    if args.igor:
        subset = ExperimentSet(id=new_experiment_id())
        subset.metadata.update(exp.metadata)
        subset.metadata["num_runs"] = len(runs)
        for run in runs:
            subset.runs[run.id] = run
        out = subset.export_igor(args.igor)
        print(os.path.abspath(out))

    return 0


def cmd_create_experiment(args) -> int:
    exp = ExperimentSet(id=new_experiment_id())
    nicknames = args.nickname or []
    run_ids = []
    for idx, path in enumerate(args.input):
        run = Run.from_file(
            path,
            default_unknown_1d_spectral_unit=config.get("default_unknown_1d_spectral_unit", "meV"),
            default_unknown_2d_spectral_unit=config.get("default_unknown_2d_spectral_unit", "meV"),
        )
        if idx < len(nicknames):
            run.nickname = nicknames[idx]
        base_id = run.id
        suffix = 1
        while run.id in exp.runs:
            suffix += 1
            run.id = f"{base_id}_{suffix}"
        exp.add_run(run)
        run_ids.append(run.id)

    unit = normalize_spectral_unit(args.unit or config.get("default_spectral_unit", config.get("unit", "meV")))
    cmap = _normalize_colormap_name(args.default_cmap) or config.get("default_colormap", config.get("colormap", "OrRd"))
    view = ViewState(
        id=new_view_id(),
        title=args.view_title or "View 1",
        run_ids=run_ids[:2],
        spectral_unit=unit,
        x_axis="energy_eV" if unit == "meV" else "shift_cm1",
        cmap=cmap,
    )
    for key in ("1A", "2A"):
        view.get_graph_config(key).cmap = cmap
    exp.add_view(view)
    exp.metadata["num_runs"] = len(exp.runs)

    output = os.path.abspath(args.output)
    os.makedirs(os.path.dirname(output) or ".", exist_ok=True)
    exp.export_hdf5(output)
    print(output)
    for run in exp.runs.values():
        print(f"{run.id}\t{run.nickname}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Agent-oriented Venkata Raman plotting and export CLI.")
    sub = parser.add_subparsers(dest="command", required=True)

    p_list = sub.add_parser("list", help="List views and runs in an ExperimentSet.")
    p_list.add_argument("experiment")
    p_list.set_defaults(func=cmd_list)

    p_plot = sub.add_parser("plot", help="Export a plot from an ExperimentSet.")
    p_plot.add_argument("experiment")
    p_plot.add_argument("--kind", choices=["auto", "map", "slice-b", "slice-c", "trace", "full"], default="auto")
    p_plot.add_argument("--run-query")
    p_plot.add_argument("--view")
    p_plot.add_argument("-o", "--output", required=True)
    p_plot.add_argument("--unit", choices=["meV", "cm-1"])
    p_plot.add_argument("--xlim", nargs=2, type=float, metavar=("MIN", "MAX"))
    p_plot.add_argument("--ylim", nargs=2, type=float, metavar=("MIN", "MAX"))
    p_plot.add_argument("--clim", nargs=2, type=float, metavar=("MIN", "MAX"))
    p_plot.add_argument("--vlim-percent", nargs=2, type=float, metavar=("MIN", "MAX"))
    p_plot.add_argument("--x-value", type=float)
    p_plot.add_argument("--y-value", type=float)
    p_plot.add_argument("--cmap")
    p_plot.add_argument("--panel", default="auto", help="Reserved for compatibility with export-plot.")
    p_plot.add_argument("--fit-overlay", choices=["off", "global", "row", "both"], default="off")
    p_plot.add_argument("--angle-slice", choices=["polar", "cartesian"], default=str(config.get("default_angle_slice_type", "polar")))
    p_plot.add_argument("--data-only", action="store_true")
    p_plot.add_argument("--title")
    p_plot.add_argument("--colorbar-label", default="Intensity")
    p_plot.add_argument("--line-color", default="black")
    p_plot.add_argument("--linewidth", type=float, default=1.2)
    p_plot.add_argument("--width", type=float, default=7.0)
    p_plot.add_argument("--height", type=float, default=5.0)
    p_plot.add_argument("--dpi", type=int, default=300)
    p_plot.set_defaults(func=cmd_plot)

    p_export = sub.add_parser("export-data", help="Export selected runs to CSV and/or Igor Text.")
    p_export.add_argument("experiment")
    p_export.add_argument("--run-query", action="append")
    p_export.add_argument("--csv-dir")
    p_export.add_argument("--igor")
    p_export.set_defaults(func=cmd_export_data)

    p_create = sub.add_parser("create-experiment", help="Create a new ExperimentSet from raw files.")
    p_create.add_argument("--input", action="append", required=True)
    p_create.add_argument("--nickname", action="append")
    p_create.add_argument("--default-cmap")
    p_create.add_argument("--unit", choices=["meV", "cm-1"])
    p_create.add_argument("--view-title")
    p_create.add_argument("-o", "--output", required=True)
    p_create.set_defaults(func=cmd_create_experiment)

    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
