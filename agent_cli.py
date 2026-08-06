from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import os
import re
import sys
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.lines import Line2D
from matplotlib.ticker import FormatStrFormatter

import analysis
import polar_area_fitting
import qe_raman
import row_fit_cli
from config_manager import config
from data_structure import (
    EV_PER_CM1,
    ExperimentSet,
    Run,
    RunType,
    ViewState,
    cm1_to_unit,
    is_hidden_run,
    new_experiment_id,
    new_run_id,
    new_view_id,
    normalize_spectral_unit,
    spectral_axis_label,
    unit_to_cm1,
)
from fast_hdf5 import (
    FastHDF5Unsupported,
    read_experiment_metadata,
    read_run_arrays,
)
from fit_overlay import (
    cached_row_fits_for_runs,
    find_fit_state,
    has_saved_row_fit_result,
    normalize_fit_overlay_mode,
    overlay_data,
    row_fit_cache_hash,
)
from plotting import centers_to_edges

import plot_export_cli


def _visible_runs(exp: ExperimentSet) -> List[Run]:
    return [run for run in exp.runs.values() if not is_hidden_run(run)]


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
        for run in _visible_runs(exp):
            return run
        raise ValueError("Experiment does not contain any runs.")

    scored = [(score, run) for run in _visible_runs(exp) if (score := _score_run(run, query)) > 0]
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


def _json_default(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    return str(value)


def _write_experiment(exp: ExperimentSet, output: str) -> str:
    output = os.path.abspath(output)
    os.makedirs(os.path.dirname(output) or ".", exist_ok=True)
    exp.metadata["num_runs"] = len(exp.runs)
    exp.export_hdf5(output)
    return output


def _reject_in_place(output: str, *inputs: Optional[str]) -> None:
    out = os.path.abspath(output)
    for path in inputs:
        if path and os.path.abspath(path) == out:
            raise ValueError("In-place writes are not supported; choose a different --output path.")


def _ensure_unique_run_id(exp: ExperimentSet, run: Run) -> None:
    base = run.id
    suffix = 1
    while run.id in exp.runs:
        suffix += 1
        run.id = f"{base}_{suffix}"


def _run_summary(run: Run) -> Dict[str, Any]:
    md = run.metadata or {}
    fit_state = md.get("fit_state") or md.get("map_fit_state")
    return {
        "id": run.id,
        "nickname": run.nickname,
        "kind": "2D" if run.is_2d else run.run_type.value,
        "run_type": run.run_type.value,
        "source_path": run.source_path,
        "raw_x_unit": md.get("raw_x_unit", ""),
        "polarization_rows": md.get("polarization_rows", []),
        "angle_rotation_summary": md.get("angle_rotation_summary", ""),
        "fit_peak_count": len((fit_state or {}).get("peaks") or []),
        "source_run_ids": md.get("source_run_ids") or md.get("fit_params_source_run_ids") or [],
    }


def _view_summary(exp: ExperimentSet, view: ViewState) -> Dict[str, Any]:
    return {
        "id": view.id,
        "title": view.title,
        "spectral_unit": view.spectral_unit,
        "run_ids": [rid for rid in view.run_ids if (run := exp.get_run(rid)) and not is_hidden_run(run)],
        "runs": [run.nickname for rid in view.run_ids if (run := exp.get_run(rid)) and not is_hidden_run(run)],
    }



def _fast_is_hidden_header(header: Dict[str, Any]) -> bool:
    md = header.get("metadata") or {}
    return bool(
        md.get("hidden_from_default_lists")
        or md.get("is_fit_cache")
        or md.get("cache_run")
    )


def _fast_has_dataset(header: Dict[str, Any], name: str) -> bool:
    return name in (header.get("datasets") or {})


def _fast_is_2d_header(header: Dict[str, Any]) -> bool:
    datasets = header.get("datasets") or {}
    return "intensity_2d" in datasets and "angle_values" in datasets


def _fast_fit_state_from_metadata(metadata: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    return metadata.get("fit_state") or metadata.get("map_fit_state")


def _fast_source_run_ids(header: Dict[str, Any]) -> List[str]:
    md = header.get("metadata") or {}
    values = md.get("source_run_ids") or md.get("fit_params_source_run_ids") or []
    return [str(value) for value in values if str(value)]


def _fast_run_summary(header: Dict[str, Any]) -> Dict[str, Any]:
    md = header.get("metadata") or {}
    fit_state = _fast_fit_state_from_metadata(md)
    return {
        "id": header.get("id", ""),
        "nickname": header.get("nickname", header.get("id", "")),
        "kind": "2D" if _fast_is_2d_header(header) else header.get("run_type", "other"),
        "run_type": header.get("run_type", "other"),
        "source_path": header.get("source_path", ""),
        "raw_x_unit": md.get("raw_x_unit", ""),
        "polarization_rows": md.get("polarization_rows", []),
        "angle_rotation_summary": md.get("angle_rotation_summary", ""),
        "fit_peak_count": len((fit_state or {}).get("peaks") or []),
        "source_run_ids": _fast_source_run_ids(header),
        "datasets": header.get("datasets") or {},
    }


def _fast_view_summary(meta: Dict[str, Any], view_id: str, view: Dict[str, Any]) -> Dict[str, Any]:
    runs = meta.get("runs") or {}
    visible_ids = [
        str(rid)
        for rid in (view.get("run_ids") or [])
        if str(rid) in runs and not _fast_is_hidden_header(runs[str(rid)])
    ]
    return {
        "id": view_id,
        "title": view.get("title", view_id),
        "spectral_unit": normalize_spectral_unit(view.get("spectral_unit", "meV")),
        "run_ids": visible_ids,
        "runs": [runs[rid].get("nickname", rid) for rid in visible_ids],
    }


def _cmd_list_fast(args) -> int:
    meta = read_experiment_metadata(args.experiment)
    visible_runs = [run for run in (meta.get("runs") or {}).values() if not _fast_is_hidden_header(run)]
    if getattr(args, "json", False):
        payload = {
            "views": [
                _fast_view_summary(meta, str(vid), view)
                for vid, view in (meta.get("views") or {}).items()
            ],
            "runs": [_fast_run_summary(run) for run in visible_runs],
        }
        print(json.dumps(payload, indent=2, default=_json_default))
        return 0
    print("Views:")
    for vid, view in (meta.get("views") or {}).items():
        info = _fast_view_summary(meta, str(vid), view)
        print(f"  {info['id']}\t{info['title']}\t{info['spectral_unit']}\t{', '.join(info['runs'])}")
    print("Runs:")
    for run in visible_runs:
        info = _fast_run_summary(run)
        print(
            f"  {info['id']}\t{info['nickname']}\t{info['kind']}\t"
            f"raw_x={info['raw_x_unit']}\tfit_peaks={info['fit_peak_count']}\t"
            f"rotation={info['angle_rotation_summary']}\t"
            f"sources={','.join(info['source_run_ids'])}\t{info['source_path']}"
        )
    return 0


_LAYER_PATTERNS = (
    ("mono", "1", re.compile(r"(?<![a-z0-9])(mono|monolayer|1l)(?![a-z0-9])", re.I)),
    ("bi", "2", re.compile(r"(?<![a-z0-9])(bi|bilayer|2l)(?![a-z0-9])", re.I)),
    ("tri", "3", re.compile(r"(?<![a-z0-9])(tri|trilayer|3l)(?![a-z0-9])", re.I)),
    ("4", "4", re.compile(r"(?<![a-z0-9])(4l|four|quad|tetra|4)(?![a-z0-9])", re.I)),
    ("bulk", "bulk", re.compile(r"(?<![a-z0-9])(bulk)(?![a-z0-9])", re.I)),
)


def _fast_parse_sample_context(view: Dict[str, Any], sources: Sequence[Dict[str, Any]], fit_header: Optional[Dict[str, Any]]) -> Dict[str, str]:
    source_md = (sources[0].get("metadata") if sources else {}) or {}
    candidates = [
        source_md.get("tag"),
        source_md.get("substance"),
        source_md.get("sample"),
        source_md.get("base"),
        source_md.get("stem"),
        view.get("title"),
        sources[0].get("nickname") if sources else "",
        fit_header.get("nickname") if fit_header else "",
    ]
    text = next((str(value).strip() for value in candidates if str(value or "").strip()), "")
    combined = " ".join(
        str(value or "")
        for value in [
            text,
            view.get("title"),
            " ".join(str(src.get("nickname", "")) for src in sources),
            fit_header.get("nickname", "") if fit_header else "",
        ]
    )

    layer = ""
    layer_number = ""
    layer_match = None
    for layer_name, number, pattern in _LAYER_PATTERNS:
        match = pattern.search(combined)
        if match:
            layer = layer_name
            layer_number = number
            layer_match = match
            break

    cleaned = re.sub(r"(?<![a-z0-9])(xx|xy|yx|yy|rr|rl|lr|ll)(?![a-z0-9])", " ", text, flags=re.I)
    for _layer_name, _number, pattern in _LAYER_PATTERNS:
        cleaned = pattern.sub(" ", cleaned)
    cleaned = re.sub(r"[_\-]+", " ", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip(" _-")
    if not cleaned and layer_match:
        cleaned = combined[: layer_match.start()].strip(" _-")
    if not cleaned:
        cleaned = str(view.get("title") or "").split()[0] if str(view.get("title") or "").split() else ""

    tag = cleaned or "unknown"
    variant = ""
    m = re.search(r"\((\d{4})\)", tag)
    if not m:
        m = re.search(r"([A-Za-z]+)(\d{4})$", tag)
    if m:
        variant = m.group(1) if len(m.groups()) == 1 else m.group(2)
    substance = re.sub(r"\(\d{4}\)", "", tag)
    substance = re.sub(r"(?<=[A-Za-z])\d{4}$", "", substance).strip(" _-")
    if substance in {"CPtS", "CsPtS"}:
        substance = "CsPtS"
    return {
        "tag": tag,
        "substance": substance or tag,
        "substance_variant": variant,
        "layer": layer,
        "layer_number": layer_number,
    }


def _fast_numeric_sort_key(header: Dict[str, Any]) -> Tuple[int, str]:
    text = f"{header.get('id', '')} {header.get('nickname', '')}"
    numbers = re.findall(r"\d+", text)
    return (int(numbers[-1]) if numbers else -1, str(header.get("id", "")))


def _fast_fit_param_index(meta: Dict[str, Any]) -> Tuple[Dict[frozenset, Dict[str, Any]], Dict[frozenset, int]]:
    grouped: Dict[frozenset, List[Dict[str, Any]]] = {}
    for header in (meta.get("runs") or {}).values():
        if header.get("run_type") != RunType.FIT_PARAMS.value:
            continue
        state = _fast_fit_state_from_metadata(header.get("metadata") or {})
        source_ids = _fast_source_run_ids(header)
        if not state or not source_ids:
            continue
        grouped.setdefault(frozenset(source_ids), []).append(header)
    latest = {}
    counts = {}
    for key, headers in grouped.items():
        headers.sort(key=_fast_numeric_sort_key)
        latest[key] = headers[-1]
        counts[key] = len(headers)
    return latest, counts


def _fast_visible_view_records(meta: Dict[str, Any]) -> List[Dict[str, Any]]:
    runs = meta.get("runs") or {}
    records = []
    for order, (vid, view) in enumerate((meta.get("views") or {}).items(), start=1):
        visible_ids = [
            str(rid)
            for rid in (view.get("run_ids") or [])
            if str(rid) in runs
            and not _fast_is_hidden_header(runs[str(rid)])
            and _fast_is_2d_header(runs[str(rid)])
        ]
        if not visible_ids:
            continue
        records.append({"id": str(vid), "order": order, "view": view, "run_ids": visible_ids})
    return records


def _fast_resolve_fit_for_view(
    meta: Dict[str, Any],
    view_record: Dict[str, Any],
    fit_index: Dict[frozenset, Dict[str, Any]],
    duplicate_counts: Dict[frozenset, int],
) -> Tuple[Optional[Dict[str, Any]], Optional[Dict[str, Any]], str, int]:
    runs = meta.get("runs") or {}
    source_key = frozenset(view_record["run_ids"])
    if source_key in fit_index:
        header = fit_index[source_key]
        state = _fast_fit_state_from_metadata(header.get("metadata") or {})
        return state, header, "fit parameters run", duplicate_counts.get(source_key, 1)

    for rid in view_record["run_ids"]:
        header = runs.get(rid)
        if not header:
            continue
        md = header.get("metadata") or {}
        state = _fast_fit_state_from_metadata(md)
        if not state:
            continue
        source_ids = [str(value) for value in (md.get("fit_params_source_run_ids") or md.get("source_run_ids") or [rid])]
        if frozenset(source_ids) == source_key:
            return state, header, "source run metadata", 1
    return None, None, "", 0


def _fast_param_triplet(params: Dict[str, Any], name: str) -> Tuple[float, float, float]:
    value = params.get(name)
    if isinstance(value, (list, tuple)):
        values = list(value)[:3]
    else:
        values = [value]
    while len(values) < 3:
        values.append(np.nan)
    out = []
    for item in values[:3]:
        try:
            out.append(float(item))
        except (TypeError, ValueError):
            out.append(np.nan)
    return out[0], out[1], out[2]


def _fast_unit_value_to_cm1(value: float, unit: str) -> float:
    return float(unit_to_cm1(np.asarray([value], dtype=float), unit)[0])


def _fast_peak_rows(path: str, x_min: Optional[float], x_max: Optional[float]) -> List[Dict[str, Any]]:
    meta = read_experiment_metadata(path)
    runs = meta.get("runs") or {}
    fit_index, duplicate_counts = _fast_fit_param_index(meta)
    rows: List[Dict[str, Any]] = []
    for view_record in _fast_visible_view_records(meta):
        view = view_record["view"]
        source_headers = [runs[rid] for rid in view_record["run_ids"] if rid in runs]
        state, fit_header, fit_source, duplicate_count = _fast_resolve_fit_for_view(meta, view_record, fit_index, duplicate_counts)
        if not state:
            continue
        unit = normalize_spectral_unit(state.get("unit", "cm-1"), "cm-1")
        context = _fast_parse_sample_context(view, source_headers, fit_header)
        for idx, peak in enumerate(state.get("peaks") or [], start=1):
            spec = peak.get("spec_params") or {}
            ang = peak.get("ang_params") or {}
            x0, x0_min, x0_max = _fast_param_triplet(spec, "x0")
            gamma, gamma_min, gamma_max = _fast_param_triplet(spec, "gamma")
            x0_cm1 = _fast_unit_value_to_cm1(x0, unit)
            if x_min is not None and x0_cm1 < float(x_min):
                continue
            if x_max is not None and x0_cm1 > float(x_max):
                continue
            gamma_cm1 = _fast_unit_value_to_cm1(gamma, unit)
            row = {
                "view_id": view_record["id"],
                "view_order": view_record["order"],
                "view_title": view.get("title", view_record["id"]),
                **context,
                "symmetry": peak.get("rule", ""),
                "peak_name": peak.get("name", ""),
                "peak_index": idx,
                "frequency_cm1": x0_cm1,
                "frequency_meV": float(cm1_to_unit(np.asarray([x0_cm1]), "meV")[0]),
                "frequency_min_cm1": _fast_unit_value_to_cm1(x0_min, unit),
                "frequency_max_cm1": _fast_unit_value_to_cm1(x0_max, unit),
                "gamma_cm1": gamma_cm1,
                "gamma_meV": float(cm1_to_unit(np.asarray([gamma_cm1]), "meV")[0]),
                "gamma_min_cm1": _fast_unit_value_to_cm1(gamma_min, unit),
                "gamma_max_cm1": _fast_unit_value_to_cm1(gamma_max, unit),
                "fit_source": fit_source,
                "fit_params_run_id": fit_header.get("id", "") if fit_header else "",
                "fit_params_run": fit_header.get("nickname", "") if fit_header else "",
                "duplicate_fit_params_for_sources": duplicate_count,
                "source_run_ids": ";".join(view_record["run_ids"]),
                "source_runs": ";".join(src.get("nickname", src.get("id", "")) for src in source_headers),
            }
            tensor_parts = []
            for name in ("a", "b", "d", "phi"):
                value, low, high = _fast_param_triplet(ang, name)
                prefix = "tensor_phi_deg" if name == "phi" else f"tensor_{name}"
                row[prefix] = value
                row[f"{prefix}_min"] = low
                row[f"{prefix}_max"] = high
                if np.isfinite(value):
                    tensor_parts.append(f"{name}={value:.12g}")
            row["raman_tensor_parameters"] = "; ".join(tensor_parts)
            rows.append(row)
    return rows


_FAST_TENSOR_COLUMNS = [
    "view_id", "view_order", "view_title", "tag", "substance", "substance_variant",
    "layer", "layer_number", "symmetry", "peak_name", "peak_index",
    "frequency_cm1", "frequency_meV", "frequency_min_cm1", "frequency_max_cm1",
    "gamma_cm1", "gamma_meV", "gamma_min_cm1", "gamma_max_cm1",
    "tensor_a", "tensor_a_min", "tensor_a_max",
    "tensor_b", "tensor_b_min", "tensor_b_max",
    "tensor_d", "tensor_d_min", "tensor_d_max",
    "tensor_phi_deg", "tensor_phi_deg_min", "tensor_phi_deg_max",
    "raman_tensor_parameters", "fit_source", "fit_params_run_id", "fit_params_run",
    "duplicate_fit_params_for_sources", "source_run_ids", "source_runs",
]


_FAST_PEAK_COLUMNS = [
    "view_id", "view_order", "view_title", "tag", "substance", "substance_variant",
    "layer", "layer_number", "symmetry", "peak_name", "peak_index",
    "frequency_cm1", "frequency_meV", "gamma_cm1", "gamma_meV",
    "fit_source", "fit_params_run_id", "fit_params_run",
    "duplicate_fit_params_for_sources", "source_run_ids", "source_runs",
]


def _write_dict_csv(path: str, columns: Sequence[str], rows: Sequence[Dict[str, Any]]) -> str:
    path = os.path.abspath(path)
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(columns), extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    return path


def cmd_export_fit_tensors(args) -> int:
    rows = _fast_peak_rows(args.experiment, args.x_min, args.x_max)
    output = _write_dict_csv(args.output, _FAST_TENSOR_COLUMNS, rows)
    print(json.dumps({"output": output, "rows": len(rows)}, indent=2, default=_json_default))
    return 0


def cmd_export_fit_peaks(args) -> int:
    rows = _fast_peak_rows(args.experiment, args.x_min, args.x_max)
    output = _write_dict_csv(args.output, _FAST_PEAK_COLUMNS, rows)
    print(json.dumps({"output": output, "rows": len(rows)}, indent=2, default=_json_default))
    return 0


_PEAK_SUMMARY_SUBSTANCE_ORDER = ["CsPdS", "CsNiS", "RPS", "CsPtS"]
_PEAK_SUMMARY_COLORS = {
    "CsPdS": "#2f318e",
    "CsNiS": "#c25b3e",
    "RPS": "#7c3a33",
    "CsPtS": "#968e66",
}
_PEAK_SUMMARY_LINEAGES = {
    "Ag1/E2g": {"marker": "o", "linestyle": "-"},
    "B1g/E2g": {"marker": "s", "linestyle": "--"},
    "Ag2/A1g": {"marker": "^", "linestyle": "-."},
    "E2g": {"marker": "o", "linestyle": "-"},
    "A1g": {"marker": "^", "linestyle": "-."},
}


def _peak_summary_unit_columns(fieldnames: Sequence[str]) -> Tuple[str, str, str]:
    fields = set(fieldnames)
    for unit, frequency, gamma in (
        ("meV", "frequency_meV", "gamma_meV"),
        ("cm-1", "frequency_cm1", "gamma_cm1"),
    ):
        if frequency in fields and gamma in fields:
            return unit, frequency, gamma
    raise ValueError("Peak summary CSV must contain frequency/gamma columns in meV or cm-1.")


def _peak_summary_graph_substance(label: str) -> str:
    text = str(label).strip()
    if text.startswith("CsPtS"):
        return "CsPtS"
    if text.startswith("RPS"):
        return "RPS"
    return text


def _peak_summary_lineages(substance: str, branch: str) -> List[str]:
    if branch == "E2g":
        return ["E2g"] if substance == "CsNiS" else ["Ag1/E2g", "B1g/E2g"]
    if branch == "Ag1":
        return ["Ag1/E2g"]
    if branch == "B1g":
        return ["B1g/E2g"]
    if branch == "A1g":
        return ["A1g"] if substance == "CsNiS" else ["Ag2/A1g"]
    if branch == "Ag2":
        return ["Ag2/A1g"]
    return []


def _read_peak_summary_csv(path: str, output_unit: str) -> Tuple[List[Dict[str, Any]], List[str]]:
    with open(path, newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames:
            raise ValueError("Peak summary CSV has no header.")
        required = {"substance_table", "layer_x (5=bulk)", "branch"}
        missing = required.difference(reader.fieldnames)
        if missing:
            raise ValueError(f"Peak summary CSV is missing columns: {', '.join(sorted(missing))}")
        source_unit, source_frequency, source_gamma = _peak_summary_unit_columns(reader.fieldnames)
        output_suffix = "cm1" if output_unit == "cm-1" else "meV"
        output_frequency = f"frequency_{output_suffix}"
        output_gamma = f"gamma_{output_suffix}"
        columns = [name for name in reader.fieldnames if name not in {source_frequency, source_gamma}]
        columns.extend([output_frequency, output_gamma])
        rows = []
        for row_order, source_row in enumerate(reader):
            frequency_cm1 = float(unit_to_cm1(float(source_row[source_frequency]), source_unit))
            gamma_cm1 = float(unit_to_cm1(abs(float(source_row[source_gamma])), source_unit))
            frequency = float(cm1_to_unit(frequency_cm1, output_unit))
            gamma = float(cm1_to_unit(gamma_cm1, output_unit))
            row = {
                key: value
                for key, value in source_row.items()
                if key not in {source_frequency, source_gamma}
            }
            row[output_frequency] = frequency
            row[output_gamma] = gamma
            row["_row_order"] = row_order
            row["_frequency"] = frequency
            row["_layer_x"] = float(source_row["layer_x (5=bulk)"])
            row["_substance"] = _peak_summary_graph_substance(source_row["substance_table"])
            rows.append(row)
    return rows, columns


def _write_peak_summary_csv(path: str, columns: Sequence[str], rows: Sequence[Dict[str, Any]]) -> str:
    cleaned = []
    for row in rows:
        cleaned.append({
            key: (f"{float(row[key]):.8f}".rstrip("0").rstrip(".") if key.startswith(("frequency_", "gamma_")) else row[key])
            for key in columns
        })
    return _write_dict_csv(path, columns, cleaned)


def _exclude_peak_summary_layers(
    rows: Sequence[Dict[str, Any]], excluded_layers: Optional[Sequence[float]]
) -> List[Dict[str, Any]]:
    excluded = {float(layer) for layer in (excluded_layers or [])}
    return [row for row in rows if float(row["_layer_x"]) not in excluded]


def _parse_peak_summary_bulk_sources(values: Optional[Sequence[str]]) -> Dict[str, str]:
    selections: Dict[str, str] = {}
    for value in values or []:
        if "=" not in value:
            raise ValueError("--bulk-source must use Substance=table-label syntax.")
        substance, table_label = (part.strip() for part in value.split("=", 1))
        if not substance or not table_label:
            raise ValueError("--bulk-source must use non-empty Substance=table-label values.")
        selections[_peak_summary_graph_substance(substance)] = table_label
    return selections


def _select_peak_summary_bulk_sources(
    rows: Sequence[Dict[str, Any]], selections: Dict[str, str]
) -> List[Dict[str, Any]]:
    if not selections:
        return list(rows)
    selected = []
    for row in rows:
        requested_label = selections.get(str(row["_substance"]))
        is_bulk = float(row["_layer_x"]) == 5.0
        if requested_label is not None and is_bulk and str(row.get("substance_table", "")) != requested_label:
            continue
        selected.append(row)
    return selected


def _filter_peak_summary_scope(
    rows: Sequence[Dict[str, Any]],
    substances: Optional[Sequence[str]],
    frequency_region: Optional[str],
    threshold: float,
) -> List[Dict[str, Any]]:
    selected_substances = {
        _peak_summary_graph_substance(substance) for substance in (substances or [])
    }
    selected = []
    for row in rows:
        if selected_substances and str(row["_substance"]) not in selected_substances:
            continue
        frequency = float(row["_frequency"])
        if frequency_region == "below" and not frequency < threshold:
            continue
        if frequency_region == "above" and not frequency > threshold:
            continue
        selected.append(row)
    return selected


def _monolayer_centered_window(
    rows: Sequence[Dict[str, Any]], window_width: float
) -> Tuple[float, float, float]:
    if window_width <= 0:
        raise ValueError("--center-monolayer-window must be positive.")
    centers = sorted({round(float(row["_frequency"]), 10) for row in rows if float(row["_layer_x"]) == 1.0})
    if len(centers) != 1:
        raise ValueError(
            "A monolayer-centered window requires exactly one monolayer peak after substance/region filtering."
        )
    center = centers[0]
    half_width = window_width / 2.0
    return center, center - half_width, center + half_width


def _draw_peak_summary(
    ax, rows: Sequence[Dict[str, Any]], line_width: float, marker_size: float
) -> None:
    entries = []
    for row in rows:
        for lineage in _peak_summary_lineages(row["_substance"], str(row.get("branch", ""))):
            entries.append((row, lineage))

    for substance in _PEAK_SUMMARY_SUBSTANCE_ORDER:
        color = _PEAK_SUMMARY_COLORS[substance]
        for lineage, style in _PEAK_SUMMARY_LINEAGES.items():
            group = [row for row, item_lineage in entries if row["_substance"] == substance and item_lineage == lineage]
            if not group:
                continue
            group.sort(key=lambda row: (row["_layer_x"], row["_row_order"], row["_frequency"]))
            primary_by_layer: Dict[float, Dict[str, Any]] = {}
            extras = []
            for row in group:
                if row["_layer_x"] not in primary_by_layer:
                    primary_by_layer[row["_layer_x"]] = row
                else:
                    extras.append(row)
            primary = [primary_by_layer[layer] for layer in sorted(primary_by_layer)]
            ax.plot(
                [row["_layer_x"] for row in primary],
                [row["_frequency"] for row in primary],
                color=color,
                marker=style["marker"],
                linestyle=style["linestyle"],
                linewidth=line_width,
                markersize=marker_size,
                markeredgewidth=line_width,
                alpha=0.9,
            )
            if extras:
                ax.scatter(
                    [row["_layer_x"] + 0.045 for row in extras],
                    [row["_frequency"] for row in extras],
                    facecolors="none",
                    edgecolors=color,
                    marker=style["marker"],
                    s=marker_size ** 2,
                    linewidths=line_width,
                    alpha=0.9,
                )


def _add_peak_summary_legends(
    ax,
    rows: Sequence[Dict[str, Any]],
    line_width: float,
    marker_size: float,
    font_size: float,
    position: str = "outside",
    inset_anchor_y: float = 0.985,
    inset_layout: str = "compact",
) -> None:
    present_substances = {
        str(row["_substance"]) for row in rows
    }
    substance_order = [
        substance for substance in _PEAK_SUMMARY_SUBSTANCE_ORDER if substance in present_substances
    ]
    present_lineages = {
        lineage
        for row in rows
        for lineage in _peak_summary_lineages(str(row["_substance"]), str(row.get("branch", "")))
    }
    substance_handles = [
        Line2D(
            [0],
            [0],
            color=_PEAK_SUMMARY_COLORS[substance],
            marker="o",
            lw=line_width,
            markersize=marker_size,
            markeredgewidth=line_width,
            label=substance,
        )
        for substance in substance_order
    ]
    lineage_handles = [
        Line2D(
            [0],
            [0],
            color="#333333",
            marker=style["marker"],
            linestyle=style["linestyle"],
            lw=line_width,
            markersize=marker_size,
            markeredgewidth=line_width,
            label=lineage,
        )
        for lineage, style in _PEAK_SUMMARY_LINEAGES.items()
        if lineage in present_lineages
    ]
    if position == "inset":
        blank = Line2D([0], [0], color="none", lw=0, label="")
        if inset_layout == "split":
            rows = max(len(substance_handles), len(lineage_handles))
            handles = (
                substance_handles
                + [blank] * (rows - len(substance_handles))
                + lineage_handles
                + [blank] * (rows - len(lineage_handles))
            )
            columns = 2
        else:
            columns = max(len(substance_handles), len(lineage_handles))
            handles = []
            for index in range(columns):
                handles.append(substance_handles[index] if index < len(substance_handles) else blank)
                handles.append(lineage_handles[index] if index < len(lineage_handles) else blank)
        legend = ax.legend(
            handles=handles,
            loc="upper center",
            bbox_to_anchor=(0.5, inset_anchor_y),
            borderaxespad=0.0,
            ncol=columns,
            fontsize=font_size,
            frameon=True,
            framealpha=0.95,
            facecolor="white",
            edgecolor="#808080",
            handlelength=1.5,
            handletextpad=0.4,
            columnspacing=0.9,
            labelspacing=0.3,
            borderpad=0.4,
        )
        legend.get_frame().set_linewidth(line_width)
        return

    substance_legend = ax.legend(
        handles=substance_handles,
        loc="upper left",
        bbox_to_anchor=(1.005, 1.0),
        borderaxespad=0.0,
        fontsize=font_size,
        frameon=False,
        handlelength=1.5,
        handletextpad=0.4,
        labelspacing=0.25,
    )
    ax.add_artist(substance_legend)
    lineage_anchor = max(0.12, 0.92 - 0.08 * len(substance_order))
    ax.legend(
        handles=lineage_handles,
        loc="upper left",
        bbox_to_anchor=(1.005, lineage_anchor),
        borderaxespad=0.0,
        fontsize=font_size,
        frameon=False,
        handlelength=1.5,
        handletextpad=0.4,
        labelspacing=0.25,
    )


def cmd_plot_peak_summary(args) -> int:
    unit = normalize_spectral_unit(args.unit)
    rows, columns = _read_peak_summary_csv(args.input_csv, unit)
    if not rows:
        raise ValueError("Peak summary CSV has no data rows.")
    bulk_sources = _parse_peak_summary_bulk_sources(args.bulk_source)
    selected_rows = _select_peak_summary_bulk_sources(rows, bulk_sources)
    selected_substances = [
        _peak_summary_graph_substance(substance) for substance in (args.substance or [])
    ]
    selected_rows = _filter_peak_summary_scope(
        selected_rows,
        selected_substances,
        args.frequency_region,
        float(args.region_threshold),
    )
    plot_rows = _exclude_peak_summary_layers(selected_rows, args.exclude_layer)
    if not plot_rows:
        raise ValueError("No peak-summary rows remain after applying the requested filters.")

    csv_output = _write_peak_summary_csv(args.csv_output, columns, plot_rows) if args.csv_output else None
    output = os.path.abspath(args.output)
    os.makedirs(os.path.dirname(output) or ".", exist_ok=True)
    tick_step = float(args.tick_step if args.tick_step is not None else (10.0 if unit == "cm-1" else 2.0))
    if tick_step <= 0:
        raise ValueError("--tick-step must be positive.")
    if args.width_cm <= 0 or args.height_cm <= 0:
        raise ValueError("--width-cm and --height-cm must be positive.")
    if args.line_width <= 0 or args.tick_length < 0:
        raise ValueError("--line-width must be positive and --tick-length must be nonnegative.")
    if not 0.0 < args.legend_anchor_y <= 1.0:
        raise ValueError("--legend-anchor-y must be greater than 0 and no greater than 1.")

    frequencies = np.asarray([row["_frequency"] for row in plot_rows], dtype=float)
    monolayer_center = None
    if args.center_monolayer_window is not None:
        if args.ylim is not None:
            raise ValueError("Use either --ylim or --center-monolayer-window, not both.")
        if args.frequency_region is None or len(set(selected_substances)) != 1:
            raise ValueError(
                "--center-monolayer-window requires one --substance and --frequency-region below/above."
            )
        monolayer_center, y_min, y_max = _monolayer_centered_window(
            plot_rows, float(args.center_monolayer_window)
        )
    elif args.ylim is not None:
        y_min, y_max = sorted((float(args.ylim[0]), float(args.ylim[1])))
    else:
        margin = tick_step * 0.35
        y_min = np.floor((float(np.nanmin(frequencies)) - margin) / tick_step) * tick_step
        y_max = np.ceil((float(np.nanmax(frequencies)) + margin) / tick_step) * tick_step

    rc = {
        "font.family": args.font_family,
        "font.size": args.font_size,
        "axes.labelsize": args.font_size,
        "axes.titlesize": args.font_size + 1.0,
        "axes.linewidth": args.line_width,
        "xtick.labelsize": args.font_size,
        "ytick.labelsize": args.font_size,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }
    with matplotlib.rc_context(rc):
        fig, ax = plt.subplots(figsize=(args.width_cm / 2.54, args.height_cm / 2.54))
        _draw_peak_summary(ax, plot_rows, args.line_width, args.marker_size)
        ax.set_xlim(0.75, 5.25)
        ax.set_xticks([1, 2, 3, 4, 5])
        ax.set_xticklabels(["1L", "2L", "3L", "4L", "Bulk"])
        ax.set_ylim(y_min, y_max)
        ax.set_yticks(np.arange(y_min, y_max + tick_step * 0.5, tick_step))
        if monolayer_center is not None:
            ax.yaxis.set_major_formatter(FormatStrFormatter("%.1f"))
        ax.tick_params(axis="both", which="major", length=args.tick_length, width=args.line_width, pad=1.5)
        for spine in ax.spines.values():
            spine.set_linewidth(args.line_width)
        if args.grid:
            ax.grid(True, color="#d9d9d9", linewidth=args.line_width, alpha=0.85)
        else:
            ax.grid(False)
        ax.set_xlabel("Layer number", labelpad=2.0)
        ax.set_ylabel("Frequency (meV)" if unit == "meV" else "Frequency (cm-1)", labelpad=2.0)
        default_title = "Raman peak positions"
        if len(set(selected_substances)) == 1 and args.frequency_region:
            operator = "<" if args.frequency_region == "below" else ">"
            default_title = f"{selected_substances[0]} ({operator}{args.region_threshold:g} cm-1)"
        ax.set_title(args.title or default_title, pad=2.0)
        _add_peak_summary_legends(
            ax,
            plot_rows,
            args.line_width,
            args.marker_size,
            max(4.0, args.font_size - 0.5),
            args.legend_position,
            args.legend_anchor_y,
            args.legend_layout,
        )
        right = 1.0 if args.legend_position == "inset" else 0.70
        fig.tight_layout(rect=[0.0, 0.0, right, 1.0], pad=0.3)
        fig.savefig(output, format="pdf")
        plt.close(fig)

    print(json.dumps({
        "input": os.path.abspath(args.input_csv),
        "csv_output": csv_output,
        "output": output,
        "rows": len(rows),
        "plotted_rows": len(plot_rows),
        "excluded_layers": [float(layer) for layer in (args.exclude_layer or [])],
        "bulk_sources": bulk_sources,
        "substances": selected_substances,
        "frequency_region": args.frequency_region,
        "region_threshold": args.region_threshold,
        "monolayer_center": monolayer_center,
        "window_width": args.center_monolayer_window,
        "unit": unit,
        "font_family": args.font_family,
        "figure_size_cm": [args.width_cm, args.height_cm],
        "line_width_pt": args.line_width,
        "tick_length_pt": args.tick_length,
        "grid": bool(args.grid),
        "tick_step": tick_step,
        "legend_position": args.legend_position,
        "legend_anchor_y": args.legend_anchor_y,
        "legend_layout": args.legend_layout,
        "ylim": [y_min, y_max],
    }, indent=2, default=_json_default))
    return 0


_TENSOR_RATIO_COLUMNS = [
    "substance", "layer", "layer_x", "view_title", "symmetry", "peak_name",
    "frequency_cm1", "tensor_a", "tensor_b", "a_over_b", "ratio_std",
    "ratio_expression", "ratio_source", "uncertainty_method",
]

_TENSOR_RATIO_UNCERTAINTY_COLUMNS = [
    "substance", "layer", "layer_number", "layer_x", "view_title", "symmetry",
    "peak_name", "peak_index", "frequency_cm1", "fit_params_run_id", "fit_params_run",
    "source_run_ids", "source_runs", "tensor_a_global", "tensor_b_global",
    "tensor_a", "tensor_a_std", "tensor_b", "tensor_b_std", "tensor_ab_cov",
    "tensor_phi_deg", "tensor_phi_deg_std", "a_over_b", "ratio_std",
    "ratio_expression", "a_over_b_global", "ratio_expression_global", "rowfit_minus_global",
    "ratio_source", "uncertainty_method", "n_observations",
    "degrees_of_freedom", "residual_rmse", "row_data_source", "cache_run_ids",
    "row_validation_status", "row_validation_message",
]

_TENSOR_ROW_AUDIT_COLUMNS = [
    "substance", "layer", "layer_number", "view_title", "symmetry", "peak_name",
    "peak_index", "frequency_cm1", "fit_params_run_id", "source_run_id",
    "source_run", "configuration", "angle_deg", "area", "gamma_cm1", "height",
    "row_data_source", "cache_run_id",
]

_GLOBAL_BOOTSTRAP_COLUMNS = [
    "substance", "layer", "layer_number", "layer_x", "view_title", "symmetry",
    "peak_name", "peak_index", "frequency_cm1", "fit_params_run_id", "fit_params_run",
    "source_run_ids", "source_runs", "tensor_a_global", "tensor_b_global",
    "tensor_phi_deg_global", "tensor_a", "tensor_b", "a_over_b", "ratio_std",
    "ratio_expression", "ratio_source", "uncertainty_method", "bootstrap_mean",
    "bootstrap_bias", "bias_over_sigma", "bootstrap_percentile_2_5",
    "bootstrap_percentile_97_5", "bootstrap_replicates_requested",
    "bootstrap_replicates_successful", "bootstrap_success_rate", "bootstrap_seed",
    "n_observations", "n_angle_clusters", "reference_scale",
    "reference_phi_shift_deg", "reference_rmse", "phase_boundary_hit",
    "ratio_boundary_fraction", "diagnostic_warning", "row_data_source",
    "cache_run_ids", "row_validation_status", "row_validation_message",
]

_GLOBAL_BOOTSTRAP_DRAW_COLUMNS = [
    "substance", "layer", "view_title", "peak_name", "peak_index",
    "fit_params_run_id", "bootstrap_seed", "replicate", "ratio",
    "fit_success",
]

_TENSOR_FREQUENCY_COLUMNS = [
    "substance", "layer", "layer_x", "view_title", "symmetry", "branch",
    "peak_name", "frequency_cm1",
]


def _optional_float(value: Any) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return np.nan
    return result if np.isfinite(result) else np.nan


def _tensor_fit_parameter(fit: polar_area_fitting.TensorFit, name: str) -> Tuple[float, float, Optional[int]]:
    try:
        index = fit.param_names.index(name)
    except ValueError:
        return np.nan, np.nan, None
    value = float(fit.params[index])
    std = np.nan
    if fit.param_std is not None and index < len(fit.param_std):
        std = _optional_float(fit.param_std[index])
    return value, std, index


def _conditional_tensor_ratio(
    a: float,
    b: float,
    covariance: Optional[np.ndarray],
    a_index: Optional[int],
    b_index: Optional[int],
) -> Tuple[float, float, str]:
    if not np.isfinite(a) or not np.isfinite(b):
        return np.nan, np.nan, ""
    if abs(b) > abs(a):
        if a == 0.0:
            return np.nan, np.nan, "b/a"
        ratio = b / a
        gradient = np.asarray([-b / (a * a), 1.0 / a], dtype=float)
        expression = "b/a"
    else:
        if b == 0.0:
            return np.nan, np.nan, "a/b"
        ratio = a / b
        gradient = np.asarray([1.0 / b, -a / (b * b)], dtype=float)
        expression = "a/b"

    ratio_std = np.nan
    if covariance is not None and a_index is not None and b_index is not None:
        covariance = np.asarray(covariance, dtype=float)
        indices = np.asarray([a_index, b_index], dtype=int)
        if covariance.ndim == 2 and np.max(indices) < covariance.shape[0]:
            subcovariance = covariance[np.ix_(indices, indices)]
            if np.isfinite(subcovariance).all():
                variance = float(gradient @ subcovariance @ gradient)
                tolerance = np.finfo(float).eps * max(1.0, float(np.max(np.abs(subcovariance)))) * 100.0
                if variance >= -tolerance:
                    ratio_std = float(np.sqrt(max(0.0, variance)))
    return float(ratio), ratio_std, expression


def _selected_tensor_uncertainty_rows(args) -> List[Dict[str, Any]]:
    frequency_ranges = args.frequency_range or [[270.0, 310.0], [350.0, 390.0]]
    bulk_views = _parse_peak_summary_bulk_sources(args.bulk_view)
    rows: List[Dict[str, Any]] = []
    for frequency_range in frequency_ranges:
        low, high = sorted(map(float, frequency_range))
        band = "low" if (low + high) * 0.5 < 340.0 else "high"
        rows.extend(_read_tensor_ratio_rows(
            args.tensor_csv,
            band,
            [low, high],
            args.exclude_layer,
            bulk_views,
        ))
    unique: Dict[Tuple[str, str, str], Dict[str, Any]] = {}
    for row in rows:
        key = (
            str(row.get("_fit_params_run_id", "")),
            str(row.get("_peak_index", "")),
            str(row.get("view_title", "")),
        )
        unique.setdefault(key, row)
    return list(unique.values())


def _row_groups_for_uncertainty(
    exp: ExperimentSet,
    params_run: Run,
    targets: Sequence[float],
    *,
    cache_only: bool,
) -> Tuple[List[List[polar_area_fitting.RowPeakFit]], str, List[str], str, str]:
    metadata = params_run.metadata or {}
    fit_state = metadata.get("fit_state") or metadata.get("map_fit_state")
    if not isinstance(fit_state, dict):
        raise ValueError(f"{params_run.nickname} has no fit state.")
    source_ids = metadata.get("source_run_ids") or metadata.get("fit_params_source_run_ids") or []
    sources = [exp.get_run(str(run_id)) for run_id in source_ids]
    sources = [run for run in sources if run is not None and run.is_2d]
    if not sources:
        raise ValueError(f"{params_run.nickname} has no linked 2D source runs.")

    saved_results = [
        has_saved_row_fit_result(exp, source, fit_state, params_run=params_run)
        for source in sources
    ]
    if all(saved_results):
        groups = cached_row_fits_for_runs(exp, sources, fit_state, targets, params_run=params_run)
        refs = [f"compact:{params_run.id}:{source.id}" for source in sources]
        return groups, "exact saved row result", refs, "cached", ""
    if cache_only:
        missing = [source.nickname for source, saved in zip(sources, saved_results) if not saved]
        raise ValueError(f"No exact current row-fit result for: {', '.join(missing)}")

    context = row_fit_cli.prepare_context(exp, params_run)
    row_fit_cli.finalize_config(context)
    success, message, results = row_fit_cli.validate_context(context)
    if not results:
        raise ValueError(f"Row-fit rebuild failed for {params_run.nickname}: {message}")
    context_targets, context_groups = row_fit_cli.validation_row_fit_groups(context)
    groups = [
        context_groups[int(np.argmin(np.abs(np.asarray(context_targets, dtype=float) - float(target))))]
        for target in targets
    ]
    status = "rebuilt-success" if success else "rebuilt-partial"
    return groups, "rebuilt in memory from saved row-fit settings", [], status, str(message)


def _append_row_audit(
    output: List[Dict[str, Any]],
    selected: Dict[str, Any],
    group: Sequence[polar_area_fitting.RowPeakFit],
    *,
    row_data_source: str,
    cache_ids: Sequence[str],
) -> None:
    cache_by_run = {row_fit.run_id: cache_id for row_fit, cache_id in zip(group, cache_ids)}
    for row_fit in group:
        count = min(len(row_fit.angles), len(row_fit.areas), len(row_fit.gammas), len(row_fit.heights))
        for index in range(count):
            output.append({
                "substance": selected["substance"],
                "layer": selected["layer"],
                "layer_number": selected.get("_layer_number", ""),
                "view_title": selected["view_title"],
                "symmetry": selected["symmetry"],
                "peak_name": selected["peak_name"],
                "peak_index": selected.get("_peak_index", ""),
                "frequency_cm1": selected["frequency_cm1"],
                "fit_params_run_id": selected.get("_fit_params_run_id", ""),
                "source_run_id": row_fit.run_id,
                "source_run": row_fit.run_label,
                "configuration": row_fit.config,
                "angle_deg": float(row_fit.angles[index]),
                "area": float(row_fit.areas[index]),
                "gamma_cm1": float(row_fit.gammas[index]),
                "height": float(row_fit.heights[index]),
                "row_data_source": row_data_source,
                "cache_run_id": cache_by_run.get(row_fit.run_id, ""),
            })


def _run_from_fast_header(
    experiment_path: str,
    header: Dict[str, Any],
    arrays: Sequence[str] = (),
) -> Run:
    values = read_run_arrays(experiment_path, str(header["id"]), arrays) if arrays else {}
    try:
        run_type = RunType(str(header.get("run_type", RunType.OTHER.value)))
    except ValueError:
        run_type = RunType.OTHER
    return Run(
        id=str(header["id"]),
        source_path=str(header.get("source_path", "")),
        source_mtime=header.get("source_mtime"),
        wl_nm=values.get("wl_nm"),
        shift_cm1=values.get("shift_cm1"),
        energy_eV=values.get("energy_eV"),
        intensity=values.get("intensity"),
        intensity_2d=values.get("intensity_2d"),
        angle_values=values.get("angle_values"),
        intensity_unit=str(header.get("intensity_unit", "au")),
        angle_unit=str(header.get("angle_unit", "deg")),
        metadata=copy.deepcopy(header.get("metadata") or {}),
        run_type=run_type,
    )


def _numeric_run_id_key(run_id: str) -> Tuple[int, str]:
    values = re.findall(r"\d+", str(run_id))
    return (int(values[-1]) if values else -1, str(run_id))


def _minimal_uncertainty_experiment(
    experiment_path: str,
    metadata: Dict[str, Any],
    params_id: str,
) -> Tuple[ExperimentSet, Run, int]:
    headers = metadata.get("runs") or {}
    params_header = headers.get(str(params_id))
    if params_header is None:
        raise ValueError(f"FitParams run '{params_id}' is missing from HDF5 metadata.")
    params_run = _run_from_fast_header(experiment_path, params_header)
    if params_run.run_type != RunType.FIT_PARAMS:
        raise ValueError(f"Run '{params_id}' is not a FitParams run.")

    exp = ExperimentSet(
        id=str(metadata.get("experiment_id") or "minimal-uncertainty"),
        metadata=copy.deepcopy(metadata.get("metadata") or {}),
        next_run_index=int(metadata.get("next_run_index", 1)),
    )
    exp.runs[params_run.id] = params_run
    params_metadata = params_run.metadata or {}
    fit_state = params_metadata.get("fit_state") or params_metadata.get("map_fit_state")
    source_ids = params_metadata.get("source_run_ids") or params_metadata.get("fit_params_source_run_ids") or []
    if not isinstance(fit_state, dict) or not source_ids:
        raise ValueError(f"FitParams run '{params_id}' has no usable fit state or source runs.")

    arrays_loaded = 0
    source_runs: List[Run] = []
    for source_id in source_ids:
        source_header = headers.get(str(source_id))
        if source_header is None:
            raise ValueError(f"Source run '{source_id}' linked by '{params_id}' is missing.")
        source_run = _run_from_fast_header(
            experiment_path,
            source_header,
            ("shift_cm1", "intensity_2d", "angle_values"),
        )
        if not source_run.is_2d:
            raise ValueError(f"Source run '{source_id}' is not a complete 2D run.")
        exp.runs[source_run.id] = source_run
        source_runs.append(source_run)
        arrays_loaded += 1

    for source_run in source_runs:
        expected_hash = row_fit_cache_hash(source_run, fit_state)
        candidates = []
        for header in headers.values():
            cache_metadata = header.get("metadata") or {}
            if cache_metadata.get("fit_cache_kind") != "row_reconstruction":
                continue
            if str(cache_metadata.get("source_run_id", "")) != source_run.id:
                continue
            if str(cache_metadata.get("fit_state_hash", "")) != expected_hash:
                continue
            candidates.append(header)
        if not candidates:
            continue
        cache_header = max(candidates, key=lambda item: _numeric_run_id_key(str(item["id"])))
        cache_run = _run_from_fast_header(
            experiment_path,
            cache_header,
            ("shift_cm1", "intensity_2d", "angle_values"),
        )
        exp.runs[cache_run.id] = cache_run
        arrays_loaded += 1
    return exp, params_run, arrays_loaded


def _stable_bootstrap_seed(base_seed: int, params_id: str, peak_index: Any) -> int:
    payload = f"{int(base_seed)}:{params_id}:{peak_index}".encode("utf-8")
    digest = hashlib.sha256(payload).digest()
    return int((int(base_seed) + int.from_bytes(digest[:4], "big")) % (2 ** 32))


def _global_bootstrap_warning(
    result: polar_area_fitting.GlobalCenteredRatioBootstrap,
    success_rate: float,
) -> str:
    warnings = []
    if success_rate < 0.95:
        warnings.append("bootstrap success below 95%")
    if result.ratio_std > 0.0 and abs(result.bootstrap_bias) > 0.5 * result.ratio_std:
        warnings.append("bootstrap bias exceeds 0.5 sigma")
    if result.phase_boundary_hit:
        warnings.append("reference phase reached local boundary")
    if result.ratio_boundary_fraction > 0.05:
        warnings.append("more than 5% of draws reached |ratio|=1 boundary")
    return "; ".join(warnings)


def _write_global_bootstrap_report(
    path: str,
    summary_rows: Sequence[Dict[str, Any]],
    *,
    experiment_path: str,
    tensor_csv: str,
    replicates: int,
    base_seed: int,
    numeric_contexts: int,
    metadata_only_contexts: int,
    exact_cache_contexts: int,
    rebuilt_contexts: int,
) -> str:
    path = os.path.abspath(path)
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    fitted = [row for row in summary_rows if row.get("ratio_expression") != "symmetry"]
    fixed = [row for row in summary_rows if row.get("ratio_expression") == "symmetry"]
    warnings = [row for row in fitted if row.get("diagnostic_warning")]
    lines = [
        "# Global-centered Raman tensor ratio bootstrap report",
        "",
        "## Scope",
        "",
        f"- Source experiment: `{os.path.abspath(experiment_path)}`",
        f"- Tensor metadata CSV: `{os.path.abspath(tensor_csv)}`",
        "- Processing: local only; no network or external service was used.",
        "- The source experiment was read only and was not modified.",
        "- Only selected source-run arrays were loaded, one FitParams context at a time.",
        "",
        "## Method",
        "",
        "Each D2h Ag tensor was converted to the symmetry-equivalent representation with the larger-magnitude component in the numerator. The plotted center is the saved global 2D-fit ratio and is never replaced by a row-area optimum.",
        "",
        "A reference angular-area model fixed that global ratio while fitting only a positive nuisance scale and a local phase adjustment within +/-45 degrees. One Rademacher weight was drawn per measured angle and shared by XX/YX observations at that angle. The row residuals were sign-flipped by cluster, added to the fixed-global-ratio reference, and the ratio, scale, and local phase were refitted.",
        "",
        f"The plotted error is the sample standard deviation of {int(replicates)} bootstrap ratios (1 sigma). The deterministic base seed is `{int(base_seed)}`; each peak receives a stable derived seed.",
        "",
        "D6h E2g and A1g ratios are fixed to -1 and +1, respectively, with zero uncertainty.",
        "",
        "## Processing summary",
        "",
        f"- Total plotted rows: {len(summary_rows)}",
        f"- Bootstrapped D2h Ag rows: {len(fitted)}",
        f"- Symmetry-fixed rows: {len(fixed)}",
        f"- Numeric FitParams contexts: {numeric_contexts}",
        f"- Metadata-only FitParams contexts: {metadata_only_contexts}",
        f"- Exact-cache numeric contexts: {exact_cache_contexts}",
        f"- Rebuilt-in-memory numeric contexts: {rebuilt_contexts}",
        f"- Diagnostic warnings: {len(warnings)}",
        "",
        "## Compact results",
        "",
        "| Substance | Layer | Peak | Global ratio | 1 sigma | Bootstrap mean | Success | Warning |",
        "|---|---:|---|---:|---:|---:|---:|---|",
    ]
    for row in summary_rows:
        ratio = float(row["a_over_b"])
        sigma = float(row["ratio_std"])
        layer_text = str(row.get("layer") or "Bulk")
        if layer_text.lower() == "bulk":
            layer_text = "Bulk"
        mean = row.get("bootstrap_mean", "")
        mean_text = "" if mean in (None, "") else f"{float(mean):.6g}"
        success = row.get("bootstrap_success_rate", "")
        success_text = "" if success in (None, "") else f"{100.0 * float(success):.1f}%"
        warning_text = str(row.get("diagnostic_warning", "")).replace("|", "\\|")
        lines.append(
            f"| {row['substance']} | {layer_text} | {row['peak_name']} | "
            f"{ratio:.6g} | {sigma:.6g} | {mean_text} | {success_text} | "
            f"{warning_text} |"
        )
    lines.extend([
        "",
        "## Interpretation limits",
        "",
        "The uncertainty describes angular row-fit variability around the saved global tensor ratio. Row-area observations are equally weighted because the cache does not store per-area standard errors. The result does not include detector calibration uncertainty or uncertainty from external sample metadata.",
        "",
    ])
    with open(path, "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines))
    return path


def _cmd_export_global_centered_bootstrap(args) -> int:
    if args.bootstrap_replicates < 2:
        raise ValueError("--bootstrap-replicates must be at least 2.")
    if not 0.0 < args.bootstrap_min_success <= 1.0:
        raise ValueError("--bootstrap-min-success must be in (0, 1].")
    metadata = read_experiment_metadata(args.experiment)
    selected_rows = _selected_tensor_uncertainty_rows(args)
    if not selected_rows:
        raise ValueError("No tensor-ratio rows remain after filtering.")

    selected_by_params: Dict[str, List[Dict[str, Any]]] = {}
    for row in selected_rows:
        params_id = str(row.get("_fit_params_run_id", ""))
        if not params_id:
            raise ValueError(f"Tensor row {row['view_title']} / {row['peak_name']} has no FitParams run ID.")
        selected_by_params.setdefault(params_id, []).append(row)

    summary_rows: List[Dict[str, Any]] = []
    audit_rows: List[Dict[str, Any]] = []
    draw_rows: List[Dict[str, Any]] = []
    numeric_contexts = metadata_only_contexts = 0
    exact_cache_contexts = rebuilt_contexts = arrays_loaded = 0

    for params_id, selections in selected_by_params.items():
        fitted_selections = [row for row in selections if row["symmetry"] == "D2h_Ag"]
        for selected in selections:
            if selected["symmetry"] == "D2h_Ag":
                continue
            ratio = -1.0 if selected["symmetry"] == "D6h_E2g" else 1.0
            summary_rows.append({
                "substance": selected["substance"],
                "layer": selected["layer"],
                "layer_number": selected.get("_layer_number", ""),
                "layer_x": selected["layer_x"],
                "view_title": selected["view_title"],
                "symmetry": selected["symmetry"],
                "peak_name": selected["peak_name"],
                "peak_index": selected.get("_peak_index", ""),
                "frequency_cm1": selected["frequency_cm1"],
                "fit_params_run_id": params_id,
                "fit_params_run": selected.get("_fit_params_run", ""),
                "source_run_ids": selected.get("_source_run_ids", ""),
                "source_runs": selected.get("_source_runs", ""),
                "tensor_a_global": selected["tensor_a"],
                "tensor_b_global": selected["tensor_b"],
                "tensor_phi_deg_global": _optional_float(selected.get("_tensor_phi_deg")),
                "tensor_a": selected["tensor_a"],
                "tensor_b": selected["tensor_b"],
                "a_over_b": ratio,
                "ratio_std": 0.0,
                "ratio_expression": "symmetry",
                "ratio_source": f"{selected['symmetry'].split('_')[-1]} symmetry limit",
                "uncertainty_method": "fixed by symmetry; no fitted ratio degree of freedom",
                "diagnostic_warning": "",
                "row_data_source": "metadata only",
                "row_validation_status": "not required",
                "_row_order": selected["_row_order"],
            })
        if not fitted_selections:
            metadata_only_contexts += 1
            continue

        numeric_contexts += 1
        exp, params_run, context_arrays = _minimal_uncertainty_experiment(
            args.experiment,
            metadata,
            params_id,
        )
        arrays_loaded += context_arrays
        targets = [float(row["frequency_cm1"]) for row in fitted_selections]
        groups, row_data_source, cache_ids, validation_status, validation_message = _row_groups_for_uncertainty(
            exp,
            params_run,
            targets,
            cache_only=bool(args.cache_only),
        )
        if validation_status not in {"cached", "rebuilt-success"}:
            raise ValueError(f"Incomplete row validation for {params_run.nickname}: {validation_message}")
        if cache_ids:
            exact_cache_contexts += 1
        else:
            rebuilt_contexts += 1

        for selected, group in zip(fitted_selections, groups):
            global_a = float(selected["tensor_a"])
            global_b = float(selected["tensor_b"])
            global_phi = float(selected.get("_tensor_phi_deg", np.nan))
            peak_seed = _stable_bootstrap_seed(args.bootstrap_seed, params_id, selected.get("_peak_index", ""))
            result = polar_area_fitting.global_centered_ratio_bootstrap(
                group,
                global_a=global_a,
                global_b=global_b,
                global_phi_deg=global_phi,
                replicates=args.bootstrap_replicates,
                seed=peak_seed,
                phi_window_deg=args.bootstrap_phi_window,
            )
            success_rate = result.successful / result.attempted
            if success_rate < args.bootstrap_min_success:
                raise ValueError(
                    f"Bootstrap success rate for {selected['view_title']} / {selected['peak_name']} "
                    f"was {success_rate:.1%}, below {args.bootstrap_min_success:.1%}."
                )
            warning = _global_bootstrap_warning(result, success_rate)
            bias_over_sigma = (
                result.bootstrap_bias / result.ratio_std if result.ratio_std > 0.0 else np.nan
            )
            summary_rows.append({
                "substance": selected["substance"],
                "layer": selected["layer"],
                "layer_number": selected.get("_layer_number", ""),
                "layer_x": selected["layer_x"],
                "view_title": selected["view_title"],
                "symmetry": selected["symmetry"],
                "peak_name": selected["peak_name"],
                "peak_index": selected.get("_peak_index", ""),
                "frequency_cm1": selected["frequency_cm1"],
                "fit_params_run_id": params_id,
                "fit_params_run": selected.get("_fit_params_run", params_run.nickname),
                "source_run_ids": selected.get("_source_run_ids", ""),
                "source_runs": selected.get("_source_runs", ""),
                "tensor_a_global": global_a,
                "tensor_b_global": global_b,
                "tensor_phi_deg_global": global_phi,
                "tensor_a": global_a,
                "tensor_b": global_b,
                "a_over_b": result.global_ratio,
                "ratio_std": result.ratio_std,
                "ratio_expression": result.ratio_expression,
                "ratio_source": "saved global 2D fit",
                "uncertainty_method": "global-centered angle-cluster wild bootstrap of row-fit areas",
                "bootstrap_mean": result.bootstrap_mean,
                "bootstrap_bias": result.bootstrap_bias,
                "bias_over_sigma": bias_over_sigma,
                "bootstrap_percentile_2_5": result.percentile_2_5,
                "bootstrap_percentile_97_5": result.percentile_97_5,
                "bootstrap_replicates_requested": result.attempted,
                "bootstrap_replicates_successful": result.successful,
                "bootstrap_success_rate": success_rate,
                "bootstrap_seed": peak_seed,
                "n_observations": result.n_observations,
                "n_angle_clusters": result.n_angle_clusters,
                "reference_scale": result.reference_scale,
                "reference_phi_shift_deg": result.reference_phi_shift_deg,
                "reference_rmse": result.reference_rmse,
                "phase_boundary_hit": result.phase_boundary_hit,
                "ratio_boundary_fraction": result.ratio_boundary_fraction,
                "diagnostic_warning": warning,
                "row_data_source": row_data_source,
                "cache_run_ids": ";".join(cache_ids),
                "row_validation_status": validation_status,
                "row_validation_message": validation_message,
                "_row_order": selected["_row_order"],
            })
            for replicate, draw in enumerate(result.draws, start=1):
                draw_rows.append({
                    "substance": selected["substance"],
                    "layer": selected["layer"],
                    "view_title": selected["view_title"],
                    "peak_name": selected["peak_name"],
                    "peak_index": selected.get("_peak_index", ""),
                    "fit_params_run_id": params_id,
                    "bootstrap_seed": peak_seed,
                    "replicate": replicate,
                    "ratio": draw,
                    "fit_success": bool(np.isfinite(draw)),
                })
            _append_row_audit(
                audit_rows,
                selected,
                group,
                row_data_source=row_data_source,
                cache_ids=cache_ids,
            )

    summary_rows.sort(key=lambda row: int(row.get("_row_order", 0)))
    output = _write_dict_csv(args.output, _GLOBAL_BOOTSTRAP_COLUMNS, summary_rows)
    row_output = _write_dict_csv(args.row_output, _TENSOR_ROW_AUDIT_COLUMNS, audit_rows)
    output_root, _output_ext = os.path.splitext(os.path.abspath(args.output))
    draw_output = _write_dict_csv(
        args.draw_output or f"{output_root}_draws.csv",
        _GLOBAL_BOOTSTRAP_DRAW_COLUMNS,
        draw_rows,
    )
    report_output = _write_global_bootstrap_report(
        args.report_output or f"{output_root}_report.md",
        summary_rows,
        experiment_path=args.experiment,
        tensor_csv=args.tensor_csv,
        replicates=args.bootstrap_replicates,
        base_seed=args.bootstrap_seed,
        numeric_contexts=numeric_contexts,
        metadata_only_contexts=metadata_only_contexts,
        exact_cache_contexts=exact_cache_contexts,
        rebuilt_contexts=rebuilt_contexts,
    )
    warning_count = sum(bool(row.get("diagnostic_warning")) for row in summary_rows)
    print(json.dumps({
        "method": "global-centered-bootstrap",
        "output": output,
        "row_output": row_output,
        "draw_output": draw_output,
        "report_output": report_output,
        "summary_rows": len(summary_rows),
        "bootstrap_draw_rows": len(draw_rows),
        "numeric_contexts": numeric_contexts,
        "metadata_only_contexts": metadata_only_contexts,
        "selected_run_arrays_loaded": arrays_loaded,
        "exact_cache_contexts": exact_cache_contexts,
        "rebuilt_contexts": rebuilt_contexts,
        "diagnostic_warnings": warning_count,
    }, indent=2))
    return 0


def cmd_export_tensor_ratio_errors(args) -> int:
    if args.method == "global-centered-bootstrap":
        return _cmd_export_global_centered_bootstrap(args)
    exp = _load_experiment(args.experiment)
    selected_rows = _selected_tensor_uncertainty_rows(args)
    if not selected_rows:
        raise ValueError("No tensor-ratio rows remain after filtering.")

    selected_by_params: Dict[str, List[Dict[str, Any]]] = {}
    for row in selected_rows:
        params_id = str(row.get("_fit_params_run_id", ""))
        if not params_id:
            raise ValueError(f"Tensor row {row['view_title']} / {row['peak_name']} has no FitParams run ID.")
        selected_by_params.setdefault(params_id, []).append(row)

    summary_rows: List[Dict[str, Any]] = []
    audit_rows: List[Dict[str, Any]] = []
    cache_count = 0
    rebuilt_count = 0
    for params_id, selections in selected_by_params.items():
        params_run = exp.get_run(params_id)
        if params_run is None or params_run.run_type != RunType.FIT_PARAMS:
            raise ValueError(f"FitParams run '{params_id}' from the tensor CSV is missing.")
        targets = [float(row["frequency_cm1"]) for row in selections]
        groups, row_data_source, cache_ids, validation_status, validation_message = _row_groups_for_uncertainty(
            exp,
            params_run,
            targets,
            cache_only=bool(args.cache_only),
        )
        if cache_ids:
            cache_count += 1
        else:
            rebuilt_count += 1
        fit_state = (params_run.metadata or {}).get("fit_state") or (params_run.metadata or {}).get("map_fit_state")

        for selected, group in zip(selections, groups):
            tensor_fit = polar_area_fitting.fit_tensor_for_peak(group, fit_state=fit_state)
            a, a_std, a_index = _tensor_fit_parameter(tensor_fit, "a")
            b, b_std, b_index = _tensor_fit_parameter(tensor_fit, "b")
            phi, phi_std, _phi_index = _tensor_fit_parameter(tensor_fit, "phi")
            symmetry = str(selected["symmetry"])
            if symmetry == "D6h_E2g":
                ratio, ratio_std, expression = -1.0, 0.0, "symmetry"
                ratio_source = "E2g symmetry limit"
                method = "fixed by E2g symmetry; no fitted ratio degree of freedom"
            elif symmetry == "D6h_A1g":
                ratio, ratio_std, expression = 1.0, 0.0, "symmetry"
                ratio_source = "A1g symmetry limit"
                method = "fixed by A1g symmetry; no fitted ratio degree of freedom"
            else:
                ratio, ratio_std, expression = _conditional_tensor_ratio(
                    a, b, tensor_fit.covariance, a_index, b_index
                )
                if not np.isfinite(ratio):
                    raise ValueError(f"Could not calculate a tensor ratio for {selected['view_title']} / {selected['peak_name']}.")
                ratio_source = f"fitted row-area {expression} (larger magnitude component in numerator)"
                method = "1-sigma delta method using full curve_fit covariance of a and b"

            global_ratio, _global_std, global_expression = _conditional_tensor_ratio(
                float(selected["tensor_a"]),
                float(selected["tensor_b"]),
                None,
                None,
                None,
            )
            if expression == "symmetry":
                global_ratio = ratio
                global_expression = "symmetry"

            covariance_ab = np.nan
            if tensor_fit.covariance is not None and a_index is not None and b_index is not None:
                covariance_ab = _optional_float(tensor_fit.covariance[a_index, b_index])
            summary_rows.append({
                "substance": selected["substance"],
                "layer": selected["layer"],
                "layer_number": selected.get("_layer_number", ""),
                "layer_x": selected["layer_x"],
                "view_title": selected["view_title"],
                "symmetry": symmetry,
                "peak_name": selected["peak_name"],
                "peak_index": selected.get("_peak_index", ""),
                "frequency_cm1": selected["frequency_cm1"],
                "fit_params_run_id": params_id,
                "fit_params_run": selected.get("_fit_params_run", params_run.nickname),
                "source_run_ids": selected.get("_source_run_ids", ""),
                "source_runs": selected.get("_source_runs", ""),
                "tensor_a_global": selected["tensor_a"],
                "tensor_b_global": selected["tensor_b"],
                "tensor_a": a,
                "tensor_a_std": a_std,
                "tensor_b": b,
                "tensor_b_std": b_std,
                "tensor_ab_cov": covariance_ab,
                "tensor_phi_deg": phi,
                "tensor_phi_deg_std": phi_std,
                "a_over_b": ratio,
                "ratio_std": ratio_std,
                "ratio_expression": expression,
                "a_over_b_global": global_ratio,
                "ratio_expression_global": global_expression,
                "rowfit_minus_global": ratio - global_ratio if np.isfinite(global_ratio) else np.nan,
                "ratio_source": ratio_source,
                "uncertainty_method": method,
                "n_observations": tensor_fit.n_observations,
                "degrees_of_freedom": tensor_fit.degrees_of_freedom,
                "residual_rmse": tensor_fit.residual_rmse,
                "row_data_source": row_data_source,
                "cache_run_ids": ";".join(cache_ids),
                "row_validation_status": validation_status,
                "row_validation_message": validation_message,
            })
            _append_row_audit(
                audit_rows,
                selected,
                group,
                row_data_source=row_data_source,
                cache_ids=cache_ids,
            )

    output = _write_dict_csv(args.output, _TENSOR_RATIO_UNCERTAINTY_COLUMNS, summary_rows)
    row_output = _write_dict_csv(args.row_output, _TENSOR_ROW_AUDIT_COLUMNS, audit_rows)
    print(json.dumps({
        "input": os.path.abspath(args.experiment),
        "tensor_csv": os.path.abspath(args.tensor_csv),
        "output": output,
        "row_output": row_output,
        "summary_rows": len(summary_rows),
        "row_audit_rows": len(audit_rows),
        "fit_params_from_exact_cache": cache_count,
        "fit_params_rebuilt_in_memory": rebuilt_count,
        "cache_only": bool(args.cache_only),
    }, indent=2, default=_json_default))
    return 0


def _read_tensor_ratio_rows(
    path: str,
    band: str,
    frequency_range: Sequence[float],
    excluded_layers: Optional[Sequence[float]],
    bulk_views: Dict[str, str],
) -> List[Dict[str, Any]]:
    with open(path, newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames:
            raise ValueError("Tensor CSV has no header.")
        required = {
            "substance", "layer_number", "view_title", "symmetry", "peak_name",
            "frequency_cm1", "tensor_a", "tensor_b",
        }
        missing = required.difference(reader.fieldnames)
        if missing:
            raise ValueError(f"Tensor CSV is missing columns: {', '.join(sorted(missing))}")
        source_rows = list(reader)

    x_min, x_max = sorted((float(frequency_range[0]), float(frequency_range[1])))
    excluded = {float(layer) for layer in (excluded_layers or [])}
    rows = []
    for row_order, source in enumerate(source_rows):
        frequency = float(source["frequency_cm1"])
        if not x_min <= frequency <= x_max:
            continue
        symmetry = str(source["symmetry"])
        if band == "low":
            if symmetry not in {"D2h_Ag", "D6h_E2g"}:
                continue
        elif symmetry not in {"D2h_Ag", "D6h_A1g"}:
            continue

        substance = _peak_summary_graph_substance(source["substance"])
        layer_text = str(source["layer_number"]).strip().lower()
        layer_label = str(source.get("layer", "")).strip().lower()
        view_title = str(source["view_title"])
        if layer_text == "bulk" or layer_label == "bulk" or "bulk" in view_title.lower():
            layer_x = 5.0
        elif layer_text:
            layer_x = float(layer_text)
        else:
            continue
        if layer_x in excluded:
            continue
        requested_view = bulk_views.get(substance)
        if layer_x == 5.0 and requested_view and view_title != requested_view:
            continue

        a = float(source["tensor_a"])
        b = float(source["tensor_b"])
        if np.isfinite(a) and np.isfinite(b) and abs(b) > abs(a) and a != 0.0:
            ratio = b / a
            ratio_expression = "b/a"
            ratio_source = "fitted b/a (|b|>|a|)"
        elif np.isfinite(a) and np.isfinite(b) and b != 0.0:
            ratio = a / b
            ratio_expression = "a/b"
            ratio_source = "fitted a/b (|a|>=|b|)"
        elif symmetry == "D6h_E2g" and band == "low":
            ratio = -1.0
            ratio_expression = "symmetry"
            ratio_source = "E2g symmetry limit"
        elif symmetry == "D6h_A1g" and band == "high":
            ratio = 1.0
            ratio_expression = "symmetry"
            ratio_source = "A1g symmetry limit"
        else:
            continue
        rows.append({
            "substance": substance,
            "layer": source.get("layer", source["layer_number"]),
            "layer_x": layer_x,
            "view_title": view_title,
            "symmetry": symmetry,
            "peak_name": source["peak_name"],
            "frequency_cm1": frequency,
            "tensor_a": a,
            "tensor_b": b,
            "a_over_b": ratio,
            "ratio_std": _optional_float(source.get("ratio_std")),
            "ratio_expression": ratio_expression,
            "ratio_source": ratio_source,
            "uncertainty_method": str(source.get("uncertainty_method", "")),
            "_fit_params_run_id": str(source.get("fit_params_run_id", "")),
            "_fit_params_run": str(source.get("fit_params_run", "")),
            "_source_run_ids": str(source.get("source_run_ids", "")),
            "_source_runs": str(source.get("source_runs", "")),
            "_peak_index": source.get("peak_index", ""),
            "_layer_number": source.get("layer_number", ""),
            "_tensor_phi_deg": source.get("tensor_phi_deg", "nan"),
            "_row_order": row_order,
        })
    return rows


def cmd_plot_tensor_ratio(args) -> int:
    frequency_range = args.frequency_range or ([270.0, 310.0] if args.band == "low" else [350.0, 390.0])
    bulk_views = _parse_peak_summary_bulk_sources(args.bulk_view)
    rows = _read_tensor_ratio_rows(
        args.input_csv,
        args.band,
        frequency_range,
        args.exclude_layer,
        bulk_views,
    )
    if not rows:
        raise ValueError("No matching tensor-ratio rows remain after filtering.")
    csv_output = _write_dict_csv(args.csv_output, _TENSOR_RATIO_COLUMNS, rows) if args.csv_output else None
    output = os.path.abspath(args.output)
    os.makedirs(os.path.dirname(output) or ".", exist_ok=True)

    y_values = np.asarray([float(row["a_over_b"]) for row in rows], dtype=float)
    y_errors = np.asarray([_optional_float(row.get("ratio_std")) for row in rows], dtype=float)
    finite_errors = np.isfinite(y_errors) & (y_errors >= 0.0)
    y_lower = y_values.copy()
    y_upper = y_values.copy()
    y_lower[finite_errors] -= y_errors[finite_errors]
    y_upper[finite_errors] += y_errors[finite_errors]
    if args.ylim:
        y_min, y_max = sorted((float(args.ylim[0]), float(args.ylim[1])))
    else:
        span = float(np.max(y_upper) - np.min(y_lower))
        margin = max(0.05, 0.2 * span)
        y_min, y_max = float(np.min(y_lower) - margin), float(np.max(y_upper) + margin)
    if args.tick_step <= 0 or args.width_cm <= 0 or args.height_cm <= 0 or args.error_capsize < 0:
        raise ValueError("Tick step and figure dimensions must be positive; error cap size cannot be negative.")

    rc = {
        "font.family": args.font_family,
        "font.size": args.font_size,
        "axes.labelsize": args.font_size,
        "axes.titlesize": args.font_size + 1.0,
        "axes.linewidth": args.line_width,
        "xtick.labelsize": args.font_size,
        "ytick.labelsize": args.font_size,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }
    with matplotlib.rc_context(rc):
        fig, ax = plt.subplots(figsize=(args.width_cm / 2.54, args.height_cm / 2.54))
        for substance in _PEAK_SUMMARY_SUBSTANCE_ORDER:
            group = [row for row in rows if row["substance"] == substance]
            if not group:
                continue
            group.sort(key=lambda row: (row["layer_x"], row["_row_order"]))
            primary_by_layer: Dict[float, Dict[str, Any]] = {}
            extras = []
            for row in group:
                if row["layer_x"] not in primary_by_layer:
                    primary_by_layer[row["layer_x"]] = row
                else:
                    extras.append(row)
            primary = [primary_by_layer[layer] for layer in sorted(primary_by_layer)]
            color = _PEAK_SUMMARY_COLORS[substance]
            ax.plot(
                [row["layer_x"] for row in primary],
                [row["a_over_b"] for row in primary],
                color=color,
                marker=args.marker,
                linestyle=args.line_style,
                linewidth=args.line_width,
                markersize=args.marker_size,
                markeredgewidth=args.line_width,
                alpha=0.9,
            )
            primary_errors = np.asarray([_optional_float(row.get("ratio_std")) for row in primary], dtype=float)
            valid_primary_errors = np.isfinite(primary_errors) & (primary_errors > 0.0)
            if np.any(valid_primary_errors):
                primary_x = np.asarray([row["layer_x"] for row in primary], dtype=float)
                primary_y = np.asarray([row["a_over_b"] for row in primary], dtype=float)
                ax.errorbar(
                    primary_x[valid_primary_errors],
                    primary_y[valid_primary_errors],
                    yerr=primary_errors[valid_primary_errors],
                    fmt="none",
                    ecolor=color,
                    elinewidth=args.line_width,
                    capsize=args.error_capsize,
                    capthick=args.line_width,
                    zorder=3,
                )
            constrained = [row for row in primary if not row["ratio_source"].startswith("fitted ")]
            if constrained and not args.filled_symmetry_limits:
                ax.scatter(
                    [row["layer_x"] for row in constrained],
                    [row["a_over_b"] for row in constrained],
                    facecolors="white",
                    edgecolors=color,
                    marker=args.marker,
                    s=args.marker_size ** 2,
                    linewidths=args.line_width,
                    zorder=4,
                )
            if extras:
                ax.scatter(
                    [row["layer_x"] + 0.045 for row in extras],
                    [row["a_over_b"] for row in extras],
                    facecolors="none",
                    edgecolors=color,
                    marker=args.marker,
                    s=args.marker_size ** 2,
                    linewidths=args.line_width,
                )
                extra_errors = np.asarray([_optional_float(row.get("ratio_std")) for row in extras], dtype=float)
                valid_extra_errors = np.isfinite(extra_errors) & (extra_errors > 0.0)
                if np.any(valid_extra_errors):
                    extra_x = np.asarray([row["layer_x"] + 0.045 for row in extras], dtype=float)
                    extra_y = np.asarray([row["a_over_b"] for row in extras], dtype=float)
                    ax.errorbar(
                        extra_x[valid_extra_errors],
                        extra_y[valid_extra_errors],
                        yerr=extra_errors[valid_extra_errors],
                        fmt="none",
                        ecolor=color,
                        elinewidth=args.line_width,
                        capsize=args.error_capsize,
                        capthick=args.line_width,
                        zorder=3,
                    )

        ax.set_xlim(0.75, 5.25)
        ax.set_xticks([1, 2, 3, 4, 5])
        ax.set_xticklabels(["1L", "2L", "3L", "4L", "Bulk"])
        ax.set_ylim(y_min, y_max)
        tick_start = np.ceil((y_min - 1e-12) / args.tick_step) * args.tick_step
        tick_stop = np.floor((y_max + 1e-12) / args.tick_step) * args.tick_step
        ax.set_yticks(np.arange(tick_start, tick_stop + args.tick_step * 0.5, args.tick_step))
        ax.tick_params(axis="both", which="major", length=args.tick_length, width=args.line_width, pad=1.5)
        for spine in ax.spines.values():
            spine.set_linewidth(args.line_width)
        ax.grid(False)
        ax.set_xlabel("Layer number", labelpad=2.0)
        ax.set_ylabel("Raman tensor ratio", labelpad=2.0)
        default_title = "Ag1/E2g tensor ratio" if args.band == "low" else "Ag2/A1g tensor ratio"
        ax.set_title(args.title or default_title, pad=2.0)

        handles = [
            Line2D([0], [0], color=_PEAK_SUMMARY_COLORS[substance], marker=args.marker,
                   linestyle=args.line_style, lw=args.line_width,
                   markersize=args.marker_size, markeredgewidth=args.line_width, label=substance)
            for substance in _PEAK_SUMMARY_SUBSTANCE_ORDER
            if any(row["substance"] == substance for row in rows)
        ]
        if not args.filled_symmetry_limits:
            handles.append(Line2D([0], [0], color="#333333", marker="o", markerfacecolor="white",
                                  lw=0, markeredgewidth=args.line_width, markersize=args.marker_size,
                                  label="symmetry limit"))
        legend = ax.legend(
            handles=handles,
            loc="upper center",
            bbox_to_anchor=(0.5, args.legend_anchor_y),
            ncol=2,
            fontsize=max(4.0, args.font_size - 0.5),
            frameon=True,
            framealpha=0.95,
            facecolor="white",
            edgecolor="#808080",
            handlelength=1.5,
            handletextpad=0.4,
            columnspacing=0.9,
            labelspacing=0.3,
            borderpad=0.4,
        )
        legend.get_frame().set_linewidth(args.line_width)
        fig.tight_layout(pad=0.3)
        fig.savefig(output, format="pdf")
        plt.close(fig)

    print(json.dumps({
        "input": os.path.abspath(args.input_csv),
        "csv_output": csv_output,
        "output": output,
        "rows": len(rows),
        "band": args.band,
        "frequency_range_cm1": list(map(float, frequency_range)),
        "excluded_layers": [float(layer) for layer in args.exclude_layer],
        "bulk_views": bulk_views,
        "figure_size_cm": [args.width_cm, args.height_cm],
        "ylim": [y_min, y_max],
        "filled_symmetry_limits": bool(args.filled_symmetry_limits),
        "marker": args.marker,
        "line_style": args.line_style,
        "error_bars": int(np.count_nonzero(finite_errors & (y_errors > 0.0))),
        "error_capsize_pt": args.error_capsize,
    }, indent=2))
    return 0


def _read_tensor_frequency_rows(
    path: str,
    frequency_min: float,
    frequency_max: float,
    low_max: float,
    high_min: float,
    excluded_layers: Optional[Sequence[float]],
    bulk_views: Dict[str, str],
) -> List[Dict[str, Any]]:
    with open(path, newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames:
            raise ValueError("Tensor CSV has no header.")
        source_rows = list(reader)
    excluded = {float(layer) for layer in (excluded_layers or [])}
    rows = []
    for row_order, source in enumerate(source_rows):
        frequency = float(source["frequency_cm1"])
        if not frequency_min <= frequency <= frequency_max:
            continue
        substance = _peak_summary_graph_substance(source["substance"])
        symmetry = str(source["symmetry"])
        if frequency <= low_max:
            source_branch = {
                "D2h_Ag": "Ag1",
                "D2h_B1g": "B1g",
                "D6h_E2g": "E2g",
            }.get(symmetry)
        elif frequency >= high_min:
            source_branch = {
                "D2h_Ag": "Ag2",
                "D6h_A1g": "A1g",
            }.get(symmetry)
        else:
            continue
        if source_branch is None:
            continue
        branches = _peak_summary_lineages(substance, source_branch)
        if not branches:
            continue

        layer_text = str(source["layer_number"]).strip().lower()
        layer_label = str(source.get("layer", "")).strip().lower()
        view_title = str(source["view_title"])
        if layer_text == "bulk" or layer_label == "bulk" or "bulk" in view_title.lower():
            layer_x = 5.0
        elif layer_text:
            layer_x = float(layer_text)
        else:
            continue
        if layer_x in excluded:
            continue
        requested_view = bulk_views.get(substance)
        if layer_x == 5.0 and requested_view and view_title != requested_view:
            continue
        for branch in branches:
            rows.append({
                "substance": substance,
                "layer": source.get("layer", source["layer_number"]),
                "layer_x": layer_x,
                "view_title": view_title,
                "symmetry": symmetry,
                "branch": branch,
                "peak_name": source["peak_name"],
                "frequency_cm1": frequency,
                "_row_order": row_order,
            })
    return rows


def cmd_plot_tensor_frequency(args) -> int:
    bulk_views = _parse_peak_summary_bulk_sources(args.bulk_view)
    rows = _read_tensor_frequency_rows(
        args.input_csv,
        args.frequency_min,
        args.frequency_max,
        args.low_max,
        args.high_min,
        args.exclude_layer,
        bulk_views,
    )
    if not rows:
        raise ValueError("No matching Ag/E2g/A1g frequency rows remain after filtering.")
    csv_output = _write_dict_csv(args.csv_output, _TENSOR_FREQUENCY_COLUMNS, rows) if args.csv_output else None
    output = os.path.abspath(args.output)
    os.makedirs(os.path.dirname(output) or ".", exist_ok=True)

    styles = {
        "Ag1/E2g": {"marker": "o", "linestyle": "-"},
        "B1g/E2g": {"marker": "s", "linestyle": "--"},
        "Ag2/A1g": {"marker": "^", "linestyle": "-."},
        "E2g": {"marker": "o", "linestyle": "-"},
        "A1g": {"marker": "^", "linestyle": "-."},
    }
    rc = {
        "font.family": args.font_family,
        "font.size": args.font_size,
        "axes.labelsize": args.font_size,
        "axes.titlesize": args.font_size + 1.0,
        "axes.linewidth": args.line_width,
        "xtick.labelsize": args.font_size,
        "ytick.labelsize": args.font_size,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }
    with matplotlib.rc_context(rc):
        fig, ax = plt.subplots(figsize=(args.width_cm / 2.54, args.height_cm / 2.54))
        for substance in _PEAK_SUMMARY_SUBSTANCE_ORDER:
            color = _PEAK_SUMMARY_COLORS[substance]
            for branch, style in styles.items():
                group = [row for row in rows if row["substance"] == substance and row["branch"] == branch]
                if not group:
                    continue
                group.sort(key=lambda row: (row["layer_x"], row["_row_order"]))
                primary_by_layer: Dict[float, Dict[str, Any]] = {}
                extras = []
                for row in group:
                    if row["layer_x"] not in primary_by_layer:
                        primary_by_layer[row["layer_x"]] = row
                    else:
                        extras.append(row)
                primary = [primary_by_layer[layer] for layer in sorted(primary_by_layer)]
                ax.plot(
                    [row["layer_x"] for row in primary],
                    [row["frequency_cm1"] for row in primary],
                    color=color,
                    marker=style["marker"],
                    linestyle=style["linestyle"],
                    linewidth=args.line_width,
                    markersize=args.marker_size,
                    markeredgewidth=args.line_width,
                    alpha=0.9,
                )
                if extras:
                    ax.scatter(
                        [row["layer_x"] + 0.045 for row in extras],
                        [row["frequency_cm1"] for row in extras],
                        facecolors="none",
                        edgecolors=color,
                        marker=style["marker"],
                        s=args.marker_size ** 2,
                        linewidths=args.line_width,
                    )

        ax.set_xlim(0.75, 5.25)
        ax.set_xticks([1, 2, 3, 4, 5])
        ax.set_xticklabels(["1L", "2L", "3L", "4L", "Bulk"])
        ax.set_ylim(args.frequency_min, args.frequency_max)
        ax.set_yticks(np.arange(args.frequency_min, args.frequency_max + args.tick_step * 0.5, args.tick_step))
        ax.tick_params(axis="both", which="major", length=args.tick_length, width=args.line_width, pad=1.5)
        for spine in ax.spines.values():
            spine.set_linewidth(args.line_width)
        ax.grid(False)
        ax.set_xlabel("Layer number", labelpad=2.0)
        ax.set_ylabel("Frequency (cm-1)", labelpad=2.0)
        ax.set_title(args.title or "Ag/B1g/A1g peak frequencies", pad=2.0)

        substance_handles = [
            Line2D([0], [0], color=_PEAK_SUMMARY_COLORS[substance], marker="o", lw=args.line_width,
                   markersize=args.marker_size, markeredgewidth=args.line_width, label=substance)
            for substance in _PEAK_SUMMARY_SUBSTANCE_ORDER
            if any(row["substance"] == substance for row in rows)
        ]
        branch_handles = [
            Line2D([0], [0], color="#333333", marker=style["marker"], linestyle=style["linestyle"],
                   lw=args.line_width, markersize=args.marker_size, markeredgewidth=args.line_width, label=branch)
            for branch, style in styles.items()
        ]
        legend_rows = max(len(substance_handles), len(branch_handles))
        blank = Line2D([0], [0], color="none", lw=0, label="")
        handles = (
            substance_handles + [blank] * (legend_rows - len(substance_handles))
            + branch_handles + [blank] * (legend_rows - len(branch_handles))
        )
        legend = ax.legend(
            handles=handles,
            loc="upper center",
            bbox_to_anchor=(0.5, args.legend_anchor_y),
            ncol=2,
            fontsize=max(4.0, args.font_size - 0.5),
            frameon=True,
            framealpha=0.95,
            facecolor="white",
            edgecolor="#808080",
            handlelength=1.5,
            handletextpad=0.4,
            columnspacing=0.9,
            labelspacing=0.3,
            borderpad=0.4,
        )
        legend.get_frame().set_linewidth(args.line_width)
        fig.tight_layout(pad=0.3)
        fig.savefig(output, format="pdf")
        plt.close(fig)

    print(json.dumps({
        "input": os.path.abspath(args.input_csv),
        "csv_output": csv_output,
        "output": output,
        "rows": len(rows),
        "frequency_range_cm1": [args.frequency_min, args.frequency_max],
        "excluded_layers": [float(layer) for layer in args.exclude_layer],
        "bulk_views": bulk_views,
        "figure_size_cm": [args.width_cm, args.height_cm],
    }, indent=2))
    return 0


def _fast_resolve_views(meta: Dict[str, Any], selectors: Optional[Sequence[str]]) -> List[Dict[str, Any]]:
    records = _fast_visible_view_records(meta)
    if not selectors:
        return records
    selected = []
    for token in selectors:
        query = str(token).strip().lower()
        exact = [rec for rec in records if rec["id"].lower() == query or str(rec["view"].get("title", "")).lower() == query]
        matches = exact or [rec for rec in records if query in rec["id"].lower() or query in str(rec["view"].get("title", "")).lower()]
        if not matches:
            raise ValueError(f"No visible view matches '{token}'.")
        if len(matches) > 1:
            names = ", ".join(f"{rec['view'].get('title', rec['id'])} ({rec['id']})" for rec in matches[:8])
            raise ValueError(f"View query '{token}' is ambiguous: {names}")
        selected.append(matches[0])
    return selected


def _fast_run_pol(header: Dict[str, Any]) -> str:
    md = header.get("metadata") or {}
    pol = str(md.get("pol") or md.get("polarization") or "").strip().lower()
    if pol:
        return pol
    text = f"{header.get('nickname', '')} {header.get('source_path', '')}".lower()
    match = re.search(r"(?<![a-z0-9])(xx|xy|yx|yy|rr|rl|lr|ll)(?![a-z0-9])", text)
    return match.group(1) if match else ""


def _fast_runs_by_pol(meta: Dict[str, Any], run_ids: Sequence[str]) -> Dict[str, Dict[str, Any]]:
    runs = meta.get("runs") or {}
    out: Dict[str, Dict[str, Any]] = {}
    for rid in run_ids:
        header = runs.get(str(rid))
        if not header:
            continue
        pol = _fast_run_pol(header)
        if pol and pol not in out:
            out[pol] = header
    if not out:
        for idx, rid in enumerate(run_ids):
            if str(rid) in runs:
                out[str(idx)] = runs[str(rid)]
    return out


def _fast_color(value: str) -> str:
    text = str(value).strip()
    if not text:
        return "#000000"
    return text if text.startswith("#") else f"#{text}"


def _fast_axis_cm1(arrays: Dict[str, Any]) -> np.ndarray:
    shift = arrays.get("shift_cm1")
    if shift is not None:
        return np.asarray(shift, dtype=float)
    energy = arrays.get("energy_eV")
    if energy is not None:
        return np.asarray(energy, dtype=float) / EV_PER_CM1
    raise ValueError("Run is missing shift_cm1/energy_eV axis.")


def _fast_oriented_rotated_arrays(path: str, header: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    arrays = read_run_arrays(path, header["id"], ["shift_cm1", "energy_eV", "angle_values", "intensity_2d"])
    shift = _fast_axis_cm1(arrays)
    angles = np.asarray(arrays.get("angle_values"), dtype=float)
    intensity = np.asarray(arrays.get("intensity_2d"), dtype=float)
    shift, angles, intensity = analysis._orient_2d_arrays(shift, angles, intensity)
    return shift, *analysis.apply_angle_rotation(angles, intensity, (header.get("metadata") or {}).get("angle_rotation"))


def _fast_angle_bin_trace(angles: np.ndarray, intensity: np.ndarray, target: float, bin_size: int) -> Tuple[float, np.ndarray]:
    if angles.size == 0:
        raise ValueError("Angle axis is empty.")
    count = max(1, min(int(bin_size), int(angles.size)))
    order = np.argsort(np.abs(np.asarray(angles, dtype=float) - float(target)))[:count]
    order = np.sort(order)
    trace = np.nanmean(intensity[order, :], axis=0)
    return float(np.nanmean(angles[order])), trace


def _fast_should_smooth(query: str, view: Dict[str, Any], header: Dict[str, Any]) -> bool:
    tokens = _tokenize(query or "")
    if not tokens:
        return False
    text = f"{view.get('title', '')} {header.get('nickname', '')} {header.get('source_path', '')}".lower()
    return all(token in text for token in tokens)


def _fast_baseline_correct(x: np.ndarray, y: np.ndarray, roi: Sequence[float]) -> np.ndarray:
    y = np.asarray(y, dtype=float)
    roi_mask = (x >= float(roi[0])) & (x <= float(roi[1])) & np.isfinite(y)
    source = y[roi_mask] if np.any(roi_mask) else y[np.isfinite(y)]
    baseline = float(np.nanpercentile(source, 5.0)) if source.size else 0.0
    return y - baseline


def _default_manifest_path(output: str) -> str:
    root, ext = os.path.splitext(os.path.abspath(output))
    return f"{root}_manifest.csv" if ext else f"{os.path.abspath(output)}_manifest.csv"


def cmd_plotc_collection(args) -> int:
    meta = read_experiment_metadata(args.experiment)
    selected_views = _fast_resolve_views(meta, args.view)
    if not selected_views:
        raise ValueError("No visible views with 2D runs were found.")
    output = os.path.abspath(args.output)
    os.makedirs(os.path.dirname(output) or ".", exist_ok=True)
    colors = [_fast_color(color) for color in (args.colors or [])] or ["#c25b3e", "#3f276b"]
    xlim = [float(args.xlim[0]), float(args.xlim[1])]
    roi = [float(args.ylim_roi[0]), float(args.ylim_roi[1])]
    unit = normalize_spectral_unit(args.unit)
    panels_per_page = max(1, int(args.panels_per_page))
    manifest_rows = []

    with PdfPages(output) as pdf:
        for page_start in range(0, len(selected_views), panels_per_page):
            page_views = selected_views[page_start: page_start + panels_per_page]
            ncols = 2 if len(page_views) > 1 else 1
            nrows = int(np.ceil(len(page_views) / ncols))
            fig, axes = plt.subplots(nrows, ncols, figsize=(5.9 * ncols, 2.65 * nrows), squeeze=False, sharex=True)
            flat_axes = axes.ravel()
            for ax in flat_axes[len(page_views):]:
                ax.axis("off")
            for ax, view_record in zip(flat_axes, page_views):
                view = view_record["view"]
                pol_headers = _fast_runs_by_pol(meta, view_record["run_ids"])
                traces = []
                for pol_idx, pol in enumerate(args.pols):
                    header = pol_headers.get(str(pol).lower())
                    if header is None:
                        continue
                    shift_cm1, angles, intensity = _fast_oriented_rotated_arrays(args.experiment, header)
                    x = cm1_to_unit(shift_cm1, unit)
                    smooth_bin = args.smooth_angle_bin if _fast_should_smooth(args.smooth_query, view, header) else 1
                    for angle_idx, angle in enumerate(args.angles):
                        actual_angle, raw_y = _fast_angle_bin_trace(angles, intensity, float(angle), smooth_bin)
                        yb = _fast_baseline_correct(x, raw_y, roi)
                        traces.append({
                            "pol": str(pol),
                            "pol_idx": pol_idx,
                            "angle": float(angle),
                            "actual_angle": actual_angle,
                            "angle_idx": angle_idx,
                            "run_id": header["id"],
                            "run": header.get("nickname", header["id"]),
                            "x": x,
                            "y": yb,
                            "bin_size": smooth_bin,
                        })
                        manifest_rows.append([
                            view_record["id"],
                            view.get("title", view_record["id"]),
                            header["id"],
                            header.get("nickname", header["id"]),
                            pol,
                            float(angle),
                            actual_angle,
                            smooth_bin,
                        ])
                if not traces:
                    ax.text(0.5, 0.5, "No requested polarizations", ha="center", va="center", transform=ax.transAxes)
                    ax.set_axis_off()
                    continue

                roi_values = []
                for trace in traces:
                    x = trace["x"]
                    y = trace["y"]
                    mask = (x >= roi[0]) & (x <= roi[1]) & np.isfinite(y)
                    if np.any(mask):
                        roi_values.append(y[mask])
                if roi_values:
                    all_roi = np.concatenate(roi_values)
                    amp = float(np.nanpercentile(all_roi, 98.0) - np.nanpercentile(all_roi, 2.0))
                else:
                    amp = 1.0
                if not np.isfinite(amp) or amp <= 0:
                    amp = 1.0
                offset_step = float(args.offset_step) if args.offset_step is not None else amp * 1.35

                plotted_roi = []
                for trace in traces:
                    x = trace["x"]
                    y = trace["y"] + trace["pol_idx"] * offset_step
                    mask = (x >= xlim[0]) & (x <= xlim[1]) & np.isfinite(y)
                    if not np.any(mask):
                        continue
                    color = colors[trace["angle_idx"] % len(colors)]
                    linestyle = "-" if trace["pol_idx"] == 0 else "--"
                    ax.plot(x[mask], y[mask], color=color, linestyle=linestyle, linewidth=1.05)
                    roi_mask = (x >= roi[0]) & (x <= roi[1]) & np.isfinite(y)
                    if np.any(roi_mask):
                        plotted_roi.append(y[roi_mask])
                    if abs(float(x[mask][-1]) - xlim[1]) < max(1e-9, abs(xlim[1]) * 1e-9):
                        label_y = float(y[mask][-1])
                    else:
                        label_y = float(y[mask][np.argmin(np.abs(x[mask] - xlim[1]))])
                    if trace["angle_idx"] == 0:
                        ax.text(xlim[1], label_y, f" {trace['pol']}", fontsize=7, va="center", color="#333333")

                if plotted_roi:
                    ycat = np.concatenate(plotted_roi)
                    ymin = float(np.nanmin(ycat))
                    ymax = float(np.nanmax(ycat))
                    margin = max((ymax - ymin) * 0.12, 0.5)
                    ax.set_ylim(ymin - margin, ymax + margin)
                ax.set_xlim(xlim)
                ax.set_title(str(view.get("title", view_record["id"])), fontsize=9)
                ax.set_ylabel("Offset intensity")
                ax.grid(True, color="#dddddd", linewidth=0.4, alpha=0.7)
                angle_handles = [
                    plt.Line2D([0], [0], color=colors[idx % len(colors)], lw=1.2, label=f"{angle:g} deg")
                    for idx, angle in enumerate(args.angles)
                ]
                pol_handles = [
                    plt.Line2D([0], [0], color="#333333", lw=1.0, linestyle="-" if idx == 0 else "--", label=str(pol))
                    for idx, pol in enumerate(args.pols)
                ]
                ax.legend(handles=angle_handles + pol_handles, fontsize=6, loc="upper right", frameon=False)
            for ax in flat_axes[-ncols:]:
                if ax.has_data():
                    ax.set_xlabel(spectral_axis_label(unit))
            fig.tight_layout()
            pdf.savefig(fig)
            plt.close(fig)

    manifest = args.manifest or _default_manifest_path(output)
    _write_table_csv(
        manifest,
        ["view_id", "view_title", "run_id", "run", "polarization", "requested_angle_deg", "actual_angle_deg", "angle_bin_size"],
        manifest_rows,
    )
    print(json.dumps({"output": output, "manifest": os.path.abspath(manifest), "views": len(selected_views), "traces": len(manifest_rows)}, indent=2, default=_json_default))
    return 0

def _parse_csv_indices(text: Optional[str]) -> List[int]:
    if not text:
        return []
    out = []
    for part in str(text).split(","):
        part = part.strip()
        if part:
            out.append(int(part))
    return out


def _target_runs(exp: ExperimentSet, queries: Optional[Sequence[str]], view_token: Optional[str]) -> List[Run]:
    runs: List[Run] = []
    if queries:
        runs = [_find_run_by_query(exp, query) for query in queries]
    elif view_token:
        view = _find_view(exp, view_token)
        if view is None:
            raise ValueError(f"No view matches '{view_token}'.")
        runs = [exp.get_run(rid) for rid in view.run_ids]
        runs = [run for run in runs if run is not None and not is_hidden_run(run)]
    else:
        raise ValueError("Specify --run-query or --view.")
    seen = set()
    unique = []
    for run in runs:
        if run.id not in seen:
            seen.add(run.id)
            unique.append(run)
    return unique


def _fold_phi(value: float) -> float:
    return float(((float(value) + 180.0) % 360.0) - 180.0)


def _shift_fit_state_phi(state: Dict[str, Any], offset: float) -> int:
    count = 0
    for peak in state.get("peaks", []) or []:
        params = peak.get("ang_params", {})
        if "phi" not in params:
            continue
        try:
            params["phi"][0] = _fold_phi(float(params["phi"][0]) + float(offset))
            count += 1
        except Exception:
            continue
    return count


def _adjust_phi_for_rotated_runs(exp: ExperimentSet, deltas_by_run: Dict[str, float]) -> List[str]:
    warnings: List[str] = []
    adjusted = set()
    for owner in exp.runs.values():
        md = owner.metadata or {}
        state = md.get("fit_state") or md.get("map_fit_state")
        if not state:
            continue
        source_ids = [str(v) for v in (md.get("source_run_ids") or md.get("fit_params_source_run_ids") or ([owner.id] if owner.is_2d else []))]
        if not source_ids:
            continue
        touched = [rid for rid in source_ids if rid in deltas_by_run]
        if not touched:
            continue
        if len(touched) != len(source_ids):
            warnings.append(f"{owner.nickname}: phi not adjusted because only part of a common fit was rotated.")
            continue
        deltas = [float(deltas_by_run[rid]) for rid in source_ids]
        if max(deltas) - min(deltas) > 1e-6:
            warnings.append(f"{owner.nickname}: phi not adjusted because common-fit runs used different rotation changes.")
            continue
        if abs(deltas[0]) <= 1e-12:
            continue
        if id(state) in adjusted:
            continue
        adjusted.add(id(state))
        count = _shift_fit_state_phi(state, deltas[0])
        if count:
            try:
                engine = analysis.MapFittingEngine()
                engine.from_dict(state)
                md["fit_parameters_text"] = engine.export_parameters_text()
            except Exception:
                pass
    return warnings


def _resolve_fit_state(exp: ExperimentSet, runs: Sequence[Run], selector: str) -> Optional[Dict[str, Any]]:
    selector = str(selector or "auto")
    if selector == "none":
        return None
    if selector != "auto":
        params_run = _find_run_by_query(exp, selector)
        state = (params_run.metadata or {}).get("fit_state") or (params_run.metadata or {}).get("map_fit_state")
        if not state:
            raise ValueError(f"Run '{params_run.nickname}' does not contain fit parameters.")
        return state
    state, _source = polar_area_fitting.fit_state_for_runs(exp, runs)
    return state


def _setup_fit_engine(exp: ExperimentSet, runs: Sequence[Run], args) -> analysis.MapFittingEngine:
    engine = analysis.MapFittingEngine()
    for idx, run in enumerate(runs[:2]):
        if not run.is_2d:
            raise ValueError(f"{run.nickname} is not a 2D run.")
        shift, angles, intensity, background_angles = analysis.display_2d_with_acquisition_angles_from_run(run)
        engine.set_data(
            idx,
            shift,
            angles,
            intensity,
            run.nickname,
            background_ang=background_angles,
        )
        fallback = "parallel" if idx == 0 else "cross"
        engine.datasets[idx]["config"] = polar_area_fitting.infer_config(run, fallback)
        engine.datasets[idx]["label"] = "Parallel" if engine.datasets[idx]["config"] == "parallel" else "Cross"
    state = _resolve_fit_state(exp, runs, args.load_params)
    if state:
        engine.from_dict(state)
    if args.x_min is not None:
        engine.x_min_limit = float(args.x_min)
    if args.x_max is not None:
        engine.x_max_limit = float(args.x_max)
    if args.clear_si_bg:
        engine.clear_si_bg()
    if args.si_bg:
        ok, msg = engine.load_si_bg(args.si_bg)
        if not ok:
            raise ValueError(f"Failed to load Si BG profile: {msg}")
    return engine


def _parse_auto_peak_spec(text: str) -> Tuple[float, Optional[float], int]:
    parts = [p.strip() for p in str(text).split(",")]
    if not parts or not parts[0]:
        raise ValueError("--auto-peak requires X[,ANGLE[,RUN_INDEX]].")
    x_value = float(parts[0])
    angle_value = float(parts[1]) if len(parts) > 1 and parts[1] else None
    dataset_index = int(parts[2]) if len(parts) > 2 and parts[2] else 0
    return x_value, angle_value, max(0, min(1, dataset_index))


def _parse_peak_spec(text: str) -> Tuple[str, float, Optional[float], Optional[str]]:
    parts = str(text).split(":")
    if len(parts) < 2:
        raise ValueError("--peak requires RULE:X0[:GAMMA[:NAME]].")
    rule = parts[0].strip()
    if rule not in analysis.RULE_METADATA:
        raise ValueError(f"Unknown rule '{rule}'.")
    x0 = float(parts[1])
    gamma = float(parts[2]) if len(parts) > 2 and parts[2] else None
    name = parts[3].strip() if len(parts) > 3 and parts[3].strip() else None
    return rule, x0, gamma, name


def _write_table_csv(path: str, headers: Sequence[str], rows: Sequence[Sequence[Any]]) -> str:
    path = os.path.abspath(path)
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        writer.writerows(rows)
    return path


def cmd_list(args) -> int:
    if getattr(args, "fast", False):
        try:
            return _cmd_list_fast(args)
        except FastHDF5Unsupported as exc:
            print(f"Warning: fast metadata listing unsupported ({exc}); falling back to full load.", file=sys.stderr)
    exp = _load_experiment(args.experiment)
    if getattr(args, "json", False):
        payload = {
            "views": [_view_summary(exp, view) for view in exp.views.values()],
            "runs": [_run_summary(run) for run in _visible_runs(exp)],
        }
        print(json.dumps(payload, indent=2, default=_json_default))
        return 0
    print("Views:")
    for view in exp.views.values():
        run_names = [run.nickname for rid in view.run_ids if (run := exp.get_run(rid)) and not is_hidden_run(run)]
        print(f"  {view.id}\t{view.title}\t{view.spectral_unit}\t{', '.join(run_names)}")
    print("Runs:")
    for run in _visible_runs(exp):
        info = _run_summary(run)
        print(
            f"  {info['id']}\t{info['nickname']}\t{info['kind']}\t"
            f"raw_x={info['raw_x_unit']}\tfit_peaks={info['fit_peak_count']}\t"
            f"rotation={info['angle_rotation_summary']}\t"
            f"sources={','.join(info['source_run_ids'])}\t{info['source_path']}"
        )
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
    selected = [_find_run_by_query(exp, query) for query in queries] if queries else _visible_runs(exp)
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


def cmd_merge(args) -> int:
    _reject_in_place(args.output, args.append)
    exp = _load_experiment(args.append) if args.append else ExperimentSet(id=new_experiment_id())
    run, summary, cosmic_report = analysis.merge_raw_to_run(
        args.seed,
        input_unit=args.input_unit,
        manual_laser_nm=args.manual_laser_nm,
        dark_value=args.dark_value,
        cosmic_policy=args.cosmic,
        cosmic_indices=_parse_csv_indices(args.cosmic_indices),
        nickname=args.nickname,
    )
    _ensure_unique_run_id(exp, run)
    exp.add_run(run)
    unit = normalize_spectral_unit(args.display_unit or config.get("default_spectral_unit", config.get("unit", "meV")))
    if args.view_title or not args.append:
        view = ViewState(
            id=new_view_id(),
            title=args.view_title or "View 1",
            run_ids=[run.id],
            spectral_unit=unit,
            x_axis="energy_eV" if unit == "meV" else "shift_cm1",
        )
        exp.add_view(view)
    output = _write_experiment(exp, args.output)
    if args.cosmic_report and cosmic_report is not None:
        report_path = os.path.abspath(args.cosmic_report)
        os.makedirs(os.path.dirname(report_path) or ".", exist_ok=True)
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(cosmic_report, f, indent=2, default=_json_default)
        summary["cosmic_report"] = report_path
    summary["output"] = output
    print(json.dumps(summary, indent=2, default=_json_default))
    return 0


def cmd_import_qe_raman(args) -> int:
    """Import a QE ph.x output as paired native XX/YX one-dimensional runs."""

    _reject_in_place(args.output, args.append, args.ph_output)
    if args.fit_params_query and not args.append:
        raise ValueError("--fit-params-query requires --append with an existing ExperimentSet.")
    exp = _load_experiment(args.append) if args.append else ExperimentSet(id=new_experiment_id())
    result = qe_raman.parse_qe_ph_output(args.ph_output)
    groups = qe_raman.usable_mode_groups(result)

    fit_run = None
    fit_state = None
    if args.fit_params_query:
        fit_run = _find_run_by_query(exp, args.fit_params_query)
        if fit_run.run_type != RunType.FIT_PARAMS:
            raise ValueError(
                f"Run '{fit_run.nickname}' matches the query but is not a Fit Parameters run."
            )
        fit_state = (fit_run.metadata or {}).get("fit_state") or (fit_run.metadata or {}).get("map_fit_state")
        if not fit_state:
            raise ValueError(f"Fit Parameters run '{fit_run.nickname}' does not contain a saved fit state.")

    rows, fit_peaks, adaptation_warnings = qe_raman.adapt_peak_rows(
        groups,
        fit_state,
        angle_deg=args.angle,
        default_fwhm_cm1=args.fwhm,
        match_tolerance_cm1=args.match_tolerance,
    )
    nickname = str(args.nickname or os.path.splitext(os.path.basename(args.ph_output))[0]).strip()
    xx_run, yx_run = qe_raman.make_qe_raman_runs(
        result,
        nickname,
        rows,
        angle_deg=args.angle,
        match_tolerance_cm1=args.match_tolerance,
        default_fwhm_cm1=args.fwhm,
        x_min_cm1=args.x_min,
        x_max_cm1=args.x_max,
        step_cm1=args.step,
        fit_params_run_id=fit_run.id if fit_run else None,
        warnings=adaptation_warnings,
    )
    for run in (xx_run, yx_run):
        _ensure_unique_run_id(exp, run)
        exp.add_run(run)

    unit = normalize_spectral_unit(
        args.display_unit or config.get("default_spectral_unit", config.get("unit", "meV"))
    )
    view = ViewState(
        id=new_view_id(),
        title=args.view_title or f"{nickname} DFT Raman",
        run_ids=[xx_run.id, yx_run.id],
        spectral_unit=unit,
        x_axis="energy_eV" if unit == "meV" else "shift_cm1",
        show_legend=True,
    )
    exp.add_view(view)
    output = _write_experiment(exp, args.output)
    payload = {
        "output": output,
        "source": os.path.abspath(args.ph_output),
        "qe_version": result.qe_version,
        "point_group": result.point_group,
        "mode_count": result.mode_count,
        "raman_peak_count": len(groups),
        "fit_params_run_id": fit_run.id if fit_run else None,
        "fit_peak_count": len(fit_peaks),
        "matched_peak_count": sum(row.get("value_source") == "fitparams" for row in rows),
        "intensity_source": xx_run.metadata.get("intensity_source"),
        "frequency_only": True,
        "runs": [
            {"id": xx_run.id, "nickname": xx_run.nickname, "polarization": "xx"},
            {"id": yx_run.id, "nickname": yx_run.nickname, "polarization": "yx"},
        ],
        "view": {"id": view.id, "title": view.title, "spectral_unit": view.spectral_unit},
        "warnings": list(dict.fromkeys(result.warnings + adaptation_warnings)),
        "disclaimer": xx_run.metadata.get("qe_intensity_disclaimer"),
    }
    print(json.dumps(payload, indent=2, default=_json_default))
    return 0


def cmd_rotate(args) -> int:
    _reject_in_place(args.output, args.experiment)
    exp = _load_experiment(args.experiment)
    runs = _target_runs(exp, args.run_query, args.view)
    if not runs:
        raise ValueError("No runs selected for rotation.")
    deltas_by_run: Dict[str, float] = {}
    rotated = []
    for run in runs:
        if not run.is_2d:
            raise ValueError(f"{run.nickname} is not a 2D run.")
        previous_settings = copy.deepcopy((run.metadata or {}).get("angle_rotation") or {})
        if args.disable:
            settings = {"enabled": False, "summary": "Rotation disabled"}
            run.metadata["angle_rotation"] = settings
            run.metadata.pop("angle_rotation_summary", None)
        else:
            settings = analysis.build_angle_rotation_settings(
                run.angle_values,
                anchor_deg=float(args.anchor_deg),
                window_start_deg=args.window_start_deg,
                cyclic_rotate=bool(args.cyclic),
            )
            run.metadata["angle_rotation"] = settings
            run.metadata["angle_rotation_summary"] = settings.get("summary", analysis.angle_rotation_summary(settings))
        deltas_by_run[run.id] = analysis.angle_rotation_transition_delta(previous_settings, settings)
        rotated.append({"run_id": run.id, "nickname": run.nickname, "settings": settings})
    warnings = _adjust_phi_for_rotated_runs(exp, deltas_by_run)
    output = _write_experiment(exp, args.output)
    print(json.dumps({"output": output, "runs": rotated, "warnings": warnings}, indent=2, default=_json_default))
    return 0


def _runs_for_fit(args, exp: ExperimentSet) -> List[Run]:
    role_map: Dict[str, Run] = {}
    for role_spec in args.role or []:
        role, _, query = role_spec.partition(":")
        role = role.strip().lower()
        if role not in {"parallel", "cross"} or not query:
            raise ValueError("--role must look like parallel:QUERY or cross:QUERY.")
        role_map[role] = _find_run_by_query(exp, query)
    if role_map:
        runs = []
        if "parallel" in role_map:
            runs.append(role_map["parallel"])
        if "cross" in role_map:
            runs.append(role_map["cross"])
        return runs
    queries = args.run_query or []
    if not (1 <= len(queries) <= 2):
        raise ValueError("fit2d requires one or two --run-query values, or --role entries.")
    return [_find_run_by_query(exp, query) for query in queries]


def cmd_fit2d(args) -> int:
    _reject_in_place(args.output, args.experiment)
    exp = _load_experiment(args.experiment)
    runs = _runs_for_fit(args, exp)
    engine = _setup_fit_engine(exp, runs, args)
    if len(runs) == 1 and args.config != "auto":
        engine.datasets[0]["config"] = args.config
        engine.datasets[0]["label"] = "Parallel" if args.config == "parallel" else "Cross"

    for spec in args.auto_peak or []:
        x_value, angle_value, dataset_index = _parse_auto_peak_spec(spec)
        estimates = analysis.estimate_peak_defaults(
            datasets=engine.datasets,
            selected_dataset_index=dataset_index,
            x_value=x_value,
            angle_value=angle_value,
            preserve_x0=True,
        )
        estimates["x0"] = x_value
        estimates.setdefault("spec_params", {})["x0"] = x_value
        engine.add_peak(rule_name=estimates.get("best_rule", "D2h_B1g"), center=x_value, estimates=estimates)

    for spec in args.peak or []:
        rule, x0, gamma, name = _parse_peak_spec(spec)
        estimates = {"x0": x0}
        if gamma is not None:
            estimates["gamma"] = gamma
        engine.add_peak(name=name, rule_name=rule, center=x0, estimates=estimates)

    if not engine.peaks:
        raise ValueError("No peaks available. Add --auto-peak or --peak, or load previous params containing peaks.")

    ok, message = engine.run_optimization()
    if not ok:
        raise ValueError(f"2D fit failed: {message}")

    state = engine.to_dict()
    text = engine.export_parameters_text()
    source_ids = [run.id for run in runs]
    for run in runs:
        run.metadata["fit_state"] = state
        run.metadata["fit_parameters_text"] = text
        run.metadata["fit_params_source_run_ids"] = source_ids

    params_run = Run(
        id=new_run_id(prefix="params"),
        source_path="",
        metadata={
            "nickname": f"FitParams_{'_'.join(run.nickname for run in runs)}",
            "fit_state": state,
            "fit_parameters_text": text,
            "source_run_ids": source_ids,
        },
        run_type=RunType.FIT_PARAMS,
    )
    _ensure_unique_run_id(exp, params_run)
    exp.add_run(params_run)

    component_paths: List[str] = []
    if args.export_components:
        os.makedirs(args.export_components, exist_ok=True)
        for rec in engine.get_peak_reconstructions():
            ds = engine.datasets[int(rec["dataset_idx"])]
            safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(rec["name"]))
            path = os.path.join(args.export_components, f"dataset{rec['dataset_idx']}_{safe}.csv")
            rows = [[angle, *matrix_row] for angle, matrix_row in zip(ds["ang"], np.asarray(rec["matrix"]))]
            component_paths.append(_write_table_csv(path, ["Angle", *[f"{x:.12g}" for x in ds["x"]]], rows))

    diagnostic_paths: List[str] = []
    if args.export_diagnostics:
        os.makedirs(args.export_diagnostics, exist_ok=True)
        for table in engine.global_fit_trace_tables():
            path = os.path.join(args.export_diagnostics, f"global_dataset{table['dataset_idx']}.csv")
            diagnostic_paths.append(_write_table_csv(path, table["headers"], table["rows"]))
        success, _msg, row_results = engine.validate_row_by_row()
        if success and row_results:
            for idx, table in enumerate(row_results):
                path = os.path.join(args.export_diagnostics, f"rowfit_dataset{idx}.csv")
                diagnostic_paths.append(_write_table_csv(path, table["headers"], table["rows_params"]))

    output = _write_experiment(exp, args.output)
    print(json.dumps({
        "output": output,
        "params_run_id": params_run.id,
        "source_run_ids": source_ids,
        "peak_count": len(engine.peaks),
        "peaks": engine.peaks,
        "component_paths": component_paths,
        "diagnostic_paths": diagnostic_paths,
    }, indent=2, default=_json_default))
    return 0


def _selected_validate_rowfit_params(exp: ExperimentSet, args) -> List[Run]:
    if args.all_fit_params:
        selected = [run for run in exp.runs.values() if run.run_type == RunType.FIT_PARAMS]
    else:
        selected = []
        for query in args.params_query or []:
            run = _find_run_by_query(exp, query)
            if run.run_type != RunType.FIT_PARAMS:
                raise ValueError(f"'{run.nickname}' is not a Fit Parameters run.")
            if all(existing.id != run.id for existing in selected):
                selected.append(run)
    if not selected:
        raise ValueError("No Fit Parameters runs were selected.")
    return selected


def _rowfit_failure_entry(params_run: Run, message: str) -> Dict[str, Any]:
    metadata = params_run.metadata or {}
    return {
        "status": "failed",
        "fit_params_run_id": params_run.id,
        "fit_params": params_run.nickname,
        "source_run_ids": list(metadata.get("source_run_ids") or metadata.get("fit_params_source_run_ids") or []),
        "source_runs": [],
        "peak_count": len(((metadata.get("fit_state") or {}).get("peaks") or [])),
        "rows_total": 0,
        "rows_succeeded": 0,
        "bound_hit_count": 0,
        "anomaly_count": 0,
        "message": str(message),
        "artifacts": {},
        "cache_run_ids": [],
    }


def cmd_validate_rowfit(args) -> int:
    if args.output:
        _reject_in_place(args.output, args.experiment)
    exp = _load_experiment(args.experiment)
    selected = _selected_validate_rowfit_params(exp, args)
    rules = row_fit_cli.load_rule_config(args.row_config)
    contexts: List[row_fit_cli.ValidationContext] = []
    entries: List[Dict[str, Any]] = []

    for params_run in selected:
        try:
            context = row_fit_cli.prepare_context(exp, params_run)
            row_fit_cli.apply_group_overrides(
                context.config,
                all_fixed=args.all_fixed,
                background=args.background_fixed,
                area=args.area_fixed,
                gamma=args.gamma_fixed,
            )
            row_fit_cli.apply_json_rules(context, rules)
            row_fit_cli.finalize_config(context)
            contexts.append(context)
        except Exception as exc:
            entries.append(_rowfit_failure_entry(params_run, str(exc)))

    row_fit_cli.ensure_rules_matched(rules)

    successful_contexts: List[row_fit_cli.ValidationContext] = []
    for context in contexts:
        success, message, _results = row_fit_cli.validate_context(context)
        entry = row_fit_cli.context_summary(context)
        entry["status"] = "success" if success else "failed"
        entry["message"] = message
        entry["artifacts"] = {}
        entry["cache_run_ids"] = []
        entries.append(entry)

        stem = row_fit_cli.safe_artifact_stem(context.params_run)
        artifact_errors: List[str] = []
        if args.csv_dir and context.results:
            try:
                entry["artifacts"]["row_csvs"] = row_fit_cli.write_context_csvs(context, args.csv_dir)
            except Exception as exc:
                artifact_errors.append(f"CSV export: {exc}")
        if args.report_dir and context.results:
            try:
                path = os.path.join(args.report_dir, f"{stem}_validation.pdf")
                entry["artifacts"]["diagnostic_pdf"] = row_fit_cli.write_diagnostic_pdf(context, path, unit=args.unit)
            except Exception as exc:
                artifact_errors.append(f"diagnostic PDF: {exc}")
        if args.polar_dir and context.results:
            try:
                path = os.path.join(args.polar_dir, f"{stem}_polar.pdf")
                entry["artifacts"]["polar_pdf"] = row_fit_cli.write_polar_pdf(
                    context,
                    path,
                    unit=args.unit,
                    panels_per_page=args.polar_panels_per_page,
                )
            except Exception as exc:
                artifact_errors.append(f"polar PDF: {exc}")
        if artifact_errors:
            entry["artifact_errors"] = artifact_errors
        if success:
            successful_contexts.append(context)

    failures = [entry for entry in entries if entry.get("status") != "success"]
    output_path = None
    persistence_errors: List[str] = []
    if args.output and successful_contexts and not (args.strict and failures):
        entry_by_id = {str(entry.get("fit_params_run_id")): entry for entry in entries}
        for context in successful_contexts:
            try:
                result_refs = row_fit_cli.persist_context(exp, context)
                entry = entry_by_id[str(context.params_run.id)]
                entry["row_result_storage"] = "compact_fitparams"
                entry["row_result_refs"] = result_refs
                # Retained as a compatibility alias for existing manifest readers.
                entry["cache_run_ids"] = result_refs
            except Exception as exc:
                persistence_errors.append(f"{context.params_run.nickname}: {exc}")
                entry = entry_by_id[str(context.params_run.id)]
                entry["status"] = "failed"
                entry["message"] = f"Compact row-result persistence failed: {exc}"
        if not persistence_errors or not args.strict:
            output_path = _write_experiment(exp, args.output)

    if args.csv_dir:
        summary_path = row_fit_cli.write_batch_summary_csv(
            os.path.join(args.csv_dir, "validate_rowfit_summary.csv"),
            entries,
        )
    else:
        summary_path = None

    final_failures = [entry for entry in entries if entry.get("status") != "success"]
    artifact_error_count = sum(len(entry.get("artifact_errors") or []) for entry in entries)
    payload = {
        "input": os.path.abspath(args.experiment),
        "output": output_path,
        "selected": len(selected),
        "successful": len(entries) - len(final_failures),
        "failed": len(final_failures),
        "strict": bool(args.strict),
        "summary_csv": summary_path,
        "persistence_errors": persistence_errors,
        "artifact_error_count": artifact_error_count,
        "results": entries,
    }
    print(json.dumps(payload, indent=2, default=_json_default))
    if not successful_contexts:
        return 1
    if final_failures or persistence_errors or artifact_error_count or (args.strict and failures):
        return 2
    return 0


def _fit_result_context(exp: ExperimentSet, args) -> Tuple[Dict[str, Any], List[Run], Run]:
    if args.params_query:
        params_run = _find_run_by_query(exp, args.params_query)
        state = (params_run.metadata or {}).get("fit_state") or (params_run.metadata or {}).get("map_fit_state")
        if not state:
            raise ValueError(f"{params_run.nickname} does not contain fit_state.")
        source_ids = params_run.metadata.get("source_run_ids") or params_run.metadata.get("fit_params_source_run_ids") or []
        sources = [exp.get_run(rid) for rid in source_ids]
        sources = [run for run in sources if run is not None and run.is_2d]
        return state, sources[:2], params_run
    if not args.run_query:
        raise ValueError("Specify --run-query or --params-query.")
    run = _find_run_by_query(exp, args.run_query)
    state, _params_run, source_ids = find_fit_state(exp, run)
    if not state:
        raise ValueError(f"No fit state found for {run.nickname}.")
    sources = [exp.get_run(rid) for rid in source_ids]
    sources = [src for src in sources if src is not None and src.is_2d]
    if run.is_2d and all(src.id != run.id for src in sources):
        sources.insert(0, run)
    return state, sources[:2], run


def _select_peak(state: Dict[str, Any], token: Optional[str]) -> Tuple[int, Dict[str, Any]]:
    peaks = state.get("peaks") or []
    if not peaks:
        raise ValueError("Fit state has no peaks.")
    if token is None:
        return 0, peaks[0]
    try:
        idx = int(token)
        if 1 <= idx <= len(peaks):
            idx -= 1
        if 0 <= idx < len(peaks):
            return idx, peaks[idx]
    except ValueError:
        pass
    for idx, peak in enumerate(peaks):
        if str(peak.get("name", "")).lower() == str(token).lower():
            return idx, peak
    raise ValueError(f"No peak matches '{token}'.")


def _fit_state_unit(state: Dict[str, Any]) -> str:
    return normalize_spectral_unit(state.get("unit", "cm-1"), "cm-1")


def _peak_center_cm1(state: Dict[str, Any], peak: Dict[str, Any]) -> float:
    value = float(peak.get("spec_params", {}).get("x0", [np.nan])[0])
    return float(unit_to_cm1(np.asarray([value], dtype=float), _fit_state_unit(state))[0])


def _plot_fit_result_slice(ax, exp: ExperimentSet, sources: Sequence[Run], state: Dict[str, Any], peak: Dict[str, Any], unit: str, compact: bool = False) -> None:
    target = _peak_center_cm1(state, peak)
    for idx, run in enumerate(sources[:2]):
        shift, angles, intensity = analysis.display_2d_from_run(run)
        ix = _nearest_index(shift, target)
        theta = np.deg2rad(angles)
        raw_color = "black" if idx == 0 else "#777777"
        global_color = "red" if idx == 0 else "#ff8a8a"
        row_color = "#1f77b4" if idx == 0 else "#7fb3df"
        linestyle = "-" if idx == 0 else "--"
        marker = "o" if idx == 0 else "s"
        ax.plot(theta, intensity[:, ix], marker=marker, linestyle="None", ms=3, color=raw_color, label=f"{run.nickname} raw")
        fit = overlay_data(exp, run, "both")
        if fit and fit.global_matrix is not None and fit.global_matrix.shape[1] > ix:
            ax.plot(theta, fit.global_matrix[:, ix], color=global_color, linestyle=linestyle, lw=1.1, label=f"{run.nickname} global")
        if fit and fit.row_matrix is not None and fit.row_matrix.shape[1] > ix:
            ax.plot(theta, fit.row_matrix[:, ix], color=row_color, linestyle=linestyle, lw=1.1, label=f"{run.nickname} row")
    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)
    ax.set_title(f"Slice @ {float(cm1_to_unit(np.asarray([target]), unit)[0]):.4g} {unit}", fontsize=9 if compact else None)
    ax.legend(fontsize=6 if compact else 7, loc="upper right", bbox_to_anchor=(1.25, 1.15) if compact else None)


def _plot_fit_result_area(ax, sources: Sequence[Run], state: Dict[str, Any], peak: Dict[str, Any], compact: bool = False) -> None:
    target = _peak_center_cm1(state, peak)
    row_fit_groups = polar_area_fitting.fit_lorentzian_rows_for_runs(
        sources[:2],
        [target],
        fit_state=state,
    )
    row_fits = row_fit_groups[0] if row_fit_groups else []
    tensor = polar_area_fitting.fit_tensor_for_peak(row_fits, fit_state=state)
    theta_dense = np.linspace(0.0, 360.0, 721)
    theta_rad = np.deg2rad(theta_dense)
    rule_func = (analysis.RULE_METADATA.get(tensor.rule) or analysis.RULE_METADATA["D2h_Ag"])["func"]
    for idx, rf in enumerate(row_fits[:2]):
        color = "#222222" if idx == 0 else "#777777"
        line_color = "red" if idx == 0 else "#ff8a8a"
        linestyle = "-" if idx == 0 else "--"
        marker = "o" if idx == 0 else "s"
        valid = np.isfinite(rf.angles) & np.isfinite(rf.areas)
        if valid.any():
            ax.scatter(np.deg2rad(rf.angles[valid]), rf.areas[valid], s=14, color=color, marker=marker, label=f"{rf.run_label} row areas")
        ax.plot(theta_rad, rule_func(theta_dense, rf.config, *tensor.params), color=line_color, linestyle=linestyle, lw=1.1, label=f"{rf.run_label} tensor")
    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)
    ax.set_title(f"{tensor.name} area fit", fontsize=9 if compact else None)
    ax.legend(fontsize=6 if compact else 7, loc="upper right", bbox_to_anchor=(1.25, 1.15) if compact else None)


def cmd_fit_results(args) -> int:
    exp = _load_experiment(args.experiment)
    state, sources, selected_run = _fit_result_context(exp, args)
    if not sources:
        raise ValueError("No source 2D runs found for fit results.")
    peak_index, peak = _select_peak(state, args.peak)
    unit = normalize_spectral_unit(args.unit or config.get("default_spectral_unit", config.get("unit", "meV")))
    output = os.path.abspath(args.output)
    os.makedirs(os.path.dirname(output) or ".", exist_ok=True)

    if args.format == "csv":
        rows = []
        for idx, item in enumerate(state.get("peaks") or [], start=1):
            spec = item.get("spec_params", {})
            rows.append([idx, item.get("name", ""), item.get("rule", ""), spec.get("x0", [np.nan])[0], spec.get("gamma", [np.nan])[0], json.dumps(item.get("ang_params", {}), default=_json_default)])
        _write_table_csv(output, ["index", "name", "rule", "x0", "gamma", "ang_params"], rows)
    else:
        mode = args.mode
        if mode == "both":
            fig = plt.figure(figsize=(8.0, 4.0), dpi=args.dpi)
            ax1 = fig.add_subplot(1, 2, 1, projection="polar")
            ax2 = fig.add_subplot(1, 2, 2, projection="polar")
            _plot_fit_result_slice(ax1, exp, sources, state, peak, unit, compact=True)
            _plot_fit_result_area(ax2, sources, state, peak, compact=True)
        else:
            fig = plt.figure(figsize=(5.0, 4.6), dpi=args.dpi)
            ax = fig.add_subplot(1, 1, 1, projection="polar")
            if mode == "slice":
                _plot_fit_result_slice(ax, exp, sources, state, peak, unit)
            else:
                _plot_fit_result_area(ax, sources, state, peak)
        fig.tight_layout()
        fig.savefig(output, bbox_inches="tight", dpi=args.dpi)
        plt.close(fig)

    print(json.dumps({
        "output": output,
        "source": selected_run.nickname,
        "source_run_ids": [run.id for run in sources],
        "peak_index": peak_index,
        "peak_name": peak.get("name", ""),
        "mode": args.mode,
        "format": args.format,
    }, indent=2, default=_json_default))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Agent-oriented Venkata Raman plotting and export CLI.")
    sub = parser.add_subparsers(dest="command", required=True)

    p_list = sub.add_parser("list", help="List views and runs in an ExperimentSet.")
    p_list.add_argument("experiment")
    p_list.add_argument("--json", action="store_true", help="Emit machine-readable JSON.")
    p_list.add_argument("--fast", action="store_true", help="Use metadata-only HDF5 listing when available.")
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

    p_merge = sub.add_parser("merge", help="Headlessly merge raw spectra into a 2D run.")
    p_merge.add_argument("--seed", required=True)
    p_merge.add_argument("-o", "--output", required=True)
    p_merge.add_argument("--append", help="Append the merged run to an existing ExperimentSet.")
    p_merge.add_argument("--input-unit", choices=["auto", "nm", "cm-1", "meV"], default="auto")
    p_merge.add_argument("--display-unit", choices=["meV", "cm-1"])
    p_merge.add_argument("--manual-laser-nm", type=float)
    p_merge.add_argument("--dark-value", type=float, default=0.0)
    p_merge.add_argument("--nickname")
    p_merge.add_argument("--view-title")
    p_merge.add_argument("--cosmic", choices=["off", "detect", "apply-auto", "apply-indices"], default="off")
    p_merge.add_argument("--cosmic-indices", help="Comma-separated cosmic candidate indices for --cosmic apply-indices.")
    p_merge.add_argument("--cosmic-report")
    p_merge.set_defaults(func=cmd_merge)

    p_qe = sub.add_parser(
        "import-qe-raman",
        help="Import a Quantum ESPRESSO ph.x output as paired frequency-only XX/YX runs.",
    )
    p_qe.add_argument("ph_output", help="Quantum ESPRESSO ph.x text output.")
    p_qe.add_argument("-o", "--output", required=True, help="New or updated ExperimentSet output path.")
    p_qe.add_argument("--append", help="Append both generated runs to an existing ExperimentSet.")
    p_qe.add_argument("--fit-params-query", help="FIT_PARAMS run query (requires --append).")
    p_qe.add_argument("--nickname", help="Nickname prefix; defaults to the QE output filename stem.")
    p_qe.add_argument("--angle", type=float, default=0.0, help="In-plane sample rotation in degrees.")
    p_qe.add_argument("--fwhm", type=float, default=qe_raman.DEFAULT_FWHM_CM1, help="Fallback FWHM in cm-1.")
    p_qe.add_argument(
        "--match-tolerance",
        type=float,
        default=qe_raman.DEFAULT_MATCH_TOLERANCE_CM1,
        help="Maximum DFT-to-fit peak distance in cm-1.",
    )
    p_qe.add_argument("--x-min", type=float, help="Optional spectrum minimum in cm-1.")
    p_qe.add_argument("--x-max", type=float, help="Optional spectrum maximum in cm-1.")
    p_qe.add_argument("--step", type=float, default=qe_raman.DEFAULT_STEP_CM1, help="Grid spacing in cm-1.")
    p_qe.add_argument("--display-unit", choices=["meV", "cm-1"])
    p_qe.add_argument("--view-title")
    p_qe.set_defaults(func=cmd_import_qe_raman)

    p_rotate = sub.add_parser("rotate", help="Apply non-destructive 360-degree rotation metadata.")
    p_rotate.add_argument("experiment")
    p_rotate.add_argument("-o", "--output", required=True)
    p_rotate.add_argument("--run-query", action="append")
    p_rotate.add_argument("--view")
    p_rotate.add_argument("--anchor-deg", type=float, default=0.0)
    p_rotate.add_argument("--window-start-deg", type=float)
    p_rotate.add_argument("--cyclic", dest="cyclic", action="store_true", default=True)
    p_rotate.add_argument("--no-cyclic", dest="cyclic", action="store_false")
    p_rotate.add_argument("--disable", action="store_true")
    p_rotate.set_defaults(func=cmd_rotate)

    p_fit = sub.add_parser("fit2d", help="Run the 2D map fitting engine headlessly.")
    p_fit.add_argument("experiment")
    p_fit.add_argument("-o", "--output", required=True)
    p_fit.add_argument("--run-query", action="append")
    p_fit.add_argument("--role", action="append", help="parallel:QUERY or cross:QUERY")
    p_fit.add_argument("--x-min", type=float)
    p_fit.add_argument("--x-max", type=float)
    p_fit.add_argument("--config", choices=["parallel", "cross", "auto"], default="auto")
    p_fit.add_argument("--auto-peak", action="append", help="X[,ANGLE[,RUN_INDEX]]")
    p_fit.add_argument("--peak", action="append", help="RULE:X0[:GAMMA[:NAME]]")
    p_fit.add_argument("--load-params", default="auto", help="auto, none, or run query for a FIT_PARAMS run.")
    p_fit.add_argument("--si-bg")
    p_fit.add_argument("--clear-si-bg", action="store_true")
    p_fit.add_argument("--advanced-si-bg", action="store_true", help="Accepted for compatibility; inferred from the Si BG JSON schema.")
    p_fit.add_argument("--export-components")
    p_fit.add_argument("--export-diagnostics")
    p_fit.set_defaults(func=cmd_fit2d)

    p_validate = sub.add_parser(
        "validate-rowfit",
        help="Run Validate(RowFit) for existing FitParams, optionally saving caches and reports.",
    )
    p_validate.add_argument("experiment")
    selection = p_validate.add_mutually_exclusive_group(required=True)
    selection.add_argument("--params-query", action="append", help="FitParams id/nickname/query. Repeat to select several.")
    selection.add_argument("--all-fit-params", action="store_true", help="Validate every Fit Parameters run.")
    p_validate.add_argument("-o", "--output", help="Copy-on-write ExperimentSet output containing successful caches.")
    p_validate.add_argument("--row-config", help="JSON rule file for per-FitParams/dataset/parameter initials, bounds, and fixed states.")
    all_group = p_validate.add_mutually_exclusive_group()
    all_group.add_argument("--fix-all", dest="all_fixed", action="store_const", const=True)
    all_group.add_argument("--free-all", dest="all_fixed", action="store_const", const=False)
    background_group = p_validate.add_mutually_exclusive_group()
    background_group.add_argument("--fix-background", dest="background_fixed", action="store_const", const=True)
    background_group.add_argument("--free-background", dest="background_fixed", action="store_const", const=False)
    area_group = p_validate.add_mutually_exclusive_group()
    area_group.add_argument("--fix-area", dest="area_fixed", action="store_const", const=True)
    area_group.add_argument("--free-area", dest="area_fixed", action="store_const", const=False)
    gamma_group = p_validate.add_mutually_exclusive_group()
    gamma_group.add_argument("--fix-gamma", dest="gamma_fixed", action="store_const", const=True)
    gamma_group.add_argument("--free-gamma", dest="gamma_fixed", action="store_const", const=False)
    p_validate.set_defaults(all_fixed=None, background_fixed=None, area_fixed=None, gamma_fixed=None)
    p_validate.add_argument("--csv-dir", help="Directory for wide row-fit CSVs and the batch summary CSV.")
    p_validate.add_argument("--report-dir", help="Directory for multi-page validation diagnostic PDFs.")
    p_validate.add_argument("--polar-dir", help="Directory for paginated polar-area PDFs.")
    p_validate.add_argument("--polar-panels-per-page", type=int, default=4)
    p_validate.add_argument("--unit", choices=["meV", "cm-1"], default="cm-1")
    p_validate.add_argument("--strict", action="store_true", help="Do not write --output if any selected FitParams fails validation.")
    p_validate.set_defaults(func=cmd_validate_rowfit)

    p_results = sub.add_parser("fit-results", help="Export compact read-only fit diagnostics.")
    p_results.add_argument("experiment")
    p_results.add_argument("--run-query")
    p_results.add_argument("--params-query")
    p_results.add_argument("-o", "--output", required=True)
    p_results.add_argument("--mode", choices=["slice", "area", "both"], default="both")
    p_results.add_argument("--peak")
    p_results.add_argument("--unit", choices=["meV", "cm-1"])
    p_results.add_argument("--format", choices=["pdf", "png", "csv"], default="pdf")
    p_results.add_argument("--dpi", type=int, default=300)
    p_results.set_defaults(func=cmd_fit_results)


    p_tensors = sub.add_parser("export-fit-tensors", help="Fast metadata-only export of fitted Raman tensor parameters.")
    p_tensors.add_argument("experiment")
    p_tensors.add_argument("-o", "--output", required=True)
    p_tensors.add_argument("--x-min", type=float, default=250.0)
    p_tensors.add_argument("--x-max", type=float, default=400.0)
    p_tensors.set_defaults(func=cmd_export_fit_tensors)

    p_peaks = sub.add_parser("export-fit-peaks", help="Fast metadata-only export of fitted peak frequencies and widths.")
    p_peaks.add_argument("experiment")
    p_peaks.add_argument("-o", "--output", required=True)
    p_peaks.add_argument("--x-min", type=float, default=250.0)
    p_peaks.add_argument("--x-max", type=float, default=400.0)
    p_peaks.set_defaults(func=cmd_export_fit_peaks)

    p_tensor_errors = sub.add_parser(
        "export-tensor-ratio-errors",
        help="Export row-area tensor ratios and 1-sigma covariance errors from current FitParams runs.",
    )
    p_tensor_errors.add_argument("experiment")
    p_tensor_errors.add_argument("tensor_csv", help="Metadata-only export-fit-tensors CSV used to select current peaks/views.")
    p_tensor_errors.add_argument("-o", "--output", required=True, help="Tensor-ratio uncertainty summary CSV.")
    p_tensor_errors.add_argument("--row-output", required=True, help="Long-form per-angle row-area audit CSV.")
    p_tensor_errors.add_argument(
        "--method",
        choices=["row-covariance", "global-centered-bootstrap"],
        default="row-covariance",
        help="Keep the legacy row-area covariance method or center bootstrap errors on saved global ratios.",
    )
    p_tensor_errors.add_argument("--draw-output", help="Bootstrap-draw CSV; defaults beside --output.")
    p_tensor_errors.add_argument("--report-output", help="Compact Markdown calculation report; defaults beside --output.")
    p_tensor_errors.add_argument("--bootstrap-replicates", type=int, default=2000)
    p_tensor_errors.add_argument("--bootstrap-seed", type=int, default=20260723)
    p_tensor_errors.add_argument("--bootstrap-min-success", type=float, default=0.95)
    p_tensor_errors.add_argument("--bootstrap-phi-window", type=float, default=45.0)
    p_tensor_errors.add_argument(
        "--frequency-range",
        nargs=2,
        type=float,
        action="append",
        metavar=("MIN", "MAX"),
        help="Repeatable frequency window; defaults to 270-310 and 350-390 cm-1.",
    )
    p_tensor_errors.add_argument("--exclude-layer", action="append", type=float, default=[])
    p_tensor_errors.add_argument("--bulk-view", action="append", default=[], metavar="SUBSTANCE=VIEW_TITLE")
    p_tensor_errors.add_argument(
        "--cache-only",
        action="store_true",
        help="Fail instead of rebuilding missing exact row caches in memory from saved row-fit settings.",
    )
    p_tensor_errors.set_defaults(func=cmd_export_tensor_ratio_errors)

    p_peak_summary = sub.add_parser(
        "plot-peak-summary",
        help="Convert a peak-summary CSV and export the established layer-dependence plot.",
    )
    p_peak_summary.add_argument("input_csv")
    p_peak_summary.add_argument("-o", "--output", required=True, help="Vector PDF output path.")
    p_peak_summary.add_argument("--csv-output", help="Optional converted CSV output path.")
    p_peak_summary.add_argument("--unit", choices=["meV", "cm-1"], default="cm-1")
    p_peak_summary.add_argument("--tick-step", type=float)
    p_peak_summary.add_argument("--ylim", nargs=2, type=float, metavar=("MIN", "MAX"))
    p_peak_summary.add_argument("--exclude-layer", action="append", type=float, default=[])
    p_peak_summary.add_argument("--bulk-source", action="append", default=[], metavar="SUBSTANCE=TABLE_LABEL")
    p_peak_summary.add_argument("--substance", action="append", default=[])
    p_peak_summary.add_argument("--frequency-region", choices=["below", "above"])
    p_peak_summary.add_argument("--region-threshold", type=float, default=340.0)
    p_peak_summary.add_argument("--center-monolayer-window", type=float)
    p_peak_summary.add_argument("--font-family", default="Arial")
    p_peak_summary.add_argument("--font-size", type=float, default=6.0)
    p_peak_summary.add_argument("--width-cm", type=float, default=7.0)
    p_peak_summary.add_argument("--height-cm", type=float, default=5.0)
    p_peak_summary.add_argument("--line-width", type=float, default=0.5)
    p_peak_summary.add_argument("--marker-size", type=float, default=3.0)
    p_peak_summary.add_argument("--tick-length", type=float, default=5.0)
    p_peak_summary.add_argument("--grid", action="store_true")
    p_peak_summary.add_argument("--legend-position", choices=["outside", "inset"], default="outside")
    p_peak_summary.add_argument("--legend-anchor-y", type=float, default=0.985)
    p_peak_summary.add_argument("--legend-layout", choices=["compact", "split"], default="compact")
    p_peak_summary.add_argument("--title")
    p_peak_summary.set_defaults(func=cmd_plot_peak_summary)

    p_tensor_ratio = sub.add_parser(
        "plot-tensor-ratio",
        help="Plot fitted Ag tensor a/b ratios by layer, including E2g/A1g symmetry limits.",
    )
    p_tensor_ratio.add_argument("input_csv")
    p_tensor_ratio.add_argument("-o", "--output", required=True, help="Vector PDF output path.")
    p_tensor_ratio.add_argument("--csv-output")
    p_tensor_ratio.add_argument("--band", choices=["low", "high"], required=True)
    p_tensor_ratio.add_argument("--frequency-range", nargs=2, type=float, metavar=("MIN", "MAX"))
    p_tensor_ratio.add_argument("--exclude-layer", action="append", type=float, default=[])
    p_tensor_ratio.add_argument("--bulk-view", action="append", default=[], metavar="SUBSTANCE=VIEW_TITLE")
    p_tensor_ratio.add_argument("--ylim", nargs=2, type=float, metavar=("MIN", "MAX"))
    p_tensor_ratio.add_argument("--tick-step", type=float, default=0.1)
    p_tensor_ratio.add_argument("--font-family", default="Arial")
    p_tensor_ratio.add_argument("--font-size", type=float, default=7.0)
    p_tensor_ratio.add_argument("--width-cm", type=float, default=6.0)
    p_tensor_ratio.add_argument("--height-cm", type=float, default=8.0)
    p_tensor_ratio.add_argument("--line-width", type=float, default=0.5)
    p_tensor_ratio.add_argument("--marker-size", type=float, default=3.5)
    p_tensor_ratio.add_argument("--marker", default="o")
    p_tensor_ratio.add_argument("--line-style", default="-")
    p_tensor_ratio.add_argument("--tick-length", type=float, default=5.0)
    p_tensor_ratio.add_argument("--error-capsize", type=float, default=2.0)
    p_tensor_ratio.add_argument("--legend-anchor-y", type=float, default=0.985)
    p_tensor_ratio.add_argument("--filled-symmetry-limits", action="store_true")
    p_tensor_ratio.add_argument("--title")
    p_tensor_ratio.set_defaults(func=cmd_plot_tensor_ratio)

    p_tensor_frequency = sub.add_parser(
        "plot-tensor-frequency",
        help="Plot Ag1/E2g and Ag2/A1g fitted frequencies from a compact tensor CSV.",
    )
    p_tensor_frequency.add_argument("input_csv")
    p_tensor_frequency.add_argument("-o", "--output", required=True, help="Vector PDF output path.")
    p_tensor_frequency.add_argument("--csv-output")
    p_tensor_frequency.add_argument("--frequency-min", type=float, default=270.0)
    p_tensor_frequency.add_argument("--frequency-max", type=float, default=390.0)
    p_tensor_frequency.add_argument("--low-max", type=float, default=310.0)
    p_tensor_frequency.add_argument("--high-min", type=float, default=350.0)
    p_tensor_frequency.add_argument("--tick-step", type=float, default=10.0)
    p_tensor_frequency.add_argument("--exclude-layer", action="append", type=float, default=[])
    p_tensor_frequency.add_argument("--bulk-view", action="append", default=[], metavar="SUBSTANCE=VIEW_TITLE")
    p_tensor_frequency.add_argument("--font-family", default="Arial")
    p_tensor_frequency.add_argument("--font-size", type=float, default=7.0)
    p_tensor_frequency.add_argument("--width-cm", type=float, default=6.0)
    p_tensor_frequency.add_argument("--height-cm", type=float, default=16.0)
    p_tensor_frequency.add_argument("--line-width", type=float, default=0.5)
    p_tensor_frequency.add_argument("--marker-size", type=float, default=3.5)
    p_tensor_frequency.add_argument("--tick-length", type=float, default=5.0)
    p_tensor_frequency.add_argument("--legend-anchor-y", type=float, default=0.65)
    p_tensor_frequency.add_argument("--title")
    p_tensor_frequency.set_defaults(func=cmd_plot_tensor_frequency)

    p_plotc = sub.add_parser("plotc-collection", help="Fast Plot C fixed-angle slice collection for visible views.")
    p_plotc.add_argument("experiment")
    p_plotc.add_argument("-o", "--output", required=True)
    p_plotc.add_argument("--manifest")
    p_plotc.add_argument("--view", action="append", help="View id/title/substring. Repeat to select multiple views.")
    p_plotc.add_argument("--angles", nargs="+", type=float, default=[88.0, 128.0])
    p_plotc.add_argument("--colors", nargs="+", default=["c25b3e", "3f276b"])
    p_plotc.add_argument("--pols", nargs="+", default=["xx", "yx"])
    p_plotc.add_argument("--unit", choices=["meV", "cm-1"], default="meV")
    p_plotc.add_argument("--xlim", nargs=2, type=float, default=[-5.0, 75.0], metavar=("MIN", "MAX"))
    p_plotc.add_argument("--ylim-roi", nargs=2, type=float, default=[15.0, 40.0], metavar=("MIN", "MAX"))
    p_plotc.add_argument("--offset-step", type=float)
    p_plotc.add_argument("--panels-per-page", type=int, default=6)
    p_plotc.add_argument("--smooth-query", default="RPS mono")
    p_plotc.add_argument("--smooth-angle-bin", type=int, default=3)
    p_plotc.set_defaults(func=cmd_plotc_collection)

    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
