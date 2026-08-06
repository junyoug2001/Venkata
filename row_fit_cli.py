from __future__ import annotations

import copy
import csv
import fnmatch
import json
import os
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.figure import Figure

import analysis
import polar_area_fitting
from data_structure import ExperimentSet, Run, cm1_to_unit, normalize_spectral_unit, unit_to_cm1
from fit_overlay import compact_experiment_fit_caches, store_compact_row_fit_results


BACKGROUND_KEYS = {"BG_Const", "BG_Slope_X", "Amp_Si"}
RULE_FIELDS = {"fitparams", "dataset", "parameter", "initial", "min", "max", "fixed"}


@dataclass
class ValidationContext:
    params_run: Run
    sources: List[Run]
    engine: analysis.MapFittingEngine
    config: Dict[str, Any]
    results: Optional[List[Dict[str, Any]]] = None
    message: str = ""

    @property
    def state(self) -> Dict[str, Any]:
        return self.engine.to_dict()


def safe_artifact_stem(run: Run) -> str:
    nickname = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(run.nickname)).strip("_.-") or "FitParams"
    run_id = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(run.id)).strip("_.-") or "run"
    return f"{nickname}__{run_id}"


def load_rule_config(path: Optional[str]) -> List[Dict[str, Any]]:
    if not path:
        return []
    with open(path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict) or not isinstance(payload.get("rules", []), list):
        raise ValueError("--row-config must contain a JSON object with a 'rules' list.")
    rules: List[Dict[str, Any]] = []
    for index, value in enumerate(payload.get("rules", []), start=1):
        if not isinstance(value, dict):
            raise ValueError(f"Row rule {index} must be a JSON object.")
        unknown = set(value) - RULE_FIELDS
        if unknown:
            raise ValueError(f"Row rule {index} has unknown field(s): {', '.join(sorted(unknown))}.")
        if "parameter" not in value:
            raise ValueError(f"Row rule {index} requires a parameter pattern.")
        if not any(key in value for key in ("initial", "min", "max", "fixed")):
            raise ValueError(f"Row rule {index} does not change initial, bounds, or fixed state.")
        if "fixed" in value and not isinstance(value["fixed"], bool):
            raise ValueError(f"Row rule {index} fixed must be a JSON boolean.")
        rule = dict(value)
        rule.setdefault("fitparams", "*")
        rule.setdefault("dataset", "*")
        rule["_index"] = index
        rule["_matches"] = 0
        rules.append(rule)
    return rules


def _glob_matches(value: Any, pattern: Any) -> bool:
    return fnmatch.fnmatchcase(str(value).lower(), str(pattern).lower())


def _parameter_group(key: str) -> Optional[str]:
    if key in BACKGROUND_KEYS:
        return "background"
    if key.endswith("_Area"):
        return "area"
    if key.endswith("_Gamma"):
        return "gamma"
    return None


def _dataset_matches(engine: analysis.MapFittingEngine, dataset_index: int, selector: Any) -> bool:
    selector = str(selector).strip().lower()
    if selector in {"", "*"}:
        return True
    if selector == str(dataset_index):
        return True
    return selector == str(engine.datasets[dataset_index].get("config", "")).strip().lower()


def _fitparams_matches(params_run: Run, pattern: Any) -> bool:
    return _glob_matches(params_run.nickname, pattern) or _glob_matches(params_run.id, pattern)


def prepare_context(exp: ExperimentSet, params_run: Run) -> ValidationContext:
    metadata = params_run.metadata or {}
    state = metadata.get("fit_state") or metadata.get("map_fit_state")
    if not isinstance(state, dict) or not state.get("peaks"):
        raise ValueError("FitParams does not contain a usable fit_state with peaks.")
    source_ids = metadata.get("source_run_ids") or metadata.get("fit_params_source_run_ids") or []
    if not source_ids:
        raise ValueError("FitParams does not link any source 2D runs.")
    sources = [exp.get_run(str(run_id)) for run_id in source_ids]
    missing = [str(run_id) for run_id, run in zip(source_ids, sources) if run is None or not run.is_2d]
    if missing:
        raise ValueError(f"Missing linked 2D source run(s): {', '.join(missing)}.")
    sources = [run for run in sources if run is not None and run.is_2d]
    if not 1 <= len(sources) <= 2:
        raise ValueError("Validate(RowFit) supports one or two linked 2D source runs.")

    engine = analysis.MapFittingEngine()
    for dataset_index, source in enumerate(sources):
        shift, angles, intensity, background_angles = analysis.display_2d_with_acquisition_angles_from_run(source)
        engine.set_data(
            dataset_index,
            shift,
            angles,
            intensity,
            source.nickname,
            background_ang=background_angles,
        )
        fallback = "parallel" if dataset_index == 0 else "cross"
        engine.datasets[dataset_index]["config"] = polar_area_fitting.infer_config(source, fallback)
    engine.from_dict(copy.deepcopy(state))
    config = engine.normalized_row_fit_config(engine.row_fit_config)
    return ValidationContext(params_run=params_run, sources=sources, engine=engine, config=config)


def apply_group_overrides(config: Dict[str, Any], *, all_fixed=None, background=None, area=None, gamma=None) -> None:
    group_values = {"background": background, "area": area, "gamma": gamma}
    for rules in config.get("datasets", {}).values():
        for key, rule in rules.items():
            group = _parameter_group(str(key))
            if group is None:
                continue
            value = all_fixed if all_fixed is not None else None
            if group_values[group] is not None:
                value = group_values[group]
            if value is not None:
                rule["fixed"] = bool(value)


def apply_json_rules(context: ValidationContext, rules: Sequence[Dict[str, Any]]) -> None:
    for rule in rules:
        if not _fitparams_matches(context.params_run, rule.get("fitparams", "*")):
            continue
        for dataset_index in context.engine.active_dataset_indices():
            if not _dataset_matches(context.engine, dataset_index, rule.get("dataset", "*")):
                continue
            dataset_rules = context.config["datasets"][str(dataset_index)]
            for key, parameter_rule in dataset_rules.items():
                if not _glob_matches(key, rule["parameter"]):
                    continue
                for field in ("initial", "min", "max", "fixed"):
                    if field in rule:
                        parameter_rule[field] = rule[field]
                rule["_matches"] += 1


def ensure_rules_matched(rules: Sequence[Dict[str, Any]]) -> None:
    unmatched = [rule for rule in rules if not int(rule.get("_matches", 0))]
    if unmatched:
        details = ", ".join(
            f"#{rule.get('_index')} {rule.get('fitparams', '*')} / {rule.get('dataset', '*')} / {rule.get('parameter')}"
            for rule in unmatched
        )
        raise ValueError(f"Row configuration rule(s) matched no parameters: {details}.")


def finalize_config(context: ValidationContext) -> Dict[str, Any]:
    context.config = context.engine.normalized_row_fit_config(context.config)
    context.engine.set_row_fit_config(context.config)
    return context.config


def validate_context(context: ValidationContext) -> Tuple[bool, str, Optional[List[Dict[str, Any]]]]:
    success, message, results = context.engine.validate_row_by_row(context.config)
    context.message = str(message)
    context.results = results
    if not success or not results:
        return False, context.message, results
    statuses = [status for result in results for status in (result.get("row_status") or [])]
    all_succeeded = bool(statuses) and all(bool(status.get("success")) for status in statuses)
    if not all_succeeded:
        succeeded = sum(bool(status.get("success")) for status in statuses)
        return False, f"{message}; only {succeeded}/{len(statuses)} rows converged", results
    return True, context.message, results


def context_summary(context: ValidationContext) -> Dict[str, Any]:
    statuses = [status for result in (context.results or []) for status in (result.get("row_status") or [])]
    anomalies = [item for status in statuses for item in (status.get("anomalies") or [])]
    bound_hits = [item for status in statuses for item in (status.get("at_bounds") or [])]
    return {
        "fit_params_run_id": context.params_run.id,
        "fit_params": context.params_run.nickname,
        "source_run_ids": [run.id for run in context.sources],
        "source_runs": [run.nickname for run in context.sources],
        "peak_count": len(context.engine.peaks),
        "rows_total": len(statuses),
        "rows_succeeded": sum(bool(status.get("success")) for status in statuses),
        "bound_hit_count": len(bound_hits),
        "anomaly_count": len(anomalies),
        "message": context.message,
    }


def persist_context(exp: ExperimentSet, context: ValidationContext) -> List[str]:
    if not context.results:
        return []
    state = copy.deepcopy(context.state)
    text = context.engine.export_parameters_text()
    source_ids = [run.id for run in context.sources]
    context.params_run.metadata["fit_state"] = copy.deepcopy(state)
    context.params_run.metadata["fit_parameters_text"] = text
    context.params_run.metadata["source_run_ids"] = list(source_ids)
    for source in context.sources:
        ids = list(source.metadata.get("fit_params_run_ids") or [])
        if context.params_run.id not in ids:
            ids.append(context.params_run.id)
        source.metadata["fit_params_run_ids"] = ids
        source.metadata["active_fit_params_run_id"] = context.params_run.id

    count = store_compact_row_fit_results(
        context.params_run,
        context.sources,
        state,
        context.results,
    )
    compact_experiment_fit_caches(exp, preferred_params_run=context.params_run)
    if count != len(context.sources):
        raise ValueError(f"Stored {count} of {len(context.sources)} compact row results.")
    return [f"compact:{context.params_run.id}:{source.id}" for source in context.sources]


def _status_columns(status: Dict[str, Any]) -> List[Any]:
    return [
        bool(status.get("success")),
        status.get("rmse", np.nan),
        str(status.get("message", "")),
        json.dumps(status.get("at_bounds") or [], ensure_ascii=False, default=str),
        json.dumps(status.get("anomalies") or [], ensure_ascii=False, default=str),
    ]


def write_context_csvs(context: ValidationContext, output_dir: str) -> List[str]:
    if not context.results:
        return []
    os.makedirs(output_dir, exist_ok=True)
    stem = safe_artifact_stem(context.params_run)
    paths: List[str] = []
    for position, result in enumerate(context.results):
        dataset_index = int(result.get("dataset_index", position))
        source = context.sources[dataset_index] if dataset_index < len(context.sources) else None
        source_stem = re.sub(r"[^A-Za-z0-9_.-]+", "_", source.nickname if source else f"dataset_{dataset_index}")
        path = os.path.abspath(os.path.join(output_dir, f"{stem}__{source_stem}_rowfit.csv"))
        headers = list(result.get("headers") or [])
        rows = result.get("rows_params") or result.get("params") or []
        statuses = result.get("row_status") or []
        with open(path, "w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(headers + ["fit_success", "rmse", "fit_message", "at_bounds_json", "anomalies_json"])
            for row_index, row in enumerate(rows):
                status = statuses[row_index] if row_index < len(statuses) else {}
                writer.writerow(list(row) + _status_columns(status))
        paths.append(path)
    return paths


def write_batch_summary_csv(path: str, entries: Sequence[Dict[str, Any]]) -> str:
    path = os.path.abspath(path)
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    fields = [
        "status", "fit_params_run_id", "fit_params", "source_run_ids", "source_runs",
        "peak_count", "rows_total", "rows_succeeded", "bound_hit_count", "anomaly_count", "message",
    ]
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for entry in entries:
            row = dict(entry)
            for key in ("source_run_ids", "source_runs"):
                row[key] = json.dumps(row.get(key) or [], ensure_ascii=False)
            writer.writerow(row)
    return path


def validation_row_fit_groups(context: ValidationContext) -> Tuple[List[float], List[List[polar_area_fitting.RowPeakFit]]]:
    if not context.results:
        raise ValueError("Validation results are required for polar-area export.")
    state = context.state
    state_unit = normalize_spectral_unit(state.get("unit", "cm-1"), "cm-1")
    targets: List[float] = []
    groups: List[List[polar_area_fitting.RowPeakFit]] = []
    for peak_index, peak in enumerate(context.engine.peaks, start=1):
        name = str(peak.get("name") or f"Peak {peak_index}")
        rule = str(peak.get("rule") or "D2h_Ag")
        center_native = float(peak.get("spec_params", {}).get("x0", [np.nan])[0])
        target_cm1 = float(unit_to_cm1(np.asarray([center_native]), state_unit)[0])
        target_group: List[polar_area_fitting.RowPeakFit] = []
        prefix = f"P{peak_index}_{name}_"
        for position, result in enumerate(context.results):
            headers = [str(value) for value in (result.get("headers") or [])]
            rows = result.get("rows_params") or result.get("params") or []
            area_column = headers.index(prefix + "Area") if prefix + "Area" in headers else None
            gamma_column = headers.index(prefix + "Gamma") if prefix + "Gamma" in headers else None
            height_column = headers.index(prefix + "Height") if prefix + "Height" in headers else None
            if area_column is None or gamma_column is None:
                raise ValueError(f"Validation result does not contain columns for {name}.")
            angles = np.asarray([float(row[0]) for row in rows], dtype=float)
            areas = np.asarray([float(row[area_column]) for row in rows], dtype=float)
            gammas_native = np.asarray([float(row[gamma_column]) for row in rows], dtype=float)
            gammas_cm1 = np.asarray(unit_to_cm1(gammas_native, state_unit), dtype=float)
            heights = np.asarray(
                [float(row[height_column]) if height_column is not None else np.nan for row in rows],
                dtype=float,
            )
            source = context.sources[position] if position < len(context.sources) else context.sources[0]
            target_group.append(polar_area_fitting.RowPeakFit(
                target=target_cm1,
                name=name,
                rule=rule,
                angles=angles,
                areas=areas,
                centers=np.full(angles.shape, target_cm1, dtype=float),
                gammas=gammas_cm1,
                heights=heights,
                config=str(result.get("config") or context.engine.datasets[position].get("config") or "parallel"),
                run_id=source.id,
                run_label=source.nickname,
            ))
        targets.append(target_cm1)
        groups.append(target_group)
    return targets, groups


def write_polar_pdf(context: ValidationContext, output_path: str, *, unit: str, panels_per_page: int = 4) -> str:
    targets, groups = validation_row_fit_groups(context)
    output_path = os.path.abspath(output_path)
    polar_area_fitting.generate_polar_area_pdf(
        ExperimentSet(id="rowfit-report"),
        context.sources,
        targets,
        output_path,
        fit_state=context.state,
        row_fit_groups=groups,
        output_unit=unit,
        export_table=False,
        panels_per_page=panels_per_page,
    )
    return output_path


def _map_extent(x: np.ndarray, angles: np.ndarray) -> Tuple[float, float, float, float]:
    x = np.asarray(x, dtype=float)
    angles = np.asarray(angles, dtype=float)
    return float(np.nanmin(x)), float(np.nanmax(x)), float(np.nanmin(angles)), float(np.nanmax(angles))


def _finite_limits(values: np.ndarray, percentiles=(1.0, 99.0)) -> Tuple[float, float]:
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return 0.0, 1.0
    low, high = np.nanpercentile(finite, percentiles)
    if not np.isfinite(low) or not np.isfinite(high) or low == high:
        low, high = float(np.nanmin(finite)), float(np.nanmax(finite))
    if low == high:
        high = low + 1.0
    return float(low), float(high)


def _summary_lines(context: ValidationContext) -> List[str]:
    summary = context_summary(context)
    lines = [
        context.params_run.nickname,
        f"FitParams ID: {context.params_run.id}",
        f"Sources: {', '.join(summary['source_runs'])}",
        f"Peaks: {summary['peak_count']}",
        f"Rows converged: {summary['rows_succeeded']}/{summary['rows_total']}",
        f"Bound hits: {summary['bound_hit_count']}",
        f"Area anomalies (>2x global): {summary['anomaly_count']}",
        f"Validation: {context.message}",
        "",
        "Resolved row parameters:",
    ]
    for dataset_index in context.engine.active_dataset_indices():
        ds = context.engine.datasets[dataset_index]
        lines.append(f"  Dataset {dataset_index} ({ds.get('label')} / {ds.get('config')}):")
        for key, rule in context.config["datasets"][str(dataset_index)].items():
            initial = rule.get("initial")
            lines.append(
                f"    {key}: initial={initial}, min={rule.get('min')}, max={rule.get('max')}, fixed={bool(rule.get('fixed'))}"
            )
    return lines


def write_diagnostic_pdf(context: ValidationContext, output_path: str, *, unit: str) -> str:
    if not context.results:
        raise ValueError("Validation results are required for a diagnostic report.")
    unit = normalize_spectral_unit(unit, "cm-1")
    output_path = os.path.abspath(output_path)
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    global_matrices = {idx: np.asarray(context.engine.reconstruct(idx), dtype=float) for idx in context.engine.active_dataset_indices()}

    with PdfPages(output_path) as pdf:
        lines = _summary_lines(context)
        for page_start in range(0, len(lines), 48):
            fig = Figure(figsize=(8.27, 11.69))
            axis = fig.add_subplot(111)
            axis.axis("off")
            axis.text(0.02, 0.98, "\n".join(lines[page_start:page_start + 48]), va="top", ha="left", family="monospace", fontsize=8)
            pdf.savefig(fig, bbox_inches="tight")
            fig.clear()

        for position, result in enumerate(context.results):
            dataset_index = int(result.get("dataset_index", position))
            source = context.sources[dataset_index]
            x_cm1 = np.asarray(result.get("x"), dtype=float)
            x_display = np.asarray(cm1_to_unit(x_cm1, unit), dtype=float)
            angles = np.asarray(result.get("ang"), dtype=float)
            raw = np.asarray(result.get("z_raw"), dtype=float)
            row = np.asarray(result.get("z_rec"), dtype=float)
            global_matrix = global_matrices[dataset_index]
            residual = raw - row
            extent = _map_extent(x_display, angles)
            vmin, vmax = _finite_limits(raw)
            residual_limit = max(abs(value) for value in _finite_limits(residual, (1.0, 99.0)))

            fig = Figure(figsize=(11.0, 7.5))
            axes = [fig.add_subplot(2, 2, index + 1) for index in range(4)]
            for axis, matrix, title in zip(
                axes,
                (raw, global_matrix, row, residual),
                ("Raw data", "Global reconstruction", "Row reconstruction", "Raw - row residual"),
            ):
                kwargs = {"origin": "lower", "aspect": "auto", "extent": extent, "interpolation": "nearest"}
                if title.endswith("residual"):
                    image = axis.imshow(matrix, cmap="coolwarm", vmin=-residual_limit, vmax=residual_limit, **kwargs)
                else:
                    image = axis.imshow(matrix, cmap="viridis", vmin=vmin, vmax=vmax, **kwargs)
                axis.set_title(title)
                axis.set_xlabel(f"Raman shift ({unit})")
                axis.set_ylabel("Angle (deg)")
                fig.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
            fig.suptitle(f"{context.params_run.nickname}\n{source.nickname}")
            fig.tight_layout(rect=(0, 0, 1, 0.94))
            pdf.savefig(fig, bbox_inches="tight")
            fig.clear()

            statuses = result.get("row_status") or []
            finite_positions = [
                index for index, status in enumerate(statuses)
                if status.get("success") and np.isfinite(float(status.get("rmse", np.nan)))
            ]
            if finite_positions:
                ordered = sorted(finite_positions, key=lambda index: float(statuses[index].get("rmse", np.inf)))
                chosen = [ordered[0], ordered[len(ordered) // 2], ordered[-1]]
                chosen = list(dict.fromkeys(chosen))
                fig = Figure(figsize=(10.0, 3.0 * len(chosen)))
                for plot_index, row_index in enumerate(chosen, start=1):
                    axis = fig.add_subplot(len(chosen), 1, plot_index)
                    axis.plot(x_display, raw[row_index], color="black", lw=0.9, label="Raw")
                    axis.plot(x_display, global_matrix[row_index], color="red", lw=0.9, label="Global")
                    axis.plot(x_display, row[row_index], color="#1f77b4", lw=0.9, label="Row")
                    status = statuses[row_index]
                    axis.set_title(f"Angle {angles[row_index]:.4g} deg; RMSE {float(status.get('rmse')):.4g}")
                    axis.set_xlabel(f"Raman shift ({unit})")
                    axis.set_ylabel("Intensity")
                    axis.legend(fontsize=8)
                fig.suptitle(f"Representative row fits: {source.nickname}")
                fig.tight_layout(rect=(0, 0, 1, 0.96))
                pdf.savefig(fig, bbox_inches="tight")
                fig.clear()
    return output_path
