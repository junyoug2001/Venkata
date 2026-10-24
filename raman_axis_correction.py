"""Atomic Raman-shift recalibration for wavelength-based merged runs."""

from __future__ import annotations

import copy
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

import analysis
import fit_overlay
from data_structure import (
    ExperimentSet,
    Run,
    RunType,
    cm1_to_unit,
    normalize_spectral_unit,
)


_AXIS_ATOL_CM1 = 1.0e-5
_DELTA_ATOL_CM1 = 1.0e-5


@dataclass(frozen=True)
class RunAxisCorrection:
    run_id: str
    nickname: str
    wavelength_nm: np.ndarray
    old_laser_nm: float
    new_laser_nm: float
    old_shift_cm1: np.ndarray
    new_shift_cm1: np.ndarray
    delta_cm1: float


@dataclass(frozen=True)
class RamanAxisCorrectionPlan:
    selected_run_id: str
    new_laser_nm: float
    corrections: Tuple[RunAxisCorrection, ...]
    fit_owner_ids: Tuple[str, ...]

    @property
    def affected_run_ids(self) -> Tuple[str, ...]:
        return tuple(item.run_id for item in self.corrections)

    def correction_for(self, run_id: str) -> RunAxisCorrection:
        for correction in self.corrections:
            if correction.run_id == str(run_id):
                return correction
        raise KeyError(run_id)


def is_merged_raman_run(run: Optional[Run]) -> bool:
    """Return whether a run has enough merge provenance for recalibration."""
    if run is None or not run.is_2d or run.shift_cm1 is None:
        return False
    metadata = run.metadata or {}
    return bool(
        run.wl_nm is not None
        or metadata.get("merged_files")
        or metadata.get("merge_pattern_hint")
    )


def _spectral_length(run: Run) -> int:
    if run.shift_cm1 is None:
        raise ValueError(f"{run.nickname}: the Raman-shift axis is missing.")
    size = int(np.asarray(run.shift_cm1).size)
    matrix = np.asarray(run.intensity_2d) if run.intensity_2d is not None else None
    if matrix is None or matrix.ndim != 2:
        raise ValueError(f"{run.nickname}: a 2D intensity matrix is required.")
    if size not in matrix.shape:
        raise ValueError(
            f"{run.nickname}: spectral axis length {size} does not match matrix shape {matrix.shape}."
        )
    return size


def _recorded_source_paths(run: Run) -> List[str]:
    metadata = run.metadata or {}
    raw = metadata.get("merged_files") or []
    paths = [str(value) for value in raw if str(value).strip()]
    if not paths and run.source_path:
        paths = [str(run.source_path)]
    return list(dict.fromkeys(os.path.abspath(path) for path in paths))


def recover_raw_wavelength_axis(run: Run) -> np.ndarray:
    """Return the original wavelength grid, with a legacy source-file fallback."""
    expected = _spectral_length(run)
    if run.wl_nm is not None:
        wavelength = np.asarray(run.wl_nm, dtype=float)
        if wavelength.ndim == 1 and wavelength.size == expected:
            if np.all(np.isfinite(wavelength)) and np.all(wavelength > 0):
                return np.array(wavelength, copy=True)
        raise ValueError(f"{run.nickname}: stored wavelength axis is invalid or has the wrong length.")

    paths = _recorded_source_paths(run)
    if not paths:
        raise ValueError(f"{run.nickname}: no original merge files were recorded.")
    grids: List[np.ndarray] = []
    missing: List[str] = []
    for path in paths:
        if not os.path.exists(path):
            missing.append(path)
            continue
        x_values, _intensity = analysis.load_1d_xy(path)
        grids.append(np.asarray(x_values, dtype=float))
    if missing:
        preview = "\n".join(missing[:3])
        suffix = "\n…" if len(missing) > 3 else ""
        raise FileNotFoundError(
            f"{run.nickname}: original merge file(s) are unavailable:\n{preview}{suffix}"
        )
    analysis.validate_merge_x_grids(grids)
    wavelength = np.asarray(grids[0], dtype=float)
    if wavelength.ndim != 1 or wavelength.size != expected:
        raise ValueError(
            f"{run.nickname}: recovered wavelength length {wavelength.size} does not match "
            f"the merged spectral length {expected}."
        )
    if not np.all(np.isfinite(wavelength)) or np.any(wavelength <= 0):
        raise ValueError(f"{run.nickname}: recovered wavelength values must be finite and positive.")
    return np.array(wavelength, copy=True)


def infer_applied_laser_nm(run: Run, wavelength_nm: np.ndarray) -> float:
    """Infer the laser used by an existing axis and validate wavelength provenance."""
    shift = np.asarray(run.shift_cm1, dtype=float)
    wavelength = np.asarray(wavelength_nm, dtype=float)
    if shift.shape != wavelength.shape:
        raise ValueError(f"{run.nickname}: Raman and wavelength axes have different shapes.")
    finite = np.isfinite(shift) & np.isfinite(wavelength) & (wavelength > 0)
    if np.count_nonzero(finite) < 2:
        raise ValueError(f"{run.nickname}: not enough finite axis values to infer the old laser.")
    inverse_laser = shift[finite] / 1.0e7 + 1.0 / wavelength[finite]
    if np.any(inverse_laser <= 0) or not np.all(np.isfinite(inverse_laser)):
        raise ValueError(f"{run.nickname}: the existing Raman axis is not a valid wavelength conversion.")
    lasers = 1.0 / inverse_laser
    laser_nm = float(np.nanmedian(lasers))
    reconstructed = analysis.raman_shift_cm1_from_wavelength_nm(wavelength, laser_nm)
    if not np.allclose(reconstructed, shift, rtol=1.0e-9, atol=_AXIS_ATOL_CM1, equal_nan=False):
        spread = float(np.nanmax(reconstructed - shift) - np.nanmin(reconstructed - shift))
        raise ValueError(
            f"{run.nickname}: the existing axis is not a single-laser wavelength conversion "
            f"(nonuniform residual {spread:.6g} cm-1)."
        )
    return laser_nm


def _fit_state(run: Run) -> Optional[Dict[str, Any]]:
    metadata = run.metadata or {}
    state = metadata.get("fit_state") or metadata.get("map_fit_state")
    return state if isinstance(state, dict) else None


def _fit_source_ids(run: Run) -> List[str]:
    metadata = run.metadata or {}
    values = metadata.get("source_run_ids") or metadata.get("fit_params_source_run_ids") or []
    ids = [str(value) for value in values if str(value)]
    if not ids and run.is_2d and _fit_state(run) is not None:
        ids = [str(run.id)]
    return list(dict.fromkeys(ids))


def _linked_fit_closure(exp: ExperimentSet, selected_run_id: str) -> Tuple[List[str], List[str]]:
    selected = str(selected_run_id)
    affected = {selected}
    owners = set()
    changed = True
    while changed:
        changed = False
        referenced_params = set()
        for source_id in list(affected):
            source = exp.get_run(source_id)
            if source is not None:
                referenced_params.update(str(value) for value in (source.metadata or {}).get("fit_params_run_ids", []))
                active = (source.metadata or {}).get("active_fit_params_run_id")
                if active:
                    referenced_params.add(str(active))
        for run in exp.runs.values():
            state = _fit_state(run)
            if state is None:
                continue
            source_ids = set(_fit_source_ids(run))
            linked = bool(source_ids & affected) or str(run.id) in referenced_params
            if not linked:
                continue
            owners.add(str(run.id))
            if not source_ids:
                source_ids = {selected}
            before = len(affected)
            affected.update(source_ids)
            changed = changed or len(affected) != before
    source_ids = []
    for run_id in sorted(affected):
        run = exp.get_run(run_id)
        if run is None:
            raise ValueError(f"A shared fit references missing run '{run_id}'.")
        if run.run_type == RunType.FIT_PARAMS:
            continue
        source_ids.append(run_id)
    return source_ids, sorted(owners)


def build_raman_axis_correction_plan(
    exp: ExperimentSet,
    selected_run_id: str,
    new_laser_nm: float,
) -> RamanAxisCorrectionPlan:
    """Validate a complete recalibration without mutating the experiment."""
    try:
        laser = float(new_laser_nm)
    except (TypeError, ValueError) as exc:
        raise ValueError("Laser wavelength must be a number in nm.") from exc
    if not np.isfinite(laser) or laser <= 0:
        raise ValueError("Laser wavelength must be finite and greater than zero.")
    selected = exp.get_run(str(selected_run_id))
    if not is_merged_raman_run(selected):
        raise ValueError("Redo Raman-shift conversion is available only for merged 2D runs.")

    source_ids, fit_owner_ids = _linked_fit_closure(exp, str(selected_run_id))
    corrections: List[RunAxisCorrection] = []
    by_id: Dict[str, RunAxisCorrection] = {}
    for run_id in source_ids:
        run = exp.get_run(run_id)
        if not is_merged_raman_run(run):
            raise ValueError(
                f"Shared fit run '{run.nickname if run else run_id}' is not a wavelength-based merged 2D run."
            )
        wavelength = recover_raw_wavelength_axis(run)
        old_shift = np.asarray(run.shift_cm1, dtype=float)
        old_laser = infer_applied_laser_nm(run, wavelength)
        new_shift = analysis.raman_shift_cm1_from_wavelength_nm(wavelength, laser)
        differences = new_shift - old_shift
        delta = float(np.nanmedian(differences))
        if not np.allclose(differences, delta, rtol=0.0, atol=_DELTA_ATOL_CM1):
            raise ValueError(f"{run.nickname}: laser correction does not produce a constant axis shift.")
        correction = RunAxisCorrection(
            run_id=str(run.id),
            nickname=run.nickname,
            wavelength_nm=wavelength,
            old_laser_nm=old_laser,
            new_laser_nm=laser,
            old_shift_cm1=np.array(old_shift, copy=True),
            new_shift_cm1=np.array(new_shift, copy=True),
            delta_cm1=delta,
        )
        corrections.append(correction)
        by_id[correction.run_id] = correction

    for owner_id in fit_owner_ids:
        owner = exp.get_run(owner_id)
        source_deltas = [by_id[run_id].delta_cm1 for run_id in _fit_source_ids(owner) if run_id in by_id]
        if not source_deltas and str(owner.id) in by_id:
            source_deltas = [by_id[str(owner.id)].delta_cm1]
        if source_deltas and not np.allclose(source_deltas, source_deltas[0], rtol=0.0, atol=_DELTA_ATOL_CM1):
            names = ", ".join(by_id[run_id].nickname for run_id in _fit_source_ids(owner) if run_id in by_id)
            raise ValueError(
                f"Shared fit '{owner.nickname}' cannot be translated because its source runs "
                f"have different axis shifts: {names}."
            )

    return RamanAxisCorrectionPlan(
        selected_run_id=str(selected_run_id),
        new_laser_nm=laser,
        corrections=tuple(corrections),
        fit_owner_ids=tuple(fit_owner_ids),
    )


def _shift_triplet(values: Any, delta: float) -> None:
    if not isinstance(values, list):
        return
    for index, value in enumerate(values[:3]):
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            continue
        if np.isfinite(numeric):
            values[index] = numeric + float(delta)


def _translate_row_fit_config(state: Dict[str, Any], delta: float) -> None:
    config = state.get("row_fit_config")
    if not isinstance(config, dict):
        return
    datasets = config.get("datasets", config)
    if not isinstance(datasets, dict):
        return
    bg_params = state.get("bg_params") if isinstance(state.get("bg_params"), list) else []
    for dataset_key, rules in datasets.items():
        if not isinstance(rules, dict):
            continue
        const_rule = rules.get("BG_Const")
        if not isinstance(const_rule, dict):
            continue
        slope_rule = rules.get("BG_Slope_X") if isinstance(rules.get("BG_Slope_X"), dict) else {}
        slope_initial = slope_rule.get("initial", "global")
        try:
            dataset_index = int(dataset_key)
        except (TypeError, ValueError):
            dataset_index = 0
        global_slope = 0.0
        if 0 <= dataset_index < len(bg_params):
            try:
                global_slope = float(bg_params[dataset_index].get("slope_x", [0.0])[0])
            except (TypeError, ValueError, IndexError):
                global_slope = 0.0
        try:
            slope = global_slope if str(slope_initial).lower() == "global" else float(slope_initial)
        except (TypeError, ValueError):
            slope = global_slope
        for key in ("initial", "min", "max"):
            value = const_rule.get(key)
            if isinstance(value, str) and value.strip().lower() == "global":
                continue
            try:
                numeric = float(value)
            except (TypeError, ValueError):
                continue
            if np.isfinite(numeric):
                const_rule[key] = numeric - slope * float(delta)


def translate_fit_state_axis(fit_state: Dict[str, Any], delta_cm1: float) -> Dict[str, Any]:
    """Return a translated fit state that reconstructs the same intensities."""
    state = copy.deepcopy(fit_state)
    unit = normalize_spectral_unit(state.get("unit", "cm-1"), "cm-1")
    delta = float(cm1_to_unit(float(delta_cm1), unit))
    for key in ("x_min_limit", "x_max_limit"):
        try:
            value = float(state.get(key))
        except (TypeError, ValueError):
            continue
        if np.isfinite(value):
            state[key] = value + delta
    for peak in state.get("peaks", []) or []:
        if isinstance(peak, dict):
            _shift_triplet((peak.get("spec_params") or {}).get("x0"), delta)
    si_params = state.get("si_bg_peak_params")
    if isinstance(si_params, dict):
        _shift_triplet(si_params.get("x0"), delta)
    for background in state.get("bg_params", []) or []:
        if not isinstance(background, dict):
            continue
        offsets = background.get("offset")
        slopes = background.get("slope_x")
        if not isinstance(offsets, list) or not isinstance(slopes, list):
            continue
        for index in range(min(3, len(offsets), len(slopes))):
            try:
                offset = float(offsets[index])
                slope = float(slopes[index])
            except (TypeError, ValueError):
                continue
            if np.isfinite(offset) and np.isfinite(slope):
                offsets[index] = offset - slope * delta
    _translate_row_fit_config(state, delta)
    return state


def _translate_row_table(record: Dict[str, Any], delta: float) -> None:
    headers = [str(value) for value in (record.get("headers") or record.get("row_fit_headers") or [])]
    rows_key = "rows" if "rows" in record else "row_fit_params"
    rows = record.get(rows_key) or []
    if not headers or not isinstance(rows, list):
        return
    indexes = {name: headers.index(name) for name in ("BG_Const", "BG_Slope_X", "B1g_Center", "Profile_X_Shift") if name in headers}
    for row in rows:
        if not isinstance(row, list):
            continue
        if "BG_Const" in indexes and "BG_Slope_X" in indexes:
            try:
                row[indexes["BG_Const"]] = float(row[indexes["BG_Const"]]) - float(row[indexes["BG_Slope_X"]]) * delta
            except (TypeError, ValueError, IndexError):
                pass
        for name in ("B1g_Center", "Profile_X_Shift"):
            if name in indexes:
                try:
                    row[indexes[name]] = float(row[indexes[name]]) + delta
                except (TypeError, ValueError, IndexError):
                    pass


def _translated_owner_metadata(
    owner: Run,
    correction_by_id: Dict[str, RunAxisCorrection],
    base_metadata: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    metadata = copy.deepcopy(base_metadata if base_metadata is not None else (owner.metadata or {}))
    state_key = "fit_state" if isinstance(metadata.get("fit_state"), dict) else "map_fit_state"
    original_state = metadata.get(state_key)
    if not isinstance(original_state, dict):
        return metadata
    source_ids = _fit_source_ids(owner)
    deltas = [correction_by_id[run_id].delta_cm1 for run_id in source_ids if run_id in correction_by_id]
    if not deltas and str(owner.id) in correction_by_id:
        deltas = [correction_by_id[str(owner.id)].delta_cm1]
    if not deltas:
        return metadata
    delta_cm1 = float(deltas[0])
    translated = translate_fit_state_axis(original_state, delta_cm1)
    metadata[state_key] = translated
    unit = normalize_spectral_unit(translated.get("unit", "cm-1"), "cm-1")
    delta_native = float(cm1_to_unit(delta_cm1, unit))
    container = metadata.get(fit_overlay.COMPACT_ROW_RESULTS_KEY)
    if isinstance(container, dict):
        container["row_fit_config"] = copy.deepcopy(translated.get("row_fit_config") or {})
        records = container.get("sources")
        if isinstance(records, dict):
            for source_id, record in records.items():
                if source_id in correction_by_id and isinstance(record, dict):
                    _translate_row_table(record, delta_native)
    try:
        engine = analysis.MapFittingEngine()
        engine.from_dict(translated)
        metadata["fit_parameters_text"] = engine.export_parameters_text()
    except Exception as exc:
        raise ValueError(f"{owner.nickname}: could not regenerate translated fit parameters: {exc}") from exc
    return metadata


def _clone_with_corrected_axis(run: Run, correction: RunAxisCorrection, metadata: Dict[str, Any]) -> Run:
    clone = copy.copy(run)
    clone.wl_nm = np.array(correction.wavelength_nm, copy=True)
    clone.shift_cm1 = np.array(correction.new_shift_cm1, copy=True)
    clone.energy_eV = analysis.ev_from_cm1(clone.shift_cm1)
    clone.metadata = metadata
    return clone


def _refresh_compact_hashes(
    exp: ExperimentSet,
    owner: Run,
    metadata: Dict[str, Any],
    staged_sources: Dict[str, Run],
) -> None:
    state = metadata.get("fit_state") or metadata.get("map_fit_state")
    container = metadata.get(fit_overlay.COMPACT_ROW_RESULTS_KEY)
    if not isinstance(state, dict) or not isinstance(container, dict):
        return
    records = container.get("sources")
    if not isinstance(records, dict):
        return
    for source_id, record in records.items():
        source = staged_sources.get(str(source_id)) or exp.get_run(str(source_id))
        if source is not None and isinstance(record, dict):
            record["fit_state_hash"] = fit_overlay.row_fit_cache_hash(source, state)


def apply_raman_axis_correction(
    exp: ExperimentSet,
    plan: RamanAxisCorrectionPlan,
) -> Dict[str, Any]:
    """Apply a validated plan atomically and return a compact audit summary."""
    if not plan.corrections:
        raise ValueError("Correction plan has no affected runs.")
    correction_by_id = {item.run_id: item for item in plan.corrections}
    timestamp = datetime.now(timezone.utc).isoformat()
    staged_metadata: Dict[str, Dict[str, Any]] = {}
    staged_sources: Dict[str, Run] = {}

    for correction in plan.corrections:
        run = exp.get_run(correction.run_id)
        if run is None:
            raise ValueError(f"Run '{correction.run_id}' disappeared before correction was applied.")
        metadata = copy.deepcopy(run.metadata or {})
        history = metadata.get("raman_axis_conversion_history")
        history = list(history) if isinstance(history, list) else []
        history.append({
            "timestamp_utc": timestamp,
            "old_laser_nm": correction.old_laser_nm,
            "new_laser_nm": correction.new_laser_nm,
            "delta_cm1": correction.delta_cm1,
            "source": "manual_redo",
        })
        metadata["raman_axis_conversion_history"] = history
        metadata["raman_axis_conversion"] = {
            "input_unit": "nm",
            "laser_nm": correction.new_laser_nm,
            "laser_source": "manual_redo",
        }
        metadata["raman_laser_nm"] = correction.new_laser_nm
        metadata["raman_laser_source"] = "manual_redo"
        metadata["raw_x_unit"] = "nm"
        staged_metadata[run.id] = metadata
        staged_sources[run.id] = _clone_with_corrected_axis(run, correction, metadata)

    for owner_id in plan.fit_owner_ids:
        owner = exp.get_run(owner_id)
        if owner is None:
            raise ValueError(f"Fit owner '{owner_id}' disappeared before correction was applied.")
        translated_metadata = _translated_owner_metadata(
            owner,
            correction_by_id,
            base_metadata=staged_metadata.get(owner_id),
        )
        staged_metadata[owner_id] = translated_metadata

    for owner_id in plan.fit_owner_ids:
        owner = exp.get_run(owner_id)
        metadata = staged_metadata[owner_id]
        _refresh_compact_hashes(exp, owner, metadata, staged_sources)

    backups = {
        run_id: (
            exp.get_run(run_id).wl_nm,
            exp.get_run(run_id).shift_cm1,
            exp.get_run(run_id).energy_eV,
            exp.get_run(run_id).metadata,
        )
        for run_id in set(staged_metadata) | set(correction_by_id)
        if exp.get_run(run_id) is not None
    }
    try:
        for correction in plan.corrections:
            run = exp.get_run(correction.run_id)
            run.wl_nm = np.array(correction.wavelength_nm, copy=True)
            run.shift_cm1 = np.array(correction.new_shift_cm1, copy=True)
            run.energy_eV = analysis.ev_from_cm1(run.shift_cm1)
            run.metadata = staged_metadata[run.id]
        for owner_id in plan.fit_owner_ids:
            exp.get_run(owner_id).metadata = staged_metadata[owner_id]
        cache_summary = fit_overlay.invalidate_fit_caches_for_sources(
            exp,
            list(correction_by_id),
            fit_owner_ids=list(plan.fit_owner_ids),
        )
    except Exception:
        for run_id, (wl_nm, shift_cm1, energy_eV, metadata) in backups.items():
            run = exp.get_run(run_id)
            if run is None:
                continue
            run.wl_nm = wl_nm
            run.shift_cm1 = shift_cm1
            run.energy_eV = energy_eV
            run.metadata = metadata
        raise

    return {
        "selected_run_id": plan.selected_run_id,
        "new_laser_nm": plan.new_laser_nm,
        "affected_run_ids": list(plan.affected_run_ids),
        "fit_owner_ids": list(plan.fit_owner_ids),
        "deltas_cm1": {item.run_id: item.delta_cm1 for item in plan.corrections},
        **cache_summary,
    }
