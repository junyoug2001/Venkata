from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

import analysis
import polar_area_fitting
from data_structure import ExperimentSet, Run, RunType


FIT_OVERLAY_MODES = ("off", "global", "row", "both")


@dataclass
class FitOverlayData:
    centers_cm1: np.ndarray
    global_matrix: Optional[np.ndarray] = None
    row_matrix: Optional[np.ndarray] = None
    source: str = ""


_ROW_RECON_CACHE: Dict[str, np.ndarray] = {}


def normalize_fit_overlay_mode(mode: Any) -> str:
    mode = str(mode or "off").strip().lower()
    return mode if mode in FIT_OVERLAY_MODES else "off"


def fit_peak_centers_cm1(fit_state: Optional[Dict[str, Any]]) -> np.ndarray:
    centers: List[float] = []
    for peak in (fit_state or {}).get("peaks", []) or []:
        try:
            value = peak.get("spec_params", {}).get("x0", [None])[0]
            center = float(value)
        except (TypeError, ValueError, IndexError):
            continue
        if np.isfinite(center):
            centers.append(center)
    return np.asarray(centers, dtype=float)


def find_fit_state(exp: Optional[ExperimentSet], run: Run) -> Tuple[Optional[Dict[str, Any]], Optional[Run], List[str]]:
    if run is None:
        return None, None, []

    metadata = run.metadata or {}
    state = metadata.get("fit_state") or metadata.get("map_fit_state")
    if state:
        source_ids = metadata.get("fit_params_source_run_ids") or metadata.get("source_run_ids") or [run.id]
        return state, None, [str(v) for v in source_ids]

    if exp is None:
        return None, None, []

    params_run = polar_area_fitting.find_fit_params_run(exp, [run.id])
    if params_run is not None:
        state = params_run.metadata.get("fit_state") or params_run.metadata.get("map_fit_state")
        source_ids = params_run.metadata.get("source_run_ids") or [run.id]
        return state, params_run, [str(v) for v in source_ids]

    return None, None, []


def _state_cache_key(exp: Optional[ExperimentSet], run: Run, fit_state: Dict[str, Any]) -> str:
    try:
        state_blob = json.dumps(fit_state, sort_keys=True, default=str)
    except TypeError:
        state_blob = repr(fit_state)
    return f"{id(exp)}:{run.id}:{state_blob}"


def _oriented_matrix(run: Run) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    return polar_area_fitting.oriented_2d(run)


def _source_runs(exp: Optional[ExperimentSet], run: Run, source_ids: Sequence[str]) -> List[Run]:
    runs: List[Run] = []
    if exp is not None:
        for rid in source_ids:
            source = exp.get_run(str(rid))
            if source is not None and source.is_2d:
                runs.append(source)

    if run is not None and run.is_2d and all(src.id != run.id for src in runs):
        runs.append(run)

    if len(runs) > 2:
        selected = [src for src in runs if src.id == run.id]
        for src in runs:
            if len(selected) >= 2:
                break
            if src.id != run.id:
                selected.append(src)
        runs = selected

    return runs


def _engine_for_run(
    exp: Optional[ExperimentSet],
    run: Run,
    fit_state: Dict[str, Any],
    source_ids: Sequence[str],
) -> Tuple[analysis.MapFittingEngine, int]:
    runs = _source_runs(exp, run, source_ids)
    if not runs:
        raise ValueError("No source 2D run is available for fit overlay.")

    engine = analysis.MapFittingEngine()
    dataset_index = 0

    for idx, source in enumerate(runs[:2]):
        shift, angles, intensity = _oriented_matrix(source)
        engine.set_data(idx, shift, angles, intensity, source.nickname)
        config_name = polar_area_fitting.infer_config(source, "parallel" if idx == 0 else "cross")
        engine.datasets[idx]["config"] = config_name
        engine.datasets[idx]["label"] = "Parallel" if config_name == "parallel" else "Cross"
        if source.id == run.id:
            dataset_index = idx

    engine.from_dict(fit_state)
    return engine, dataset_index


def _matching_total_reconstruction(exp: Optional[ExperimentSet], run: Run) -> Optional[np.ndarray]:
    if exp is None or run is None:
        return None

    for candidate in exp.runs.values():
        metadata = candidate.metadata or {}
        if not candidate.is_2d:
            continue
        if metadata.get("source_run_id") != run.id:
            continue
        component = str(metadata.get("fit_component", "")).lower()
        is_total = component == "total" or "totalfit" in candidate.nickname.lower()
        if not (metadata.get("is_reconstruction") and is_total):
            continue
        try:
            _shift, _angles, matrix = _oriented_matrix(candidate)
        except Exception:
            continue
        return matrix
    return None


def _global_matrix(exp: Optional[ExperimentSet], run: Run, fit_state: Optional[Dict[str, Any]], source_ids: Sequence[str]) -> Optional[np.ndarray]:
    if fit_state:
        try:
            engine, dataset_index = _engine_for_run(exp, run, fit_state, source_ids)
            return np.asarray(engine.reconstruct(dataset_index), dtype=float)
        except Exception:
            pass
    return _matching_total_reconstruction(exp, run)


def _row_matrix(exp: Optional[ExperimentSet], run: Run, fit_state: Optional[Dict[str, Any]], source_ids: Sequence[str]) -> Optional[np.ndarray]:
    if not fit_state:
        return None

    cache_key = _state_cache_key(exp, run, fit_state)
    if cache_key in _ROW_RECON_CACHE:
        return _ROW_RECON_CACHE[cache_key]

    try:
        engine, dataset_index = _engine_for_run(exp, run, fit_state, source_ids)
        active_indices = [
            idx
            for idx, ds in enumerate(engine.datasets)
            if ds["z"] is not None and ds["x"] is not None and ds["ang"] is not None
        ]
        result_index = active_indices.index(dataset_index)
        success, _message, results = engine.validate_row_by_row()
        if not success or not results:
            return None
        matrix = np.asarray(results[result_index]["z_rec"], dtype=float)
    except Exception:
        return None

    _ROW_RECON_CACHE[cache_key] = matrix
    return matrix


def overlay_data(exp: Optional[ExperimentSet], run: Run, mode: Any) -> Optional[FitOverlayData]:
    mode = normalize_fit_overlay_mode(mode)
    if mode == "off" or run is None or run.run_type == RunType.FIT_PARAMS or not run.is_2d:
        return None

    fit_state, params_run, source_ids = find_fit_state(exp, run)
    centers = fit_peak_centers_cm1(fit_state)
    global_matrix = _global_matrix(exp, run, fit_state, source_ids) if mode in {"global", "both"} else None
    row_matrix = _row_matrix(exp, run, fit_state, source_ids) if mode in {"row", "both"} else None

    if centers.size == 0 and global_matrix is None and row_matrix is None:
        return None

    source = params_run.id if params_run is not None else ("run_metadata" if fit_state else "reconstruction")
    return FitOverlayData(
        centers_cm1=centers,
        global_matrix=global_matrix,
        row_matrix=row_matrix,
        source=source,
    )

