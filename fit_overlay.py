from __future__ import annotations

import copy
import hashlib
import json
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

import analysis
import polar_area_fitting
from data_structure import (
    ExperimentSet,
    Run,
    RunType,
    is_hidden_run,
    new_run_id,
    normalize_spectral_unit,
    unit_to_cm1,
)


FIT_OVERLAY_MODES = ("off", "global", "row", "both")
ROW_FIT_CACHE_KIND = "row_reconstruction"
ROW_FIT_CACHE_VERSION = 3
GLOBAL_FIT_CACHE_KIND = "global_reconstruction"
COMPACT_ROW_RESULTS_KEY = "row_fit_results"
COMPACT_ROW_RESULTS_VERSION = 1

_CACHE_LINK_KEYS = {
    "row_fit_cache_run_id",
    "fit_row_cache_run_id",
    "row_fit_cache_hash",
    "row_fit_cache_run_ids",
    "row_fit_cache_hashes",
    "global_fit_cache_run_id",
    "global_fit_cache_hash",
    "global_fit_cache_run_ids",
    "global_fit_cache_hashes",
}
_DUPLICATED_SOURCE_FIT_KEYS = {
    "fit_state",
    "map_fit_state",
    "fit_parameters_text",
    "fit_params_source_run_ids",
}


@dataclass
class FitOverlayPeak:
    name: str
    center_cm1: float
    rule: str = ""


@dataclass
class FitOverlayData:
    centers_cm1: np.ndarray
    peaks: Tuple[FitOverlayPeak, ...] = ()
    global_matrix: Optional[np.ndarray] = None
    row_matrix: Optional[np.ndarray] = None
    source: str = ""


_ROW_RECON_CACHE: Dict[str, np.ndarray] = {}
_GLOBAL_RECON_CACHE: Dict[str, np.ndarray] = {}


def _stable_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, default=str)


def _array_digest(values: Any) -> Dict[str, Any]:
    if values is None:
        return {"shape": [], "sha256": ""}
    arr = np.asarray(values, dtype=float)
    if arr.ndim == 0:
        arr = arr.reshape(1)
    arr = np.ascontiguousarray(arr)
    return {
        "shape": list(arr.shape),
        "sha256": hashlib.sha256(arr.tobytes()).hexdigest(),
    }


def row_fit_cache_hash(run: Run, fit_state: Dict[str, Any], *, version: Optional[int] = None) -> str:
    """Stable key for row-by-row reconstructions tied to fit state and displayed axes."""
    try:
        shift, angles, intensity, background_angles = analysis.display_2d_with_acquisition_angles_from_run(run)
        shape = list(np.asarray(intensity).shape)
    except Exception:
        shift = run.shift_cm1 if run is not None else []
        angles = run.angle_values if run is not None else []
        background_angles = angles
        shape = list(np.asarray(run.intensity_2d).shape) if run is not None and run.intensity_2d is not None else []
    source_mtime = getattr(run, "source_mtime", None)
    try:
        source_mtime = 0.0 if source_mtime is None else float(source_mtime)
    except (TypeError, ValueError):
        source_mtime = 0.0
    payload = {
        "kind": ROW_FIT_CACHE_KIND,
        "version": ROW_FIT_CACHE_VERSION if version is None else int(version),
        "source_run_id": getattr(run, "id", ""),
        "source_path": getattr(run, "source_path", ""),
        "source_mtime": source_mtime,
        "shape": shape,
        "shift_cm1": _array_digest(shift),
        "angles_deg": _array_digest(angles),
        "background_angles_deg": _array_digest(background_angles),
        "fit_state": fit_state or {},
    }
    return hashlib.sha256(_stable_json(payload).encode("utf-8")).hexdigest()


def normalize_fit_overlay_mode(mode: Any) -> str:
    mode = str(mode or "off").strip().lower()
    return mode if mode in FIT_OVERLAY_MODES else "off"


def fit_peak_centers_cm1(fit_state: Optional[Dict[str, Any]]) -> np.ndarray:
    return np.asarray([peak.center_cm1 for peak in fit_overlay_peaks(fit_state)], dtype=float)


def fit_overlay_peaks(fit_state: Optional[Dict[str, Any]]) -> Tuple[FitOverlayPeak, ...]:
    peaks: List[FitOverlayPeak] = []
    for idx, peak in enumerate((fit_state or {}).get("peaks", []) or [], start=1):
        name = str(peak.get("name") or f"Peak {idx}").strip() or f"Peak {idx}"
        rule = str(peak.get("rule") or "").strip()
        try:
            value = peak.get("spec_params", {}).get("x0", [None])[0]
            center = float(value)
        except (TypeError, ValueError, IndexError):
            continue
        if np.isfinite(center):
            peaks.append(FitOverlayPeak(name=name, center_cm1=center, rule=rule))
    return tuple(peaks)


def find_fit_state(exp: Optional[ExperimentSet], run: Run) -> Tuple[Optional[Dict[str, Any]], Optional[Run], List[str]]:
    if run is None:
        return None, None, []

    if exp is not None:
        params_run = polar_area_fitting.find_fit_params_run(exp, [run.id])
        if params_run is not None:
            state = params_run.metadata.get("fit_state") or params_run.metadata.get("map_fit_state")
            source_ids = params_run.metadata.get("source_run_ids") or [run.id]
            return state, params_run, [str(v) for v in source_ids]

    metadata = run.metadata or {}
    state = metadata.get("fit_state") or metadata.get("map_fit_state")
    if state:
        source_ids = metadata.get("fit_params_source_run_ids") or metadata.get("source_run_ids") or [run.id]
        return state, None, [str(v) for v in source_ids]

    return None, None, []


def _state_cache_key(exp: Optional[ExperimentSet], run: Run, fit_state: Dict[str, Any]) -> str:
    try:
        return row_fit_cache_hash(run, fit_state)
    except Exception:
        try:
            state_blob = _stable_json(fit_state)
        except TypeError:
            state_blob = repr(fit_state)
        return f"{getattr(run, 'id', '')}:{state_blob}"


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
        shift, angles, intensity, background_angles = analysis.display_2d_with_acquisition_angles_from_run(source)
        engine.set_data(
            idx,
            shift,
            angles,
            intensity,
            source.nickname,
            background_ang=background_angles,
        )
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


def _global_cache_hash(run: Run, fit_state: Dict[str, Any]) -> str:
    return f"global:{row_fit_cache_hash(run, fit_state)}"


def _global_cache_valid(cache_run: Optional[Run], source_run: Run, cache_hash: str) -> bool:
    if cache_run is None or not cache_run.is_2d:
        return False
    metadata = cache_run.metadata or {}
    if not metadata.get("is_fit_cache") or metadata.get("fit_cache_kind") != GLOBAL_FIT_CACHE_KIND:
        return False
    if str(metadata.get("source_run_id", "")) != str(source_run.id):
        return False
    if str(metadata.get("fit_state_hash", "")) != str(cache_hash):
        return False
    try:
        shift, angles, source_matrix = _oriented_matrix(source_run)
    except Exception:
        return False
    matrix = np.asarray(cache_run.intensity_2d, dtype=float)
    if matrix.shape != np.asarray(source_matrix).shape:
        return False
    if cache_run.shift_cm1 is None or cache_run.angle_values is None:
        return False
    if len(cache_run.shift_cm1) != len(shift) or len(cache_run.angle_values) != len(angles):
        return False
    return bool(
        np.allclose(cache_run.shift_cm1, shift, equal_nan=True)
        and np.allclose(cache_run.angle_values, angles, equal_nan=True)
    )


def _linked_global_cache_ids(source_run: Run, params_run: Optional[Run]) -> List[str]:
    ids: List[str] = []
    md = source_run.metadata or {}
    value = md.get("global_fit_cache_run_id")
    if value:
        ids.append(str(value))
    value = _cache_id_mapping_value(md.get("global_fit_cache_run_ids"), source_run.id)
    if value:
        ids.append(value)
    if params_run is not None:
        pmd = params_run.metadata or {}
        value = _cache_id_mapping_value(pmd.get("global_fit_cache_run_ids"), source_run.id)
        if value:
            ids.append(value)
    return ids


def _find_persistent_global_cache(
    exp: Optional[ExperimentSet],
    source_run: Run,
    cache_hash: str,
    params_run: Optional[Run] = None,
) -> Optional[Run]:
    if exp is None:
        return None
    seen: set[str] = set()
    for cache_id in _linked_global_cache_ids(source_run, params_run):
        if cache_id in seen:
            continue
        seen.add(cache_id)
        candidate = exp.get_run(cache_id)
        if _global_cache_valid(candidate, source_run, cache_hash):
            return candidate
    for candidate in exp.runs.values():
        if candidate.id in seen:
            continue
        if _global_cache_valid(candidate, source_run, cache_hash):
            return candidate
    return None


def _link_global_fit_cache(source_run: Run, cache_run: Run, cache_hash: str, params_run: Optional[Run]) -> None:
    if not isinstance(source_run.metadata, dict):
        source_run.metadata = {}
    source_run.metadata["global_fit_cache_run_id"] = cache_run.id
    source_run.metadata["global_fit_cache_hash"] = cache_hash
    _set_cache_id_mapping(source_run.metadata, "global_fit_cache_run_ids", source_run.id, cache_run.id)
    _set_cache_id_mapping(source_run.metadata, "global_fit_cache_hashes", source_run.id, cache_hash)

    if params_run is not None:
        if not isinstance(params_run.metadata, dict):
            params_run.metadata = {}
        _set_cache_id_mapping(params_run.metadata, "global_fit_cache_run_ids", source_run.id, cache_run.id)
        _set_cache_id_mapping(params_run.metadata, "global_fit_cache_hashes", source_run.id, cache_hash)
        if isinstance(cache_run.metadata, dict):
            cache_run.metadata["fit_params_run_id"] = params_run.id


def persist_global_fit_cache(
    exp: Optional[ExperimentSet],
    source_run: Run,
    fit_state: Dict[str, Any],
    matrix: Any,
    *,
    source_ids: Optional[Sequence[str]] = None,
    params_run: Optional[Run] = None,
) -> Optional[Run]:
    """Create or update a hidden 2D run containing global fit reconstruction."""
    if exp is None or source_run is None or not source_run.is_2d or not fit_state:
        return None

    if params_run is None:
        try:
            params_run = polar_area_fitting.find_fit_params_run(exp, [source_run.id])
        except Exception:
            params_run = None

    try:
        shift, angles, source_matrix = _oriented_matrix(source_run)
        matrix = np.asarray(matrix, dtype=float)
    except Exception:
        return None
    if matrix.shape != np.asarray(source_matrix).shape:
        if matrix.T.shape == np.asarray(source_matrix).shape:
            matrix = matrix.T
        else:
            return None

    cache_hash = _global_cache_hash(source_run, fit_state)
    cache_run = _find_persistent_global_cache(exp, source_run, cache_hash, params_run)
    if cache_run is None:
        cache_run = Run(
            id=_unique_cache_run_id(exp),
            source_path=source_run.source_path,
            source_mtime=source_run.source_mtime,
            shift_cm1=np.asarray(shift, dtype=float).copy(),
            energy_eV=np.asarray(source_run.energy_eV, dtype=float).copy() if source_run.energy_eV is not None and len(source_run.energy_eV) == len(shift) else None,
            intensity_2d=np.asarray(matrix, dtype=float).copy(),
            angle_values=np.asarray(angles, dtype=float).copy(),
            intensity_unit=source_run.intensity_unit,
            angle_unit=source_run.angle_unit,
            metadata={
                "nickname": f"{source_run.nickname}_GlobalFitCache",
                "hidden_from_default_lists": True,
            },
            run_type=RunType.RUN_2D,
        )
        exp.add_run(cache_run)
    else:
        cache_run.shift_cm1 = np.asarray(shift, dtype=float).copy()
        cache_run.energy_eV = np.asarray(source_run.energy_eV, dtype=float).copy() if source_run.energy_eV is not None and len(source_run.energy_eV) == len(shift) else None
        cache_run.intensity_2d = np.asarray(matrix, dtype=float).copy()
        cache_run.angle_values = np.asarray(angles, dtype=float).copy()
        cache_run.intensity_unit = source_run.intensity_unit
        cache_run.angle_unit = source_run.angle_unit

    cache_run.metadata.update({
        "nickname": f"{source_run.nickname}_GlobalFitCache",
        "hidden_from_default_lists": True,
        "is_fit_cache": True,
        "cache_run": True,
        "fit_cache_kind": GLOBAL_FIT_CACHE_KIND,
        "source_run_id": source_run.id,
        "source_run_ids": [str(v) for v in (source_ids or [source_run.id])],
        "fit_state_hash": cache_hash,
        "fit_params_run_id": params_run.id if params_run is not None else None,
    })
    _link_global_fit_cache(source_run, cache_run, cache_hash, params_run)
    _GLOBAL_RECON_CACHE[cache_hash] = np.asarray(cache_run.intensity_2d, dtype=float)
    return cache_run


def _global_matrix(exp: Optional[ExperimentSet], run: Run, fit_state: Optional[Dict[str, Any]], source_ids: Sequence[str]) -> Optional[np.ndarray]:
    if fit_state:
        cache_key = _global_cache_hash(run, fit_state)
        persistent = _find_persistent_global_cache(exp, run, cache_key)
        if persistent is not None:
            matrix = np.asarray(persistent.intensity_2d, dtype=float)
            _GLOBAL_RECON_CACHE[cache_key] = matrix
            _link_global_fit_cache(run, persistent, cache_key, None)
            return matrix
        if cache_key in _GLOBAL_RECON_CACHE:
            return _GLOBAL_RECON_CACHE[cache_key]
        try:
            engine, dataset_index = _engine_for_run(exp, run, fit_state, source_ids)
            matrix = np.asarray(engine.reconstruct(dataset_index), dtype=float)
            _GLOBAL_RECON_CACHE[cache_key] = matrix
            return matrix
        except Exception:
            pass
    return _matching_total_reconstruction(exp, run)


def _table_for_metadata(rows: Any) -> List[List[Any]]:
    out: List[List[Any]] = []
    if rows is None:
        return out
    for row in rows:
        clean_row: List[Any] = []
        for value in row:
            try:
                numeric = float(value)
            except (TypeError, ValueError):
                clean_row.append(str(value))
                continue
            clean_row.append(numeric if np.isfinite(numeric) else None)
        out.append(clean_row)
    return out


def _result_rows(result: Dict[str, Any]) -> Any:
    rows = result.get("rows_params")
    if rows is None:
        rows = result.get("params")
    return rows


def _json_safe(value: Any) -> Any:
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, np.ndarray):
        return [_json_safe(item) for item in value.tolist()]
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, float) and not np.isfinite(value):
        return None
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def compact_row_fit_record(source_run: Run, fit_state: Dict[str, Any], result: Dict[str, Any]) -> Dict[str, Any]:
    """Return the matrix-free, JSON-safe representation of one row-fit result."""
    return {
        "source_run_id": str(source_run.id),
        "fit_state_hash": row_fit_cache_hash(source_run, fit_state),
        "dataset_index": int(result.get("dataset_index", 0)),
        "config": str(result.get("config") or polar_area_fitting.infer_config(source_run)),
        "headers": [str(value) for value in (result.get("headers") or [])],
        "rows": _table_for_metadata(_result_rows(result)),
        "row_status": _json_safe(result.get("row_status") or []),
        "message": str(result.get("message") or ""),
    }


def store_compact_row_fit_results(
    params_run: Run,
    source_runs: Sequence[Run],
    fit_state: Dict[str, Any],
    results: Sequence[Dict[str, Any]],
) -> int:
    """Store row parameters/statuses on FitParams without spectra, axes, or matrices."""
    if params_run is None or params_run.run_type != RunType.FIT_PARAMS:
        raise ValueError("Compact row results require a Fit Parameters run.")
    result_by_dataset = {
        int(result.get("dataset_index", index)): result
        for index, result in enumerate(results or [])
        if isinstance(result, dict)
    }
    records: Dict[str, Dict[str, Any]] = {}
    for dataset_index, source_run in enumerate(source_runs):
        result = result_by_dataset.get(dataset_index)
        if result is None:
            continue
        records[str(source_run.id)] = compact_row_fit_record(source_run, fit_state, result)
    params_run.metadata[COMPACT_ROW_RESULTS_KEY] = {
        "version": COMPACT_ROW_RESULTS_VERSION,
        "row_fit_config": _json_safe((fit_state or {}).get("row_fit_config") or {}),
        "sources": records,
    }
    return len(records)


def get_compact_row_fit_record(
    source_run: Run,
    fit_state: Optional[Dict[str, Any]],
    params_run: Optional[Run],
) -> Optional[Dict[str, Any]]:
    if source_run is None or not fit_state or params_run is None:
        return None
    container = (params_run.metadata or {}).get(COMPACT_ROW_RESULTS_KEY)
    if not isinstance(container, dict) or int(container.get("version", 0) or 0) != COMPACT_ROW_RESULTS_VERSION:
        return None
    records = container.get("sources")
    if not isinstance(records, dict):
        return None
    record = records.get(str(source_run.id))
    if not isinstance(record, dict):
        return None
    try:
        expected = row_fit_cache_hash(source_run, fit_state)
    except Exception:
        return None
    if str(record.get("fit_state_hash", "")) != expected:
        return None
    headers = record.get("headers")
    rows = record.get("rows")
    if not isinstance(headers, list) or not isinstance(rows, list) or not headers or not rows:
        return None
    return record


def has_saved_row_fit_result(
    exp: Optional[ExperimentSet],
    source_run: Run,
    fit_state: Optional[Dict[str, Any]],
    *,
    params_run: Optional[Run] = None,
) -> bool:
    if get_compact_row_fit_record(source_run, fit_state, params_run) is not None:
        return True
    return get_persistent_row_fit_cache(exp, source_run, fit_state, params_run=params_run) is not None


def _cache_id_mapping_value(mapping: Any, source_run_id: str) -> Optional[str]:
    if not isinstance(mapping, dict):
        return None
    value = mapping.get(source_run_id) or mapping.get(str(source_run_id))
    return str(value) if value else None


def _set_cache_id_mapping(metadata: Dict[str, Any], key: str, source_run_id: str, value: str) -> None:
    mapping = metadata.get(key)
    if not isinstance(mapping, dict):
        mapping = {}
    mapping[str(source_run_id)] = str(value)
    metadata[key] = mapping


def _linked_cache_ids(source_run: Run, params_run: Optional[Run]) -> List[str]:
    ids: List[str] = []
    md = source_run.metadata or {}
    for key in ("row_fit_cache_run_id", "fit_row_cache_run_id"):
        value = md.get(key)
        if value:
            ids.append(str(value))
    value = _cache_id_mapping_value(md.get("row_fit_cache_run_ids"), source_run.id)
    if value:
        ids.append(value)
    if params_run is not None:
        pmd = params_run.metadata or {}
        value = _cache_id_mapping_value(pmd.get("row_fit_cache_run_ids"), source_run.id)
        if value:
            ids.append(value)
    return ids


def _row_cache_valid(cache_run: Optional[Run], source_run: Run, cache_hash: str) -> bool:
    if cache_run is None or not cache_run.is_2d:
        return False
    metadata = cache_run.metadata or {}
    if not metadata.get("is_fit_cache") or metadata.get("fit_cache_kind") != ROW_FIT_CACHE_KIND:
        return False
    if str(metadata.get("source_run_id", "")) != str(source_run.id):
        return False
    if str(metadata.get("fit_state_hash", "")) != str(cache_hash):
        return False
    try:
        shift, angles, source_matrix = _oriented_matrix(source_run)
    except Exception:
        return False
    matrix = np.asarray(cache_run.intensity_2d, dtype=float)
    if matrix.shape != np.asarray(source_matrix).shape:
        return False
    if cache_run.shift_cm1 is None or cache_run.angle_values is None:
        return False
    if len(cache_run.shift_cm1) != len(shift) or len(cache_run.angle_values) != len(angles):
        return False
    return bool(
        np.allclose(cache_run.shift_cm1, shift, equal_nan=True)
        and np.allclose(cache_run.angle_values, angles, equal_nan=True)
    )


def _find_persistent_row_cache(
    exp: Optional[ExperimentSet],
    source_run: Run,
    cache_hash: str,
    params_run: Optional[Run] = None,
) -> Optional[Run]:
    if exp is None:
        return None
    seen: set[str] = set()
    for cache_id in _linked_cache_ids(source_run, params_run):
        if cache_id in seen:
            continue
        seen.add(cache_id)
        candidate = exp.get_run(cache_id)
        if _row_cache_valid(candidate, source_run, cache_hash):
            return candidate
    for candidate in exp.runs.values():
        if candidate.id in seen:
            continue
        if _row_cache_valid(candidate, source_run, cache_hash):
            return candidate
    return None


def _find_legacy_row_cache_for_migration(
    exp: ExperimentSet,
    source_run: Run,
    fit_state: Dict[str, Any],
    params_run: Run,
) -> Optional[Run]:
    """Find the newest legacy matrix cache with the exact current validity hash."""
    acceptable_hashes = {row_fit_cache_hash(source_run, fit_state)}
    linked_ids = _linked_cache_ids(source_run, params_run)
    candidates: List[Run] = []
    seen_ids: set[str] = set()
    for cache_id in linked_ids:
        candidate = exp.get_run(cache_id)
        if candidate is not None and candidate.id not in seen_ids:
            candidates.append(candidate)
            seen_ids.add(candidate.id)
    for candidate in reversed(list(exp.runs.values())):
        if candidate.id not in seen_ids:
            candidates.append(candidate)
            seen_ids.add(candidate.id)
    for candidate in candidates:
        metadata = candidate.metadata or {}
        linked_params_id = metadata.get("fit_params_run_id")
        if linked_params_id and str(linked_params_id) != str(params_run.id):
            continue
        stored_hash = str(metadata.get("fit_state_hash", ""))
        if stored_hash not in acceptable_hashes:
            continue
        if _row_cache_valid(candidate, source_run, stored_hash):
            return candidate
    return None


def _find_replaceable_row_cache(
    exp: Optional[ExperimentSet],
    source_run: Run,
    params_run: Optional[Run],
) -> Optional[Run]:
    """Return the cache explicitly linked to one FitParams/source pair.

    Unlike normal cache lookup this intentionally ignores the fit-state hash so
    callers can replace a stale cache without accumulating hidden runs.  A
    generic source-level link is not sufficient because one source may belong
    to multiple FitParams runs.
    """
    if exp is None or params_run is None:
        return None
    mapping = (params_run.metadata or {}).get("row_fit_cache_run_ids")
    cache_id = _cache_id_mapping_value(mapping, source_run.id)
    if not cache_id:
        return None
    candidate = exp.get_run(cache_id)
    if candidate is None or not candidate.is_2d:
        return None
    metadata = candidate.metadata or {}
    if not metadata.get("is_fit_cache") or metadata.get("fit_cache_kind") != ROW_FIT_CACHE_KIND:
        return None
    if str(metadata.get("source_run_id", "")) != str(source_run.id):
        return None
    linked_params_id = metadata.get("fit_params_run_id")
    if linked_params_id and str(linked_params_id) != str(params_run.id):
        return None
    return candidate


def _unique_cache_run_id(exp: ExperimentSet) -> str:
    base = new_run_id(prefix="fitcache")
    run_id = base
    suffix = 2
    while run_id in exp.runs:
        run_id = f"{base}_{suffix}"
        suffix += 1
    return run_id


def _link_row_fit_cache(source_run: Run, cache_run: Run, cache_hash: str, params_run: Optional[Run]) -> None:
    if not isinstance(source_run.metadata, dict):
        source_run.metadata = {}
    source_run.metadata["row_fit_cache_run_id"] = cache_run.id
    source_run.metadata["row_fit_cache_hash"] = cache_hash
    _set_cache_id_mapping(source_run.metadata, "row_fit_cache_run_ids", source_run.id, cache_run.id)
    _set_cache_id_mapping(source_run.metadata, "row_fit_cache_hashes", source_run.id, cache_hash)

    if params_run is not None:
        if not isinstance(params_run.metadata, dict):
            params_run.metadata = {}
        _set_cache_id_mapping(params_run.metadata, "row_fit_cache_run_ids", source_run.id, cache_run.id)
        _set_cache_id_mapping(params_run.metadata, "row_fit_cache_hashes", source_run.id, cache_hash)
        if isinstance(cache_run.metadata, dict):
            cache_run.metadata["fit_params_run_id"] = params_run.id


def persist_row_fit_cache(
    exp: Optional[ExperimentSet],
    source_run: Run,
    fit_state: Dict[str, Any],
    result: Dict[str, Any],
    *,
    source_ids: Optional[Sequence[str]] = None,
    params_run: Optional[Run] = None,
    replace_stale: bool = False,
) -> Optional[Run]:
    """Create or update a hidden 2D run containing row-by-row fit reconstruction."""
    if exp is None or source_run is None or not source_run.is_2d or not fit_state or not result:
        return None

    if params_run is None:
        try:
            params_run = polar_area_fitting.find_fit_params_run(exp, [source_run.id])
        except Exception:
            params_run = None

    try:
        shift, angles, source_matrix = _oriented_matrix(source_run)
        matrix = np.asarray(result.get("z_rec"), dtype=float)
    except Exception:
        return None
    if matrix.shape != np.asarray(source_matrix).shape:
        if matrix.T.shape == np.asarray(source_matrix).shape:
            matrix = matrix.T
        else:
            return None

    cache_hash = row_fit_cache_hash(source_run, fit_state)
    cache_run = _find_persistent_row_cache(exp, source_run, cache_hash, params_run)
    if cache_run is None and replace_stale:
        cache_run = _find_replaceable_row_cache(exp, source_run, params_run)
    if cache_run is None:
        cache_run = Run(
            id=_unique_cache_run_id(exp),
            source_path=source_run.source_path,
            source_mtime=source_run.source_mtime,
            shift_cm1=np.asarray(shift, dtype=float).copy(),
            energy_eV=np.asarray(source_run.energy_eV, dtype=float).copy() if source_run.energy_eV is not None and len(source_run.energy_eV) == len(shift) else None,
            intensity_2d=np.asarray(matrix, dtype=float).copy(),
            angle_values=np.asarray(angles, dtype=float).copy(),
            intensity_unit=source_run.intensity_unit,
            angle_unit=source_run.angle_unit,
            metadata={
                "nickname": f"{source_run.nickname}_RowFitCache",
                "hidden_from_default_lists": True,
            },
            run_type=RunType.RUN_2D,
        )
        exp.add_run(cache_run)
    else:
        cache_run.shift_cm1 = np.asarray(shift, dtype=float).copy()
        cache_run.energy_eV = np.asarray(source_run.energy_eV, dtype=float).copy() if source_run.energy_eV is not None and len(source_run.energy_eV) == len(shift) else None
        cache_run.intensity_2d = np.asarray(matrix, dtype=float).copy()
        cache_run.angle_values = np.asarray(angles, dtype=float).copy()
        cache_run.intensity_unit = source_run.intensity_unit
        cache_run.angle_unit = source_run.angle_unit

    cache_run.metadata.update({
        "nickname": f"{source_run.nickname}_RowFitCache",
        "hidden_from_default_lists": True,
        "is_fit_cache": True,
        "cache_run": True,
        "fit_cache_kind": ROW_FIT_CACHE_KIND,
        "source_run_id": source_run.id,
        "source_run_ids": [str(v) for v in (source_ids or [source_run.id])],
        "fit_state_hash": cache_hash,
        "fit_params_run_id": params_run.id if params_run is not None else None,
        "row_fit_headers": [str(v) for v in (result.get("headers") or [])],
        "row_fit_params": _table_for_metadata(_result_rows(result)),
    })
    _link_row_fit_cache(source_run, cache_run, cache_hash, params_run)
    _ROW_RECON_CACHE[cache_hash] = np.asarray(cache_run.intensity_2d, dtype=float)
    return cache_run


def link_existing_row_fit_caches(
    exp: Optional[ExperimentSet],
    source_runs: Sequence[Run],
    fit_state: Dict[str, Any],
    *,
    params_run: Optional[Run] = None,
) -> None:
    if exp is None or not fit_state:
        return
    for source_run in source_runs:
        if source_run is None or not source_run.is_2d:
            continue
        try:
            cache_hash = row_fit_cache_hash(source_run, fit_state)
        except Exception:
            continue
        cache_run = _find_persistent_row_cache(exp, source_run, cache_hash, params_run)
        if cache_run is not None:
            _link_row_fit_cache(source_run, cache_run, cache_hash, params_run)


def get_persistent_row_fit_cache(
    exp: Optional[ExperimentSet],
    source_run: Run,
    fit_state: Optional[Dict[str, Any]],
    *,
    params_run: Optional[Run] = None,
) -> Optional[Run]:
    """Return a valid saved row-fit cache run without computing a new fit."""
    if exp is None or source_run is None or not source_run.is_2d or not fit_state:
        return None
    try:
        cache_hash = row_fit_cache_hash(source_run, fit_state)
    except Exception:
        return None
    cache_run = _find_persistent_row_cache(exp, source_run, cache_hash, params_run)
    if cache_run is not None:
        _link_row_fit_cache(source_run, cache_run, cache_hash, params_run)
    return cache_run


def _record_parameter_arrays(record: Dict[str, Any]) -> Tuple[List[str], List[List[Any]]]:
    return (
        [str(value) for value in (record.get("headers") or [])],
        list(record.get("rows") or record.get("row_fit_params") or []),
    )


def reconstruct_compact_row_matrix(
    exp: Optional[ExperimentSet],
    source_run: Run,
    fit_state: Dict[str, Any],
    source_ids: Sequence[str],
    record: Dict[str, Any],
) -> np.ndarray:
    """Rebuild a row-fit map from compact fitted parameters in memory."""
    engine, dataset_index = _engine_for_run(exp, source_run, fit_state, source_ids)
    ds = engine.datasets[dataset_index]
    x = np.asarray(ds["x"], dtype=float)
    angles = np.asarray(ds["ang"], dtype=float)
    headers, rows = _record_parameter_arrays(record)
    if len(rows) != len(angles):
        raise ValueError("Compact row-result count does not match the source angular axis.")

    matrix = np.zeros((len(angles), len(x)), dtype=float)
    for row_index, row in enumerate(rows):
        values: Dict[str, float] = {}
        for column, header in enumerate(headers):
            if column >= len(row) or row[column] is None:
                continue
            try:
                values[header] = float(row[column])
            except (TypeError, ValueError):
                continue

        angle = float(angles[row_index])
        y = values.get("BG_Const", 0.0) + values.get("BG_Slope_X", 0.0) * x
        config_name = str(ds.get("config", "parallel"))
        if getattr(engine, "si_bg_mode", "none") == "advanced_si_bg_v2":
            y = y + engine.evaluate_advanced_si_bg(
                x,
                np.full(x.shape, angle, dtype=float),
                config_name,
                values.get("Amp_Si", 0.0),
                values.get("Amp_B1g_Peak", 0.0),
                values.get("B1g_Center", engine.si_bg_peak_params["x0"][0]),
                values.get("B1g_Gamma", engine.si_bg_peak_params["gamma"][0]),
                values.get("B1g_Phi", engine.si_bg_peak_params["phi"][0]),
                normalization_theta=angles,
            )
        elif getattr(engine, "si_bg_mode", "none") != "none":
            y = y + values.get("Amp_Si", 0.0) * engine.evaluate_si_bg(x, config_name)

        for peak_index, peak in enumerate(engine.peaks, start=1):
            prefix = f"P{peak_index}_{peak.get('name', '')}"
            area = values.get(f"{prefix}_Area", 0.0)
            gamma = values.get(f"{prefix}_Gamma", peak.get("spec_params", {}).get("gamma", [1.0])[0])
            center = peak.get("spec_params", {}).get("x0", [0.0])[0]
            y = y + float(area) * analysis.lorentzian_normalized(x, float(center), float(gamma))
        matrix[row_index, :] = y
    return matrix


def cached_row_fits_for_runs(
    exp: Optional[ExperimentSet],
    runs: Sequence[Run],
    fit_state: Optional[Dict[str, Any]],
    targets: Sequence[float],
    *,
    params_run: Optional[Run] = None,
) -> List[List[polar_area_fitting.RowPeakFit]]:
    """Read inspected peak areas/Gammas from compact results or a legacy cache."""
    if exp is None or not fit_state:
        raise ValueError("A saved Fit Parameters state is required for cached polar-area export.")
    state_peaks = list(fit_state.get("peaks") or [])
    if not state_peaks:
        raise ValueError("The selected Fit Parameters state contains no peaks.")
    unit = normalize_spectral_unit(fit_state.get("unit", "cm-1"), "cm-1")
    indexed_peaks: List[Tuple[int, float, Dict[str, Any]]] = []
    for peak_index, peak in enumerate(state_peaks):
        try:
            center = float(peak.get("spec_params", {}).get("x0", [np.nan])[0])
            center = float(unit_to_cm1(np.asarray([center], dtype=float), unit)[0])
        except Exception:
            continue
        if np.isfinite(center):
            indexed_peaks.append((peak_index, center, peak))
    if not indexed_peaks:
        raise ValueError("The selected Fit Parameters state contains no finite peak centers.")

    if params_run is None:
        params_run = polar_area_fitting.find_fit_params_run(exp, [run.id for run in runs])
    table_by_run: Dict[str, Dict[str, Any]] = {}
    for run in runs:
        record = get_compact_row_fit_record(run, fit_state, params_run)
        if record is not None:
            table_by_run[str(run.id)] = record
            continue
        cache_run = get_persistent_row_fit_cache(exp, run, fit_state, params_run=params_run)
        if cache_run is not None:
            table_by_run[str(run.id)] = {
                "headers": (cache_run.metadata or {}).get("row_fit_headers") or [],
                "rows": (cache_run.metadata or {}).get("row_fit_params") or [],
            }
        else:
            raise ValueError(
                f"No matching saved row-fit result exists for '{run.nickname}'. "
                "Open the 2D fit inspector, run Validate, choose Save Caches, and commit the fitting dialog."
            )

    groups: List[List[polar_area_fitting.RowPeakFit]] = []
    for target in [float(value) for value in targets]:
        peak_index, center, peak = min(indexed_peaks, key=lambda item: abs(item[1] - target))
        prefix = f"P{peak_index + 1}_"
        group: List[polar_area_fitting.RowPeakFit] = []
        for run_index, run in enumerate(runs):
            headers, rows = _record_parameter_arrays(table_by_run[str(run.id)])
            if not headers or not rows:
                raise ValueError(f"The saved row-fit result for '{run.nickname}' has no parameter table.")
            area_col = gamma_col = height_col = None
            for column, header in enumerate(headers):
                if not header.startswith(prefix):
                    continue
                if header.endswith("_Area"):
                    area_col = column
                elif header.endswith("_Gamma"):
                    gamma_col = column
                elif header.endswith("_Height"):
                    height_col = column
            if area_col is None:
                raise ValueError(
                    f"The saved row-fit result for '{run.nickname}' does not contain peak {peak_index + 1} areas."
                )
            angle_col = headers.index("Angle") if "Angle" in headers else 0

            def column_values(column: Optional[int], default: float = np.nan) -> np.ndarray:
                values: List[float] = []
                for row in rows:
                    try:
                        values.append(float(row[column]) if column is not None else default)
                    except (TypeError, ValueError, IndexError):
                        values.append(default)
                return np.asarray(values, dtype=float)

            angles = column_values(angle_col)
            areas = column_values(area_col)
            gammas = column_values(gamma_col)
            heights = column_values(height_col)
            if not np.isfinite(areas).any():
                raise ValueError(f"The saved row-fit result for '{run.nickname}' has no finite areas for peak {peak_index + 1}.")
            group.append(
                polar_area_fitting.RowPeakFit(
                    target=target,
                    name=str(peak.get("name") or f"Peak {peak_index + 1}"),
                    rule=str(peak.get("rule") or "D2h_Ag"),
                    angles=angles,
                    areas=areas,
                    centers=np.full(angles.shape, center, dtype=float),
                    gammas=gammas,
                    heights=heights,
                    config=polar_area_fitting.infer_config(run, "parallel" if run_index == 0 else "cross"),
                    run_id=run.id,
                    run_label=run.nickname,
                )
            )
        groups.append(group)
    return groups


def _is_automatic_fit_cache(run: Run) -> bool:
    metadata = run.metadata or {}
    return bool(
        metadata.get("is_fit_cache")
        or (
            metadata.get("cache_run")
            and metadata.get("fit_cache_kind") in {ROW_FIT_CACHE_KIND, GLOBAL_FIT_CACHE_KIND}
        )
    )


def compact_experiment_fit_caches(
    exp: ExperimentSet,
    *,
    preferred_params_run: Optional[Run] = None,
) -> Dict[str, int]:
    """Migrate exact legacy row tables, then remove all automatic matrix caches."""
    migrated = 0
    params_runs = [run for run in exp.runs.values() if run.run_type == RunType.FIT_PARAMS]
    for params_run in params_runs:
        metadata = params_run.metadata or {}
        state = metadata.get("fit_state") or metadata.get("map_fit_state")
        if not isinstance(state, dict):
            continue
        source_ids = [str(value) for value in (metadata.get("source_run_ids") or [])]
        sources = [exp.get_run(run_id) for run_id in source_ids]
        sources = [run for run in sources if run is not None and run.is_2d]
        current = metadata.get(COMPACT_ROW_RESULTS_KEY)
        records = dict(current.get("sources") or {}) if isinstance(current, dict) else {}
        for source in sources:
            if get_compact_row_fit_record(source, state, params_run) is not None:
                continue
            cache = _find_legacy_row_cache_for_migration(exp, source, state, params_run)
            if cache is None:
                continue
            cache_metadata = cache.metadata or {}
            headers = cache_metadata.get("row_fit_headers") or []
            rows = cache_metadata.get("row_fit_params") or []
            if not headers or not rows:
                continue
            records[str(source.id)] = {
                "source_run_id": str(source.id),
                "fit_state_hash": row_fit_cache_hash(source, state),
                "dataset_index": source_ids.index(str(source.id)) if str(source.id) in source_ids else 0,
                "config": polar_area_fitting.infer_config(source),
                "headers": [str(value) for value in headers],
                "rows": _table_for_metadata(rows),
                "row_status": _json_safe(cache_metadata.get("row_status") or []),
                "message": "Migrated from legacy matrix cache",
            }
            migrated += 1
        if records:
            metadata[COMPACT_ROW_RESULTS_KEY] = {
                "version": COMPACT_ROW_RESULTS_VERSION,
                "row_fit_config": _json_safe((state or {}).get("row_fit_config") or {}),
                "sources": records,
            }

    removed_ids = [run.id for run in exp.runs.values() if _is_automatic_fit_cache(run)]
    for run_id in removed_ids:
        exp.runs.pop(run_id, None)

    source_links: Dict[str, List[str]] = {}
    for params_run in params_runs:
        for source_id in (params_run.metadata or {}).get("source_run_ids") or []:
            source_links.setdefault(str(source_id), []).append(str(params_run.id))
        for key in _CACHE_LINK_KEYS:
            params_run.metadata.pop(key, None)

    stripped = 0
    preferred_id = str(preferred_params_run.id) if preferred_params_run is not None else None
    preferred_sources = set(
        str(value) for value in ((preferred_params_run.metadata or {}).get("source_run_ids") or [])
    ) if preferred_params_run is not None else set()
    for source_id, params_ids in source_links.items():
        source = exp.get_run(source_id)
        if source is None:
            continue
        metadata = source.metadata
        for key in _CACHE_LINK_KEYS | _DUPLICATED_SOURCE_FIT_KEYS:
            if key in metadata:
                metadata.pop(key, None)
                stripped += 1
        metadata["fit_params_run_ids"] = list(dict.fromkeys(params_ids))
        active = metadata.get("active_fit_params_run_id")
        if preferred_id and source_id in preferred_sources:
            active = preferred_id
        if active not in metadata["fit_params_run_ids"]:
            active = metadata["fit_params_run_ids"][-1]
        metadata["active_fit_params_run_id"] = active

    _ROW_RECON_CACHE.clear()
    _GLOBAL_RECON_CACHE.clear()
    return {
        "migrated_row_results": migrated,
        "removed_matrix_caches": len(removed_ids),
        "stripped_source_metadata": stripped,
    }


def _unique_params_nickname(exp: ExperimentSet, base: str) -> str:
    existing = {run.nickname for run in exp.runs.values()}
    if base not in existing:
        return base
    suffix = 2
    while f"{base}_{suffix}" in existing:
        suffix += 1
    return f"{base}_{suffix}"


def commit_fit_params_transaction(
    exp: ExperimentSet,
    source_runs: Sequence[Run],
    fit_state: Dict[str, Any],
    fit_parameters_text: str,
    *,
    existing_params_run: Optional[Run] = None,
    save_as_new: bool = False,
    row_results: Optional[Sequence[Dict[str, Any]]] = None,
    explicit_runs: Optional[Sequence[Run]] = None,
) -> Tuple[Run, Dict[str, int]]:
    """Atomically commit one dialog session and compact automatic caches."""
    source_runs = [run for run in source_runs if run is not None and run.is_2d]
    if not source_runs:
        raise ValueError("At least one 2D source run is required.")
    explicit_runs = list(explicit_runs or [])
    staged_by_id: Dict[str, List[Run]] = {}
    for run in explicit_runs:
        staged_by_id.setdefault(str(run.id), []).append(run)
    duplicates = {
        run_id: runs
        for run_id, runs in staged_by_id.items()
        if len(runs) > 1
    }
    if duplicates:
        details = "; ".join(
            f"{run_id} ({', '.join(run.nickname for run in runs)})"
            for run_id, runs in duplicates.items()
        )
        raise ValueError(f"Duplicate run ID(s) inside staged exports: {details}")
    for run in explicit_runs:
        existing = exp.get_run(run.id)
        if existing is None:
            continue
        visibility = "hidden" if is_hidden_run(existing) else "visible"
        raise ValueError(
            f"Staged run '{run.nickname}' uses ID '{run.id}', which already belongs "
            f"to {visibility} run '{existing.nickname}'."
        )

    original_runs = dict(exp.runs)
    original_metadata = {run_id: copy.deepcopy(run.metadata) for run_id, run in exp.runs.items()}
    try:
        update_existing = (
            not save_as_new
            and existing_params_run is not None
            and existing_params_run.id in exp.runs
            and existing_params_run.run_type == RunType.FIT_PARAMS
        )
        if update_existing:
            target = existing_params_run
        else:
            names = "_".join(run.nickname for run in source_runs)
            base_nickname = (
                existing_params_run.nickname
                if save_as_new and existing_params_run is not None
                else f"FitParams_{names}"
            )
            target = Run(
                id=new_run_id(prefix="params"),
                source_path="",
                metadata={"nickname": _unique_params_nickname(exp, base_nickname)},
                run_type=RunType.FIT_PARAMS,
            )
            while target.id in exp.runs:
                target.id = new_run_id(prefix="params")
            exp.add_run(target)

        target.metadata["fit_state"] = copy.deepcopy(fit_state)
        target.metadata["fit_parameters_text"] = str(fit_parameters_text)
        target.metadata["source_run_ids"] = [str(run.id) for run in source_runs]
        if row_results:
            store_compact_row_fit_results(target, source_runs, fit_state, row_results)
        else:
            container = target.metadata.get(COMPACT_ROW_RESULTS_KEY)
            if isinstance(container, dict):
                records = container.get("sources") or {}
                valid = {
                    str(run.id): records[str(run.id)]
                    for run in source_runs
                    if str(run.id) in records
                    and str(records[str(run.id)].get("fit_state_hash", "")) == row_fit_cache_hash(run, fit_state)
                }
                if valid:
                    target.metadata[COMPACT_ROW_RESULTS_KEY] = {
                        "version": COMPACT_ROW_RESULTS_VERSION,
                        "row_fit_config": _json_safe((fit_state or {}).get("row_fit_config") or {}),
                        "sources": valid,
                    }
                else:
                    target.metadata.pop(COMPACT_ROW_RESULTS_KEY, None)

        for run in explicit_runs:
            exp.add_run(run)

        for source in source_runs:
            ids = list(source.metadata.get("fit_params_run_ids") or [])
            if target.id not in ids:
                ids.append(target.id)
            source.metadata["fit_params_run_ids"] = ids
            source.metadata["active_fit_params_run_id"] = target.id
        summary = compact_experiment_fit_caches(exp, preferred_params_run=target)
        summary["compact_row_results"] = len(
            ((target.metadata.get(COMPACT_ROW_RESULTS_KEY) or {}).get("sources") or {})
        )
        summary["explicit_runs"] = len(explicit_runs)
        return target, summary
    except Exception:
        exp.runs.clear()
        exp.runs.update(original_runs)
        for run_id, metadata in original_metadata.items():
            if run_id in exp.runs:
                exp.runs[run_id].metadata = metadata
        raise


def _row_matrix(
    exp: Optional[ExperimentSet],
    run: Run,
    fit_state: Optional[Dict[str, Any]],
    source_ids: Sequence[str],
    params_run: Optional[Run] = None,
) -> Optional[np.ndarray]:
    if not fit_state:
        return None

    cache_key = _state_cache_key(exp, run, fit_state)
    compact = get_compact_row_fit_record(run, fit_state, params_run)
    if compact is not None:
        if cache_key in _ROW_RECON_CACHE:
            return _ROW_RECON_CACHE[cache_key]
        try:
            matrix = reconstruct_compact_row_matrix(exp, run, fit_state, source_ids, compact)
            _ROW_RECON_CACHE[cache_key] = matrix
            return matrix
        except Exception:
            pass

    persistent = _find_persistent_row_cache(exp, run, cache_key, params_run)
    if persistent is not None:
        matrix = np.asarray(persistent.intensity_2d, dtype=float)
        _ROW_RECON_CACHE[cache_key] = matrix
        _link_row_fit_cache(run, persistent, cache_key, params_run)
        return matrix

    if cache_key in _ROW_RECON_CACHE:
        return _ROW_RECON_CACHE[cache_key]

    # Row-by-row fitting is expensive and must not run on a drawing path.
    return None


def overlay_data(exp: Optional[ExperimentSet], run: Run, mode: Any) -> Optional[FitOverlayData]:
    mode = normalize_fit_overlay_mode(mode)
    if mode == "off" or run is None or run.run_type == RunType.FIT_PARAMS or not run.is_2d:
        return None

    fit_state, params_run, source_ids = find_fit_state(exp, run)
    peaks = fit_overlay_peaks(fit_state)
    centers = np.asarray([peak.center_cm1 for peak in peaks], dtype=float)
    global_matrix = _global_matrix(exp, run, fit_state, source_ids) if mode in {"global", "both"} else None
    row_matrix = _row_matrix(exp, run, fit_state, source_ids, params_run=params_run) if mode in {"row", "both"} else None

    if centers.size == 0 and global_matrix is None and row_matrix is None:
        return None

    source = params_run.id if params_run is not None else ("run_metadata" if fit_state else "reconstruction")
    return FitOverlayData(
        centers_cm1=centers,
        peaks=peaks,
        global_matrix=global_matrix,
        row_matrix=row_matrix,
        source=source,
    )
