"""Circular-polarization helpers shared by merge and GUI code."""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Iterable, Mapping, Optional, Sequence

import numpy as np

from data_structure import ExperimentSet, Run, RunType, ViewState, new_run_id


CPL_POLARIZATION_ORDER = ("RR", "RL", "LL", "LR")
CPL_OPPOSITE = {"RR": "LL", "LL": "RR", "RL": "LR", "LR": "RL"}
CPL_OVERLAY_COLOR = "#d62728"
CPL_OVERLAY_LINEWIDTH = 0.6


def _base_polarization(label: Any) -> Optional[str]:
    text = str(label or "").strip().upper()
    for polarization in CPL_POLARIZATION_ORDER:
        if text == polarization or text.startswith(f"{polarization} ") or text.startswith(f"{polarization}·"):
            return polarization
    return None


def polarization_rows(run: Run) -> tuple[str, ...]:
    metadata = run.metadata if isinstance(run.metadata, Mapping) else {}
    labels = metadata.get("merge_row_labels") or metadata.get("polarization_rows") or ()
    if not isinstance(labels, (list, tuple)):
        return ()
    return tuple(str(label) for label in labels)


def is_cpl_run(run: Optional[Run]) -> bool:
    if run is None or run.intensity_2d is None or run.angle_values is None:
        return False
    metadata = run.metadata if isinstance(run.metadata, Mapping) else {}
    if metadata.get("cpl_virtual"):
        return True
    if str(metadata.get("merge_axis_kind", "")).strip().lower() != "polarization":
        return False
    labels = polarization_rows(run)
    return bool(labels) and all(_base_polarization(label) in CPL_OPPOSITE for label in labels)


def display_angles(run: Run, stored_values: Sequence[float]) -> np.ndarray:
    """Return 90-degree display coordinates for an ordinary four-row CPL map."""
    values = np.asarray(stored_values, dtype=float)
    labels = polarization_rows(run)
    if not is_cpl_run(run) or len(labels) != values.size:
        return values
    bases = tuple(_base_polarization(label) for label in labels)
    if len(labels) == 4 and set(bases) == set(CPL_POLARIZATION_ORDER):
        return np.arange(4, dtype=float) * 90.0
    return values


def _opposite_row_permutation(labels: Sequence[str]) -> tuple[int, ...]:
    bases = tuple(_base_polarization(label) for label in labels)
    if len(labels) != 4 or set(bases) != set(CPL_POLARIZATION_ORDER):
        raise ValueError("CPL opposite-chirality generation requires one RR, RL, LL, and LR row.")
    indices = {polarization: index for index, polarization in enumerate(bases)}
    return tuple(indices[CPL_OPPOSITE[polarization]] for polarization in bases)


def create_opposite_chirality_run(source: Run) -> Run:
    """Create a persisted 2D virtual map with opposite-helicity rows exchanged."""
    if not is_cpl_run(source):
        raise ValueError(f"'{source.nickname}' is not a complete CPL map.")
    labels = polarization_rows(source)
    permutation = _opposite_row_permutation(labels)
    matrix = np.asarray(source.intensity_2d)
    if matrix.ndim != 2 or matrix.shape[0] != len(labels):
        raise ValueError("CPL row labels do not match the source intensity matrix.")

    metadata = deepcopy(source.metadata)
    metadata.update(
        {
            "nickname": f"{source.nickname} (opposite chirality)",
            "cpl_virtual": True,
            "source_kind": "cpl_opposite_chirality_virtual",
            "cpl_virtual_source_run_id": source.id,
            "cpl_opposite_mapping": dict(CPL_OPPOSITE),
            "cpl_row_permutation": list(permutation),
        }
    )
    return Run(
        id=new_run_id("cpl"),
        source_path=source.source_path,
        source_mtime=source.source_mtime,
        wl_nm=None if source.wl_nm is None else np.asarray(source.wl_nm).copy(),
        shift_cm1=None if source.shift_cm1 is None else np.asarray(source.shift_cm1).copy(),
        energy_eV=None if source.energy_eV is None else np.asarray(source.energy_eV).copy(),
        intensity=None,
        intensity_2d=matrix[np.asarray(permutation, dtype=int), :].copy(),
        angle_values=np.asarray(source.angle_values, dtype=float).copy(),
        intensity_unit=source.intensity_unit,
        angle_unit=source.angle_unit,
        metadata=metadata,
        run_type=RunType.RUN_2D,
        raw_table=None,
    )


def synchronize_opposite_chirality_run(source: Run, virtual: Run) -> None:
    """Refresh an existing virtual map from its current CPL source arrays."""
    replacement = create_opposite_chirality_run(source)
    virtual.source_path = replacement.source_path
    virtual.source_mtime = replacement.source_mtime
    virtual.wl_nm = replacement.wl_nm
    virtual.shift_cm1 = replacement.shift_cm1
    virtual.energy_eV = replacement.energy_eV
    virtual.intensity = None
    virtual.intensity_2d = replacement.intensity_2d
    virtual.angle_values = replacement.angle_values
    virtual.intensity_unit = replacement.intensity_unit
    virtual.angle_unit = replacement.angle_unit
    virtual.metadata = replacement.metadata
    virtual.run_type = RunType.RUN_2D
    virtual.raw_table = None


def find_opposite_chirality_run(experiment: ExperimentSet, source_run_id: str) -> Optional[Run]:
    for run in experiment.runs.values():
        metadata = run.metadata if isinstance(run.metadata, Mapping) else {}
        if metadata.get("cpl_virtual") and str(metadata.get("cpl_virtual_source_run_id")) == str(source_run_id):
            return run
    return None


def primary_run_ids(view_state: ViewState, experiment: ExperimentSet, limit: int = 2) -> list[str]:
    """Return primary view runs while excluding explicit overlay-only runs."""
    result: list[str] = []
    for run_id in view_state.run_ids:
        if experiment.get_run(run_id) is None:
            continue
        config = view_state.run_configs.get(run_id)
        if config is not None and config.overlay_target:
            continue
        result.append(run_id)
        if len(result) >= limit:
            break
    return result


def attach_opposite_chirality_overlays(
    experiment: ExperimentSet,
    view_state: ViewState,
    source_run_ids: Iterable[str],
) -> tuple[list[Run], list[str]]:
    """Attach or reuse CPL mirrors for primary sources in a view.

    Returns newly created virtual runs and compact warnings. Existing virtual
    runs are reused so repeatedly adding the same source remains idempotent.
    """
    created: list[Run] = []
    warnings: list[str] = []
    for source_run_id in source_run_ids:
        source = experiment.get_run(source_run_id)
        if source is None or not is_cpl_run(source) or (source.metadata or {}).get("cpl_virtual"):
            continue
        primaries = primary_run_ids(view_state, experiment)
        if source.id not in primaries:
            continue
        slot = primaries.index(source.id) + 1
        try:
            virtual = find_opposite_chirality_run(experiment, source.id)
            if virtual is None:
                virtual = create_opposite_chirality_run(source)
                experiment.add_run(virtual)
                created.append(virtual)
            else:
                synchronize_opposite_chirality_run(source, virtual)
        except ValueError as exc:
            warnings.append(f"{source.nickname}: {exc}")
            continue

        if virtual.id not in view_state.run_ids:
            view_state.run_ids.append(virtual.id)
        config = view_state.get_run_config(virtual.id)
        config.overlay_target = f"{slot}C"
        style = config.get_style("Overlay")
        style.color = CPL_OVERLAY_COLOR
        style.linewidth = CPL_OVERLAY_LINEWIDTH
        style.linestyle = "-"
        style.alpha = 0.9
        style.marker = ""
    return created, warnings
