"""Shared description of the non-spectral axis of a merged Raman map.

``Run.angle_values`` remains the storage coordinate for backwards-compatible
HDF5 files, but a merge can now describe those values as temperature,
polarization, magnetic field, or an acquisition index instead of an angle.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Optional, Sequence

import numpy as np


@dataclass(frozen=True)
class RowAxisSpec:
    """Display contract for a map's row axis."""

    kind: str = "angle"
    unit: str = "deg"
    label: str = "Angle (deg)"
    major_step: Optional[float] = 90.0
    tick_positions: Optional[np.ndarray] = None
    tick_labels: Optional[tuple[str, ...]] = None

    @property
    def is_angle(self) -> bool:
        return self.kind == "angle"

    @property
    def is_categorical(self) -> bool:
        return self.tick_positions is not None and self.tick_labels is not None

    @property
    def uses_polar_plot(self) -> bool:
        return self.is_angle and not self.is_categorical

    def format_value(self, value: float, row_index: Optional[int] = None) -> str:
        if self.is_categorical and row_index is not None and self.tick_labels is not None:
            if 0 <= row_index < len(self.tick_labels):
                return self.tick_labels[row_index]
        if self.kind == "angle":
            return f"{float(value):.1f} deg"
        if self.unit:
            return f"{float(value):.3g} {self.unit}"
        return f"{float(value):.3g}"


def row_axis_from_metadata(
    metadata: Optional[Mapping[str, Any]],
    values: Optional[Sequence[float]],
) -> RowAxisSpec:
    """Return a robust display axis for legacy and newly merged runs.

    Categorical labels are intentionally accepted only when their count is
    identical to the stored numeric row coordinates.  That prevents malformed
    metadata from silently assigning labels to the wrong spectra.
    """

    md = dict(metadata or {})
    kind = str(md.get("merge_axis_kind") or md.get("raw_y_unit") or "angle").strip().lower()
    unit = str(md.get("merge_axis_unit") or "").strip()
    vals = np.asarray(values if values is not None else [], dtype=float)
    labels_raw = md.get("merge_row_labels")
    if not labels_raw and kind == "polarization":
        labels_raw = md.get("polarization_rows") or md.get("merge_unique_xxxx")
    labels = tuple(str(v) for v in labels_raw) if isinstance(labels_raw, (list, tuple)) else ()

    if kind in {"polarization", "categorical"}:
        if vals.size and len(labels) == vals.size:
            return RowAxisSpec(
                kind="polarization",
                unit="polarization",
                label="Polarization",
                major_step=None,
                tick_positions=vals.copy(),
                tick_labels=labels,
            )
        # Keep a usable numeric map if legacy metadata is incomplete.
        return RowAxisSpec(kind="angle", unit="deg", label="Angle (deg)", major_step=90.0)

    if vals.size and len(labels) == vals.size and str(md.get("repeat_mode") or "").lower() == "separate":
        if kind in {"temperature", "temp"}:
            label = f"Temperature ({unit or 'K'})"
        elif kind in {"magnetic_field", "field", "magnetic"}:
            label = f"Magnetic field ({unit or 'T'})"
        elif kind == "angle":
            label = "Angle (deg)"
        else:
            label = "Acquisition index"
        return RowAxisSpec(
            kind=kind,
            unit=unit,
            label=label,
            major_step=None,
            tick_positions=vals.copy(),
            tick_labels=labels,
        )

    if kind in {"temperature", "temp"}:
        return RowAxisSpec(kind="temperature", unit=unit or "K", label=f"Temperature ({unit or 'K'})", major_step=None)
    if kind in {"magnetic_field", "field", "magnetic"}:
        return RowAxisSpec(kind="magnetic_field", unit=unit or "T", label=f"Magnetic field ({unit or 'T'})", major_step=None)
    if kind in {"index", "acquisition_index"}:
        return RowAxisSpec(kind="index", unit=unit, label="Acquisition index", major_step=None)

    return RowAxisSpec(kind="angle", unit="deg", label="Angle (deg)", major_step=90.0)
