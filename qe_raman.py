"""Quantum ESPRESSO ``ph.x`` Raman import helpers.

This module deliberately has no wxPython dependency.  It parses the final
Gamma-point diagonalization from a ``ph.x`` text output, optionally adapts a
Venkata Fit Parameters state into relative XX/YX peak heights, and creates two
native one-dimensional :class:`data_structure.Run` objects.

The ``ph.x`` text output does not contain the mode-projected Raman tensors
needed for exact polarization intensities.  Spectra made here are therefore
frequency-based reconstructions.  Exact QE intensities remain a separate
``fildyn``/``dynmat.x`` workflow.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import math
import os
import re
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
from scipy.optimize import linear_sum_assignment

import analysis
from data_structure import Run, RunType, new_run_id, normalize_spectral_unit, shift_to_eV, unit_to_cm1


DEFAULT_FWHM_CM1 = 4.0
DEFAULT_STEP_CM1 = 0.25
DEFAULT_MATCH_TOLERANCE_CM1 = 15.0
QE_RAMAN_SCHEMA_VERSION = 1

_FLOAT = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[EeDd][-+]?\d+)?"
_VERSION_RE = re.compile(r"Program\s+PHONON\s+v\.\s*([^\s]+)", re.IGNORECASE)
_ATOM_RE = re.compile(r"number\s+of\s+atoms/cell\s*=\s*(\d+)", re.IGNORECASE)
_DIAG_RE = re.compile(r"Diagonalizing\s+the\s+dynamical\s+matrix", re.IGNORECASE)
_Q_RE = re.compile(
    rf"q\s*=\s*\(\s*({_FLOAT})\s+({_FLOAT})\s+({_FLOAT})\s*\)",
    re.IGNORECASE,
)
_MODE_RE = re.compile(
    rf"freq\s*\(\s*(\d+)\s*\)\s*=\s*({_FLOAT})\s*\[THz\]\s*=\s*({_FLOAT})\s*\[cm-1\]",
    re.IGNORECASE,
)
_SYMMETRY_HEADER_RE = re.compile(
    r"Mode\s+symmetry\s*,\s*(.*?)\s+point\s+group\s*:",
    re.IGNORECASE,
)
_SYMMETRY_ROW_RE = re.compile(
    rf"^[ \t]*freq[ \t]*\([ \t]*(\d+)[ \t]*-[ \t]*(\d+)[ \t]*\)[ \t]*=[ \t]*({_FLOAT})[ \t]*\[cm-1\][ \t]*-->[ \t]*(\S+)(?:[ \t]+([^\r\n]*?))?[ \t]*$",
    re.IGNORECASE | re.MULTILINE,
)


class QeRamanError(ValueError):
    """Raised when a QE output cannot produce a safe Raman reconstruction."""


def _number(value: str) -> float:
    return float(str(value).replace("D", "E").replace("d", "e"))


def _json_float(value: Any) -> Optional[float]:
    try:
        converted = float(value)
    except (TypeError, ValueError):
        return None
    return converted if math.isfinite(converted) else None


def _param_value(value: Any) -> Optional[float]:
    if isinstance(value, (list, tuple, np.ndarray)):
        if len(value) == 0:
            return None
        value = value[0]
    return _json_float(value)


def _raman_activity(activity: str) -> bool:
    tokens = [token.strip().upper() for token in str(activity or "").replace(" ", "").split("+")]
    return "R" in tokens


@dataclass(frozen=True)
class QeMode:
    mode_number: int
    frequency_thz: float
    frequency_cm1: float


@dataclass(frozen=True)
class QeModeGroup:
    mode_start: int
    mode_end: int
    frequencies_cm1: Tuple[float, ...]
    center_cm1: float
    degeneracy: int
    symmetry: str = ""
    activity: str = ""
    raman_active: bool = False


@dataclass
class QePhResult:
    source_path: str
    qe_version: str
    job_done: bool
    atom_count: int
    q_point: Tuple[float, float, float]
    point_group: str
    modes: List[QeMode]
    mode_groups: List[QeModeGroup]
    activity_labels_present: bool
    warnings: List[str] = field(default_factory=list)

    @property
    def mode_count(self) -> int:
        return len(self.modes)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "source_path": self.source_path,
            "qe_version": self.qe_version,
            "job_done": bool(self.job_done),
            "atom_count": int(self.atom_count),
            "q_point": [float(value) for value in self.q_point],
            "point_group": self.point_group,
            "mode_count": self.mode_count,
            "modes": [asdict(mode) for mode in self.modes],
            "mode_groups": [
                {
                    **asdict(group),
                    "frequencies_cm1": list(group.frequencies_cm1),
                }
                for group in self.mode_groups
            ],
            "activity_labels_present": bool(self.activity_labels_present),
            "warnings": list(self.warnings),
        }


def parse_qe_ph_text(text: str, source_path: str = "") -> QePhResult:
    """Parse the final complete Gamma-point diagonalization in QE ``ph.x`` text."""

    if not isinstance(text, str) or not text.strip():
        raise QeRamanError("The selected file is empty.")
    version_matches = list(_VERSION_RE.finditer(text))
    if not version_matches:
        raise QeRamanError("This does not look like a Quantum ESPRESSO ph.x output file.")

    diagonalizations = list(_DIAG_RE.finditer(text))
    if not diagonalizations:
        raise QeRamanError("No dynamical-matrix diagonalization was found in the QE output.")
    final_start = diagonalizations[-1].start()
    tail = text[final_start:]

    q_match = _Q_RE.search(tail)
    if q_match is None:
        raise QeRamanError("The final diagonalization does not identify its q point.")
    q_point = tuple(_number(value) for value in q_match.groups())
    if any(abs(value) > 1e-7 for value in q_point):
        formatted = ", ".join(f"{value:.7g}" for value in q_point)
        raise QeRamanError(f"Only Gamma-point results are supported; the final q point is ({formatted}).")

    atom_matches = list(_ATOM_RE.finditer(text[:final_start]))
    if not atom_matches:
        raise QeRamanError("The QE output does not report the number of atoms per cell.")
    atom_count = int(atom_matches[-1].group(1))
    expected_mode_count = 3 * atom_count

    modes = [
        QeMode(int(match.group(1)), _number(match.group(2)), _number(match.group(3)))
        for match in _MODE_RE.finditer(tail)
    ]
    expected_numbers = list(range(1, expected_mode_count + 1))
    actual_numbers = [mode.mode_number for mode in modes]
    if actual_numbers != expected_numbers:
        raise QeRamanError(
            "The final QE frequency block is incomplete: "
            f"expected modes 1-{expected_mode_count}, found {len(modes)} complete frequency rows."
        )

    symmetry_header = _SYMMETRY_HEADER_RE.search(tail)
    point_group = ""
    groups: List[QeModeGroup] = []
    warnings: List[str] = []
    if symmetry_header is not None:
        point_group = " ".join(symmetry_header.group(1).split())
        symmetry_text = tail[symmetry_header.end():]
        mode_by_number = {mode.mode_number: mode for mode in modes}
        for match in _SYMMETRY_ROW_RE.finditer(symmetry_text):
            start = int(match.group(1))
            end = int(match.group(2))
            if start < 1 or end < start or end > expected_mode_count:
                continue
            grouped_modes = [mode_by_number[index] for index in range(start, end + 1) if index in mode_by_number]
            if len(grouped_modes) != end - start + 1:
                continue
            frequencies = tuple(mode.frequency_cm1 for mode in grouped_modes)
            activity = str(match.group(5) or "").strip().split()[0] if str(match.group(5) or "").strip() else ""
            groups.append(
                QeModeGroup(
                    mode_start=start,
                    mode_end=end,
                    frequencies_cm1=frequencies,
                    center_cm1=float(np.mean(frequencies)),
                    degeneracy=end - start + 1,
                    symmetry=str(match.group(4) or "").strip(),
                    activity=activity,
                    raman_active=_raman_activity(activity),
                )
            )

    if groups:
        covered = [number for group in groups for number in range(group.mode_start, group.mode_end + 1)]
        if covered != expected_numbers:
            warnings.append(
                "The final symmetry table does not cover every mode; uncovered positive modes are omitted."
            )
    else:
        groups = [
            QeModeGroup(
                mode_start=mode.mode_number,
                mode_end=mode.mode_number,
                frequencies_cm1=(mode.frequency_cm1,),
                center_cm1=mode.frequency_cm1,
                degeneracy=1,
                raman_active=True,
            )
            for mode in modes
        ]
        warnings.append(
            "The final QE block has no usable symmetry/activity table; all positive modes are included."
        )

    activity_labels_present = any(bool(group.activity) for group in groups)
    if not activity_labels_present:
        groups = [
            QeModeGroup(**{**asdict(group), "raman_active": True})
            for group in groups
        ]
        if not any("activity" in warning for warning in warnings):
            warnings.append("Raman/IR activity labels are absent; all positive modes are included.")
    elif any(not group.activity for group in groups):
        warnings.append("Some symmetry rows have no Raman/IR activity label and are not treated as Raman-active.")

    job_done = bool(re.search(r"\bJOB\s+DONE\.?", tail, re.IGNORECASE))
    if not job_done:
        warnings.append("QE did not print JOB DONE, but the final Gamma-point frequency block is complete.")

    return QePhResult(
        source_path=os.path.abspath(source_path) if source_path else "",
        qe_version=version_matches[-1].group(1),
        job_done=job_done,
        atom_count=atom_count,
        q_point=q_point,
        point_group=point_group,
        modes=modes,
        mode_groups=groups,
        activity_labels_present=activity_labels_present,
        warnings=warnings,
    )


def parse_qe_ph_output(path: str) -> QePhResult:
    """Read and parse a QE ``ph.x`` output file."""

    path = os.path.abspath(os.path.expanduser(path))
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as handle:
            text = handle.read()
    except OSError as exc:
        raise QeRamanError(f"Could not read QE output: {exc}") from exc
    return parse_qe_ph_text(text, source_path=path)


def usable_mode_groups(result: QePhResult) -> List[QeModeGroup]:
    """Return positive Raman-active groups, or the documented label-free fallback."""

    groups = [group for group in result.mode_groups if group.center_cm1 > 0 and group.raman_active]
    if not groups:
        raise QeRamanError("No positive usable Raman modes remain after activity and frequency filtering.")
    return groups


def fit_peaks_from_state(fit_state: Optional[Mapping[str, Any]]) -> Tuple[List[Dict[str, Any]], List[str]]:
    """Convert persisted Fit Parameters peaks to a compact cm^-1 representation."""

    if not fit_state:
        return [], []
    raw_unit = str(fit_state.get("unit", "") or "").strip()
    unit = normalize_spectral_unit(raw_unit or "meV")
    warnings: List[str] = []
    normalized_tokens = {"mev", "cm-1", "cm^-1", "cm1", "cm⁻¹"}
    if raw_unit.lower() not in normalized_tokens:
        warnings.append(
            f"Fit Parameters unit '{raw_unit or '(missing)'}' is not recognized; affected mappings use fallback values."
        )

    converted: List[Dict[str, Any]] = []
    for index, peak in enumerate(fit_state.get("peaks") or []):
        spec_params = peak.get("spec_params") or {}
        x0_raw = _param_value(spec_params.get("x0"))
        gamma_raw = _param_value(spec_params.get("gamma"))
        x0_cm1 = None
        gamma_cm1 = None
        if x0_raw is not None:
            x0_cm1 = _json_float(np.asarray(unit_to_cm1(x0_raw, unit)).reshape(-1)[0])
        if gamma_raw is not None:
            gamma_cm1 = _json_float(np.asarray(unit_to_cm1(gamma_raw, unit)).reshape(-1)[0])
        rule = str(peak.get("rule") or "")
        ang_params = peak.get("ang_params") or {}
        rule_definition = analysis.RULE_METADATA.get(rule)
        parameter_values: Dict[str, Optional[float]] = {}
        if rule_definition is not None:
            parameter_values = {
                name: _param_value(ang_params.get(name))
                for name in rule_definition["params"]
            }
        problems: List[str] = []
        if raw_unit.lower() not in normalized_tokens:
            problems.append("invalid spectral unit")
        if x0_cm1 is None:
            problems.append("invalid fitted center")
        if gamma_cm1 is None or gamma_cm1 <= 0:
            problems.append("invalid fitted Gamma")
        if rule_definition is None:
            problems.append(f"unknown tensor rule '{rule or '(missing)'}'")
        elif any(value is None for value in parameter_values.values()):
            problems.append("invalid tensor parameters")
        name = str(peak.get("name") or f"Peak {index + 1}")
        converted.append(
            {
                "fit_index": index,
                "name": name,
                "display_name": f"{index + 1}: {name}",
                "center_cm1": x0_cm1,
                "gamma_cm1": gamma_cm1,
                "rule": rule,
                "ang_params": parameter_values,
                "valid": not problems,
                "warning": "; ".join(problems),
            }
        )
    return converted, warnings


def auto_match_groups(
    groups: Sequence[QeModeGroup],
    fit_peaks: Sequence[Mapping[str, Any]],
    tolerance_cm1: float = DEFAULT_MATCH_TOLERANCE_CM1,
) -> Dict[int, int]:
    """Deterministically minimize total one-to-one center-frequency distance."""

    tolerance_cm1 = float(tolerance_cm1)
    if not math.isfinite(tolerance_cm1) or tolerance_cm1 < 0:
        raise QeRamanError("The Fit Parameters matching tolerance must be a finite non-negative value.")
    valid_fit_indices = [
        index for index, peak in enumerate(fit_peaks)
        if _json_float(peak.get("center_cm1")) is not None
    ]
    if not groups or not valid_fit_indices:
        return {}
    centers = np.asarray([group.center_cm1 for group in groups], dtype=float)
    fit_centers = np.asarray([float(fit_peaks[index]["center_cm1"]) for index in valid_fit_indices], dtype=float)
    costs = np.abs(centers[:, None] - fit_centers[None, :])
    # Stable, tiny tie breakers preserve the physical cost while making equal
    # assignments deterministic across SciPy versions.
    costs = costs + np.arange(costs.shape[0], dtype=float)[:, None] * 1e-12
    costs = costs + np.arange(costs.shape[1], dtype=float)[None, :] * 1e-15
    row_indices, column_indices = linear_sum_assignment(costs)
    mapping: Dict[int, int] = {}
    for row_index, column_index in zip(row_indices.tolist(), column_indices.tolist()):
        fit_index = valid_fit_indices[column_index]
        distance = abs(float(groups[row_index].center_cm1) - float(fit_peaks[fit_index]["center_cm1"]))
        if distance <= tolerance_cm1:
            mapping[row_index] = fit_index
    return mapping


def validate_mapping(mapping: Mapping[int, Optional[int]]) -> None:
    """Reject a manual table mapping that reuses the same fitted peak."""

    assigned = [int(value) for value in mapping.values() if value is not None and int(value) >= 0]
    duplicates = sorted({value for value in assigned if assigned.count(value) > 1})
    if duplicates:
        labels = ", ".join(str(value + 1) for value in duplicates)
        raise QeRamanError(f"A fitted peak can be assigned only once; duplicate Fit Parameters peak(s): {labels}.")


def adapt_peak_rows(
    groups: Sequence[QeModeGroup],
    fit_state: Optional[Mapping[str, Any]] = None,
    *,
    angle_deg: float = 0.0,
    default_fwhm_cm1: float = DEFAULT_FWHM_CM1,
    match_tolerance_cm1: float = DEFAULT_MATCH_TOLERANCE_CM1,
    mapping: Optional[Mapping[int, Optional[int]]] = None,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[str]]:
    """Build editable peak-table rows from QE groups and optional Fit Parameters."""

    angle_deg = float(angle_deg)
    default_fwhm_cm1 = float(default_fwhm_cm1)
    if not math.isfinite(angle_deg):
        raise QeRamanError("The sample rotation must be finite.")
    if not math.isfinite(default_fwhm_cm1) or default_fwhm_cm1 <= 0:
        raise QeRamanError("The default FWHM must be a finite positive value.")

    fit_peaks, warnings = fit_peaks_from_state(fit_state)
    if mapping is None:
        resolved_mapping: Dict[int, int] = auto_match_groups(groups, fit_peaks, match_tolerance_cm1)
    else:
        resolved_mapping = {
            int(row): int(fit_index)
            for row, fit_index in mapping.items()
            if fit_index is not None and int(fit_index) >= 0
        }
        validate_mapping(resolved_mapping)

    reconstructed: Dict[int, Tuple[float, float, float]] = {}
    invalid_mapping_messages: List[str] = []
    for row_index, fit_index in sorted(resolved_mapping.items()):
        if row_index < 0 or row_index >= len(groups) or fit_index < 0 or fit_index >= len(fit_peaks):
            invalid_mapping_messages.append(f"Row {row_index + 1} has an out-of-range Fit Parameters mapping.")
            continue
        fit_peak = fit_peaks[fit_index]
        if not fit_peak.get("valid"):
            invalid_mapping_messages.append(
                f"{fit_peak['display_name']} is invalid ({fit_peak.get('warning')}); the matched QE peak uses fallback values."
            )
            continue
        definition = analysis.RULE_METADATA[fit_peak["rule"]]
        parameters = [fit_peak["ang_params"][name] for name in definition["params"]]
        try:
            theta = np.asarray([angle_deg], dtype=float)
            xx_rule = float(np.asarray(definition["func"](theta, "parallel", *parameters), dtype=float)[0])
            yx_rule = float(np.asarray(definition["func"](theta, "cross", *parameters), dtype=float)[0])
            gamma_cm1 = float(fit_peak["gamma_cm1"])
            xx_height = xx_rule / (math.pi * gamma_cm1)
            yx_height = yx_rule / (math.pi * gamma_cm1)
            if not all(math.isfinite(value) and value >= 0 for value in (xx_height, yx_height, gamma_cm1)):
                raise ValueError("non-finite reconstructed height")
            reconstructed[row_index] = (xx_height, yx_height, 2.0 * gamma_cm1)
        except Exception as exc:
            invalid_mapping_messages.append(
                f"{fit_peak['display_name']} could not be evaluated ({exc}); the matched QE peak uses fallback values."
            )

    if reconstructed:
        common_scale = max(max(values[0], values[1]) for values in reconstructed.values())
        if not math.isfinite(common_scale) or common_scale <= 0:
            invalid_mapping_messages.append(
                "Mapped tensor rules reconstruct zero intensity in both polarizations; fallback values are used."
            )
            reconstructed.clear()
        else:
            reconstructed = {
                row_index: (values[0] / common_scale, values[1] / common_scale, values[2])
                for row_index, values in reconstructed.items()
            }

    rows: List[Dict[str, Any]] = []
    for row_index, group in enumerate(groups):
        fit_index = resolved_mapping.get(row_index)
        fit_peak = fit_peaks[fit_index] if fit_index is not None and 0 <= fit_index < len(fit_peaks) else None
        adapted = reconstructed.get(row_index)
        rows.append(
            {
                "include": True,
                "group_index": row_index,
                "mode_start": group.mode_start,
                "mode_end": group.mode_end,
                "mode_range": str(group.mode_start) if group.mode_start == group.mode_end else f"{group.mode_start}-{group.mode_end}",
                "frequencies_cm1": [float(value) for value in group.frequencies_cm1],
                "dft_frequency_cm1": float(group.center_cm1),
                "degeneracy": int(group.degeneracy),
                "symmetry": group.symmetry,
                "activity": group.activity,
                "matched_fit_index": fit_index,
                "matched_fit_name": fit_peak.get("name", "") if fit_peak else "",
                "matched_fit_frequency_cm1": fit_peak.get("center_cm1") if fit_peak else None,
                "match_distance_cm1": (
                    abs(float(group.center_cm1) - float(fit_peak["center_cm1"]))
                    if fit_peak and fit_peak.get("center_cm1") is not None
                    else None
                ),
                "xx_height": float(adapted[0]) if adapted else 1.0,
                "yx_height": float(adapted[1]) if adapted else 1.0,
                "fwhm_cm1": float(adapted[2]) if adapted else default_fwhm_cm1,
                "value_source": "fitparams" if adapted else "uniform_fallback",
            }
        )

    if fit_state and not resolved_mapping:
        warnings.append(
            f"No Fit Parameters peaks match a QE group within {float(match_tolerance_cm1):g} cm-1; uniform fallback values are used."
        )
    warnings.extend(invalid_mapping_messages)
    return rows, fit_peaks, warnings


def validate_peak_rows(rows: Sequence[Mapping[str, Any]], fit_peak_count: Optional[int] = None) -> None:
    """Validate the user-editable mapping and spectrum fields before creation."""

    mapping: Dict[int, int] = {}
    included = 0
    for index, row in enumerate(rows):
        if not bool(row.get("include", True)):
            continue
        included += 1
        center = _json_float(row.get("dft_frequency_cm1"))
        xx = _json_float(row.get("xx_height"))
        yx = _json_float(row.get("yx_height"))
        fwhm = _json_float(row.get("fwhm_cm1"))
        if center is None or center <= 0:
            raise QeRamanError(f"Peak row {index + 1} has an invalid positive DFT frequency.")
        if xx is None or xx < 0 or yx is None or yx < 0:
            raise QeRamanError(f"Peak row {index + 1} must have finite non-negative XX/YX heights.")
        if fwhm is None or fwhm <= 0:
            raise QeRamanError(f"Peak row {index + 1} must have a finite positive FWHM.")
        fit_index = row.get("matched_fit_index")
        if fit_index is not None and int(fit_index) >= 0:
            fit_index = int(fit_index)
            if fit_peak_count is not None and fit_index >= fit_peak_count:
                raise QeRamanError(f"Peak row {index + 1} refers to a missing Fit Parameters peak.")
            mapping[index] = fit_index
    if included == 0:
        raise QeRamanError("At least one positive QE peak must be included.")
    validate_mapping(mapping)


def build_spectra(
    rows: Sequence[Mapping[str, Any]],
    *,
    x_min_cm1: Optional[float] = None,
    x_max_cm1: Optional[float] = None,
    step_cm1: float = DEFAULT_STEP_CM1,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, float]]:
    """Generate pair-normalized XX/YX spectra from height-normalized Lorentzians."""

    validate_peak_rows(rows)
    included = [row for row in rows if bool(row.get("include", True))]
    step_cm1 = float(step_cm1)
    if not math.isfinite(step_cm1) or step_cm1 <= 0:
        raise QeRamanError("The spectral grid spacing must be a finite positive value.")
    auto_min = min(float(row["dft_frequency_cm1"]) - 5.0 * float(row["fwhm_cm1"]) for row in included)
    auto_max = max(float(row["dft_frequency_cm1"]) + 5.0 * float(row["fwhm_cm1"]) for row in included)
    x_min = auto_min if x_min_cm1 is None else float(x_min_cm1)
    x_max = auto_max if x_max_cm1 is None else float(x_max_cm1)
    if not math.isfinite(x_min) or not math.isfinite(x_max) or x_max <= x_min:
        raise QeRamanError("The spectral range must have finite limits with maximum greater than minimum.")
    intervals = int(math.ceil((x_max - x_min) / step_cm1))
    x = x_min + np.arange(intervals + 1, dtype=float) * step_cm1
    xx = np.zeros_like(x)
    yx = np.zeros_like(x)
    for row in included:
        center = float(row["dft_frequency_cm1"])
        gamma = float(row["fwhm_cm1"]) / 2.0
        profile = (gamma * gamma) / ((x - center) ** 2 + gamma * gamma)
        xx += float(row["xx_height"]) * profile
        yx += float(row["yx_height"]) * profile
    common_scale = float(max(np.nanmax(xx), np.nanmax(yx)))
    if math.isfinite(common_scale) and common_scale > 0:
        xx = xx / common_scale
        yx = yx / common_scale
    else:
        raise QeRamanError("The selected peak heights produce an empty spectrum.")
    settings = {
        "x_min_cm1": float(x[0]),
        "x_max_cm1": float(x[-1]),
        "requested_x_min_cm1": float(x_min),
        "requested_x_max_cm1": float(x_max),
        "step_cm1": step_cm1,
        "normalization_scale": common_scale,
    }
    return x, xx, yx, settings


def make_qe_raman_runs(
    result: QePhResult,
    nickname_prefix: str,
    rows: Sequence[Mapping[str, Any]],
    *,
    angle_deg: float = 0.0,
    match_tolerance_cm1: float = DEFAULT_MATCH_TOLERANCE_CM1,
    default_fwhm_cm1: float = DEFAULT_FWHM_CM1,
    x_min_cm1: Optional[float] = None,
    x_max_cm1: Optional[float] = None,
    step_cm1: float = DEFAULT_STEP_CM1,
    fit_params_run_id: Optional[str] = None,
    warnings: Optional[Iterable[str]] = None,
) -> Tuple[Run, Run]:
    """Create native XX/YX RUN_1D objects with complete regeneration metadata."""

    nickname_prefix = str(nickname_prefix or "").strip()
    if not nickname_prefix:
        raise QeRamanError("A nickname prefix is required.")
    validate_peak_rows(rows)
    x, xx, yx, spectral_settings = build_spectra(
        rows,
        x_min_cm1=x_min_cm1,
        x_max_cm1=x_max_cm1,
        step_cm1=step_cm1,
    )
    source_path = os.path.abspath(result.source_path) if result.source_path else ""
    try:
        source_mtime = os.path.getmtime(source_path) if source_path else None
    except OSError:
        source_mtime = None
    final_rows = [dict(row) for row in rows]
    intensity_source = "fitparams" if any(row.get("value_source") == "fitparams" for row in final_rows) else "uniform"
    all_warnings = list(result.warnings)
    all_warnings.extend(str(value) for value in (warnings or []) if str(value))
    disclaimer = (
        "Frequency-only QE reconstruction; XX/YX values are relative display intensities "
        "and are not ab-initio QE polarization intensities."
    )
    pair_id = new_run_id(prefix="qe_pair")
    common_metadata = {
        "raw_dim": "1d",
        "raw_x_unit": "cm-1",
        "raw_y_unit": "relative intensity",
        "source_kind": "qe_ph_output",
        "specialized_generated_run": True,
        "qe_raman_generated": True,
        "qe_raman_schema_version": QE_RAMAN_SCHEMA_VERSION,
        "qe_raman_pair_id": pair_id,
        "frequency_only": True,
        "polarization_intensities_are_ab_initio": False,
        "intensity_source": intensity_source,
        "fit_params_run_id": fit_params_run_id,
        "sample_rotation_deg": float(angle_deg),
        "match_tolerance_cm1": float(match_tolerance_cm1),
        "default_fwhm_cm1": float(default_fwhm_cm1),
        "source_path": source_path,
        "source_mtime": source_mtime,
        "qe_parser_result": result.to_dict(),
        "qe_peak_mappings": final_rows,
        "qe_spectrum_settings": spectral_settings,
        "qe_warnings": all_warnings,
        "qe_intensity_disclaimer": disclaimer,
        "regeneration_workflow": "File -> Import Quantum ESPRESSO Raman...",
    }

    def _run(polarization: str, intensity: np.ndarray) -> Run:
        metadata = dict(common_metadata)
        metadata.update(
            {
                "nickname": f"{nickname_prefix}_DFT_{polarization}",
                "sample": nickname_prefix,
                "pol": polarization,
                "polarization": polarization,
            }
        )
        return Run(
            id=new_run_id(prefix=f"qe_{polarization}"),
            source_path=source_path,
            source_mtime=source_mtime,
            shift_cm1=np.asarray(x, dtype=float),
            energy_eV=shift_to_eV(x),
            intensity=np.asarray(intensity, dtype=float),
            intensity_unit="au",
            metadata=metadata,
            run_type=RunType.RUN_1D,
        )

    return _run("xx", xx), _run("yx", yx)
