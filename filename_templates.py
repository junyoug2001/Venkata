"""JSON-backed filename parsing and merge-family discovery.

The catalog keeps filename conventions out of the merge algorithm.  Built-ins
ship with the application; users can disable them or add templates in the
ignored user overlay without modifying project code.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
import re
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np


CATALOG_FILENAME = "merge_filename_templates.json"
USER_OVERLAY_FILENAME = "merge_filename_templates.user.json"
POLARIZATION_ORDER = ("RR", "RL", "LL", "LR")


class TemplateValidationError(ValueError):
    """Raised when a selected catalog entry cannot be used safely."""


@dataclass(frozen=True)
class MergeProfile:
    id: str
    label: str
    axis: Mapping[str, Any]
    family_fields: Tuple[str, ...]
    repeat_fields: Tuple[str, ...]


@dataclass(frozen=True)
class FilenameTemplate:
    id: str
    label: str
    priority: int
    enabled: bool
    pattern: str
    regex: re.Pattern[str]
    converters: Mapping[str, str]
    x_unit_hint: str
    examples: Tuple[str, ...]
    profiles: Tuple[MergeProfile, ...]
    source: str = "built-in"


@dataclass(frozen=True)
class FilenameMatch:
    path: str
    template: FilenameTemplate
    fields: Mapping[str, Any]

    def profile(self, profile_id: str = "auto") -> MergeProfile:
        if not self.template.profiles:
            raise TemplateValidationError(f"Template '{self.template.id}' has no merge profile.")
        if profile_id in {None, "", "auto"}:
            return self.template.profiles[0]
        for profile in self.template.profiles:
            if profile.id == profile_id:
                return profile
        raise TemplateValidationError(
            f"Template '{self.template.id}' has no profile '{profile_id}'."
        )


@dataclass(frozen=True)
class MergeRowPlan:
    """File order and row grouping derived from one template/profile."""

    files: Tuple[str, ...]
    raw_row_keys: Tuple[str, ...]
    raw_repeat_keys: Tuple[str, ...]
    primitive_keys: Tuple[str, ...]
    primitive_values: np.ndarray
    primitive_labels: Tuple[str, ...]
    axis_kind: str
    axis_unit: str
    is_categorical: bool


@dataclass
class TemplateCatalog:
    templates: List[FilenameTemplate]
    errors: List[str] = field(default_factory=list)

    def available_templates(self, include_disabled: bool = False) -> List[FilenameTemplate]:
        values = self.templates if include_disabled else [t for t in self.templates if t.enabled]
        # A local laboratory rule is an intentional override, even when its
        # author leaves the default priority unchanged.  Within each source,
        # priority remains explicit and equal priorities are reported as an
        # ambiguity by ``select``.
        return sorted(values, key=lambda t: (0 if t.source == "user" else 1, -t.priority, t.id))

    def get(self, template_id: str) -> FilenameTemplate:
        for template in self.templates:
            if template.id == template_id:
                return template
        raise TemplateValidationError(f"Unknown filename template '{template_id}'.")

    def matches(self, path: str, include_disabled: bool = False) -> List[FilenameMatch]:
        base = Path(path).name
        matches: List[FilenameMatch] = []
        for template in self.available_templates(include_disabled=include_disabled):
            found = template.regex.fullmatch(base)
            if not found:
                continue
            matches.append(FilenameMatch(str(path), template, _convert_fields(found.groupdict(), template.converters)))
        return matches

    def select(self, path: str, template_id: str = "auto") -> Optional[FilenameMatch]:
        if template_id not in {None, "", "auto"}:
            template = self.get(str(template_id))
            if not template.enabled:
                raise TemplateValidationError(f"Template '{template_id}' is disabled.")
            found = template.regex.fullmatch(Path(path).name)
            if not found:
                raise TemplateValidationError(f"'{Path(path).name}' does not match template '{template_id}'.")
            return FilenameMatch(str(path), template, _convert_fields(found.groupdict(), template.converters))
        matches = self.matches(path)
        if not matches:
            return None
        first = matches[0]
        tied = [
            m for m in matches
            if m.template.source == first.template.source and m.template.priority == first.template.priority
        ]
        if len(tied) > 1:
            ids = ", ".join(m.template.id for m in tied)
            raise TemplateValidationError(
                f"Ambiguous filename template for '{Path(path).name}': {ids}. Select one explicitly."
            )
        return first


def _project_dir() -> Path:
    return Path(__file__).resolve().parent


def builtin_catalog_path() -> Path:
    return _project_dir() / CATALOG_FILENAME


def user_overlay_path() -> Path:
    return _project_dir() / USER_OVERLAY_FILENAME


def _read_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise TemplateValidationError(f"{path.name} must contain a JSON object.")
    return payload


def _compile_template(raw: Mapping[str, Any], source: str) -> FilenameTemplate:
    template_id = str(raw.get("id") or "").strip()
    pattern = str(raw.get("pattern") or "")
    if not template_id or not pattern:
        raise TemplateValidationError("Each filename template requires id and pattern.")
    try:
        regex = re.compile(pattern)
    except re.error as exc:
        raise TemplateValidationError(f"Template '{template_id}' has invalid regex: {exc}") from exc
    x_unit_hint = str(raw.get("x_unit_hint", "auto"))
    if x_unit_hint not in {"auto", "nm", "cm-1", "meV"}:
        raise TemplateValidationError(f"Template '{template_id}' has invalid x_unit_hint '{x_unit_hint}'.")
    profiles: List[MergeProfile] = []
    for profile_raw in raw.get("merge_profiles") or []:
        if not isinstance(profile_raw, Mapping):
            raise TemplateValidationError(f"Template '{template_id}' has a non-object profile.")
        profile_id = str(profile_raw.get("id") or "").strip()
        axis = profile_raw.get("axis") or {}
        if not profile_id or not isinstance(axis, Mapping) or not axis.get("source"):
            raise TemplateValidationError(f"Template '{template_id}' has an incomplete merge profile.")
        kind = str(axis.get("kind", "numeric"))
        if kind not in {"numeric", "indexed_step", "categorical"}:
            raise TemplateValidationError(f"Template '{template_id}' profile '{profile_id}' has invalid axis kind.")
        profiles.append(
            MergeProfile(
                id=profile_id,
                label=str(profile_raw.get("label") or profile_id),
                axis=dict(axis),
                family_fields=tuple(str(v) for v in profile_raw.get("family_fields") or []),
                repeat_fields=tuple(str(v) for v in profile_raw.get("repeat_fields") or []),
            )
        )
    return FilenameTemplate(
        id=template_id,
        label=str(raw.get("label") or template_id),
        priority=int(raw.get("priority", 0)),
        enabled=bool(raw.get("enabled", True)),
        pattern=pattern,
        regex=regex,
        converters=dict(raw.get("converters") or {}),
        x_unit_hint=x_unit_hint,
        examples=tuple(str(v) for v in raw.get("examples") or []),
        profiles=tuple(profiles),
        source=source,
    )


def load_catalog(
    builtins_path: Optional[Path] = None,
    overlay_path: Optional[Path] = None,
) -> TemplateCatalog:
    """Load built-ins plus a failure-tolerant per-user overlay."""
    builtins_path = builtins_path or builtin_catalog_path()
    overlay_path = overlay_path or user_overlay_path()
    payload = _read_json(builtins_path)
    if payload.get("schema_version") != 1:
        raise TemplateValidationError("Unsupported merge filename template schema version.")
    errors: List[str] = []
    templates: List[FilenameTemplate] = []
    seen: set[str] = set()
    for raw in payload.get("templates") or []:
        template = _compile_template(raw, "built-in")
        if template.id in seen:
            raise TemplateValidationError(f"Duplicate built-in template id '{template.id}'.")
        seen.add(template.id)
        templates.append(template)

    overlay: Dict[str, Any] = {}
    if overlay_path.exists():
        try:
            overlay = _read_json(overlay_path)
        except Exception as exc:  # Keep trusted built-ins usable.
            errors.append(f"Could not load {overlay_path.name}: {exc}")
    disabled = {str(v) for v in overlay.get("disabled_builtin_ids") or []}
    if disabled:
        templates = [
            FilenameTemplate(**{**template.__dict__, "enabled": False}) if template.id in disabled else template
            for template in templates
        ]
    for raw in overlay.get("templates") or []:
        try:
            template = _compile_template(raw, "user")
            if template.id in seen:
                raise TemplateValidationError(f"Duplicate template id '{template.id}'.")
            seen.add(template.id)
            templates.append(template)
        except Exception as exc:
            errors.append(str(exc))
    return TemplateCatalog(templates=templates, errors=errors)


def read_user_overlay(path: Optional[Path] = None) -> Dict[str, Any]:
    path = path or user_overlay_path()
    if not path.exists():
        return {"disabled_builtin_ids": [], "templates": []}
    payload = _read_json(path)
    payload.setdefault("disabled_builtin_ids", [])
    payload.setdefault("templates", [])
    return payload


def write_user_overlay(payload: Mapping[str, Any], path: Optional[Path] = None) -> None:
    path = path or user_overlay_path()
    clean = {
        "disabled_builtin_ids": list(payload.get("disabled_builtin_ids") or []),
        "templates": list(payload.get("templates") or []),
    }
    # Validate before replacing an existing user file.
    test_path = path.with_suffix(path.suffix + ".validation")
    try:
        test_path.write_text(json.dumps(clean), encoding="utf-8")
        load_catalog(overlay_path=test_path)
    finally:
        if test_path.exists():
            test_path.unlink()
    path.write_text(json.dumps(clean, indent=2) + "\n", encoding="utf-8")


def _convert_fields(raw: Mapping[str, Optional[str]], converters: Mapping[str, str]) -> Dict[str, Any]:
    result: Dict[str, Any] = {key: value for key, value in raw.items() if value is not None}
    for key, converter in converters.items():
        if key not in result:
            continue
        value = str(result[key])
        if converter == "int":
            result[key] = int(value)
        elif converter == "float":
            result[key] = float(value.replace(",", "."))
        elif converter == "unit_float":
            number = re.search(r"[-+]?\d+(?:[,.]\d+)?", value)
            result[key] = float(number.group(0).replace(",", ".")) if number else float("nan")
    # Custom templates may capture a signed field without committing the
    # shipped catalog to a particular filename convention.  Normalizing here
    # gives merge profiles a stable ``magnetic_field_T`` source.
    if "magnetic_field_value" in result:
        try:
            value = float(str(result["magnetic_field_value"]).replace(",", "."))
            unit = str(result.get("magnetic_field_unit", "T")).strip().lower()
            factor = {"t": 1.0, "mt": 1e-3, "g": 1e-4, "gauss": 1e-4}.get(unit)
            if factor is not None:
                result["magnetic_field_T"] = value * factor
        except (TypeError, ValueError):
            pass
    return result


def filename_metadata(path: str, template_id: str = "auto") -> Dict[str, Any]:
    """Return canonical and legacy-compatible metadata from a catalog match."""
    match = load_catalog().select(path, template_id=template_id)
    if match is None:
        return {}
    fields = dict(match.fields)
    fields["filename_template_id"] = match.template.id
    fields["filename_template_label"] = match.template.label
    fields["filename_x_unit_hint"] = match.template.x_unit_hint
    if "polarization" in fields:
        fields["pol"] = fields["polarization"]
    if "temperature_K" in fields:
        fields["temp_K"] = fields["temperature_K"]
        fields["temp_str"] = f"{fields['temperature_K']:g}K"
    if "laser_nm" in fields:
        fields["laser_nm_str"] = f"{fields['laser_nm']:g}nm"
    if "power_mW" in fields:
        fields["power"] = fields["power_mW"]
    if "row_index" in fields:
        fields["angle_idx"] = str(fields["row_index"]).zfill(4)
    if "rep_idx" in fields:
        fields["rep_idx"] = str(fields["rep_idx"]).zfill(4)
    fields["has_angular"] = "angle_step_deg" in fields and "row_index" in fields
    return fields


def _axis_value(match: FilenameMatch, profile: MergeProfile) -> Tuple[Any, str]:
    axis = profile.axis
    source = str(axis["source"])
    if source not in match.fields:
        raise TemplateValidationError(f"'{Path(match.path).name}' is missing axis field '{source}'.")
    kind = str(axis.get("kind", "numeric"))
    raw = match.fields[source]
    if kind == "categorical":
        return str(raw), str(raw)
    if kind == "numeric":
        value = float(raw)
        return value, f"{value:g} {axis.get('unit', '')}".strip()
    origin = float(axis.get("index_origin", 1))
    if axis.get("step_source"):
        step = float(match.fields[str(axis["step_source"])])
    elif axis.get("step_from_filename"):
        found = re.search(r"(-?\d+(?:\.\d+)?)deg", Path(match.path).name)
        step = abs(float(found.group(1))) if found else 1.0
    else:
        step = 1.0
    value = (float(raw) - origin) * step
    return value, f"{value:g} {axis.get('unit', '')}".strip()


def _key(fields: Mapping[str, Any], names: Sequence[str]) -> Tuple[Any, ...]:
    return tuple(fields.get(name) for name in names)


def _repeat_display(fields: Mapping[str, Any], names: Sequence[str]) -> str:
    """Human-friendly repeat text without exposing export bookkeeping."""
    values: List[str] = []
    for name in names:
        value = fields.get(name)
        if value is None or name == "export_index":
            continue
        if name == "scan_repeat":
            values.append(f"{int(value):02d}")
        elif name in {"rep_idx", "row_index"}:
            values.append(f"{int(value):04d}")
        else:
            values.append(str(value))
    return " · ".join(values) or "run"


def discover_family(
    seed_file: str,
    *,
    template_id: str = "auto",
    profile_id: str = "auto",
    repeat_mode: str = "average",
) -> Tuple[FilenameMatch, MergeProfile, List[FilenameMatch], MergeRowPlan]:
    """Find the files compatible with the selected template/profile."""
    if repeat_mode not in {"average", "separate"}:
        raise TemplateValidationError("repeat_mode must be 'average' or 'separate'.")
    catalog = load_catalog()
    seed = catalog.select(seed_file, template_id=template_id)
    if seed is None:
        raise TemplateValidationError(f"No filename template matches '{Path(seed_file).name}'.")
    profile = seed.profile(profile_id)
    family = _key(seed.fields, profile.family_fields)
    matches: List[FilenameMatch] = []
    for entry in Path(seed_file).resolve().parent.iterdir():
        if not entry.is_file():
            continue
        try:
            matched = catalog.select(str(entry), template_id=seed.template.id)
        except TemplateValidationError:
            continue
        if matched and _key(matched.fields, profile.family_fields) == family:
            matches.append(matched)
    if not matches:
        matches = [seed]

    axis_info: List[Tuple[FilenameMatch, Any, str, Tuple[Any, ...], str]] = []
    for matched in matches:
        value, label = _axis_value(matched, profile)
        repeat = _key(matched.fields, profile.repeat_fields)
        axis_info.append((matched, value, label, repeat, _repeat_display(matched.fields, profile.repeat_fields)))
    axis = profile.axis
    kind = str(axis.get("kind", "numeric"))
    order = [str(v) for v in axis.get("order") or POLARIZATION_ORDER]

    def sort_key(item):
        _match, value, _label, repeat, _repeat_label = item
        if kind == "categorical":
            text = str(value)
            return (order.index(text) if text in order else len(order), text, repeat, Path(_match.path).name)
        return (float(value), repeat, Path(_match.path).name)

    axis_info.sort(key=sort_key)
    files = tuple(item[0].path for item in axis_info)
    raw_row_keys = tuple(str(item[1]) for item in axis_info)
    raw_repeat_keys = tuple("|".join(str(v) for v in item[3] if v is not None) for item in axis_info)
    grouped: Dict[str, Tuple[Any, str, str]] = {}
    for (_match, value, label, repeat, repeat_label), raw_key, repeat_key in zip(axis_info, raw_row_keys, raw_repeat_keys):
        primitive_key = raw_key if repeat_mode == "average" else f"{raw_key}|{repeat_key}"
        display_label = label if repeat_mode == "average" else f"{label} · {repeat_label}"
        grouped.setdefault(primitive_key, (value, display_label, repeat_key))
    primitive_keys = tuple(grouped.keys())
    if repeat_mode == "separate":
        primitive_values = np.arange(len(primitive_keys), dtype=float)
        categorical = True
    elif kind == "categorical":
        primitive_values = np.asarray(
            [order.index(str(grouped[key][0])) if str(grouped[key][0]) in order else idx for idx, key in enumerate(primitive_keys)],
            dtype=float,
        )
        if str(axis.get("unit") or "").strip().lower() == "polarization":
            primitive_values *= 90.0
        categorical = True
    else:
        primitive_values = np.asarray([float(grouped[key][0]) for key in primitive_keys], dtype=float)
        categorical = False
    primitive_labels = tuple(grouped[key][1] for key in primitive_keys)
    return seed, profile, [item[0] for item in axis_info], MergeRowPlan(
        files=files,
        raw_row_keys=raw_row_keys,
        raw_repeat_keys=raw_repeat_keys,
        primitive_keys=primitive_keys,
        primitive_values=primitive_values,
        primitive_labels=primitive_labels,
        axis_kind="polarization" if kind == "categorical" else ("angle" if axis.get("unit") == "deg" else str(profile.id)),
        axis_unit=str(axis.get("unit") or ""),
        is_categorical=categorical,
    )
