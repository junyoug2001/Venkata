from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, Mapping, Optional, Sequence

import h5py


class FastHDF5Unsupported(RuntimeError):
    """Raised when a .dat file does not expose the metadata attrs used here."""


def _decode_attr(value: Any) -> Any:
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return value


def _json_attr(attrs: Mapping[str, Any], key: str, default: Any) -> Any:
    if key not in attrs:
        return default
    raw = _decode_attr(attrs[key])
    if raw in (None, ""):
        return default
    try:
        return json.loads(raw)
    except (TypeError, json.JSONDecodeError) as exc:
        raise FastHDF5Unsupported(f"Invalid JSON metadata attr '{key}'.") from exc


def _nickname_from_metadata(run_id: str, metadata: Mapping[str, Any]) -> str:
    name = metadata.get("nickname")
    if isinstance(name, str) and name.strip():
        return name.strip()
    pieces = []
    for key in ("sample", "pol", "temp_str"):
        value = metadata.get(key)
        if value:
            pieces.append(str(value))
    if pieces:
        return "_".join(pieces)
    sample = metadata.get("sample")
    if isinstance(sample, str) and sample.strip():
        return sample.strip()
    return str(run_id)


def _run_header_from_group(run_id: str, group: h5py.Group) -> Dict[str, Any]:
    attrs = group.attrs
    metadata = _json_attr(attrs, "metadata_json", {})
    if not isinstance(metadata, dict):
        metadata = {}
    datasets = {
        name: {"shape": tuple(ds.shape), "dtype": str(ds.dtype)}
        for name, ds in group.items()
        if isinstance(ds, h5py.Dataset)
    }
    run_type = _decode_attr(attrs.get("run_type", "other"))
    if run_type == "other":
        if metadata.get("derived"):
            run_type = "derived run"
        elif "intensity_2d" in datasets and "angle_values" in datasets:
            run_type = "2d run"
        elif "intensity" in datasets:
            run_type = "1d run"
    source_path = _decode_attr(attrs.get("source_path", ""))
    intensity_unit = _decode_attr(attrs.get("intensity_unit", "au"))
    angle_unit = _decode_attr(attrs.get("angle_unit", "deg"))
    return {
        "id": str(run_id),
        "nickname": _nickname_from_metadata(run_id, metadata),
        "source_path": "" if source_path is None else str(source_path),
        "source_mtime": float(attrs.get("source_mtime", 0.0) or 0.0),
        "intensity_unit": str(intensity_unit or "au"),
        "angle_unit": str(angle_unit or "deg"),
        "run_type": str(run_type),
        "metadata": metadata,
        "datasets": datasets,
    }


def read_experiment_metadata(path: str | Path) -> Dict[str, Any]:
    """
    Read ExperimentSet metadata, view JSON, and run attrs without dataset reads.

    The returned dictionaries are intentionally plain JSON-like objects so CLI
    commands can use them without importing the full GUI/data model stack.
    """
    path = str(path)
    with h5py.File(path, "r") as h5:
        if "metadata" not in h5 or "runs" not in h5:
            raise FastHDF5Unsupported("Missing expected HDF5 groups: metadata/runs.")

        exp_id = _decode_attr(h5.attrs.get("experiment_id", ""))
        metadata_group = h5["metadata"]
        experiment_metadata = _json_attr(metadata_group.attrs, "json", {})
        views_json = {}
        if "views" in h5 and "json" in h5["views"].attrs:
            views_json = _json_attr(h5["views"].attrs, "json", {})
        elif "views" in h5:
            raise FastHDF5Unsupported("Views group is missing the JSON attr.")

        runs = {
            str(run_id): _run_header_from_group(str(run_id), group)
            for run_id, group in h5["runs"].items()
            if isinstance(group, h5py.Group)
        }
        return {
            "path": path,
            "experiment_id": "" if exp_id is None else str(exp_id),
            "metadata": experiment_metadata if isinstance(experiment_metadata, dict) else {},
            "next_run_index": int(metadata_group.attrs.get("next_run_index", 1)),
            "views": views_json if isinstance(views_json, dict) else {},
            "runs": runs,
        }


def iter_run_headers(path: str | Path) -> Iterator[Dict[str, Any]]:
    meta = read_experiment_metadata(path)
    yield from meta["runs"].values()


def iter_fit_params(path: str | Path) -> Iterator[Dict[str, Any]]:
    for header in iter_run_headers(path):
        if header.get("run_type") != "fit parameters":
            continue
        metadata = header.get("metadata") or {}
        state = metadata.get("fit_state") or metadata.get("map_fit_state")
        if state:
            item = dict(header)
            item["fit_state"] = state
            yield item


def read_run_arrays(path: str | Path, run_id: str, arrays: Sequence[str]) -> Dict[str, Any]:
    """Read only explicitly requested datasets for one run."""
    out: Dict[str, Any] = {}
    with h5py.File(str(path), "r") as h5:
        try:
            run_group = h5["runs"][str(run_id)]
        except KeyError as exc:
            raise KeyError(f"Run id '{run_id}' not found.") from exc
        for name in arrays:
            if name in run_group:
                out[name] = run_group[name][:]
            else:
                out[name] = None
    return out
