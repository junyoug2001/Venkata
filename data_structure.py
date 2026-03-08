

"""
Core data model and basic I/O utilities for the Raman GUI.

This module is intentionally GUI-agnostic:

- Defines core data classes (Run / ViewState / ExperimentSet).
- Provides basic filename parsing and table loading.
- Implements simple unit conversions (nm → wavenumber, Raman shift cm^-1 / eV).
- Contains no wxPython or plotting code.
"""

# -------------------------
# Units and simple constants
# -------------------------

#: conversion factor: 1 nm = 1e-7 cm
CM_PER_NM: float = 1e-7

#: 1 cm^-1 = 1 / 8065.544 eV
EV_PER_CM1: float = 1.0 / 8065.544

from dataclasses import dataclass, field, asdict
from typing import Optional, List, Dict, Any, Tuple, Union
from enum import Enum, auto

import os
import re

import numpy as np
import pandas as pd


# -------------------------
# Utility helpers
# -------------------------

# Filename pattern adapted from the legacy raman_plotter_5.py script.
# This encodes the standard naming convention used in your lab, but
# keeps all parsing logic self-contained in this file.
FNAME_RE = re.compile(
    r'^'
    r'(?P<category>\d{8})_'
    r'(?P<sample>[^_]+)_'
    r'(?P<inttime>\d+sx\d+)_'
    r'(?P<grating>\d+g)_'
    r'(?P<slit>\d+mu)_'
    r'(?:(?P<cm_val>\d+cm-1)_)?'  # Optional field like 350cm-1
    r'(?P<laser_nm>\d+(?:,\d+)?nm)_'
    r'(?P<power>[^_]+)_'
    r'(?P<pol>(xx|xy|yx|yy|RL|LR|LL|RR))_'
    r'(?P<temp>\d+K)'
    r'(?P<angular>-8deg_(?P<angle_idx>\d{4})_(?P<rep_idx>\d{4}))?'
    r'(?P<tail>.*?)$'
)


def parse_filename(path: str) -> Dict[str, Any]:
    """
    Parse a Raman data filename into a structured metadata dictionary.

    This is adapted from the legacy raman_plotter_5.py logic and
    reflects the standard naming convention used in your files:

        YYYYMMDD_sample_inttime_grating_slit_laser_nm_power_pol_temp[-8deg_idx_rep]tail.ext

    If the filename does not match this pattern, a dictionary with
    all fields set to None (or sensible defaults) is returned.

    Parameters
    ----------
    path : str
        Full path to the data file.

    Returns
    -------
    Dict[str, Any]
        Dictionary with keys:
        - "base": original basename (with extension)
        - "stem": basename without extension
        - "parts": underscore-separated tokens from the stem
        - "category", "sample", "inttime", "grating", "slit"
        - "laser_nm_str", "laser_nm" (float, if parsable)
        - "power", "pol"
        - "temp_str", "temp_K" (float, if parsable)
        - "has_angular", "angle_idx", "rep_idx", "tail"
    """
    base = os.path.basename(path)
    stem, _ = os.path.splitext(base)
    parts = stem.split("_")

    info: Dict[str, Any] = {
        "base": base,
        "stem": stem,
        "parts": parts,
        "category": None,
        "sample": None,
        "inttime": None,
        "grating": None,
        "slit": None,
        "cm_val": None,
        "laser_nm_str": None,
        "laser_nm": None,
        "power": None,
        "pol": None,
        "temp_str": None,
        "temp_K": None,
        "has_angular": False,
        "angle_idx": None,
        "rep_idx": None,
        "tail": None,
    }

    m = FNAME_RE.match(base)
    if not m:
        # Filename does not follow the strict pattern; return best-effort info.
        # We still keep stem/parts filled for manual inspection.
        return info

    gd = m.groupdict()

    info["category"] = gd.get("category")
    info["sample"] = gd.get("sample")
    info["inttime"] = gd.get("inttime")
    info["grating"] = gd.get("grating")
    info["slit"] = gd.get("slit")
    info["cm_val"] = gd.get("cm_val")
    info["laser_nm_str"] = gd.get("laser_nm")
    info["power"] = gd.get("power")
    info["pol"] = gd.get("pol")
    info["temp_str"] = gd.get("temp")
    info["has_angular"] = gd.get("angular") is not None
    info["angle_idx"] = gd.get("angle_idx")
    info["rep_idx"] = gd.get("rep_idx")
    info["tail"] = gd.get("tail")

    # Numeric conversions
    temp = gd.get("temp")
    if temp and temp.endswith("K"):
        try:
            info["temp_K"] = float(temp[:-1])
        except ValueError:
            pass

    laser_field = gd.get("laser_nm")
    if laser_field:
        # e.g. '532nm' or '532,0nm'
        nm_str = laser_field.replace("nm", "").replace(",", ".")
        try:
            info["laser_nm"] = float(nm_str)
        except ValueError:
            pass

    return info


def load_table(path: str) -> Tuple[Union[Tuple[np.ndarray, np.ndarray],
                                   Tuple[np.ndarray, np.ndarray, np.ndarray]],
                                   Optional[List[str]]]:
    """
    Load a simple text / table file and return numeric arrays instead of a DataFrame.

    The behavior is intentionally minimal and aimed at your current data:

    - 1D data (spectra):
        * Assumed to have exactly two numeric columns.
        * Returns: ((x, y), header_labels)
          where x is x-axis array (e.g. wavelength or shift),
                y is intensity array.
          header_labels is a list of column names if detected, else None.

    - 2D data (e.g. angular_matrix_meV.csv):
        * First column: y-axis values (e.g. angle in degrees).
        * Remaining columns: intensity matrix, column-wise.
        * Column labels (except the first) are treated as x-axis values when
          they can be parsed as floats; otherwise, a simple index grid is used.
        * Returns: ((x_axis, y_axis, intensity_matrix), None)

    This function does not try to infer physical units; that is the job of
    the caller (e.g. converting meV → eV → cm^-1).
    """
    ext = os.path.splitext(path)[1].lower()

    if ext in {".txt", ".dat"}:
        df = pd.read_csv(path, sep=r"\s+", comment="#", header=None)
    elif ext == ".csv":
        df = pd.read_csv(path, comment="#")
    elif ext == ".tsv":
        df = pd.read_csv(path, sep="\t", comment="#")
    elif ext in {".xlsx", ".xls"}:
        df = pd.read_excel(path)
    else:
        # Fallback: let pandas try to infer
        df = pd.read_csv(path, comment="#")

    # Drop completely empty columns
    df = df.dropna(axis=1, how="all")

    ncols = df.shape[1]
    if ncols < 2:
        raise ValueError(f"File {path} must have at least 2 columns, got {ncols}.")

    labels = None

    # 1D: exactly two columns → (x, y)
    if ncols == 2:
        # Check if current columns are meaningful (not just 0, 1 integers)
        is_default_cols = False
        try:
            # Check if columns are RangeIndex or similar integers
            if list(df.columns) == [0, 1]:
                is_default_cols = True
        except:
            pass

        if not is_default_cols:
            # Columns might be headers
            # Check if they look like strings
            if all(isinstance(c, str) for c in df.columns) and not str(df.columns[0]).isdigit():
                labels = list(df.columns)
        
        # If columns were default integers (likely header=None), check the first row content
        if labels is None:
            # Check if first row is non-numeric
            first_row_is_numeric = False
            try:
                pd.to_numeric(df.iloc[0], errors='raise')
                first_row_is_numeric = True
            except ValueError:
                pass
            
            if not first_row_is_numeric:
                # Promote first row to header
                labels = df.iloc[0].astype(str).tolist()
                df = df.iloc[1:]
                # Convert data to numeric
                df = df.apply(pd.to_numeric, errors='coerce')
                df = df.dropna(how='any')

        x = df.iloc[:, 0].to_numpy(dtype=float)
        y = df.iloc[:, 1].to_numpy(dtype=float)
        return (x, y), labels

    # 2D: first column = y-axis, remaining columns = intensity (x-grid in header)
    y_axis = df.iloc[:, 0].to_numpy(dtype=float)
    data = df.iloc[:, 1:]

    # Try to parse column labels as floats for x-axis
    try:
        x_axis = np.array([float(str(c)) for c in data.columns], dtype=float)
    except Exception:
        x_axis = np.arange(data.shape[1], dtype=float)

    intensity = data.to_numpy(dtype=float)

    return (x_axis, y_axis, intensity), None


def nm_to_wavenumber(nm: np.ndarray) -> np.ndarray:
    """
    Convert wavelength in nm to absolute wavenumber in cm^-1.

    Note: this is *not* the Raman shift; it is simply 1 / lambda in cm.
    Raman shift requires a reference laser wavelength.
    """
    nm = np.asarray(nm, dtype=float)
    lam_cm = nm * CM_PER_NM
    return 1.0 / lam_cm


def raman_shift_cm1(laser_nm: float, wl_nm: np.ndarray) -> np.ndarray:
    """
    Compute Raman shift (in cm^-1) given a laser wavelength (nm)
    and an array of scattered wavelengths (nm).

    shift = 1/lambda_laser(cm) - 1/lambda_scattered(cm)
    """
    wl_nm = np.asarray(wl_nm, dtype=float)
    lam0_cm = laser_nm * CM_PER_NM
    lam_cm = wl_nm * CM_PER_NM
    return (1.0 / lam0_cm) - (1.0 / lam_cm)


def shift_to_eV(shift_cm: np.ndarray) -> np.ndarray:
    """
    Convert Raman shift in cm^-1 to energy in eV.
    """
    shift_cm = np.asarray(shift_cm, dtype=float)
    return shift_cm * EV_PER_CM1


SUPPORTED_SPECTRAL_UNITS = ("meV", "cm-1")


def normalize_spectral_unit(unit: Optional[str], default: str = "meV") -> str:
    """Return a supported public spectral unit name."""
    value = str(unit or default or "meV").strip().lower()
    if value in {"mev", "milliev", "millielectronvolt", "millielectronvolts"}:
        return "meV"
    if value in {"cm-1", "cm^-1", "cm⁻¹", "cm1", "wavenumber", "wavenumbers"}:
        return "cm-1"
    if "mev" in value:
        return "meV"
    if "cm-1" in value or "cm^-1" in value or "cm⁻¹" in value or "cm1" in value or ("cm" in value and "-1" in value):
        return "cm-1"
    return normalize_spectral_unit(default, "meV") if value else "meV"


def cm1_to_mev(value):
    """Convert Raman shift from cm^-1 to meV."""
    return np.asarray(value, dtype=float) * EV_PER_CM1 * 1000.0


def mev_to_cm1(value):
    """Convert Raman shift from meV to cm^-1."""
    return np.asarray(value, dtype=float) / (EV_PER_CM1 * 1000.0)


def cm1_to_unit(value, unit: str):
    unit = normalize_spectral_unit(unit)
    if unit == "meV":
        return cm1_to_mev(value)
    return np.asarray(value, dtype=float)


def unit_to_cm1(value, unit: str):
    unit = normalize_spectral_unit(unit)
    if unit == "meV":
        return mev_to_cm1(value)
    return np.asarray(value, dtype=float)


def spectral_xlim_from_cm1(xlim: Optional[Tuple[float, float]], unit: str) -> Optional[Tuple[float, float]]:
    if xlim is None:
        return None
    converted = cm1_to_unit(np.asarray(xlim, dtype=float), unit)
    return (float(converted[0]), float(converted[1]))


def spectral_xlim_to_cm1(xlim: Optional[Tuple[float, float]], unit: str) -> Optional[Tuple[float, float]]:
    if xlim is None:
        return None
    converted = unit_to_cm1(np.asarray(xlim, dtype=float), unit)
    return (float(converted[0]), float(converted[1]))


def alternate_spectral_unit(unit: str) -> str:
    return "cm-1" if normalize_spectral_unit(unit) == "meV" else "meV"


def spectral_axis_label(unit: str, *, latex: bool = True) -> str:
    unit = normalize_spectral_unit(unit)
    if unit == "meV":
        return "Raman shift (meV)"
    return "Raman shift (cm$^{-1}$)" if latex else "Raman shift (cm-1)"


def spectral_axis_for_run(run, unit: str) -> Optional[np.ndarray]:
    unit = normalize_spectral_unit(unit)
    if run is None:
        return None
    if unit == "meV":
        if getattr(run, "energy_eV", None) is not None:
            return np.asarray(run.energy_eV, dtype=float) * 1000.0
        if getattr(run, "shift_cm1", None) is not None:
            return cm1_to_mev(run.shift_cm1)
    if getattr(run, "shift_cm1", None) is not None:
        return np.asarray(run.shift_cm1, dtype=float)
    if getattr(run, "energy_eV", None) is not None:
        return mev_to_cm1(np.asarray(run.energy_eV, dtype=float) * 1000.0)
    return None


def infer_intensity_unit(y: np.ndarray) -> str:
    """
    Decide whether an array of intensities is effectively integer-like
    (=> 'count') or not (=> 'au').

    - Integer dtypes are treated as 'count'.
    - Float dtypes are checked against rounding with a small tolerance.
    """
    y = np.asarray(y)
    if y.size == 0:
        return "au"

    if np.issubdtype(y.dtype, np.integer):
        return "count"

    y_float = y.astype(float)
    rounded = np.rint(y_float)
    if np.allclose(y_float, rounded, rtol=0.0, atol=1e-9):
        return "count"
    return "au"


# -------------------------
# Core data classes
# -------------------------


class RunType(str, Enum):
    RUN_1D = "1d run"
    RUN_2D = "2d run"
    DERIVED = "derived run"
    FIT_PARAMS = "fit parameters"
    OTHER = "other"

@dataclass
class PlotStyle:
    """Style attributes for a single plot component."""
    visible: bool = True
    color: str = "black"
    linestyle: str = "-"  # '-', '--', '-.', ':'
    linewidth: float = 1.0
    marker: str = ""
    markersize: float = 5.0
    alpha: float = 1.0

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PlotStyle":
        return cls(**data)

@dataclass
class RunViewConfig:
    """Configuration for how a Run is displayed in a View."""
    # Keyed by component name: e.g. "Map", "SliceH", "SliceV", or "A", "B", "C"
    # For now, let's use "A", "B", "C" to match the ViewPanel layout.
    styles: Dict[str, PlotStyle] = field(default_factory=dict)
    overlay_target: Optional[str] = None  # e.g. "1B", "1C"
    
    # Derived run configuration overrides
    derived_n_points: Optional[int] = None
    derived_autorange: Optional[bool] = None
    derived_range: Optional[Tuple[float, float]] = None

    def get_style(self, component: str) -> PlotStyle:
        if component not in self.styles:
            self.styles[component] = PlotStyle()
        return self.styles[component]

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "RunViewConfig":
        styles_raw = data.get("styles", {})
        styles = {k: PlotStyle.from_dict(v) for k, v in styles_raw.items()}
        overlay_target = data.get("overlay_target")
        derived_n_points = data.get("derived_n_points")
        derived_autorange = data.get("derived_autorange")
        derived_range = data.get("derived_range")
        if derived_range: derived_range = tuple(derived_range)
        
        return cls(styles=styles, overlay_target=overlay_target,
                   derived_n_points=derived_n_points,
                   derived_autorange=derived_autorange,
                   derived_range=derived_range)

@dataclass
class GraphViewConfig:
    """Persisted axis and color settings for one visible graph panel."""
    xlim: Optional[Tuple[float, float]] = None
    ylim: Optional[Tuple[float, float]] = None
    vmin: Optional[float] = None
    vmax: Optional[float] = None
    cmap: Optional[str] = None
    clim: Optional[Tuple[float, float]] = None

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "GraphViewConfig":
        if data is None:
            return cls()

        def _tuple_or_none(value):
            if value is None:
                return None
            if isinstance(value, (list, tuple)) and len(value) == 2:
                try:
                    return (float(value[0]), float(value[1]))
                except (TypeError, ValueError):
                    return None
            return None

        return cls(
            xlim=_tuple_or_none(data.get("xlim")),
            ylim=_tuple_or_none(data.get("ylim")),
            vmin=None if data.get("vmin") is None else float(data.get("vmin")),
            vmax=None if data.get("vmax") is None else float(data.get("vmax")),
            cmap=data.get("cmap"),
            clim=_tuple_or_none(data.get("clim")),
        )

@dataclass
class Run:
    """
    One Raman dataset.
    ...
    """

    # Identity / origin
    id: str
    source_path: str
    source_mtime: Optional[float] = None  # os.path.getmtime, if known

    # Axes
    wl_nm: Optional[np.ndarray] = None        # raw wavelength axis (nm)
    shift_cm1: Optional[np.ndarray] = None    # Raman shift in cm^-1
    energy_eV: Optional[np.ndarray] = None    # Raman shift in eV

    # Data
    intensity: Optional[np.ndarray] = None        # 1D spectrum
    intensity_2d: Optional[np.ndarray] = None     # 2D map (e.g. angle × shift)
    angle_values: Optional[np.ndarray] = None     # angle axis for 2D data, in degrees

    # Units and additional metadata
    intensity_unit: str = "au"    # 'count' or 'au'
    angle_unit: str = "deg"       # 'deg' or 'rad'
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    # Run Type
    run_type: RunType = RunType.OTHER

    # Optional copy of the raw loaded table
    raw_table: Optional[pd.DataFrame] = None


    # --- label helpers ---

    @property
    def nickname(self) -> str:
        """
        Human-friendly name for this run.

        The primary storage is metadata["nickname"]. If this key is missing
        or empty, we attempt to construct it from metadata in 'sample-pol-unit' order.
        If that fails, we fall back to:
        - metadata["sample"] if available, otherwise
        - the internal id.

        This keeps the core data model independent from any particular GUI
        while still providing a stable, human-readable label.
        """
        md = self.metadata or {}
        name = md.get("nickname")
        if isinstance(name, str) and name.strip():
            return name.strip()

        # Attempt to construct from sample-pol-unit
        sample = md.get("sample")
        pol = md.get("pol")
        # 'unit' might refer to temperature or the energy unit (cm-1/meV)
        # We try temp_str first, then check if x_unit is in metadata
        unit = md.get("temp_str") or md.get("raw_x_unit")
        
        parts = []
        if sample: parts.append(str(sample))
        if pol: parts.append(str(pol))
        if unit: parts.append(str(unit))
        
        if parts:
            return "_".join(parts)

        if isinstance(sample, str) and sample.strip():
            return sample.strip()

        return self.id

    @nickname.setter
    def nickname(self, value: Optional[str]) -> None:
        """
        Set the human-friendly nickname for this run.

        The value is stored in metadata["nickname"] as a stripped string.
        """
        if not isinstance(self.metadata, dict):
            self.metadata = {}
        
        if value is None:
            self.metadata["nickname"] = None
        else:
            self.metadata["nickname"] = str(value).strip()

    # --- convenience ---

    @property
    def is_2d(self) -> bool:
        """Return True if this Run stores 2D data."""
        return self.intensity_2d is not None and self.angle_values is not None

    @property
    def n_points(self) -> int:
        """Number of points in the primary 1D axis, if available."""
        if self.shift_cm1 is not None:
            return int(self.shift_cm1.size)
        if self.wl_nm is not None:
            return int(self.wl_nm.size)
        if self.intensity is not None:
            return int(self.intensity.size)
        return 0
    
    def evaluate(self, x: np.ndarray) -> np.ndarray:
        """
        Evaluate the run at given x values.
        For RUN_1D, this might imply interpolation (not implemented yet).
        For DERIVED, this evaluates the stored formula.
        """
        if self.run_type == RunType.DERIVED:
            formula = self.metadata.get("formula")
            params = self.metadata.get("formula_params", {})
            if formula:
                # Safe evaluation environment
                allowed_locals = {"x": x, "np": np}
                allowed_locals.update(params)
                try:
                    return eval(formula, {"__builtins__": {}}, allowed_locals)
                except Exception as e:
                    # Fallback or error logging?
                    # For now return zeros of same shape
                    print(f"Error evaluating formula '{formula}': {e}")
                    return np.zeros_like(x)
        
        # Fallback for non-derived: return intensity if shapes match?
        # Interpolation logic would go here.
        return np.zeros_like(x)

    def to_dict_summary(self) -> Dict[str, Any]:
        """
        Lightweight JSON-friendly summary (no full arrays).

        Useful for experiment summaries and quick inspection without
        serializing all numeric data.
        """
        return {
            "id": self.id,
            "source_path": self.source_path,
            "source_mtime": self.source_mtime,
            "intensity_unit": self.intensity_unit,
            "angle_unit": self.angle_unit,
            "shapes": {
                "wl_nm": None if self.wl_nm is None else list(self.wl_nm.shape),
                "shift_cm1": None if self.shift_cm1 is None else list(self.shift_cm1.shape),
                "energy_eV": None if self.energy_eV is None else list(self.energy_eV.shape),
                "intensity": None if self.intensity is None else list(self.intensity.shape),
                "intensity_2d": None if self.intensity_2d is None else list(self.intensity_2d.shape),
                "angle_values": None if self.angle_values is None else list(self.angle_values.shape),
            },
            "metadata_keys": list(self.metadata.keys()),
        }

    def export_csv(self, base_filepath: Optional[str] = None, output_dir: Optional[str] = None):
        """
        Export the Run data to CSV files.
        For 2D runs: exports two files (cm-1 and meV).
        For 1D runs: exports one file with the best available X-axis.
        Runs carrying map-fit parameter text also export a companion .dat file.

        If `base_filepath` is provided, it is used. Otherwise, a filename is
        generated from metadata and saved in `output_dir` (if provided) or
        the run's source directory.
        """
        if self.run_type == RunType.FIT_PARAMS and self.metadata.get("fit_parameters_text"):
            self._export_fit_parameters_text(base_filepath, output_dir)
            return

        if self.is_2d:
            self._export_csv_2d(base_filepath, output_dir)
        else:
            self.export_csv_1d(base_filepath, output_dir)

        if self.metadata.get("fit_parameters_text"):
            self._export_fit_parameters_text(base_filepath, output_dir)

    def _export_fit_parameters_text(self, base_filepath: Optional[str] = None, output_dir: Optional[str] = None) -> str:
        base_filepath = self._resolve_export_path(base_filepath, output_dir)
        base, _ext = os.path.splitext(base_filepath)
        path = f"{base}_fit_params.dat"
        text = str(self.metadata.get("fit_parameters_text", ""))
        source_ids = self.metadata.get("source_run_ids") or self.metadata.get("fit_params_source_run_ids")
        with open(path, "w", encoding="utf-8") as f:
            f.write("# Venkata map fitting parameters\n")
            f.write(f"# Run: {self.nickname} ({self.id})\n")
            if source_ids:
                f.write(f"# Source run ids: {', '.join(str(v) for v in source_ids)}\n")
            f.write(text.rstrip())
            f.write("\n")
        return path

    def _export_csv_2d(self, base_filepath: Optional[str] = None, output_dir: Optional[str] = None):
        """
        Internal helper to export 2D runs (cm-1 and meV matrices).
        """
        if self.shift_cm1 is None or self.angle_values is None or self.intensity_2d is None:
            raise ValueError("Run is missing data for 2D CSV export.")

        base_filepath = self._resolve_export_path(base_filepath, output_dir)

        # --- cm-1 export ---
        df_cm1 = pd.DataFrame(
            data=self.intensity_2d,
            index=self.angle_values,
            columns=self.shift_cm1,
        )
        df_cm1.index.name = "angle_deg"
        df_cm1.columns.name = "Raman shift (cm-1)"
        
        base, ext = os.path.splitext(base_filepath)
        if not ext:
            ext = ".csv"
        cm1_path = f"{base}_cm-1{ext}"
        
        df_cm1.to_csv(cm1_path)

        # --- meV export ---
        if self.energy_eV is not None:
            energy_mev = self.energy_eV * 1000.0
            df_mev = pd.DataFrame(
                data=self.intensity_2d,
                index=self.angle_values,
                columns=energy_mev,
            )
            df_mev.index.name = "angle_deg"
            df_mev.columns.name = "Energy (meV)"
            
            mev_path = f"{base}_meV{ext}"
            df_mev.to_csv(mev_path)

    def export_csv_1d(self, base_filepath: Optional[str] = None, output_dir: Optional[str] = None):
        """
        Export 1D Run data to a CSV file with a header.
        """
        if self.intensity is None:
            # Maybe it's a derived run that needs evaluation?
            # For now, if no intensity array, we can't export static CSV easily unless we evaluate it.
            # But derived runs often don't have stored intensity.
            # If it's derived, we might need to generate x values.
            if self.run_type == RunType.DERIVED:
                # Generate dummy X or use default range
                x_range = self.metadata.get("default_range", (0, 100))
                n_points = self.metadata.get("default_n_points", 100)
                x = np.linspace(x_range[0], x_range[1], n_points)
                y = self.evaluate(x)
                x_label = "x"
                y_label = "y"
            else:
                raise ValueError("Run is missing intensity data for 1D CSV export.")
        else:
            y = self.intensity
            y_label = "Intensity"
            
            if self.shift_cm1 is not None and len(self.shift_cm1) == len(y):
                x = self.shift_cm1
                x_label = "Raman Shift (cm-1)"
            elif self.wl_nm is not None and len(self.wl_nm) == len(y):
                x = self.wl_nm
                x_label = "Wavelength (nm)"
            elif self.energy_eV is not None and len(self.energy_eV) == len(y):
                x = self.energy_eV
                x_label = "Energy (eV)"
            elif self.angle_values is not None and len(self.angle_values) == len(y):
                x = self.angle_values
                x_label = "Angle (deg)"
            else:
                x = np.arange(len(y))
                x_label = "Index"

        base_filepath = self._resolve_export_path(base_filepath, output_dir)
        
        df = pd.DataFrame({x_label: x, y_label: y})
        df.to_csv(base_filepath, index=False)

    def _resolve_export_path(self, base_filepath: Optional[str], output_dir: Optional[str]) -> str:
        """Helper to determine the output filename/path."""
        if base_filepath:
            return base_filepath
            
        # Determine the directory to save in
        if output_dir:
            save_dir = output_dir
        else:
            save_dir = os.path.dirname(self.source_path) if self.source_path else '.'
        
        # Rebuild filename from individual metadata components
        parts = []
        # Use nickname if available and safe, or construct from metadata
        if self.metadata.get("nickname"):
             # Sanitize nickname for filename
             safe_nick = "".join([c for c in self.metadata["nickname"] if c.isalnum() or c in (' ', '-', '_')]).strip()
             parts.append(safe_nick.replace(" ", "_"))
        else:
            for key in ["category", "sample", "inttime", "grating", "slit", "cm_val", "laser_nm_str", "power", "pol", "temp_str"]:
                val = self.metadata.get(key)
                if val:
                    parts.append(str(val))
            
            if self.metadata.get("has_angular"):
                parts.append("angular_matrix")
            
            tail = self.metadata.get("tail")
            if tail and not self.metadata.get("has_angular"):
                parts.append(str(tail))

        base_name = "_".join(parts)
        if not base_name:
            base_name = f"Run_{self.id}"
        
        if 'merged_files' in self.metadata:
            base_name += "_merged"
        
        # Ensure it ends with .csv if no extension provided in the end
        if not base_name.lower().endswith(".csv"):
            base_name += ".csv"
            
        return os.path.join(save_dir, base_name)


    def reload_data(self) -> None:
        """
        Reload data from the source file, updating arrays and metadata.
        Preserves the current ID and nickname.
        """
        if not self.source_path or not os.path.exists(self.source_path):
            raise FileNotFoundError(f"Source file not found: {self.source_path}")

        # Load fresh instance
        fresh = Run.from_file(self.source_path)

        # Update attributes
        self.source_mtime = fresh.source_mtime
        self.wl_nm = fresh.wl_nm
        self.shift_cm1 = fresh.shift_cm1
        self.energy_eV = fresh.energy_eV
        self.intensity = fresh.intensity
        self.intensity_2d = fresh.intensity_2d
        self.angle_values = fresh.angle_values
        self.intensity_unit = fresh.intensity_unit
        self.angle_unit = fresh.angle_unit
        
        # Merge metadata (preserve manual nickname)
        current_nickname = self.metadata.get("nickname")
        self.metadata.update(fresh.metadata)
        if current_nickname:
            self.metadata["nickname"] = current_nickname

    # --- construction helpers ---

    @classmethod
    def from_formula(cls, formula: str, params: Dict[str, float] = None, 
                     n_points: int = 100, x_range: Tuple[float, float] = (0, 100),
                     autorange: bool = False,
                     nickname: str = "Formula Run",
                     x_unit: str = "x") -> "Run":
        """
        Create a Derived Run defined by a formula.
        """
        metadata = {
            "nickname": nickname,
            "raw_dim": "1d",
            "derived": True,
            "formula": formula,
            "formula_params": params or {},
            "default_n_points": n_points,
            "default_autorange": autorange,
            "default_range": x_range,
            "raw_x_unit": x_unit,
        }
        
        return cls(
            id=new_run_id(prefix="derived"),
            source_path="",
            source_mtime=None,
            intensity=None, # Calculated on fly
            intensity_2d=None,
            angle_values=None,
            intensity_unit="au",
            angle_unit="deg",
            metadata=metadata,
            run_type=RunType.DERIVED,
            raw_table=None,
        )

    @classmethod
    def from_arrays(cls, x: np.ndarray, y: np.ndarray, x_label: str = "x", y_label: str = "y", nickname: str = "Derived Run") -> "Run":
        """
        Create a 1D Run from x and y arrays.
        Useful for creating derived runs (e.g. slices).
        """
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)
        
        metadata = {
            "nickname": nickname,
            "raw_dim": "1d",
            "raw_x_unit": x_label,
            "raw_y_unit": y_label,
        }

        # Try to infer if x is shift_cm1, energy_eV, or something else based on label?
        # For now, let's store it generic, but if x_label suggests shift/eV, populate those fields.
        
        shift_cm1 = None
        energy_eV = None
        wl_nm = None
        angle_values = None
        
        x_label_lower = str(x_label).lower()

        # Simple heuristic mapping
        if "mev" in x_label_lower:
            energy_eV = x / 1000.0
            shift_cm1 = energy_eV / EV_PER_CM1
        elif "cm" in x_label_lower and ("-1" in x_label_lower or "^-1" in x_label_lower):
            shift_cm1 = x
            energy_eV = shift_to_eV(shift_cm1)
        elif "ev" in x_label_lower or "energy" in x_label_lower:
            energy_eV = x
            shift_cm1 = energy_eV / EV_PER_CM1
        elif "nm" in x_label_lower:
            wl_nm = x
        elif "angle" in x_label_lower or "deg" in x_label_lower:
            angle_values = x
        else:
            # Default fallback for generic data
            shift_cm1 = x
            energy_eV = shift_to_eV(shift_cm1)
            
        return cls(
            id=new_run_id(prefix="derived"),
            source_path="", # No source file
            source_mtime=None,
            wl_nm=wl_nm,
            shift_cm1=shift_cm1,
            energy_eV=energy_eV,
            intensity=y,
            intensity_2d=None,
            angle_values=angle_values,
            intensity_unit=infer_intensity_unit(y),
            angle_unit="deg",
            metadata=metadata,
            run_type=RunType.RUN_1D,
            raw_table=None,
        )

    @classmethod
    def from_file(
        cls,
        path: str,
        default_unknown_1d_spectral_unit: str = "meV",
        default_unknown_2d_spectral_unit: str = "meV",
    ) -> "Run":
        """
        Build a Run from a data file using load_table and parse_filename.

        Logic:
        - Use load_table(path):
            * (x, y)  -> treat as 1D spectrum.
            * (x, y, Z) -> treat as 2D map (e.g. angular_matrix_meV.csv).
        - For 2D case:
            * x is interpreted from filename/header when possible.
            * otherwise x defaults to ``default_unknown_2d_spectral_unit``.
            * y is assumed to be angle in degrees.
            * Converts meV -> eV -> cm^-1 for the x-axis.
        - For 1D case:
            * Only stores raw x and intensity in metadata for now; detailed
              axis semantics will be handled later in the analysis layer.
        """
        arrays, header_labels = load_table(path)
        info = parse_filename(path)
        metadata = dict(info)
        default_unknown_1d_spectral_unit = normalize_spectral_unit(default_unknown_1d_spectral_unit)
        default_unknown_2d_spectral_unit = normalize_spectral_unit(default_unknown_2d_spectral_unit)

        try:
            mtime = os.path.getmtime(path)
        except OSError:
            mtime = None

        # 1D: (x, y)
        if isinstance(arrays, tuple) and len(arrays) == 2:
            x, y = arrays
            x = np.asarray(x, dtype=float)
            y = np.asarray(y, dtype=float)

            metadata.setdefault("raw_dim", "1d")
            metadata.setdefault("raw_x_unit", "unknown")
            metadata.setdefault("raw_y_unit", "intensity")
            
            # Infer axes from header if available
            shift_cm1 = None
            angle_values = None
            energy_eV = None
            wl_nm = None
            
            # Default to shift_cm1 if no better match found
            default_x = x
            
            # 1. Check filename for explicit units first (Priority)
            fname_lower = os.path.basename(path).lower()
            found_in_fname = False
            
            if "cm-1" in fname_lower or "cm1" in fname_lower:
                shift_cm1 = x
                energy_eV = shift_to_eV(shift_cm1)
                metadata["raw_x_unit"] = "cm-1"
                found_in_fname = True
                default_x = None
            elif "mev" in fname_lower:
                energy_eV = x / 1000.0
                shift_cm1 = energy_eV / EV_PER_CM1
                metadata["raw_x_unit"] = "meV"
                found_in_fname = True
                default_x = None
            # Basic check for 'eV' but avoid matching 'level' etc if possible. 
            # Assuming '_ev' or 'ev.' or ' ev' pattern or just presence if user says so.
            elif "ev" in fname_lower and "level" not in fname_lower and "dev" not in fname_lower: 
                energy_eV = x
                shift_cm1 = energy_eV / EV_PER_CM1
                metadata["raw_x_unit"] = "eV"
                found_in_fname = True
                default_x = None
            elif "nm" in fname_lower and "wl" in fname_lower: # explicit wavelength hint
                wl_nm = x
                metadata["raw_x_unit"] = "nm"
                found_in_fname = True
                default_x = None

            # 2. If not found in filename, check headers
            if not found_in_fname and header_labels and len(header_labels) > 0:
                x_label = str(header_labels[0]).lower()
                metadata["raw_x_unit"] = str(header_labels[0]) # store original label
                
                if any(k in x_label for k in ["angle", "deg", "theta"]):
                    angle_values = x
                    default_x = None 
                elif "mev" in x_label:
                    energy_eV = x / 1000.0
                    shift_cm1 = energy_eV / EV_PER_CM1
                    metadata["raw_x_unit"] = "meV"
                    default_x = None
                elif any(k in x_label for k in ["ev", "energy"]):
                    energy_eV = x
                    # attempt auto-conversion
                    shift_cm1 = energy_eV / EV_PER_CM1
                    metadata["raw_x_unit"] = "eV"
                    default_x = None
                elif any(k in x_label for k in ["nm", "wave"]):
                    wl_nm = x
                    default_x = None
                elif any(k in x_label for k in ["cm-1", "raman", "shift", "wavenumber"]):
                    shift_cm1 = x
                    energy_eV = shift_to_eV(shift_cm1)
                    metadata["raw_x_unit"] = "cm-1"
                    default_x = None
            
            if default_x is not None:
                if default_unknown_1d_spectral_unit == "meV":
                    energy_eV = default_x / 1000.0
                    shift_cm1 = energy_eV / EV_PER_CM1
                else:
                    shift_cm1 = default_x
                    energy_eV = shift_to_eV(shift_cm1)
                metadata["raw_x_unit"] = default_unknown_1d_spectral_unit

            return cls(
                id=new_run_id(),
                source_path=os.path.abspath(path),
                source_mtime=mtime,
                wl_nm=wl_nm,
                shift_cm1=shift_cm1,
                energy_eV=energy_eV,
                intensity=y,
                intensity_2d=None,
                angle_values=angle_values,
                intensity_unit=infer_intensity_unit(y),
                angle_unit="deg",
                metadata=metadata,
                run_type=RunType.RUN_1D,
                raw_table=None,
            )

        # 2D: (x_axis, y_axis, matrix)
        if isinstance(arrays, tuple) and len(arrays) == 3:
            x_raw, y_axis, intensity = arrays
            x_raw = np.asarray(x_raw, dtype=float)
            y_axis = np.asarray(y_axis, dtype=float)
            intensity = np.asarray(intensity, dtype=float)

            # Check filename for units
            fname_lower = os.path.basename(path).lower()
            
            if "cm-1" in fname_lower or "cm1" in fname_lower:
                # Treat as Raman shift (cm^-1)
                shift_cm1 = x_raw
                energies_eV = shift_to_eV(shift_cm1)
                metadata.setdefault("raw_x_unit", "cm-1")
            elif "mev" in fname_lower:
                # Treat as meV
                energies_eV = x_raw / 1000.0
                shift_cm1 = energies_eV / EV_PER_CM1
                metadata.setdefault("raw_x_unit", "meV")
            else:
                if default_unknown_2d_spectral_unit == "meV":
                    energies_eV = x_raw / 1000.0
                    shift_cm1 = energies_eV / EV_PER_CM1
                else:
                    shift_cm1 = x_raw
                    energies_eV = shift_to_eV(shift_cm1)
                metadata.setdefault("raw_x_unit", default_unknown_2d_spectral_unit)

            metadata.setdefault("raw_dim", "2d")
            metadata.setdefault("raw_y_unit", "deg")

            return cls(
                id=new_run_id(),
                source_path=os.path.abspath(path),
                source_mtime=mtime,
                wl_nm=None,
                shift_cm1=shift_cm1,
                energy_eV=energies_eV,
                intensity=None,
                intensity_2d=intensity,
                angle_values=y_axis,
                intensity_unit=infer_intensity_unit(intensity),
                angle_unit="deg",
                metadata=metadata,
                run_type=RunType.RUN_2D,
                raw_table=None,
            )

        raise ValueError("Unsupported data shape returned by load_table(path).")


@dataclass
class ViewState:
    """
    Visualization state for a single GUI 'View' tab.

    This does not depend on any particular plotting backend. It just
    tells the GUI which runs to display and which options to use.
    """

    id: str
    title: str = "View"

    # Which runs are displayed in this view (0–2 for now)
    run_ids: List[str] = field(default_factory=list)

    # Styling configuration per run. Key is run_id.
    run_configs: Dict[str, RunViewConfig] = field(default_factory=dict)

    # Display options
    x_axis: str = "shift_cm1"   # Legacy: "shift_cm1" or "energy_eV"
    spectral_unit: str = "meV"
    angle_slice_type: str = "polar"
    slice_x_binning: int = 1
    slice_y_binning: int = 1
    slice_binning_mode: str = "cross"
    fit_overlay_mode: str = "off"
    show_secondary_unit_axis: bool = True
    normalize: bool = False
    show_legend: bool = False
    
    # Persisted plot settings
    xlim: Optional[Tuple[float, float]] = None
    ylim: Optional[Tuple[float, float]] = None
    vmin: float = 0.0
    vmax: float = 100.0
    cmap: str = "OrRd"
    graph_configs: Dict[str, GraphViewConfig] = field(default_factory=dict)

    # Curve-fit placeholders (to be filled by analysis / GUI)
    active_fit_id: Optional[str] = None
    fit_params: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.spectral_unit = normalize_spectral_unit(self.spectral_unit)
        if self.angle_slice_type not in {"polar", "cartesian"}:
            self.angle_slice_type = "polar"
        self.slice_x_binning = self._coerce_binning(self.slice_x_binning)
        self.slice_y_binning = self._coerce_binning(self.slice_y_binning)
        self.slice_binning_mode = str(self.slice_binning_mode).lower()
        if self.slice_binning_mode not in {"cross", "box"}:
            self.slice_binning_mode = "cross"
        self.fit_overlay_mode = str(self.fit_overlay_mode).lower()
        if self.fit_overlay_mode not in {"off", "global", "row", "both"}:
            self.fit_overlay_mode = "off"

    @staticmethod
    def _coerce_binning(value: Any) -> int:
        try:
            return max(1, int(value))
        except (TypeError, ValueError):
            return 1

    @property
    def n_runs(self) -> int:
        return len(self.run_ids)
    
    def get_run_config(self, run_id: str) -> RunViewConfig:
        if run_id not in self.run_configs:
            self.run_configs[run_id] = RunViewConfig()
        return self.run_configs[run_id]

    def get_graph_config(self, graph_id: str) -> GraphViewConfig:
        """
        Return per-panel plot settings for keys such as ``1A`` or ``2C``.

        The older view-level ``xlim``/``ylim``/``vmin``/``vmax``/``cmap`` fields
        remain in place for backward compatibility. Newer saves mirror the
        primary map into those fields so older code can still make sense of it.
        """
        if graph_id not in self.graph_configs:
            self.graph_configs[graph_id] = GraphViewConfig()
        return self.graph_configs[graph_id]

    def seed_legacy_graph_configs(self) -> None:
        """Populate graph configs from legacy view-level settings when needed."""
        if self.graph_configs:
            return

        legacy_vmin = self.vmin
        legacy_vmax = self.vmax
        legacy_cmap = self.cmap

        for slot in ("1", "2"):
            map_cfg = self.get_graph_config(f"{slot}A")
            map_cfg.xlim = tuple(self.xlim) if self.xlim else None
            map_cfg.ylim = tuple(self.ylim) if self.ylim else None
            map_cfg.vmin = legacy_vmin
            map_cfg.vmax = legacy_vmax
            map_cfg.cmap = legacy_cmap

            # Plot B uses angle on its x-axis; Plot C uses shift on its x-axis.
            b_cfg = self.get_graph_config(f"{slot}B")
            b_cfg.xlim = tuple(self.ylim) if self.ylim else None

            c_cfg = self.get_graph_config(f"{slot}C")
            c_cfg.xlim = tuple(self.xlim) if self.xlim else None

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ViewState":
        run_configs_raw = data.get("run_configs", {})
        run_configs = {k: RunViewConfig.from_dict(v) for k, v in run_configs_raw.items()}
        graph_configs_raw = data.get("graph_configs", {})
        graph_configs = {k: GraphViewConfig.from_dict(v) for k, v in graph_configs_raw.items()}
        
        # Remove these from dict before unpacking to avoid double init
        base_data = dict(data)
        for key in ("run_configs", "graph_configs"):
            if key in base_data:
                del base_data[key]

        allowed = set(cls.__dataclass_fields__.keys())
        base_data = {k: v for k, v in base_data.items() if k in allowed}
        if "spectral_unit" in base_data:
            base_data["spectral_unit"] = normalize_spectral_unit(base_data.get("spectral_unit"))
        elif base_data.get("x_axis") == "energy_eV":
            base_data["spectral_unit"] = "meV"
        else:
            base_data["spectral_unit"] = "meV"
        if base_data.get("angle_slice_type") not in {"polar", "cartesian"}:
            base_data["angle_slice_type"] = "polar"
        base_data["slice_x_binning"] = cls._coerce_binning(base_data.get("slice_x_binning", 1))
        base_data["slice_y_binning"] = cls._coerce_binning(base_data.get("slice_y_binning", 1))
        base_data["slice_binning_mode"] = str(base_data.get("slice_binning_mode", "cross")).lower()
        if base_data.get("slice_binning_mode") not in {"cross", "box"}:
            base_data["slice_binning_mode"] = "cross"
        base_data["fit_overlay_mode"] = str(base_data.get("fit_overlay_mode", "off")).lower()
        if base_data.get("fit_overlay_mode") not in {"off", "global", "row", "both"}:
            base_data["fit_overlay_mode"] = "off"
        base_data["show_secondary_unit_axis"] = bool(base_data.get("show_secondary_unit_axis", True))
        for key in ("xlim", "ylim"):
            value = base_data.get(key)
            if isinstance(value, (list, tuple)) and len(value) == 2:
                try:
                    base_data[key] = (float(value[0]), float(value[1]))
                except (TypeError, ValueError):
                    base_data[key] = None
        view = cls(run_configs=run_configs, graph_configs=graph_configs, **base_data)
        view.seed_legacy_graph_configs()
        return view

@dataclass
class ExperimentSet:
    """
    Collection of Runs + Views + experiment-level metadata.

    This is the main object that the GUI will manipulate and that can
    be saved/loaded as a whole (JSON for light summaries, HDF5 for full
    numeric arrays).

    The ExperimentSet is also responsible for assigning and managing
    human-friendly run nicknames (Run01, Run02, ...) via a simple
    monotonically increasing counter.
    """

    id: str
    runs: Dict[str, Run] = field(default_factory=dict)
    views: Dict[str, ViewState] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    #: Next index used for assigning default run nicknames, e.g. "Run01".
    next_run_index: int = 1

    # ---- Run management ----
    def new_run_nickname(self) -> str:
        """
        Generate the next default run nickname of the form "RunNN".
        """
        nickname = f"Run{self.next_run_index:02d}"
        self.next_run_index += 1
        return nickname

    def add_run(self, run: Run) -> None:
        """
        Add a Run to the experiment.

        If the run does not yet have a nickname in its metadata, assign
        a default one of the form "RunNN" using this experiment's
        monotonically increasing index.
        """
        if not isinstance(run.metadata, dict):
            run.metadata = {}

        # Only assign a nickname if one is not already present
        nickname = run.metadata.get("nickname")
        if not (isinstance(nickname, str) and nickname.strip()):
            run.nickname = self.new_run_nickname()

        self.runs[run.id] = run

    def remove_run(self, run_id: str) -> None:
        if run_id in self.runs:
            del self.runs[run_id]
        # Detach from any views that reference this run
        for v in self.views.values():
            if run_id in v.run_ids:
                v.run_ids = [rid for rid in v.run_ids if rid != run_id]

    def get_run(self, run_id: str) -> Optional[Run]:
        return self.runs.get(run_id)

    def get_run_nickname(self, run_id: str) -> str:
        """
        Return the nickname for the given run id.

        Falls back to the run id itself if the run is missing or if
        no nickname is defined.
        """
        run = self.runs.get(run_id)
        if run is None:
            return run_id
        return run.nickname

    def set_run_nickname(self, run_id: str, name: str) -> None:
        """
        Set or update the nickname for the given run id.

        If the run does not exist, this function is a no-op.
        """
        run = self.runs.get(run_id)
        if run is None:
            return
        run.nickname = name

    # ---- View management ----

    def add_view(self, view: ViewState) -> None:
        self.views[view.id] = view

    def remove_view(self, id: int) -> None:
        pass

    def get_view(self, view_id: str) -> Optional[ViewState]:
        return self.views.get(view_id)

    # ---- Serialization helpers ----

    def to_json_dict(self) -> Dict[str, Any]:
        """
        Lightweight JSON representation of the experiment.

        This is not meant to hold full numeric arrays, only metadata and
        shape information. Use export_hdf5 for full numeric storage.
        """
        return {
            "id": self.id,
            "metadata": self.metadata,
            "runs": {rid: r.to_dict_summary() for rid, r in self.runs.items()},
            "views": {vid: asdict(v) for vid, v in self.views.items()},
            "next_run_index": self.next_run_index,
        }

    def export_json(self, path: str) -> None:
        """Export summary metadata to a JSON file."""
        import json

        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_json_dict(), f, indent=2, ensure_ascii=False)

    def export_igor(self, path: str) -> str:
        """
        Export numeric runs to an Igor Text (.itx) file without external deps.

        The file recreates dataset folders shaped like the reference PXP data
        folder ``root:CsPdS_bi_xx``:

        - ``root:<run_name>:wave0...waveN`` are per-x vertical slices.
        - ``root:<run_name>:<run_name>_xaxis:wave0...waveN`` are scalar x
          coordinate waves matching those slices.
        - ``wave2D`` is an x-by-angle matrix.
        - ``wave2D_interp`` is a uniformly scaled ImageTool companion wave
          interpolated from the nonuniform x-axis, matching the reference PXP
          profile/mask state.
        - ``wave1Dx`` is the Raman-shift x axis in cm-1.
        - ``wave1Dx_meV`` is the same axis in meV.
        - ``wavey`` mirrors the last angle column of ``wave2D``, matching the
          reference PXP folders.

        Native .pxp files are packed binary Igor experiments. If a .pxp path is
        passed, this method writes a sibling .itx file instead of creating a
        mislabeled text file that Igor Pro cannot open.
        """
        root, ext = os.path.splitext(path)
        if ext.lower() in {".pxp", ".pxt"}:
            path = root + ".itx"
        elif not ext:
            path = path + ".itx"

        def sanitize_wave_name(name: str, used: set[str]) -> str:
            name = re.sub(r"\W+", "_", str(name or "wave")).strip("_")
            if not name:
                name = "wave"
            if name[0].isdigit():
                name = f"w_{name}"
            name = name[:31]

            base = name
            suffix = 1
            while name in used:
                extra = f"_{suffix}"
                name = f"{base[:31 - len(extra)]}{extra}"
                suffix += 1
            used.add(name)
            return name

        def sanitize_local_name(name: str, fallback: str, used: set[str]) -> str:
            return sanitize_wave_name(name or fallback, used)

        def format_value(value: float) -> str:
            value = float(value)
            if not np.isfinite(value):
                return "NaN"
            return f"{value:.12g}"

        def write_wave(f, wave_name: str, data: np.ndarray, wave_type: str = "double") -> None:
            if wave_type == "byte":
                arr = np.asarray(data, dtype=np.uint8)
                wave_flag = "/B/U"
            elif wave_type == "single":
                arr = np.asarray(data, dtype=np.float32)
                wave_flag = ""
            else:
                arr = np.asarray(data, dtype=float)
                wave_flag = "/D"

            if arr.ndim == 0:
                arr = arr.reshape(1)
            elif arr.ndim > 2:
                arr = arr.reshape(arr.shape[0], -1)

            def _format_array_value(value) -> str:
                if wave_type == "byte":
                    return str(int(value))
                return format_value(float(value))

            if arr.ndim == 1:
                f.write(f"WAVES{wave_flag}/N=({arr.shape[0]}) {wave_name}\n")
                f.write("BEGIN\n")
                for value in arr:
                    f.write(f"{_format_array_value(value)}\n")
                f.write("END\n")
                return

            f.write(f"WAVES{wave_flag}/N=({arr.shape[0]},{arr.shape[1]}) {wave_name}\n")
            f.write("BEGIN\n")
            for row in arr:
                f.write("\t".join(_format_array_value(v) for v in row))
                f.write("\n")
            f.write("END\n")

        def write_x_scale(f, wave_name: str, start: float, delta: float) -> None:
            if np.isfinite(start) and np.isfinite(delta) and delta != 0:
                f.write(f"X SetScale/P x, {format_value(start)}, {format_value(delta)}, {wave_name}\n")

        def make_interpolated_image(
            x_values: np.ndarray,
            image_xy: np.ndarray,
        ) -> Tuple[np.ndarray, Optional[float], Optional[float]]:
            """
            Build the uniformly scaled ImageTool image stored in the reference PXP.

            The raw wave2D keeps the nonuniform x grid through the scalar xaxis
            folder. ImageTool's saved profile/mask state is tied to a uniformly
            scaled companion wave, so we export both.
            """
            x = np.asarray(x_values, dtype=float)
            image = np.asarray(image_xy, dtype=float)
            if x.ndim != 1 or image.ndim != 2 or image.shape[0] != x.size or x.size < 2:
                return image, None, None

            finite_x = np.isfinite(x)
            if finite_x.sum() < 2:
                return image, None, None

            x = x[finite_x]
            image = image[finite_x, :]
            order = np.argsort(x)
            x = x[order]
            image = image[order, :]

            unique_x, inverse = np.unique(x, return_inverse=True)
            if unique_x.size < 2:
                return image, None, None
            if unique_x.size != x.size:
                merged = np.empty((unique_x.size, image.shape[1]), dtype=float)
                for idx in range(unique_x.size):
                    merged[idx, :] = np.nanmean(image[inverse == idx, :], axis=0)
                image = merged
                x = unique_x

            x_start = float(np.ceil(np.nanmin(x)))
            x_stop = float(np.floor(np.nanmax(x)))
            if not np.isfinite(x_start) or not np.isfinite(x_stop) or x_stop <= x_start:
                x_start = float(x[0])
                x_stop = float(x[-1])
            if x_stop <= x_start:
                return image, None, None

            interp_count = max(1000, int(x_values.size))
            x_interp = np.linspace(x_start, x_stop, interp_count)
            image_interp = np.empty((interp_count, image.shape[1]), dtype=float)
            for col in range(image.shape[1]):
                y = image[:, col]
                finite = np.isfinite(y)
                if finite.sum() >= 2:
                    image_interp[:, col] = np.interp(x_interp, x[finite], y[finite])
                elif finite.sum() == 1:
                    image_interp[:, col] = float(y[finite][0])
                else:
                    image_interp[:, col] = np.nan

            delta = float((x_stop - x_start) / (interp_count - 1))
            return image_interp, x_start, delta

        def folder_path(folder_name: str) -> str:
            return f"root:{folder_name}"

        def write_set_folder(f, folder_name: str) -> None:
            f.write("X SetDataFolder root:\n")
            f.write(f"X NewDataFolder/O {folder_path(folder_name)}\n")
            f.write(f"X SetDataFolder {folder_path(folder_name)}\n")

        used_names: set[str] = set()
        used_folders: set[str] = set()

        with open(path, "w", encoding="utf-8") as f:
            f.write("IGOR\n")
            f.write("X SetDataFolder root:\n")

            for run_id, run in self.runs.items():
                if run.is_2d:
                    folder_name = sanitize_local_name(run.nickname, run_id, used_folders)
                    write_set_folder(f, folder_name)

                    local_used: set[str] = set()
                    intensity = np.asarray(run.intensity_2d, dtype=float)
                    x_axis = np.asarray(run.shift_cm1, dtype=float) if run.shift_cm1 is not None else np.arange(intensity.shape[1], dtype=float)
                    if run.angle_values is not None:
                        y_axis = np.asarray(run.angle_values, dtype=float)
                    else:
                        y_axis = np.arange(intensity.shape[0], dtype=float)

                    if intensity.shape != (y_axis.size, x_axis.size):
                        if intensity.shape == (x_axis.size, y_axis.size):
                            intensity_yx = intensity.T
                        else:
                            raise ValueError(
                                f"Run '{run.nickname}' has incompatible 2D shape {intensity.shape} "
                                f"for axes ({y_axis.size}, {x_axis.size})."
                            )
                    else:
                        intensity_yx = intensity

                    # Reference PXP stores wave2D as (x, y), while Venkata stores
                    # intensity_2d as (y, x).
                    intensity_xy = np.asarray(intensity_yx.T, dtype=float)

                    for idx in range(intensity_xy.shape[0]):
                        write_wave(f, f"wave{idx}", intensity_xy[idx, :])

                    wave2d_name = sanitize_wave_name("wave2D", local_used)
                    write_wave(f, wave2d_name, intensity_xy, wave_type="single")
                    wave2d_interp, interp_start, interp_delta = make_interpolated_image(x_axis, intensity_xy)
                    wave2d_interp_name = sanitize_wave_name("wave2D_interp", local_used)
                    write_wave(f, wave2d_interp_name, wave2d_interp, wave_type="single")
                    if interp_start is not None and interp_delta is not None:
                        write_x_scale(f, wave2d_interp_name, interp_start, interp_delta)

                    write_wave(f, sanitize_wave_name("wave1Dx", local_used), x_axis, wave_type="single")
                    if run.energy_eV is not None and len(run.energy_eV) == x_axis.size:
                        x_mev = np.asarray(run.energy_eV, dtype=float) * 1000.0
                    else:
                        x_mev = x_axis * EV_PER_CM1 * 1000.0
                    write_wave(f, sanitize_wave_name("wave1Dx_meV", local_used), x_mev, wave_type="single")
                    # In the reference PXP folders, wavey exactly matches the
                    # last angle column of wave2D. ImageTool uses these helper
                    # waves for profile state, so using the first column makes
                    # the displayed image look right while the slice/profile
                    # values start from the wrong trace.
                    write_wave(f, sanitize_wave_name("wavey", local_used), intensity_xy[:, -1], wave_type="single")
                    roi_name = sanitize_wave_name("M_ROIMask", local_used)
                    write_wave(f, roi_name, np.ones(wave2d_interp.shape, dtype=np.uint8), wave_type="byte")
                    if interp_start is not None and interp_delta is not None:
                        write_x_scale(f, roi_name, interp_start, interp_delta)
                    write_wave(f, sanitize_wave_name("proc_ROIx", local_used), np.array([], dtype=float), wave_type="single")
                    write_wave(f, sanitize_wave_name("proc_ROIy", local_used), np.array([], dtype=float), wave_type="single")
                    write_wave(f, sanitize_wave_name("sel_ROIx", local_used), np.array([], dtype=float), wave_type="single")
                    write_wave(f, sanitize_wave_name("sel_ROIy", local_used), np.array([], dtype=float), wave_type="single")

                    x_folder = sanitize_wave_name(f"{folder_name}_xaxis", set())
                    f.write(f"X NewDataFolder/O {folder_path(folder_name)}:{x_folder}\n")
                    f.write(f"X SetDataFolder {folder_path(folder_name)}:{x_folder}\n")
                    for idx, x_val in enumerate(x_axis):
                        write_wave(f, f"wave{idx}", np.array([x_val], dtype=float))
                    f.write(f"X SetDataFolder {folder_path(folder_name)}\n")

                elif run.intensity is not None:
                    folder_name = sanitize_local_name(run.nickname, run_id, used_folders)
                    write_set_folder(f, folder_name)

                    local_used: set[str] = set()
                    wave_name = sanitize_wave_name("trace", local_used)
                    write_wave(f, wave_name, run.intensity)

                    if run.shift_cm1 is not None:
                        x_name = sanitize_wave_name("x_cm1", local_used)
                        write_wave(f, x_name, run.shift_cm1)
                    elif run.angle_values is not None:
                        x_name = sanitize_wave_name("x_deg", local_used)
                        write_wave(f, x_name, run.angle_values)
                    elif run.wl_nm is not None:
                        x_name = sanitize_wave_name("x_nm", local_used)
                        write_wave(f, x_name, run.wl_nm)
                    elif run.energy_eV is not None:
                        x_name = sanitize_wave_name("x_eV", local_used)
                        write_wave(f, x_name, run.energy_eV)

            f.write("X SetDataFolder root:\n\n")

        return path

    def export_hdf5(self, path: str) -> None:
        """
        Export the experiment (including numeric arrays) to an HDF5 file.

        HDF5 layout (high-level sketch):
        - attrs:
            - experiment_id = self.id
        - metadata/json : experiment-level metadata as JSON string
        - runs/<run_id>/... : all arrays + scalar attrs for each Run
        - views/json : JSON string with serialized ViewState dicts
        """
        import h5py
        import json

        with h5py.File(path, "w") as h5:
            h5.attrs["experiment_id"] = self.id

            # Experiment-level metadata as JSON
            md_grp = h5.create_group("metadata")
            md_grp.attrs["json"] = json.dumps(self.metadata, ensure_ascii=False)
            md_grp.attrs["next_run_index"] = self.next_run_index

            # Runs
            runs_grp = h5.create_group("runs")
            for rid, r in self.runs.items():
                rg = runs_grp.create_group(rid)

                # Basic scalar attrs
                rg.attrs["source_path"] = r.source_path
                rg.attrs["source_mtime"] = r.source_mtime if r.source_mtime is not None else 0.0
                rg.attrs["intensity_unit"] = r.intensity_unit
                rg.attrs["angle_unit"] = r.angle_unit
                rg.attrs["run_type"] = r.run_type.value
                rg.attrs["metadata_json"] = json.dumps(r.metadata, ensure_ascii=False)

                # Numeric arrays
                if r.wl_nm is not None:
                    rg.create_dataset("wl_nm", data=r.wl_nm)
                if r.shift_cm1 is not None:
                    rg.create_dataset("shift_cm1", data=r.shift_cm1)
                if r.energy_eV is not None:
                    rg.create_dataset("energy_eV", data=r.energy_eV)
                if r.intensity is not None:
                    rg.create_dataset("intensity", data=r.intensity)
                if r.intensity_2d is not None:
                    rg.create_dataset("intensity_2d", data=r.intensity_2d)
                if r.angle_values is not None:
                    rg.create_dataset("angle_values", data=r.angle_values)

            # Views: store as a JSON string for simplicity
            views_grp = h5.create_group("views")
            views_grp.attrs["json"] = json.dumps(
                {vid: asdict(v) for vid, v in self.views.items()},
                ensure_ascii=False,
            )

    @classmethod
    def from_hdf5(cls, path: str) -> "ExperimentSet":
        """
        Load an experiment from an HDF5 file.
        """
        import h5py
        import json

        with h5py.File(path, "r") as h5:
            # ID
            exp_id = h5.attrs.get("experiment_id", new_experiment_id())
            if isinstance(exp_id, bytes):
                exp_id = exp_id.decode("utf-8")

            # Metadata
            md_grp = h5["metadata"]
            metadata = json.loads(md_grp.attrs["json"])
            next_run_index = int(md_grp.attrs.get("next_run_index", 1))

            # Runs
            runs = {}
            if "runs" in h5:
                runs_grp = h5["runs"]
                for rid in runs_grp:
                    rg = runs_grp[rid]

                    # Scalar attrs
                    source_path = rg.attrs["source_path"]
                    if isinstance(source_path, bytes):
                        source_path = source_path.decode("utf-8")
                    
                    source_mtime = rg.attrs.get("source_mtime")
                    intensity_unit = rg.attrs.get("intensity_unit", "au")
                    if isinstance(intensity_unit, bytes):
                        intensity_unit = intensity_unit.decode("utf-8")
                        
                    angle_unit = rg.attrs.get("angle_unit", "deg")
                    if isinstance(angle_unit, bytes):
                        angle_unit = angle_unit.decode("utf-8")
                    
                    run_type_str = rg.attrs.get("run_type", RunType.OTHER.value)
                    if isinstance(run_type_str, bytes):
                        run_type_str = run_type_str.decode("utf-8")
                    try:
                        run_type = RunType(run_type_str)
                    except ValueError:
                        run_type = RunType.OTHER
                        
                    run_md = json.loads(rg.attrs["metadata_json"])

                    # Arrays (helper)
                    def read_ds(name):
                        return rg[name][:] if name in rg else None
                    
                    intensity_2d = read_ds("intensity_2d")

                    r = Run(
                        id=rid,
                        source_path=source_path,
                        source_mtime=source_mtime,
                        wl_nm=read_ds("wl_nm"),
                        shift_cm1=read_ds("shift_cm1"),
                        energy_eV=read_ds("energy_eV"),
                        intensity=read_ds("intensity"),
                        intensity_2d=intensity_2d,
                        angle_values=read_ds("angle_values"),
                        intensity_unit=intensity_unit,
                        angle_unit=angle_unit,
                        metadata=run_md,
                        run_type=run_type,
                        raw_table=None, 
                    )
                    
                    # Infer run_type for legacy files
                    if r.run_type == RunType.OTHER:
                        if r.metadata.get("derived"):
                            r.run_type = RunType.DERIVED
                        elif r.intensity_2d is not None:
                            r.run_type = RunType.RUN_2D
                        else:
                            r.run_type = RunType.RUN_1D
                            
                    runs[rid] = r

            # Views
            views = {}
            if "views" in h5:
                views_grp = h5["views"]
                if "json" in views_grp.attrs:
                    views_dict = json.loads(views_grp.attrs["json"])
                    for vid, vdata in views_dict.items():
                        # Reconstruct ViewState
                        views[vid] = ViewState.from_dict(vdata)

            return cls(
                id=exp_id,
                runs=runs,
                views=views,
                metadata=metadata,
                next_run_index=next_run_index
            )


# -------------------------
# ID helper functions
# -------------------------


def new_run_id(prefix: str = "run") -> str:
    """Generate a simple time-based run id."""
    import time
    return f"{prefix}_{int(time.time() * 1000)}"


def new_view_id(prefix: str = "view") -> str:
    """Generate a simple time-based view id."""
    import time
    return f"{prefix}_{int(time.time() * 1000)}"


def new_experiment_id(prefix: str = "exp") -> str:
    """Generate a simple time-based experiment id."""
    import time
    return f"{prefix}_{int(time.time() * 1000)}"


# -------------------------
# Quick test helper (manual use only)
# -------------------------




def test_with_files(filename1: str, filename2: str) -> None:
    """
    Quick manual test for data_structure.py.

    - Creates an ExperimentSet.
    - Parses the two filenames and builds Run objects with metadata.
    - Prints a summary of each Run.
    - Produces simple 2D matplotlib plots:
        * x-axis: Raman shift in cm^-1 (bottom)
        * top x-axis: Raman shift in eV
        * y-axis: angle (deg)
        * color: intensity
    """
    import matplotlib.pyplot as plt

    exp = ExperimentSet(id=new_experiment_id())

    run1 = Run.from_file(filename1)
    run2 = Run.from_file(filename2)

    exp.add_run(run1)
    exp.add_run(run2)

    # Print metadata summaries
    for i, run in enumerate((run1, run2), start=1):
        md = run.metadata
        print(f"Run {i}:")
        print(f"  id           : {run.id}")
        print(f"  source_path  : {run.source_path}")
        print(f"  category     : {md.get('category')}")
        print(f"  sample       : {md.get('sample')}")
        print(f"  inttime      : {md.get('inttime')}")
        print(f"  grating      : {md.get('grating')}")
        print(f"  slit         : {md.get('slit')}")
        print(f"  laser_nm_str : {md.get('laser_nm_str')}")
        print(f"  laser_nm     : {md.get('laser_nm')}")
        print(f"  power        : {md.get('power')}")
        print(f"  pol          : {md.get('pol')}")
        print(f"  temp_str     : {md.get('temp_str') or md.get('temp')}")
        print(f"  temp_K       : {md.get('temp_K')}")
        print(f"  has_angular  : {md.get('has_angular')}")
        print(f"  angle_idx    : {md.get('angle_idx')}")
        print(f"  rep_idx      : {md.get('rep_idx')}")
        print(f"  tail         : {md.get('tail')}")
        print(f"  angle_values : {None if run.angle_values is None else run.angle_values.shape}")
        print(f"  shift_cm1    : {None if run.shift_cm1 is None else run.shift_cm1.shape}")
        print(f"  energy_eV    : {None if run.energy_eV is None else run.energy_eV.shape}")
        print(f"  intensity_2d : {None if run.intensity_2d is None else run.intensity_2d.shape}")
        print(f"  intensity_unit: {run.intensity_unit}")
        print("")

    # Prepare plots
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
    runs = [run1, run2]
    titles = [
        f"Run 1: {os.path.basename(run1.source_path)}",
        f"Run 2: {os.path.basename(run2.source_path)}",
    ]

    for ax, run, title in zip(axes, runs, titles):
        if run.intensity_2d is None or run.shift_cm1 is None or run.angle_values is None:
            ax.text(0.5, 0.5, "No 2D data", ha="center", va="center")
            ax.set_title(title)
            continue

        x_cm1 = np.asarray(run.shift_cm1, dtype=float)
        y_angle = np.asarray(run.angle_values, dtype=float)
        Z = np.asarray(run.intensity_2d, dtype=float)

        # Ensure orientation matches (angle, shift)
        if Z.shape == (y_angle.size, x_cm1.size):
            Z_plot = Z
        elif Z.shape == (x_cm1.size, y_angle.size):
            Z_plot = Z.T
        else:
            # Fallback: let imshow handle it; axes will be index-based
            Z_plot = Z
            x_cm1 = np.arange(Z_plot.shape[1], dtype=float)
            y_angle = np.arange(Z_plot.shape[0], dtype=float)

        im = ax.pcolormesh(x_cm1, y_angle, Z_plot, shading="auto")
        ax.set_xlabel("Raman shift (cm$^{-1}$)")
        ax.set_title(title)
        if ax is axes[0]:
            ax.set_ylabel("Angle (deg)")

        cb = fig.colorbar(im, ax=ax)
        cb.set_label("Intensity (arb. units)")

        # Top x-axis in eV
        def cm1_to_eV(x):
            return x * EV_PER_CM1

        def eV_to_cm1(e):
            return e / EV_PER_CM1

        ax_top = ax.secondary_xaxis("top", functions=(cm1_to_eV, eV_to_cm1))
        ax_top.set_xlabel("Raman shift (eV)")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    import sys

    if len(sys.argv) >= 3:
        test_with_files(sys.argv[1], sys.argv[2])
    else:
        print("Usage: python data_structure.py file1 file2")
        print("This will parse the filenames, build two 2D Runs,")
        print("print their metadata, and show simple 2D maps.")
