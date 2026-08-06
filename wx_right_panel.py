import io
import os
import analysis
from merge_runs_gui import MergeRunsDialog
from typing import List, Optional, Dict, Any, Tuple

import wx
import numpy as np

import matplotlib as mpl
import matplotlib.ticker as mticker
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.backends.backend_wxagg import (
    FigureCanvasWxAgg as FigureCanvas,
    NavigationToolbar2WxAgg as NavigationToolbar,
)

from data_structure import (
    ExperimentSet,
    Run,
    ViewState,
    new_experiment_id,
    new_view_id,
    RunType,
    alternate_spectral_unit,
    cm1_to_unit,
    normalize_spectral_unit,
    spectral_axis_for_run,
    spectral_axis_label,
    spectral_xlim_from_cm1,
    spectral_xlim_to_cm1,
    unit_to_cm1,
)
from config_manager import config
from fit_overlay import normalize_fit_overlay_mode, overlay_data
from plotting import RamanPlotter2d, AngularPlotter, SlicePlotter
from wx_left_lower_panel import PlotConfigPanel


class RamanToolbar(NavigationToolbar):
    """
    Thin wrapper around NavigationToolbar so we can hook the Home button.

    When the user presses Home, we first let matplotlib restore the original
    view (axis limits etc.), and then we ask the owning ViewPanel to
    re-apply the last chosen slice so that Plot B/C are not left blank.
    """
    def __init__(self, canvas, owner_panel: "ViewPanel"):
        self._owner_panel = owner_panel
        super().__init__(canvas)

    def home(self, *args, **kwargs):
        super().home(*args, **kwargs)
        if hasattr(self._owner_panel, "go_home"):
            self._owner_panel.go_home()
        if hasattr(self._owner_panel, "_refresh_after_home"):
            self._owner_panel._refresh_after_home()


class ViewPanel(wx.Panel):
    """
    One 'View' tab, showing up to one active Run as three horizontal plots:

    - Plot A: 2D color map (angle vs Raman shift).
    - Plot B: Angular slice from Plot A.
    - Plot C: Spectral slice from Plot A.

    Refactored to use plotting.py classes.
    """

    def __init__(self, parent, view_label: str, plot_config_panel: PlotConfigPanel, on_run_created=None, on_fit_request=None):
        super().__init__(parent)

        self.view_label = view_label
        self.plot_config_panel = plot_config_panel
        self.on_run_created = on_run_created
        self.on_fit_request = on_fit_request
        self.current_run_id: Optional[str] = None
        self.current_run: Optional[Run] = None
        self._experiment: Optional[ExperimentSet] = None
        self._view_state: Optional[ViewState] = None

        # Matplotlib figure and canvas
        self.figure = Figure(figsize=(9, 4.5))
        self.canvas = FigureCanvas(self, -1, self.figure)

        # Plotter instances (initialized in _draw_runs)
        self.plotterA1: Optional[RamanPlotter2d] = None
        self.plotterB1: Optional[AngularPlotter] = None
        self.plotterC1: Optional[SlicePlotter] = None

        self.plotterA2: Optional[RamanPlotter2d] = None
        self.plotterB2: Optional[AngularPlotter] = None
        self.plotterC2: Optional[SlicePlotter] = None

        # Click event connection id
        self._cid_click = None
        self._cid_resize = None

        # Highlight mode: "none", "click", "line_profile", "peak_fit"
        self.highlight_mode: str = "click"

        # Angle slice rendering type for Plot B: "polar" or "cartesian"
        self.angle_slice_type: str = "polar"
        self._last_drawn_runs: List[Run] = []

        # Guard flag to avoid recursive callbacks when syncing zoom between axes
        self._syncing_limits: bool = False
        self._limit_cb_ids: list[tuple[object, int]] = []

        # Persistent selection state (indices)
        self._sel_idx1: Optional[tuple[int, int]] = None
        self._sel_idx2: Optional[tuple[int, int]] = None

        # Track contrast state
        self._contrast_percent = (0.0, 100.0)
        self.current_cmap = "OrRd"

        self._suppress_config_save = False

        # Optional second run to visualize in the bottom row
        self._second_run: Optional[Run] = None

        # Layout
        sizer = wx.BoxSizer(wx.VERTICAL)
        label = wx.StaticText(self, label=view_label)
        label.SetForegroundColour(wx.Colour(50, 50, 50))
        sizer.Add(label, 0, wx.ALL, 4)

        self.toolbar = RamanToolbar(self.canvas, self)
        self.toolbar.Realize()
        sizer.Add(self.toolbar, 0, wx.EXPAND | wx.LEFT | wx.RIGHT, 4)

        sizer.Add(self.canvas, 1, wx.EXPAND | wx.ALL, 4)
        self.SetSizer(sizer)

        self._init_empty_figure()

    def _refresh_after_home(self) -> None:
        self._apply_selection_from_indices(reason="home")

    def go_home(self):
        """Reset the view to the full data range."""
        if not self.current_run:
            return

        # Determine full ranges from the primary run
        # We assume Plot A (Map) is the master for Shift and Angle limits.

        # 1. Shift Range (X-axis of Plot A, X-axis of Plot C)
        xmin, xmax = 0.0, 1.0
        x_axis = spectral_axis_for_run(self.current_run, self.get_spectral_unit())
        if x_axis is not None:
            xmin = np.min(x_axis)
            xmax = np.max(x_axis)

        # 2. Angle Range (Y-axis of Plot A, X-axis of Plot B)
        ymin, ymax = 0.0, 1.0
        if self.current_run.angle_values is not None:
            try:
                _shift, display_angles, _I = self._display_2d_for_run(self.current_run)
                ymin = np.min(display_angles)
                ymax = np.max(display_angles)
            except Exception:
                ymin = np.min(self.current_run.angle_values)
                ymax = np.max(self.current_run.angle_values)

        # Apply to Plot A (Map)
        # This should trigger sync callbacks to update C(x) and B(x)
        if self.plotterA1 and self.plotterA1.ax:
            self.plotterA1.ax.set_xlim(xmin, xmax)
            self.plotterA1.ax.set_ylim(ymin, ymax)

        # Reset Intensity limits for B and C (Auto-scale)
        # We don't have a specific "full intensity range" stored easily without scanning data,
        # but relim() + autoscale_view() usually works if data is plotted.
        if self.plotterB1 and self.plotterB1.ax:
            self.plotterB1.ax.relim()
            self.plotterB1.ax.autoscale_view()

        if self.plotterC1 and self.plotterC1.ax:
            self.plotterC1.ax.relim()
            self.plotterC1.ax.autoscale_view()

        self.canvas.draw_idle()

    # --- Public API for external controls ---

    def _iter_graph_plotters(self):
        return [
            ("1A", self.plotterA1),
            ("1B", self.plotterB1),
            ("1C", self.plotterC1),
            ("2A", self.plotterA2),
            ("2B", self.plotterB2),
            ("2C", self.plotterC2),
        ]

    def _primary_graph_key(self) -> str:
        run = self.current_run
        if run and run.run_type in (RunType.RUN_1D, RunType.DERIVED):
            x_unit = str(run.metadata.get("raw_x_unit", "")).lower()
            if "deg" in x_unit or "angle" in x_unit:
                return "1B"
            return "1C"
        return "1A"

    def get_spectral_unit(self) -> str:
        if self._view_state:
            return normalize_spectral_unit(self._view_state.spectral_unit)
        return "meV"

    def _display_2d_for_run(self, run: Run):
        return analysis.display_2d_from_run(run)

    def set_spectral_unit(self, unit: str) -> None:
        unit = normalize_spectral_unit(unit)
        if self._view_state and self._view_state.spectral_unit != unit:
            self._save_all_graph_configs()
            self._view_state.spectral_unit = unit
            self._view_state.x_axis = "energy_eV" if unit == "meV" else "shift_cm1"
            self._draw_runs(self._last_drawn_runs)
            self._apply_saved_graph_configs()
        elif self._view_state:
            self._view_state.spectral_unit = unit

    @staticmethod
    def _coerce_binning(value: Any) -> int:
        try:
            return max(1, int(value))
        except (TypeError, ValueError):
            return 1

    def get_slice_binning_config(self) -> Dict[str, Any]:
        if not self._view_state:
            return {"x_bin": 1, "y_bin": 1, "mode": "cross"}
        return {
            "x_bin": self._coerce_binning(self._view_state.slice_x_binning),
            "y_bin": self._coerce_binning(self._view_state.slice_y_binning),
            "mode": "box" if self._view_state.slice_binning_mode == "box" else "cross",
        }

    def set_slice_binning(self, x_bin: Optional[int] = None, y_bin: Optional[int] = None, mode: Optional[str] = None) -> None:
        if not self._view_state:
            return
        if x_bin is not None:
            self._view_state.slice_x_binning = self._coerce_binning(x_bin)
        if y_bin is not None:
            self._view_state.slice_y_binning = self._coerce_binning(y_bin)
        if mode is not None:
            self._view_state.slice_binning_mode = "box" if mode == "box" else "cross"
        self._apply_selection_from_indices(reason="binning")

    def get_fit_overlay_mode(self) -> str:
        if not self._view_state:
            return "off"
        return normalize_fit_overlay_mode(self._view_state.fit_overlay_mode)

    def set_fit_overlay_mode(self, mode: str) -> None:
        if not self._view_state:
            return
        mode = normalize_fit_overlay_mode(mode)
        if self._view_state.fit_overlay_mode != mode:
            self._view_state.fit_overlay_mode = mode
            self._apply_selection_from_indices(reason="fit-overlay")

    def _slice_window(self, center: int, width: int, size: int) -> slice:
        width = self._coerce_binning(width)
        if size <= 0:
            return slice(0, 0)
        center = max(0, min(int(center), size - 1))
        left = (width - 1) // 2
        right = width // 2
        start = max(0, center - left)
        stop = min(size, center + right + 1)
        return slice(start, stop)

    def _nanmean_axis(self, values: np.ndarray, axis: int) -> np.ndarray:
        arr = np.asarray(values, dtype=float)
        finite = np.isfinite(arr)
        counts = finite.sum(axis=axis)
        sums = np.where(finite, arr, 0.0).sum(axis=axis)
        result = np.full_like(sums, np.nan, dtype=float)
        np.divide(sums, counts, out=result, where=counts > 0)
        return result

    def _nan_moving_average(self, values: np.ndarray, width: int) -> np.ndarray:
        values = np.asarray(values, dtype=float)
        width = self._coerce_binning(width)
        if width <= 1 or values.size == 0:
            return values
        out = np.empty(values.shape, dtype=float)
        for idx in range(values.size):
            window = self._slice_window(idx, width, values.size)
            chunk = values[window]
            finite = chunk[np.isfinite(chunk)]
            out[idx] = float(np.mean(finite)) if finite.size else np.nan
        return out

    def _slice_data_for_indices(self, intensity: np.ndarray, ix: int, iy: int, *, mode: Optional[str] = None) -> Tuple[np.ndarray, np.ndarray]:
        I = np.asarray(intensity, dtype=float)
        cfg = self.get_slice_binning_config()
        x_bin = cfg["x_bin"]
        y_bin = cfg["y_bin"]
        mode = cfg["mode"] if mode is None else ("box" if mode == "box" else "cross")

        x_window = self._slice_window(ix, x_bin, I.shape[1])
        y_window = self._slice_window(iy, y_bin, I.shape[0])

        angular_slice = self._nanmean_axis(I[:, x_window], axis=1)
        spectral_slice = self._nanmean_axis(I[y_window, :], axis=0)

        if mode == "box":
            angular_slice = self._nan_moving_average(angular_slice, x_bin)
            spectral_slice = self._nan_moving_average(spectral_slice, y_bin)

        return angular_slice, spectral_slice

    def _format_binning_suffix(self, plot_type: str) -> str:
        cfg = self.get_slice_binning_config()
        mode = cfg["mode"]
        x_bin = cfg["x_bin"]
        y_bin = cfg["y_bin"]
        if mode == "box":
            if x_bin == 1 and y_bin == 1:
                return ""
            return f", box {x_bin}x{y_bin}"
        width = x_bin if plot_type == "B" else y_bin
        return "" if width == 1 else f", bin {width}"

    def _highlight_click_allowed(self, event) -> bool:
        if not bool(config.get("highlight_requires_modifier", False)):
            return True
        key = str(getattr(event, "key", "") or "").lower()
        return any(token in key for token in ("control", "ctrl", "cmd", "command", "super", "meta"))

    def _clear_fit_overlay_artists(self, *plotters) -> None:
        for plotter in plotters:
            artists = getattr(plotter, "_fit_overlay_artists", [])
            for artist in artists:
                try:
                    artist.remove()
                except Exception:
                    pass
            plotter._fit_overlay_artists = []

    def _track_fit_overlay_artist(self, plotter, artist) -> None:
        artists = getattr(plotter, "_fit_overlay_artists", [])
        artists.append(artist)
        plotter._fit_overlay_artists = artists

    def _add_fit_overlay_trace(self, plotter, x_data, y_data, *, color: str, style: str, label: str) -> None:
        if plotter is None or x_data is None or y_data is None:
            return
        x_data = np.asarray(x_data, dtype=float)
        y_data = np.asarray(y_data, dtype=float)
        if x_data.size != y_data.size or x_data.size == 0:
            return
        before = len(getattr(plotter.ax, "lines", [])) if getattr(plotter, "ax", None) else 0
        plotter.add_trace(x_data, y_data, color=color, style=style, alpha=0.9, label=label)
        if getattr(plotter, "ax", None) and len(plotter.ax.lines) > before:
            self._track_fit_overlay_artist(plotter, plotter.ax.lines[-1])

    def _apply_fit_overlay(self, run: Run, ix: int, iy: int, pA: RamanPlotter2d, pB: AngularPlotter, pC: SlicePlotter) -> None:
        self._clear_fit_overlay_artists(pA, pB, pC)
        mode = self.get_fit_overlay_mode()
        if mode == "off" or not self._experiment:
            return

        try:
            shift_axis = spectral_axis_for_run(run, self.get_spectral_unit())
            _shift, angles, _display_I = self._display_2d_for_run(run)
            data = overlay_data(self._experiment, run, mode)
        except Exception:
            return
        if data is None or shift_axis is None:
            return

        if pA and pA.ax and data.centers_cm1.size:
            centers_display = cm1_to_unit(data.centers_cm1, self.get_spectral_unit())
            for center in centers_display:
                if not np.isfinite(center):
                    continue
                artist = pA.ax.axvline(center, color="white", linestyle=":", linewidth=1.1, alpha=0.75)
                self._track_fit_overlay_artist(pA, artist)

        if data.global_matrix is not None:
            angular, spectral = self._slice_data_for_indices(data.global_matrix, ix, iy)
            self._add_fit_overlay_trace(pB, angles, angular, color="red", style="-", label="Global fit")
            self._add_fit_overlay_trace(pC, shift_axis, spectral, color="red", style="-", label="Global fit")

        if data.row_matrix is not None:
            angular, spectral = self._slice_data_for_indices(data.row_matrix, ix, iy)
            self._add_fit_overlay_trace(pB, angles, angular, color="#1f77b4", style="--", label="Row fit")
            self._add_fit_overlay_trace(pC, shift_axis, spectral, color="#1f77b4", style="--", label="Row fit")

    def _uses_spectral_x(self, key: str) -> bool:
        return key.endswith(("A", "C"))

    def _stored_xlim_for_axis(self, key: str, xlim):
        if xlim is None:
            return None
        if self._uses_spectral_x(key):
            return spectral_xlim_to_cm1(tuple(float(v) for v in xlim), self.get_spectral_unit())
        return tuple(float(v) for v in xlim)

    def _axis_xlim_from_stored(self, key: str, xlim):
        if xlim is None:
            return None
        if self._uses_spectral_x(key):
            return spectral_xlim_from_cm1(tuple(float(v) for v in xlim), self.get_spectral_unit())
        return tuple(float(v) for v in xlim)

    def _spectral_x_conversion(self):
        unit = self.get_spectral_unit()
        other = alternate_spectral_unit(unit)
        if unit == "meV":
            forward = lambda x: unit_to_cm1(x, "meV")
            inverse = lambda x: cm1_to_unit(x, "meV")
        else:
            forward = lambda x: cm1_to_unit(x, "meV")
            inverse = lambda x: unit_to_cm1(x, "meV")
        return (forward, inverse), spectral_axis_label(other)

    def _plotter_for_graph_key(self, key: str):
        for graph_key, plotter in self._iter_graph_plotters():
            if graph_key == key:
                return plotter
        return None

    def _save_graph_config(self, key: str, plotter) -> None:
        if not self._view_state or plotter is None or not getattr(plotter, "ax", None):
            return
        ax = plotter.ax
        if not ax.has_data() and key.endswith(("B", "C")):
            return

        cfg = self._view_state.get_graph_config(key)
        cfg.xlim = self._stored_xlim_for_axis(key, ax.get_xlim())
        cfg.ylim = tuple(float(v) for v in ax.get_ylim())

        if key.endswith("A") and getattr(plotter, "mesh", None) is not None:
            cfg.vmin = float(self._contrast_percent[0])
            cfg.vmax = float(self._contrast_percent[1])
            cmap = plotter.mesh.get_cmap()
            cfg.cmap = cmap.name if cmap is not None else self.current_cmap

            # Mirror primary map settings into legacy fields for old experiments.
            if key == "1A":
                self._view_state.xlim = cfg.xlim
                self._view_state.ylim = cfg.ylim
                self._view_state.vmin = cfg.vmin
                self._view_state.vmax = cfg.vmax
                self._view_state.cmap = cfg.cmap or self.current_cmap

    def _save_all_graph_configs(self) -> None:
        if self._suppress_config_save:
            return
        for key, plotter in self._iter_graph_plotters():
            self._save_graph_config(key, plotter)

    def save_current_plot_config(self) -> None:
        """Persist the live axes/colormap settings into the current ViewState."""
        self._save_all_graph_configs()

    def _apply_saved_graph_configs(self) -> None:
        if not self._view_state:
            return

        self._view_state.seed_legacy_graph_configs()
        self._suppress_config_save = True
        old_sync = self._syncing_limits
        self._syncing_limits = True
        try:
            for key, plotter in self._iter_graph_plotters():
                if plotter is None or not getattr(plotter, "ax", None):
                    continue
                cfg = self._view_state.graph_configs.get(key)
                if cfg is None:
                    continue

                ax = plotter.ax
                if cfg.xlim is not None and not (key.endswith("B") and self.angle_slice_type == "polar"):
                    ax.set_xlim(self._axis_xlim_from_stored(key, cfg.xlim))
                if cfg.ylim is not None:
                    ax.set_ylim(cfg.ylim)

                if key.endswith("A") and getattr(plotter, "mesh", None) is not None:
                    cmap = cfg.cmap or self.current_cmap
                    if cmap:
                        try:
                            plotter.mesh.set_cmap(cmap)
                        except Exception:
                            pass
                    if cfg.clim is not None:
                        plotter.set_clim(cfg.clim[0], cfg.clim[1])
                    elif cfg.vmin is not None and cfg.vmax is not None:
                        plotter.set_contrast(cfg.vmin, cfg.vmax)
        finally:
            self._syncing_limits = old_sync
            self._suppress_config_save = False

    def get_plot_limits(self):
        plotter = self._plotter_for_graph_key(self._primary_graph_key())
        if plotter and plotter.ax:
            return plotter.ax.get_xlim(), plotter.ax.get_ylim()
        return (None, None), (None, None)

    def set_x_range(self, xmin, xmax, unit):
        # We delegate unit conversion handling to the caller or do it here.
        # The plotter expects whatever unit it was rendered with (usually cm-1).
        target_key = self._primary_graph_key()
        target = self._plotter_for_graph_key(target_key)
        if self.current_run and self.current_run.is_2d:
            target_key = "1A"
            target = self.plotterA1

        if not target or not target.ax:
            return

        target_unit = self.get_spectral_unit()
        input_unit = normalize_spectral_unit(unit)
        if self._uses_spectral_x(target_key) and input_unit != target_unit:
            x_cm1 = unit_to_cm1(np.asarray([xmin, xmax], dtype=float), input_unit)
            x_axis = cm1_to_unit(x_cm1, target_unit)
            xmin, xmax = float(x_axis[0]), float(x_axis[1])

        target.ax.set_xlim(xmin, xmax)
        self._save_all_graph_configs()
        self.canvas.draw_idle()

    def set_y_range(self, ymin, ymax):
        target = self._plotter_for_graph_key(self._primary_graph_key())
        if self.current_run and self.current_run.is_2d:
            target = self.plotterA1

        if target and target.ax:
            target.ax.set_ylim(ymin, ymax)
            self._save_all_graph_configs()
            self.canvas.draw_idle()

    def set_vlim(self, vmin, vmax):
        self._contrast_percent = (vmin, vmax)
        if self._view_state:
            self._view_state.vmin = vmin
            self._view_state.vmax = vmax
            for key in ("1A", "2A"):
                cfg = self._view_state.get_graph_config(key)
                cfg.vmin = float(vmin)
                cfg.vmax = float(vmax)
                cfg.clim = None

        if self.plotterA1:
            self.plotterA1.set_contrast(vmin, vmax)
        if self.plotterA2:
            self.plotterA2.set_contrast(vmin, vmax)

        # Update absolute value displays in config panel
        if self.plot_config_panel:
            v_abs = self.get_absolute_vlim_for_percentiles(vmin, vmax)
            self.plot_config_panel.update_absolute_vlim_display(v_abs[0], v_abs[1])

    def set_vlim_absolute(self, vmin, vmax):
        """Set absolute intensity limits for contrast."""
        if self._view_state:
            for key in ("1A", "2A"):
                cfg = self._view_state.get_graph_config(key)
                cfg.clim = (float(vmin), float(vmax))
                cfg.vmin = self._contrast_percent[0]
                cfg.vmax = self._contrast_percent[1]
        if self.plotterA1:
            self.plotterA1.set_clim(vmin, vmax)
        if self.plotterA2:
            self.plotterA2.set_clim(vmin, vmax)
        self.canvas.draw_idle()

    def get_roi_vlim(self) -> Tuple[float, float]:
        """Calculate min/max values within the current Region of Interest (ROI)."""
        lims = []
        if self.plotterA1:
            lims.append(self.plotterA1.get_roi_limits())
        if self.plotterA2:
            lims.append(self.plotterA2.get_roi_limits())

        if not lims:
            return 0.0, 1.0

        vmin = min(l[0] for l in lims)
        vmax = max(l[1] for l in lims)
        return vmin, vmax

    def get_absolute_vlim_for_percentiles(self, p_min: float, p_max: float) -> Tuple[float, float]:
        """Get absolute intensity values corresponding to given percentiles."""
        if not self.plotterA1:
            return 0.0, 1.0
        # Use Plot 1A as reference
        v_min = self.plotterA1.get_value_at_percentile(p_min)
        v_max = self.plotterA1.get_value_at_percentile(p_max)
        return v_min, v_max

    def set_colormap(self, cmap_name: str):
        self.current_cmap = cmap_name
        if self._view_state:
            self._view_state.cmap = cmap_name
            for key in ("1A", "2A"):
                self._view_state.get_graph_config(key).cmap = cmap_name

        if self.plotterA1 and self.plotterA1.mesh:
            self.plotterA1.mesh.set_cmap(cmap_name)
        if self.plotterA2 and self.plotterA2.mesh:
            self.plotterA2.mesh.set_cmap(cmap_name)
        self.canvas.draw_idle()

    def refresh_current_colormap(self) -> None:
        """Re-apply the currently selected colormap after its definition changes."""
        self.set_colormap(self.get_colormap())

    def get_colormap(self) -> str:
        if self._view_state and self._view_state.cmap:
            return self._view_state.cmap
        if self.plotterA1 and self.plotterA1.mesh:
            return self.plotterA1.mesh.get_cmap().name
        return 'OrRd'

    def _capture_pixel_size(self, plotter, long_edge_px: int = 1200) -> Tuple[int, int]:
        """Use the live panel aspect ratio and a fixed long edge for bitmap exports."""
        ax = getattr(plotter, "ax", None)
        if ax is None:
            return long_edge_px, long_edge_px

        try:
            self.canvas.draw()
            renderer = self.figure.canvas.get_renderer()
            bbox = ax.get_window_extent(renderer)
            width = float(bbox.width)
            height = float(bbox.height)
        except Exception:
            width, height = 4.0, 3.0

        if width <= 1 or height <= 1:
            width, height = 4.0, 3.0

        scale = long_edge_px / max(width, height)
        width_px = int(round(width * scale))
        height_px = int(round(height * scale))
        width_px = max(64, min(2400, width_px))
        height_px = max(64, min(2400, height_px))
        return width_px, height_px

    def _new_capture_figure(self, plotter, *, dpi: int = 200, data_only: bool = False):
        width_px, height_px = self._capture_pixel_size(plotter)
        fig = Figure(figsize=(width_px / dpi, height_px / dpi), dpi=dpi)
        FigureCanvasAgg(fig)

        source_ax = getattr(plotter, "ax", None)
        projection = "polar" if getattr(source_ax, "name", "") == "polar" else None
        if data_only:
            ax = fig.add_axes([0, 0, 1, 1], projection=projection)
            ax.set_axis_off()
        else:
            ax = fig.add_subplot(1, 1, 1, projection=projection)
        return fig, ax

    def _visible_map_region(self, plotter):
        data_raw = getattr(plotter, "_data", None)
        x_edges_raw = getattr(plotter, "_x_edges", None)
        y_edges_raw = getattr(plotter, "_y_edges", None)
        if data_raw is None or x_edges_raw is None or y_edges_raw is None:
            return None

        data = np.asarray(data_raw, dtype=float)
        x_edges = np.asarray(x_edges_raw, dtype=float)
        y_edges = np.asarray(y_edges_raw, dtype=float)
        if data.ndim != 2 or x_edges.ndim != 1 or y_edges.ndim != 1:
            return None
        if data.shape[0] == 0 or data.shape[1] == 0:
            return None
        if x_edges.size != data.shape[1] + 1 or y_edges.size != data.shape[0] + 1:
            return None

        xlim = plotter.ax.get_xlim()
        ylim = plotter.ax.get_ylim()
        xlo, xhi = sorted(float(v) for v in xlim)
        ylo, yhi = sorted(float(v) for v in ylim)

        ix0 = max(0, int(np.searchsorted(x_edges, xlo, side="right") - 1))
        ix0 = min(data.shape[1] - 1, ix0)
        ix1 = min(data.shape[1], int(np.searchsorted(x_edges, xhi, side="left")))
        iy0 = max(0, int(np.searchsorted(y_edges, ylo, side="right") - 1))
        iy0 = min(data.shape[0] - 1, iy0)
        iy1 = min(data.shape[0], int(np.searchsorted(y_edges, yhi, side="left")))

        if ix1 <= ix0:
            ix1 = min(data.shape[1], ix0 + 1)
        if iy1 <= iy0:
            iy1 = min(data.shape[0], iy0 + 1)

        return x_edges[ix0:ix1 + 1], y_edges[iy0:iy1 + 1], data[iy0:iy1, ix0:ix1]

    def _is_helper_line(self, plotter, line) -> bool:
        return line in {
            getattr(plotter, "_vline", None),
            getattr(plotter, "_vline_polar", None),
            getattr(plotter, "_hline", None),
        }

    def _line_data_for_capture(self, source_ax, line, *, trim_to_view: bool):
        x = np.asarray(line.get_xdata(orig=False))
        y = np.asarray(line.get_ydata(orig=False))
        if x.ndim != 1 or y.ndim != 1 or x.size != y.size:
            return x, y

        if trim_to_view and getattr(source_ax, "name", "") != "polar" and x.size > 2:
            try:
                x_numeric = x.astype(float)
                xlo, xhi = sorted(float(v) for v in source_ax.get_xlim())
                mask = (x_numeric >= xlo) & (x_numeric <= xhi)
                if np.any(mask):
                    idx = np.flatnonzero(mask)
                    start = max(0, int(idx[0]) - 1)
                    stop = min(x.size, int(idx[-1]) + 2)
                    x = x[start:stop]
                    y = y[start:stop]
            except (TypeError, ValueError):
                pass

        return x, y

    def _copy_line_artist(self, target_ax, source_ax, line, *, trim_to_view: bool = False) -> None:
        x, y = self._line_data_for_capture(source_ax, line, trim_to_view=trim_to_view)
        target_ax.plot(
            x,
            y,
            color=line.get_color(),
            linewidth=line.get_linewidth(),
            linestyle=line.get_linestyle(),
            marker=line.get_marker(),
            markersize=line.get_markersize(),
            markerfacecolor=line.get_markerfacecolor(),
            markeredgecolor=line.get_markeredgecolor(),
            alpha=line.get_alpha(),
            drawstyle=line.get_drawstyle(),
            label=line.get_label(),
        )

    def _copy_axes_view(self, target_ax, source_ax, *, include_text: bool) -> None:
        if getattr(source_ax, "name", "") == "polar":
            target_ax.set_theta_zero_location("N")
            target_ax.set_theta_direction(-1)
            target_ax.set_xlim(source_ax.get_xlim())
            target_ax.set_ylim(source_ax.get_ylim())
        else:
            target_ax.set_xlim(source_ax.get_xlim())
            target_ax.set_ylim(source_ax.get_ylim())

        if include_text:
            target_ax.set_title(source_ax.get_title())
            target_ax.set_xlabel(source_ax.get_xlabel())
            target_ax.set_ylabel(source_ax.get_ylabel())

    def _render_bitmap_data_bytes(self, plotter, dpi: int = 200) -> bytes:
        fig, ax = self._new_capture_figure(plotter, dpi=dpi, data_only=True)
        source_ax = getattr(plotter, "ax", None)

        if isinstance(plotter, RamanPlotter2d) and getattr(plotter, "mesh", None) is not None:
            region = self._visible_map_region(plotter)
            if region is None:
                raise RuntimeError("Could not extract the visible map data region.")
            x_edges, y_edges, data = region
            mesh = ax.pcolormesh(
                x_edges,
                y_edges,
                data,
                shading="auto",
                cmap=plotter.mesh.get_cmap(),
            )
            mesh.set_clim(*plotter.mesh.get_clim())
            ax.set_xlim(source_ax.get_xlim())
            ax.set_ylim(source_ax.get_ylim())
        else:
            if source_ax is None:
                raise RuntimeError("Could not find a source plot for bitmap capture.")
            self._copy_axes_view(ax, source_ax, include_text=False)
            for line in source_ax.get_lines():
                if line.get_visible() and not self._is_helper_line(plotter, line):
                    self._copy_line_artist(ax, source_ax, line, trim_to_view=True)

        buf = io.BytesIO()
        fig.savefig(
            buf,
            format="png",
            dpi=dpi,
            facecolor="white",
            edgecolor="white",
            pad_inches=0,
        )
        return buf.getvalue()

    def _render_vector_panel_bytes(self, plotter, fmt: str, dpi: int = 200) -> bytes:
        source_ax = getattr(plotter, "ax", None)
        if source_ax is None:
            raise RuntimeError("Could not find a source plot for vector capture.")

        fig, ax = self._new_capture_figure(plotter, dpi=dpi, data_only=False)
        self._copy_axes_view(ax, source_ax, include_text=True)

        for line in source_ax.get_lines():
            if line.get_visible():
                self._copy_line_artist(ax, source_ax, line, trim_to_view=True)

        if source_ax.get_legend() is not None:
            ax.legend()

        fig.tight_layout(pad=0.35)
        buf = io.BytesIO()
        with mpl.rc_context({"svg.fonttype": "none", "pdf.fonttype": 42, "ps.fonttype": 42}):
            fig.savefig(
                buf,
                format=fmt,
                dpi=dpi,
                bbox_inches="tight",
                facecolor="white",
                edgecolor="white",
                metadata={"Creator": "Venkata"},
            )
        return buf.getvalue()

    def _copy_bitmap_bytes_to_clipboard(self, png_bytes: bytes) -> None:
        image = wx.Image(io.BytesIO(png_bytes), wx.BITMAP_TYPE_PNG)
        if not image.IsOk():
            raise RuntimeError("Could not render bitmap for clipboard.")
        bitmap = wx.Bitmap(image)
        data = wx.BitmapDataObject(bitmap)
        if not wx.TheClipboard.Open():
            raise RuntimeError("Could not open the system clipboard.")
        try:
            wx.TheClipboard.SetData(data)
            wx.TheClipboard.Flush()
        finally:
            wx.TheClipboard.Close()

    def _copy_vector_bytes_to_clipboard(self, svg_bytes: bytes, pdf_bytes: bytes) -> None:
        data = wx.DataObjectComposite()

        svg_obj = wx.CustomDataObject(wx.DataFormat("image/svg+xml"))
        svg_obj.SetData(svg_bytes)
        data.Add(svg_obj, True)

        pdf_obj = wx.CustomDataObject(wx.DataFormat("application/pdf"))
        pdf_obj.SetData(pdf_bytes)
        data.Add(pdf_obj)

        data.Add(wx.TextDataObject(svg_bytes.decode("utf-8", errors="replace")))

        if not wx.TheClipboard.Open():
            raise RuntimeError("Could not open the system clipboard.")
        try:
            wx.TheClipboard.SetData(data)
            wx.TheClipboard.Flush()
        finally:
            wx.TheClipboard.Close()

    def _copy_panel_to_clipboard(self, plotter, mode: str) -> None:
        try:
            if mode == "bitmap":
                self._copy_bitmap_bytes_to_clipboard(self._render_bitmap_data_bytes(plotter))
            else:
                svg_bytes = self._render_vector_panel_bytes(plotter, "svg")
                pdf_bytes = self._render_vector_panel_bytes(plotter, "pdf")
                self._copy_vector_bytes_to_clipboard(svg_bytes, pdf_bytes)
        except Exception as exc:
            wx.MessageBox(f"Failed to copy panel: {exc}", "Copy Plot", wx.OK | wx.ICON_ERROR)

    def get_plot_config(self) -> Dict[str, Any]:
        # Prefer live limits from plot if available
        xlim, ylim = self.get_plot_limits()
        graph_key = self._primary_graph_key()

        # Fallback to ViewState if live limits are invalid (e.g. plot not drawn yet)
        if (xlim[0] is None or ylim[0] is None) and self._view_state:
            cfg = self._view_state.graph_configs.get(graph_key)
            if cfg:
                if cfg.xlim: xlim = self._axis_xlim_from_stored(graph_key, cfg.xlim)
                if cfg.ylim: ylim = cfg.ylim
            if xlim[0] is None and self._view_state.xlim:
                xlim = self._axis_xlim_from_stored(graph_key, self._view_state.xlim)
            if ylim[0] is None and self._view_state.ylim:
                ylim = self._view_state.ylim

        if xlim[0] is None: xlim = (0, 1)
        if ylim[0] is None: ylim = (0, 1)

        vmin = self._contrast_percent[0]
        vmax = self._contrast_percent[1]

        if self._view_state:
            cfg = self._view_state.graph_configs.get("1A")
            vmin = cfg.vmin if cfg and cfg.vmin is not None else self._view_state.vmin
            vmax = cfg.vmax if cfg and cfg.vmax is not None else self._view_state.vmax

        slice_cfg = self.get_slice_binning_config()
        return {
            'graph': graph_key,
            'unit': self.get_spectral_unit(),
            'xlim': xlim,
            'ylim': ylim,
            'vmin_p': vmin,
            'vmax_p': vmax,
            'cmap': self.get_colormap(),
            'slice_x_binning': slice_cfg["x_bin"],
            'slice_y_binning': slice_cfg["y_bin"],
            'slice_binning_mode': slice_cfg["mode"],
            'fit_overlay_mode': self.get_fit_overlay_mode(),
        }

    # --- Internal Logic ---

    def _notify_limits_changed(self):
        self._save_all_graph_configs()
        if self.plot_config_panel and self.plotterA1 and self.plotterA1.ax:
            xlim_display, ylim = self.plotterA1.ax.get_xlim(), self.plotterA1.ax.get_ylim()
            if xlim_display and xlim_display[0] is not None:
                self.plot_config_panel.set_x_range(xlim_display[0], xlim_display[1])

            if ylim and ylim[0] is not None:
                self.plot_config_panel.set_y_range(ylim[0], ylim[1])


    def _disconnect_limit_sync_callbacks(self) -> None:
        if not getattr(self, "_limit_cb_ids", None):
            return
        for registry, cid in self._limit_cb_ids:
            try:
                registry.disconnect(cid)
            except Exception:
                pass
        self._limit_cb_ids.clear()

    def _rebind_limit_sync_callbacks(self) -> None:
        """
        Re-attach axis synchronization callbacks.
        Note: We access .ax directly from the plotters.
        """
        self._disconnect_limit_sync_callbacks()

        # Helpers to get axes safely
        axA1 = self.plotterA1.ax if self.plotterA1 else None
        axB1 = self.plotterB1.ax if self.plotterB1 else None
        axC1 = self.plotterC1.ax if self.plotterC1 else None

        axA2 = self.plotterA2.ax if self.plotterA2 else None
        axB2 = self.plotterB2.ax if self.plotterB2 else None
        axC2 = self.plotterC2.ax if self.plotterC2 else None

        # ---------------------------------------------------------
        # Sync Logic Groups
        # ---------------------------------------------------------

        # Group 1: Shift (X-axis) -> A1, C1, A2, C2
        def sync_shift(source_ax):
            if self._syncing_limits: return
            self._syncing_limits = True
            try:
                xlim = source_ax.get_xlim()

                # Apply to A1, C1
                if axA1 and axA1 != source_ax: axA1.set_xlim(xlim)
                if axC1 and axC1 != source_ax: axC1.set_xlim(xlim)

                # Apply to A2, C2
                if axA2 and axA2 != source_ax: axA2.set_xlim(xlim)
                if axC2 and axC2 != source_ax: axC2.set_xlim(xlim)

            finally:
                self._syncing_limits = False
            self._notify_limits_changed()

        # Group 2: Angle -> A1(y), A2(y), B1(x), B2(x)
        def sync_angle(source_ax, is_y_axis=True):
            if self._syncing_limits: return
            self._syncing_limits = True
            try:
                # Get the canonical angle range
                if is_y_axis:
                    angle_lim = source_ax.get_ylim()
                else:
                    angle_lim = source_ax.get_xlim()

                # Apply to A1 (Y), A2 (Y)
                if axA1 and axA1 != source_ax: axA1.set_ylim(angle_lim)
                if axA2 and axA2 != source_ax: axA2.set_ylim(angle_lim)

                # Apply to B1 (X), B2 (X) - ONLY if not polar
                if self.angle_slice_type != "polar":
                    if axB1 and axB1 != source_ax: axB1.set_xlim(angle_lim)
                    if axB2 and axB2 != source_ax: axB2.set_xlim(angle_lim)

            finally:
                self._syncing_limits = False
            self._notify_limits_changed()

        # Group 3: Intensity/Radial Scale - REMOVED to avoid wrongful sync between B and C
        # If we want to sync intensity between Run 1 and Run 2 on the SAME plot type (e.g. C1 and C2),
        # that would be fine, but cross-plot sync (B vs C) is often confusing as slices might have different scales.

        # ---------------------------------------------------------
        # Connect Callbacks
        # ---------------------------------------------------------

        def save_only(ax):
            if self._syncing_limits:
                return
            self._save_all_graph_configs()

        # Run 1
        if axA1:
            # A1 X -> Shift
            cid = axA1.callbacks.connect("xlim_changed", sync_shift)
            self._limit_cb_ids.append((axA1.callbacks, cid))
            # A1 Y -> Angle
            cid = axA1.callbacks.connect("ylim_changed", lambda ax: sync_angle(ax, is_y_axis=True))
            self._limit_cb_ids.append((axA1.callbacks, cid))

        if axC1:
            # C1 X -> Shift
            cid = axC1.callbacks.connect("xlim_changed", sync_shift)
            self._limit_cb_ids.append((axC1.callbacks, cid))
            # C1 Y -> Intensity (Independent)
            cid = axC1.callbacks.connect("ylim_changed", save_only)
            self._limit_cb_ids.append((axC1.callbacks, cid))

        if axB1:
            if self.angle_slice_type != "polar":
                # B1 X -> Angle
                cid = axB1.callbacks.connect("xlim_changed", lambda ax: sync_angle(ax, is_y_axis=False))
                self._limit_cb_ids.append((axB1.callbacks, cid))
            # B1 Y -> Intensity (Independent)
            cid = axB1.callbacks.connect("ylim_changed", save_only)
            self._limit_cb_ids.append((axB1.callbacks, cid))

        # Run 2
        if axA2:
            # A2 X -> Shift
            cid = axA2.callbacks.connect("xlim_changed", sync_shift)
            self._limit_cb_ids.append((axA2.callbacks, cid))
            # A2 Y -> Angle
            cid = axA2.callbacks.connect("ylim_changed", lambda ax: sync_angle(ax, is_y_axis=True))
            self._limit_cb_ids.append((axA2.callbacks, cid))

        if axC2:
            # C2 X -> Shift
            cid = axC2.callbacks.connect("xlim_changed", sync_shift)
            self._limit_cb_ids.append((axC2.callbacks, cid))
            cid = axC2.callbacks.connect("ylim_changed", save_only)
            self._limit_cb_ids.append((axC2.callbacks, cid))

        if axB2:
            if self.angle_slice_type != "polar":
                # B2 X -> Angle
                cid = axB2.callbacks.connect("xlim_changed", lambda ax: sync_angle(ax, is_y_axis=False))
                self._limit_cb_ids.append((axB2.callbacks, cid))
            cid = axB2.callbacks.connect("ylim_changed", save_only)
            self._limit_cb_ids.append((axB2.callbacks, cid))


    def _init_empty_figure(self, message: str = "No data") -> None:
        self.figure.clf()
        ax = self.figure.add_subplot(1, 1, 1)
        ax.text(0.5, 0.5, message, ha="center", va="center", transform=ax.transAxes)
        ax.set_axis_off()

        self.plotterA1 = self.plotterB1 = self.plotterC1 = None
        self.plotterA2 = self.plotterB2 = self.plotterC2 = None

        self._disconnect_limit_sync_callbacks()

        if self._cid_click:
            self.canvas.mpl_disconnect(self._cid_click)
            self._cid_click = None
        if self._cid_resize:
            self.canvas.mpl_disconnect(self._cid_resize)
            self._cid_resize = None

    def _apply_selection_from_indices(self, *, reason: str = "") -> None:
        if not self.plotterA1 or self._sel_idx1 is None:
            return

        # Prevent sync callbacks from propagating the "reset" that happens during render()
        self._syncing_limits = True
        try:
            def _update_run_plots(run: Run, ix: int, iy: int, pA: RamanPlotter2d, pB: AngularPlotter, pC: SlicePlotter):
                if run.run_type == RunType.DERIVED:
                    # Derived Run Logic (Formula)
                    run_config = self._view_state.get_run_config(run.id) if self._view_state else None
                    if not run_config: return

                    n_points = run_config.derived_n_points or run.metadata.get("default_n_points", 100)
                    autorange = run_config.derived_autorange
                    if autorange is None: autorange = run.metadata.get("default_autorange", False)

                    def get_eval_x(target_ax, default_range):
                        if autorange and target_ax:
                            xlim = target_ax.get_xlim()
                            return np.linspace(xlim[0], xlim[1], n_points)
                        else:
                            rng = run_config.derived_range or default_range or (0, 100)
                            return np.linspace(rng[0], rng[1], n_points)

                    x_unit = run.metadata.get("raw_x_unit", "")

                    if "deg" not in x_unit: # Assume Spectral
                        run_unit = normalize_spectral_unit(x_unit, self.get_spectral_unit())
                        if autorange and pC.ax:
                            x_c = get_eval_x(pC.ax, run.metadata.get("default_range"))
                            eval_x = cm1_to_unit(unit_to_cm1(x_c, self.get_spectral_unit()), run_unit)
                        else:
                            eval_x = get_eval_x(pC.ax, run.metadata.get("default_range"))
                            x_c = cm1_to_unit(unit_to_cm1(eval_x, run_unit), self.get_spectral_unit())
                        y_c = run.evaluate(eval_x)
                        styleC = run_config.get_style("C")
                        conversion, secondary_label = self._spectral_x_conversion()
                        show_secondary = True if self._view_state is None else self._view_state.show_secondary_unit_axis
                        pC.render(
                            x_c, y_c,
                            title=f"{run.nickname} (Derived)",
                            xlabel=spectral_axis_label(self.get_spectral_unit()),
                            x_unit_conversion=conversion if show_secondary else None,
                            secondary_x_label=secondary_label,
                            style=styleC,
                        )
                        if pB.ax: pB.ax.clear(); pB.ax.set_axis_off()
                        if pA.ax: pA.ax.clear(); pA.ax.set_axis_off()

                    else: # Assume Angular
                        x_b = get_eval_x(pB.ax, run.metadata.get("default_range"))
                        y_b = run.evaluate(x_b)
                        styleB = run_config.get_style("B")
                        pB.render(x_b, y_b, mode=self.angle_slice_type, title=f"{run.nickname} (Derived)", style=styleB)
                        if pC.ax: pC.ax.clear(); pC.ax.set_axis_off()
                        if pA.ax: pA.ax.clear(); pA.ax.set_axis_off()
                    return

                if run.run_type == RunType.RUN_1D:
                    # Static 1D Run Logic (Extracted or Imported)
                    run_config = self._view_state.get_run_config(run.id) if self._view_state else None

                    # Determine X and Y
                    # Try standard fields
                    x_data = None
                    x_label = ""
                    y_data = run.intensity

                    if run.shift_cm1 is not None:
                        x_data = run.shift_cm1
                        x_label = "shift"
                    elif run.angle_values is not None:
                        x_data = run.angle_values
                        x_label = "angle"
                    elif run.wl_nm is not None:
                        x_data = run.wl_nm
                        x_label = "nm"
                    elif run.energy_eV is not None:
                        x_data = run.energy_eV
                        x_label = "eV"

                    # Fallback to metadata hint if array is None but metadata says what it is
                    # (This handles the case where from_arrays put x in shift_cm1 but we need to match it)
                    # Actually from_arrays populates one of the arrays.

                    if x_data is None: return # Can't plot

                    if x_label == "angle" or "deg" in run.metadata.get("raw_x_unit", ""):
                        # Plot on B
                        styleB = run_config.get_style("B") if run_config else None
                        pB.render(x_data, y_data, mode=self.angle_slice_type, title=f"{run.nickname} (1D)", style=styleB)
                        if pC.ax: pC.ax.clear(); pC.ax.set_axis_off()
                        if pA.ax: pA.ax.clear(); pA.ax.set_axis_off()
                    else:
                        # Plot on C (default for spectral or unknown)
                        x_data = spectral_axis_for_run(run, self.get_spectral_unit())
                        if x_data is None:
                            return
                        styleC = run_config.get_style("C") if run_config else None
                        conversion, secondary_label = self._spectral_x_conversion()
                        show_secondary = True if self._view_state is None else self._view_state.show_secondary_unit_axis
                        pC.render(
                            x_data, y_data,
                            title=f"{run.nickname} (1D)",
                            xlabel=spectral_axis_label(self.get_spectral_unit()),
                            x_unit_conversion=conversion if show_secondary else None,
                            secondary_x_label=secondary_label,
                            style=styleC,
                        )
                        if pB.ax: pB.ax.clear(); pB.ax.set_axis_off()
                        if pA.ax: pA.ax.clear(); pA.ax.set_axis_off()
                    return

                # Standard 2D Run Logic
                # 1. Get Data
                shift_cm1, angles, I = self._display_2d_for_run(run)
                shift_axis = spectral_axis_for_run(run, self.get_spectral_unit())
                if shift_axis is None:
                    return

                # 2. Get coords from index
                x_sel, y_sel = pA.get_coords_from_index(ix, iy)

                # 3. Update Highlights
                pA.set_highlight(x_sel, y_sel, visible=True)
                pB.set_highlight(y_sel, visible=True)
                pC.set_highlight(x_sel, visible=True)

                # 4. Update Slice Data
                # Get style from view_state
                run_config = self._view_state.get_run_config(run.id) if self._view_state else None
                styleB = run_config.get_style("B") if run_config else None
                styleC = run_config.get_style("C") if run_config else None
                angular_slice, spectral_slice = self._slice_data_for_indices(I, ix, iy)

                # Plot B: Intensity vs Angle at selected Shift (column ix)
                if ix < I.shape[1]:
                    x_title = f"{x_sel:.2f} {self.get_spectral_unit()}"
                    pB.render(angles, angular_slice, mode=self.angle_slice_type,
                              title=f"Angular Slice @ {x_title}{self._format_binning_suffix('B')}",
                              style=styleB)

                # Plot C: Intensity vs Shift at selected Angle (row iy)
                if iy < I.shape[0]:
                    # SlicePlotter creates a new line in render(), which is efficient enough here
                    conversion, secondary_label = self._spectral_x_conversion()
                    show_secondary = True if self._view_state is None else self._view_state.show_secondary_unit_axis
                    pC.render(shift_axis, spectral_slice,
                              title=f"Spectral Slice @ {y_sel:.1f} deg{self._format_binning_suffix('C')}",
                              xlabel=spectral_axis_label(self.get_spectral_unit()),
                              x_unit_conversion=conversion if show_secondary else None,
                              secondary_x_label=secondary_label,
                              style=styleC)

                self._apply_fit_overlay(run, ix, iy, pA, pB, pC)

                # 5. Restore Zoom (Sync limits from A to B/C)
                # Since render() cleared the axes, we must re-apply the current zoom from pA.
                if pA.ax:
                    # Sync C (Shift)
                    if pC.ax:
                        pC.ax.set_xlim(pA.ax.get_xlim())

                    # Sync B (Angle) if Cartesian
                    if self.angle_slice_type != "polar" and pB.ax:
                        pB.ax.set_xlim(pA.ax.get_ylim())

            # Update Run 1
            _update_run_plots(self.current_run, self._sel_idx1[0], self._sel_idx1[1],
                              self.plotterA1, self.plotterB1, self.plotterC1)

            # Update Run 2
            if self._second_run and self.plotterA2:
                # Sync indices if needed
                if self._sel_idx2 is None:
                    # Naive sync: same indices (assuming same grid)
                    # Or physical sync? Let's do physical sync logic
                    # For now, let's reuse idx1 if idx2 is missing, assuming close grids
                    self._sel_idx2 = self._sel_idx1

                _update_run_plots(self._second_run, self._sel_idx2[0], self._sel_idx2[1],
                                  self.plotterA2, self.plotterB2, self.plotterC2)

            # --- Overlays ---
            if self._view_state and self._experiment:
                for rid in self._view_state.run_ids:
                    r_overlay = self._experiment.get_run(rid)
                    if not r_overlay: continue

                    cfg = self._view_state.get_run_config(rid)
                    target_str = cfg.overlay_target
                    if not target_str: continue

                    # Handle multiple targets (comma-separated)
                    targets = [t.strip() for t in target_str.split(",") if t.strip()]

                    for target in targets:
                        # Identify target plotter and reference selection
                        target_plotter = None
                        ref_run = None
                        ref_idx = None

                        if target == "1B":
                            target_plotter = self.plotterB1
                            ref_run = self.current_run
                            ref_idx = self._sel_idx1
                        elif target == "1C":
                            target_plotter = self.plotterC1
                            ref_run = self.current_run
                            ref_idx = self._sel_idx1
                        elif target == "2B":
                            target_plotter = self.plotterB2
                            ref_run = self._second_run
                            ref_idx = self._sel_idx2
                        elif target == "2C":
                            target_plotter = self.plotterC2
                            ref_run = self._second_run
                            ref_idx = self._sel_idx2

                        if not target_plotter or not ref_run or not ref_idx:
                            continue

                        # Overlay logic
                        x_data = None
                        y_data = None
                        style = cfg.get_style("Overlay")
                        label = f"{r_overlay.nickname}"

                        # --- DERIVED RUNS ---
                        if r_overlay.run_type == RunType.DERIVED:
                             # Use auto-range from target plotter or internal config
                             n_points = cfg.derived_n_points or r_overlay.metadata.get("default_n_points", 100)
                             autorange = cfg.derived_autorange
                             if autorange is None: autorange = r_overlay.metadata.get("default_autorange", False)
                             overlay_unit = normalize_spectral_unit(r_overlay.metadata.get("raw_x_unit", self.get_spectral_unit()), self.get_spectral_unit())

                             if autorange and target_plotter.ax:
                                 xlim = target_plotter.ax.get_xlim()
                                 x_data = np.linspace(xlim[0], xlim[1], n_points)
                                 eval_x = cm1_to_unit(unit_to_cm1(x_data, self.get_spectral_unit()), overlay_unit) if "C" in target else x_data
                             else:
                                 rng = cfg.derived_range or r_overlay.metadata.get("default_range", (0, 100))
                                 eval_x = np.linspace(rng[0], rng[1], n_points)
                                 x_data = cm1_to_unit(unit_to_cm1(eval_x, overlay_unit), self.get_spectral_unit()) if "C" in target else eval_x

                             y_data = r_overlay.evaluate(eval_x)
                             label += " (D)"

                        # --- STATIC 1D RUNS ---
                        elif r_overlay.run_type == RunType.RUN_1D:
                            y_data = r_overlay.intensity
                            # Infer X based on target plot type
                            if "B" in target: # Angular plot (Angle vs Intensity)
                                if r_overlay.angle_values is not None: x_data = r_overlay.angle_values
                                elif r_overlay.shift_cm1 is not None and "angle" in r_overlay.metadata.get("raw_x_unit",""): x_data = r_overlay.shift_cm1 # Fallback
                            elif "C" in target: # Spectral plot (Shift vs Intensity)
                                if r_overlay.shift_cm1 is not None: x_data = spectral_axis_for_run(r_overlay, self.get_spectral_unit())
                                elif r_overlay.angle_values is not None and "cm" in r_overlay.metadata.get("raw_x_unit",""): x_data = r_overlay.angle_values # Fallback

                        # --- STANDARD 2D RUNS (Slicing) ---
                        else:
                            # Get physical coords from reference selection
                            if ref_run.shift_cm1 is not None and ref_run.angle_values is not None:
                                ref_shift_axis, ref_angles, _ref_I = self._display_2d_for_run(ref_run)
                                ref_shift = ref_shift_axis[ref_idx[0]]
                                ref_angle = ref_angles[ref_idx[1]]

                                if (
                                    r_overlay.shift_cm1 is not None
                                    and r_overlay.angle_values is not None
                                    and r_overlay.intensity_2d is not None
                                ):
                                    ov_shift, ov_angles, ov_I = self._display_2d_for_run(r_overlay)
                                    idx_shift = int(np.abs(ov_shift - ref_shift).argmin())
                                    idx_angle = int(np.abs(ov_angles - ref_angle).argmin())
                                    angular_slice, spectral_slice = self._slice_data_for_indices(ov_I, idx_shift, idx_angle)

                                    if "B" in target: # Angular slice overlay
                                        x_data = ov_angles
                                        y_data = angular_slice
                                        display_val = float(cm1_to_unit(ov_shift[idx_shift], self.get_spectral_unit()))
                                        label += f" @ {display_val:.1f} {self.get_spectral_unit()}"
                                    elif "C" in target: # Spectral slice overlay
                                        x_data = spectral_axis_for_run(r_overlay, self.get_spectral_unit())
                                        y_data = spectral_slice
                                        label += f" @ {ov_angles[idx_angle]:.1f}"

                        # Plot if data is valid
                        if x_data is not None and y_data is not None and style:
                            if x_data.size != y_data.size:
                                # Safety for 2D slicing mismatch
                                if y_data.ndim > 1:
                                     if x_data.size == y_data.shape[0]: y_data = y_data[:, 0] # or similar
                                     elif x_data.size == y_data.shape[1]: y_data = y_data[0, :]

                            if x_data.size == y_data.size:
                                target_plotter.add_trace(
                                    x_data, y_data,
                                    color=style.color,
                                    style=style.linestyle,
                                    alpha=style.alpha,
                                    label=label,
                                    marker=style.marker,
                                    markersize=style.markersize
                                )

            # Post-render: Synchronize Radial/Intensity Scale for Plot B
            if self.plotterB1 and self.plotterB1.ax and self.plotterB2 and self.plotterB2.ax:
                ylim1 = self.plotterB1.ax.get_ylim()
                ylim2 = self.plotterB2.ax.get_ylim()
                common_max = max(ylim1[1], ylim2[1])
                self.plotterB1.ax.set_ylim(0, common_max)
                self.plotterB2.ax.set_ylim(0, common_max)

        finally:
            self._rebind_limit_sync_callbacks()
            self._syncing_limits = False

        self.figure.subplots_adjust(top=0.862)
        self.canvas.draw_idle()

    def _draw_runs(self, runs: List[Run]) -> None:
        if not runs:
            self._init_empty_figure(message="No runs to draw.")
            return

        runs = runs[:2]
        self._last_drawn_runs = list(runs)
        self.figure.clf()

        # Layout
        gs = self.figure.add_gridspec(
            2, 3, height_ratios=[1.0, 1.0], width_ratios=[2.0, 1.0, 1.0],
            hspace=0.6, wspace=0.5, left=0.1, right=0.95
        )

        spectral_unit = self.get_spectral_unit()
        conversion, secondary_label = self._spectral_x_conversion()
        show_secondary = True if self._view_state is None else self._view_state.show_secondary_unit_axis

        # --- Run 1 ---
        run0 = runs[0]
        axA1 = self.figure.add_subplot(gs[0, 0])
        axB1 = self.figure.add_subplot(gs[0, 1], projection="polar" if self.angle_slice_type == "polar" else None)
        axC1 = self.figure.add_subplot(gs[0, 2])

        self.plotterA1 = RamanPlotter2d(axA1)
        self.plotterB1 = AngularPlotter(axB1)
        self.plotterC1 = SlicePlotter(axC1)

        nickname0 = self._experiment.get_run_nickname(run0.id) if self._experiment else run0.nickname
        x0 = spectral_axis_for_run(run0, spectral_unit)
        _shift0, angles0, intensity0 = self._display_2d_for_run(run0)
        self.plotterA1.render(
            x0, angles0, intensity0,
            title=f"{nickname0}: 2D map",
            xlabel=spectral_axis_label(spectral_unit),
            cmap=self.current_cmap,
            x_unit_conversion=conversion if show_secondary else None,
            secondary_x_label=secondary_label,
        )
        self.plotterA1.set_highlight_mode(self.highlight_mode)

        # --- Run 2 ---
        if len(runs) > 1:
            run1 = runs[1]
            axA2 = self.figure.add_subplot(gs[1, 0])
            axB2 = self.figure.add_subplot(gs[1, 1], projection="polar" if self.angle_slice_type == "polar" else None)
            axC2 = self.figure.add_subplot(gs[1, 2])

            self.plotterA2 = RamanPlotter2d(axA2)
            self.plotterB2 = AngularPlotter(axB2)
            self.plotterC2 = SlicePlotter(axC2)

            nickname1 = self._experiment.get_run_nickname(run1.id) if self._experiment else run1.nickname
            x1 = spectral_axis_for_run(run1, spectral_unit)
            _shift1, angles1, intensity1 = self._display_2d_for_run(run1)
            self.plotterA2.render(
                x1, angles1, intensity1,
                title=f"{nickname1}: 2D map",
                xlabel=spectral_axis_label(spectral_unit),
                cmap=self.current_cmap,
                x_unit_conversion=conversion if show_secondary else None,
                secondary_x_label=secondary_label,
            )
            self.plotterA2.set_highlight_mode(self.highlight_mode)
        else:
            self.plotterA2 = self.plotterB2 = self.plotterC2 = None
            # Fill empty space
            self.figure.add_subplot(gs[1, 0]).set_axis_off()
            self.figure.add_subplot(gs[1, 1]).set_axis_off()
            self.figure.add_subplot(gs[1, 2]).set_axis_off()

        # Connect Callbacks
        self._rebind_limit_sync_callbacks()

        if self._cid_click: self.canvas.mpl_disconnect(self._cid_click)
        self._cid_click = self.canvas.mpl_connect("button_press_event", self._on_canvas_click)

        self.figure.subplots_adjust(top=0.862)
        self._apply_selection_from_indices()
        self._apply_saved_graph_configs()
        self.canvas.draw_idle()

    def _on_canvas_click(self, event):
        # 0. Handle Right Click on Background (outside axes)
        if event.button == 3 and event.inaxes is None:
            menu = wx.Menu()
            highlight_menu = wx.Menu()
            modes = ["none", "click", "line_profile", "peak_fit"]
            for m in modes:
                item = highlight_menu.AppendRadioItem(wx.ID_ANY, m.capitalize())
                if self.highlight_mode == m: item.Check(True)
                self.Bind(wx.EVT_MENU, lambda e, mode=m: self._set_highlight_mode(mode), item)
            menu.AppendSubMenu(highlight_menu, "Highlight")
            self.PopupMenu(menu)
            menu.Destroy()
            return

        if event.inaxes is None:
            return

        # Determine which plotter was clicked
        clicked_plotter = None
        is_run1 = False

        if self.plotterA1 and event.inaxes == self.plotterA1.ax:
            clicked_plotter = self.plotterA1
            is_run1 = True
        elif self.plotterA2 and event.inaxes == self.plotterA2.ax:
            clicked_plotter = self.plotterA2
            is_run1 = False
        elif self.plotterB1 and event.inaxes == self.plotterB1.ax:
            clicked_plotter = self.plotterB1
            is_run1 = True
        elif self.plotterB2 and event.inaxes == self.plotterB2.ax:
            clicked_plotter = self.plotterB2
            is_run1 = False
        elif self.plotterC1 and event.inaxes == self.plotterC1.ax:
            clicked_plotter = self.plotterC1
            is_run1 = True
        elif self.plotterC2 and event.inaxes == self.plotterC2.ax:
            clicked_plotter = self.plotterC2
            is_run1 = False

        if not clicked_plotter:
            return

        # 1. Update selection indices if we clicked on Plot A
        if (
            event.button == 1
            and clicked_plotter in [self.plotterA1, self.plotterA2]
            and self._highlight_click_allowed(event)
        ):
            indices = clicked_plotter.get_index_at(event.xdata, event.ydata)
            if indices:
                ix, iy = indices
                if is_run1:
                    self._sel_idx1 = (ix, iy)
                    self._sel_idx2 = (ix, iy) # Simple sync
                else:
                    self._sel_idx2 = (ix, iy)
                    self._sel_idx1 = (ix, iy)
                self._apply_selection_from_indices(reason="click")

        # 2. Handle Right Click Menu
        if event.button == 3:
            menu = wx.Menu()

            # Helper to add Highlight mode submenu
            highlight_menu = wx.Menu()
            modes = ["none", "click", "line_profile", "peak_fit"]
            for m in modes:
                item = highlight_menu.AppendRadioItem(wx.ID_ANY, m.capitalize())
                if self.highlight_mode == m: item.Check(True)
                self.Bind(wx.EVT_MENU, lambda e, mode=m: self._set_highlight_mode(mode), item)
            menu.AppendSubMenu(highlight_menu, "Highlight")
            menu.AppendSeparator()

            item_copy_bitmap = menu.Append(wx.ID_ANY, "Copy data bitmap")
            self.Bind(wx.EVT_MENU, lambda e, p=clicked_plotter: self._copy_panel_to_clipboard(p, "bitmap"), item_copy_bitmap)

            if clicked_plotter in [self.plotterB1, self.plotterB2, self.plotterC1, self.plotterC2]:
                item_copy_vector = menu.Append(wx.ID_ANY, "Copy panel vector")
                self.Bind(wx.EVT_MENU, lambda e, p=clicked_plotter: self._copy_panel_to_clipboard(p, "vector"), item_copy_vector)

            menu.AppendSeparator()

            # Plot B: Angular Slice (Angle vs Intensity)
            if clicked_plotter in [self.plotterB1, self.plotterB2]:
                target_run = self.current_run if is_run1 else self._second_run

                # Submenu for slice type
                sub = wx.Menu()
                item_cart = sub.AppendRadioItem(wx.ID_ANY, "Cartesian")
                item_polar = sub.AppendRadioItem(wx.ID_ANY, "Polar")
                if self.angle_slice_type == "polar": item_polar.Check(True)
                else: item_cart.Check(True)
                self.Bind(wx.EVT_MENU, lambda e: self._set_angle_slice_type("cartesian"), item_cart)
                self.Bind(wx.EVT_MENU, lambda e: self._set_angle_slice_type("polar"), item_polar)
                menu.AppendSubMenu(sub, "Angle slice type")

                menu.AppendSeparator()
                item_create = menu.Append(wx.ID_ANY, "Create separate run")
                self.Bind(wx.EVT_MENU, lambda e: self._create_run_from_trace(clicked_plotter, target_run, "B"), item_create)

                item_fit = menu.Append(wx.ID_ANY, "Curve Fit...")
                self.Bind(wx.EVT_MENU, lambda e: self._request_curve_fit(clicked_plotter, target_run, "B"), item_fit)

            # Plot C: Spectral Slice (Shift vs Intensity)
            elif clicked_plotter in [self.plotterC1, self.plotterC2]:
                target_run = self.current_run if is_run1 else self._second_run

                item_create = menu.Append(wx.ID_ANY, "Create separate run")
                self.Bind(wx.EVT_MENU, lambda e: self._create_run_from_trace(clicked_plotter, target_run, "C"), item_create)

                item_fit = menu.Append(wx.ID_ANY, "Curve Fit...")
                self.Bind(wx.EVT_MENU, lambda e: self._request_curve_fit(clicked_plotter, target_run, "C"), item_fit)

            # Plot A: Map
            elif clicked_plotter in [self.plotterA1, self.plotterA2]:
                item_fit = menu.Append(wx.ID_ANY, "Curve Fit (Row-by-Row)...")
                target_run = self.current_run if is_run1 else self._second_run
                self.Bind(wx.EVT_MENU, lambda e: self._request_2d_fit(target_run), item_fit)

            self.PopupMenu(menu)
            menu.Destroy()
            return

    def _request_2d_fit(self, source_run: Run):
        """
        Request a 2D (batch) curve fit.
        """
        if not source_run or not self.on_fit_request:
            return

        try:
            shift, angles, I = self._display_2d_for_run(source_run)
        except Exception:
            return
        shift_display = spectral_axis_for_run(source_run, self.get_spectral_unit())

        if shift is None or shift_display is None or I is None:
            return

        # Pass the full 2D array as y_data
        self.on_fit_request(source_run, "A", shift_display, I, spectral_axis_label(self.get_spectral_unit(), latex=False))

    def _request_curve_fit(self, plotter, source_run: Run, plot_type: str):
        """
        Extract data and request a curve fit operation.
        """
        if not plotter or not source_run or not self.on_fit_request:
            return

        # Logic similar to _create_run_from_trace to extract x/y
        shift = None
        angles = None
        I = None
        if source_run.is_2d:
            try:
                shift, angles, I = self._display_2d_for_run(source_run)
            except Exception:
                shift, angles, I = None, None, None
        shift_display = spectral_axis_for_run(source_run, self.get_spectral_unit())

        x_data, y_data = None, None

        # 2D Case
        if I is not None and shift is not None and angles is not None:
             selection = self._sel_idx1 if source_run == self.current_run else self._sel_idx2
             if selection is None:
                 return
             ix, iy = selection
             angular_slice, spectral_slice = self._slice_data_for_indices(I, ix, iy)

             if plot_type == "B" and ix < I.shape[1]:
                 x_data = angles
                 y_data = angular_slice
             elif plot_type == "C" and iy < I.shape[0]:
                 x_data = shift_display
                 y_data = spectral_slice

        # 1D Case (if user clicked on a 1D run overlay) - Wait, overlays don't trigger this menu yet.
        # The menu is attached to the AXES click.
        # The logic `target_run = self.current_run` picks the PRIMARY run.
        # If the primary run is 1D, we handle it here.
        elif source_run.run_type == RunType.RUN_1D:
             y_data = source_run.intensity
             if plot_type == "C":
                 x_data = spectral_axis_for_run(source_run, self.get_spectral_unit())
             elif plot_type == "B":
                 x_data = source_run.angle_values

        if x_data is not None and y_data is not None:
            x_label = "Angle (deg)" if plot_type == "B" else spectral_axis_label(self.get_spectral_unit(), latex=False)
            self.on_fit_request(source_run, plot_type, x_data, y_data, x_label)

    def _create_run_from_trace(self, plotter, source_run: Run, plot_type: str):
        """
        Extract data from the plotter's active trace and create a new Run.
        plot_type: "B" (Angular) or "C" (Spectral)
        """
        if not plotter or not source_run:
            return

        # We need to grab the data currently being displayed.
        # Plotters usually have 'ax.lines' or similar.
        # But our plotters (AngularPlotter, SlicePlotter) don't expose data directly easily
        # unless we stored it.
        # However, we know what data is displayed based on self._sel_idx1 / self._sel_idx2.

        # Re-derive the data logic from _apply_selection_from_indices
        shift, angles, I = self._display_2d_for_run(source_run)
        shift_display = spectral_axis_for_run(source_run, self.get_spectral_unit())
        if shift_display is None:
            shift_display = shift

        selection = self._sel_idx1 if source_run == self.current_run else self._sel_idx2
        if selection is None:
            return
        ix, iy = selection
        angular_slice, spectral_slice = self._slice_data_for_indices(I, ix, iy)

        x_data, y_data = None, None
        x_label, y_label = "", "Intensity (au)"
        nickname = ""

        if plot_type == "B": # Angular Slice @ fixed shift (ix)
            if ix >= I.shape[1]: return
            # x = angles, y = intensity
            x_data = angles
            y_data = angular_slice
            val = shift[ix]
            nickname = f"{source_run.nickname}_ang_{val:.1f}cm-1"
            x_label = "Angle (deg)"

        elif plot_type == "C": # Spectral Slice @ fixed angle (iy)
            if iy >= I.shape[0]: return
            # x = shift, y = intensity
            x_data = shift_display
            y_data = spectral_slice
            val = angles[iy]
            nickname = f"{source_run.nickname}_spec_{val:.1f}deg"
            x_label = spectral_axis_label(self.get_spectral_unit(), latex=False)

        if x_data is not None and y_data is not None:
            new_run = Run.from_arrays(x_data, y_data, x_label=x_label, y_label=y_label, nickname=nickname)
            if self.on_run_created:
                self.on_run_created(new_run)
            else:
                # Fallback: try adding to experiment directly if possible, though update won't happen
                if self._experiment:
                    self._experiment.add_run(new_run)
                    wx.MessageBox(f"Created run {nickname}, but UI might not refresh.", "Info")

    def _popup_angle_slice_type_menu(self, mpl_event) -> None:
        menu = wx.Menu()
        sub = wx.Menu()
        item_cart = sub.AppendRadioItem(wx.ID_ANY, "Cartesian")
        item_polar = sub.AppendRadioItem(wx.ID_ANY, "Polar")

        if self.angle_slice_type == "polar": item_polar.Check(True)
        else: item_cart.Check(True)

        self.Bind(wx.EVT_MENU, lambda e: self._set_angle_slice_type("cartesian"), item_cart)
        self.Bind(wx.EVT_MENU, lambda e: self._set_angle_slice_type("polar"), item_polar)
        menu.AppendSubMenu(sub, "Angle slice type")

        self.PopupMenu(menu)
        menu.Destroy()

    def _set_angle_slice_type(self, mode: str) -> None:
        if mode == self.angle_slice_type: return
        self.angle_slice_type = mode
        if self._view_state:
            self._view_state.angle_slice_type = mode
        # Re-draw everything because axes projection needs to change
        self._draw_runs(self._last_drawn_runs)
        self._apply_selection_from_indices()

    def _set_highlight_mode(self, mode: str):
        self.highlight_mode = mode
        if self.plotterA1: self.plotterA1.set_highlight_mode(mode)
        if self.plotterA2: self.plotterA2.set_highlight_mode(mode)
        self.canvas.draw_idle()

    def set_view_model(self, experiment: ExperimentSet, view_state: ViewState, preserve_state: bool = False) -> None:
        # 1. Determine if we should preserve interactive state (Zoom, Selection)
        # We preserve if requested AND the primary runs currently displayed match the new ones.

        new_ids = []
        # Pre-calculate which runs will be loaded
        for rid in view_state.run_ids:
            r = experiment.runs.get(rid)
            if r:
                new_ids.append(r.id)
                if len(new_ids) == 2: break

        should_preserve = False
        if preserve_state:
            old_ids = []
            if self.current_run: old_ids.append(self.current_run.id)
            if self._second_run: old_ids.append(self._second_run.id)

            should_preserve = (len(old_ids) > 0) and (old_ids == new_ids)

        saved_xlim = (None, None)
        saved_ylim = (None, None)
        saved_sel1 = None
        saved_sel2 = None
        saved_unit = self.get_spectral_unit()

        if should_preserve:
            saved_xlim, saved_ylim = self.get_plot_limits()
            saved_sel1 = self._sel_idx1
            saved_sel2 = self._sel_idx2

        # 2. Reset / Load Model
        self._experiment = experiment
        self._view_state = view_state
        self._view_state.seed_legacy_graph_configs()
        self.angle_slice_type = view_state.angle_slice_type
        self.current_run_id = None
        self.current_run = None
        self._second_run = None

        # Reset indices initially (will override if preserving)
        self._sel_idx1 = (0, 0)
        self._sel_idx2 = (0, 0)

        runs: List[Run] = []
        for rid in new_ids:
            runs.append(experiment.runs[rid])

        if not runs:
            self._init_empty_figure("No runs selected.")
            self.canvas.draw_idle()
            return

        self.current_run = runs[0]
        self.current_run_id = runs[0].id
        self._second_run = runs[1] if len(runs) > 1 else None

        # 3. Restore Config from ViewState
        # Colormap and Contrast are typically driven by the panel/view_state
        map_cfg = view_state.graph_configs.get("1A")
        self.current_cmap = (
            map_cfg.cmap if map_cfg and map_cfg.cmap else
            view_state.cmap if view_state.cmap else "OrRd"
        )
        if map_cfg and map_cfg.vmin is not None and map_cfg.vmax is not None:
            self._contrast_percent = (map_cfg.vmin, map_cfg.vmax)
        elif view_state.vmin is not None and view_state.vmax is not None:
            self._contrast_percent = (view_state.vmin, view_state.vmax)

        # 4. Restore Selection Indices (before draw)
        if should_preserve:
            if saved_sel1: self._sel_idx1 = saved_sel1
            if saved_sel2: self._sel_idx2 = saved_sel2

        self._draw_runs(runs)

        # 5. Restore Zoom Limits
        # Priority: Preserved Live Limits > per-graph persisted limits applied by _draw_runs.
        if should_preserve and saved_xlim[0] is not None:
             self.set_x_range(saved_xlim[0], saved_xlim[1], unit=saved_unit)

        if should_preserve and saved_ylim[0] is not None:
             self.set_y_range(saved_ylim[0], saved_ylim[1])

        # Apply contrast unless an absolute color scale was saved for the map.
        if not (map_cfg and map_cfg.clim is not None):
            self.set_vlim(self._contrast_percent[0], self._contrast_percent[1])
        self._apply_saved_graph_configs()
