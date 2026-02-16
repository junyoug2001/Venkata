import os
import analysis
from merge_runs_gui import MergeRunsDialog
from typing import List, Optional, Dict, Any, Tuple

import wx
import numpy as np

import matplotlib.ticker as mticker
from matplotlib.figure import Figure
from matplotlib.backends.backend_wxagg import (
    FigureCanvasWxAgg as FigureCanvas,
    NavigationToolbar2WxAgg as NavigationToolbar,
)

from data_structure import (
    ExperimentSet,
    Run,
    ViewState,
    EV_PER_CM1,
    new_experiment_id,
    new_view_id,
    RunType
)
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
        self.canvas.Bind(wx.EVT_CONTEXT_MENU, self._on_context_menu)

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
        if self.current_run.shift_cm1 is not None:
            xmin = np.min(self.current_run.shift_cm1)
            xmax = np.max(self.current_run.shift_cm1)
        
        # 2. Angle Range (Y-axis of Plot A, X-axis of Plot B)
        ymin, ymax = 0.0, 1.0
        if self.current_run.angle_values is not None:
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

    def get_plot_limits(self):
        if self.plotterA1 and self.plotterA1.ax:
            return self.plotterA1.ax.get_xlim(), self.plotterA1.ax.get_ylim()
        return (None, None), (None, None)

    def set_x_range(self, xmin, xmax, unit):
        # We delegate unit conversion handling to the caller or do it here.
        # The plotter expects whatever unit it was rendered with (usually cm-1).
        if not self.plotterA1 or not self.plotterA1.ax:
            return
        
        # If input is meV, convert to cm-1 if that's the base unit
        if unit == 'meV':
            xmin = xmin / EV_PER_CM1 / 1000.0
            xmax = xmax / EV_PER_CM1 / 1000.0
        
        self.plotterA1.ax.set_xlim(xmin, xmax)
        self.canvas.draw_idle()

    def set_y_range(self, ymin, ymax):
        if self.plotterA1 and self.plotterA1.ax:
            self.plotterA1.ax.set_ylim(ymin, ymax)
            self.canvas.draw_idle()

    def set_vlim(self, vmin, vmax):
        self._contrast_percent = (vmin, vmax)
        if self._view_state:
            self._view_state.vmin = vmin
            self._view_state.vmax = vmax
            
        if self.plotterA1:
            self.plotterA1.set_contrast(vmin, vmax)
        if self.plotterA2:
            self.plotterA2.set_contrast(vmin, vmax)

    def set_colormap(self, cmap_name: str):
        self.current_cmap = cmap_name
        if self._view_state:
            self._view_state.cmap = cmap_name
            
        if self.plotterA1 and self.plotterA1.mesh:
            self.plotterA1.mesh.set_cmap(cmap_name)
        if self.plotterA2 and self.plotterA2.mesh:
            self.plotterA2.mesh.set_cmap(cmap_name)
        self.canvas.draw_idle()

    def get_colormap(self) -> str:
        if self._view_state and self._view_state.cmap:
            return self._view_state.cmap
        if self.plotterA1 and self.plotterA1.mesh:
            return self.plotterA1.mesh.get_cmap().name
        return 'OrRd'

    def get_plot_config(self) -> Dict[str, Any]:
        # Prefer live limits from plot if available
        xlim, ylim = self.get_plot_limits()
        
        # Fallback to ViewState if live limits are invalid (e.g. plot not drawn yet)
        if (xlim[0] is None or ylim[0] is None) and self._view_state:
            if self._view_state.xlim: xlim = self._view_state.xlim
            if self._view_state.ylim: ylim = self._view_state.ylim
            
        if xlim[0] is None: xlim = (0, 1)
        if ylim[0] is None: ylim = (0, 1)
        
        vmin = self._contrast_percent[0]
        vmax = self._contrast_percent[1]
        
        if self._view_state:
            vmin = self._view_state.vmin
            vmax = self._view_state.vmax
        
        return {
            'xlim': xlim,
            'ylim': ylim,
            'vmin_p': vmin,
            'vmax_p': vmax,
            'cmap': self.get_colormap()
        }

    # --- Internal Logic ---

    def _notify_limits_changed(self):
        if self.plot_config_panel and self.plotterA1 and self.plotterA1.ax:
            xlim_cm1, ylim = self.plotterA1.ax.get_xlim(), self.plotterA1.ax.get_ylim()
            
            current_unit = self.plot_config_panel.get_x_unit()
            if xlim_cm1 and xlim_cm1[0] is not None:
                if current_unit == 'meV':
                    xlim_display = (xlim_cm1[0] * EV_PER_CM1 * 1000, xlim_cm1[1] * EV_PER_CM1 * 1000)
                else: # cm-1
                    xlim_display = xlim_cm1
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
        
        if axB1:
            if self.angle_slice_type != "polar":
                # B1 X -> Angle
                cid = axB1.callbacks.connect("xlim_changed", lambda ax: sync_angle(ax, is_y_axis=False))
                self._limit_cb_ids.append((axB1.callbacks, cid))
            # B1 Y -> Intensity (Independent)

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
            
        if axB2:
            if self.angle_slice_type != "polar":
                # B2 X -> Angle
                cid = axB2.callbacks.connect("xlim_changed", lambda ax: sync_angle(ax, is_y_axis=False))
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
                        x_c = get_eval_x(pC.ax, run.metadata.get("default_range"))
                        y_c = run.evaluate(x_c)
                        styleC = run_config.get_style("C")
                        pC.render(x_c, y_c, title=f"{run.nickname} (Derived)", style=styleC)
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
                        styleC = run_config.get_style("C") if run_config else None
                        pC.render(x_data, y_data, title=f"{run.nickname} (1D)", style=styleC)
                        if pB.ax: pB.ax.clear(); pB.ax.set_axis_off()
                        if pA.ax: pA.ax.clear(); pA.ax.set_axis_off()
                    return

                # Standard 2D Run Logic
                # 1. Get Data
                shift = np.asarray(run.shift_cm1, dtype=float)
                angles = np.asarray(run.angle_values, dtype=float)
                I = np.asarray(run.intensity_2d, dtype=float)
                
                # Ensure orientation (transpose if needed to match axes)
                if I.shape != (angles.size, shift.size):
                    if I.shape == (shift.size, angles.size):
                        I = I.T
                    else:
                        return # Data mismatch

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

                # Plot B: Intensity vs Angle at selected Shift (column ix)
                if ix < I.shape[1]:
                    pB.render(angles, I[:, ix], mode=self.angle_slice_type, 
                              title=f"Angular Slice @ {x_sel:.1f} cm$^{{-1}}$",
                              style=styleB)
                
                # Plot C: Intensity vs Shift at selected Angle (row iy)
                if iy < I.shape[0]:
                    # SlicePlotter creates a new line in render(), which is efficient enough here
                    pC.render(shift, I[iy, :], 
                              title=f"Spectral Slice @ {y_sel:.1f} deg",
                              style=styleC)

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
                             
                             if autorange and target_plotter.ax:
                                 xlim = target_plotter.ax.get_xlim()
                                 x_data = np.linspace(xlim[0], xlim[1], n_points)
                             else:
                                 rng = cfg.derived_range or r_overlay.metadata.get("default_range", (0, 100))
                                 x_data = np.linspace(rng[0], rng[1], n_points)
                             
                             y_data = r_overlay.evaluate(x_data)
                             label += " (D)"

                        # --- STATIC 1D RUNS ---
                        elif r_overlay.run_type == RunType.RUN_1D:
                            y_data = r_overlay.intensity
                            # Infer X based on target plot type
                            if "B" in target: # Angular plot (Angle vs Intensity)
                                if r_overlay.angle_values is not None: x_data = r_overlay.angle_values
                                elif r_overlay.shift_cm1 is not None and "angle" in r_overlay.metadata.get("raw_x_unit",""): x_data = r_overlay.shift_cm1 # Fallback
                            elif "C" in target: # Spectral plot (Shift vs Intensity)
                                if r_overlay.shift_cm1 is not None: x_data = r_overlay.shift_cm1
                                elif r_overlay.angle_values is not None and "cm" in r_overlay.metadata.get("raw_x_unit",""): x_data = r_overlay.angle_values # Fallback

                        # --- STANDARD 2D RUNS (Slicing) ---
                        else:
                            # Get physical coords from reference selection
                            if ref_run.shift_cm1 is not None and ref_run.angle_values is not None:
                                ref_shift = ref_run.shift_cm1[ref_idx[0]]
                                ref_angle = ref_run.angle_values[ref_idx[1]]

                                if "B" in target: # Angular slice overlay
                                    if r_overlay.shift_cm1 is not None and r_overlay.intensity_2d is not None:
                                        idx_ov = (np.abs(r_overlay.shift_cm1 - ref_shift)).argmin()
                                        x_data = r_overlay.angle_values
                                        y_data = r_overlay.intensity_2d[:, idx_ov]
                                        label += f" @ {r_overlay.shift_cm1[idx_ov]:.1f}"
                                    
                                elif "C" in target: # Spectral slice overlay
                                    if r_overlay.angle_values is not None and r_overlay.intensity_2d is not None:
                                        idx_ov = (np.abs(r_overlay.angle_values - ref_angle)).argmin()
                                        x_data = r_overlay.shift_cm1
                                        y_data = r_overlay.intensity_2d[idx_ov, :]
                                        label += f" @ {r_overlay.angle_values[idx_ov]:.1f}"

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

        # Helper for secondary axis (cm-1 -> meV)
        def cm_to_mev(x): return x * EV_PER_CM1 * 1000
        def mev_to_cm(x): return x / (EV_PER_CM1 * 1000)
        
        # --- Run 1 ---
        run0 = runs[0]
        axA1 = self.figure.add_subplot(gs[0, 0])
        axB1 = self.figure.add_subplot(gs[0, 1], projection="polar" if self.angle_slice_type == "polar" else None)
        axC1 = self.figure.add_subplot(gs[0, 2])

        self.plotterA1 = RamanPlotter2d(axA1)
        self.plotterB1 = AngularPlotter(axB1)
        self.plotterC1 = SlicePlotter(axC1)

        nickname0 = self._experiment.get_run_nickname(run0.id) if self._experiment else run0.nickname
        self.plotterA1.render(
            run0.shift_cm1, run0.angle_values, run0.intensity_2d,
            title=f"{nickname0}: 2D map",
            cmap=self.current_cmap,
            x_unit_conversion=(cm_to_mev, mev_to_cm)
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
            self.plotterA2.render(
                run1.shift_cm1, run1.angle_values, run1.intensity_2d,
                title=f"{nickname1}: 2D map",
                cmap=self.current_cmap,
                x_unit_conversion=(cm_to_mev, mev_to_cm)
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
        self.canvas.draw_idle()

    def _on_canvas_click(self, event):
        if event.button == 3: # Right click
            # Plot B: Angular Slice (Angle vs Intensity)
            if (self.plotterB1 and event.inaxes == self.plotterB1.ax) or \
               (self.plotterB2 and event.inaxes == self.plotterB2.ax):
                
                # Identify which run/plotter
                target_plotter = self.plotterB1 if event.inaxes == self.plotterB1.ax else self.plotterB2
                target_run = self.current_run if target_plotter == self.plotterB1 else self._second_run
                
                menu = wx.Menu()
                
                # Submenu for slice type
                sub = wx.Menu()
                item_cart = sub.AppendRadioItem(wx.ID_ANY, "Cartesian")
                item_polar = sub.AppendRadioItem(wx.ID_ANY, "Polar")
                if self.angle_slice_type == "polar": item_polar.Check(True)
                else: item_cart.Check(True)
                self.Bind(wx.EVT_MENU, lambda e: self._set_angle_slice_type("cartesian"), item_cart)
                self.Bind(wx.EVT_MENU, lambda e: self._set_angle_slice_type("polar"), item_polar)
                menu.AppendSubMenu(sub, "Angle slice type")
                
                # Item for creating separate run
                menu.AppendSeparator()
                item_create = menu.Append(wx.ID_ANY, "Create separate run")
                self.Bind(wx.EVT_MENU, lambda e: self._create_run_from_trace(target_plotter, target_run, "B"), item_create)
                
                item_fit = menu.Append(wx.ID_ANY, "Curve Fit...")
                self.Bind(wx.EVT_MENU, lambda e: self._request_curve_fit(target_plotter, target_run, "B"), item_fit)
                
                self.PopupMenu(menu)
                menu.Destroy()
                return

            # Plot C: Spectral Slice (Shift vs Intensity)
            if (self.plotterC1 and event.inaxes == self.plotterC1.ax) or \
               (self.plotterC2 and event.inaxes == self.plotterC2.ax):
                
                target_plotter = self.plotterC1 if event.inaxes == self.plotterC1.ax else self.plotterC2
                target_run = self.current_run if target_plotter == self.plotterC1 else self._second_run
                
                menu = wx.Menu()
                item_create = menu.Append(wx.ID_ANY, "Create separate run")
                self.Bind(wx.EVT_MENU, lambda e: self._create_run_from_trace(target_plotter, target_run, "C"), item_create)

                item_fit = menu.Append(wx.ID_ANY, "Curve Fit...")
                self.Bind(wx.EVT_MENU, lambda e: self._request_curve_fit(target_plotter, target_run, "C"), item_fit)
                
                self.PopupMenu(menu)
                menu.Destroy()
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
        
        if not clicked_plotter:
            return

        if event.button == 3: # Right click on Map
            menu = wx.Menu()
            item_fit = menu.Append(wx.ID_ANY, "Curve Fit (Row-by-Row)...")
            target_run = self.current_run if is_run1 else self._second_run
            self.Bind(wx.EVT_MENU, lambda e: self._request_2d_fit(target_run), item_fit)
            self.PopupMenu(menu)
            menu.Destroy()
            return

        # Get indices from plotter
        indices = clicked_plotter.get_index_at(event.xdata, event.ydata)
        if not indices:
            return
        ix, iy = indices

        # Update Indices state
        if is_run1:
            self._sel_idx1 = (ix, iy)
            # Optional: Map to idx2 via physical coords if needed
            self._sel_idx2 = (ix, iy) # Simple sync for now
        else:
            self._sel_idx2 = (ix, iy)
            self._sel_idx1 = (ix, iy) # Simple sync

        self._apply_selection_from_indices(reason="click")

    def _request_2d_fit(self, source_run: Run):
        """
        Request a 2D (batch) curve fit.
        """
        if not source_run or not self.on_fit_request:
            return

        shift = np.asarray(source_run.shift_cm1, dtype=float) if source_run.shift_cm1 is not None else None
        angles = np.asarray(source_run.angle_values, dtype=float) if source_run.angle_values is not None else None
        I = np.asarray(source_run.intensity_2d, dtype=float) if source_run.intensity_2d is not None else None

        if shift is None or I is None:
            return

        # Ensure orientation: rows = angles, cols = shift
        # We want to fit along shift (columns) for each angle (row)
        if I.shape != (angles.size, shift.size):
             if I.shape == (shift.size, angles.size): 
                 I = I.T
             else: 
                 return

        # Pass the full 2D array as y_data
        self.on_fit_request(source_run, "A", shift, I)

    def _request_curve_fit(self, plotter, source_run: Run, plot_type: str):
        """
        Extract data and request a curve fit operation.
        """
        if not plotter or not source_run or not self.on_fit_request:
            return

        # Logic similar to _create_run_from_trace to extract x/y
        shift = np.asarray(source_run.shift_cm1, dtype=float) if source_run.shift_cm1 is not None else None
        angles = np.asarray(source_run.angle_values, dtype=float) if source_run.angle_values is not None else None
        I = np.asarray(source_run.intensity_2d, dtype=float) if source_run.intensity_2d is not None else None

        x_data, y_data = None, None
        
        # 2D Case
        if I is not None and shift is not None and angles is not None:
             if I.shape != (angles.size, shift.size):
                 if I.shape == (shift.size, angles.size): I = I.T
                 else: return

             ix, iy = (self._sel_idx1 if source_run == self.current_run else self._sel_idx2)
             
             if plot_type == "B" and ix < I.shape[1]:
                 x_data = angles
                 y_data = I[:, ix]
             elif plot_type == "C" and iy < I.shape[0]:
                 x_data = shift
                 y_data = I[iy, :]
                 
        # 1D Case (if user clicked on a 1D run overlay) - Wait, overlays don't trigger this menu yet.
        # The menu is attached to the AXES click.
        # The logic `target_run = self.current_run` picks the PRIMARY run.
        # If the primary run is 1D, we handle it here.
        elif source_run.run_type == RunType.RUN_1D:
             y_data = source_run.intensity
             if plot_type == "C":
                 x_data = source_run.shift_cm1
             elif plot_type == "B":
                 x_data = source_run.angle_values

        if x_data is not None and y_data is not None:
            self.on_fit_request(source_run, plot_type, x_data, y_data)

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
        shift = np.asarray(source_run.shift_cm1, dtype=float)
        angles = np.asarray(source_run.angle_values, dtype=float)
        I = np.asarray(source_run.intensity_2d, dtype=float)
        
        if I.shape != (angles.size, shift.size):
             if I.shape == (shift.size, angles.size): I = I.T
             else: return 

        ix, iy = (self._sel_idx1 if source_run == self.current_run else self._sel_idx2)
        
        x_data, y_data = None, None
        x_label, y_label = "", "Intensity (au)"
        nickname = ""
        
        if plot_type == "B": # Angular Slice @ fixed shift (ix)
            if ix >= I.shape[1]: return
            # x = angles, y = intensity
            x_data = angles
            y_data = I[:, ix]
            val = shift[ix]
            nickname = f"{source_run.nickname}_ang_{val:.1f}cm-1"
            x_label = "Angle (deg)"
            
        elif plot_type == "C": # Spectral Slice @ fixed angle (iy)
            if iy >= I.shape[0]: return
            # x = shift, y = intensity
            x_data = shift
            y_data = I[iy, :]
            val = angles[iy]
            nickname = f"{source_run.nickname}_spec_{val:.1f}deg"
            x_label = "Raman shift (cm-1)"

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
        # Re-draw everything because axes projection needs to change
        self._draw_runs(self._last_drawn_runs)
        self._apply_selection_from_indices()

    def _on_context_menu(self, event):
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

        if should_preserve:
            saved_xlim, saved_ylim = self.get_plot_limits()
            saved_sel1 = self._sel_idx1
            saved_sel2 = self._sel_idx2

        # 2. Reset / Load Model
        self._experiment = experiment
        self._view_state = view_state
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
        self.current_cmap = view_state.cmap if view_state.cmap else "OrRd"
        if view_state.vmin is not None and view_state.vmax is not None:
            self._contrast_percent = (view_state.vmin, view_state.vmax)

        # 4. Restore Selection Indices (before draw)
        if should_preserve:
            if saved_sel1: self._sel_idx1 = saved_sel1
            if saved_sel2: self._sel_idx2 = saved_sel2

        self._draw_runs(runs)
        
        # 5. Restore Zoom Limits
        # Priority: Preserved Live Limits > ViewState Persisted Limits > Default (Auto)
        
        # X Limits
        if should_preserve and saved_xlim[0] is not None:
             self.set_x_range(saved_xlim[0], saved_xlim[1], unit="cm-1")
        elif view_state.xlim:
             self.set_x_range(view_state.xlim[0], view_state.xlim[1], unit="cm-1")
             
        # Y Limits
        if should_preserve and saved_ylim[0] is not None:
             self.set_y_range(saved_ylim[0], saved_ylim[1])
        elif view_state.ylim:
             self.set_y_range(view_state.ylim[0], view_state.ylim[1])
            
        # Apply contrast
        self.set_vlim(self._contrast_percent[0], self._contrast_percent[1])
