import analysis
from analysis import MergeDiscoverOptions, MergePreviewOptions, MergeDiscoverResult, MergePreviewResult
from cosmic_review_gui import CosmicReviewDialog
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any, Tuple, Sequence

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
    infer_intensity_unit,
    parse_filename,
)


class MergeRunsToolbar(NavigationToolbar):
    def __init__(self, canvas, owner_dialog: "MergeRunsDialog"):
        self._owner_dialog = owner_dialog
        super().__init__(canvas)

    def home(self, *args, **kwargs):
        super().home(*args, **kwargs)
        if hasattr(self._owner_dialog, "_restore_home_view"):
            self._owner_dialog._restore_home_view()

class MergeRunsDialog(wx.Dialog):
    """Merge-run dialog implementing Stage-1/Stage-2 preview pipeline.

    Stage-1 (Discover):
      - analysis.discover_merge(MergeDiscoverOptions)
      - returns wavelength-axis preview matrix (angle × wavelength)

    Stage-2 (Preview):
      - analysis.preview_merge(discover_result, MergePreviewOptions)
      - optional cosmic correction + Raman-shift axis construction

    GUI responsibilities:
      - Render top image plot (matrix) with click-based highlight
      - Render bottom 1D spectral slice at selected angle (row)
      - Provide vmin/vmax sliders to adjust image contrast
      - Provide right-panel controls to set cosmic/manual-laser options
      - Preview/Revert/Apply flow

    Notes
    -----
    - Native file dialogs cannot host custom checkboxes reliably across platforms.
      All options are therefore collected inside this dialog.
    - This dialog is designed so Stage-1 is always available; Stage-2 is optional.
    """

    def __init__(self, parent: wx.Window, paths: List[str], log_cb=None):
        super().__init__(
            parent,
            title="Merge Runs",
            style=wx.DEFAULT_DIALOG_STYLE | wx.RESIZE_BORDER,
        )

        self._paths = list(paths)
        self._log_cb = log_cb
        self.result_run: Optional[Run] = None

        # -----------------------------
        # Internal state
        # -----------------------------
        self._disc_original: Optional[MergeDiscoverResult] = None
        self._disc_active: Optional[MergeDiscoverResult] = None
        self._disc_cosmic_applied: Optional[MergeDiscoverResult] = None
        self._prev: Optional[MergePreviewResult] = None
        self._in_preview_mode: bool = False

        self._original_raw_xlim: Optional[Tuple[float, float]] = None
        self._original_raw_ylim: Optional[Tuple[float, float]] = None
        self._original_preview_xlim: Optional[Tuple[float, float]] = None
        self._original_preview_ylim: Optional[Tuple[float, float]] = None

        # Selection (indices) in the current displayed matrix
        self._sel_row: int = 0
        self._sel_col: int = 0

        # Matplotlib artists
        self._mesh = None  # QuadMesh from pcolormesh
        self._vline = None
        self._hline = None
        self._laser_vline = None
        self._secax_ev = None

        # Colorbar handle (must be removed/recreated across redraws)
        self._cbar = None

        # Persist contrast settings across Discover/Preview/Revert.
        # Stored as (vmin_percentile, vmax_percentile) in [0, 100].
        self._contrast_percent = (0.0, 100.0)

        # Cached displayed data
        self._x = None  # np.ndarray
        self._y = None  # np.ndarray
        self._I = None  # np.ndarray (ny, nx)
        # For non-uniform grids we render with pcolormesh using bin edges.
        self._x_centers = None  # np.ndarray (nx,)
        self._y_centers = None  # np.ndarray (ny,)
        self._x_edges = None    # np.ndarray (nx+1,)
        self._y_edges = None    # np.ndarray (ny+1,)

        # 50% of parent size
        pw, ph = parent.GetSize()
        self.SetSize((max(980, int(pw * 0.5)), max(680, int(ph * 0.5))))

        root = wx.BoxSizer(wx.HORIZONTAL)
        self.SetSizer(root)

        # ----------------
        # Left: plotting area (3/4)
        # ----------------
        plot_panel = wx.Panel(self)
        plot_sizer = wx.BoxSizer(wx.VERTICAL)
        plot_panel.SetSizer(plot_sizer)

        self.figure = Figure(figsize=(10, 7))
        self.canvas = FigureCanvas(plot_panel, -1, self.figure)
        self.toolbar = MergeRunsToolbar(self.canvas, self)
        self.toolbar.Realize()

        plot_sizer.Add(self.toolbar, 0, wx.EXPAND | wx.ALL, 2)
        plot_sizer.Add(self.canvas, 1, wx.EXPAND | wx.ALL, 2)

        # Two plots stacked
        self.figure.clf()
        gs = self.figure.add_gridspec(2, 1, height_ratios=[3.0, 1.4], hspace=0.35)
        self.ax_top = self.figure.add_subplot(gs[0, 0])
        self.ax_bot = self.figure.add_subplot(gs[1, 0])

        # Click handler on the canvas
        self._cid_click = self.canvas.mpl_connect("button_press_event", self._on_canvas_click)

        # ----------------
        # Right: controls area (1/4)
        # ----------------
        ctrl_panel = wx.Panel(self)
        ctrl_sizer = wx.BoxSizer(wx.VERTICAL)
        ctrl_panel.SetSizer(ctrl_sizer)

        # Info box
        box_info = wx.StaticBoxSizer(wx.StaticBox(ctrl_panel, label="Detected files"), wx.VERTICAL)
        ctrl_sizer.Add(box_info, 1, wx.EXPAND | wx.ALL, 6)

        self.lbl_pattern = wx.StaticText(ctrl_panel, label="Pattern: (not discovered)")
        box_info.Add(self.lbl_pattern, 0, wx.ALL, 4)

        self.lbl_counts = wx.StaticText(ctrl_panel, label="xxxx: -, yyyy: -")
        box_info.Add(self.lbl_counts, 0, wx.ALL, 4)

        box_info.Add(wx.StaticText(ctrl_panel, label="Files:"), 0, wx.LEFT | wx.RIGHT | wx.TOP, 4)
        self.txt_files = wx.TextCtrl(
            ctrl_panel,
            value="\n".join(self._paths),
            style=wx.TE_MULTILINE | wx.TE_READONLY,
        )
        box_info.Add(self.txt_files, 1, wx.EXPAND | wx.ALL, 4)

        # Options box
        box_opt = wx.StaticBoxSizer(wx.StaticBox(ctrl_panel, label="Preview options"), wx.VERTICAL)
        ctrl_sizer.Add(box_opt, 0, wx.EXPAND | wx.LEFT | wx.RIGHT | wx.BOTTOM, 6)

        # Dark value
        row_dark = wx.BoxSizer(wx.HORIZONTAL)
        self.txt_dark_value = wx.TextCtrl(ctrl_panel, value="600", size=(80, -1))
        row_dark.Add(wx.StaticText(ctrl_panel, label="Dark value"), 0, wx.ALL | wx.ALIGN_CENTER_VERTICAL, 2)
        row_dark.Add(self.txt_dark_value, 0, wx.ALL | wx.ALIGN_CENTER_VERTICAL, 2)
        box_opt.Add(row_dark, 0, wx.ALL, 2)

        # Cosmic correction/review (will open a dedicated dialog; no inline checkbox)
        row_cosmic = wx.BoxSizer(wx.HORIZONTAL)
        self.btn_cosmic = wx.Button(ctrl_panel, label="Cosmic...")
        self.btn_cosmic_revert = wx.Button(ctrl_panel, label="Revert cosmic")
        self.btn_cosmic_revert.Enable(False)
        row_cosmic.Add(self.btn_cosmic, 0, wx.ALL, 2)
        row_cosmic.Add(self.btn_cosmic_revert, 0, wx.ALL, 2)
        box_opt.Add(row_cosmic, 0, wx.ALL, 2)

        # Manual laser wavelength
        row_manual = wx.BoxSizer(wx.HORIZONTAL)
        self.chk_manual_laser = wx.CheckBox(ctrl_panel, label="Manual laser")
        self.chk_manual_laser.SetValue(False)
        self.txt_manual_nm = wx.TextCtrl(ctrl_panel, value="", size=(80, -1))
        self.txt_manual_nm.Enable(False)
        lbl_nm = wx.StaticText(ctrl_panel, label="nm")

        row_manual.Add(self.chk_manual_laser, 0, wx.ALL | wx.ALIGN_CENTER_VERTICAL, 2)
        row_manual.Add(self.txt_manual_nm, 0, wx.ALL | wx.ALIGN_CENTER_VERTICAL, 2)
        row_manual.Add(lbl_nm, 0, wx.ALL | wx.ALIGN_CENTER_VERTICAL, 2)
        box_opt.Add(row_manual, 0, wx.ALL, 2)

        # Raman x-axis checkbox
        row_raman = wx.BoxSizer(wx.HORIZONTAL)
        self.chk_raman_x_axis = wx.CheckBox(ctrl_panel, label="Use Raman shift as x-axis")
        self.chk_raman_x_axis.SetValue(False)
        row_raman.Add(self.chk_raman_x_axis, 0, wx.ALIGN_CENTER_VERTICAL | wx.ALL, 2)

        self.rad_raman_unit_cm1 = wx.RadioButton(ctrl_panel, label="cm-1", style=wx.RB_GROUP)
        self.rad_raman_unit_mev = wx.RadioButton(ctrl_panel, label="meV")
        self.rad_raman_unit_cm1.SetValue(True)
        self.rad_raman_unit_cm1.Enable(False)
        self.rad_raman_unit_mev.Enable(False)

        row_raman.Add(self.rad_raman_unit_cm1, 0, wx.ALIGN_CENTER_VERTICAL | wx.ALL, 2)
        row_raman.Add(self.rad_raman_unit_mev, 0, wx.ALIGN_CENTER_VERTICAL | wx.ALL, 2)
        box_opt.Add(row_raman, 0, wx.ALL, 2)

        # Contrast controls (bottom of right panel)
        box_contrast = wx.StaticBoxSizer(wx.StaticBox(ctrl_panel, label="Contrast"), wx.VERTICAL)
        ctrl_sizer.Add(box_contrast, 0, wx.EXPAND | wx.LEFT | wx.RIGHT | wx.BOTTOM, 6)

        self.slider_vmin = wx.Slider(ctrl_panel, minValue=0, maxValue=1000, value=0, style=wx.SL_HORIZONTAL)
        self.slider_vmax = wx.Slider(ctrl_panel, minValue=0, maxValue=1000, value=1000, style=wx.SL_HORIZONTAL)
        box_contrast.Add(wx.StaticText(ctrl_panel, label="vmin"), 0, wx.LEFT | wx.RIGHT | wx.TOP, 4)
        box_contrast.Add(self.slider_vmin, 0, wx.EXPAND | wx.LEFT | wx.RIGHT | wx.BOTTOM, 4)
        box_contrast.Add(wx.StaticText(ctrl_panel, label="vmax"), 0, wx.LEFT | wx.RIGHT | wx.TOP, 4)
        box_contrast.Add(self.slider_vmax, 0, wx.EXPAND | wx.LEFT | wx.RIGHT | wx.BOTTOM, 4)

        # Buttons
        btn_row = wx.BoxSizer(wx.HORIZONTAL)
        self.btn_preview = wx.Button(ctrl_panel, label="Preview")
        self.btn_revert = wx.Button(ctrl_panel, label="Revert")
        self.btn_apply = wx.Button(ctrl_panel, label="Apply")
        self.btn_close = wx.Button(ctrl_panel, id=wx.ID_CANCEL, label="Close")

        self.btn_revert.Enable(False)
        self.btn_apply.Enable(False)

        btn_row.Add(self.btn_preview, 0, wx.ALL, 4)
        btn_row.Add(self.btn_revert, 0, wx.ALL, 4)
        btn_row.AddStretchSpacer(1)
        btn_row.Add(self.btn_apply, 0, wx.ALL, 4)
        btn_row.Add(self.btn_close, 0, wx.ALL, 4)
        ctrl_sizer.Add(btn_row, 0, wx.EXPAND | wx.LEFT | wx.RIGHT | wx.BOTTOM, 6)

        # Root: add panels with 3:1 ratio
        root.Add(plot_panel, 3, wx.EXPAND | wx.ALL, 6)
        root.Add(ctrl_panel, 1, wx.EXPAND | wx.ALL, 6)

        # Bind UI events
        self.Bind(wx.EVT_BUTTON, self._on_preview, self.btn_preview)
        self.Bind(wx.EVT_BUTTON, self._on_revert, self.btn_revert)
        self.Bind(wx.EVT_BUTTON, self._on_apply, self.btn_apply)
        self.Bind(wx.EVT_BUTTON, self._on_cosmic_button, self.btn_cosmic)
        self.Bind(wx.EVT_BUTTON, self._on_cosmic_revert, self.btn_cosmic_revert)

        self.Bind(wx.EVT_CHECKBOX, self._on_toggle_manual, self.chk_manual_laser)
        self.Bind(wx.EVT_CHECKBOX, self._on_toggle_raman_x_axis, self.chk_raman_x_axis)
        self.Bind(wx.EVT_TEXT, self._on_manual_nm_text, self.txt_manual_nm)
        self.Bind(wx.EVT_SLIDER, self._on_contrast_slider, self.slider_vmin)
        self.Bind(wx.EVT_SLIDER, self._on_contrast_slider, self.slider_vmax)
        self.Bind(wx.EVT_TEXT, self._on_dark_value_change, self.txt_dark_value)

        # Run Stage-1 discovery immediately
        self._run_discover()

        self.Layout()

    # -----------------------------
    # Logging helper
    # -----------------------------

    def _log(self, msg: str) -> None:
        if self._log_cb is not None:
            try:
                self._log_cb(msg)
            except Exception:
                pass

    # -----------------------------
    # Stage-1 / Stage-2
    # -----------------------------

    def _run_discover(self) -> None:
        if not self._paths:
            self._show_message("No seed file was provided.")
            return

        seed = self._paths[0]
        try:
            dark_value_str = self.txt_dark_value.GetValue()
            try:
                dark_value = float(dark_value_str)
            except ValueError:
                dark_value = 0.0
            
            self._disc = analysis.discover_merge(
                MergeDiscoverOptions(seed_file=seed, dark_value=dark_value)
            )
            self._disc_original = self._disc
            self._disc_active = self._disc_original
            self._disc_cosmic_applied = None
            self.btn_cosmic_revert.Enable(False)
        except Exception as e:
            self._disc = None
            self._show_message(f"Discover failed: {e}")
            return

        # Update right panel info
        self.lbl_pattern.SetLabel(f"Pattern: {self._disc.pattern_hint}")
        self.lbl_counts.SetLabel(
            f"xxxx: {len(self._disc.unique_xxxx)}, yyyy: {len(self._disc.unique_yyyy)}"
        )
        self.txt_files.SetValue("\n".join(self._disc.files))

        # Set manual laser textbox to the auto-detected candidate value.
        # The user can then enable the checkbox to use/edit it.
        if hasattr(self._disc, "candidate_laser_nm") and self._disc.candidate_laser_nm is not None:
            val_str = f"{self._disc.candidate_laser_nm:.4f}"
            self.txt_manual_nm.SetValue(val_str)

        # Render raw preview (wavelength axis)
        self._prev = None
        self._in_preview_mode = False
        self.btn_revert.Enable(False)
        self.btn_apply.Enable(False)

        self._render_raw_from_discover(self._disc_active)

    def _on_dark_value_change(self, event) -> None:
        if self.ax_top is not None:
            current_xlim = self.ax_top.get_xlim()
            current_ylim = self.ax_top.get_ylim()
        else:
            current_xlim = None
            current_ylim = None

        self._run_discover()

        if current_xlim is not None and current_ylim is not None and self.ax_top is not None:
            self.ax_top.set_xlim(current_xlim)
            self.ax_top.set_ylim(current_ylim)
            self.canvas.draw_idle()



    def _build_preview_options(self) -> Optional[MergePreviewOptions]:
        disc_for_opts = self._disc_active if self._disc_active is not None else self._disc
        if disc_for_opts is None:
            return None

        manual_nm = None
        if self.chk_manual_laser.GetValue():
            s = self.txt_manual_nm.GetValue().strip()
            if not s:
                self._show_message("Manual laser is enabled, but no wavelength is provided.")
                return None
            try:
                manual_nm = float(s)
            except ValueError:
                self._show_message("Manual laser wavelength must be a number (nm).")
                return None

        raw_files = getattr(disc_for_opts, "raw_files", None)
        files = raw_files if raw_files else disc_for_opts.files
        seed_file = files[0] if files else self._paths[0]
        opts = MergePreviewOptions(
            seed_file=seed_file,
            files=list(files),
            cosmic_enable=True,
            manual_laser_nm=manual_nm,
            interactive_confirm=False,
            require_laser_nm=False,
            use_raman_x=self.chk_raman_x_axis.GetValue(),
            raman_x_mode="cm-1" if self.rad_raman_unit_cm1.GetValue() else "meV",
        )
        return opts

    def _on_preview(self, event) -> None:
        if self._disc is None:
            return

        opts = self._build_preview_options()
        if opts is None:
            return


        try:
            disc_for_preview = self._disc_active if self._disc_active is not None else self._disc
            self._prev = analysis.preview_merge(disc_for_preview, opts, prompt=None)
        except ValueError as e:
            if str(e) == "wavelength_nm must be > 0":
                opts.use_raman_x = True
                try:
                    self._prev = analysis.preview_merge(disc_for_preview, opts, prompt=None)
                except Exception as inner_e:
                    self._prev = None
                    self._show_message(f"Preview failed (after retry): {inner_e}")
                    return
            else:
                self._prev = None
                self._show_message(f"Preview failed: {e}")
                return
        except Exception as e:
            self._prev = None
            self._show_message(f"Preview failed: {e}")
            return

        self._in_preview_mode = True
        self.btn_revert.Enable(True)
        self.btn_apply.Enable(True)
        self._render_preview_from_result()

    def _on_revert(self, event) -> None:
        self._prev = None
        self._in_preview_mode = False
        self.btn_revert.Enable(False)
        self.btn_apply.Enable(False)
        self._original_preview_xlim = None
        self._original_preview_ylim = None
        self._render_raw_from_discover(self._disc_active)

    def _on_cosmic_revert(self, event) -> None:
        """Revert the wavelength-angle matrix to the original (pre-cosmic) state.

        Behavior depends on current mode:
        - Raw mode: simply redraw the wavelength-axis matrix.
        - Preview mode: redo preview immediately so the GUI stays consistent with the
          now-reverted wavelength matrix.
        """
        if self._disc_original is None:
            return

        # Restore original wavelength-axis discover state.
        self._disc_active = self._disc_original
        self._disc_cosmic_applied = None
        self.btn_cosmic_revert.Enable(False)

        if not self._in_preview_mode:
            # Raw mode: just redraw wavelength-axis matrix.
            self._render_raw_from_discover(self._disc_active)
            return

        # Preview mode: re-run preview based on the reverted wavelength matrix.
        opts = self._build_preview_options()
        if opts is None:
            # If options are invalid, fall back to raw view.
            self._prev = None
            self._in_preview_mode = False
            self.btn_revert.Enable(False)
            self.btn_apply.Enable(False)
            self._render_raw_from_discover(self._disc_active)
            return

        try:
            self._prev = analysis.preview_merge(self._disc_active, opts, prompt=None)
        except Exception as e:
            # Keep the app in a consistent state: drop preview and show raw.
            self._prev = None
            self._in_preview_mode = False
            self.btn_revert.Enable(False)
            self.btn_apply.Enable(False)
            self._show_message(f"Preview failed after cosmic revert: {e}")
            # Show raw matrix so the user can continue.
            self._render_raw_from_discover(self._disc_active)
            return

        # Successfully re-generated preview.
        self._in_preview_mode = True
        self.btn_revert.Enable(True)
        self.btn_apply.Enable(True)
        self._render_preview_from_result()

    def _on_apply(self, event) -> None:
        if self._prev is None:
            return

        new_run_id = new_experiment_id() # Reusing new_experiment_id for unique run ID
        source_path = self._paths[0] # Using the first path as a representative source path

        metadata = parse_filename(source_path)
        if self._disc is not None:
            metadata["merge_pattern_hint"] = self._disc.pattern_hint
            metadata["merge_unique_xxxx"] = self._disc.unique_xxxx
            metadata["merge_unique_yyyy"] = self._disc.unique_yyyy
            metadata["merged_files"] = self._disc.files
            # Also capture the title for the new run's nickname
            metadata["nickname"] = None

        # Determine intensity unit based on the original data, if possible.
        # For now, default to "au".
        intensity_unit = "au" 
        if self._disc and self._disc.raw_intensity_matrix is not None:
            intensity_unit = infer_intensity_unit(self._disc.raw_intensity_matrix)

        new_run = Run(
            id=new_run_id,
            source_path=source_path,
            source_mtime=None, # Not directly available from MergePreviewResult
            shift_cm1=self._prev.raman_shift_cm1,
            energy_eV=self._prev.energy_ev,
            intensity=None, # This is 2D data, so 1D intensity is None
            intensity_2d=self._prev.intensity_matrix,
            angle_values=self._prev.angle_values,
            intensity_unit=intensity_unit,
            angle_unit="deg", # Assuming degrees for angles
            metadata=metadata,
            raw_table=None,
        )

        self.result_run = new_run
        self._log(f"MergeRunsDialog: Created new 2D run with ID {new_run.id}")
        self.EndModal(wx.ID_OK)

    # -----------------------------
    # Rendering
    # -----------------------------

    @staticmethod
    def _centers_to_edges(x: np.ndarray) -> np.ndarray:
        """Convert 1D center coordinates to bin edges.

        Supports non-uniform grids.
        For n centers, returns n+1 edges.
        """
        x = np.asarray(x, dtype=float)
        if x.ndim != 1 or x.size < 2:
            if x.size == 1:
                return np.array([x[0] - 0.5, x[0] + 0.5], dtype=float)
            return np.array([0.0, 1.0], dtype=float)
        dx = np.diff(x)
        edges = np.empty(x.size + 1, dtype=float)
        edges[1:-1] = x[:-1] + 0.5 * dx
        edges[0] = x[0] - 0.5 * dx[0]
        edges[-1] = x[-1] + 0.5 * dx[-1]
        return edges

    def _clear_colorbar(self) -> None:
        """Remove an existing colorbar (if any).

        Matplotlib colorbars create their own Axes. If we don't remove them
        explicitly, repeated redraws can leave orphaned colorbar axes and also
        interfere with tight_layout/axes extents.
        """
        if self._cbar is None:
            return
        try:
            self._cbar.remove()
        except Exception:
            pass
        self._cbar = None

    def _update_colorbar(self) -> None:
        """Create a fresh colorbar for the current mesh (self._mesh)."""
        self._clear_colorbar()
        if self._mesh is None:
            return
        self._cbar = self.figure.colorbar(
            self._mesh,
            ax=self.ax_top,
            orientation="vertical",
            pad=0.02,
            label="Intensity",
        )

    def _render_raw_from_discover(self, disc: Optional[MergeDiscoverResult] = None) -> None:
        if disc is None:
            disc = self._disc_active if self._disc_active is not None else self._disc
        if disc is None:
            return

        self._secax_ev = None
        self._clear_colorbar()
        self.ax_top.clear()
        self.ax_bot.clear()
        self._mesh = None

        x_centers = np.asarray(disc.wavelength_nm, dtype=float)
        y_centers = np.asarray(disc.angle_values, dtype=float)
        I_display = getattr(disc, "primitive_matrix", None)
        if I_display is None:
            I_display = disc.intensity_matrix
        self._x = x_centers
        self._y = y_centers
        self._I = np.asarray(I_display, dtype=float)

        if self._I.ndim != 2:
            self._show_message("Invalid matrix shape from discover stage.")
            return

        ny, nx = self._I.shape
        self._sel_row = int(np.clip(self._sel_row, 0, ny - 1))
        self._sel_col = int(np.clip(self._sel_col, 0, nx - 1))

        self._x_centers = x_centers
        self._y_centers = y_centers
        self._x_edges = self._centers_to_edges(self._x_centers)
        self._y_edges = self._centers_to_edges(self._y_centers)

        # Use pcolormesh so non-uniform x-grid is represented correctly.
        self._mesh = self.ax_top.pcolormesh(
            self._x_edges,
            self._y_edges,
            self._I,
            shading="auto",
        )
        # Lock axes to edge extents.
        self.ax_top.set_xlim(float(self._x_edges[0]), float(self._x_edges[-1]))
        self.ax_top.set_ylim(float(self._y_edges[0]), float(self._y_edges[-1]))

        if self._original_raw_xlim is None:
            self._original_raw_xlim = self.ax_top.get_xlim()
            self._original_raw_ylim = self.ax_top.get_ylim()

        # Add/update colorbar for this mesh.
        self._update_colorbar()
        self.ax_top.set_xlabel("Wavelength (nm)")
        self.ax_top.set_ylabel("Angle (deg)")
        self.ax_top.set_title(disc.title)

        # Keep user-selected contrast across Discover/Preview/Revert.
        # Re-apply contrast to the newly created mesh.
        self._apply_contrast_from_sliders()

        # Manual laser line on wavelength axis (raw mode)
        self._laser_vline = None
        if self.chk_manual_laser.GetValue():
            self._update_manual_laser_line()

        # Initialize highlight and slice
        self._ensure_highlight_artists()
        self._update_highlight_from_indices()
        self._update_slice_plot_raw()

        self.figure.tight_layout()
        self.canvas.draw_idle()

    def _render_preview_from_result(self) -> None:
        if self._prev is None:
            return

        self._clear_colorbar()
        self.ax_top.clear()
        self.ax_bot.clear()
        self._mesh = None

        self._x = np.asarray(self._prev.raman_shift_cm1, dtype=float)
        self._y = np.asarray(self._prev.angle_values, dtype=float)
        self._I = np.asarray(self._prev.intensity_matrix, dtype=float)

        ny, nx = self._I.shape
        self._sel_row = int(np.clip(self._sel_row, 0, ny - 1))
        self._sel_col = int(np.clip(self._sel_col, 0, nx - 1))

        self._x_centers = np.asarray(self._prev.raman_shift_cm1, dtype=float)
        self._y_centers = np.asarray(self._prev.angle_values, dtype=float)
        self._x_edges = self._centers_to_edges(self._x_centers)
        self._y_edges = self._centers_to_edges(self._y_centers)

        self._mesh = self.ax_top.pcolormesh(
            self._x_edges,
            self._y_edges,
            self._I,
            shading="auto",
        )
        self.ax_top.set_xlim(float(self._x_edges[0]), float(self._x_edges[-1]))
        self.ax_top.set_ylim(float(self._y_edges[0]), float(self._y_edges[-1]))

        if self._original_preview_xlim is None:
            self._original_preview_xlim = self.ax_top.get_xlim()
            self._original_preview_ylim = self.ax_top.get_ylim()

        self._update_colorbar()

        self.ax_top.set_xlabel("Raman shift (cm$^{-1}$)")
        self.ax_top.set_ylabel("Angle (deg)")
        self.ax_top.set_title(self._prev.title)

        # Top secondary axis in eV (if available)
        self._secax_ev = None
        if self._prev.energy_ev is not None and np.all(np.isfinite(self._prev.energy_ev)):
            # Use linear conversion constant so axis is stable.
            def cm1_to_ev(x_cm1):
                return np.asarray(x_cm1, dtype=float) * EV_PER_CM1

            def ev_to_cm1(x_ev):
                return np.asarray(x_ev, dtype=float) / EV_PER_CM1

            self._secax_ev = self.ax_top.secondary_xaxis("top", functions=(cm1_to_ev, ev_to_cm1))
            self._secax_ev.set_xlabel("Energy shift (eV)")

        # Keep user-selected contrast across Discover/Preview/Revert.
        # Re-apply contrast to the newly created mesh.
        self._apply_contrast_from_sliders()

        # Laser overlay: in corrected mode, laser corresponds to 0 cm^-1.
        # If manual laser is used, emphasize 0 with a green line.
        # If per-row laser varies, we keep the same indicator and allow later refinement.
        self._laser_vline = self.ax_top.axvline(
            0.0,
            color="green",
            alpha=0.6,
            linewidth=1.2,
        )

        # Initialize highlight and slice
        self._ensure_highlight_artists()
        self._update_highlight_from_indices()
        self._update_slice_plot_preview()

        self.figure.tight_layout()
        self.canvas.draw_idle()

    def _restore_home_view(self):
        if self._in_preview_mode:
            if self._original_preview_xlim is not None and self._original_preview_ylim is not None:
                self.ax_top.set_xlim(self._original_preview_xlim)
                self.ax_top.set_ylim(self._original_preview_ylim)
                self.canvas.draw_idle()
        else:
            if self._original_raw_xlim is not None and self._original_raw_ylim is not None:
                self.ax_top.set_xlim(self._original_raw_xlim)
                self.ax_top.set_ylim(self._original_raw_ylim)
                self.canvas.draw_idle()

    def _show_message(self, msg: str) -> None:
        self._clear_colorbar()
        self.ax_top.clear()
        self.ax_bot.clear()
        self.ax_top.text(
            0.5,
            0.5,
            msg,
            ha="center",
            va="center",
            transform=self.ax_top.transAxes,
        )
        self.ax_top.set_axis_off()
        self.ax_bot.set_axis_off()
        self.canvas.draw_idle()

    # -----------------------------
    # Highlight + slice
    # -----------------------------

    def _ensure_highlight_artists(self) -> None:
        if self._vline is None:
            self._vline = self.ax_top.axvline(0.0, color="red", alpha=0.35, linewidth=1.0)
        if self._hline is None:
            self._hline = self.ax_top.axhline(0.0, color="red", alpha=0.35, linewidth=1.0)

    def _update_highlight_from_indices(self) -> None:
        if self._mesh is None or self._I is None:
            return
        if self._x_edges is None or self._y_edges is None:
            return
        ny, nx = self._I.shape
        ix = int(np.clip(self._sel_col, 0, nx - 1))
        iy = int(np.clip(self._sel_row, 0, ny - 1))

        # Highlight at the CENTER of the selected cell.
        x_hl = 0.5 * (float(self._x_edges[ix]) + float(self._x_edges[ix + 1]))
        y_hl = 0.5 * (float(self._y_edges[iy]) + float(self._y_edges[iy + 1]))

        self._vline.set_xdata([x_hl, x_hl])
        self._hline.set_ydata([y_hl, y_hl])

    def _update_slice_plot_raw(self) -> None:
        if self._I is None or self._x_centers is None:
            return
        ny, nx = self._I.shape
        r = int(np.clip(self._sel_row, 0, ny - 1))
        c = int(np.clip(self._sel_col, 0, nx - 1))
        self.ax_bot.clear()

        x_centers = self._x_centers
        x_hl = float(x_centers[c])
        if self._x_edges is not None and c + 1 < len(self._x_edges):
            x_hl = 0.5 * (float(self._x_edges[c]) + float(self._x_edges[c + 1]))

        y_hl = None
        if self._y_centers is not None and r < len(self._y_centers):
            y_hl = float(self._y_centers[r])
        if y_hl is None:
            y_hl = float(r)

        self.ax_bot.plot(x_centers, self._I[r, :], "-k", linewidth=1.2)
        self.ax_bot.axvline(x_hl, color="red", alpha=0.3, linewidth=1.0)
        self.ax_bot.set_title(f"Spectral slice ({x_hl:.3f} nm, {y_hl:.1f} deg)")
        self.ax_bot.set_xlabel("Wavelength (nm)")
        self.ax_bot.set_ylabel("Intensity")
        self.ax_bot.relim()
        self.ax_bot.autoscale_view(scalex=True, scaley=True)

    def _update_slice_plot_preview(self) -> None:
        if self._I is None or self._x is None or self._y is None:
            return
        ny, nx = self._I.shape
        r = int(np.clip(self._sel_row, 0, ny - 1))
        self.ax_bot.clear()
        self.ax_bot.plot(self._x, self._I[r, :], "-k", linewidth=1.2)
        self.ax_bot.set_xlabel("Raman shift (cm$^{-1}$)")
        self.ax_bot.set_ylabel("Intensity")
        self.ax_bot.set_title("Spectral slice (selected angle row)")
        self.ax_bot.relim()
        self.ax_bot.autoscale_view(scalex=True, scaley=True)

    # -----------------------------
    # Event handlers
    # -----------------------------

    def _on_canvas_click(self, event) -> None:
        if event.inaxes is not self.ax_top:
            return
        if event.xdata is None or event.ydata is None:
            return
        if self._mesh is None or self._I is None:
            return
        if self._x_edges is None or self._y_edges is None:
            return

        x = float(event.xdata)
        y = float(event.ydata)

        ny, nx = self._I.shape
        ix = int(np.searchsorted(self._x_edges, x, side="right") - 1)
        iy = int(np.searchsorted(self._y_edges, y, side="right") - 1)
        ix = int(np.clip(ix, 0, nx - 1))
        iy = int(np.clip(iy, 0, ny - 1))

        self._sel_col = ix
        self._sel_row = iy

        self._ensure_highlight_artists()
        self._update_highlight_from_indices()

        if self._in_preview_mode:
            self._update_slice_plot_preview()
        else:
            self._update_slice_plot_raw()

        self.canvas.draw_idle()

    def _on_toggle_raman_x_axis(self, event):
        enabled = self.chk_raman_x_axis.GetValue()
        self.rad_raman_unit_cm1.Enable(enabled)
        self.rad_raman_unit_mev.Enable(enabled)

    def _on_toggle_manual(self, event) -> None:
        enabled = bool(self.chk_manual_laser.GetValue())
        self.txt_manual_nm.Enable(enabled)
        if not enabled:
            # Remove manual laser vline if present
            if self._laser_vline is not None:
                try:
                    self._laser_vline.remove()
                except Exception:
                    pass
                self._laser_vline = None
            self.canvas.draw_idle()
        else:
            self._update_manual_laser_line()

    def _on_manual_nm_text(self, event) -> None:
        if not self.chk_manual_laser.GetValue():
            return
        # Update line live as the user types
        self._update_manual_laser_line()
    
        

    def _update_manual_laser_line(self) -> None:
        if self._in_preview_mode:
            # In corrected mode, manual laser is represented by 0 cm^-1 indicator.
            # Do not re-draw here.
            return
        if self._mesh is None or self._x is None:
            return
        s = self.txt_manual_nm.GetValue().strip()
        if not s:
            # If no value, hide the line
            if self._laser_vline is not None:
                self._laser_vline.set_visible(False)
                self.canvas.draw_idle()
            return
        try:
            nm = float(s)
        except ValueError:
            return

        if self._laser_vline is None:
            self._laser_vline = self.ax_top.axvline(nm, color="green", alpha=0.6, linewidth=1.2)
        else:
            self._laser_vline.set_xdata([nm, nm])
            self._laser_vline.set_visible(True)
        self.canvas.draw_idle()

    def _on_contrast_slider(self, event) -> None:
        self._apply_contrast_from_sliders()

    def _apply_contrast_from_sliders(self) -> None:
        if self._mesh is None or self._I is None:
            return

        # Map slider values [0..1000] to percentiles [0..100]
        a = float(self.slider_vmin.GetValue()) / 10.0
        b = float(self.slider_vmax.GetValue()) / 10.0
        if b < a:
            a, b = b, a

        # Persist percentile settings so contrast stays stable across redraws.
        self._contrast_percent = (a, b)

        data = np.asarray(self._I, float)
        finite = data[np.isfinite(data)]
        if finite.size == 0:
            return

        vmin = float(np.percentile(finite, a))
        vmax = float(np.percentile(finite, b))
        if not np.isfinite(vmin) or not np.isfinite(vmax):
            return
        if vmax <= vmin:
            vmax = vmin + 1e-12

        self._mesh.set_clim(vmin=vmin, vmax=vmax)
        self.canvas.draw_idle()

    def _on_cosmic_button(self, event) -> None:
        if self._disc_active is None and self._disc is None:
            self._show_message("No discovery matrix available.")
            return
        disc = self._disc_active if self._disc_active is not None else self._disc
        if disc is None:
            return
        if self.btn_cosmic_revert.IsEnabled():
            self._on_cosmic_revert(None)

        opts = self._build_preview_options()
        if opts is None:
            return

        try:
            cosmic_result = analysis.cosmic_discover_for_gui(disc, opts)
        except Exception as e:
            self._show_message(f"Cosmic discover failed: {e}")
            return

        peaks = list(getattr(cosmic_result, "peaks", []))
        evidence = getattr(cosmic_result, "evidence", {}) or {}

        if len(peaks) == 0:
            self._show_message(f"No cosmic peaks detected.\nEvidence: {evidence}")
            return

        candidates: List[Dict[str, Any]] = []
        for p in peaks:
            candidates.append(
                {
                    "row_index": getattr(p, "row_index", None),
                    "col_index": getattr(p, "col_index", None),
                    "is_confirmed_cosmic": getattr(p, "is_confirmed_cosmic", False),
                    "xxxx": getattr(p, "xxxx", None),
                    "yyyy": getattr(p, "yyyy", None),
                    "angle_deg": getattr(p, "angle_deg", None),
                    "center_wavelength_nm": getattr(p, "center_wavelength_nm", None),
                    "intensity": getattr(p, "intensity", None),
                    "fwhm_nm": getattr(p, "fwhm_nm", None),
                    "test_results": getattr(p, "test_results", {}),
                }
            )

        dlg = CosmicReviewDialog(
            self,
            x_centers=np.asarray(disc.wavelength_nm, float),
            y_centers=np.asarray(disc.angle_values, float),
            intensity_matrix=np.asarray(disc.intensity_matrix, float),
            raw_intensity_matrix=getattr(disc, "raw_intensity_matrix", None),
            raw_rows_by_xxxx=getattr(disc, "raw_rows_by_xxxx", None),
            primitive_xxxx=getattr(disc, "primitive_xxxx", None),
            unique_xxxx=getattr(disc, "unique_xxxx", None),
            candidates=candidates,
            evidence=evidence,
            contrast_percent=self._contrast_percent,
        )
        try:
            res = dlg.ShowModal()
            if res != wx.ID_OK:
                return
            remove_mask = dlg.get_remove_mask()
            self._contrast_percent = dlg.get_contrast_percent()
        finally:
            dlg.Destroy()

        try:
            applied_prev = analysis.cosmic_apply_for_gui(disc, cosmic_result, remove_mask)
        except Exception as e:
            self._show_message(f"Cosmic apply failed: {e}")
            return

        disc2 = MergeDiscoverResult(
            files=list(disc.files),
            pattern_hint=disc.pattern_hint,
            unique_xxxx=list(disc.unique_xxxx),
            unique_yyyy=list(disc.unique_yyyy),
            wavelength_nm=np.asarray(disc.wavelength_nm, float),
            angle_values=np.asarray(applied_prev.angle_values, float),
            intensity_matrix=np.asarray(applied_prev.intensity_matrix, float),
            title="Preview (cosmic removed, wavelength axis)",
            candidate_laser_nm=getattr(disc, "candidate_laser_nm", None),
        )

        self._disc_cosmic_applied = disc2
        self._disc_active = disc2
        self.btn_cosmic_revert.Enable(True)


        # If we are in preview mode, immediately recompute preview from the
        # updated wavelength-angle matrix so the UI reflects cosmic removal.
        if self._in_preview_mode:
            opts2 = self._build_preview_options()
            if opts2 is None:
                # Fall back to raw view if options are invalid.
                self._prev = None
                self._in_preview_mode = False
                self.btn_revert.Enable(False)
                self.btn_apply.Enable(False)
                self._render_raw_from_discover(self._disc_active)
                return
            try:
                self._prev = analysis.preview_merge(self._disc_active, opts2, prompt=None)
                self.ax_top.set_title(self._prev.title+("x axis converted to Raman shift"))
            except Exception as e:
                # Keep consistent: drop preview and show raw matrix.
                self._prev = None
                self._in_preview_mode = False
                self.btn_revert.Enable(False)
                self.btn_apply.Enable(False)
                self._show_message(f"Preview failed after cosmic apply: {e}")
                self._render_raw_from_discover(self._disc_active)
                return

            self._in_preview_mode = True
            self.btn_revert.Enable(True)
            self.btn_apply.Enable(True)
            self._render_preview_from_result()
            return

        # Raw mode: just redraw wavelength-axis matrix.
        self._render_raw_from_discover(self._disc_active)
