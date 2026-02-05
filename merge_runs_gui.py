import analysis
from analysis import MergeDiscoverOptions, MergePreviewOptions, MergeDiscoverResult, MergeCosmicOptions, MergeCosmicResult
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
from plotting import RamanPlotter2d, SlicePlotter


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

        # Selection (indices)
        self._sel_row: int = 0
        self._sel_col: int = 0

        # Plotters
        self.plotterA: Optional[RamanPlotter2d] = None
        self.plotterC: Optional[SlicePlotter] = None

        # Manual laser line artist (specific to this dialog)
        self._laser_vline = None

        # Persist contrast settings
        self._contrast_percent = (0.0, 100.0)
        
        # Synchronization state
        self._syncing_limits = False
        self._limit_cb_ids = []

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

        # Initial empty layout
        self._init_plot_layout()

        # Click handler
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

        # Cosmic correction/review
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

        # Contrast controls
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

        # Root layout
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

    def _init_plot_layout(self):
        """Reset figure and recreate plotters. This fixes shrinking due to colorbar."""
        self.figure.clf()
        
        # Disconnect old sync callbacks
        for reg, cid in self._limit_cb_ids:
            try: reg.disconnect(cid)
            except: pass
        self._limit_cb_ids.clear()

        gs = self.figure.add_gridspec(2, 1, height_ratios=[3.0, 1.4], hspace=0.35)
        self.ax_top = self.figure.add_subplot(gs[0, 0])
        self.ax_bot = self.figure.add_subplot(gs[1, 0])
        
        self.plotterA = RamanPlotter2d(self.ax_top)
        self.plotterC = SlicePlotter(self.ax_bot)

        # Re-bind sync callbacks
        def sync_A_to_C(ax):
            if self._syncing_limits: return
            self._syncing_limits = True
            try:
                self.plotterC.ax.set_xlim(ax.get_xlim())
            finally:
                self._syncing_limits = False
        
        def sync_C_to_A(ax):
            if self._syncing_limits: return
            self._syncing_limits = True
            try:
                self.plotterA.ax.set_xlim(ax.get_xlim())
            finally:
                self._syncing_limits = False

        self._limit_cb_ids.append((self.ax_top.callbacks, self.ax_top.callbacks.connect("xlim_changed", sync_A_to_C)))
        self._limit_cb_ids.append((self.ax_bot.callbacks, self.ax_bot.callbacks.connect("xlim_changed", sync_C_to_A)))

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
            self._disc = analysis.discover_merge(
                MergeDiscoverOptions(seed_file=seed)
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

        if hasattr(self._disc, "candidate_laser_nm") and self._disc.candidate_laser_nm is not None:
            val_str = f"{self._disc.candidate_laser_nm:.4f}"
            self.txt_manual_nm.SetValue(val_str)

        self._prev = None
        self._in_preview_mode = False
        self.btn_revert.Enable(False)
        self.btn_apply.Enable(False)

        self._render_raw_from_discover(self._disc_active)

    def _on_dark_value_change(self, event) -> None:
        if not self._in_preview_mode:
            self._render_raw_from_discover(self._disc_active)

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
        
        self.txt_dark_value.Enable(False)

        try:
            dark_value = float(self.txt_dark_value.GetValue())
        except ValueError:
            dark_value = 600.0

        disc_for_preview = self._disc_active if self._disc_active is not None else self._disc
        
        if dark_value != 0.0:
            intensity_matrix_sub = np.asarray(disc_for_preview.intensity_matrix, dtype=float) - dark_value
            intensity_matrix_sub[intensity_matrix_sub < 0] = 0
            
            primitive_matrix_sub = np.asarray(disc_for_preview.primitive_matrix, dtype=float) - dark_value
            primitive_matrix_sub[primitive_matrix_sub < 0] = 0

            disc_for_preview.intensity_matrix=intensity_matrix_sub
            disc_for_preview.primitive_matrix=primitive_matrix_sub

        try:
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
        self.txt_dark_value.Enable(True)
        self._prev = None
        self._in_preview_mode = False
        self.btn_revert.Enable(False)
        self.btn_apply.Enable(False)
        self._original_preview_xlim = None
        self._original_preview_ylim = None
        self._render_raw_from_discover(self._disc_active)

    def _on_cosmic_revert(self, event) -> None:
        if self._disc_original is None:
            return

        self._disc_active = self._disc_original
        self._disc_cosmic_applied = None
        self.btn_cosmic_revert.Enable(False)

        if not self._in_preview_mode:
            self._render_raw_from_discover(self._disc_active)
            return

        opts = self._build_preview_options()
        if opts is None:
            self._prev = None
            self._in_preview_mode = False
            self.btn_revert.Enable(False)
            self.btn_apply.Enable(False)
            self._render_raw_from_discover(self._disc_active)
            return

        try:
            self._prev = analysis.preview_merge(self._disc_active, opts, prompt=None)
        except Exception as e:
            self._prev = None
            self._in_preview_mode = False
            self.btn_revert.Enable(False)
            self.btn_apply.Enable(False)
            self._show_message(f"Preview failed after cosmic revert: {e}")
            self._render_raw_from_discover(self._disc_active)
            return

        self._in_preview_mode = True
        self.btn_revert.Enable(True)
        self.btn_apply.Enable(True)
        self._render_preview_from_result()

    def _on_apply(self, event) -> None:
        if self._prev is None:
            return

        new_run_id = new_experiment_id() 
        source_path = self._paths[0] 

        metadata = parse_filename(source_path)
        if self._disc is not None:
            metadata["merge_pattern_hint"] = self._disc.pattern_hint
            metadata["merge_unique_xxxx"] = self._disc.unique_xxxx
            metadata["merge_unique_yyyy"] = self._disc.unique_yyyy
            metadata["merged_files"] = self._disc.files
            metadata["nickname"] = None

        intensity_unit = "au" 
        if self._disc and self._disc.raw_intensity_matrix is not None:
            intensity_unit = infer_intensity_unit(self._disc.raw_intensity_matrix)

        new_run = Run(
            id=new_run_id,
            source_path=source_path,
            source_mtime=None,
            shift_cm1=self._prev.raman_shift_cm1,
            energy_eV=self._prev.energy_ev,
            intensity=None,
            intensity_2d=self._prev.intensity_matrix,
            angle_values=self._prev.angle_values,
            intensity_unit=intensity_unit,
            angle_unit="deg",
            metadata=metadata,
            raw_table=None,
        )

        self.result_run = new_run
        self._log(f"MergeRunsDialog: Created new 2D run with ID {new_run.id}")
        self.EndModal(wx.ID_OK)

    # -----------------------------
    # Rendering
    # -----------------------------

    def _render_raw_from_discover(self, disc: Optional[MergeDiscoverResult] = None) -> None:
        if disc is None:
            disc = self._disc_active if self._disc_active is not None else self._disc
        if disc is None:
            return

        # Fully reset layout to prevent colorbar shrinking issues
        self._init_plot_layout()
        
        self._secax_ev = None
        self._laser_vline = None

        x_centers = np.asarray(disc.wavelength_nm, dtype=float)
        y_centers = np.asarray(disc.angle_values, dtype=float)
        I_display = getattr(disc, "cosmic_matrix", None)
        if I_display is None:
            I_display = getattr(disc, "primitive_matrix", None)
            if I_display is None:
                I_display = disc.intensity_matrix

        try:
            dark_value = float(self.txt_dark_value.GetValue())
        except ValueError:
            dark_value = 0.0

        if dark_value != 0.0:
            I_display = np.asarray(I_display, dtype=float).copy()
            I_display -= dark_value
            I_display[I_display < 0] = 0

        self._x = x_centers
        self._y = y_centers
        self._I = np.asarray(I_display, dtype=float)

        if self._I.ndim != 2:
            self._show_message("Invalid matrix shape from discover stage.")
            return

        ny, nx = self._I.shape
        self._sel_row = int(np.clip(self._sel_row, 0, ny - 1))
        self._sel_col = int(np.clip(self._sel_col, 0, nx - 1))

        # Render Plot A (2D Map)
        self.plotterA.render(
            self._x, self._y, self._I,
            title=disc.title,
            xlabel="Wavelength (nm)",
            ylabel="Angle (deg)"
        )
        
        if self._original_raw_xlim is None:
            self._original_raw_xlim = self.plotterA.ax.get_xlim()
            self._original_raw_ylim = self.plotterA.ax.get_ylim()

        self._apply_contrast_from_sliders()

        if self.chk_manual_laser.GetValue():
            self._update_manual_laser_line()

        # Update Highlight & Slice
        self._update_highlight_and_slice()

        self.figure.tight_layout()
        self.canvas.draw_idle()

    def _render_preview_from_result(self) -> None:
        if self._prev is None:
            return

        # Fully reset layout to prevent colorbar shrinking issues
        self._init_plot_layout()

        self._laser_vline = None

        self._x = np.asarray(self._prev.raman_shift_cm1, dtype=float)
        self._y = np.asarray(self._prev.angle_values, dtype=float)
        self._I = np.asarray(self._prev.intensity_matrix, dtype=float)

        ny, nx = self._I.shape
        self._sel_row = int(np.clip(self._sel_row, 0, ny - 1))
        self._sel_col = int(np.clip(self._sel_col, 0, nx - 1))

        # Setup secondary axis helpers if energy_ev is available
        x_unit_conv = None
        if self._prev.energy_ev is not None and np.all(np.isfinite(self._prev.energy_ev)):
            def cm1_to_ev(x_cm1): return np.asarray(x_cm1, dtype=float) * EV_PER_CM1
            def ev_to_cm1(x_ev): return np.asarray(x_ev, dtype=float) / EV_PER_CM1
            x_unit_conv = (cm1_to_ev, ev_to_cm1)

        self.plotterA.render(
            self._x, self._y, self._I,
            title=self._prev.title,
            xlabel="Raman shift (cm$^{-1}$)",
            ylabel="Angle (deg)",
            x_unit_conversion=x_unit_conv
        )

        if self._original_preview_xlim is None:
            self._original_preview_xlim = self.plotterA.ax.get_xlim()
            self._original_preview_ylim = self.plotterA.ax.get_ylim()

        self._apply_contrast_from_sliders()

        # Laser overlay (0 cm-1)
        self._laser_vline = self.plotterA.ax.axvline(
            0.0, color="green", alpha=0.6, linewidth=1.2
        )

        self._update_highlight_and_slice()

        self.figure.tight_layout()
        self.canvas.draw_idle()

    def _restore_home_view(self):
        if self._in_preview_mode:
            if self._original_preview_xlim is not None:
                self.plotterA.ax.set_xlim(self._original_preview_xlim)
                self.plotterA.ax.set_ylim(self._original_preview_ylim)
                self.canvas.draw_idle()
        else:
            if self._original_raw_xlim is not None:
                self.plotterA.ax.set_xlim(self._original_raw_xlim)
                self.plotterA.ax.set_ylim(self._original_raw_ylim)
                self.canvas.draw_idle()

    def _show_message(self, msg: str) -> None:
        self.plotterA.clear()
        self.plotterC.clear()
        self.ax_top.text(
            0.5, 0.5, msg,
            ha="center", va="center", transform=self.ax_top.transAxes,
        )
        self.ax_top.set_axis_off()
        self.ax_bot.set_axis_off()
        self.canvas.draw_idle()

    # -----------------------------
    # Highlight + slice
    # -----------------------------

    def _update_highlight_and_slice(self) -> None:
        if self._I is None or self._x is None:
            return
        
        # 1. Update Highlight on Plot A
        x_sel, y_sel = self.plotterA.get_coords_from_index(self._sel_col, self._sel_row)
        self.plotterA.set_highlight(x_sel, y_sel)

        # 2. Update Plot C (Spectral Slice)
        r = int(np.clip(self._sel_row, 0, self._I.shape[0] - 1))
        
        slice_title = "Spectral slice"
        slice_xlabel = "Wavelength (nm)"
        if self._in_preview_mode:
            slice_title = "Spectral slice (selected angle row)"
            slice_xlabel = "Raman shift (cm$^{-1}$)"
        else:
            slice_title = f"Spectral slice ({x_sel:.3f} nm, {y_sel:.1f} deg)"

        self.plotterC.render(
            self._x, self._I[r, :],
            title=slice_title,
            xlabel=slice_xlabel,
            ylabel="Intensity"
        )
        self.plotterC.set_highlight(x_sel)

    # -----------------------------
    # Event handlers
    # -----------------------------

    def _on_canvas_click(self, event) -> None:
        if event.inaxes is not self.ax_top:
            return
        
        indices = self.plotterA.get_index_at(event.xdata, event.ydata)
        if not indices:
            return
        
        self._sel_col, self._sel_row = indices
        self._update_highlight_and_slice()

    def _on_toggle_raman_x_axis(self, event):
        enabled = self.chk_raman_x_axis.GetValue()
        self.rad_raman_unit_cm1.Enable(enabled)
        self.rad_raman_unit_mev.Enable(enabled)

    def _on_toggle_manual(self, event) -> None:
        enabled = bool(self.chk_manual_laser.GetValue())
        self.txt_manual_nm.Enable(enabled)
        if not enabled:
            if self._laser_vline is not None:
                try: self._laser_vline.remove()
                except: pass
                self._laser_vline = None
            self.canvas.draw_idle()
        else:
            self._update_manual_laser_line()

    def _on_manual_nm_text(self, event) -> None:
        if not self.chk_manual_laser.GetValue():
            return
        self._update_manual_laser_line()

    def _update_manual_laser_line(self) -> None:
        if self._in_preview_mode:
            return
        if self._mesh is None and self.plotterA.mesh is None:
            return
            
        s = self.txt_manual_nm.GetValue().strip()
        if not s:
            if self._laser_vline: self._laser_vline.set_visible(False)
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
        if self.plotterA is None:
            return

        a = float(self.slider_vmin.GetValue()) / 10.0
        b = float(self.slider_vmax.GetValue()) / 10.0
        if b < a:
            a, b = b, a

        self._contrast_percent = (a, b)
        self.plotterA.set_contrast(a, b)

    def _on_cosmic_button(self, event) -> None:
        if self._disc_active is None and self._disc is None:
            self._show_message("No discovery matrix available.")
            return
        disc = self._disc_active if self._disc_active is not None else self._disc
        if disc is None:
            return
        if self.btn_cosmic_revert.IsEnabled():
            self._on_cosmic_revert(None)

        try:
            dark_value = float(self.txt_dark_value.GetValue())
        except ValueError:
            dark_value = 0

        cosmic_options = MergeCosmicOptions(
            files=disc.files,
            pattern_hint=disc.pattern_hint,
            unique_xxxx=disc.unique_xxxx,
            unique_yyyy=disc.unique_yyyy,
            wavelength_nm=disc.wavelength_nm,
            angle_values=disc.angle_values,
            intensity_matrix=disc.intensity_matrix,
            title=disc.title,
            candidate_laser_nm=disc.candidate_laser_nm,
            raw_files=disc.raw_files,
            raw_xxxx=disc.raw_xxxx,
            raw_yyyy=disc.raw_yyyy,
            raw_angle_values=disc.raw_angle_values,
            raw_intensity_matrix=disc.raw_intensity_matrix,
            raw_rows_by_xxxx=disc.raw_rows_by_xxxx,
            primitive_xxxx=disc.primitive_xxxx,
            primitive_matrix=disc.primitive_matrix,
            cosmic_matrix=disc.cosmic_matrix,
            dark_value=dark_value
        )

        try:
            cosmic_result = analysis.discover_cosmics(cosmic_options)
        except Exception as e:
            self._show_message(f"Cosmic discover failed: {e}")
            return
        
        if not cosmic_result.peaks:
            self._show_message(f"No cosmic peaks detected.\nEvidence: {cosmic_result.evidence}")
            return

        dlg = CosmicReviewDialog(
            self,
            cosmic_result=cosmic_result,
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
            disc2 = analysis.apply_cosmic_removal(cosmic_result, remove_mask)
        except Exception as e:
            self._show_message(f"Cosmic apply failed: {e}")
            return

        self._disc_cosmic_applied = disc2
        self._disc_active = disc2
        self.btn_cosmic_revert.Enable(True)


        if self._in_preview_mode:
            opts2 = self._build_preview_options()
            if opts2 is None:
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

        self._render_raw_from_discover(self._disc_active)
