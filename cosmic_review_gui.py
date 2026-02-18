import analysis
from analysis import MergeDiscoverOptions, MergePreviewOptions, MergeDiscoverResult, MergePreviewResult
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
)
from plotting import RamanPlotter2d, SlicePlotter
from config_manager import config

class CosmicReviewDialog(wx.Dialog):
    """Cosmic-ray review dialog.

    - Left: matrix plot (angle × wavelength) + 1D slice at selected row.
    - Right: candidate list (checked = remove) + evidence + contrast sliders.
    """

    def __init__(
        self,
        parent: wx.Window,
        *,
        cosmic_result: "analysis.MergeCosmicResult",
        contrast_percent: Tuple[float, float],
    ):
        super().__init__(parent, title="Cosmic Review", style=wx.DEFAULT_DIALOG_STYLE | wx.RESIZE_BORDER)

        self._x_centers = np.asarray(cosmic_result.wavelength_nm, dtype=float)
        self._y_centers = np.asarray(cosmic_result.angle_values, dtype=float)
        self._I = np.asarray(cosmic_result.intensity_matrix, dtype=float) - cosmic_result.dark_value
        
        # Check for shape mismatch
        ny, nx = self._I.shape
        ly = len(self._y_centers)
        lx = len(self._x_centers)
        if ny != ly or nx != lx:
             raise ValueError(f"Shape Mismatch: Data ({ny}, {nx}) vs Axes")

        self._raw_I = (
            np.asarray(cosmic_result.raw_intensity_matrix, dtype=float)
            if cosmic_result.raw_intensity_matrix is not None
            else None
        )
        self._raw_rows_by_xxxx = dict(cosmic_result.raw_rows_by_xxxx) if cosmic_result.raw_rows_by_xxxx else {}
        self._primitive_xxxx = list(cosmic_result.primitive_xxxx) if cosmic_result.primitive_xxxx else None
        self._unique_xxxx = list(cosmic_result.unique_xxxx) if cosmic_result.unique_xxxx else None
        
        peaks = list(getattr(cosmic_result, "peaks", []))
        evidence = getattr(cosmic_result, "evidence", {}) or {}
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

        self._candidates = candidates
        self._evidence = dict(evidence) if evidence is not None else {}
        self._general_evidence = self._evidence
        self._contrast_percent = (float(contrast_percent[0]), float(contrast_percent[1]))
        self._dark_value = cosmic_result.dark_value
        self._cosmic_result = cosmic_result # Store for re-detection

        self._sel_row = 0
        self._sel_col = 0
        self._selected_candidate_info: Optional[Dict[str, Any]] = None

        self.plotterA: Optional[RamanPlotter2d] = None
        self.plotterC: Optional[SlicePlotter] = None
        
        self._syncing_limits = False
        self._limit_cb_ids = []

        root = wx.BoxSizer(wx.HORIZONTAL)
        self.SetSizer(root)


        # Left: plots
        plot_panel = wx.Panel(self)
        plot_sizer = wx.BoxSizer(wx.VERTICAL)
        plot_panel.SetSizer(plot_sizer)

        self.figure = Figure(figsize=(9, 6))
        self.canvas = FigureCanvas(plot_panel, -1, self.figure)
        self.toolbar = NavigationToolbar(self.canvas)
        self.toolbar.Realize()

        plot_sizer.Add(self.toolbar, 0, wx.EXPAND | wx.ALL, 2)
        plot_sizer.Add(self.canvas, 1, wx.EXPAND | wx.ALL, 2)

        # Initial layout
        self._init_plot_layout()

        self._cid_click = self.canvas.mpl_connect("button_press_event", self._on_canvas_click)

        # Right: controls
        ctrl_panel = wx.Panel(self)
        ctrl_sizer = wx.BoxSizer(wx.VERTICAL)
        ctrl_panel.SetSizer(ctrl_sizer)

        # Evidence
        box_ev = wx.StaticBoxSizer(wx.StaticBox(ctrl_panel, label="Evidence"), wx.VERTICAL)
        ctrl_sizer.Add(box_ev, 0, wx.EXPAND | wx.ALL, 6)
        self.txt_evidence = wx.TextCtrl(
            ctrl_panel,
            value=self._format_evidence(self._evidence),
            style=wx.TE_MULTILINE | wx.TE_READONLY,
        )
        self.txt_evidence.SetMinSize((340, 90))
        box_ev.Add(self.txt_evidence, 1, wx.EXPAND | wx.ALL, 4)

        # Detection Threshold Control
        box_thresh = wx.StaticBoxSizer(wx.StaticBox(ctrl_panel, label="Detection Threshold"), wx.VERTICAL)
        ctrl_sizer.Add(box_thresh, 0, wx.EXPAND | wx.LEFT | wx.RIGHT | wx.BOTTOM, 6)

        row_inputs = wx.BoxSizer(wx.HORIZONTAL)
        
        self.txt_threshold = wx.TextCtrl(ctrl_panel, value=f"{cosmic_result.intensity_thresh:.1f}", size=(70, -1))
        self.txt_ratio = wx.TextCtrl(ctrl_panel, value=f"{cosmic_result.comparison_factor:.1f}", size=(70, -1))
        self.btn_detect = wx.Button(ctrl_panel, label="Detect")
        
        row_inputs.Add(wx.StaticText(ctrl_panel, label="Min Height:"), 0, wx.ALIGN_CENTER_VERTICAL | wx.ALL, 2)
        row_inputs.Add(self.txt_threshold, 1, wx.EXPAND | wx.ALL, 2)
        row_inputs.Add(wx.StaticText(ctrl_panel, label="Ratio:"), 0, wx.ALIGN_CENTER_VERTICAL | wx.ALL, 2)
        row_inputs.Add(self.txt_ratio, 1, wx.EXPAND | wx.ALL, 2)
        row_inputs.Add(self.btn_detect, 0, wx.ALL, 2)
        
        box_thresh.Add(row_inputs, 0, wx.EXPAND)
        
        self.Bind(wx.EVT_BUTTON, self._on_re_detect, self.btn_detect)

        # Candidate list
        box_list = wx.StaticBoxSizer(wx.StaticBox(ctrl_panel, label="Candidates (checked = remove)"), wx.VERTICAL)
        ctrl_sizer.Add(box_list, 1, wx.EXPAND | wx.LEFT | wx.RIGHT | wx.BOTTOM, 6)

        self.chk_sort_prominence = wx.CheckBox(ctrl_panel, label="Show by Prominence")
        self.chk_sort_prominence.SetValue(True)
        box_list.Add(self.chk_sort_prominence, 0, wx.ALL, 4)

        self.chk_list = wx.CheckListBox(ctrl_panel, choices=[])
        box_list.Add(self.chk_list, 1, wx.EXPAND | wx.ALL, 4)

        for i, c in enumerate(self._candidates):
            c["original_index"] = i
            c["is_checked"] = bool(c.get("is_confirmed_cosmic", False))
        
        self.Bind(wx.EVT_CHECKBOX, self._on_sort_change, self.chk_sort_prominence)
        self.Bind(wx.EVT_CHECKLISTBOX, self._on_candidate_check_changed, self.chk_list)

        self._sort_and_refresh_list()

        self.Bind(wx.EVT_LISTBOX, self._on_candidate_select, self.chk_list)

        # Contrast sliders
        box_contrast = wx.StaticBoxSizer(wx.StaticBox(ctrl_panel, label="Contrast (percentiles)"), wx.VERTICAL)
        ctrl_sizer.Add(box_contrast, 0, wx.EXPAND | wx.LEFT | wx.RIGHT | wx.BOTTOM, 6)

        self.slider_vmin = wx.Slider(
            ctrl_panel, minValue=0, maxValue=1000, value=int(self._contrast_percent[0] * 10), style=wx.SL_HORIZONTAL
        )
        self.slider_vmax = wx.Slider(
            ctrl_panel, minValue=0, maxValue=1000, value=int(self._contrast_percent[1] * 10), style=wx.SL_HORIZONTAL
        )

        box_contrast.Add(wx.StaticText(ctrl_panel, label="vmin"), 0, wx.LEFT | wx.RIGHT | wx.TOP, 4)
        box_contrast.Add(self.slider_vmin, 0, wx.EXPAND | wx.LEFT | wx.RIGHT | wx.BOTTOM, 4)
        box_contrast.Add(wx.StaticText(ctrl_panel, label="vmax"), 0, wx.LEFT | wx.RIGHT | wx.TOP, 4)
        box_contrast.Add(self.slider_vmax, 0, wx.EXPAND | wx.LEFT | wx.RIGHT | wx.BOTTOM, 4)

        self.Bind(wx.EVT_SLIDER, self._on_contrast_slider, self.slider_vmin)
        self.Bind(wx.EVT_SLIDER, self._on_contrast_slider, self.slider_vmax)

        # Buttons
        btns = wx.BoxSizer(wx.HORIZONTAL)
        self.btn_ok = wx.Button(ctrl_panel, id=wx.ID_OK, label="Apply")
        self.btn_cancel = wx.Button(ctrl_panel, id=wx.ID_CANCEL, label="Cancel")
        btns.AddStretchSpacer(1)
        btns.Add(self.btn_ok, 0, wx.ALL, 4)
        btns.Add(self.btn_cancel, 0, wx.ALL, 4)
        ctrl_sizer.Add(btns, 0, wx.EXPAND | wx.ALL, 6)

        root.Add(plot_panel, 3, wx.EXPAND | wx.ALL, 6)
        root.Add(ctrl_panel, 1, wx.EXPAND | wx.ALL, 6)

        # Initial render
        self._render_matrix()
        self._update_highlight_and_slice()

        self.SetSize((1150, 740))
        self.Layout()

    def get_remove_mask(self) -> List[bool]:        
        sorted_by_original_index = sorted(self._candidates, key=lambda c: c['original_index'])
        return [c['is_checked'] for c in sorted_by_original_index]

    def get_cosmic_result(self) -> "analysis.MergeCosmicResult":
        return self._cosmic_result

    def _on_re_detect(self, event):
        try:
            h_val = float(self.txt_threshold.GetValue())
            r_val = float(self.txt_ratio.GetValue())
        except ValueError:
            wx.MessageBox("Invalid values. Please enter numbers for Height and Ratio.", "Error", wx.OK | wx.ICON_ERROR)
            return

        self._cosmic_result.intensity_thresh = h_val
        self._cosmic_result.comparison_factor = r_val
        
        # Save to global config
        config.set("cosmic_threshold", h_val)
        config.set("cosmic_ratio", r_val)
        
        try:
            new_res = analysis.discover_cosmics(self._cosmic_result)
        except Exception as e:
            wx.MessageBox(f"Detection failed: {e}", "Error", wx.OK | wx.ICON_ERROR)
            return
            
        # Store for return
        self._cosmic_result = new_res

        # Re-populate candidates
        self._candidates = []
        for i, p in enumerate(new_res.peaks):
            self._candidates.append(
                {
                    "row_index": p.row_index,
                    "col_index": p.col_index,
                    "is_confirmed_cosmic": p.is_confirmed_cosmic,
                    "xxxx": p.xxxx,
                    "yyyy": p.yyyy,
                    "angle_deg": p.angle_deg,
                    "center_wavelength_nm": p.center_wavelength_nm,
                    "intensity": p.intensity,
                    "fwhm_nm": p.fwhm_nm,
                    "test_results": p.test_results,
                    "original_index": i,
                    "is_checked": bool(p.is_confirmed_cosmic) # Use the logic-based flag
                }
            )
        
        self._evidence = new_res.evidence
        self._general_evidence = self._evidence
        self.txt_evidence.SetValue(self._format_evidence(self._evidence))
        self._selected_candidate_info = None
        
        self._sort_and_refresh_list()
        self._render_matrix()
        self._update_point_highlights()
        self._update_highlight_and_slice()
        self.Layout()

    def _sort_and_refresh_list(self):
        # Sort the internal list
        if self.chk_sort_prominence.IsChecked():
            # Sort by prominence (ratio) descending
            self._candidates.sort(key=lambda c: c.get('test_results', {}).get('ratio', 0.0), reverse=True)
        else:
            # Sort by original index ascending
            self._candidates.sort(key=lambda c: c['original_index'])

        # Repopulate the wx.CheckListBox
        self.chk_list.Clear()
        items = [self._format_candidate(c) for c in self._candidates]
        self.chk_list.InsertItems(items, 0)

        # Re-apply the checked state
        for i, c in enumerate(self._candidates):
            self.chk_list.Check(i, c['is_checked'])
    
    def _on_sort_change(self, event):
        self._sort_and_refresh_list()

    def _on_candidate_check_changed(self, event):
        idx = event.GetSelection()
        if 0 <= idx < len(self._candidates):
            self._candidates[idx]['is_checked'] = self.chk_list.IsChecked(idx)
            self._update_point_highlights()

    def _update_point_highlights(self):
        """Send coordinates of checked candidates to the 2D plotter for visualization."""
        if not self.plotterA:
            return
            
        x_hlts = []
        y_hlts = []
        for c in self._candidates:
            if c.get('is_checked'):
                wl = c.get("center_wavelength_nm")
                ang = c.get("angle_deg")
                if wl is not None and ang is not None:
                    x_hlts.append(wl)
                    y_hlts.append(ang)
        
        # Use Turquoise (#40E0D0) for blue-greenish highlight
        self.plotterA.set_points(x_hlts, y_hlts, color="#40E0D0", size=40)


    def get_contrast_percent(self) -> Tuple[float, float]:
        a = float(self.slider_vmin.GetValue()) / 10.0
        b = float(self.slider_vmax.GetValue()) / 10.0
        if b < a:
            a, b = b, a
        return (a, b)

    @staticmethod
    def _centers_to_edges(x: np.ndarray) -> np.ndarray:
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

    @staticmethod
    def _format_evidence(evidence: Dict[str, object]) -> str:
        if not evidence:
            return "(no evidence reported)"
        lines = []
        for k in sorted(evidence.keys()):
            v = evidence[k]
            if isinstance(v, float):
                lines.append(f"{k}: {v:.3g}")
            else:
                lines.append(f"{k}: {v}")
        return "\n".join(lines)


    @staticmethod
    def _format_candidate(c: Dict[str, Any]) -> str:
        display_index = c.get('original_index', -1) + 1
        label = f"#{display_index}"
        
        ang = c.get("angle_deg", None)
        wl = c.get("center_wavelength_nm", None)
        inten = c.get("intensity", None)
        fwhm = c.get("fwhm_nm", None)
        xxxx = c.get("xxxx", None)
        yyyy = c.get("yyyy", None)

        parts = [label]
        if ang is not None:
            parts.append(f"{float(ang):.1f} deg")
        if wl is not None:
            parts.append(f"{float(wl):.3f} nm")
        if inten is not None and np.isfinite(float(inten)):
            parts.append(f"I={float(inten):.3g}")
        if fwhm is not None and np.isfinite(float(fwhm)):
            parts.append(f"FWHM={float(fwhm):.3g} nm")
        if xxxx is not None or yyyy is not None:
            parts.append(f"({xxxx},{yyyy})")
        return " | ".join(parts)

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

    def _render_matrix(self) -> None:
        # Reset layout before rendering to fix colorbar shrinking and reset axes
        self._init_plot_layout()
        
        self.plotterA.render(
            self._x_centers, self._y_centers, self._I,
            title="Cosmic review (wavelength axis)",
            xlabel="Wavelength (nm)",
            ylabel="Angle (deg)"
        )
        self._update_point_highlights()
        self._apply_contrast_from_sliders()
        self.figure.tight_layout()
        self.canvas.draw_idle()

    def _update_highlight_and_slice(self) -> None:
        ny, nx = self._I.shape
        r = int(np.clip(self._sel_row, 0, ny - 1))
        c = int(np.clip(self._sel_col, 0, nx - 1))

        # Update Highlight
        x_hl, y_hl = self.plotterA.get_coords_from_index(c, r)
        self.plotterA.set_highlight(x_hl, y_hl)

        # Update Slice
        yline = self._I[r, :]
        self.plotterC.render(
            self._x_centers, yline,
            title=f"Spectral slice({x_hl:.3f} nm, {y_hl:.1f} deg)",
            xlabel="Wavelength (nm)",
            ylabel="Intensity"
        )
        
        # Overlay raw data traces
        if self._raw_I is not None and self._raw_rows_by_xxxx:
            xxxx_key = None
            if self._primitive_xxxx is not None and r < len(self._primitive_xxxx):
                xxxx_key = self._primitive_xxxx[r]
            if xxxx_key is None and self._unique_xxxx is not None and r < len(self._unique_xxxx):
                xxxx_key = self._unique_xxxx[r]
            if xxxx_key is None:
                xxxx_key = str(r)
            rows = self._raw_rows_by_xxxx.get(xxxx_key, [])
            for rr in rows:
                if rr < 0 or rr >= self._raw_I.shape[0]:
                    continue
                raw = self._raw_I[rr, :]
                if self._dark_value != 0.0:
                    raw = raw - self._dark_value
                    raw[raw < 0] = 0
                n = min(raw.shape[0], self._x_centers.shape[0])
                
                self.plotterC.add_trace(
                    self._x_centers[:n], raw[:n], 
                    color="black", style=":", alpha=0.6
                )

        # Highlight position
        self.plotterC.set_highlight(x_hl)
        
        # Extra dot for specific candidate (manual artist on ax)
        if self._selected_candidate_info and self._raw_I is not None:
            raw_r = self._selected_candidate_info.get("row_index")
            raw_c = self._selected_candidate_info.get("col_index")
            dot_y = float(yline[c])
            if raw_r is not None and raw_c is not None:
                if 0 <= raw_r < self._raw_I.shape[0] and 0 <= raw_c < self._raw_I.shape[1]:
                    dot_y = self._raw_I[raw_r, raw_c]
            self.plotterC.ax.plot([x_hl], [dot_y], "o", color="red", markersize=4)

        self.canvas.draw_idle()

    def _on_canvas_click(self, event) -> None:
        if event.inaxes is not self.ax_top:
            return
        
        indices = self.plotterA.get_index_at(event.xdata, event.ydata)
        if not indices:
            return
        
        self._sel_col, self._sel_row = indices
        self._selected_candidate_info = None  # Clear selected candidate on manual click
        self.txt_evidence.SetValue(self._format_evidence(self._general_evidence))
        
        self._update_highlight_and_slice()

    def _on_candidate_select(self, event) -> None:
        idx = int(self.chk_list.GetSelection())
        if idx < 0 or idx >= len(self._candidates):
            self._selected_candidate_info = None
            return
        c = self._candidates[idx]
        self._selected_candidate_info = c

        angle = c.get("angle_deg", None)
        wavelength = c.get("center_wavelength_nm", None)

        if angle is not None:
            self._sel_row = int(np.argmin(np.abs(self._y_centers - float(angle))))
        
        if wavelength is not None:
            self._sel_col = int(np.argmin(np.abs(self._x_centers - float(wavelength))))

        # Show peak-specific evidence
        peak_evidence = c.get("test_results", {})
        self.txt_evidence.SetValue(self._format_evidence(peak_evidence))
        
        self._update_highlight_and_slice()

    def _on_contrast_slider(self, event) -> None:
        self._apply_contrast_from_sliders()

    def _apply_contrast_from_sliders(self) -> None:
        if not self.plotterA:
            return
        a = float(self.slider_vmin.GetValue()) / 10.0
        b = float(self.slider_vmax.GetValue()) / 10.0
        if b < a:
            a, b = b, a
        self.plotterA.set_contrast(a, b)