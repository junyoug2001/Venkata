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

class CosmicReviewDialog(wx.Dialog):
    """Cosmic-ray review dialog.

    - Left: matrix plot (angle × wavelength) + 1D slice at selected row.
    - Right: candidate list (checked = remove) + evidence + contrast sliders.
    """

    def __init__(
        self,
        parent: wx.Window,
        *,
        x_centers: np.ndarray,
        y_centers: np.ndarray,
        intensity_matrix: np.ndarray,
        raw_intensity_matrix: Optional[np.ndarray] = None,
        raw_rows_by_xxxx: Optional[Dict[str, List[int]]] = None,
        primitive_xxxx: Optional[Sequence[str]] = None,
        unique_xxxx: Optional[Sequence[str]] = None,
        candidates: List[Dict[str, Any]],
        evidence: Dict[str, object],
        contrast_percent: Tuple[float, float],
        dark_value: float = 0.0,
    ):
        super().__init__(parent, title="Cosmic Review", style=wx.DEFAULT_DIALOG_STYLE | wx.RESIZE_BORDER)

        self._x_centers = np.asarray(x_centers, dtype=float)
        self._y_centers = np.asarray(y_centers, dtype=float)
        self._I = np.asarray(intensity_matrix, dtype=float)
        self._raw_I = (
            np.asarray(raw_intensity_matrix, dtype=float)
            if raw_intensity_matrix is not None
            else None
        )
        self._raw_rows_by_xxxx = dict(raw_rows_by_xxxx) if raw_rows_by_xxxx else {}
        self._primitive_xxxx = list(primitive_xxxx) if primitive_xxxx else None
        self._unique_xxxx = list(unique_xxxx) if unique_xxxx else None
        self._candidates = list(candidates)
        self._evidence = dict(evidence) if evidence is not None else {}
        self._general_evidence = self._evidence  # Store general evidence
        self._contrast_percent = (float(contrast_percent[0]), float(contrast_percent[1]))
        self._dark_value = dark_value

        self._sel_row = 0
        self._sel_col = 0
        self._selected_candidate_info: Optional[Dict[str, Any]] = None

        self._x_edges = self._centers_to_edges(self._x_centers)
        self._y_edges = self._centers_to_edges(self._y_centers)

        self._mesh = None
        self._cbar = None
        self._vline = None
        self._hline = None
        # Slice highlight artists (bottom plot)
        self._slice_vline = None
        self._slice_dot = None

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

        self.figure.clf()
        gs = self.figure.add_gridspec(2, 1, height_ratios=[3.0, 1.4], hspace=0.35)
        self.ax_top = self.figure.add_subplot(gs[0, 0])
        self.ax_bot = self.figure.add_subplot(gs[1, 0])

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
        self._ensure_highlight()
        self._update_highlight()
        self._update_slice()

        self.SetSize((1150, 740))
        self.Layout()

    def get_remove_mask(self) -> List[bool]:
        
        sorted_by_original_index = sorted(self._candidates, key=lambda c: c['original_index'])
        return [c['is_checked'] for c in sorted_by_original_index]

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

    def _clear_colorbar(self) -> None:
        if self._cbar is None:
            return
        try:
            self._cbar.remove()
        except Exception:
            pass
        self._cbar = None

    def _render_matrix(self) -> None:
        self.ax_top.clear()
        self.ax_bot.clear()
        self._clear_colorbar()

        self._mesh = self.ax_top.pcolormesh(self._x_edges, self._y_edges, self._I, shading="auto")
        self.ax_top.set_xlim(float(self._x_edges[0]), float(self._x_edges[-1]))
        self.ax_top.set_ylim(float(self._y_edges[0]), float(self._y_edges[-1]))
        self.ax_top.set_xlabel("Wavelength (nm)")
        self.ax_top.set_ylabel("Angle (deg)")
        self.ax_top.set_title("Cosmic review (wavelength axis)")

        self._cbar = self.figure.colorbar(self._mesh, ax=self.ax_top, pad=0.02, label="Intensity")

        self._apply_contrast_from_sliders()
        self.figure.tight_layout()
        self.canvas.draw_idle()

    def _ensure_highlight(self) -> None:
        if self._vline is None:
            self._vline = self.ax_top.axvline(0.0, color="red", alpha=0.35, linewidth=1.0)
        if self._hline is None:
            self._hline = self.ax_top.axhline(0.0, color="red", alpha=0.35, linewidth=1.0)

    def _update_highlight(self) -> None:
        ny, nx = self._I.shape
        ix = int(np.clip(self._sel_col, 0, nx - 1))
        iy = int(np.clip(self._sel_row, 0, ny - 1))
        x_hl = 0.5 * (float(self._x_edges[ix]) + float(self._x_edges[ix + 1]))
        y_hl = 0.5 * (float(self._y_edges[iy]) + float(self._y_edges[iy + 1]))
        self._vline.set_xdata([x_hl, x_hl])
        self._hline.set_ydata([y_hl, y_hl])

    def _update_slice(self) -> None:
        ny, nx = self._I.shape
        r = int(np.clip(self._sel_row, 0, ny - 1))
        c = int(np.clip(self._sel_col, 0, nx - 1))

        # Cell-center coordinates for title and highlight
        x_hl = 0.5 * (float(self._x_edges[c]) + float(self._x_edges[c + 1]))
        y_hl = 0.5 * (float(self._y_edges[r]) + float(self._y_edges[r + 1]))

        self.ax_bot.clear()
        yline = self._I[r, :]
        self.ax_bot.plot(self._x_centers, yline, "-k", linewidth=1.2)

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
                self.ax_bot.plot(
                    self._x_centers[:n],
                    raw[:n],
                    ":",
                    color="black",
                    alpha=0.6,
                    linewidth=1.0,
                )

        # Red highlight at selected wavelength index
        self.ax_bot.axvline(x_hl, color="red", alpha=0.6, linewidth=1.2)
        try:
            # For the highlight dot, prioritize the exact intensity from the raw
            # data trace if a specific candidate is selected.
            dot_y = float(yline[c])
            if self._selected_candidate_info and self._raw_I is not None:
                raw_r = self._selected_candidate_info.get("row_index")
                raw_c = self._selected_candidate_info.get("col_index")
                if raw_r is not None and raw_c is not None:
                    if 0 <= raw_r < self._raw_I.shape[0] and 0 <= raw_c < self._raw_I.shape[1]:
                        dot_y = self._raw_I[raw_r, raw_c]
            self.ax_bot.plot([x_hl], [dot_y], "o", color="red", markersize=4)
        except Exception:
            pass

        self.ax_bot.set_xlabel("Wavelength (nm)")
        self.ax_bot.set_ylabel("Intensity")
        self.ax_bot.set_title(f"Spectral slice({x_hl:.3f} nm, {y_hl:.1f} deg)")

        self.ax_bot.relim()
        self.ax_bot.autoscale_view(scalex=True, scaley=True)

    def _snap_to_cell(self, x: float, y: float) -> Tuple[int, int]:
        ny, nx = self._I.shape
        ix = int(np.searchsorted(self._x_edges, x, side="right") - 1)
        iy = int(np.searchsorted(self._y_edges, y, side="right") - 1)
        ix = int(np.clip(ix, 0, nx - 1))
        iy = int(np.clip(iy, 0, ny - 1))
        return ix, iy

    def _on_canvas_click(self, event) -> None:
        if event.inaxes is not self.ax_top:
            return
        if event.xdata is None or event.ydata is None:
            return
        x = float(event.xdata)
        y = float(event.ydata)
        ix, iy = self._snap_to_cell(x, y)
        self._sel_col = ix
        self._sel_row = iy
        self._selected_candidate_info = None  # Clear selected candidate on manual click
        self.txt_evidence.SetValue(self._format_evidence(self._general_evidence))
        self._ensure_highlight()
        self._update_highlight()
        self._update_slice()
        self.canvas.draw_idle()

    def _on_candidate_select(self, event) -> None:
        idx = int(self.chk_list.GetSelection())
        if idx < 0 or idx >= len(self._candidates):
            self._selected_candidate_info = None
            return
        c = self._candidates[idx]
        self._selected_candidate_info = c

        angle = c.get("angle_deg", None)
        wavelength = c.get("center_wavelength_nm", None)

        # The dialog displays the primitive matrix, so the highlight must be
        # positioned by matching the peak's physical angle and wavelength
        # against the axes of the primitive matrix (`_y_centers`, `_x_centers`).
        # Using the peak's `row_index` (from the raw matrix) is incorrect here.
        if angle is not None:
            self._sel_row = int(np.argmin(np.abs(self._y_centers - float(angle))))
        
        if wavelength is not None:
            self._sel_col = int(np.argmin(np.abs(self._x_centers - float(wavelength))))

        # Show peak-specific evidence
        peak_evidence = c.get("test_results", {})
        self.txt_evidence.SetValue(self._format_evidence(peak_evidence))
        self._ensure_highlight()
        self._update_highlight()
        self._update_slice()
        self.canvas.draw_idle()

    def _on_contrast_slider(self, event) -> None:
        self._apply_contrast_from_sliders()

    def _apply_contrast_from_sliders(self) -> None:
        if self._mesh is None:
            return
        a = float(self.slider_vmin.GetValue()) / 10.0
        b = float(self.slider_vmax.GetValue()) / 10.0
        if b < a:
            a, b = b, a

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
