"""Desktop dialog for importing Quantum ESPRESSO Raman frequencies."""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional, Sequence, Tuple

import wx
import wx.grid as wxgrid
from matplotlib.backends.backend_wxagg import FigureCanvasWxAgg as FigureCanvas
from matplotlib.figure import Figure

from data_structure import ExperimentSet, Run, RunType
import qe_raman


class QeRamanImportDialog(wx.Dialog):
    """Parse, map, preview, and create paired QE Raman runs."""

    COL_INCLUDE = 0
    COL_FREQUENCY = 1
    COL_MODES = 2
    COL_SYMMETRY = 3
    COL_ACTIVITY = 4
    COL_FIT = 5
    COL_XX = 6
    COL_YX = 7
    COL_FWHM = 8

    def __init__(
        self,
        parent: wx.Window,
        experiment: ExperimentSet,
        *,
        initial_path: str = "",
    ) -> None:
        super().__init__(
            parent,
            title="Import Quantum ESPRESSO Raman",
            style=wx.DEFAULT_DIALOG_STYLE | wx.RESIZE_BORDER,
        )
        self.experiment = experiment
        self.result: Optional[qe_raman.QePhResult] = None
        self.groups: List[qe_raman.QeModeGroup] = []
        self.rows: List[Dict[str, Any]] = []
        self.fit_peaks: List[Dict[str, Any]] = []
        self.adaptation_warnings: List[str] = []
        self.created_runs: Optional[Tuple[Run, Run]] = None
        self.created_view_title = ""
        self._populating_grid = False
        self._fit_choice_to_index: Dict[str, Optional[int]] = {"— unmatched —": None}

        self._build_controls(initial_path)
        self._bind_events()
        self.SetMinSize((1120, 790))
        self.SetSize((1220, 880))
        self.CentreOnParent()
        if initial_path and os.path.isfile(initial_path):
            wx.CallAfter(self._load_output)

    def _build_controls(self, initial_path: str) -> None:
        root = wx.BoxSizer(wx.VERTICAL)

        source_row = wx.BoxSizer(wx.HORIZONTAL)
        source_row.Add(wx.StaticText(self, label="QE ph.x output"), 0, wx.ALIGN_CENTER_VERTICAL | wx.RIGHT, 8)
        self.file_picker = wx.FilePickerCtrl(
            self,
            path=initial_path,
            message="Select Quantum ESPRESSO ph.x output",
            wildcard="QE output (*.out)|*.out|All files (*.*)|*.*",
            style=wx.FLP_OPEN | wx.FLP_FILE_MUST_EXIST | wx.FLP_USE_TEXTCTRL,
        )
        source_row.Add(self.file_picker, 1, wx.EXPAND | wx.RIGHT, 8)
        self.load_button = wx.Button(self, label="Load / Remap")
        source_row.Add(self.load_button, 0)
        root.Add(source_row, 0, wx.EXPAND | wx.ALL, 10)

        settings = wx.FlexGridSizer(rows=2, cols=7, vgap=5, hgap=10)
        settings.AddGrowableCol(0, 2)
        settings.AddGrowableCol(1, 2)
        self.nickname_ctrl = self._setting(settings, "Nickname prefix", wx.TextCtrl(self))
        fit_names = ["None (uniform XX/YX)"] + [run.nickname for run in self._fit_runs()]
        self.fit_choice = self._setting(settings, "Fit Parameters", wx.Choice(self, choices=fit_names))
        self.fit_choice.SetSelection(0)
        self.angle_ctrl = self._setting(settings, "Rotation (deg)", wx.TextCtrl(self, value="0", style=wx.TE_PROCESS_ENTER))
        self.tolerance_ctrl = self._setting(settings, "Match tol. (cm⁻¹)", wx.TextCtrl(self, value="15", style=wx.TE_PROCESS_ENTER))
        self.fwhm_ctrl = self._setting(settings, "Fallback FWHM", wx.TextCtrl(self, value="4", style=wx.TE_PROCESS_ENTER))
        self.xmin_ctrl = self._setting(settings, "Range min (auto)", wx.TextCtrl(self, value="", style=wx.TE_PROCESS_ENTER))
        self.xmax_ctrl = self._setting(settings, "Range max (auto)", wx.TextCtrl(self, value="", style=wx.TE_PROCESS_ENTER))
        self.step_ctrl = self._setting(settings, "Grid step (cm⁻¹)", wx.TextCtrl(self, value="0.25", style=wx.TE_PROCESS_ENTER))
        self.preview_button = self._setting(settings, "", wx.Button(self, label="Update Preview"))
        settings.AddSpacer(1)
        settings.AddSpacer(1)
        settings.AddSpacer(1)
        settings.AddSpacer(1)
        settings.AddSpacer(1)
        root.Add(settings, 0, wx.EXPAND | wx.LEFT | wx.RIGHT | wx.BOTTOM, 10)

        disclaimer = wx.StaticText(
            self,
            label=(
                "This is a frequency-only reconstruction. It does not calculate ab-initio QE XX/YX "
                "polarization intensities; Fit Parameters, when selected, supply only relative display heights and widths."
            ),
        )
        disclaimer.SetForegroundColour(wx.Colour(145, 72, 0))
        disclaimer.Wrap(1160)
        root.Add(disclaimer, 0, wx.EXPAND | wx.LEFT | wx.RIGHT | wx.BOTTOM, 10)

        self.grid = wxgrid.Grid(self)
        self.grid.CreateGrid(0, 9)
        labels = ["Use", "DFT cm⁻¹", "Modes", "Symmetry", "Activity", "Matched fitted peak", "XX height", "YX height", "FWHM cm⁻¹"]
        widths = [45, 95, 75, 90, 70, 260, 95, 95, 100]
        for column, (label, width) in enumerate(zip(labels, widths)):
            self.grid.SetColLabelValue(column, label)
            self.grid.SetColSize(column, width)
        self.grid.SetRowLabelSize(45)
        self.grid.EnableDragColSize(True)
        root.Add(self.grid, 1, wx.EXPAND | wx.LEFT | wx.RIGHT, 10)

        self.warning_text = wx.TextCtrl(
            self,
            style=wx.TE_MULTILINE | wx.TE_READONLY | wx.BORDER_SIMPLE,
            size=(-1, 72),
        )
        root.Add(wx.StaticText(self, label="Import notes and warnings"), 0, wx.LEFT | wx.RIGHT | wx.TOP, 10)
        root.Add(self.warning_text, 0, wx.EXPAND | wx.LEFT | wx.RIGHT | wx.BOTTOM, 10)

        self.figure = Figure(figsize=(8.0, 2.7), tight_layout=True)
        self.axes = self.figure.add_subplot(111)
        self.canvas = FigureCanvas(self, -1, self.figure)
        self.canvas.SetMinSize((-1, 235))
        root.Add(self.canvas, 0, wx.EXPAND | wx.LEFT | wx.RIGHT | wx.BOTTOM, 10)

        buttons = self.CreateButtonSizer(wx.OK | wx.CANCEL)
        self.ok_button = self.FindWindowById(wx.ID_OK)
        if self.ok_button is not None:
            self.ok_button.SetLabel("Create Runs")
            self.ok_button.Enable(False)
        root.Add(buttons, 0, wx.EXPAND | wx.ALL, 10)
        self.SetSizer(root)

    def _setting(self, parent: wx.Sizer, label: str, control: wx.Window) -> wx.Window:
        box = wx.BoxSizer(wx.VERTICAL)
        box.Add(wx.StaticText(self, label=label), 0, wx.BOTTOM, 2)
        box.Add(control, 0, wx.EXPAND)
        parent.Add(box, 1, wx.EXPAND)
        return control

    def _bind_events(self) -> None:
        self.file_picker.Bind(wx.EVT_FILEPICKER_CHANGED, self._on_source_changed)
        self.load_button.Bind(wx.EVT_BUTTON, self._on_load)
        self.fit_choice.Bind(wx.EVT_CHOICE, self._on_remap)
        self.preview_button.Bind(wx.EVT_BUTTON, self._on_preview)
        self.grid.Bind(wxgrid.EVT_GRID_CELL_CHANGED, self._on_grid_changed)
        for control in (self.angle_ctrl, self.tolerance_ctrl, self.fwhm_ctrl):
            control.Bind(wx.EVT_TEXT_ENTER, self._on_remap)
        for control in (self.xmin_ctrl, self.xmax_ctrl, self.step_ctrl):
            control.Bind(wx.EVT_TEXT_ENTER, self._on_preview)
        if self.ok_button is not None:
            self.ok_button.Bind(wx.EVT_BUTTON, self._on_create)

    def _fit_runs(self) -> List[Run]:
        return [run for run in self.experiment.runs.values() if run.run_type == RunType.FIT_PARAMS]

    def _selected_fit_run(self) -> Optional[Run]:
        index = self.fit_choice.GetSelection() - 1
        fit_runs = self._fit_runs()
        return fit_runs[index] if 0 <= index < len(fit_runs) else None

    def _selected_fit_state(self) -> Optional[Dict[str, Any]]:
        run = self._selected_fit_run()
        if run is None:
            return None
        state = (run.metadata or {}).get("fit_state") or (run.metadata or {}).get("map_fit_state")
        if not state:
            raise qe_raman.QeRamanError(
                f"Fit Parameters run '{run.nickname}' does not contain a saved fit state."
            )
        return state

    @staticmethod
    def _float(control: wx.TextCtrl, label: str, *, optional: bool = False) -> Optional[float]:
        text = control.GetValue().strip()
        if optional and not text:
            return None
        try:
            return float(text)
        except ValueError as exc:
            raise qe_raman.QeRamanError(f"{label} must be a number.") from exc

    def _options(self) -> Dict[str, Optional[float]]:
        return {
            "angle": self._float(self.angle_ctrl, "Sample rotation"),
            "tolerance": self._float(self.tolerance_ctrl, "Matching tolerance"),
            "fwhm": self._float(self.fwhm_ctrl, "Fallback FWHM"),
            "x_min": self._float(self.xmin_ctrl, "Range minimum", optional=True),
            "x_max": self._float(self.xmax_ctrl, "Range maximum", optional=True),
            "step": self._float(self.step_ctrl, "Grid step"),
        }

    def _on_source_changed(self, _event: wx.Event) -> None:
        self.result = None
        if self.ok_button is not None:
            self.ok_button.Enable(False)
        wx.CallAfter(self._load_output)

    def _on_load(self, _event: wx.Event) -> None:
        self._load_output()

    def _load_output(self) -> bool:
        path = self.file_picker.GetPath().strip()
        if not path:
            wx.MessageBox("Choose a Quantum ESPRESSO ph.x output file.", "QE Raman Import", wx.OK | wx.ICON_INFORMATION, self)
            return False
        try:
            result = qe_raman.parse_qe_ph_output(path)
            groups = qe_raman.usable_mode_groups(result)
            self.result = result
            self.groups = groups
            if not self.nickname_ctrl.GetValue().strip():
                stem = os.path.splitext(os.path.basename(path))[0]
                self.nickname_ctrl.SetValue(stem[:-3] if stem.lower().endswith("_ph") else stem)
            self._remap_rows()
            if self.ok_button is not None:
                self.ok_button.Enable(True)
            return True
        except Exception as exc:
            self.result = None
            if self.ok_button is not None:
                self.ok_button.Enable(False)
            self._show_error(exc)
            return False

    def _on_remap(self, _event: wx.Event) -> None:
        if self.result is None:
            self._load_output()
            return
        try:
            self._remap_rows()
        except Exception as exc:
            self._show_error(exc)

    def _remap_rows(self, mapping: Optional[Dict[int, Optional[int]]] = None) -> None:
        options = self._options()
        rows, fit_peaks, warnings = qe_raman.adapt_peak_rows(
            self.groups,
            self._selected_fit_state(),
            angle_deg=float(options["angle"]),
            default_fwhm_cm1=float(options["fwhm"]),
            match_tolerance_cm1=float(options["tolerance"]),
            mapping=mapping,
        )
        self.rows = rows
        self.fit_peaks = fit_peaks
        self.adaptation_warnings = warnings
        self._populate_grid()
        self._update_notes()
        self._update_preview()

    def _fit_label(self, fit_index: Optional[int]) -> str:
        if fit_index is None or fit_index < 0 or fit_index >= len(self.fit_peaks):
            return "— unmatched —"
        peak = self.fit_peaks[fit_index]
        center = peak.get("center_cm1")
        suffix = f" ({float(center):.3f} cm⁻¹)" if center is not None else ""
        return f"{peak['display_name']}{suffix}"

    def _populate_grid(self) -> None:
        self._populating_grid = True
        try:
            current_rows = self.grid.GetNumberRows()
            target_rows = len(self.rows)
            if current_rows < target_rows:
                self.grid.AppendRows(target_rows - current_rows)
            elif current_rows > target_rows:
                self.grid.DeleteRows(0, current_rows - target_rows)
            choices = ["— unmatched —"] + [self._fit_label(index) for index in range(len(self.fit_peaks))]
            self._fit_choice_to_index = {label: (None if index == 0 else index - 1) for index, label in enumerate(choices)}
            for row_index, row in enumerate(self.rows):
                self.grid.SetCellValue(row_index, self.COL_INCLUDE, "1" if row.get("include", True) else "0")
                self.grid.SetCellRenderer(row_index, self.COL_INCLUDE, wxgrid.GridCellBoolRenderer())
                self.grid.SetCellEditor(row_index, self.COL_INCLUDE, wxgrid.GridCellBoolEditor())
                self.grid.SetCellValue(row_index, self.COL_FREQUENCY, f"{float(row['dft_frequency_cm1']):.6f}")
                self.grid.SetCellValue(row_index, self.COL_MODES, str(row.get("mode_range", "")))
                self.grid.SetCellValue(row_index, self.COL_SYMMETRY, str(row.get("symmetry", "")))
                self.grid.SetCellValue(row_index, self.COL_ACTIVITY, str(row.get("activity", "")))
                self.grid.SetCellValue(row_index, self.COL_FIT, self._fit_label(row.get("matched_fit_index")))
                self.grid.SetCellEditor(row_index, self.COL_FIT, wxgrid.GridCellChoiceEditor(choices, allowOthers=False))
                self.grid.SetCellValue(row_index, self.COL_XX, f"{float(row['xx_height']):.8g}")
                self.grid.SetCellValue(row_index, self.COL_YX, f"{float(row['yx_height']):.8g}")
                self.grid.SetCellValue(row_index, self.COL_FWHM, f"{float(row['fwhm_cm1']):.8g}")
                for column in (self.COL_FREQUENCY, self.COL_MODES, self.COL_SYMMETRY, self.COL_ACTIVITY):
                    self.grid.SetReadOnly(row_index, column, True)
                    self.grid.SetCellBackgroundColour(row_index, column, wx.Colour(242, 242, 242))
        finally:
            self._populating_grid = False

    def _rows_from_grid(self) -> List[Dict[str, Any]]:
        updated = [dict(row) for row in self.rows]
        for row_index, row in enumerate(updated):
            row["include"] = self.grid.GetCellValue(row_index, self.COL_INCLUDE).strip().lower() in {"1", "true", "yes"}
            label = self.grid.GetCellValue(row_index, self.COL_FIT)
            row["matched_fit_index"] = self._fit_choice_to_index.get(label)
            if row["matched_fit_index"] is None:
                row["matched_fit_name"] = ""
            row["xx_height"] = float(self.grid.GetCellValue(row_index, self.COL_XX))
            row["yx_height"] = float(self.grid.GetCellValue(row_index, self.COL_YX))
            row["fwhm_cm1"] = float(self.grid.GetCellValue(row_index, self.COL_FWHM))
        qe_raman.validate_peak_rows(updated, fit_peak_count=len(self.fit_peaks))
        return updated

    def _on_grid_changed(self, event: wxgrid.GridEvent) -> None:
        if self._populating_grid:
            event.Skip()
            return
        try:
            edited_rows = self._rows_from_grid()
            if event.GetCol() == self.COL_FIT:
                mapping = {
                    index: row.get("matched_fit_index")
                    for index, row in enumerate(edited_rows)
                    if row.get("matched_fit_index") is not None
                }
                include_flags = [row["include"] for row in edited_rows]
                self._remap_rows(mapping=mapping)
                for row, include in zip(self.rows, include_flags):
                    row["include"] = include
                self._populate_grid()
            else:
                self.rows = edited_rows
                self._update_preview()
        except Exception as exc:
            self._show_error(exc)
            self._populate_grid()
        event.Skip()

    def _on_preview(self, _event: wx.Event) -> None:
        try:
            self.rows = self._rows_from_grid()
            self._update_preview()
        except Exception as exc:
            self._show_error(exc)

    def _update_notes(self) -> None:
        notes: List[str] = []
        if self.result is not None:
            notes.extend(self.result.warnings)
        notes.extend(self.adaptation_warnings)
        if self.result is not None:
            notes.insert(
                0,
                f"QE {self.result.qe_version}; {self.result.mode_count} modes; "
                f"{len(self.groups)} positive Raman peaks; point group {self.result.point_group or '(not reported)'}."
            )
        if not notes:
            notes.append("Ready. Peak centers come from QE; displayed XX/YX intensities are relative reconstructions.")
        self.warning_text.SetValue("\n".join(dict.fromkeys(notes)))

    def _update_preview(self) -> None:
        options = self._options()
        x, xx, yx, _settings = qe_raman.build_spectra(
            self.rows,
            x_min_cm1=options["x_min"],
            x_max_cm1=options["x_max"],
            step_cm1=float(options["step"]),
        )
        self.axes.clear()
        self.axes.plot(x, xx, label="XX", color="#2457A7", linewidth=1.35)
        self.axes.plot(x, yx, label="YX", color="#C34232", linewidth=1.35)
        self.axes.set_xlabel("Raman shift (cm⁻¹)")
        self.axes.set_ylabel("Relative intensity")
        self.axes.set_ylim(bottom=0)
        self.axes.legend(loc="upper right", frameon=False)
        self.axes.grid(alpha=0.18)
        self.canvas.draw_idle()

    def _on_create(self, _event: wx.Event) -> None:
        try:
            if self.result is None and not self._load_output():
                return
            self.rows = self._rows_from_grid()
            options = self._options()
            nickname = self.nickname_ctrl.GetValue().strip()
            fit_run = self._selected_fit_run()
            self.created_runs = qe_raman.make_qe_raman_runs(
                self.result,
                nickname,
                self.rows,
                angle_deg=float(options["angle"]),
                match_tolerance_cm1=float(options["tolerance"]),
                default_fwhm_cm1=float(options["fwhm"]),
                x_min_cm1=options["x_min"],
                x_max_cm1=options["x_max"],
                step_cm1=float(options["step"]),
                fit_params_run_id=fit_run.id if fit_run else None,
                warnings=self.adaptation_warnings,
            )
            self.created_view_title = f"{nickname} DFT Raman"
            self.EndModal(wx.ID_OK)
        except Exception as exc:
            self._show_error(exc)

    def _show_error(self, exc: Exception) -> None:
        wx.MessageBox(str(exc), "QE Raman Import", wx.OK | wx.ICON_ERROR, self)

    def get_created_runs(self) -> Tuple[Run, Run]:
        if self.created_runs is None:
            raise RuntimeError("QE Raman runs have not been created.")
        return self.created_runs
