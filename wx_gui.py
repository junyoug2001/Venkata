"""
Minimal VESTA-like GUI skeleton using wxPython, now wired to the
data_structure ExperimentSet / Run model for early visualization tests.

Usage
-----
python wx_gui.py file1.csv file2.csv

- Creates a single ExperimentSet.
- Builds up to two Run objects from the provided files.
- Populates:
    * Files tab   : simple directory browser + file preview.
    * Runs tab    : list of runs in the current ExperimentSet.
    * Experiment  : tree view of experiment metadata and runs.
    * Log         : textual log of operations.

Layout
------
- Main horizontal splitter:
    [ Left controls ] | [ Right view notebook ]

- Left controls:
    - Vertical splitter:
        * Top:   Notebook (Files / Runs / Experiment / Log)
        * Bottom Notebook (Preview / Curve Fit)

- Right view notebook:
    - Multiple "View N" tabs (initially one tab).
    - Menu: View -> New View Tab (adds a new tab).

Notes
-----
- This is still a structural GUI; plotting is not implemented yet.
- The purpose is to validate the early pipeline:
    CLI args -> ExperimentSet / Runs -> GUI panels.
"""

from __future__ import annotations

import copy
import os
import analysis
import polar_area_fitting
import fit_overlay
from merge_runs_gui import MergeRunsDialog
from two_d_map_fitting_gui import MapFittingDialog
from normalization_gui import NormalizeDialog
from qe_raman_gui import QeRamanImportDialog
from typing import List, Optional, Dict, Any, Tuple

import wx
from wx.lib.agw import aui
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
    cm1_to_unit,
    is_hidden_run,
    normalize_spectral_unit,
    new_experiment_id,
    new_view_id,
    new_run_id,
    RunType,
    unit_to_cm1,
)
from plotting import RamanPlotter2d, AngularPlotter, SlicePlotter

from wx_left_panel import FilesPanel, RunsPanel, ExperimentPanel, LogPanel
from wx_left_lower_panel import PreviewPanel, FitResultsPanel, PlotConfigPanel, AppearancesPanel, PreferencesPanel
from wx_right_panel import RamanToolbar, ViewPanel
from config_manager import config


class PolarAreaFittingDialog(wx.Dialog):
    def __init__(
        self,
        parent,
        runs: List[Run],
        default_targets: List[float],
        default_colors: List[str],
        default_output: str,
        default_unit: str = "cm-1",
    ):
        super().__init__(parent, title="Generate Polar Area Fittings", size=(920, 680))
        self.default_colors = default_colors
        self.runs = list(runs)
        self.color_cell_editors = []
        self.row_color_editors = []
        self.current_unit = normalize_spectral_unit(default_unit, "cm-1")

        sizer = wx.BoxSizer(wx.VERTICAL)
        run_label = ", ".join(run.nickname for run in runs)
        sizer.Add(wx.StaticText(self, label=f"Runs: {run_label}"), 0, wx.ALL | wx.EXPAND, 10)

        grid = wx.FlexGridSizer(0, 2, 6, 8)
        grid.AddGrowableCol(1, 1)

        self.choice_unit = wx.Choice(self, choices=["cm-1", "meV"])
        self.choice_unit.SetStringSelection(self.current_unit)
        self.choice_unit.Bind(wx.EVT_CHOICE, self._on_unit_change)
        grid.Add(wx.StaticText(self, label="Spectral unit:"), 0, wx.ALIGN_CENTER_VERTICAL)
        grid.Add(self.choice_unit, 0, wx.EXPAND)

        display_targets = cm1_to_unit(np.asarray(default_targets, dtype=float), self.current_unit)
        target_text = ", ".join(f"{v:.6g}" for v in display_targets)
        self.txt_targets = wx.TextCtrl(self, value=target_text)
        self.lbl_targets = wx.StaticText(self, label="")
        grid.Add(self.lbl_targets, 0, wx.ALIGN_CENTER_VERTICAL)
        grid.Add(self.txt_targets, 1, wx.EXPAND)

        self.file_output = wx.FilePickerCtrl(
            self,
            path=default_output,
            message="Save polar fit PDF",
            wildcard="PDF files (*.pdf)|*.pdf",
            style=wx.FLP_SAVE | wx.FLP_OVERWRITE_PROMPT | wx.FLP_USE_TEXTCTRL,
        )
        grid.Add(wx.StaticText(self, label="Output PDF:"), 0, wx.ALIGN_CENTER_VERTICAL)
        grid.Add(self.file_output, 1, wx.EXPAND)

        peak_window = float(cm1_to_unit(8.0, self.current_unit))
        self.txt_peak_window = wx.TextCtrl(self, value=f"{peak_window:.6g}")
        self.lbl_peak_window = wx.StaticText(self, label="")
        grid.Add(self.lbl_peak_window, 0, wx.ALIGN_CENTER_VERTICAL)
        grid.Add(self.txt_peak_window, 0, wx.EXPAND)

        center_window = float(cm1_to_unit(2.0, self.current_unit))
        self.txt_center_window = wx.TextCtrl(self, value=f"{center_window:.6g}")
        self.lbl_center_window = wx.StaticText(self, label="")
        grid.Add(self.lbl_center_window, 0, wx.ALIGN_CENTER_VERTICAL)
        grid.Add(self.txt_center_window, 0, wx.EXPAND)

        self.chk_normalize = wx.CheckBox(self, label="Normalize each polar plot")
        self.chk_normalize.SetValue(True)
        grid.AddSpacer(1)
        grid.Add(self.chk_normalize, 0, wx.EXPAND)

        self.chk_use_cache = wx.CheckBox(self, label="Use saved inspection row-fit cache")
        self.chk_use_cache.SetValue(True)
        self.chk_use_cache.SetToolTip(
            "Export the exact areas and Gammas saved from Validate/inspection. "
            "Turn this off only to run a separate row fit from the raw map."
        )
        grid.AddSpacer(1)
        grid.Add(self.chk_use_cache, 0, wx.EXPAND)

        sizer.Add(grid, 0, wx.ALL | wx.EXPAND, 10)

        color_header = wx.BoxSizer(wx.HORIZONTAL)
        color_header.Add(wx.StaticText(self, label="Plot colors by fit and polarization:"), 0, wx.ALIGN_CENTER_VERTICAL | wx.RIGHT, 12)
        btn_refresh_colors = wx.Button(self, label="Refresh rows from target peaks")
        color_header.Add(btn_refresh_colors, 0)
        sizer.Add(color_header, 0, wx.LEFT | wx.RIGHT | wx.TOP, 10)

        self.color_scroll = wx.ScrolledWindow(self, style=wx.VSCROLL | wx.BORDER_SIMPLE)
        self.color_scroll.SetScrollRate(0, 10)
        self.color_table_sizer = wx.FlexGridSizer(0, 4, 6, 10)
        self.color_table_sizer.AddGrowableCol(1, 1)
        self.color_table_sizer.AddGrowableCol(2, 1)
        self.color_table_sizer.AddGrowableCol(3, 1)
        self.color_scroll.SetSizer(self.color_table_sizer)
        sizer.Add(self.color_scroll, 1, wx.ALL | wx.EXPAND, 10)

        btn_refresh_colors.Bind(wx.EVT_BUTTON, self._on_refresh_color_rows)
        self._update_unit_labels()
        self._rebuild_color_table(list(display_targets))

        sizer.Add(self.CreateButtonSizer(wx.OK | wx.CANCEL), 0, wx.ALL | wx.ALIGN_RIGHT, 10)
        self.SetSizer(sizer)

    @staticmethod
    def _html_color(colour: wx.Colour) -> str:
        return colour.GetAsString(wx.C2S_HTML_SYNTAX).upper()

    def _make_color_editor(self, parent, color: str, on_change=None):
        panel = wx.Panel(parent)
        editor_sizer = wx.BoxSizer(wx.HORIZONTAL)
        picker = wx.ColourPickerCtrl(panel, colour=wx.Colour(color))
        text = wx.TextCtrl(panel, value=self._html_color(picker.GetColour()), style=wx.TE_PROCESS_ENTER, size=(88, -1))
        editor_sizer.Add(picker, 0, wx.ALIGN_CENTER_VERTICAL | wx.RIGHT, 5)
        editor_sizer.Add(text, 1, wx.ALIGN_CENTER_VERTICAL)
        panel.SetSizer(editor_sizer)
        editor = {"panel": panel, "picker": picker, "text": text}

        def picker_changed(event):
            value = self._html_color(event.GetColour())
            text.ChangeValue(value)
            text.SetBackgroundColour(wx.NullColour)
            if on_change:
                on_change(value)

        def text_changed(event):
            value = text.GetValue().strip()
            colour = wx.Colour(value)
            if colour.IsOk():
                picker.SetColour(colour)
                normalized = self._html_color(colour)
                text.ChangeValue(normalized)
                text.SetBackgroundColour(wx.NullColour)
                if on_change:
                    on_change(normalized)
            else:
                text.SetBackgroundColour(wx.Colour(255, 220, 220))
            text.Refresh()
            event.Skip()

        picker.Bind(wx.EVT_COLOURPICKER_CHANGED, picker_changed)
        text.Bind(wx.EVT_TEXT_ENTER, text_changed)
        text.Bind(wx.EVT_KILL_FOCUS, text_changed)
        return editor

    def _set_color_editor(self, editor, color: str) -> None:
        colour = wx.Colour(color)
        if not colour.IsOk():
            return
        editor["picker"].SetColour(colour)
        editor["text"].ChangeValue(self._html_color(colour))
        editor["text"].SetBackgroundColour(wx.NullColour)
        editor["text"].Refresh()

    def _set_row_color(self, row_idx: int, color: str) -> None:
        if row_idx >= len(self.color_cell_editors):
            return
        for editor in self.color_cell_editors[row_idx].values():
            self._set_color_editor(editor, color)

    def _set_column_color(self, polarization: str, color: str) -> None:
        for row in self.color_cell_editors:
            self._set_color_editor(row[polarization], color)

    def _update_unit_labels(self) -> None:
        unit_label = "meV" if self.current_unit == "meV" else "cm-1"
        self.lbl_targets.SetLabel(f"Target peaks ({unit_label}):")
        self.lbl_peak_window.SetLabel(f"Peak window ({unit_label}):")
        self.lbl_center_window.SetLabel(f"Center tolerance ({unit_label}):")

    def _on_unit_change(self, event) -> None:
        new_unit = normalize_spectral_unit(self.choice_unit.GetStringSelection(), self.current_unit)
        if new_unit == self.current_unit:
            return
        try:
            targets_cm1 = unit_to_cm1(np.asarray(self._parse_targets(), dtype=float), self.current_unit)
            peak_window_cm1 = float(unit_to_cm1(float(self.txt_peak_window.GetValue()), self.current_unit))
            center_window_cm1 = float(unit_to_cm1(float(self.txt_center_window.GetValue()), self.current_unit))
        except Exception:
            wx.MessageBox("Enter valid target peaks and fitting windows before changing units.", "Polar Area Fittings", wx.OK | wx.ICON_ERROR)
            self.choice_unit.SetStringSelection(self.current_unit)
            return
        self.current_unit = new_unit
        display_targets = cm1_to_unit(targets_cm1, new_unit)
        self.txt_targets.ChangeValue(", ".join(f"{value:.6g}" for value in display_targets))
        self.txt_peak_window.ChangeValue(f"{float(cm1_to_unit(peak_window_cm1, new_unit)):.6g}")
        self.txt_center_window.ChangeValue(f"{float(cm1_to_unit(center_window_cm1, new_unit)):.6g}")
        self._update_unit_labels()
        self._rebuild_color_table(list(display_targets))

    def _run_label_for_config(self, config_name: str) -> str:
        aliases = ("xx", "parallel") if config_name == "parallel" else ("yx", "cross")
        for run in self.runs:
            nickname = (run.nickname or "").lower()
            if any(alias in nickname for alias in aliases):
                return run.nickname
        index = 0 if config_name == "parallel" else 1
        return self.runs[index].nickname if index < len(self.runs) else ""

    def _rebuild_color_table(self, targets: List[float]) -> None:
        previous = []
        for row in self.color_cell_editors:
            previous.append({key: self._html_color(editor["picker"].GetColour()) for key, editor in row.items()})

        self.color_table_sizer.Clear(delete_windows=True)
        self.color_cell_editors = []
        self.row_color_editors = []

        self.color_table_sizer.Add(wx.StaticText(self.color_scroll, label="Fit / peak"), 0, wx.ALIGN_CENTER_VERTICAL)
        self.color_table_sizer.Add(wx.StaticText(self.color_scroll, label="Common row color"), 0, wx.ALIGN_CENTER_VERTICAL)
        for config_name, heading in (("parallel", "XX"), ("cross", "YX")):
            run_label = self._run_label_for_config(config_name)
            label = heading if not run_label else f"{heading}  ({run_label})"
            column_box = wx.BoxSizer(wx.VERTICAL)
            column_box.Add(wx.StaticText(self.color_scroll, label=label), 0, wx.ALIGN_CENTER_HORIZONTAL | wx.BOTTOM, 3)
            initial = self.default_colors[0] if self.default_colors else "#000000"
            editor = self._make_color_editor(
                self.color_scroll,
                initial,
                on_change=lambda color, cfg=config_name: self._set_column_color(cfg, color),
            )
            column_box.Add(editor["panel"], 0, wx.EXPAND)
            self.color_table_sizer.Add(column_box, 1, wx.EXPAND)

        for idx, target in enumerate(targets):
            base_color = self.default_colors[idx] if idx < len(self.default_colors) else "#000000"
            saved = previous[idx] if idx < len(previous) else {}
            self.color_table_sizer.Add(
                wx.StaticText(self.color_scroll, label=f"Fit {idx + 1}: {target:.6g} {self.current_unit}"),
                0,
                wx.ALIGN_CENTER_VERTICAL,
            )
            row_editor = self._make_color_editor(
                self.color_scroll,
                base_color,
                on_change=lambda color, row_idx=idx: self._set_row_color(row_idx, color),
            )
            self.row_color_editors.append(row_editor)
            self.color_table_sizer.Add(row_editor["panel"], 1, wx.EXPAND)

            cells = {}
            for config_name in ("parallel", "cross"):
                editor = self._make_color_editor(self.color_scroll, saved.get(config_name, base_color))
                cells[config_name] = editor
                self.color_table_sizer.Add(editor["panel"], 1, wx.EXPAND)
            self.color_cell_editors.append(cells)

        self.color_scroll.Layout()
        self.color_scroll.FitInside()
        self.Layout()

    def _parse_targets(self) -> List[float]:
        target_raw = self.txt_targets.GetValue().replace(",", " ").split()
        return [float(v) for v in target_raw]

    def _on_refresh_color_rows(self, event) -> None:
        try:
            targets = self._parse_targets()
        except ValueError:
            wx.MessageBox("Enter valid numeric target peaks before refreshing the color table.", "Polar Area Fittings", wx.OK | wx.ICON_ERROR)
            return
        self._rebuild_color_table(targets)

    def _read_color_editor(self, editor) -> str:
        value = editor["text"].GetValue().strip()
        colour = wx.Colour(value)
        if not colour.IsOk():
            raise ValueError(f"Invalid RGB hex color: {value}")
        return self._html_color(colour)

    def get_values(self) -> Dict[str, Any]:
        display_targets = self._parse_targets()
        if len(display_targets) != len(self.color_cell_editors):
            self._rebuild_color_table(display_targets)
        targets = [float(value) for value in unit_to_cm1(np.asarray(display_targets, dtype=float), self.current_unit)]
        colors = [
            {
                "parallel": self._read_color_editor(row["parallel"]),
                "cross": self._read_color_editor(row["cross"]),
            }
            for row in self.color_cell_editors
        ]
        return {
            "targets": targets,
            "output": self.file_output.GetPath(),
            "peak_window": float(unit_to_cm1(float(self.txt_peak_window.GetValue()), self.current_unit)),
            "center_window": float(unit_to_cm1(float(self.txt_center_window.GetValue()), self.current_unit)),
            "unit": self.current_unit,
            "normalize": self.chk_normalize.GetValue(),
            "use_cache": self.chk_use_cache.GetValue(),
            "colors": colors,
        }


class FitParamsChoiceDialog(wx.Dialog):
    def __init__(self, parent, choices: List[Run]):
        super().__init__(parent, title="Choose Previous Fit Parameters", size=(620, 320))
        self.choices = choices
        sizer = wx.BoxSizer(wx.VERTICAL)
        sizer.Add(wx.StaticText(self, label="Multiple previous fit parameter sets match these run(s)."), 0, wx.ALL | wx.EXPAND, 8)
        self.list = wx.ListBox(self)
        for run in choices:
            state = run.metadata.get("fit_state") or {}
            source_ids = ", ".join(str(v) for v in (run.metadata.get("source_run_ids") or run.metadata.get("fit_params_source_run_ids") or []))
            peaks = len(state.get("peaks") or [])
            x0 = state.get("x_min_limit", "")
            x1 = state.get("x_max_limit", "")
            self.list.Append(f"{run.nickname} | sources: {source_ids} | peaks: {peaks} | range: {x0:g}-{x1:g}" if isinstance(x0, (int, float)) and isinstance(x1, (int, float)) else f"{run.nickname} | sources: {source_ids} | peaks: {peaks}")
        if choices:
            self.list.SetSelection(0)
        sizer.Add(self.list, 1, wx.ALL | wx.EXPAND, 8)
        sizer.Add(self.CreateButtonSizer(wx.OK | wx.CANCEL), 0, wx.ALL | wx.ALIGN_RIGHT, 8)
        self.SetSizer(sizer)

    def selected_run(self) -> Optional[Run]:
        idx = self.list.GetSelection()
        if idx == wx.NOT_FOUND:
            return None
        return self.choices[idx]


class AngleRotationDialog(wx.Dialog):
    def __init__(self, parent, run: Run):
        super().__init__(parent, title="Rotate 360° Dataset", size=(460, 330))
        self.run = run
        angles = np.asarray(run.angle_values, dtype=float)
        self.angle_min = float(np.nanmin(angles))
        self.angle_max = float(np.nanmax(angles))
        self.span = self.angle_max - self.angle_min

        existing = (run.metadata or {}).get("angle_rotation") or {}
        sizer = wx.BoxSizer(wx.VERTICAL)
        sizer.Add(wx.StaticText(self, label=f"Run: {run.nickname}"), 0, wx.ALL | wx.EXPAND, 8)
        sizer.Add(wx.StaticText(self, label=f"Angle range: {self.angle_min:.6g} to {self.angle_max:.6g} deg ({angles.size} rows)"), 0, wx.LEFT | wx.RIGHT | wx.BOTTOM, 8)

        self.chk_enable = wx.CheckBox(self, label="Enable rotation")
        self.chk_enable.SetValue(bool(existing.get("enabled", True)))
        sizer.Add(self.chk_enable, 0, wx.LEFT | wx.RIGHT | wx.BOTTOM, 8)

        self.chk_cyclic = wx.CheckBox(self, label="Cyclic rotate")
        self.chk_cyclic.SetValue(bool(existing.get("cyclic_rotate", True)))
        sizer.Add(self.chk_cyclic, 0, wx.LEFT | wx.RIGHT | wx.BOTTOM, 8)

        grid = wx.FlexGridSizer(0, 2, 6, 8)
        grid.AddGrowableCol(1, 1)
        self.txt_anchor = wx.TextCtrl(self, value=f"{float(existing.get('anchor_deg', 0.0)):.6g}")
        grid.Add(wx.StaticText(self, label="Anchor angle:"), 0, wx.ALIGN_CENTER_VERTICAL)
        grid.Add(self.txt_anchor, 1, wx.EXPAND)

        default_start = existing.get("window_start_deg")
        if default_start is None:
            default_start = self.angle_min
        self.txt_window_start = wx.TextCtrl(self, value=f"{float(default_start):.6g}")
        grid.Add(wx.StaticText(self, label="Window start:"), 0, wx.ALIGN_CENTER_VERTICAL)
        grid.Add(self.txt_window_start, 1, wx.EXPAND)
        sizer.Add(grid, 0, wx.ALL | wx.EXPAND, 8)

        self.info = wx.StaticText(self, label="")
        sizer.Add(self.info, 0, wx.ALL | wx.EXPAND, 8)

        if self.span <= 360.0 + 1e-9:
            self.txt_window_start.Disable()

        btns = self.CreateButtonSizer(wx.OK | wx.CANCEL)
        sizer.Add(btns, 0, wx.ALL | wx.ALIGN_RIGHT, 8)
        self.SetSizer(sizer)

        self.chk_enable.Bind(wx.EVT_CHECKBOX, self.on_change)
        self.chk_cyclic.Bind(wx.EVT_CHECKBOX, self.on_change)
        self.txt_anchor.Bind(wx.EVT_TEXT, self.on_change)
        self.txt_window_start.Bind(wx.EVT_TEXT, self.on_change)
        self.on_change(None)

    def on_change(self, event):
        if not self.chk_enable.GetValue():
            self.info.SetLabel("Rotation will be disabled.")
            return
        try:
            settings = self.get_settings(validate=True)
            self.info.SetLabel(settings.get("summary", ""))
        except Exception as exc:
            self.info.SetLabel(f"Invalid rotation: {exc}")

    def get_settings(self, validate: bool = False) -> Dict[str, Any]:
        if not self.chk_enable.GetValue():
            return {"enabled": False, "summary": "Rotation disabled"}
        anchor = float(self.txt_anchor.GetValue())
        window_start = None
        if self.span > 360.0 + 1e-9:
            window_start = float(self.txt_window_start.GetValue())
        settings = analysis.build_angle_rotation_settings(
            self.run.angle_values,
            anchor_deg=anchor,
            window_start_deg=window_start,
            period_deg=360.0,
            cyclic_rotate=self.chk_cyclic.GetValue(),
        )
        if validate and self.span > 360.0 + 1e-9 and settings.get("included_rows", 0) <= 1:
            raise ValueError("Window includes too few rows.")
        return settings


# -----------------------------
# Main frame
# -----------------------------


class MainFrame(wx.Frame):
    def __init__(self, initial_files: Optional[List[str]] = None):
        size = config.get("window_size", [1400, 800])
        super().__init__(
            None,
            title="Venkata - GUI Based Raman Analysis",
            size=(size[0], size[1]),
        )

        # Core experiment model
        self.experiment = ExperimentSet(id=new_experiment_id())

        # Simple counter for naming view tabs
        self._next_view_index = 1

        # Map notebook page index -> view_id in ExperimentSet.views
        self._view_page_to_id = {}
        # Map view_id -> ViewPanel instance
        self._view_id_to_panel = {}

        self._build_menu_bar()
        self._build_layout()

        # Status bar
        self.CreateStatusBar()

        self.Bind(wx.EVT_CLOSE, self.on_close)

        # Load any initial files passed from CLI
        if initial_files:
            self.load_initial_files(initial_files)
        else:
            last_dir = config.get("last_directory", "")
            if last_dir and os.path.isdir(last_dir):
                self.files_panel.set_directory(last_dir)

        self.Centre()
        self.Show()

    def on_close(self, event):
        """Save settings and close the app."""
        self._persist_view_panel_states()
        size = self.GetSize()
        config.set("window_size", [size.width, size.height])
        config.save()
        self.Destroy()

    # -------- menu --------

    def _build_menu_bar(self):
        menubar = wx.MenuBar()

        # File menu
        file_menu = wx.Menu()
        item_open = file_menu.Append(wx.ID_OPEN, "Open Experiment...\tCtrl-O")
        item_save = file_menu.Append(wx.ID_SAVE, "Save Experiment...\tCtrl-S")
        item_export_igor = file_menu.Append(wx.ID_ANY, "Export to Igor Pro...")
        file_menu.AppendSeparator()
        item_import = file_menu.Append(wx.ID_ANY, "Import Run...\tCtrl-I")
        item_import_qe = file_menu.Append(wx.ID_ANY, "Import Quantum ESPRESSO Raman...")
        item_new_formula = file_menu.Append(wx.ID_ANY, "New Formula Run...")
        item_merge = file_menu.Append(wx.ID_ANY, "Merge Run...\tCtrl-Shift-I")
        item_calc = file_menu.Append(wx.ID_ANY, "Calculate from run...")
        file_menu.AppendSeparator()
        item_quit = file_menu.Append(wx.ID_EXIT, "Quit\tCtrl-Q")

        self.Bind(wx.EVT_MENU, self.on_open_experiment, item_open)
        self.Bind(wx.EVT_MENU, self.on_save_experiment, item_save)
        self.Bind(wx.EVT_MENU, self.on_export_igor, item_export_igor)
        self.Bind(wx.EVT_MENU, self.on_import_run_dialog, item_import)
        self.Bind(wx.EVT_MENU, self.on_import_qe_raman, item_import_qe)
        self.Bind(wx.EVT_MENU, self.on_new_formula_run, item_new_formula)
        self.Bind(wx.EVT_MENU, self.on_merge_run_dialog, item_merge)
        self.Bind(wx.EVT_MENU, self.on_calculate_from_run, item_calc)
        self.Bind(wx.EVT_MENU, self.on_quit, item_quit)
        menubar.Append(file_menu, "&File")

        # View menu
        view_menu = wx.Menu()
        item_new_view = view_menu.Append(wx.ID_ANY, "New View Tab\tCtrl+T")
        item_close_view = view_menu.Append(wx.ID_ANY, "Close View Tab\tCtrl+W")

        self.Bind(wx.EVT_MENU, self.on_new_view, item_new_view)
        self.Bind(wx.EVT_MENU, self.on_close_current_view, item_close_view)
        menubar.Append(view_menu, "&View")

        # Tools menu
        tools_menu = wx.Menu()
        item_2d_fit = tools_menu.Append(wx.ID_ANY, "Make 2D Fit...")
        item_rotate = tools_menu.Append(wx.ID_ANY, "Rotate 360° Dataset...")
        item_polar_fit = tools_menu.Append(wx.ID_ANY, "Generate Polar Area Fittings...")
        item_normalize = tools_menu.Append(wx.ID_ANY, "Normalize...")
        self.Bind(wx.EVT_MENU, self.on_2d_map_fitting, item_2d_fit)
        self.Bind(wx.EVT_MENU, self.on_rotate_360_dataset, item_rotate)
        self.Bind(wx.EVT_MENU, self.on_generate_polar_area_fittings, item_polar_fit)
        self.Bind(wx.EVT_MENU, self.on_normalize, item_normalize)
        menubar.Append(tools_menu, "&Tools")

        self.SetMenuBar(menubar)

    def _refresh_left_panels(self) -> None:
        """Refresh left-side panels from the current experiment model."""
        try:
            self.runs_panel.refresh_from_experiment(self.experiment)
        except Exception:
            pass
        try:
            self.experiment_panel.refresh_from_experiment(self.experiment)
        except Exception:
            pass
        try:
            self.preferences_panel.update_experiment(self.experiment)
        except Exception:
            pass

        # Refresh appearances panel if a view is active
        try:
            view_id = self._get_current_view_id()
            if view_id:
                view_state = self.experiment.get_view(view_id)
                self.appearances_panel.update_view(view_state, self.experiment)
            else:
                self.appearances_panel.update_view(None, None)
        except Exception:
            pass
        try:
            self._update_fit_results_panel()
        except Exception:
            pass

    def on_merge_run_dialog(self, event):
        """Open the merge-run dialog.

        Allows multiple file selection. If multiple files correspond to different
        experiment seeds, the dialog is shown sequentially for each seed.
        """
        with wx.FileDialog(
            self,
            message="Select 1D run files to merge",
            wildcard="Data files (*.csv;*.txt)|*.csv;*.txt|All files (*.*)|*.*",
            style=wx.FD_OPEN | wx.FD_FILE_MUST_EXIST | wx.FD_MULTIPLE,
        ) as dlg:
            if dlg.ShowModal() != wx.ID_OK:
                return
            paths = dlg.GetPaths()

        if not paths:
            return

        # Group paths by seed patterns to identify unique merge operations
        unique_seeds = {}
        for p in paths:
            abs_p = os.path.abspath(p)
            d = os.path.dirname(abs_p)
            base = os.path.basename(abs_p)
            m = analysis.POLARIZATION_IDX_RE.match(base)
            if m:
                key = (
                    d,
                    "pol",
                    analysis.polarization_family_prefix(m.group("prefix")),
                    m.group("middle"),
                    m.group("ext"),
                )
            else:
                m = analysis.NORMAL_IDX_TWO_RE.match(base)
                if not m:
                    m = analysis.NORMAL_IDX_ONE_RE.match(base)
            if m and not analysis.POLARIZATION_IDX_RE.match(base):
                # Experiment pattern: group by (dir, prefix, ext)
                key = (d, m.group("prefix"), m.group("ext"))
            elif not analysis.POLARIZATION_IDX_RE.match(base):
                # Standalone file: each is its own seed
                key = (d, base, "")

            if key not in unique_seeds:
                unique_seeds[key] = abs_p

        # Process each unique seed
        for seed_path in unique_seeds.values():
            self._run_merge_dialog_for_seed(seed_path)

    def _run_merge_dialog_for_seed(self, seed_path: str):
        """Helper to run MergeRunsDialog for a specific seed path."""
        md = MergeRunsDialog(self, [seed_path], log_cb=self.log_panel.append_log, experiment=self.experiment)
        try:
            if md.ShowModal() == wx.ID_OK and md.result_run:
                new_run = md.result_run
                self.experiment.add_run(new_run)
                nickname = self.experiment.get_run_nickname(new_run.id)
                self.log_panel.append_log(
                    f"Merged run created as {nickname} ({new_run.id}) from seed {os.path.basename(seed_path)}."
                )
                self._refresh_left_panels()
                self._refresh_all_view_panels()
            else:
                self.log_panel.append_log(f"Merge operation cancelled for seed {os.path.basename(seed_path)}.")
        finally:
            md.Destroy()

    def on_calculate_from_run(self, event):
        """
        Open a dialog to select a run, then perform a calculation (currently angle summation).
        """
        # Get list of runs
        run_ids = [rid for rid, run in self.experiment.runs.items() if not is_hidden_run(run)]
        if not run_ids:
            wx.MessageBox("No runs available.", "Error", wx.OK | wx.ICON_ERROR)
            return

        choices = [f"{self.experiment.get_run_nickname(rid)} ({rid})" for rid in run_ids]

        with wx.SingleChoiceDialog(self, "Select a run to calculate from:", "Calculate from Run", choices) as dlg:
            if dlg.ShowModal() != wx.ID_OK:
                return
            selection = dlg.GetSelection()
            run_id = run_ids[selection]

        run = self.experiment.get_run(run_id)
        if not run:
             return

        # For now, we only support Angle Summation on 2D runs.
        if not run.is_2d:
             wx.MessageBox("Selected run is not a 2D run. Only angle summation of 2D runs is currently supported.", "Info", wx.OK | wx.ICON_INFORMATION)
             return

        # Perform summation
        # intensity_2d shape is (angles, shifts)
        # Sum along axis 0 to get (shifts,)
        if run.intensity_2d is None:
             return

        summed_intensity = np.sum(run.intensity_2d, axis=0)

        new_nickname = f"{run.nickname}_summed"

        new_metadata = run.metadata.copy()
        new_metadata["nickname"] = new_nickname
        new_metadata["calculation"] = "angle_sum"
        new_metadata["source_run_id"] = run.id
        new_metadata["raw_dim"] = "1d"

        new_run = Run(
            id=new_run_id(prefix="calc"),
            source_path=run.source_path,
            source_mtime=run.source_mtime,
            wl_nm=run.wl_nm,
            shift_cm1=run.shift_cm1,
            energy_eV=run.energy_eV,
            intensity=summed_intensity,
            intensity_2d=None,
            angle_values=None,
            intensity_unit=run.intensity_unit,
            angle_unit=run.angle_unit,
            metadata=new_metadata,
            run_type=RunType.RUN_1D,
            raw_table=None
        )

        self.experiment.add_run(new_run)
        self.log_panel.append_log(f"Created summed run: {new_nickname}")
        self._refresh_left_panels()

    def on_2d_map_fitting(self, event):
        """
        Open the 2D map fitting tool.
        Supports selecting one or two 2D runs, OR one FIT_PARAMS run.
        """
        run_ids = self._selected_or_current_view_run_ids()

        run1, run2 = None, None
        params_run = None
        single_config = "parallel"

        if len(run_ids) == 1:
            selected_run = self.experiment.get_run(run_ids[0])
            if selected_run is None:
                return
            if selected_run.run_type == RunType.FIT_PARAMS:
                params_run = selected_run
                source_ids = selected_run.metadata.get("source_run_ids") or selected_run.metadata.get("fit_params_source_run_ids") or []

                if len(source_ids) == 2:
                    run1 = self.experiment.get_run(source_ids[0])
                    run2 = self.experiment.get_run(source_ids[1])

                    if not (run1 and run2):
                        wx.MessageBox("One or more source runs for this FIT_PARAMS run could not be found.", "Error", wx.OK | wx.ICON_ERROR)
                        return
                elif len(source_ids) == 1:
                    run1 = self.experiment.get_run(source_ids[0])
                    if run1 is None:
                        wx.MessageBox("The source run for this FIT_PARAMS run could not be found.", "Error", wx.OK | wx.ICON_ERROR)
                        return
                    single_config = polar_area_fitting.infer_config(run1)
                else:
                    wx.MessageBox("Selected run is not a valid FIT_PARAMS run.", "Error", wx.OK | wx.ICON_ERROR)
                    return
            else:
                if not selected_run.is_2d:
                    wx.MessageBox("Selected run must be a 2D run or a Fit Parameters run.", "Selection Error", wx.OK | wx.ICON_ERROR)
                    return
                choices = ["Parallel (XX)", "Cross (YX)"]
                guessed = polar_area_fitting.infer_config(selected_run)
                with wx.SingleChoiceDialog(
                    self,
                    "How should this single selected run be interpreted?",
                    "2D Fit Polarization",
                    choices,
                ) as dlg:
                    dlg.SetSelection(0 if guessed == "parallel" else 1)
                    if dlg.ShowModal() != wx.ID_OK:
                        return
                    single_config = "parallel" if dlg.GetSelection() == 0 else "cross"
                run1 = selected_run
        elif len(run_ids) == 2:
            run1 = self.experiment.get_run(run_ids[0])
            run2 = self.experiment.get_run(run_ids[1])

            if not (run1 and run2 and run1.is_2d and run2.is_2d):
                wx.MessageBox("Both selected runs must be 2D runs.", "Selection Error", wx.OK | wx.ICON_ERROR)
                return
            if not self._validate_2d_fit_run(run1) or not self._validate_2d_fit_run(run2):
                return
            role_result = self._resolve_two_run_fit_roles(run1, run2)
            if role_result is None:
                return
            run1, run2 = role_result
        else:
            wx.MessageBox("Please select one or two 2D runs, one Fit Parameters run, or open a view containing runs.", "Selection Error", wx.OK | wx.ICON_ERROR)
            return

        for candidate in [run for run in (run1, run2) if run is not None]:
            if not self._validate_2d_fit_run(candidate):
                return

        if params_run is None:
            resolved = self._resolve_previous_fit_params([run for run in (run1, run2) if run is not None])
            if resolved is not None:
                params_run = resolved
                source_ids = params_run.metadata.get("source_run_ids") or params_run.metadata.get("fit_params_source_run_ids") or []
                source_runs = [self.experiment.get_run(rid) for rid in source_ids]
                source_runs = [run for run in source_runs if run is not None and run.is_2d]
                if len(source_runs) == 2:
                    run1, run2 = source_runs[0], source_runs[1]
                elif len(source_runs) == 1:
                    run1, run2 = source_runs[0], None
                    single_config = polar_area_fitting.infer_config(run1)

        dlg = MapFittingDialog(self, run1, run2, params_run=params_run, single_config=single_config, roles_preassigned=(run2 is not None))
        modal_result = dlg.ShowModal()
        if modal_result == wx.ID_OK and dlg.commit_action:
            try:
                target, summary = dlg.commit_to_experiment(save_as_new=dlg.commit_action == "save_new")
                self.log_panel.append_log(
                    f"Committed {target.nickname}; removed {summary.get('removed_matrix_caches', 0)} automatic matrix cache(s)."
                )
                self._refresh_left_panels()
                self._refresh_all_view_panels()
            except Exception as exc:
                wx.MessageBox(f"Could not commit 2D fit:\n{exc}", "2D Fit Commit", wx.OK | wx.ICON_ERROR)
        dlg.Destroy()

    def _validate_2d_fit_run(self, run: Run) -> bool:
        try:
            shift, angles, intensity = analysis.display_2d_from_run(run)
            if shift.size == 0 or angles.size == 0 or intensity.size == 0:
                raise ValueError("empty display arrays")
            if intensity.shape != (angles.size, shift.size):
                raise ValueError(f"shape {intensity.shape} does not match angle/shift axes")
            if not np.any(np.isfinite(intensity)):
                raise ValueError("no finite intensity values")
            return True
        except Exception as exc:
            wx.MessageBox(f"{run.nickname} cannot be used for 2D fitting:\n{exc}", "2D Fit Compatibility", wx.OK | wx.ICON_ERROR)
            return False

    def _resolve_two_run_fit_roles(self, run1: Run, run2: Run) -> Optional[Tuple[Run, Run]]:
        c1 = polar_area_fitting.infer_config(run1, "unknown")
        c2 = polar_area_fitting.infer_config(run2, "unknown")
        if c1 == "parallel" and c2 == "cross":
            return run1, run2
        if c1 == "cross" and c2 == "parallel":
            return run2, run1
        choices = [
            f"{run1.nickname}: Parallel (XX), {run2.nickname}: Cross (YX)",
            f"{run2.nickname}: Parallel (XX), {run1.nickname}: Cross (YX)",
        ]
        with wx.SingleChoiceDialog(
            self,
            "Assign polarization roles for the common 2D fit.",
            "2D Fit Polarization",
            choices,
        ) as dlg:
            dlg.SetSelection(0)
            if dlg.ShowModal() != wx.ID_OK:
                return None
            return (run1, run2) if dlg.GetSelection() == 0 else (run2, run1)

    def on_rotate_360_dataset(self, event):
        run_ids = self._selected_or_current_view_run_ids()
        runs = [self.experiment.get_run(rid) for rid in run_ids]
        runs = [run for run in runs if run is not None]
        if not runs:
            wx.MessageBox("Please select one or more 2D runs, or open a view containing 2D runs.", "Rotate 360° Dataset", wx.OK | wx.ICON_INFORMATION)
            return
        bad = [run.nickname for run in runs if not run.is_2d]
        if bad:
            wx.MessageBox("All selected/current-view runs must be 2D angular maps for batch rotation.", "Rotate 360° Dataset", wx.OK | wx.ICON_INFORMATION)
            return

        representative = next(
            (
                run for run in runs
                if run.angle_values is not None
                and np.nanmax(np.asarray(run.angle_values, dtype=float)) - np.nanmin(np.asarray(run.angle_values, dtype=float)) > 360.0 + 1e-9
            ),
            runs[0],
        )

        with AngleRotationDialog(self, representative) as dlg:
            if dlg.ShowModal() != wx.ID_OK:
                return
            try:
                template = dlg.get_settings()
            except Exception as exc:
                wx.MessageBox(f"Could not apply rotation: {exc}", "Rotate 360° Dataset", wx.OK | wx.ICON_ERROR)
                return

        updated = []
        failed = []
        rotation_deltas_by_run: Dict[str, float] = {}
        for run in runs:
            try:
                previous_settings = copy.deepcopy((run.metadata or {}).get("angle_rotation") or {})
                if template.get("enabled"):
                    angles = np.asarray(run.angle_values, dtype=float)
                    span = float(np.nanmax(angles) - np.nanmin(angles))
                    window_start = template.get("window_start_deg") if span > 360.0 + 1e-9 else None
                    settings = analysis.build_angle_rotation_settings(
                        run.angle_values,
                        anchor_deg=float(template.get("anchor_deg", 0.0)),
                        window_start_deg=window_start,
                        period_deg=float(template.get("period_deg", 360.0)),
                        cyclic_rotate=bool(template.get("cyclic_rotate", True)),
                    )
                else:
                    settings = {"enabled": False, "summary": "Rotation disabled"}
                run.metadata["angle_rotation"] = settings
                if settings.get("enabled"):
                    run.metadata["angle_rotation_summary"] = settings.get("summary", analysis.angle_rotation_summary(settings))
                else:
                    run.metadata.pop("angle_rotation_summary", None)
                updated.append(run)
                rotation_deltas_by_run[run.id] = analysis.angle_rotation_transition_delta(previous_settings, settings)
            except Exception as exc:
                failed.append(f"{run.nickname}: {exc}")

        if updated:
            names = ", ".join(run.nickname for run in updated)
            self.log_panel.append_log(f"Updated 360° rotation for {len(updated)} run(s): {names}")
            warnings = self._adjust_phi_for_rotated_runs(rotation_deltas_by_run)
            for warning in warnings:
                self.log_panel.append_log(warning)
            if warnings:
                wx.MessageBox("\n".join(warnings), "Fit Phi Adjustment", wx.OK | wx.ICON_WARNING)
        if failed:
            wx.MessageBox("Some runs could not be rotated:\n" + "\n".join(failed), "Rotate 360° Dataset", wx.OK | wx.ICON_WARNING)
        self._refresh_all_view_panels()
        self._refresh_left_panels()

    def _fold_phi(self, value: float) -> float:
        return float(((value + 180.0) % 360.0) - 180.0)

    def _shift_fit_state_phi(self, state: Dict[str, Any], offset: float) -> int:
        count = 0
        for peak in state.get("peaks", []) or []:
            params = peak.get("ang_params", {})
            if "phi" not in params:
                continue
            try:
                params["phi"][0] = self._fold_phi(float(params["phi"][0]) + float(offset))
                count += 1
            except Exception:
                continue
        return count

    def _adjust_phi_for_rotated_runs(self, deltas_by_run: Dict[str, float]) -> List[str]:
        warnings: List[str] = []
        adjusted_state_ids = set()

        def maybe_adjust(owner: Run, state: Optional[Dict[str, Any]], source_ids: List[str]) -> None:
            if not state:
                return
            source_ids_clean = [str(v) for v in source_ids if str(v)]
            if not source_ids_clean:
                source_ids_clean = [owner.id]
            touched = [rid for rid in source_ids_clean if rid in deltas_by_run]
            if not touched:
                return
            if len(touched) != len(source_ids_clean):
                warnings.append(f"{owner.nickname}: fit phi not adjusted because only part of a common fit was rotated.")
                return
            deltas = [float(deltas_by_run[rid]) for rid in source_ids_clean]
            if max(deltas) - min(deltas) > 1e-6:
                warnings.append(f"{owner.nickname}: fit phi not adjusted because common-fit runs used different rotation changes.")
                return
            if abs(deltas[0]) <= 1e-12:
                return
            state_id = id(state)
            if state_id in adjusted_state_ids:
                return
            adjusted_state_ids.add(state_id)
            count = self._shift_fit_state_phi(state, deltas[0])
            if count:
                try:
                    engine = analysis.MapFittingEngine()
                    engine.from_dict(state)
                    owner.metadata["fit_parameters_text"] = engine.export_parameters_text()
                except Exception:
                    pass
                self.log_panel.append_log(f"{owner.nickname}: adjusted {count} phi parameter(s) by {deltas[0]:.6g} deg.")

        for run in self.experiment.runs.values():
            md = run.metadata or {}
            state = md.get("fit_state") or md.get("map_fit_state")
            source_ids = md.get("source_run_ids") or md.get("fit_params_source_run_ids") or ([run.id] if run.is_2d else [])
            maybe_adjust(run, state, [str(v) for v in source_ids])

        return warnings

    def _resolve_previous_fit_params(self, runs: List[Run]) -> Optional[Run]:
        runs = [run for run in runs if run is not None and run.is_2d]
        if not runs:
            return None
        selected_ids = {run.id for run in runs}
        active_ids = {
            str((run.metadata or {}).get("active_fit_params_run_id"))
            for run in runs
            if (run.metadata or {}).get("active_fit_params_run_id")
        }
        if len(active_ids) == 1:
            active = self.experiment.get_run(next(iter(active_ids)))
            if active is not None and active.run_type == RunType.FIT_PARAMS:
                active_sources = set(str(value) for value in (active.metadata or {}).get("source_run_ids", []))
                if selected_ids.issubset(active_sources):
                    return active
        candidates: List[Run] = []
        persisted_candidates: List[Run] = []

        for run in self.experiment.runs.values():
            if run.run_type != RunType.FIT_PARAMS:
                continue
            state = run.metadata.get("fit_state") or run.metadata.get("map_fit_state")
            source_ids = set(str(v) for v in (run.metadata.get("source_run_ids") or run.metadata.get("fit_params_source_run_ids") or []))
            if state and selected_ids and selected_ids.issubset(source_ids):
                candidates.append(run)
                persisted_candidates.append(run)

        if len(persisted_candidates) == 1:
            return persisted_candidates[0]

        for run in runs:
            state = (run.metadata or {}).get("fit_state") or (run.metadata or {}).get("map_fit_state")
            if not state:
                continue
            source_ids = [str(v) for v in (run.metadata or {}).get("fit_params_source_run_ids", [])] or [r.id for r in runs]
            transient = Run(
                id=new_run_id(prefix="params"),
                source_path="",
                metadata={
                    "nickname": f"StoredParams_{run.nickname}",
                    "fit_state": state,
                    "fit_parameters_text": (run.metadata or {}).get("fit_parameters_text", ""),
                    "source_run_ids": source_ids,
                },
                run_type=RunType.FIT_PARAMS,
            )
            candidates.append(transient)

        unique: List[Run] = []
        seen = set()
        for candidate in candidates:
            state = candidate.metadata.get("fit_state") or candidate.metadata.get("map_fit_state")
            key = (candidate.id if candidate.run_type == RunType.FIT_PARAMS and candidate.id in self.experiment.runs else f"state:{id(state)}", tuple(candidate.metadata.get("source_run_ids", [])))
            if key not in seen:
                seen.add(key)
                unique.append(candidate)
        if not unique:
            return None
        if len(unique) == 1:
            return unique[0]
        with FitParamsChoiceDialog(self, unique) as dlg:
            if dlg.ShowModal() != wx.ID_OK:
                return None
            return dlg.selected_run()

    def _selected_runs_for_polar_area_fit(self) -> Tuple[List[Run], Optional[Dict[str, Any]]]:
        run_ids = self._selected_or_current_view_run_ids()
        if not run_ids:
            raise ValueError("Please select one fitted 2D run, two fitted 2D runs, one Fit Parameters run, or open a view containing fitted 2D runs.")

        selected = [self.experiment.get_run(rid) for rid in run_ids]
        selected = [run for run in selected if run is not None]
        if len(selected) == 1 and selected[0].run_type == RunType.FIT_PARAMS:
            params_run = selected[0]
            runs = polar_area_fitting.runs_from_fit_selection(self.experiment, params_run)
            if not runs:
                raise ValueError("The selected Fit Parameters run does not point to any available 2D source runs.")
            return runs, params_run.metadata.get("fit_state")

        if len(selected) > 2:
            raise ValueError("Please select at most two 2D runs.")
        if not all(run.is_2d for run in selected):
            raise ValueError("Selected runs must be 2D runs, unless selecting one Fit Parameters run.")

        params_run = self._resolve_previous_fit_params(selected)
        if params_run is not None:
            return selected, params_run.metadata.get("fit_state") or params_run.metadata.get("map_fit_state")
        for run in selected:
            fit_state = (run.metadata or {}).get("fit_state") or (run.metadata or {}).get("map_fit_state")
            if fit_state:
                return selected, fit_state
        return selected, None

    def on_generate_polar_area_fittings(self, event):
        """Generate vector PDF polar plots from row-by-row Lorentzian areas."""
        using_current_view = not bool(self.runs_panel._get_selected_run_ids())
        try:
            runs, fit_state = self._selected_runs_for_polar_area_fit()
        except Exception as exc:
            wx.MessageBox(str(exc), "Polar Area Fittings", wx.OK | wx.ICON_ERROR)
            return

        default_targets = []
        if fit_state:
            fit_state_unit = normalize_spectral_unit(fit_state.get("unit", "cm-1"), "cm-1")
            for peak in fit_state.get("peaks", []):
                try:
                    value = float(peak.get("spec_params", {}).get("x0", [])[0])
                    default_targets.append(float(unit_to_cm1(value, fit_state_unit)))
                except Exception:
                    pass
        if not default_targets and runs and runs[0].shift_cm1 is not None:
            default_targets = [float(np.nanmedian(runs[0].shift_cm1))]

        cmap_name = polar_area_fitting.view_cmap_for_runs(self.experiment, runs) or config.get("default_colormap", config.get("colormap", "OrRd"))
        default_colors = polar_area_fitting.colors_from_cmap(cmap_name, max(len(default_targets), 6))
        last_dir = config.get("last_directory", "") or os.getcwd()
        safe_name = "_".join(run.nickname for run in runs)
        safe_name = "".join(c if c.isalnum() or c in "-_" else "_" for c in safe_name)
        default_output = os.path.join(last_dir, f"{safe_name}_polar_area_fits.pdf")
        default_unit = "cm-1"
        if using_current_view:
            view_id = self._get_current_view_id()
            view_state = self.experiment.get_view(view_id) if view_id else None
            if view_state is not None:
                default_unit = normalize_spectral_unit(view_state.spectral_unit, "cm-1")

        with PolarAreaFittingDialog(self, runs, default_targets, default_colors, default_output, default_unit=default_unit) as dlg:
            if dlg.ShowModal() != wx.ID_OK:
                return
            try:
                opts = dlg.get_values()
            except Exception as exc:
                wx.MessageBox(f"Invalid options: {exc}", "Polar Area Fittings", wx.OK | wx.ICON_ERROR)
                return

        if not opts["output"]:
            wx.MessageBox("Choose an output PDF path.", "Polar Area Fittings", wx.OK | wx.ICON_ERROR)
            return

        use_cache = bool(opts.get("use_cache", True))
        progress = wx.ProgressDialog(
            "Polar Area Fittings",
            "Loading saved inspection fits and exporting PDF..." if use_cache else "Running independent row-by-row fits and exporting PDF...",
            parent=self,
            style=wx.PD_APP_MODAL | wx.PD_ELAPSED_TIME,
        )
        progress.Pulse()
        try:
            row_fit_groups = (
                fit_overlay.cached_row_fits_for_runs(self.experiment, runs, fit_state, opts["targets"])
                if use_cache else None
            )
            out_path, row_fits, tensor_fits = polar_area_fitting.generate_polar_area_pdf(
                self.experiment,
                runs,
                opts["targets"],
                opts["output"],
                fit_state=fit_state,
                colors=opts["colors"],
                peak_window=opts["peak_window"],
                center_window=opts["center_window"],
                normalize=opts["normalize"],
                row_fit_groups=row_fit_groups,
                output_unit=opts["unit"],
            )
        except Exception as exc:
            progress.Destroy()
            wx.MessageBox(f"Failed to generate polar area fittings: {exc}", "Polar Area Fittings", wx.OK | wx.ICON_ERROR)
            return
        progress.Destroy()

        for run in runs:
            run.metadata["last_polar_area_fit_pdf"] = out_path
            run.metadata["last_polar_area_fit_targets"] = opts["targets"]
            run.metadata["last_polar_area_fit_unit"] = opts["unit"]

        self.log_panel.append_log(
            f"Generated polar area fitting PDF for {len(runs)} run(s), "
            f"{len(row_fits)} row-fit series, {len(tensor_fits)} tensor fit(s), "
            f"unit={opts['unit']}, source={'saved inspection cache' if use_cache else 'independent raw-data refit'}: {out_path}"
        )
        wx.MessageBox(f"Saved:\n{out_path}", "Polar Area Fittings", wx.OK | wx.ICON_INFORMATION)

    def on_normalize(self, event):
        """
        Open the normalization tool. Requires a 2D run to be selected.
        """
        # Get list of runs
        run_ids = [rid for rid, run in self.experiment.runs.items() if not is_hidden_run(run)]
        if not run_ids:
            wx.MessageBox("No runs available.", "Error", wx.OK | wx.ICON_ERROR)
            return

        choices = [f"{self.experiment.get_run_nickname(rid)} ({rid})" for rid in run_ids]

        with wx.SingleChoiceDialog(self, "Select a run to normalize:", "Normalization", choices) as dlg:
            if dlg.ShowModal() != wx.ID_OK:
                return
            selection = dlg.GetSelection()
            run_id = run_ids[selection]

        run = self.experiment.get_run(run_id)
        if not run:
             return

        if not run.is_2d:
             wx.MessageBox("Selected run is not a 2D run.", "Info", wx.OK | wx.ICON_INFORMATION)
             return

        dlg = NormalizeDialog(self, run)
        if dlg.ShowModal() == wx.ID_OK:
            if dlg.result_run:
                self.experiment.add_run(dlg.result_run)
                self.log_panel.append_log(f"Normalized run created: {dlg.result_run.nickname}")
                self._refresh_left_panels()
        dlg.Destroy()

    # -------- layout --------

    def _build_layout(self):
        # root sizer for the frame
        root_sizer = wx.BoxSizer(wx.VERTICAL)
        self.SetSizer(root_sizer)

        # main horizontal splitter: left controls | right views
        self.main_splitter = wx.SplitterWindow(
            self, style=wx.SP_LIVE_UPDATE | wx.SP_3D
        )

        root_sizer.Add(self.main_splitter, 1, wx.EXPAND)

        # left side: panel containing its own vertical splitter
        left_panel = wx.Panel(self.main_splitter)
        left_sizer = wx.BoxSizer(wx.VERTICAL)
        left_panel.SetSizer(left_sizer)

        self.left_splitter = wx.SplitterWindow(
            left_panel, style=wx.SP_LIVE_UPDATE | wx.SP_3D
        )
        left_sizer.Add(self.left_splitter, 1, wx.EXPAND)

        # --- top notebook (Files / Runs / Experiment / Log) ---
        top_notebook = wx.Notebook(self.left_splitter, style=wx.NB_TOP)

        self.files_panel = FilesPanel(
            top_notebook,
            on_file_activated=self.on_file_activated,
            on_import_run=self.on_import_run_from_files_panel,
        )
        self.runs_panel = RunsPanel(
            top_notebook,
            on_add_to_current_view=self.on_add_runs_to_current_view,
            on_add_to_new_view=self.on_add_runs_to_new_view,
            on_view_remove=self.on_remove_view,
            on_view_duplicate=self.on_duplicate_view,
            on_view_remove_run=self.on_remove_run_from_view,
            on_rename_run=self.on_rename_run,
            on_export_run=self.on_export_run,
            on_update_from_file=self.on_update_run_from_file,
            on_remove_run=self.on_remove_run,
            on_edit_formula=self.on_edit_formula,
        )
        self.experiment_panel = ExperimentPanel(
            top_notebook,
            on_edit_formula=self.on_edit_formula,
            on_edit_derived_props=self.on_edit_derived_props
        )
        self.preferences_panel = PreferencesPanel(
            top_notebook,
            on_use_current_view=self.on_use_current_view_as_defaults,
        )
        self.log_panel = LogPanel(top_notebook)

        top_notebook.AddPage(self.files_panel, "Files")
        top_notebook.AddPage(self.runs_panel, "Experiment")
        top_notebook.AddPage(self.experiment_panel, "Metadata")
        top_notebook.AddPage(self.preferences_panel, "Preferences")
        top_notebook.AddPage(self.log_panel, "Log")

        # --- bottom notebook (Preview / Curve Fit) ---
        bottom_notebook = wx.Notebook(self.left_splitter, style=wx.NB_TOP)

        self.preview_panel = PreviewPanel(bottom_notebook)
        self.fit_results_panel = FitResultsPanel(bottom_notebook)
        self.plot_config_panel = PlotConfigPanel(
            bottom_notebook,
            on_reset=self.on_plot_reset,
            on_view_config_change=self._update_fit_results_panel,
        )
        self.appearances_panel = AppearancesPanel(
            bottom_notebook,
            on_rename_run=self.on_rename_run,
            on_style_change=self.on_style_change
        )

        bottom_notebook.AddPage(self.preview_panel, "Preview")
        bottom_notebook.AddPage(self.plot_config_panel, "Plot Config.")
        bottom_notebook.AddPage(self.appearances_panel, "Appearance")
        bottom_notebook.AddPage(self.fit_results_panel, "Fit Results")

        # Put the two notebooks into the left vertical splitter
        # Initial sash position: ~40% of left panel height.
        self.left_splitter.SplitHorizontally(top_notebook, bottom_notebook)
        self.left_splitter.SetMinimumPaneSize(80)
        self.left_splitter.SetSashGravity(0.4)

        # --- right: notebook of Views ---
        view_notebook_style = (
            aui.AUI_NB_TOP
            | aui.AUI_NB_SCROLL_BUTTONS
            | aui.AUI_NB_WINDOWLIST_BUTTON
            | getattr(aui, "AUI_NB_NO_TAB_FOCUS", 0)
        )
        self.view_notebook = aui.AuiNotebook(
            self.main_splitter,
            agwStyle=view_notebook_style,
        )

        # view tabs will be created after splitter/layout is set up

        # split left and right; adjust sash after layout based on real splitter width
        self.main_splitter.SplitVertically(left_panel, self.view_notebook)
        self.main_splitter.SetMinimumPaneSize(200)
        # SashGravity controls how extra space is distributed when resizing.
        # 1/3 means the left pane tends to stay about one third of the width when resizing.
        self.main_splitter.SetSashGravity(1.0 / 3.0)

        # Defer initial sash positioning until after the first layout pass,
        # so GetClientSize() returns a meaningful width.
        wx.CallAfter(self._set_initial_split_ratio)

        # After layout, create the initial view tab and view state
        wx.CallAfter(self._init_view_tab)

        # Bind AUI notebook page changed.
        self.view_notebook.Bind(aui.EVT_AUINOTEBOOK_PAGE_CHANGED, self.on_view_page_changed)

    def _init_view_tab(self):
        self.add_view_tab()
        # Manually trigger target setting for the initial view
        panel = self.get_current_view_panel()
        if panel:
            self.plot_config_panel.set_target_view(panel)

    def _update_view_notebook_style(self) -> None:
        """Refresh wrapped view tabs."""
        if not hasattr(self, "view_notebook"):
            return
        if hasattr(self.view_notebook, "refresh_tab_layout"):
            self.view_notebook.refresh_tab_layout()
        else:
            self.view_notebook.Layout()
        self.main_splitter.Layout()

    def on_use_current_view_as_defaults(self):
        panel = self.get_current_view_panel()
        view_id = self._get_current_view_id()
        view_state = self.experiment.get_view(view_id) if view_id else None
        if not panel or not view_state:
            return
        panel.save_current_plot_config()
        map_cfg = view_state.graph_configs.get("1A")
        config.set("default_spectral_unit", normalize_spectral_unit(view_state.spectral_unit))
        config.set("default_colormap", panel.get_colormap())
        config.set("default_vmin_percent", float(map_cfg.vmin if map_cfg and map_cfg.vmin is not None else view_state.vmin))
        config.set("default_vmax_percent", float(map_cfg.vmax if map_cfg and map_cfg.vmax is not None else view_state.vmax))
        config.set("default_angle_slice_type", view_state.angle_slice_type)
        config.set("default_slice_x_binning", view_state.slice_x_binning)
        config.set("default_slice_y_binning", view_state.slice_y_binning)
        config.set("default_slice_binning_mode", view_state.slice_binning_mode)
        config.set("show_secondary_unit_axis", view_state.show_secondary_unit_axis)

    def _set_initial_split_ratio(self):
        """
        Set initial left:right split to approximately 1/3 : 2/3,
        using the actual client width of the splitter after layout.
        """
        width = self.main_splitter.GetClientSize().width
        if width <= 0:
            return
        sash_pos = int(width * (1.0 / 3.0))
        self.main_splitter.SetSashPosition(sash_pos)

    # -------- experiment / runs --------

    def load_initial_files(self, file_paths: List[str]) -> None:
        """
        Build Run objects from the given file paths and add them to the ExperimentSet.
        """
        self.import_runs_from_paths(file_paths, set_files_dir=True)

    def import_runs_from_paths(self, file_paths: List[str], *, set_files_dir: bool = True) -> None:
        """Import runs from file paths using the same pipeline as CLI startup."""
        if not file_paths:
            return

        loaded_dirs = []

        for path in file_paths:
            if not path:
                continue
            try:
                run = Run.from_file(
                    path,
                    default_unknown_1d_spectral_unit=config.get("default_unknown_1d_spectral_unit", "meV"),
                    default_unknown_2d_spectral_unit=config.get("default_unknown_2d_spectral_unit", "meV"),
                )
            except Exception as e:
                self.log_panel.append_log(f"Failed to load {path}: {e}")
                continue

            self.experiment.add_run(run)
            nickname = self.experiment.get_run_nickname(run.id)
            self.log_panel.append_log(
                f"Loaded file '{os.path.basename(path)}' as run {nickname}({run.id})"
            )
            loaded_dirs.append(os.path.dirname(os.path.abspath(path)))

        # Update simple experiment-level metadata
        self.experiment.metadata["num_runs"] = len(self.experiment.runs)

        # Refresh tabs
        self._refresh_left_panels()

        # Initialize Files tab directory to the first loaded directory
        if set_files_dir and loaded_dirs:
            target_dir = loaded_dirs[0]
            self.files_panel.set_directory(target_dir)
            config.set("last_directory", target_dir)

    def on_import_run_from_files_panel(self, path: str) -> None:
        """Called from FilesPanel context menu: import a single file as a run."""
        if not path:
            return
        self.import_runs_from_paths([path], set_files_dir=True)

    def on_import_run_dialog(self, event) -> None:
        """File -> Import Run... opens file picker and imports as runs."""
        wildcard = "Data files (*.csv;*.txt;*.tvf)|*.csv;*.txt;*.tvf|All files (*.*)|*.*"
        dlg = wx.FileDialog(
            self,
            message="Import Run",
            wildcard=wildcard,
            style=wx.FD_OPEN | wx.FD_FILE_MUST_EXIST | wx.FD_MULTIPLE,
        )
        try:
            if dlg.ShowModal() != wx.ID_OK:
                return
            paths = dlg.GetPaths()
        finally:
            dlg.Destroy()

        self.import_runs_from_paths(list(paths), set_files_dir=True)

    def on_import_qe_raman(self, event) -> None:
        """Import a QE ph.x output as paired frequency-only XX/YX runs."""
        initial_directory = str(config.get("last_directory", "") or "")
        initial_path = initial_directory if os.path.isfile(initial_directory) else ""
        dlg = QeRamanImportDialog(self, self.experiment, initial_path=initial_path)
        try:
            if dlg.ShowModal() != wx.ID_OK:
                return
            runs = dlg.get_created_runs()
            for run in runs:
                base_id = run.id
                suffix = 1
                while run.id in self.experiment.runs:
                    suffix += 1
                    run.id = f"{base_id}_{suffix}"
                self.experiment.add_run(run)
            self.experiment.metadata["num_runs"] = len(self.experiment.runs)
            self.add_view_tab(
                run_ids=[run.id for run in runs],
                title=dlg.created_view_title,
            )
            source_path = runs[0].source_path
            if source_path:
                source_directory = os.path.dirname(os.path.abspath(source_path))
                config.set("last_directory", source_directory)
                self.files_panel.set_directory(source_directory)
            self._refresh_left_panels()
            self.log_panel.append_log(
                f"Imported QE Raman frequencies as {runs[0].nickname} and {runs[1].nickname}."
            )
            self.log_panel.append_log(str(runs[0].metadata.get("qe_intensity_disclaimer", "")))
        finally:
            dlg.Destroy()


    def on_new_formula_run(self, event):
        """Create a new derived run with a default formula."""
        # Create a default sine wave
        new_run = Run.from_formula(
            formula="np.sin(x/10.0 * freq) * amp",
            params={"amp": 100.0, "freq": 1.0},
            n_points=200,
            x_range=(0, 500),
            nickname="New Formula Run",
            x_unit=normalize_spectral_unit(config.get("default_spectral_unit", "meV")),
        )
        self.experiment.add_run(new_run)
        self.log_panel.append_log(f"Created new formula run: {new_run.id}")
        self._refresh_left_panels()

    def on_export_run(self, run_ids: List[str]):
        if not run_ids:
            return

        # Ask the user for a directory ONCE.
        with wx.DirDialog(
            self,
            f"Choose directory to save {len(run_ids)} exported runs",
            style=wx.DD_DEFAULT_STYLE | wx.DD_DIR_MUST_EXIST,
        ) as dlg:
            if dlg.ShowModal() != wx.ID_OK:
                return

            output_dir = dlg.GetPath()

        exported_count = 0
        failed_count = 0

        for run_id in run_ids:
            run = self.experiment.get_run(run_id)
            if run is None:
                self.log_panel.append_log(f"Error: Run {run_id} not found for export.")
                failed_count += 1
                continue

            try:
                run.export_csv(output_dir=output_dir)
                exported_count += 1
            except Exception as e:
                self.log_panel.append_log(f"Failed to export run {run.nickname}: {e}")
                failed_count += 1

        self.log_panel.append_log(f"Finished export operation to {output_dir}.")
        self.log_panel.append_log(f"Successfully exported {exported_count} runs.")
        if failed_count > 0:
            self.log_panel.append_log(f"Failed to export {failed_count} runs.")

    def on_edit_formula(self, run_id: str, new_formula: str, new_params: Dict[str, float]) -> None:
        """
        Update the formula and parameters for a Derived Run and refresh views.
        """
        run = self.experiment.get_run(run_id)
        if not run or run.run_type != RunType.DERIVED:
            return

        run.metadata["formula"] = new_formula
        run.metadata["formula_params"] = new_params

        nickname = self.experiment.get_run_nickname(run_id)
        self.log_panel.append_log(f"Updated formula for derived run '{nickname}' ({run_id})")

        # Refresh all views that might be displaying this run
        for view_id, view_state in self.experiment.views.items():
            if run_id in view_state.run_ids:
                self._update_view_plot(view_id, preserve_state=True)

        # Also refresh metadata tree to show updated formula
        self.experiment_panel.refresh_from_experiment(self.experiment)

    def on_edit_derived_props(self, run_id: str, n_points: int, autorange: bool, x_range: Tuple[float, float]) -> None:
        """
        Update derived run properties and refresh views.
        """
        run = self.experiment.get_run(run_id)
        if not run or run.run_type != RunType.DERIVED:
            return

        run.metadata["default_n_points"] = n_points
        run.metadata["default_autorange"] = autorange
        run.metadata["default_range"] = x_range

        self.log_panel.append_log(f"Updated properties for derived run {run.nickname}")

        # Refresh views
        for view_id, view_state in self.experiment.views.items():
            if run_id in view_state.run_ids:
                # Update run_config override if it exists, or just clear it to use metadata defaults?
                # The logic in ViewPanel._update_run_plots uses run_config override OR metadata default.
                # If run_config has an override, editing metadata won't change the plot unless we clear the override.
                # For now, let's just refresh. If the user previously set a View-specific override in Appearance panel,
                # that should probably persist. If they want to reset, they'd use Appearance panel.
                # Here we are editing the "global" default for the Run.
                self._update_view_plot(view_id, preserve_state=True)

        self.experiment_panel.refresh_from_experiment(self.experiment)

    # -------- view-tab helpers --------

    def add_view_tab(self, run_ids: Optional[List[str]] = None, title: Optional[str] = None):
        """
        Create a new View tab and corresponding ViewState in the ExperimentSet.

        Parameters
        ----------
        run_ids : list of str, optional
            If provided, these run IDs will be attached to the new view.
        """
        idx = len(self.experiment.views) + 1
        view_id = new_view_id()
        title = str(title or f"View {idx}")

        default_unit = normalize_spectral_unit(config.get("default_spectral_unit", config.get("unit", "meV")))
        default_cmap = config.get("default_colormap", config.get("colormap", "OrRd"))
        default_vmin = float(config.get("default_vmin_percent", 0.0))
        default_vmax = float(config.get("default_vmax_percent", 100.0))
        view_state = ViewState(
            id=view_id,
            title=title,
            run_ids=run_ids or [],
            spectral_unit=default_unit,
            x_axis="energy_eV" if default_unit == "meV" else "shift_cm1",
            angle_slice_type=config.get("default_angle_slice_type", "polar"),
            slice_x_binning=config.get("default_slice_x_binning", 1),
            slice_y_binning=config.get("default_slice_y_binning", 1),
            slice_binning_mode=str(config.get("default_slice_binning_mode", "cross")).lower(),
            show_secondary_unit_axis=bool(config.get("show_secondary_unit_axis", True)),
            vmin=default_vmin,
            vmax=default_vmax,
            cmap=default_cmap,
        )
        for key in ("1A", "2A"):
            graph_cfg = view_state.get_graph_config(key)
            graph_cfg.vmin = default_vmin
            graph_cfg.vmax = default_vmax
            graph_cfg.cmap = default_cmap
        self.experiment.add_view(view_state)

        panel = ViewPanel(
            self.view_notebook,
            view_label=title, plot_config_panel=self.plot_config_panel,
            on_run_created=self.on_run_created,
            on_fit_request=self.on_curve_fit_request
        )
        self.view_notebook.AddPage(panel, title, select=True)
        self._update_view_notebook_style()

        page_index = self.view_notebook.GetPageCount() - 1
        self._view_page_to_id[page_index] = view_id
        self._view_id_to_panel[view_id] = panel

        # Initialize plot for this view
        panel.set_view_model(self.experiment, view_state)
        self.view_notebook.SetSelection(page_index)
        self.plot_config_panel.set_target_view(panel)
        self.appearances_panel.update_view(view_state, self.experiment)

        # Update the Experiment tab's views list and metadata tree
        self.runs_panel.refresh_from_experiment(self.experiment)
        self.experiment_panel.refresh_from_experiment(self.experiment)
        return view_state

    def on_run_created(self, new_run: Run, overlay_target_run: Optional[Run] = None, overlay_plot_type: str = None) -> None:
        """Callback from ViewPanel or CurveFitPanel when a new run is created."""
        self.experiment.add_run(new_run)
        nickname = self.experiment.get_run_nickname(new_run.id)
        self.log_panel.append_log(
            f"Created new run '{nickname}' ({new_run.id})."
        )

        # Handle automatic overlay if requested
        if overlay_target_run and overlay_plot_type:
            view_id = self._get_current_view_id()
            if view_id:
                view_state = self.experiment.get_view(view_id)
                # Check if target run is in this view
                slot_prefix = None
                if len(view_state.run_ids) > 0 and view_state.run_ids[0] == overlay_target_run.id:
                    slot_prefix = "1"
                elif len(view_state.run_ids) > 1 and view_state.run_ids[1] == overlay_target_run.id:
                    slot_prefix = "2"

                if slot_prefix:
                    target_code = f"{slot_prefix}{overlay_plot_type}"

                    # Add new run to view if not present (it shouldn't be yet)
                    if new_run.id not in view_state.run_ids:
                        view_state.run_ids.append(new_run.id)

                    # Set overlay config
                    cfg = view_state.get_run_config(new_run.id)
                    cfg.overlay_target = target_code

                    # Also default the style color to red or something distinct?
                    # For now default black is fine, or random.

                    self.log_panel.append_log(f"Overlaying {nickname} onto {target_code}")

        self._refresh_left_panels()
        # If we modified the view, we need to refresh it
        if overlay_target_run:
            self._refresh_all_view_panels()

    def on_curve_fit_request(self, source_run: Run, plot_type: str, x_data: np.ndarray, y_data: np.ndarray, x_label: Optional[str] = None):
        """
        Handle legacy plot context fit requests.
        2D map fitting now lives in the popup; the lower-left panel is read-only.
        """
        if plot_type == "A" and source_run is not None and source_run.is_2d:
            single_config = polar_area_fitting.infer_config(source_run)
            dlg = MapFittingDialog(self, source_run, None, single_config=single_config)
            modal_result = dlg.ShowModal()
            if modal_result == wx.ID_OK and dlg.commit_action:
                try:
                    target, summary = dlg.commit_to_experiment(save_as_new=dlg.commit_action == "save_new")
                    self.log_panel.append_log(
                        f"Committed {target.nickname}; removed {summary.get('removed_matrix_caches', 0)} automatic matrix cache(s)."
                    )
                    self._refresh_left_panels()
                    self._refresh_all_view_panels()
                except Exception as exc:
                    wx.MessageBox(f"Could not commit 2D fit:\n{exc}", "2D Fit Commit", wx.OK | wx.ICON_ERROR)
            dlg.Destroy()
            self._update_fit_results_panel(source_run)
            return

        self._update_fit_results_panel(source_run)
        parent = self.fit_results_panel.GetParent()
        for i in range(parent.GetPageCount()):
            if parent.GetPage(i) == self.fit_results_panel:
                parent.SetSelection(i)
                break
        wx.MessageBox("Interactive fitting is now done in Tools -> Make 2D Fit. This panel shows saved fit results.", "Fit Results", wx.OK | wx.ICON_INFORMATION)


    # -------- view management helpers --------

    def _get_current_view_id(self) -> Optional[str]:
        """Return the view_id for the currently selected view tab, if any."""
        page_index = self.view_notebook.GetSelection()
        if page_index == wx.NOT_FOUND:
            return None
        return self._view_page_to_id.get(page_index)

    def _update_view_plot(self, view_id: str, preserve_state: bool = False) -> None:
        """
        Refresh the plotting for the specified view, if its panel exists.
        """
        view_state = self.experiment.get_view(view_id)
        if view_state is None:
            return
        panel = self._view_id_to_panel.get(view_id)
        if panel is None:
            return
        panel.set_view_model(self.experiment, view_state, preserve_state=preserve_state)
        # Also update appearances tab if relevant
        if view_id == self._get_current_view_id():
            self.appearances_panel.update_view(view_state, self.experiment)

    def get_current_view_panel(self) -> Optional[ViewPanel]:
        view_id = self._get_current_view_id()
        if view_id:
            return self._view_id_to_panel.get(view_id)
        return None

    def _selected_or_current_view_run_ids(self) -> List[str]:
        run_ids = self.runs_panel._get_selected_run_ids()
        if run_ids:
            return run_ids

        panel = self.get_current_view_panel()
        if panel is not None:
            drawn = [run.id for run in getattr(panel, "_last_drawn_runs", []) if run is not None]
            if drawn:
                return drawn

        view_id = self._get_current_view_id()
        view_state = self.experiment.get_view(view_id) if view_id else None
        return list(view_state.run_ids) if view_state else []

    def _current_fit_results_run(self) -> Optional[Run]:
        selected = self._selected_or_current_view_run_ids()
        if len(selected) == 1:
            run = self.experiment.get_run(selected[0])
            if run is not None:
                return run
        panel = self.get_current_view_panel()
        if panel is not None and panel.current_run is not None:
            return panel.current_run
        return None

    def _update_fit_results_panel(self, run: Optional[Run] = None) -> None:
        if not hasattr(self, "fit_results_panel"):
            return
        view_id = self._get_current_view_id()
        view_state = self.experiment.get_view(view_id) if view_id else None
        self.fit_results_panel.update_for_run(self.experiment, run or self._current_fit_results_run(), view_state)

    def on_plot_reset(self):
        view_id = self._get_current_view_id()
        if view_id:
            self._update_view_plot(view_id, preserve_state=False)
        self._update_fit_results_panel()

    def on_view_page_changed(self, event=None):
        view_panel = self.get_current_view_panel()
        self.plot_config_panel.set_target_view(view_panel)

        # Sync Appearances Panel
        view_id = self._get_current_view_id()
        if view_id:
            view_state = self.experiment.get_view(view_id)
            self.appearances_panel.update_view(view_state, self.experiment)
        else:
            self.appearances_panel.update_view(None, None)
        self._update_fit_results_panel()

        if event is not None:
            event.Skip()

    def on_add_runs_to_current_view(self, run_ids: List[str]) -> None:
        """
        Attach the given runs to the currently selected view.
        If no view exists yet, a new view is created.
        """
        if not run_ids:
            return

        view_id = self._get_current_view_id()
        if view_id is None:
            # No view yet: create a new one with these runs
            self.add_view_tab(run_ids=run_ids)
            labels = [
                f"{self.experiment.get_run_nickname(rid)}({rid})"
                for rid in run_ids
            ]
            self.log_panel.append_log(
                f"Created new view with runs: {', '.join(labels)}"
            )
            return

        view_state = self.experiment.get_view(view_id)
        if view_state is None:
            return

        # Extend unique run IDs
        for rid in run_ids:
            if rid not in view_state.run_ids:
                view_state.run_ids.append(rid)

        labels = [
            f"{self.experiment.get_run_nickname(rid)}({rid})"
            for rid in run_ids
        ]
        self.log_panel.append_log(
            f"Added runs {', '.join(labels)} to view {view_state.title}"
        )

        # Refresh the views listing and update plotting
        self.runs_panel.refresh_from_experiment(self.experiment)
        self._update_view_plot(view_id)
        self.experiment_panel.refresh_from_experiment(self.experiment)

    def on_add_runs_to_new_view(self, run_ids: List[str]) -> None:
        """
        Create a new view and attach the given runs to it.
        """
        if not run_ids:
            return

        self.add_view_tab(run_ids=run_ids)
        labels = [
            f"{self.experiment.get_run_nickname(rid)}({rid})"
            for rid in run_ids
        ]
        self.log_panel.append_log(
            f"Created new view with runs: {', '.join(labels)}"
        )

    def on_remove_view(self, view_id: str) -> None:
        """
        Remove a view from the ExperimentSet and its corresponding notebook tab.
        """
        view_state = self.experiment.get_view(view_id)
        if view_state is None:
            return

        title = view_state.title
        # Remove panel mapping if present
        if view_id in self._view_id_to_panel:
            del self._view_id_to_panel[view_id]

        # Find the notebook page index corresponding to this view
        index_to_remove = None
        for idx, vid in self._view_page_to_id.items():
            if vid == view_id:
                index_to_remove = idx
                break

        if index_to_remove is not None and 0 <= index_to_remove < self.view_notebook.GetPageCount():
            self.view_notebook.DeletePage(index_to_remove)
            self._update_view_notebook_style()

            # Rebuild the page-index -> view_id mapping after deletion
            old_map = self._view_page_to_id
            new_map = {}
            for old_idx, vid in old_map.items():
                if vid == view_id:
                    continue
                if old_idx > index_to_remove:
                    new_idx = old_idx - 1
                else:
                    new_idx = old_idx
                new_map[new_idx] = vid
            self._view_page_to_id = new_map

        # Remove from the experiment
        if view_id in self.experiment.views:
            del self.experiment.views[view_id]

        self.runs_panel.refresh_from_experiment(self.experiment)
        self.experiment_panel.refresh_from_experiment(self.experiment)
        self.log_panel.append_log(f"Removed view {title}")

    def on_duplicate_view(self, view_id: str) -> None:
        """
        Duplicate an existing view (including its run list) into a new view/tab.
        """
        view_state = self.experiment.get_view(view_id)
        if view_state is None:
            return

        run_ids = list(view_state.run_ids)
        self.add_view_tab(run_ids=run_ids)

        if run_ids:
            labels = [
                f"{self.experiment.get_run_nickname(rid)}({rid})"
                for rid in run_ids
            ]
            runs_str = ", ".join(labels)
        else:
            runs_str = "(no runs)"

        self.log_panel.append_log(
            f"Duplicated view {view_state.title} with runs: {runs_str}"
        )
        self.runs_panel.refresh_from_experiment(self.experiment)
        self.experiment_panel.refresh_from_experiment(self.experiment)

    def on_remove_run_from_view(self, view_id: str, run_id: str) -> None:
        """
        Remove a single run from the specified view.
        """
        view_state = self.experiment.get_view(view_id)
        if view_state is None:
            return

        if run_id in view_state.run_ids:
            view_state.run_ids = [rid for rid in view_state.run_ids if rid != run_id]
            label = f"{self.experiment.get_run_nickname(run_id)}({run_id})"
            self.log_panel.append_log(
                f"Removed run {label} from view {view_state.title}"
            )
            self.runs_panel.refresh_from_experiment(self.experiment)
            self._update_view_plot(view_id)
            self.experiment_panel.refresh_from_experiment(self.experiment)

    def on_rename_run(self, run_id: str, new_name: str) -> None:
        """
        Apply a new nickname for the given run and update the panels.

        The nickname is stored in run.metadata['nickname'] and managed by
        the ExperimentSet. This method is called from RunsPanel after an
        inline rename operation.
        """
        new_name = str(new_name).strip()
        if not new_name:
            return

        run = self.experiment.get_run(run_id)
        if run is None:
            return

        # Update the model and refresh views that display the nickname.
        self.experiment.set_run_nickname(run_id, new_name)
        self.runs_panel.refresh_from_experiment(self.experiment)
        self.experiment_panel.refresh_from_experiment(self.experiment)
        self._refresh_all_view_panels()
        self.log_panel.append_log(f"Renamed run {run_id} to {new_name}")

    def on_style_change(self, run_id: str, attr: str, value: Any, component: Optional[str] = None) -> None:
        """
        Update run style based on edits in AppearancesPanel.
        attr: 'visible', 'color', 'linestyle', 'linewidth', 'style_string'
        value: bool or str
        component: "A", "B", "C" or None (apply to all)
        """
        view_id = self._get_current_view_id()
        if not view_id:
            return

        view_state = self.experiment.get_view(view_id)
        if not view_state:
            return

        run_config = view_state.get_run_config(run_id)

        # Determine targets: specific component or all
        targets = [component] if component else ["A", "B", "C"]

        for comp in targets:
            style = run_config.get_style(comp)
            if attr == "style_string":
                parts = [p.strip() for p in str(value).split(",")]
                if len(parts) > 0 and parts[0]: style.color = parts[0]
                if len(parts) > 1 and parts[1]: style.linestyle = parts[1]
                if len(parts) > 2 and parts[2]:
                    try:
                        style.linewidth = float(parts[2])
                    except ValueError: pass
                if len(parts) > 3: style.marker = parts[3]
                if len(parts) > 4 and parts[4]:
                    try:
                        style.markersize = float(parts[4])
                    except ValueError: pass
            elif attr == "visible":
                style.visible = bool(value)
            elif attr == "color":
                style.color = str(value)
            elif attr == "linestyle":
                style.linestyle = str(value)
            elif attr == "linewidth":
                try:
                    style.linewidth = float(value)
                except ValueError:
                    pass
            elif attr == "overlay_target":
                run_config.overlay_target = str(value) if str(value) != "None" else None
            elif attr == "derived_n_points":
                run_config.derived_n_points = int(value)
            elif attr == "derived_autorange":
                run_config.derived_autorange = bool(value)
            elif attr == "derived_range":
                if isinstance(value, (tuple, list)) and len(value) == 2:
                    run_config.derived_range = tuple(value)

        self.log_panel.append_log(f"Updated style '{attr}' for run {run_id} (comp={component or 'all'})")
        # If overlay_target changed, we might need a full redraw because the primary run list might change
        if attr in ["overlay_target", "derived_n_points", "derived_autorange", "derived_range"]:
            self._update_view_plot(view_id, preserve_state=True)
        else:
            self._update_view_plot(view_id, preserve_state=True)

    def on_update_run_from_file(self, run_ids: List[str]) -> None:
        """
        Reload data for the specified runs from their source files.
        """
        updated_any = False
        specialized = []
        for rid in run_ids:
            run = self.experiment.get_run(rid)
            if run:
                if (run.metadata or {}).get("source_kind") == "qe_ph_output":
                    specialized.append(run.nickname)
                    self.log_panel.append_log(
                        f"Skipped generic reload for {run.nickname}; regenerate it with File -> Import Quantum ESPRESSO Raman."
                    )
                    continue
                try:
                    run.reload_data()
                    self.log_panel.append_log(f"Reloaded data for {run.nickname} ({run.source_path})")
                    updated_any = True
                except Exception as e:
                    self.log_panel.append_log(f"Failed to reload {run.nickname}: {e}")

        if specialized:
            wx.MessageBox(
                "QE-generated Raman runs cannot use generic Update from file.\n\n"
                "Use File -> Import Quantum ESPRESSO Raman... to regenerate them.",
                "Quantum ESPRESSO Raman",
                wx.OK | wx.ICON_INFORMATION,
                self,
            )

        if updated_any:
            self._refresh_left_panels()
            self._refresh_all_view_panels()

    def on_remove_run(self, run_ids: List[str]) -> None:
        """
        Remove the specified runs from the experiment.
        """
        if not run_ids:
            return

        for run_id in run_ids:
            run = self.experiment.get_run(run_id)
            if run is None:
                continue

            nickname = run.nickname
            self.experiment.remove_run(run_id)
            self.log_panel.append_log(f"Removed run {nickname} ({run_id})")

        self.runs_panel.refresh_from_experiment(self.experiment)
        self.experiment_panel.refresh_from_experiment(self.experiment)
        self._refresh_all_view_panels()

    def on_save_experiment(self, event=None) -> None:
        """
        Save the current ExperimentSet to an HDF5 file.
        """
        self._persist_view_panel_states()
        wildcard = "HDF5 files (*.h5;*.hdf5)|*.h5;*.hdf5|All files (*.*)|*.*"
        with wx.FileDialog(
            self, message="Save Experiment",
            wildcard=wildcard,
            style=wx.FD_SAVE | wx.FD_OVERWRITE_PROMPT
        ) as dlg:
            if dlg.ShowModal() == wx.ID_OK:
                path = dlg.GetPath()
                try:
                    self.experiment.export_hdf5(path)
                    self.log_panel.append_log(f"Experiment saved to {path}")
                except Exception as e:
                    wx.MessageBox(f"Failed to save experiment: {e}", "Error", wx.OK | wx.ICON_ERROR)

    def on_export_igor(self, event=None) -> None:
        """
        Export the current ExperimentSet to an Igor Pro file.
        """
        wildcard = "Igor Text image package (*.itx)|*.itx|All files (*.*)|*.*"
        with wx.FileDialog(
            self, message="Export to Igor Pro",
            wildcard=wildcard,
            style=wx.FD_SAVE | wx.FD_OVERWRITE_PROMPT
        ) as dlg:
            if dlg.ShowModal() == wx.ID_OK:
                path = dlg.GetPath()
                try:
                    written_path = self.experiment.export_igor(path)
                    self.log_panel.append_log(f"Igor image package exported to {written_path}")
                except Exception as e:
                    wx.MessageBox(f"Failed to export to Igor Pro: {e}", "Error", wx.OK | wx.ICON_ERROR)

    def on_open_experiment(self, event=None) -> None:
        """
        Load an ExperimentSet from an HDF5 file.
        """
        wildcard = "HDF5 files (*.h5;*.hdf5)|*.h5;*.hdf5|All files (*.*)|*.*"
        with wx.FileDialog(
            self, message="Open Experiment",
            wildcard=wildcard,
            style=wx.FD_OPEN | wx.FD_FILE_MUST_EXIST
        ) as dlg:
            if dlg.ShowModal() == wx.ID_OK:
                path = dlg.GetPath()
                try:
                    new_exp = ExperimentSet.from_hdf5(path)
                    self.experiment = new_exp
                    self._refresh_left_panels()
                    self._recreate_view_tabs_from_experiment()
                    self.log_panel.append_log(f"Experiment loaded from {path}")
                except Exception as e:
                    wx.MessageBox(f"Failed to load experiment: {e}", "Error", wx.OK | wx.ICON_ERROR)

    def _recreate_view_tabs_from_experiment(self):
        # Clear existing tabs
        while self.view_notebook.GetPageCount() > 0:
            self.view_notebook.DeletePage(0)
        self._update_view_notebook_style()
        self._view_page_to_id.clear()
        self._view_id_to_panel.clear()

        # Add tabs back
        for vid, vstate in self.experiment.views.items():
            panel = ViewPanel(
                self.view_notebook,
                view_label=vstate.title,
                plot_config_panel=self.plot_config_panel,
                on_run_created=self.on_run_created,
                on_fit_request=self.on_curve_fit_request
            )
            self.view_notebook.AddPage(panel, vstate.title)
            self._update_view_notebook_style()
            page_index = self.view_notebook.GetPageCount() - 1
            self._view_page_to_id[page_index] = vid
            self._view_id_to_panel[vid] = panel
            panel.set_view_model(self.experiment, vstate)

        # Ensure panels are connected to the current (first) view
        if self.view_notebook.GetPageCount() > 0:
            # Manually trigger updates as if the page changed
            self.view_notebook.SetSelection(0)
            self.on_view_page_changed()

    def _refresh_all_view_panels(self) -> None:
        for view_id in list(self.experiment.views.keys()):
            self._update_view_plot(view_id)

    def _persist_view_panel_states(self) -> None:
        for panel in list(self._view_id_to_panel.values()):
            if hasattr(panel, "save_current_plot_config"):
                panel.save_current_plot_config()


    # -------- callbacks from panels --------

    def on_file_activated(self, path: str) -> None:
        """
        Called by FilesPanel when a file is double-clicked.

        For now:
        - Reads a small chunk of the file and shows it in the Preview tab.
        - Logs the action.
        """
        try:
            with open(path, "r", encoding="utf-8", errors="replace") as f:
                text = f.read(8000)  # ~8 kB is enough for a quick preview
        except OSError as e:
            msg = f"Failed to open file for preview: {path} ({e})"
            self.log_panel.append_log(msg)
            self.preview_panel.show_text(msg)
            return

        self.preview_panel.show_text(text)
        self.log_panel.append_log(f"Previewed file: {path}")

    # -------- event handlers --------

    def on_new_view(self, event):
        self.add_view_tab()

    def on_close_current_view(self, event):
        """Close the currently selected view tab, with confirmation if it has runs."""
        view_id = self._get_current_view_id()
        if not view_id:
            return

        view_state = self.experiment.get_view(view_id)
        if not view_state:
            return

        # If the view has attached runs, ask for confirmation
        if view_state.run_ids:
            msg = (
                f"View '{view_state.title}' has {len(view_state.run_ids)} run(s) attached.\n"
                "Are you sure you want to close it?"
            )
            dlg = wx.MessageDialog(self, msg, "Close View", wx.YES_NO | wx.NO_DEFAULT | wx.ICON_WARNING)
            if dlg.ShowModal() != wx.ID_YES:
                dlg.Destroy()
                return
            dlg.Destroy()

        self.on_remove_view(view_id)

    def on_quit(self, event):
        self.Close()





# -----------------------------
# App entry
# -----------------------------


class RamanApp(wx.App):
    def __init__(self, filenames: Optional[List[str]] = None):
        self._filenames = filenames or []
        super().__init__(False)

    def OnInit(self):
        self.frame = MainFrame(initial_files=self._filenames)
        self.SetTopWindow(self.frame)
        return True


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] in {"export-plot", "--export-plot"}:
        from plot_export_cli import main as export_plot_main
        raise SystemExit(export_plot_main(sys.argv[2:]))

    if len(sys.argv) > 1 and sys.argv[1] == "agent":
        from agent_cli import main as agent_main
        raise SystemExit(agent_main(sys.argv[2:]))

    filenames = sys.argv[1:]
    app = RamanApp(filenames)
    app.MainLoop()
