
import wx
import wx.grid
import wx.lib.dialogs
import copy
import os
import csv
import threading
import traceback
import numpy as np
import matplotlib
from matplotlib.backends.backend_wxagg import FigureCanvasWxAgg as FigureCanvas
from matplotlib.backends.backend_wxagg import NavigationToolbar2WxAgg as NavigationToolbar
from matplotlib.figure import Figure
import analysis
import polar_area_fitting
from config_manager import config
from data_structure import Run, RunType, new_run_id
from fit_overlay import commit_fit_params_transaction
from typing import List, Optional, Dict, Any, Tuple

class ParamsGrid(wx.grid.Grid):
    def __init__(self, parent):
        super().__init__(parent)
        self.CreateGrid(0, 4)
        for i, l in enumerate(["Param", "Value", "Min", "Max"]):
            self.SetColLabelValue(i, l)
        self.SetRowLabelSize(0)

    def load_data(self, labels, data_lists):
        if self.GetNumberRows() > 0:
            self.DeleteRows(0, self.GetNumberRows())
        for label, dlist in zip(labels, data_lists):
            row = self.GetNumberRows()
            self.InsertRows(row, 1)
            self.SetCellValue(row, 0, label)
            self.SetReadOnly(row, 0, True)
            for i in range(3):
                self.SetCellValue(row, i+1, f"{dlist[i]:.6g}")
                if label == "Profile_X_Shift":
                    self.SetReadOnly(row, i+1, True)

    def save_to_lists(self, data_lists):
        for row in range(self.GetNumberRows()):
            for i in range(3):
                try:
                    data_lists[row][i] = float(self.GetCellValue(row, i+1))
                except:
                    pass

class RowFitSettingsDialog(wx.Dialog):
    def __init__(self, parent, engine, row_fit_config=None):
        super().__init__(parent, title="Row-by-Row Fit Setup", size=(760, 560), style=wx.DEFAULT_DIALOG_STYLE | wx.RESIZE_BORDER)
        self.engine = engine
        self.config = engine.normalized_row_fit_config(row_fit_config)
        self.current_dataset = 0
        panel, sizer = wx.Panel(self), wx.BoxSizer(wx.VERTICAL)
        note = wx.StaticText(panel, label="Global-fit parameters are inherited as a read-only snapshot. Use 'global' for each row's inherited starting value, enter a number to override it, or check Fixed to keep that value constant.")
        note.Wrap(720); sizer.Add(note, 0, wx.ALL | wx.EXPAND, 10)
        bulk = wx.StaticBoxSizer(wx.HORIZONTAL, panel, "Bulk fixed controls (all datasets)")
        self.bulk_fixed_checks = {}
        for group, label in (
            ("all", "Fix all parameters"),
            ("background", "Fix all backgrounds"),
            ("area", "Fix all areas"),
            ("gamma", "Fix all Gammas"),
        ):
            checkbox = wx.CheckBox(panel, label=label)
            checkbox.Bind(wx.EVT_CHECKBOX, lambda event, selected=group: self.on_bulk_fixed(event, selected))
            self.bulk_fixed_checks[group] = checkbox
            bulk.Add(checkbox, 0, wx.ALL | wx.ALIGN_CENTER_VERTICAL, 5)
        sizer.Add(bulk, 0, wx.LEFT | wx.RIGHT | wx.BOTTOM | wx.EXPAND, 8)
        line = wx.BoxSizer(wx.HORIZONTAL); line.Add(wx.StaticText(panel, label="Dataset:"), 0, wx.ALL | wx.ALIGN_CENTER_VERTICAL, 5)
        self.dataset_choice = wx.Choice(panel, choices=[engine.datasets[i]["label"] for i in engine.active_dataset_indices()]); self.dataset_choice.SetSelection(0); self.dataset_choice.Bind(wx.EVT_CHOICE, self.on_dataset_change)
        line.Add(self.dataset_choice, 1, wx.ALL | wx.EXPAND, 5)
        reset = wx.Button(panel, label="Reset This Dataset"); reset.Bind(wx.EVT_BUTTON, self.on_reset); line.Add(reset, 0, wx.ALL, 5); sizer.Add(line, 0, wx.EXPAND)
        self.grid = wx.grid.Grid(panel); self.grid.CreateGrid(0, 5); self.grid.SetRowLabelSize(0)
        for col, label in enumerate(["Parameter", "Initial", "Min", "Max", "Fixed"]): self.grid.SetColLabelValue(col, label)
        self.grid.SetColSize(0, 260)
        for col in (1, 2, 3): self.grid.SetColSize(col, 125)
        self.grid.SetColSize(4, 65); sizer.Add(self.grid, 1, wx.ALL | wx.EXPAND, 8)
        buttons = wx.StdDialogButtonSizer()
        ok_button = wx.Button(panel, wx.ID_OK); cancel_button = wx.Button(panel, wx.ID_CANCEL)
        buttons.AddButton(ok_button); buttons.AddButton(cancel_button); buttons.Realize(); ok_button.Bind(wx.EVT_BUTTON, self.on_ok)
        sizer.Add(buttons, 0, wx.ALL | wx.ALIGN_RIGHT, 8)
        panel.SetSizer(sizer); self._load_dataset(0)

    @staticmethod
    def _format(value):
        if isinstance(value, str):
            text = value.strip()
            normalized = text.lower()
            if normalized in {"inf", "+inf", "infinity", "+infinity"}:
                return "inf"
            if normalized in {"-inf", "-infinity"}:
                return "-inf"
            try:
                value = float(text)
            except ValueError:
                return text
        else:
            try:
                value = float(value)
            except (TypeError, ValueError):
                return str(value)
        if np.isposinf(value):
            return "inf"
        if np.isneginf(value):
            return "-inf"
        return f"{value:.8g}"

    def _dataset_index(self, position=None):
        position = self.current_dataset if position is None else position
        active = self.engine.active_dataset_indices()
        return active[position] if 0 <= position < len(active) else None

    def _save_dataset(self):
        ds_idx = self._dataset_index()
        if ds_idx is None: return
        rules = self.config["datasets"][str(ds_idx)]
        for row, definition in enumerate(self.engine.row_fit_parameter_definitions(ds_idx)):
            rules[definition["key"]] = {"initial": self.grid.GetCellValue(row, 1).strip() or "global", "min": self.grid.GetCellValue(row, 2).strip() or "-inf", "max": self.grid.GetCellValue(row, 3).strip() or "inf", "fixed": self.grid.GetCellValue(row, 4).strip().lower() in {"1", "true", "yes"}}

    def _load_dataset(self, position):
        self.current_dataset = position; ds_idx = self._dataset_index(position)
        if self.grid.GetNumberRows(): self.grid.DeleteRows(0, self.grid.GetNumberRows())
        if ds_idx is None: return
        definitions, rules = self.engine.row_fit_parameter_definitions(ds_idx), self.config["datasets"][str(ds_idx)]
        self.grid.AppendRows(len(definitions))
        for row, definition in enumerate(definitions):
            rule = rules[definition["key"]]; self.grid.SetCellValue(row, 0, definition["label"]); self.grid.SetReadOnly(row, 0, True)
            self.grid.SetCellValue(row, 1, "global" if rule["initial"] == "global" else self._format(rule["initial"])); self.grid.SetCellValue(row, 2, self._format(rule["min"])); self.grid.SetCellValue(row, 3, self._format(rule["max"]))
            self.grid.SetCellRenderer(row, 4, wx.grid.GridCellBoolRenderer()); self.grid.SetCellEditor(row, 4, wx.grid.GridCellBoolEditor()); self.grid.SetCellValue(row, 4, "1" if rule["fixed"] else "")
        self._sync_bulk_fixed_checks()

    @staticmethod
    def _fixed_group_for_key(key):
        if key in {"BG_Const", "BG_Slope_X", "Amp_Si"}:
            return "background"
        if str(key).endswith("_Area"):
            return "area"
        if str(key).endswith("_Gamma"):
            return "gamma"
        return None

    def _sync_bulk_fixed_checks(self):
        group_states = {}
        for group in ("background", "area", "gamma"):
            matching = []
            for rules in self.config.get("datasets", {}).values():
                matching.extend(
                    bool(rule.get("fixed", False))
                    for key, rule in rules.items()
                    if self._fixed_group_for_key(key) == group
                )
            group_states[group] = bool(matching) and all(matching)
            self.bulk_fixed_checks[group].SetValue(group_states[group])
        self.bulk_fixed_checks["all"].SetValue(all(group_states.values()))

    def on_bulk_fixed(self, event, group):
        self._save_dataset()
        fixed = bool(self.bulk_fixed_checks[group].GetValue())
        selected_groups = {"background", "area", "gamma"} if group == "all" else {group}
        for rules in self.config.get("datasets", {}).values():
            for key, rule in rules.items():
                if self._fixed_group_for_key(key) in selected_groups:
                    rule["fixed"] = fixed
        self._load_dataset(self.current_dataset)

    def on_dataset_change(self, event):
        self._save_dataset(); self._load_dataset(self.dataset_choice.GetSelection())

    def on_reset(self, event):
        ds_idx = self._dataset_index(); defaults = self.engine.normalized_row_fit_config({}); self.config["datasets"][str(ds_idx)] = defaults["datasets"][str(ds_idx)]; self._load_dataset(self.current_dataset)

    def on_ok(self, event):
        try: self._save_dataset(); self.config = self.engine.normalized_row_fit_config(self.config)
        except Exception as exc: wx.MessageBox(str(exc), "Invalid Row-Fit Settings", wx.OK | wx.ICON_ERROR); return
        self.EndModal(wx.ID_OK)

    def get_config(self): return copy.deepcopy(self.config)


class PolarAreaFitDialog(wx.Dialog):
    def __init__(self, parent, engine, results, initial_peak_index=0, on_point_selected=None):
        super().__init__(
            parent,
            title="Row-Fit Polar Peak Areas",
            size=(850, 760),
            style=wx.DEFAULT_DIALOG_STYLE | wx.RESIZE_BORDER | wx.MAXIMIZE_BOX,
        )
        self.engine = engine
        self.results = results
        self.on_point_selected = on_point_selected
        self.selected_point = None
        self._pick_targets = {}
        self.peak_index = max(0, min(int(initial_peak_index), max(0, len(engine.peaks) - 1)))
        panel = wx.Panel(self)
        sizer = wx.BoxSizer(wx.VERTICAL)
        peak_row = wx.BoxSizer(wx.HORIZONTAL)
        peak_row.Add(wx.StaticText(panel, label="Peak:"), 0, wx.ALL | wx.ALIGN_CENTER_VERTICAL, 6)
        choices = []
        for peak in engine.peaks:
            center = float(peak.get("spec_params", {}).get("x0", [np.nan])[0])
            choices.append(f"{peak.get('name', 'Peak')} @ {center:.5g} {engine.unit}")
        self.choice_peak = wx.Choice(panel, choices=choices)
        if choices:
            self.choice_peak.SetSelection(self.peak_index)
        self.choice_peak.Bind(wx.EVT_CHOICE, self.on_peak_choice)
        peak_row.Add(self.choice_peak, 1, wx.ALL | wx.EXPAND, 6)
        sizer.Add(peak_row, 0, wx.EXPAND)
        self.figure = Figure()
        self.canvas = FigureCanvas(panel, -1, self.figure)
        self.canvas.mpl_connect("pick_event", self.on_pick)
        self.toolbar = NavigationToolbar(self.canvas)
        sizer.Add(self.toolbar, 0, wx.EXPAND)
        sizer.Add(self.canvas, 1, wx.EXPAND)
        self.summary = wx.TextCtrl(panel, style=wx.TE_MULTILINE | wx.TE_READONLY, size=(-1, 105))
        sizer.Add(self.summary, 0, wx.ALL | wx.EXPAND, 6)
        close_button = wx.Button(panel, wx.ID_CLOSE, label="Close")
        close_button.Bind(wx.EVT_BUTTON, lambda event: self.EndModal(wx.ID_CLOSE))
        sizer.Add(close_button, 0, wx.ALL | wx.ALIGN_RIGHT, 6)
        panel.SetSizer(sizer)
        self.plot_peak()

    def on_peak_choice(self, event):
        selection = self.choice_peak.GetSelection()
        if selection != wx.NOT_FOUND:
            self.peak_index = selection
            self.selected_point = None
            self.plot_peak()

    def on_pick(self, event):
        target = self._pick_targets.get(event.artist)
        if target is None or not len(event.ind):
            return
        point_index = int(event.ind[0])
        row_indices = target["row_indices"]
        if point_index >= len(row_indices):
            return
        row_index = int(row_indices[point_index])
        result_position = int(target["result_position"])
        self.selected_point = (result_position, row_index, int(self.peak_index))
        if callable(self.on_point_selected):
            self.on_point_selected(result_position, row_index, int(self.peak_index))
        self.plot_peak()

    def _row_fits(self):
        peak = self.engine.peaks[self.peak_index]
        center = float(peak.get("spec_params", {}).get("x0", [np.nan])[0])
        target_cm1 = self.engine.convert_between_units(center, self.engine.unit, "cm-1")
        row_fits = []
        result_positions = []
        prefix = f"P{self.peak_index + 1}_"
        for result_position, result in enumerate(self.results):
            headers = [str(value) for value in (result.get("headers") or [])]
            rows = result.get("rows_params") or result.get("params") or []
            angle_column = headers.index("Angle") if "Angle" in headers else 0
            area_column = next((index for index, header in enumerate(headers) if header.startswith(prefix) and header.endswith("_Area")), None)
            gamma_column = next((index for index, header in enumerate(headers) if header.startswith(prefix) and header.endswith("_Gamma")), None)
            height_column = next((index for index, header in enumerate(headers) if header.startswith(prefix) and header.endswith("_Height")), None)
            if area_column is None:
                continue

            def column(index, default=np.nan):
                values = []
                for row in rows:
                    try:
                        values.append(float(row[index]) if index is not None else default)
                    except Exception:
                        values.append(default)
                return np.asarray(values, dtype=float)

            angles = column(angle_column)
            row_fits.append(polar_area_fitting.RowPeakFit(
                target=float(target_cm1),
                name=str(peak.get("name") or f"Peak {self.peak_index + 1}"),
                rule=str(peak.get("rule") or "D2h_Ag"),
                angles=angles,
                areas=column(area_column),
                centers=np.full(angles.shape, float(target_cm1), dtype=float),
                gammas=column(gamma_column),
                heights=column(height_column),
                config=str(result.get("config") or "parallel"),
                run_id=f"dataset-{result.get('dataset_index', result_position)}",
                run_label=str(result.get("label") or f"Dataset {result_position + 1}"),
            ))
            result_positions.append(result_position)
        return peak, row_fits, result_positions

    def plot_peak(self):
        self.figure.clear()
        self._pick_targets = {}
        axis = self.figure.add_subplot(111, projection="polar")
        if not self.engine.peaks:
            axis.text(0.5, 0.5, "No fitted peaks", ha="center", va="center", transform=axis.transAxes)
            self.canvas.draw_idle()
            return
        try:
            peak, row_fits, result_positions = self._row_fits()
            tensor = polar_area_fitting.fit_tensor_for_peak(row_fits, fit_state=self.engine.to_dict())
            rule = analysis.RULE_METADATA.get(tensor.rule) or analysis.RULE_METADATA["D2h_Ag"]
            theta_dense = np.linspace(0.0, 360.0, 721)
            theta_rad = np.deg2rad(theta_dense)
            global_params = [float(peak["ang_params"][name][0]) for name in rule["params"]]
            global_gamma = abs(float(peak["spec_params"]["gamma"][0])) + 1e-9
            anomaly_total = 0
            palette = {
                "parallel": {"points": "#222222", "row": "#d62728", "global": "#1f77b4", "marker": "o", "style": "-"},
                "cross": {"points": "#777777", "row": "#ff7f0e", "global": "#17becf", "marker": "s", "style": "-"},
            }
            for row_fit, result_position in zip(row_fits, result_positions):
                style = palette.get(row_fit.config, palette["parallel"])
                valid = np.isfinite(row_fit.angles) & np.isfinite(row_fit.areas)
                if valid.any():
                    points = axis.scatter(
                        np.deg2rad(row_fit.angles[valid]),
                        row_fit.areas[valid],
                        color=style["points"],
                        marker=style["marker"],
                        s=28,
                        picker=7,
                        label=f"{row_fit.run_label} row areas (clickable)",
                    )
                    self._pick_targets[points] = {
                        "result_position": result_position,
                        "row_indices": np.flatnonzero(valid),
                    }
                row_curve = rule["func"](theta_dense, row_fit.config, *tensor.params)
                global_curve = rule["func"](theta_dense, row_fit.config, *global_params) / global_gamma
                axis.plot(theta_rad, row_curve, color=style["row"], linestyle=style["style"], linewidth=1.5, label=f"{row_fit.run_label} row tensor")
                axis.plot(theta_rad, global_curve, color=style["global"], linestyle=":", linewidth=1.4, label=f"{row_fit.run_label} global")
                result = self.results[result_position]
                statuses = result.get("row_status") or []
                anomaly_rows = []
                for row_index, status in enumerate(statuses):
                    anomalies = status.get("anomalies", []) if isinstance(status, dict) else []
                    if any(int(item.get("peak_index", -1)) == self.peak_index for item in anomalies):
                        anomaly_rows.append(row_index)
                anomaly_rows = [index for index in anomaly_rows if index < row_fit.angles.size and np.isfinite(row_fit.areas[index])]
                if anomaly_rows:
                    anomaly_total += len(anomaly_rows)
                    anomaly_points = axis.scatter(
                        np.deg2rad(row_fit.angles[anomaly_rows]),
                        row_fit.areas[anomaly_rows],
                        color="#d00000",
                        edgecolor="white",
                        marker="*",
                        s=100,
                        linewidth=0.7,
                        picker=7,
                        zorder=10,
                        label=f"{row_fit.run_label} >2× anomaly",
                    )
                    self._pick_targets[anomaly_points] = {
                        "result_position": result_position,
                        "row_indices": np.asarray(anomaly_rows, dtype=int),
                    }
                if self.selected_point is not None:
                    selected_result, selected_row, selected_peak = self.selected_point
                    if selected_result == result_position and selected_peak == self.peak_index and selected_row < row_fit.angles.size and np.isfinite(row_fit.areas[selected_row]):
                        axis.scatter(
                            [np.deg2rad(row_fit.angles[selected_row])],
                            [row_fit.areas[selected_row]],
                            facecolors="none",
                            edgecolors="#00b7ff",
                            marker="o",
                            s=165,
                            linewidth=2.2,
                            zorder=12,
                            label="Current inspection cursor",
                        )
            axis.set_theta_zero_location("E")
            axis.set_theta_direction(-1)
            axis.grid(True, alpha=0.35)
            axis.set_title(f"{tensor.name} polar area fit")
            axis.legend(fontsize=7, loc="upper right", bbox_to_anchor=(1.32, 1.15))
            self.figure.subplots_adjust(left=0.08, right=0.78, top=0.9, bottom=0.08)
            parameter_text = ", ".join(f"{name}={value:.6g}" for name, value in zip(tensor.param_names, tensor.params))
            selection_text = ""
            if self.selected_point is not None:
                selected_result, selected_row, _selected_peak = self.selected_point
                fit_position = result_positions.index(selected_result) if selected_result in result_positions else -1
                if fit_position >= 0 and selected_row < row_fits[fit_position].angles.size:
                    selected_fit = row_fits[fit_position]
                    selection_text = f"\nSelected: {selected_fit.run_label}, row {selected_row}, angle {selected_fit.angles[selected_row]:.5g}°"
            self.summary.SetValue(
                f"Peak: {tensor.name}\nRule: {tensor.rule}\nRow-area tensor parameters: {parameter_text}\n"
                f"Detected >2× global-area anomalies: {anomaly_total}{selection_text}\n"
                "Click any row-area marker to move the 2D inspection cursor to that row and peak."
            )
        except Exception as exc:
            axis.text(0.5, 0.5, f"Cannot fit polar areas\n{exc}", ha="center", va="center", transform=axis.transAxes)
            self.summary.SetValue(str(exc))
        self.canvas.draw_idle()


class ValidationFrame(wx.Dialog):
    def __init__(self, results, parent=None, engine=None, global_matrices=None, row_fit_config=None, initial_selection=None):
        super().__init__(
            parent,
            title="Row-by-Row Fit Inspector",
            size=(1450, 950),
            style=wx.DEFAULT_DIALOG_STYLE | wx.RESIZE_BORDER | wx.MAXIMIZE_BOX,
        )
        self.results = results
        self.engine = engine
        self.anomaly_threshold = 2.0
        self.engine.annotate_row_fit_anomalies(self.results, threshold=self.anomaly_threshold)
        self.row_fit_config = engine.normalized_row_fit_config(row_fit_config) if engine is not None else {}
        self.global_matrices = {int(item["dataset_index"]): np.asarray(item["matrix"]) for item in (global_matrices or [])}
        self.global_param_tables = {int(table["dataset_idx"]): table for table in engine.global_fit_trace_tables()}
        self.parent_dialog = parent if hasattr(parent, "_persist_row_fit_caches") else None
        self.selected_dataset_idx = 0
        self.slice_angle_idx = 0
        self.slice_shift_idx = 0
        if self.results and isinstance(initial_selection, dict):
            requested_dataset = int(initial_selection.get("dataset_index", 0))
            for position, result in enumerate(self.results):
                if int(result.get("dataset_index", position)) == requested_dataset:
                    self.selected_dataset_idx = position
                    break
            selected_result = self.results[self.selected_dataset_idx]
            x_axis = np.asarray(selected_result.get("x", []), dtype=float)
            angle_axis = np.asarray(selected_result.get("ang", []), dtype=float)
            requested_x = initial_selection.get("x_value")
            requested_angle = initial_selection.get("angle_value")
            if x_axis.size and requested_x is not None:
                self.slice_shift_idx = int(np.nanargmin(np.abs(x_axis - float(requested_x))))
            if angle_axis.size and requested_angle is not None:
                self.slice_angle_idx = int(np.nanargmin(np.abs(angle_axis - float(requested_angle))))
        self.init_ui()
        self.update_plots()
        self.load_row_grid()

    def init_ui(self):
        main_sizer = wx.BoxSizer(wx.VERTICAL)
        splitter = wx.SplitterWindow(self)
        left_panel = wx.Panel(splitter)
        right_panel = wx.Panel(splitter)

        left_sizer = wx.BoxSizer(wx.VERTICAL)
        heading = wx.StaticText(left_panel, label="Row-by-Row Fit Reconstruction")
        heading_font = heading.GetFont()
        heading_font.SetWeight(wx.FONTWEIGHT_BOLD)
        heading.SetFont(heading_font)
        left_sizer.Add(heading, 0, wx.ALL, 8)

        selection_grid = wx.FlexGridSizer(2, 2, 5, 5)
        selection_grid.AddGrowableCol(1, 1)
        selection_grid.Add(wx.StaticText(left_panel, label="Dataset:"), 0, wx.ALIGN_CENTER_VERTICAL)
        self.choice_dataset = wx.Choice(left_panel, choices=[str(res.get("label", f"dataset {idx + 1}")) for idx, res in enumerate(self.results)])
        if self.results:
            self.choice_dataset.SetSelection(self.selected_dataset_idx)
        self.choice_dataset.Bind(wx.EVT_CHOICE, self.on_dataset_choice)
        selection_grid.Add(self.choice_dataset, 1, wx.EXPAND)
        selection_grid.Add(wx.StaticText(left_panel, label="Row:"), 0, wx.ALIGN_CENTER_VERTICAL)
        self.spin_row = wx.SpinCtrl(left_panel, min=0, max=max(0, len(self.results[0].get("ang", [])) - 1) if self.results else 0)
        self.spin_row.Bind(wx.EVT_SPINCTRL, self.on_row_spin)
        self.spin_row.Bind(wx.EVT_TEXT_ENTER, self.on_row_spin)
        if self.results:
            selected_angles = self.results[self.selected_dataset_idx].get("ang", [])
            self.spin_row.SetRange(0, max(0, len(selected_angles) - 1))
            self.spin_row.SetValue(self.slice_angle_idx)
        selection_grid.Add(self.spin_row, 1, wx.EXPAND)
        left_sizer.Add(selection_grid, 0, wx.LEFT | wx.RIGHT | wx.BOTTOM | wx.EXPAND, 8)

        self.row_label = wx.StaticText(left_panel, label="Row parameters")
        left_sizer.Add(self.row_label, 0, wx.LEFT | wx.RIGHT | wx.TOP | wx.EXPAND, 8)
        self.row_status = wx.StaticText(left_panel, label="")
        self.row_status.Wrap(330)
        left_sizer.Add(self.row_status, 0, wx.ALL | wx.EXPAND, 8)

        self.param_grid = wx.grid.Grid(left_panel)
        self.param_grid.CreateGrid(0, 4)
        self.param_grid.SetColLabelValue(0, "Parameter")
        self.param_grid.SetColLabelValue(1, "Row")
        self.param_grid.SetColLabelValue(2, "Global")
        self.param_grid.SetColLabelValue(3, "Ratio")
        self.param_grid.SetRowLabelSize(0)
        self.param_grid.SetColSize(0, 155)
        self.param_grid.SetColSize(1, 75)
        self.param_grid.SetColSize(2, 75)
        self.param_grid.SetColSize(3, 55)
        left_sizer.Add(self.param_grid, 1, wx.EXPAND | wx.ALL, 8)

        finite_values = []
        for result in self.results:
            values = np.asarray(result.get("z_raw", []), dtype=float)
            finite_values.extend(values[np.isfinite(values)].tolist())
        if finite_values:
            default_vmin, default_vmax = np.nanpercentile(finite_values, [1, 99])
        else:
            default_vmin, default_vmax = 0.0, 1.0
        display_grid = wx.FlexGridSizer(2, 2, 5, 5)
        display_grid.AddGrowableCol(1, 1)
        self.txt_vmin = wx.TextCtrl(left_panel, value=f"{default_vmin:.6g}", style=wx.TE_PROCESS_ENTER)
        self.txt_vmax = wx.TextCtrl(left_panel, value=f"{default_vmax:.6g}", style=wx.TE_PROCESS_ENTER)
        self.txt_vmin.Bind(wx.EVT_TEXT_ENTER, self.on_viz)
        self.txt_vmax.Bind(wx.EVT_TEXT_ENTER, self.on_viz)
        display_grid.AddMany([
            wx.StaticText(left_panel, label="VMin:"), self.txt_vmin,
            wx.StaticText(left_panel, label="VMax:"), self.txt_vmax,
        ])
        left_sizer.Add(display_grid, 0, wx.LEFT | wx.RIGHT | wx.TOP | wx.EXPAND, 8)

        action_grid = wx.FlexGridSizer(0, 2, 5, 5)
        for label, handler in [
            ("Row Settings...", self.on_settings),
            ("Refit Selected Row", self.on_refit_row),
            ("Refit All Rows", self.on_refit_all),
            ("Apply Row Edit", self.on_apply_row_edit),
            ("Next Anomaly", self.on_next_anomaly),
            ("Polar Area Fit...", self.on_polar_area_fit),
            ("Save Caches", self.on_save_caches),
            ("Export Areas", self.on_export_areas),
        ]:
            button = wx.Button(left_panel, label=label)
            button.Bind(wx.EVT_BUTTON, handler)
            action_grid.Add(button, 1, wx.EXPAND)
        action_grid.AddGrowableCol(0, 1)
        action_grid.AddGrowableCol(1, 1)
        left_sizer.Add(action_grid, 0, wx.ALL | wx.EXPAND, 8)
        close_button = wx.Button(left_panel, label="Close")
        close_button.Bind(wx.EVT_BUTTON, lambda event: self.Close())
        left_sizer.Add(close_button, 0, wx.LEFT | wx.RIGHT | wx.BOTTOM | wx.EXPAND, 8)
        left_panel.SetSizer(left_sizer)

        right_sizer = wx.BoxSizer(wx.VERTICAL)
        self.fig = Figure()
        self.canvas = FigureCanvas(right_panel, -1, self.fig)
        self.toolbar = NavigationToolbar(self.canvas)
        right_sizer.Add(self.toolbar, 0, wx.EXPAND)
        right_sizer.Add(self.canvas, 1, wx.EXPAND)
        right_panel.SetSizer(right_sizer)

        gs = self.fig.add_gridspec(2, 4, height_ratios=[1.2, 1], wspace=0.3, hspace=0.3,
                                   left=0.05, right=0.98, bottom=0.08, top=0.95)
        self.ax_raw1 = self.fig.add_subplot(gs[0, 0])
        self.ax_rec1 = self.fig.add_subplot(gs[0, 1], sharex=self.ax_raw1, sharey=self.ax_raw1)
        self.ax_spec1 = self.fig.add_subplot(gs[1, 0], sharex=self.ax_raw1)
        self.ax_ang1 = self.fig.add_subplot(gs[1, 1])
        self.ax_raw2 = self.fig.add_subplot(gs[0, 2], sharex=self.ax_raw1, sharey=self.ax_raw1)
        self.ax_rec2 = self.fig.add_subplot(gs[0, 3], sharex=self.ax_raw1, sharey=self.ax_raw1)
        self.ax_spec2 = self.fig.add_subplot(gs[1, 2], sharex=self.ax_raw1)
        self.ax_ang2 = self.fig.add_subplot(gs[1, 3])
        self.canvas.mpl_connect('button_press_event', self.on_click)

        splitter.SplitVertically(left_panel, right_panel, 360)
        splitter.SetMinimumPaneSize(280)
        main_sizer.Add(splitter, 1, wx.EXPAND)
        self.SetSizer(main_sizer)

    def on_export_areas(self, e):
        with wx.DirDialog(self, "Choose Output Directory for Row-Fit Area CSVs") as dlg:
            if dlg.ShowModal() != wx.ID_OK:
                return
            out_dir = dlg.GetPath()
        for idx, res in enumerate(self.results):
            label = str(res.get("label", f"dataset_{idx+1}")).replace(" ", "_").replace("(", "").replace(")", "")
            path = os.path.join(out_dir, f"RowFit_Areas_{label}.csv")
            headers = res.get("headers") or []
            rows = res.get("rows_params") or res.get("params") or []
            with open(path, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(headers)
                writer.writerows(rows)
        wx.MessageBox(f"Exported row-fit CSVs to:\n{out_dir}", "Export Complete", wx.OK | wx.ICON_INFORMATION)

    def on_click(self, e):
        if self.toolbar.mode != '': return
        ds_idx = -1
        if e.inaxes in [self.ax_raw1, self.ax_rec1]: ds_idx = 0
        elif e.inaxes in [self.ax_raw2, self.ax_rec2]: ds_idx = 1

        if ds_idx != -1 and ds_idx < len(self.results):
            res = self.results[ds_idx]
            if e.xdata is not None and e.ydata is not None:
                self.slice_shift_idx = np.abs(res["x"] - e.xdata).argmin()
                self.slice_angle_idx = np.abs(res["ang"] - e.ydata).argmin()
                self.selected_dataset_idx = ds_idx
                if self.choice_dataset.GetCount() > ds_idx:
                    self.choice_dataset.SetSelection(ds_idx)
                self.spin_row.SetRange(0, max(0, len(res["ang"]) - 1))
                self.spin_row.SetValue(int(self.slice_angle_idx))
                self.load_row_grid()
                self.update_plots()

    def on_dataset_choice(self, event):
        idx = self.choice_dataset.GetSelection()
        if idx == wx.NOT_FOUND:
            return
        self.selected_dataset_idx = idx
        res = self.results[idx]
        self.spin_row.SetRange(0, max(0, len(res.get("ang", [])) - 1))
        self.slice_angle_idx = min(self.slice_angle_idx, max(0, len(res.get("ang", [])) - 1))
        self.slice_shift_idx = min(self.slice_shift_idx, max(0, len(res.get("x", [])) - 1))
        self.spin_row.SetValue(int(self.slice_angle_idx))
        self.load_row_grid()
        self.update_plots()

    def on_row_spin(self, event):
        self.slice_angle_idx = int(self.spin_row.GetValue())
        self.load_row_grid()
        self.update_plots()

    def on_viz(self, event):
        self.update_plots()

    def _selected_result(self):
        if not self.results:
            return None
        idx = max(0, min(int(self.selected_dataset_idx), len(self.results) - 1))
        return self.results[idx]

    def load_row_grid(self):
        res = self._selected_result()
        if res is None:
            return
        headers = list(res.get("headers") or [])
        rows = res.get("rows_params") or res.get("params") or []
        row_idx = max(0, min(int(self.slice_angle_idx), len(rows) - 1)) if rows else 0
        if self.param_grid.GetNumberRows():
            self.param_grid.DeleteRows(0, self.param_grid.GetNumberRows())
        angle = np.asarray(res.get("ang", []), dtype=float)
        angle_text = f" ({angle[row_idx]:.3g}°)" if row_idx < angle.size else ""
        self.row_label.SetLabel(f"Row parameters: {res.get('label', '')} row {row_idx}{angle_text}")
        if not rows:
            self.row_status.SetLabel("")
            return
        statuses = res.get("row_status") or []
        info = statuses[row_idx] if row_idx < len(statuses) else None
        anomalies = info.get("anomalies", []) if isinstance(info, dict) else []
        if info:
            rmse = info.get("rmse", np.nan); rmse_text = f"; RMSE {float(rmse):.4g}" if rmse is not None and np.isfinite(float(rmse)) else ""
            status = ("OK" if info.get("success") else "FAILED") + rmse_text
            if info.get("at_bounds"): status += "; bounds: " + ", ".join(info["at_bounds"][:3])
            elif not info.get("success"): status += "; " + str(info.get("message", ""))
        else: status = self._row_status_text(headers, rows, row_idx)
        if anomalies:
            anomaly_labels = []
            for anomaly in anomalies[:3]:
                ratio = anomaly.get("ratio")
                ratio_text = f"{float(ratio):.2f}×" if ratio is not None and np.isfinite(float(ratio)) else "global≈0"
                anomaly_labels.append(f"{anomaly.get('peak_name', 'peak')} {ratio_text}")
            status += "; ANOMALY: " + ", ".join(anomaly_labels)
        self.row_status.SetLabel(status)
        colour = wx.Colour(190, 25, 25) if anomalies or (info and not info.get("success")) else (wx.Colour(170, 90, 0) if info and info.get("at_bounds") else wx.Colour(20, 110, 45))
        self.row_status.SetForegroundColour(colour)
        self.param_grid.AppendRows(len(headers))
        row = rows[row_idx]
        dataset_index = int(res.get("dataset_index", self.selected_dataset_idx))
        global_table = self.global_param_tables.get(dataset_index, {})
        global_headers = [str(value) for value in (global_table.get("headers") or [])]
        global_rows = global_table.get("rows") or []
        global_row = global_rows[row_idx] if row_idx < len(global_rows) else []
        anomalous_parameters = {str(anomaly.get("parameter", "")) for anomaly in anomalies}
        bound_parameters = {str(parameter) for parameter in (info.get("at_bounds", []) if isinstance(info, dict) else [])}
        for idx, header in enumerate(headers):
            self.param_grid.SetCellValue(idx, 0, str(header))
            self.param_grid.SetReadOnly(idx, 0, True)
            try:
                value = row[idx]
            except Exception:
                value = np.nan
            self.param_grid.SetCellValue(idx, 1, f"{float(value):.8g}" if np.isfinite(float(value)) else "nan")
            global_value = np.nan
            if header in global_headers:
                global_index = global_headers.index(header)
                try:
                    global_value = float(global_row[global_index])
                except Exception:
                    global_value = np.nan
            self.param_grid.SetCellValue(idx, 2, f"{global_value:.8g}" if np.isfinite(global_value) else "")
            ratio_text = ""
            if header.endswith(("_Area", "_Gamma")) and np.isfinite(float(value)) and np.isfinite(global_value):
                if abs(global_value) > 1e-12:
                    ratio_text = f"{float(value) / abs(global_value):.3g}×"
                elif float(value) > 0:
                    ratio_text = "∞"
            self.param_grid.SetCellValue(idx, 3, ratio_text)
            self.param_grid.SetReadOnly(idx, 2, True)
            self.param_grid.SetReadOnly(idx, 3, True)
            if header in anomalous_parameters:
                for column in range(4):
                    self.param_grid.SetCellBackgroundColour(idx, column, wx.Colour(255, 215, 215))
                self.param_grid.SetCellTextColour(idx, 1, wx.Colour(175, 0, 0))
            if header.endswith("_Gamma") and header in bound_parameters:
                for column in range(4):
                    self.param_grid.SetCellBackgroundColour(idx, column, wx.Colour(255, 242, 153))
                self.param_grid.SetCellTextColour(idx, 1, wx.Colour(110, 75, 0))
            if idx == 0:
                self.param_grid.SetReadOnly(idx, 1, True)
        self.param_grid.AutoSizeRows()

    def _row_status_text(self, headers, rows, row_idx: int) -> str:
        try:
            table = np.asarray(rows, dtype=float)
        except Exception:
            return ""
        if table.ndim != 2 or row_idx >= table.shape[0]:
            return ""
        notes = []
        row = table[row_idx]
        for col, header in enumerate(headers):
            if col >= table.shape[1] or col >= row.size:
                continue
            value = row[col]
            column = table[:, col]
            finite = column[np.isfinite(column)]
            if not np.isfinite(value):
                notes.append(f"{header}=NaN")
                continue
            if header.endswith("_Gamma") and finite.size:
                lo = float(np.nanmin(finite))
                hi = float(np.nanmax(finite))
                if abs(value - lo) < 1e-9 or abs(value - hi) < 1e-9:
                    notes.append(f"{header} bound-like")
            if header.endswith(("_Area", "_Height")) and finite.size >= 5:
                med = float(np.nanmedian(finite))
                mad = float(np.nanmedian(np.abs(finite - med)))
                scale = mad if mad > 0 else float(np.nanstd(finite))
                if scale > 0 and abs(value - med) > 10.0 * scale:
                    notes.append(f"{header} outlier")
        return "; ".join(notes[:3])

    def on_apply_row_edit(self, event):
        res = self._selected_result()
        if res is None:
            return
        rows = res.get("rows_params") or res.get("params") or []
        if not rows:
            return
        row_idx = max(0, min(int(self.slice_angle_idx), len(rows) - 1))
        row = list(rows[row_idx])
        for idx in range(min(len(row), self.param_grid.GetNumberRows())):
            if idx == 0:
                continue
            try:
                row[idx] = float(self.param_grid.GetCellValue(idx, 1))
            except Exception:
                row[idx] = np.nan
        rows[row_idx] = row
        res["rows_params"] = rows
        res["params"] = rows
        self._reconstruct_result_row(res, row_idx)
        statuses = res.setdefault("row_status", [{} for _ in rows])
        while len(statuses) < len(rows): statuses.append({})
        statuses[row_idx] = {"success": True, "message": "Manually edited", "n_points": 0, "rmse": np.nan, "at_bounds": []}
        self.engine.annotate_row_fit_anomalies(self.results, threshold=self.anomaly_threshold)
        self.load_row_grid()
        self.update_plots()

    def on_next_anomaly(self, event):
        positions = []
        for dataset_position, result in enumerate(self.results):
            for row_index, status in enumerate(result.get("row_status") or []):
                if isinstance(status, dict) and status.get("anomalies"):
                    positions.append((dataset_position, row_index))
        if not positions:
            wx.MessageBox("No peak areas exceed 2× their frozen global-fit values.", "Row-Fit Anomalies", wx.OK | wx.ICON_INFORMATION)
            return
        current = (int(self.selected_dataset_idx), int(self.slice_angle_idx))
        next_position = positions[0]
        for position in positions:
            if position > current:
                next_position = position
                break
        self.selected_dataset_idx, self.slice_angle_idx = next_position
        result = self.results[self.selected_dataset_idx]
        self.choice_dataset.SetSelection(self.selected_dataset_idx)
        self.spin_row.SetRange(0, max(0, len(result.get("ang", [])) - 1))
        self.spin_row.SetValue(self.slice_angle_idx)
        self.load_row_grid()
        self.update_plots()

    def on_polar_area_fit(self, event):
        if not self.engine.peaks:
            wx.MessageBox("No fitted peaks are available for a polar area fit.", "Polar Area Fit", wx.OK | wx.ICON_INFORMATION)
            return
        initial_peak = 0
        result = self._selected_result()
        statuses = (result.get("row_status") or []) if result else []
        selected_anomalies = []
        if self.slice_angle_idx < len(statuses) and isinstance(statuses[self.slice_angle_idx], dict):
            selected_anomalies = statuses[self.slice_angle_idx].get("anomalies") or []
            anomalies = selected_anomalies
            if anomalies:
                initial_peak = max(0, int(anomalies[0].get("peak_index", 0)))
        if not selected_anomalies:
            selected_x = None
            if result is not None:
                x_axis = np.asarray(result.get("x", []), dtype=float)
                if x_axis.size:
                    selected_x = float(x_axis[min(self.slice_shift_idx, x_axis.size - 1)])
            if selected_x is not None:
                centers = np.asarray([float(peak.get("spec_params", {}).get("x0", [np.nan])[0]) for peak in self.engine.peaks], dtype=float)
                if np.isfinite(centers).any():
                    initial_peak = int(np.nanargmin(np.abs(centers - selected_x)))
        dialog = PolarAreaFitDialog(
            self,
            self.engine,
            self.results,
            initial_peak_index=initial_peak,
            on_point_selected=self.on_polar_point_selected,
        )
        try:
            dialog.ShowModal()
        finally:
            dialog.Destroy()

    def on_polar_point_selected(self, result_position, row_index, peak_index):
        if not (0 <= int(result_position) < len(self.results)):
            return
        if not (0 <= int(peak_index) < len(self.engine.peaks)):
            return
        self.selected_dataset_idx = int(result_position)
        result = self.results[self.selected_dataset_idx]
        angles = np.asarray(result.get("ang", []), dtype=float)
        x_axis = np.asarray(result.get("x", []), dtype=float)
        if angles.size:
            self.slice_angle_idx = max(0, min(int(row_index), angles.size - 1))
        peak_center = float(self.engine.peaks[int(peak_index)].get("spec_params", {}).get("x0", [np.nan])[0])
        if x_axis.size and np.isfinite(peak_center):
            self.slice_shift_idx = int(np.nanargmin(np.abs(x_axis - peak_center)))
        self.choice_dataset.SetSelection(self.selected_dataset_idx)
        self.spin_row.SetRange(0, max(0, angles.size - 1))
        self.spin_row.SetValue(self.slice_angle_idx)
        self.load_row_grid()
        self.update_plots()

    def _sync_config_to_parent(self):
        self.engine.set_row_fit_config(self.row_fit_config)
        if self.parent_dialog is not None:
            self.parent_dialog.engine.set_row_fit_config(self.row_fit_config)

    def on_settings(self, event):
        dlg = RowFitSettingsDialog(self, self.engine, self.row_fit_config)
        try:
            if dlg.ShowModal() == wx.ID_OK: self.row_fit_config = dlg.get_config(); self._sync_config_to_parent()
        finally: dlg.Destroy()

    def _run_refit(self, only_rows=None):
        progress = wx.ProgressDialog("Row Fit", "Fitting selected rows...", parent=self, style=wx.PD_APP_MODAL | wx.PD_ELAPSED_TIME); progress.Pulse()
        try: success, message, results = self.engine.validate_row_by_row(self.row_fit_config, only_rows=only_rows, existing_results=self.results if only_rows is not None else None)
        finally: progress.Destroy()
        if not success or not results: wx.MessageBox(message, "Row Fit Failed", wx.OK | wx.ICON_ERROR); return
        self.results = results; self._sync_config_to_parent(); self.load_row_grid(); self.update_plots()

    def on_refit_all(self, event): self._run_refit()

    def on_refit_row(self, event):
        res = self._selected_result()
        if res is not None: self._run_refit({int(res.get("dataset_index", self.selected_dataset_idx)): [int(self.slice_angle_idx)]})

    def on_save_caches(self, event):
        self.on_apply_row_edit(event)
        if self.parent_dialog is None:
            wx.MessageBox("This validation window is not connected to a fitting dialog, so caches cannot be saved here.", "Save Caches", wx.OK | wx.ICON_INFORMATION)
            return
        count = self.parent_dialog._persist_row_fit_caches(self.results)
        wx.MessageBox(
            f"Staged {count} compact row-fit result(s).\nThey will be saved only when the main fitting dialog is committed.",
            "Stage Row Results",
            wx.OK | wx.ICON_INFORMATION,
        )

    def _reconstruct_result_row(self, res, row_idx: int):
        if self.engine is None:
            return
        headers = list(res.get("headers") or [])
        rows = res.get("rows_params") or res.get("params") or []
        if not rows or row_idx >= len(rows):
            return
        ds_idx = int(res.get("dataset_index", 0))
        x = np.asarray(res.get("x"), dtype=float)
        angle = float(rows[row_idx][0])
        values = {header: float(rows[row_idx][idx]) for idx, header in enumerate(headers) if idx < len(rows[row_idx])}
        bg_const = values.get("BG_Const", 0.0)
        bg_slope = values.get("BG_Slope_X", 0.0)
        y = bg_const + bg_slope * x
        if self.engine.si_bg_mode == "advanced_si_bg_v2":
            config_mode = res.get("config") or self.engine.datasets[ds_idx].get("config", "parallel")
            y += self.engine.evaluate_advanced_si_bg(
                x,
                np.full_like(x, angle, dtype=float),
                config_mode,
                values.get("Amp_Si", 0.0),
                values.get("Amp_B1g_Peak", 0.0),
                values.get("B1g_Center", self.engine.si_bg_peak_params["x0"][0]),
                values.get("B1g_Gamma", self.engine.si_bg_peak_params["gamma"][0]),
                values.get("B1g_Phi", self.engine.si_bg_peak_params["phi"][0]),
            )
        else:
            config_mode = res.get("config") or self.engine.datasets[ds_idx].get("config", "parallel")
            y += values.get("Amp_Si", 0.0) * self.engine.evaluate_si_bg(x, config_mode)

        for peak_idx, peak in enumerate(self.engine.peaks, start=1):
            name = str(peak.get("name", ""))
            prefix = f"P{peak_idx}_{name}_"
            area = values.get(prefix + "Area")
            gamma = values.get(prefix + "Gamma")
            if area is None or gamma is None or not np.isfinite(area) or not np.isfinite(gamma):
                continue
            center = float(peak.get("spec_params", {}).get("x0", [np.nan])[0])
            y += area * analysis.lorentzian_normalized(x, center, gamma)
        z_rec = np.asarray(res.get("z_rec"), dtype=float)
        if 0 <= row_idx < z_rec.shape[0] and z_rec.shape[1] == y.size:
            z_rec[row_idx, :] = y
            res["z_rec"] = z_rec

    def update_plots(self):
        saved_xlim = self.ax_raw1.get_xlim()
        saved_ylim = self.ax_raw1.get_ylim()
        selected = self._selected_result()
        selected_x = None
        selected_angle = None
        if selected is not None:
            selected_x_axis = np.asarray(selected.get("x", []), dtype=float)
            selected_angles = np.asarray(selected.get("ang", []), dtype=float)
            if selected_x_axis.size:
                selected_x = float(selected_x_axis[min(self.slice_shift_idx, selected_x_axis.size - 1)])
            if selected_angles.size:
                selected_angle = float(selected_angles[min(self.slice_angle_idx, selected_angles.size - 1)])
        try:
            display_vmin = float(self.txt_vmin.GetValue())
            display_vmax = float(self.txt_vmax.GetValue())
            if not np.isfinite(display_vmin) or not np.isfinite(display_vmax) or display_vmin >= display_vmax:
                raise ValueError
        except Exception:
            display_vmin = display_vmax = None

        def plot_res(idx, ax_raw, ax_rec, ax_spec, ax_ang):
            res = self.results[idx]
            x = np.asarray(res["x"], dtype=float)
            ang = np.asarray(res["ang"], dtype=float)
            z_raw = np.asarray(res["z_raw"], dtype=float)
            z_rec = np.asarray(res["z_rec"], dtype=float)
            ds_idx = int(res.get("dataset_index", idx))
            z_global = self.global_matrices.get(ds_idx)
            if z_global is None:
                z_global = self.engine.reconstruct(ds_idx)
            z_global = np.asarray(z_global, dtype=float)
            if display_vmin is None:
                vmin, vmax = np.nanpercentile(z_raw, [1, 99])
            else:
                vmin, vmax = display_vmin, display_vmax
            local_angle_idx = int(np.nanargmin(np.abs(ang - selected_angle))) if ang.size and selected_angle is not None else 0
            local_shift_idx = int(np.nanargmin(np.abs(x - selected_x))) if x.size and selected_x is not None else 0

            ax_raw.clear(); ax_rec.clear()
            ax_raw.pcolormesh(x, ang, z_raw, vmin=vmin, vmax=vmax, cmap='inferno', shading='auto')
            ax_raw.set_title(f"{res['label']} Data")
            ax_rec.pcolormesh(x, ang, z_rec, vmin=vmin, vmax=vmax, cmap='inferno', shading='auto')
            anomaly_rows = [
                row_index for row_index, status in enumerate(res.get("row_status") or [])
                if isinstance(status, dict) and status.get("anomalies") and row_index < ang.size
            ]
            anomaly_suffix = f" ({len(anomaly_rows)} anomalous rows)" if anomaly_rows else ""
            ax_rec.set_title(f"Row-Fit Reconstruction{anomaly_suffix}")
            ax_raw.set_ylabel("Angle")

            for axis in [ax_raw, ax_rec]:
                if ang.size:
                    axis.axhline(ang[local_angle_idx], color='w', ls='--', alpha=0.65)
                if x.size:
                    axis.axvline(x[local_shift_idx], color='w', ls='--', alpha=0.65)
                if self.engine.x_min_limit > np.nanmin(x):
                    axis.axvspan(np.nanmin(x), self.engine.x_min_limit, color='gray', alpha=0.35)
                if self.engine.x_max_limit < np.nanmax(x):
                    axis.axvspan(self.engine.x_max_limit, np.nanmax(x), color='gray', alpha=0.35)

            ax_spec.clear()
            if ang.size:
                ax_spec.plot(x, z_raw[local_angle_idx, :], color='black', alpha=0.65)
                ax_spec.plot(x, z_global[local_angle_idx, :], color='tab:blue')
                ax_spec.plot(x, z_rec[local_angle_idx, :], color='tab:red')
                ax_spec.set_title(f"Spectrum @ {ang[local_angle_idx]:.1f}°")
                ax_spec.legend(["Data", "Global", "Row"], fontsize=8)
                if x.size:
                    ax_spec.axvline(x[local_shift_idx], color='0.4', ls='--', alpha=0.5)

            ax_ang.clear()
            if x.size:
                ax_ang.plot(ang, z_raw[:, local_shift_idx], color='black', alpha=0.65)
                ax_ang.plot(ang, z_global[:, local_shift_idx], color='tab:blue')
                ax_ang.plot(ang, z_rec[:, local_shift_idx], color='tab:red')
                ax_ang.set_title(f"Angular @ {x[local_shift_idx]:.1f} cm-1")
                ax_ang.legend(["Data", "Global", "Row"], fontsize=8)
                if ang.size:
                    ax_ang.axvline(ang[local_angle_idx], color='0.4', ls='--', alpha=0.5)

        if len(self.results) > 0:
            plot_res(0, self.ax_raw1, self.ax_rec1, self.ax_spec1, self.ax_ang1)
        if len(self.results) > 1:
            plot_res(1, self.ax_raw2, self.ax_rec2, self.ax_spec2, self.ax_ang2)
        else:
            for ax in [self.ax_raw2, self.ax_rec2, self.ax_spec2, self.ax_ang2]:
                ax.clear()

        if saved_xlim != (0.0, 1.0) and saved_ylim != (0.0, 1.0):
            self.ax_raw1.set_xlim(saved_xlim)
            self.ax_raw1.set_ylim(saved_ylim)
            self.ax_ang1.set_xlim(saved_ylim)
            self.ax_ang2.set_xlim(saved_ylim)

        self.canvas.draw()


class ExportOptionsDialog(wx.Dialog):
    def __init__(self, parent, peak_names):
        super().__init__(parent, title="Export Options", size=(400, 500))
        self.peak_names = peak_names
        self.init_ui()

    def init_ui(self):
        sizer = wx.BoxSizer(wx.VERTICAL)

        sizer.Add(wx.StaticText(self, label="Select data to export to experiment:"), 0, wx.ALL, 10)

        self.cb_full_rec = wx.CheckBox(self, label="Full Reconstruction (Total)")
        self.cb_full_rec.SetValue(True)
        sizer.Add(self.cb_full_rec, 0, wx.LEFT|wx.RIGHT|wx.BOTTOM, 10)

        self.cb_polar_areas = wx.CheckBox(self, label="Peak-wise Polar Peak Area Map (Angle vs Peak)")
        self.cb_polar_areas.SetValue(True)
        sizer.Add(self.cb_polar_areas, 0, wx.LEFT|wx.RIGHT|wx.BOTTOM, 10)

        self.cb_background = wx.CheckBox(self, label="Background Reconstruction")
        self.cb_background.SetValue(False)
        sizer.Add(self.cb_background, 0, wx.LEFT|wx.RIGHT|wx.BOTTOM, 10)

        self.cb_bg_subtracted = wx.CheckBox(self, label="Background-Subtracted Map")
        self.cb_bg_subtracted.SetValue(False)
        sizer.Add(self.cb_bg_subtracted, 0, wx.LEFT|wx.RIGHT|wx.BOTTOM, 10)

        sizer.Add(wx.StaticLine(self), 0, wx.EXPAND|wx.ALL, 5)
        sizer.Add(wx.StaticText(self, label="Individual Peak Reconstructions:"), 0, wx.ALL, 10)

        self.peak_cbs = []
        for name in self.peak_names:
            cb = wx.CheckBox(self, label=f"Peak: {name}")
            cb.SetValue(False)
            self.peak_cbs.append(cb)
            sizer.Add(cb, 0, wx.LEFT|wx.RIGHT|wx.BOTTOM, 10)

        btn_sizer = self.CreateButtonSizer(wx.OK | wx.CANCEL)
        sizer.Add(btn_sizer, 0, wx.ALL | wx.ALIGN_RIGHT, 10)

        self.SetSizer(sizer)

    def get_options(self):
        return {
            "full_rec": self.cb_full_rec.GetValue(),
            "polar_areas": self.cb_polar_areas.GetValue(),
            "background": self.cb_background.GetValue(),
            "bg_subtracted": self.cb_bg_subtracted.GetValue(),
            "peaks": [cb.GetValue() for cb in self.peak_cbs]
        }

class MapFittingDialog(wx.Dialog):
    def __init__(
        self,
        parent,
        run1: Run,
        run2: Optional[Run] = None,
        params_run: Optional[Run] = None,
        single_config: str = "parallel",
        roles_preassigned: bool = False,
    ):
        super().__init__(parent, title=f"Joint 2D Fitting", size=(1400, 900),
                         style=wx.DEFAULT_DIALOG_STYLE | wx.RESIZE_BORDER | wx.MAXIMIZE_BOX)

        self.engine = analysis.MapFittingEngine()
        self.params_run = params_run

        self.single_config = "cross" if str(single_config).lower().startswith("cross") else "parallel"

        if run2 is None:
            self.run1, self.run2 = run1, None
            self.engine.datasets[0]["config"] = self.single_config
            self.engine.datasets[0]["label"] = "Parallel (XX)" if self.single_config == "parallel" else "Cross (YX)"
        else:
            # Determine which is XX and which is YX based on pol metadata
            r1_pol = str(run1.metadata.get("pol", "")).lower()
            r2_pol = str(run2.metadata.get("pol", "")).lower()

            swap = False
            if not roles_preassigned and ("yx" in r1_pol or "xy" in r1_pol or "cross" in r1_pol) and not ("xx" in r2_pol or "para" in r2_pol):
                swap = True

            if swap:
                self.run1, self.run2 = run2, run1
            else:
                self.run1, self.run2 = run1, run2

        if self.run1.is_2d:
            shift1, angles1, intensity1, background_angles1 = analysis.display_2d_with_acquisition_angles_from_run(self.run1)
            self.engine.set_data(
                0,
                shift1,
                angles1,
                intensity1,
                self.run1.nickname,
                background_ang=background_angles1,
            )
            self.engine.datasets[0]["config"] = self.single_config if self.run2 is None else "parallel"
            self.engine.datasets[0]["label"] = "Parallel (XX)" if self.engine.datasets[0]["config"] == "parallel" else "Cross (YX)"
        if self.run2 is not None and self.run2.is_2d:
            shift2, angles2, intensity2, background_angles2 = analysis.display_2d_with_acquisition_angles_from_run(self.run2)
            self.engine.set_data(
                1,
                shift2,
                angles2,
                intensity2,
                self.run2.nickname,
                background_ang=background_angles2,
            )

        if params_run:
            # The dialog is transactional: never let engine edits alias saved metadata.
            self.engine.from_dict(copy.deepcopy(params_run.metadata.get("fit_state", {})))

        if self.run2 is None:
            self.SetTitle(f"2D Fitting: {self.run1.nickname} ({self.engine.datasets[0]['label']})")
        else:
            self.SetTitle(f"Joint 2D Fitting: {self.run1.nickname} & {self.run2.nickname}")

        self.last_sel = 0
        self.selected_slice = self._default_selected_slice()

        self.result_runs = []
        self.pending_row_results = None
        self.commit_action = None
        self.committed_params_run = None
        self.commit_summary = None
        self._fit_running = False
        self._fit_cancel_requested = False
        self._fit_token = 0
        self._fit_progress = None
        self._fit_timer = wx.Timer(self)
        self.Bind(wx.EVT_TIMER, self._on_fit_timer, self._fit_timer)
        self.Bind(wx.EVT_CHAR_HOOK, self._on_char_hook)

        self.init_ui()
        self.refresh_ui()
        self.update_plots()
        if getattr(self.engine, "si_bg_load_warning", ""):
            wx.MessageBox(self.engine.si_bg_load_warning, "Si BG Profile", wx.OK | wx.ICON_WARNING)

    def _source_runs(self) -> List[Run]:
        return [r for r in (self.run1, self.run2) if r is not None]

    def _source_run_ids(self) -> List[str]:
        return [r.id for r in self._source_runs()]

    def _experiment(self):
        return getattr(self.GetParent(), "experiment", None)

    def _active_dataset_indices(self) -> List[int]:
        return self.engine.active_dataset_indices()

    def _default_selected_slice(self) -> Dict[str, Any]:
        active = self.engine.active_dataset_indices()
        ds_idx = active[0] if active else 0
        ds = self.engine.datasets[ds_idx]
        x_val = None
        angle_val = None
        if ds["x"] is not None and len(ds["x"]):
            x_val = float(ds["x"][len(ds["x"]) // 2])
        if ds["ang"] is not None and len(ds["ang"]):
            angle_val = float(ds["ang"][0])
        return {"dataset_index": ds_idx, "x_value": x_val, "angle_value": angle_val}

    def _slice_indices(self, dataset_index: int) -> Tuple[int, int]:
        ds = self.engine.datasets[dataset_index]
        x = np.asarray(ds["x"], dtype=float)
        ang = np.asarray(ds["ang"], dtype=float)
        x_value = self.selected_slice.get("x_value")
        angle_value = self.selected_slice.get("angle_value")
        ix = int(np.nanargmin(np.abs(x - x_value))) if x.size and x_value is not None and np.isfinite(float(x_value)) else 0
        iy = int(np.nanargmin(np.abs(ang - angle_value))) if ang.size and angle_value is not None and np.isfinite(float(angle_value)) else 0
        return ix, iy

    def _bg_count(self) -> int:
        return len(self._active_dataset_indices())

    def _selection_bg_dataset(self, sel: int) -> Optional[int]:
        active = self._active_dataset_indices()
        if 0 <= sel < len(active):
            return active[sel]
        return None

    def _selection_peak_index(self, sel: int) -> int:
        return sel - self._bg_count()

    def _bg_label(self, dataset_index: int) -> str:
        ds = self.engine.datasets[dataset_index]
        config_name = "XX/Parallel" if ds.get("config") == "parallel" else "YX/Cross"
        return f"Background ({config_name})"

    def _update_si_bg_label(self) -> None:
        if not hasattr(self, "si_bg_label"):
            return
        mode = getattr(self.engine, "si_bg_mode", "none")
        if mode and mode != "none":
            src = os.path.basename(getattr(self.engine, "si_bg_source_path", "") or "")
            self.si_bg_label.SetLabel(f"Si BG: {mode}" + (f" ({src})" if src else ""))
        else:
            self.si_bg_label.SetLabel("Si BG: none")

    def _show_error_details(self, title: str, message: str) -> None:
        text = str(message)
        if "\n" in text or len(text) > 500:
            dlg = wx.lib.dialogs.ScrolledMessageDialog(self, text, title)
            dlg.ShowModal()
            dlg.Destroy()
        else:
            wx.MessageBox(text, title, wx.OK | wx.ICON_ERROR)

    def make_fit_params_run(self) -> Run:
        names = "_".join(r.nickname for r in self._source_runs())
        text = self.engine.export_parameters_text()
        params_run = Run(
            id=new_run_id(prefix="params"),
            source_path="",
            metadata={
                "nickname": f"FitParams_{names}",
                "fit_state": copy.deepcopy(self.engine.to_dict()),
                "fit_parameters_text": text,
                "source_run_ids": self._source_run_ids(),
            },
            run_type=RunType.FIT_PARAMS
        )
        return params_run

    def _persist_row_fit_caches(self, results) -> int:
        if not results:
            return 0
        self.pending_row_results = copy.deepcopy(list(results))
        return len(self.pending_row_results)

    def commit_to_experiment(self, *, save_as_new: bool = False):
        self.save_grid(self.peak_list.GetSelection())
        exp = self._experiment()
        if exp is None:
            raise ValueError("The fitting dialog is not connected to an experiment.")
        target, summary = commit_fit_params_transaction(
            exp,
            self._source_runs(),
            self.engine.to_dict(),
            self.engine.export_parameters_text(),
            existing_params_run=self.params_run,
            save_as_new=save_as_new,
            row_results=self.pending_row_results,
            explicit_runs=self.result_runs,
        )
        self.committed_params_run = target
        self.commit_summary = summary
        return target, summary

    def init_ui(self):
        main_sizer = wx.BoxSizer(wx.VERTICAL)

        splitter = wx.SplitterWindow(self)
        left_panel = wx.Panel(splitter)
        right_panel = wx.Panel(splitter)

        # Left Panel (Controls)
        left_sizer = wx.BoxSizer(wx.VERTICAL)

        # Range
        range_box = wx.FlexGridSizer(2, 2, 5, 5)
        self.txt_min_x = wx.TextCtrl(left_panel, value="", style=wx.TE_PROCESS_ENTER)
        self.txt_max_x = wx.TextCtrl(left_panel, value="", style=wx.TE_PROCESS_ENTER)
        self.txt_min_x.Bind(wx.EVT_TEXT_ENTER, self.on_range_change)
        self.txt_max_x.Bind(wx.EVT_TEXT_ENTER, self.on_range_change)
        range_box.AddMany([wx.StaticText(left_panel, label="Min X:"), self.txt_min_x,
                           wx.StaticText(left_panel, label="Max X:"), self.txt_max_x])
        left_sizer.Add(range_box, 0, wx.ALL|wx.EXPAND, 5)

        # Peaks
        self.peak_list = wx.ListBox(left_panel)
        self.peak_list.Bind(wx.EVT_LISTBOX, self.on_peak_sel)
        self.peak_list.Bind(wx.EVT_LISTBOX_DCLICK, self.on_peak_rename_req)
        self.peak_list.Bind(wx.EVT_KEY_DOWN, self.on_peak_list_key)
        left_sizer.Add(wx.StaticText(left_panel, label="Peaks:"), 0, wx.ALL, 5)
        left_sizer.Add(self.peak_list, 0, wx.EXPAND|wx.ALL, 5)

        p_box = wx.BoxSizer(wx.HORIZONTAL)
        btn_add = wx.Button(left_panel, label="Add Peak")
        btn_add.Bind(wx.EVT_BUTTON, self.on_add_peak)
        btn_rem = wx.Button(left_panel, label="Remove")
        btn_rem.Bind(wx.EVT_BUTTON, self.on_rem_peak)
        p_box.Add(btn_add, 1, wx.ALL, 2)
        p_box.Add(btn_rem, 1, wx.ALL, 2)
        left_sizer.Add(p_box, 0, wx.EXPAND)

        self.rule_combo = wx.ComboBox(left_panel, choices=list(analysis.RULE_METADATA.keys()), style=wx.CB_READONLY)
        self.rule_combo.Bind(wx.EVT_COMBOBOX, self.on_rule_change)
        left_sizer.Add(self.rule_combo, 0, wx.EXPAND|wx.ALL, 5)

        self.auto_summary = wx.StaticText(left_panel, label="Auto: -")
        left_sizer.Add(self.auto_summary, 0, wx.EXPAND|wx.LEFT|wx.RIGHT|wx.BOTTOM, 5)

        si_box = wx.BoxSizer(wx.VERTICAL)
        self.si_bg_label = wx.StaticText(left_panel, label="Si BG: none")
        si_box.Add(self.si_bg_label, 0, wx.EXPAND|wx.LEFT|wx.RIGHT|wx.BOTTOM, 2)
        si_btns = wx.BoxSizer(wx.HORIZONTAL)
        btn_si_default = wx.Button(left_panel, label="Use Default Si BG")
        btn_si_choose = wx.Button(left_panel, label="Choose Si BG...")
        btn_si_clear = wx.Button(left_panel, label="Clear")
        btn_si_default.Bind(wx.EVT_BUTTON, self.on_use_default_si_bg)
        btn_si_choose.Bind(wx.EVT_BUTTON, self.on_choose_si_bg)
        btn_si_clear.Bind(wx.EVT_BUTTON, self.on_clear_si_bg)
        si_btns.Add(btn_si_default, 1, wx.ALL, 2)
        si_btns.Add(btn_si_choose, 1, wx.ALL, 2)
        si_btns.Add(btn_si_clear, 0, wx.ALL, 2)
        si_box.Add(si_btns, 0, wx.EXPAND)
        left_sizer.Add(si_box, 0, wx.EXPAND|wx.LEFT|wx.RIGHT|wx.BOTTOM, 5)

        self.grid = ParamsGrid(left_panel)
        left_sizer.Add(self.grid, 1, wx.EXPAND|wx.ALL, 5)

        # Viz limits
        v_box = wx.FlexGridSizer(2, 2, 5, 5)
        self.txt_vmin = wx.TextCtrl(left_panel, value="0", style=wx.TE_PROCESS_ENTER)
        self.txt_vmax = wx.TextCtrl(left_panel, value="1000", style=wx.TE_PROCESS_ENTER)
        self.txt_vmin.Bind(wx.EVT_TEXT_ENTER, self.on_viz)
        self.txt_vmax.Bind(wx.EVT_TEXT_ENTER, self.on_viz)
        v_box.AddMany([wx.StaticText(left_panel, label="VMin:"), self.txt_vmin,
                           wx.StaticText(left_panel, label="VMax:"), self.txt_vmax])
        left_sizer.Add(v_box, 0, wx.ALL|wx.EXPAND, 5)

        # Actions
        act_box = wx.FlexGridSizer(3, 2, 5, 5)
        self._fit_action_buttons = []
        for lbl, cb in [("Preview", self.on_preview), ("FIT Global", self.on_fit),
                        ("Validate (RowFit)", self.on_validate), ("Export Text", self.on_export_text),
                        ("Import Text", self.on_import_text), ("Export CSV Diag", self.on_export_diagnostics)]:
            btn = wx.Button(left_panel, label=lbl)
            btn.Bind(wx.EVT_BUTTON, cb)
            self._fit_action_buttons.append(btn)
            act_box.Add(btn, 1, wx.EXPAND)
        left_sizer.Add(act_box, 0, wx.ALL|wx.EXPAND, 5)

        self.chk_fit_rows_after_global = wx.CheckBox(
            left_panel,
            label="Also fit all rows after global fit (slow)",
        )
        self.chk_fit_rows_after_global.SetValue(False)
        left_sizer.Add(self.chk_fit_rows_after_global, 0, wx.LEFT|wx.RIGHT|wx.BOTTOM, 7)

        left_panel.SetSizer(left_sizer)

        # Right Panel (Plots)
        right_sizer = wx.BoxSizer(wx.VERTICAL)
        self.fig = Figure()
        self.canvas = FigureCanvas(right_panel, -1, self.fig)
        self.toolbar = NavigationToolbar(self.canvas)
        right_sizer.Add(self.toolbar, 0, wx.EXPAND)
        right_sizer.Add(self.canvas, 1, wx.EXPAND)
        right_panel.SetSizer(right_sizer)

        gs = self.fig.add_gridspec(2, 4, height_ratios=[1.2, 1], wspace=0.3, hspace=0.3,
                                   left=0.05, right=0.98, bottom=0.08, top=0.95)

        self.ax_raw1 = self.fig.add_subplot(gs[0, 0])
        self.ax_rec1 = self.fig.add_subplot(gs[0, 1], sharex=self.ax_raw1, sharey=self.ax_raw1)
        self.ax_spec1 = self.fig.add_subplot(gs[1, 0], sharex=self.ax_raw1)
        self.ax_ang1  = self.fig.add_subplot(gs[1, 1])

        self.ax_raw2 = self.fig.add_subplot(gs[0, 2], sharex=self.ax_raw1, sharey=self.ax_raw1)
        self.ax_rec2 = self.fig.add_subplot(gs[0, 3], sharex=self.ax_raw1, sharey=self.ax_raw1)
        self.ax_spec2 = self.fig.add_subplot(gs[1, 2], sharex=self.ax_raw1)
        self.ax_ang2  = self.fig.add_subplot(gs[1, 3])

        self.canvas.mpl_connect('button_press_event', self.on_click)

        splitter.SplitVertically(left_panel, right_panel, 350)
        main_sizer.Add(splitter, 1, wx.EXPAND)

        # Dialog buttons. Only these commit handlers may mutate the experiment.
        self.btn_export_exp = wx.Button(self, label="Export to Experiment...")
        self.btn_export_exp.Bind(wx.EVT_BUTTON, self.on_export_to_experiment)
        self.btn_save_new = wx.Button(self, label="Save as new params")
        self.btn_ok = wx.Button(self, wx.ID_OK, "OK")
        self.btn_cancel = wx.Button(self, wx.ID_CANCEL, "Cancel")
        self.btn_save_new.Bind(wx.EVT_BUTTON, self.on_save_as_new)
        self.btn_ok.Bind(wx.EVT_BUTTON, self.on_ok)
        self.btn_cancel.Bind(wx.EVT_BUTTON, self.on_cancel)

        h_sizer = wx.BoxSizer(wx.HORIZONTAL)
        h_sizer.Add(self.btn_export_exp, 0, wx.ALL | wx.ALIGN_CENTER_VERTICAL, 10)
        h_sizer.AddStretchSpacer()
        h_sizer.Add(self.btn_save_new, 0, wx.ALL | wx.ALIGN_CENTER_VERTICAL, 5)
        h_sizer.Add(self.btn_ok, 0, wx.ALL | wx.ALIGN_CENTER_VERTICAL, 5)
        h_sizer.Add(self.btn_cancel, 0, wx.ALL | wx.ALIGN_CENTER_VERTICAL, 5)

        main_sizer.Add(h_sizer, 0, wx.EXPAND)

        self.SetSizer(main_sizer)

    def on_ok(self, event):
        self.commit_action = "update"
        self.EndModal(wx.ID_OK)

    def on_save_as_new(self, event):
        self.commit_action = "save_new"
        self.EndModal(wx.ID_OK)

    def on_cancel(self, event):
        self.commit_action = None
        self.EndModal(wx.ID_CANCEL)

    def peak_identifier_label(self, idx, peak):
        rule = str(peak.get("rule", ""))
        name = str(peak.get("name", rule))
        try:
            center = float(peak.get("spec_params", {}).get("x0", [np.nan])[0])
        except Exception:
            center = np.nan
        center_text = f" @ {center:.4g}" if np.isfinite(center) else ""
        rule_text = f"{rule}{center_text}"
        looks_auto = bool(peak.get("auto_name", False)) or name == rule or name.startswith(f"{rule}(")
        if name and not looks_auto:
            rule_text = f"{name} | {rule_text}"
        return f"{idx + 1}: {rule_text}"

    def set_auto_summary(self, text):
        if not hasattr(self, "auto_summary"):
            return
        label = str(text).strip() if text else "Auto: -"
        self.auto_summary.SetLabel(label)
        try:
            self.auto_summary.Wrap(320)
        except Exception:
            pass

    def _load_si_bg_path(self, path: str, *, save_if_no_default: bool = False) -> bool:
        path = str(path or "").strip()
        if not path:
            wx.MessageBox("Choose a Si BG JSON profile first.", "Si BG Profile", wx.OK | wx.ICON_INFORMATION)
            return False
        success, msg = self.engine.load_si_bg(path)
        if success:
            if save_if_no_default and not config.get("default_si_bg_profile_path", ""):
                config.set("default_si_bg_profile_path", path)
            self._update_si_bg_label()
            self.refresh_ui()
            self.update_plots()
            wx.MessageBox(msg, "Si BG Profile", wx.OK | wx.ICON_INFORMATION)
            return True
        self._show_error_details("Si BG Profile", f"Failed to load Si BG profile:\n{msg}")
        self._update_si_bg_label()
        return False

    def on_use_default_si_bg(self, event):
        self._load_si_bg_path(config.get("default_si_bg_profile_path", ""), save_if_no_default=False)

    def on_choose_si_bg(self, event):
        base = config.get("default_si_bg_profile_path", "") or config.get("last_directory", "") or os.getcwd()
        default_dir = base if os.path.isdir(base) else os.path.dirname(base)
        with wx.FileDialog(self, "Choose Si BG JSON Profile", defaultDir=default_dir, wildcard="JSON files (*.json)|*.json", style=wx.FD_OPEN | wx.FD_FILE_MUST_EXIST) as dlg:
            if dlg.ShowModal() != wx.ID_OK:
                return
            self._load_si_bg_path(dlg.GetPath(), save_if_no_default=True)

    def on_clear_si_bg(self, event):
        self.engine.clear_si_bg()
        self._update_si_bg_label()
        self.refresh_ui()
        self.update_plots()

    def refresh_ui(self):
        curr = self.peak_list.GetSelection()
        self.peak_list.Clear()
        for ds_idx in self._active_dataset_indices():
            self.peak_list.Append(self._bg_label(ds_idx))
        for i, p in enumerate(self.engine.peaks):
            self.peak_list.Append(self.peak_identifier_label(i, p))
        sel = curr if curr != -1 else 0
        if self.peak_list.GetCount():
            sel = max(0, min(sel, self.peak_list.GetCount() - 1))
            self.peak_list.SetSelection(sel)
        else:
            sel = -1
        self.last_sel = self.peak_list.GetSelection()
        self.load_grid()
        self.txt_min_x.SetValue(f"{self.engine.x_min_limit:.6g}")
        self.txt_max_x.SetValue(f"{self.engine.x_max_limit:.6g}")
        self._update_si_bg_label()

    def on_peak_sel(self, e):
        self.save_grid(self.last_sel)
        self.last_sel = self.peak_list.GetSelection()
        self.load_grid()

    def on_peak_list_key(self, e):
        keycode = e.GetKeyCode()
        if keycode in [wx.WXK_RETURN, wx.WXK_NUMPAD_ENTER, wx.WXK_F2]:
            self.on_peak_rename_req(None)
        else:
            e.Skip()

    def on_peak_rename_req(self, e):
        sel = self.peak_list.GetSelection()
        peak_idx = self._selection_peak_index(sel)
        if 0 <= peak_idx < len(self.engine.peaks):
            peak = self.engine.peaks[peak_idx]
            dlg = wx.TextEntryDialog(self, "Enter new name for peak:", "Rename Peak", peak["name"])
            if dlg.ShowModal() == wx.ID_OK:
                self.engine.rename_peak(peak_idx, dlg.GetValue())
                self.refresh_ui()
                self.peak_list.SetSelection(sel)
                self.load_grid()
            dlg.Destroy()

    def load_grid(self):
        sel = self.peak_list.GetSelection()
        bg_ds_idx = self._selection_bg_dataset(sel)
        if bg_ds_idx is not None:
            self.rule_combo.Disable()
            self.set_auto_summary("Auto: -")
            bg = self.engine._bg_for_dataset(bg_ds_idx)
            labels = ["Offset", "Slope_X", "Slope_Theta (pre-rotation angle)", "Amp_Si"]
            data = [bg["offset"], bg["slope_x"], bg["slope_theta"], bg["amp_si"]]
            if self.engine.si_bg_mode == "advanced_si_bg_v2":
                labels.extend(["Amp_B1g_Peak", "B1g_Center", "B1g_Gamma", "B1g_Phi", "Profile_X_Shift"])
                data.extend([bg["amp_b1g_peak"], self.engine.si_bg_peak_params["x0"], self.engine.si_bg_peak_params["gamma"], self.engine.si_bg_peak_params["phi"], [self.engine.get_si_profile_shift(), 0.0, 0.0]])
            self.grid.load_data(labels, data)
        else:
            peak_idx = self._selection_peak_index(sel)
            if not (0 <= peak_idx < len(self.engine.peaks)):
                return
            self.rule_combo.Enable()
            p = self.engine.peaks[peak_idx]
            self.rule_combo.SetValue(p["rule"])
            self.set_auto_summary(p.get("auto_rule_summary") or f"Auto: {p['rule']}")
            lbls = ["Center (x0)", "Width (G)"] + analysis.RULE_METADATA[p["rule"]]["params"]
            dst = [p["spec_params"]["x0"], p["spec_params"]["gamma"]] + [p["ang_params"].get(n, [0.0, -np.inf, np.inf]) for n in analysis.RULE_METADATA[p["rule"]]["params"]]
            self.grid.load_data(lbls, dst)

    def save_grid(self, idx):
        bg_ds_idx = self._selection_bg_dataset(idx)
        if bg_ds_idx is not None:
            bg = self.engine._bg_for_dataset(bg_ds_idx)
            data = [bg["offset"], bg["slope_x"], bg["slope_theta"], bg["amp_si"]]
            if self.engine.si_bg_mode == "advanced_si_bg_v2":
                data.extend([bg["amp_b1g_peak"], self.engine.si_bg_peak_params["x0"], self.engine.si_bg_peak_params["gamma"], self.engine.si_bg_peak_params["phi"], [self.engine.get_si_profile_shift(), 0.0, 0.0]])
            self.grid.save_to_lists(data)
        else:
            peak_idx = self._selection_peak_index(idx)
            if not (0 <= peak_idx < len(self.engine.peaks)):
                return
            p = self.engine.peaks[peak_idx]
            self.grid.save_to_lists([p["spec_params"]["x0"], p["spec_params"]["gamma"]] + [p["ang_params"].get(n, [0.0, -np.inf, np.inf]) for n in analysis.RULE_METADATA[p["rule"]]["params"]])

    def on_add_peak(self, e):
        self.save_grid(self.peak_list.GetSelection())
        rule_name = self.rule_combo.GetValue() or "D2h_B1g"
        ds_idx = int(self.selected_slice.get("dataset_index", 0))
        if ds_idx not in self._active_dataset_indices():
            ds_idx = self._active_dataset_indices()[0] if self._active_dataset_indices() else 0
        ds = self.engine.datasets[ds_idx]
        x_val = self.selected_slice.get("x_value")
        angle_val = self.selected_slice.get("angle_value")
        ix, iy = self._slice_indices(ds_idx) if ds["x"] is not None and ds["ang"] is not None else (0, 0)
        if ds["x"] is not None and len(ds["x"]):
            x_val = float(ds["x"][ix])
        estimates = {}
        try:
            if ds["x"] is not None and ds["ang"] is not None and ds["z"] is not None:
                estimates = analysis.estimate_peak_defaults(
                    datasets=self.engine.datasets,
                    selected_dataset_index=ds_idx,
                    x_value=x_val,
                    angle_value=angle_val if angle_val is not None else float(ds["ang"][iy]),
                    rule_name=rule_name,
                    preserve_x0=True,
                )
                estimates["x0"] = x_val
                estimates.setdefault("spec_params", {})["x0"] = x_val
                rule_name = estimates.get("best_rule", rule_name)
        except Exception:
            estimates = {
                "best_rule": rule_name,
                "auto_rule_score": None,
                "auto_rule_summary": f"Auto: fallback {rule_name}",
            }

        new_peak = self.engine.add_peak(rule_name=rule_name, center=x_val, estimates=estimates)
        self.refresh_ui()
        try:
            self.peak_list.SetSelection(self.engine.peaks.index(new_peak) + self._bg_count())
        except ValueError:
            self.peak_list.SetSelection(self.peak_list.GetCount() - 1)
        self.on_peak_sel(None)

    def on_rem_peak(self, e):
        sel = self.peak_list.GetSelection()
        peak_idx = self._selection_peak_index(sel)
        if 0 <= peak_idx < len(self.engine.peaks):
            self.engine.peaks.pop(peak_idx)
            self.refresh_ui()

    def on_rule_change(self, e):
        sel = self.peak_list.GetSelection()
        peak_idx = self._selection_peak_index(sel)
        if 0 <= peak_idx < len(self.engine.peaks):
            rule_name = self.rule_combo.GetValue()
            peak = self.engine.peaks[peak_idx]
            x0 = float(peak.get("spec_params", {}).get("x0", [0.0])[0])
            ds_idx = int(self.selected_slice.get("dataset_index", 0))
            if ds_idx not in self._active_dataset_indices():
                ds_idx = self._active_dataset_indices()[0] if self._active_dataset_indices() else 0
            ds = self.engine.datasets[ds_idx]
            angle_val = self.selected_slice.get("angle_value")
            if angle_val is None and ds.get("ang") is not None and len(ds["ang"]):
                _ix, iy = self._slice_indices(ds_idx)
                angle_val = float(ds["ang"][iy])
            estimates = {}
            try:
                estimates = analysis.estimate_peak_defaults(
                    datasets=self.engine.datasets,
                    selected_dataset_index=ds_idx,
                    x_value=x0,
                    angle_value=angle_val,
                    rule_name=rule_name,
                    preserve_x0=True,
                    force_rule=True,
                )
                estimates["x0"] = x0
                estimates.setdefault("spec_params", {})["x0"] = x0
            except Exception:
                estimates = {"auto_rule_summary": f"Auto: fallback {rule_name}", "auto_rule_score": None}
            self.engine.set_peak_rule(peak_idx, rule_name, estimates=estimates)
            self.refresh_ui()
            self.peak_list.SetSelection(sel)
            self.load_grid()
            self.update_plots()

    def on_preview(self, e):
        self.save_grid(self.peak_list.GetSelection())
        self.update_plots()

    def on_viz(self, e):
        self.update_plots()

    def on_range_change(self, e):
        try:
            self.engine.x_min_limit = float(self.txt_min_x.GetValue())
            self.engine.x_max_limit = float(self.txt_max_x.GetValue())
            self.update_plots()
        except:
            pass

    def _set_fit_actions_enabled(self, enabled: bool) -> None:
        for btn in getattr(self, "_fit_action_buttons", []):
            try:
                btn.Enable(enabled)
            except Exception:
                pass
        try:
            self.btn_export_exp.Enable(enabled)
        except Exception:
            pass
        try:
            self.chk_fit_rows_after_global.Enable(enabled)
        except Exception:
            pass
        for name in ("btn_save_new", "btn_ok", "btn_cancel"):
            try:
                getattr(self, name).Enable(enabled)
            except Exception:
                pass

    def _on_char_hook(self, event):
        if event.GetKeyCode() == wx.WXK_ESCAPE and self._fit_running:
            self._request_global_fit_abort()
            return
        event.Skip()

    def _request_global_fit_abort(self) -> None:
        if not self._fit_running:
            return
        self._fit_cancel_requested = True
        if self._fit_timer.IsRunning():
            self._fit_timer.Stop()
        if self._fit_progress is not None:
            try:
                self._fit_progress.Destroy()
            except Exception:
                pass
            self._fit_progress = None
        self._fit_running = False
        self._set_fit_actions_enabled(True)

    def _on_fit_timer(self, event):
        if self._fit_progress is not None:
            try:
                message = "Running global fit..."
                if getattr(self, "_fit_rows_after_global", False):
                    message = "Running global fit and preparing row-fit cache..."
                keep_going = self._fit_progress.Pulse(message)
                if isinstance(keep_going, tuple):
                    keep_going = keep_going[0]
                if keep_going is False:
                    self._request_global_fit_abort()
            except Exception:
                pass

    def _clone_engine_for_fit(self) -> analysis.MapFittingEngine:
        fit_engine = analysis.MapFittingEngine()
        for idx, ds in enumerate(self.engine.datasets):
            if ds["x"] is not None and ds["ang"] is not None and ds["z"] is not None:
                fit_engine.set_data(
                    idx,
                    ds["x"],
                    ds["ang"],
                    ds["z"],
                    ds.get("nickname", "Run"),
                    background_ang=ds.get("background_ang"),
                )
            fit_engine.datasets[idx]["config"] = ds.get("config", fit_engine.datasets[idx]["config"])
            fit_engine.datasets[idx]["label"] = ds.get("label", fit_engine.datasets[idx]["label"])
        fit_engine.from_dict(copy.deepcopy(self.engine.to_dict()))
        return fit_engine

    def _finish_global_fit(
        self,
        token: int,
        success: bool,
        msg: str,
        fitted_state: Optional[Dict[str, Any]] = None,
        global_fit_results: Optional[List[Dict[str, Any]]] = None,
        row_fit_results: Optional[List[Dict[str, Any]]] = None,
        row_fit_msg: str = "",
    ) -> None:
        if token != self._fit_token:
            return
        if self._fit_cancel_requested:
            self._fit_running = False
            return
        if self._fit_timer.IsRunning():
            self._fit_timer.Stop()
        if self._fit_progress is not None:
            try:
                self._fit_progress.Destroy()
            except Exception:
                pass
            self._fit_progress = None
        self._fit_running = False
        self._set_fit_actions_enabled(True)
        if success:
            if fitted_state is not None:
                self.engine.from_dict(fitted_state)
            cache_count = self._persist_row_fit_caches(row_fit_results) if row_fit_results else 0
            self.load_grid()
            self.update_plots()
            if row_fit_results:
                wx.MessageBox(
                    f"Global Fit Complete\nStaged {cache_count} compact row-fit result(s).",
                    "Success",
                )
            elif row_fit_msg:
                wx.MessageBox(
                    f"Global Fit Complete\n\nRow-by-row result was not staged:\n{row_fit_msg}",
                    "Success",
                    wx.OK | wx.ICON_WARNING,
                )
            else:
                wx.MessageBox("Global Fit Complete", "Success")
        else:
            self._show_error_details("Fit Failed", msg)

    def on_fit(self, e):
        if self._fit_running:
            return
        self.save_grid(self.peak_list.GetSelection())
        self._fit_token += 1
        fit_token = self._fit_token
        self._fit_cancel_requested = False
        self._fit_running = True
        self._fit_rows_after_global = bool(self.chk_fit_rows_after_global.GetValue())
        self._set_fit_actions_enabled(False)
        progress_message = "Running global fit...\nPress Esc to abort."
        if self._fit_rows_after_global:
            progress_message = "Running global fit and preparing row-fit cache...\nPress Esc to abort."
        self._fit_progress = wx.ProgressDialog(
            "Fitting",
            progress_message,
            parent=self,
            style=wx.PD_APP_MODAL | wx.PD_ELAPSED_TIME | wx.PD_CAN_ABORT,
        )
        self._fit_progress.Bind(wx.EVT_CHAR_HOOK, self._on_char_hook)
        self._fit_progress.Pulse("Starting global fit...")
        self._fit_timer.Start(150)
        fit_engine = self._clone_engine_for_fit()

        def worker():
            try:
                success, msg = fit_engine.run_optimization()
                fitted_state = copy.deepcopy(fit_engine.to_dict()) if success else None
                global_fit_results = None
                row_fit_results = None
                row_fit_msg = ""
                if success:
                    global_fit_results = [
                        {"dataset_index": ds_idx, "matrix": fit_engine.reconstruct(ds_idx)}
                        for ds_idx in fit_engine.active_dataset_indices()
                    ]
                    if self._fit_rows_after_global:
                        row_success, row_fit_msg, row_fit_results = fit_engine.validate_row_by_row()
                        if not row_success:
                            row_fit_results = None
            except Exception as exc:
                success, msg = False, f"{exc}\n\n{traceback.format_exc()}"
                fitted_state = None
                global_fit_results = None
                row_fit_results = None
                row_fit_msg = ""
            wx.CallAfter(
                self._finish_global_fit,
                fit_token,
                success,
                msg,
                fitted_state,
                global_fit_results,
                row_fit_results,
                row_fit_msg,
            )

        threading.Thread(target=worker, daemon=True).start()

    def on_validate(self, e):
        self.save_grid(self.peak_list.GetSelection())
        validation_engine = self._clone_engine_for_fit()
        settings = RowFitSettingsDialog(self, validation_engine, validation_engine.row_fit_config)
        try:
            if settings.ShowModal() != wx.ID_OK: return
            row_fit_config = settings.get_config()
        finally: settings.Destroy()
        validation_engine.set_row_fit_config(row_fit_config)
        self.engine.set_row_fit_config(row_fit_config)
        prog = wx.ProgressDialog("Validating", "Running Row-by-Row Fit...", parent=self, style=wx.PD_APP_MODAL|wx.PD_ELAPSED_TIME)
        prog.Pulse()
        success, msg, results = validation_engine.validate_row_by_row(row_fit_config)
        prog.Destroy()

        if success and results:
            self._persist_row_fit_caches(results)
            global_matrices = [{"dataset_index": ds_idx, "matrix": validation_engine.reconstruct(ds_idx)} for ds_idx in validation_engine.active_dataset_indices()]
            vf = ValidationFrame(
                results,
                self,
                engine=validation_engine,
                global_matrices=global_matrices,
                row_fit_config=row_fit_config,
                initial_selection=self.selected_slice,
            )
            try:
                vf.ShowModal()
            finally:
                vf.Destroy()
        else:
            self._show_error_details("Validation Failed", msg)

    def on_export_text(self, e):
        text = self.engine.export_parameters_text()
        dlg = wx.lib.dialogs.ScrolledMessageDialog(self, text, "Exported Parameters")
        dlg.ShowModal()
        dlg.Destroy()

    def on_import_text(self, e):
        with wx.FileDialog(self, "Import Fitting Parameters", wildcard="TXT files (*.txt)|*.txt",
                          style=wx.FD_OPEN | wx.FD_FILE_MUST_EXIST) as file_dlg:
            if file_dlg.ShowModal() == wx.ID_CANCEL: return
            path = file_dlg.GetPath()
            try:
                with open(path, "r") as f:
                    text = f.read()
                self.engine.import_parameters_text(text)
                self.refresh_ui()
                self.update_plots()
                wx.MessageBox("Parameters imported successfully.", "Success")
            except Exception as ex:
                wx.MessageBox(f"Import failed: {ex}", "Error", wx.ICON_ERROR)

    def on_export_diagnostics(self, e):
        self.save_grid(self.peak_list.GetSelection())
        with wx.DirDialog(self, "Choose Output Directory for Fit Diagnostic CSVs") as dlg:
            if dlg.ShowModal() != wx.ID_OK:
                return
            out_dir = dlg.GetPath()
        count = 0
        for table in self.engine.global_fit_trace_tables():
            label = str(table["label"]).replace(" ", "_").replace("(", "").replace(")", "")
            path = os.path.join(out_dir, f"GlobalFit_Params_{label}.csv")
            with open(path, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(table["headers"])
                writer.writerows(table["rows"])
            count += 1
        success, _msg, results = self.engine.validate_row_by_row()
        if success and results:
            self._persist_row_fit_caches(results)
            for idx, res in enumerate(results):
                label = str(res.get("label", f"dataset_{idx+1}")).replace(" ", "_").replace("(", "").replace(")", "")
                path = os.path.join(out_dir, f"RowFit_Areas_{label}.csv")
                with open(path, "w", newline="", encoding="utf-8") as f:
                    writer = csv.writer(f)
                    writer.writerow(res.get("headers") or [])
                    writer.writerows(res.get("rows_params") or res.get("params") or [])
                count += 1
        wx.MessageBox(f"Exported {count} diagnostic CSV file(s) to:\n{out_dir}", "Export Complete", wx.OK | wx.ICON_INFORMATION)

    def on_click(self, e):
        if self.toolbar.mode != '': return
        clicked_ax = e.inaxes
        ds_idx = -1
        if clicked_ax in [self.ax_raw1, self.ax_rec1]: ds_idx = 0
        elif clicked_ax in [self.ax_raw2, self.ax_rec2]: ds_idx = 1

        if ds_idx != -1:
            ds = self.engine.datasets[ds_idx]
            if ds["x"] is not None and ds["ang"] is not None and e.xdata is not None and e.ydata is not None:
                ix = int(np.nanargmin(np.abs(ds["x"] - e.xdata)))
                iy = int(np.nanargmin(np.abs(ds["ang"] - e.ydata)))
                self.selected_slice = {
                    "dataset_index": ds_idx,
                    "x_value": float(ds["x"][ix]),
                    "angle_value": float(ds["ang"][iy]),
                }
                self.update_plots()

    def update_plots(self):
        if self.engine.datasets[0]["z"] is None: return

        saved_xlim = self.ax_raw1.get_xlim()
        saved_ylim = self.ax_raw1.get_ylim()

        def plot_set(ds_idx, ax_raw, ax_rec, ax_spec, ax_ang):
            ds = self.engine.datasets[ds_idx]
            if ds["z"] is None: return

            z_raw = ds["z"]
            z_rec = self.engine.reconstruct(ds_idx)
            x = ds["x"]
            ang = ds["ang"]
            slice_ix, slice_iy = self._slice_indices(ds_idx)

            try:
                vmin, vmax = float(self.txt_vmin.GetValue()), float(self.txt_vmax.GetValue())
            except:
                vmin, vmax = np.nanpercentile(z_raw, [1, 99])

            ax_raw.clear(); ax_rec.clear()
            ax_raw.pcolormesh(x, ang, z_raw, vmin=vmin, vmax=vmax, cmap='inferno', shading='auto')
            ax_raw.set_title(f"{ds['label']} Raw")
            ax_raw.set_ylabel("Angle")

            ax_rec.pcolormesh(x, ang, z_rec, vmin=vmin, vmax=vmax, cmap='inferno', shading='auto')
            ax_rec.set_title("Reconstruction")
            ax_rec.set_yticklabels([])

            for ax in [ax_raw, ax_rec]:
                if self.engine.x_min_limit > x.min():
                    ax.axvspan(x.min(), self.engine.x_min_limit, color='gray', alpha=0.5)
                if self.engine.x_max_limit < x.max():
                    ax.axvspan(self.engine.x_max_limit, x.max(), color='gray', alpha=0.5)

                if 0 <= slice_iy < len(ang):
                    ax.axhline(ang[slice_iy], color='w', ls='--', alpha=0.5)
                if 0 <= slice_ix < len(x):
                    ax.axvline(x[slice_ix], color='w', ls='--', alpha=0.5)
                trans = ax.get_xaxis_transform()
                for peak in self.engine.peaks:
                    x0 = peak.get("spec_params", {}).get("x0", [None])[0]
                    try:
                        x0 = float(x0)
                    except Exception:
                        continue
                    if np.isfinite(x0):
                        ax.plot(x0, 0, marker='^', color='red', markersize=6, transform=trans, clip_on=False)
                        ax.text(x0, -0.05, peak.get("name", ""), color='red', ha='center', va='top', transform=trans, clip_on=False, fontsize=7)

            ax_spec.clear()
            if 0 <= slice_iy < len(ang):
                cur_ang = ang[slice_iy]
                ax_spec.plot(x, z_raw[slice_iy,:], 'k', alpha=0.5)
                ax_spec.plot(x, z_rec[slice_iy,:], 'r')
                ax_spec.set_title(f"Spec @ {cur_ang:.1f}°")
                if self.engine.x_min_limit > x.min():
                    ax_spec.axvspan(x.min(), self.engine.x_min_limit, color='gray', alpha=0.2)
                if self.engine.x_max_limit < x.max():
                    ax_spec.axvspan(self.engine.x_max_limit, x.max(), color='gray', alpha=0.2)

            ax_ang.clear()
            if 0 <= slice_ix < len(x):
                cur_shift = x[slice_ix]
                ax_ang.plot(ang, z_raw[:,slice_ix], 'k', alpha=0.5)
                ax_ang.plot(ang, z_rec[:,slice_ix], 'r')
                ax_ang.set_title(f"Ang @ {cur_shift:.1f} cm-1")

        plot_set(0, self.ax_raw1, self.ax_rec1, self.ax_spec1, self.ax_ang1)
        plot_set(1, self.ax_raw2, self.ax_rec2, self.ax_spec2, self.ax_ang2)

        if saved_xlim != (0.0, 1.0) and saved_ylim != (0.0, 1.0):
            self.ax_raw1.set_xlim(saved_xlim)
            self.ax_raw1.set_ylim(saved_ylim)
            self.ax_ang1.set_xlim(saved_ylim)
            self.ax_ang2.set_xlim(saved_ylim)

        self.canvas.draw()

    @staticmethod
    def _clean_export_metadata(source_run: Run) -> Dict[str, Any]:
        metadata = copy.deepcopy(source_run.metadata or {})
        for key in (
            "fit_state", "map_fit_state", "fit_parameters_text", "fit_params_source_run_ids",
            "row_fit_cache_run_id", "row_fit_cache_hash", "row_fit_cache_run_ids", "row_fit_cache_hashes",
            "global_fit_cache_run_id", "global_fit_cache_hash", "global_fit_cache_run_ids", "global_fit_cache_hashes",
            "active_fit_params_run_id", "fit_params_run_ids",
        ):
            metadata.pop(key, None)
        return metadata

    def on_export_to_experiment(self, e):
        """Stage explicitly requested Run objects until the main dialog commits."""
        self.save_grid(self.peak_list.GetSelection())

        peak_names = [p["name"] for p in self.engine.peaks]
        with ExportOptionsDialog(self, peak_names) as dlg:
            if dlg.ShowModal() != wx.ID_OK:
                return
            options = dlg.get_options()

        self.result_runs = []

        # 1. Full Reconstruction
        if options["full_rec"]:
            for i in range(2):
                orig_run = self.run1 if i == 0 else self.run2
                if orig_run is None or not orig_run.is_2d: continue
                total_rec = self.engine.reconstruct(i)

                new_nickname = f"{orig_run.nickname}_TotalFit"
                new_metadata = self._clean_export_metadata(orig_run)
                new_metadata.pop("angle_rotation", None)
                new_metadata.pop("angle_rotation_summary", None)
                new_metadata["nickname"] = new_nickname
                new_metadata["is_reconstruction"] = True
                new_metadata["fit_component"] = "Total"
                new_metadata["source_run_id"] = orig_run.id

                new_run = Run(
                    id=new_run_id(prefix="fit"),
                    source_path=orig_run.source_path,
                    source_mtime=orig_run.source_mtime,
                    wl_nm=None,
                    shift_cm1=orig_run.shift_cm1,
                    energy_eV=orig_run.energy_eV,
                    intensity=None,
                    intensity_2d=total_rec,
                    angle_values=self.engine.datasets[i]["ang"],
                    intensity_unit=orig_run.intensity_unit,
                    angle_unit=orig_run.angle_unit,
                    metadata=new_metadata,
                    run_type=RunType.RUN_2D,
                    raw_table=None
                )
                self.result_runs.append(new_run)

        # 2. Polar Areas
        if options["polar_areas"]:
            for i in range(2):
                orig_run = self.run1 if i == 0 else self.run2
                if orig_run is None or not orig_run.is_2d: continue
                areas = self.engine.get_peak_polar_areas(i)
                if areas is None: continue

                new_nickname = f"{orig_run.nickname}_PolarAreas"
                new_run = Run(
                    id=new_run_id(prefix="polar"),
                    source_path="",
                    shift_cm1=np.arange(areas.shape[1]), # Peak index
                    angle_values=self.engine.datasets[i]["ang"],
                    intensity_2d=areas,
                    metadata={
                        "nickname": new_nickname,
                        "source_run_id": orig_run.id,
                        "peak_names": peak_names
                    },
                    run_type=RunType.RUN_2D
                )
                self.result_runs.append(new_run)

        # 3. Individual Peak Reconstructions
        recons = self.engine.get_peak_reconstructions()
        for rec in recons:
            ds_idx = rec["dataset_idx"]
            orig_run = self.run1 if ds_idx == 0 else self.run2
            if not (orig_run and orig_run.is_2d): continue
            name = rec["name"]

            # Check if this peak was selected
            if name == "Background":
                if not options.get("background"):
                    continue
            else:
                try:
                    peak_idx = peak_names.index(name)
                    if not options["peaks"][peak_idx]:
                        continue
                except ValueError:
                    continue

            matrix = rec["matrix"]
            new_nickname = f"{orig_run.nickname}_{name}_fit" if name != "Background" else f"{orig_run.nickname}_BackgroundFit"

            new_metadata = self._clean_export_metadata(orig_run)
            new_metadata.pop("angle_rotation", None)
            new_metadata.pop("angle_rotation_summary", None)
            new_metadata["nickname"] = new_nickname
            new_metadata["is_reconstruction"] = True
            new_metadata["fit_component"] = name
            new_metadata["source_run_id"] = orig_run.id

            new_run = Run(
                id=new_run_id(prefix="fit"),
                source_path=orig_run.source_path,
                source_mtime=orig_run.source_mtime,
                wl_nm=None,
                shift_cm1=orig_run.shift_cm1,
                energy_eV=orig_run.energy_eV,
                intensity=None,
                intensity_2d=matrix,
                angle_values=self.engine.datasets[ds_idx]["ang"],
                intensity_unit=orig_run.intensity_unit,
                angle_unit=orig_run.angle_unit,
                metadata=new_metadata,
                run_type=RunType.RUN_2D,
                raw_table=None
            )
            self.result_runs.append(new_run)

        if options.get("bg_subtracted"):
            for ds_idx in self.engine.active_dataset_indices():
                orig_run = self.run1 if ds_idx == 0 else self.run2
                if not (orig_run and orig_run.is_2d):
                    continue
                matrix = self.engine.get_background_subtracted(ds_idx)
                if matrix is None:
                    continue
                new_metadata = self._clean_export_metadata(orig_run)
                new_metadata.pop("angle_rotation", None)
                new_metadata.pop("angle_rotation_summary", None)
                new_metadata["nickname"] = f"{orig_run.nickname}_BGSubtracted"
                new_metadata["fit_component"] = "BackgroundSubtracted"
                new_metadata["source_run_id"] = orig_run.id
                new_run = Run(
                    id=new_run_id(prefix="bgsub"),
                    source_path=orig_run.source_path,
                    source_mtime=orig_run.source_mtime,
                    wl_nm=None,
                    shift_cm1=orig_run.shift_cm1,
                    energy_eV=orig_run.energy_eV,
                    intensity=None,
                    intensity_2d=matrix,
                    angle_values=self.engine.datasets[ds_idx]["ang"],
                    intensity_unit=orig_run.intensity_unit,
                    angle_unit=orig_run.angle_unit,
                    metadata=new_metadata,
                    run_type=RunType.RUN_2D,
                    raw_table=None,
                )
                self.result_runs.append(new_run)

        # Count how many for each run
        r1_count = len([r for r in self.result_runs if r.metadata.get("source_run_id") == self.run1.id])
        lines = [f"Created {len(self.result_runs)} runs total.", f"- {self.run1.nickname}: {r1_count} runs"]
        if self.run2 is not None:
            r2_count = len([r for r in self.result_runs if r.metadata.get("source_run_id") == self.run2.id])
            lines.append(f"- {self.run2.nickname}: {r2_count} runs")
        msg = "\n".join(lines)
        wx.MessageBox(
            msg + "\n\nThese runs are staged and will be added only when you press OK or Save as new params.",
            "Export Staged",
        )
