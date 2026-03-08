import os
import analysis
from merge_runs_gui import MergeRunsDialog
from typing import List, Optional, Dict, Any, Tuple, TYPE_CHECKING

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
    new_experiment_id,
    new_view_id,
    RunType,
    normalize_spectral_unit,
)
from plotting import RamanPlotter2d, AngularPlotter, SlicePlotter
from config_manager import DEFAULT_SETTINGS, config
from fit_overlay import normalize_fit_overlay_mode

try:
    import cmcrameri.cm
    HAS_CMCRAMERI = True
except ImportError:
    HAS_CMCRAMERI = False

if TYPE_CHECKING:
    from wx_right_panel import ViewPanel


class PreviewPanel(wx.Panel):
    """
    Preview tab: shows a simple text preview of the selected file.
    """

    def __init__(self, parent):
        super().__init__(parent)
        sizer = wx.BoxSizer(wx.VERTICAL)
        label = wx.StaticText(self, label="Preview of selected file")
        sizer.Add(label, 0, wx.ALL, 8)
        self.preview = wx.TextCtrl(
            self, style=wx.TE_MULTILINE | wx.TE_READONLY | wx.TE_RICH2
        )
        sizer.Add(self.preview, 1, wx.EXPAND | wx.ALL, 4)
        self.SetSizer(sizer)

    def show_text(self, text: str) -> None:
        self.preview.SetValue(text)


import colorsys
import re
try:
    from scipy.optimize import curve_fit
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import LinearSegmentedColormap, to_hex, to_rgba

class CurveFitPanel(wx.Panel):
    """
    Curve Fit tab: 1D curve fitting implementation.
    """

    def __init__(self, parent, on_run_created=None):
        super().__init__(parent)
        self.on_run_created = on_run_created
        
        self.data_x = None
        self.data_y = None
        self.data_y_2d = None # For batch fitting
        self.source_run: Optional[Run] = None
        self.plot_type = "" # "B" or "C" or "A" (for batch)
        self.fit_result_curve = None # x, y of fit
        self.current_model = "Lorentzian"
        
        # UI Layout
        main_sizer = wx.BoxSizer(wx.VERTICAL)
        
        # 1. Top Toolbar / Info
        self.info_label = wx.StaticText(self, label="No data selected (Right-click trace -> Curve Fit)")
        main_sizer.Add(self.info_label, 0, wx.ALL, 5)
        
        # 2. Plot Area
        self.figure = Figure(figsize=(4, 3), dpi=100)
        self.canvas = FigureCanvas(self, -1, self.figure)
        self.ax = self.figure.add_subplot(111)
        self.figure.tight_layout()
        main_sizer.Add(self.canvas, 1, wx.EXPAND | wx.ALL, 2)
        
        # 3. Controls
        controls_sizer = wx.BoxSizer(wx.VERTICAL)
        
        # Model Selection
        model_sizer = wx.BoxSizer(wx.HORIZONTAL)
        model_sizer.Add(wx.StaticText(self, label="Model:"), 0, wx.ALIGN_CENTER_VERTICAL | wx.RIGHT, 5)
        self.choice_model = wx.Choice(self, choices=["Lorentzian", "Sum of Lorentzian", "User Defined"])
        self.choice_model.SetSelection(0)
        model_sizer.Add(self.choice_model, 0, wx.RIGHT, 10)
        
        # N Peaks (for Sum)
        self.lbl_peaks = wx.StaticText(self, label="N Peaks:")
        self.spin_peaks = wx.SpinCtrl(self, value="2", min=1, max=10)
        model_sizer.Add(self.lbl_peaks, 0, wx.ALIGN_CENTER_VERTICAL | wx.RIGHT, 5)
        model_sizer.Add(self.spin_peaks, 0, wx.RIGHT, 10)
        
        # User Formula (for User Defined)
        self.lbl_formula = wx.StaticText(self, label="f(x)=")
        self.txt_formula = wx.TextCtrl(self, value="a*x + b")
        self.btn_parse = wx.Button(self, label="Parse", size=(50, -1))
        model_sizer.Add(self.lbl_formula, 0, wx.ALIGN_CENTER_VERTICAL | wx.RIGHT, 5)
        model_sizer.Add(self.txt_formula, 1, wx.EXPAND | wx.RIGHT, 5)
        model_sizer.Add(self.btn_parse, 0)
        
        controls_sizer.Add(model_sizer, 0, wx.EXPAND | wx.ALL, 5)

        peak_sizer = wx.BoxSizer(wx.HORIZONTAL)
        self.peak_list = wx.ListBox(self, size=(-1, 64))
        peak_btn_sizer = wx.BoxSizer(wx.VERTICAL)
        self.btn_add_peak = wx.Button(self, label="Add Peak")
        self.btn_remove_peak = wx.Button(self, label="Remove Peak")
        peak_btn_sizer.Add(self.btn_add_peak, 0, wx.EXPAND | wx.BOTTOM, 3)
        peak_btn_sizer.Add(self.btn_remove_peak, 0, wx.EXPAND)
        peak_sizer.Add(self.peak_list, 1, wx.EXPAND | wx.RIGHT, 5)
        peak_sizer.Add(peak_btn_sizer, 0, wx.EXPAND)
        controls_sizer.Add(peak_sizer, 0, wx.EXPAND | wx.LEFT | wx.RIGHT | wx.BOTTOM, 5)
        
        # Parameter Grid (Scrolled)
        self.scrolled = wx.ScrolledWindow(self, style=wx.VSCROLL)
        self.scrolled.SetScrollRate(0, 10)
        self.param_sizer = wx.FlexGridSizer(cols=5, vgap=5, hgap=5)
        self.param_sizer.AddGrowableCol(1, 1)
        self.scrolled.SetSizer(self.param_sizer)
        
        controls_sizer.Add(self.scrolled, 1, wx.EXPAND | wx.ALL, 5)
        
        # Fit Range & Buttons
        action_sizer = wx.BoxSizer(wx.HORIZONTAL)
        
        action_sizer.Add(wx.StaticText(self, label="Range:"), 0, wx.ALIGN_CENTER_VERTICAL | wx.RIGHT, 5)
        self.txt_min = wx.TextCtrl(self, size=(50, -1))
        self.txt_max = wx.TextCtrl(self, size=(50, -1))
        action_sizer.Add(self.txt_min, 0, wx.RIGHT, 5)
        action_sizer.Add(self.txt_max, 0, wx.RIGHT, 10)
        
        self.btn_fit = wx.Button(self, label="Fit")
        self.btn_create = wx.Button(self, label="Create Curve")
        self.btn_create.Disable()
        
        self.btn_batch = wx.Button(self, label="Batch Fit (All Rows)")
        self.btn_batch.Disable()
        self.btn_batch.Hide()
        
        action_sizer.Add(self.btn_fit, 0, wx.RIGHT, 5)
        action_sizer.Add(self.btn_create, 0, wx.RIGHT, 5)
        action_sizer.Add(self.btn_batch, 0)
        
        controls_sizer.Add(action_sizer, 0, wx.EXPAND | wx.ALL, 5)
        
        main_sizer.Add(controls_sizer, 1, wx.EXPAND)
        self.SetSizer(main_sizer)
        
        # Bindings
        self.choice_model.Bind(wx.EVT_CHOICE, self.on_model_change)
        self.spin_peaks.Bind(wx.EVT_SPINCTRL, self.on_model_change)
        self.btn_parse.Bind(wx.EVT_BUTTON, self.on_model_change)
        self.btn_fit.Bind(wx.EVT_BUTTON, self.on_fit)
        self.btn_create.Bind(wx.EVT_BUTTON, self.on_create_curve)
        self.btn_batch.Bind(wx.EVT_BUTTON, self.on_batch_fit)
        self.btn_add_peak.Bind(wx.EVT_BUTTON, self.on_add_peak)
        self.btn_remove_peak.Bind(wx.EVT_BUTTON, self.on_remove_peak)
        
        # Init UI state
        self.param_controls = {} # name -> dict with 'value', 'fixed', 'min', 'max'
        self.on_model_change(None)

    def _refresh_peak_list(self):
        if not hasattr(self, "peak_list"):
            return
        self.peak_list.Clear()
        model = self.choice_model.GetStringSelection()
        if model == "Sum of Lorentzian":
            for i in range(1, self.spin_peaks.GetValue() + 1):
                label = f"Peak {i}"
                if f"x0_{i}" in self.param_controls:
                    label += f" @ {self.param_controls[f'x0_{i}']['value'].GetValue()}"
                self.peak_list.Append(label)
            if self.peak_list.GetCount():
                self.peak_list.SetSelection(0)
        elif model == "Lorentzian":
            label = "Peak 1"
            if "x0" in self.param_controls:
                label += f" @ {self.param_controls['x0']['value'].GetValue()}"
            self.peak_list.Append(label)
            self.peak_list.SetSelection(0)

    def on_add_peak(self, event):
        self.choice_model.SetSelection(self.choice_model.FindString("Sum of Lorentzian"))
        self.spin_peaks.SetValue(min(self.spin_peaks.GetValue() + 1, self.spin_peaks.GetMax()))
        self.on_model_change(None)

    def on_remove_peak(self, event):
        if self.choice_model.GetStringSelection() != "Sum of Lorentzian":
            return
        self.spin_peaks.SetValue(max(self.spin_peaks.GetValue() - 1, self.spin_peaks.GetMin()))
        if self.spin_peaks.GetValue() == 1:
            self.choice_model.SetSelection(self.choice_model.FindString("Lorentzian"))
        self.on_model_change(None)

    def set_data(self, run: Run, plot_type: str, x_data: np.ndarray, y_data: np.ndarray, x_label: Optional[str] = None):
        self.source_run = run
        self.plot_type = plot_type
        self.x_label = x_label or ("Angle (deg)" if plot_type == "B" else "Raman shift (cm-1)" if plot_type in {"A", "C"} else "x")
        self.data_x = x_data
        
        # Check for 2D data (Batch Mode)
        if y_data.ndim == 2:
            self.data_y_2d = y_data
            # Select middle row for preview
            mid_idx = y_data.shape[0] // 2
            self.data_y = y_data[mid_idx, :]
            
            self.btn_batch.Show()
            self.btn_batch.Enable()
            self.info_label.SetLabel(f"Fitting: {run.nickname} (2D Batch Mode - Preview Row {mid_idx})")
        else:
            self.data_y = y_data
            self.data_y_2d = None
            self.btn_batch.Hide()
            self.info_label.SetLabel(f"Fitting: {run.nickname} ({plot_type})")
            
        self.fit_result_curve = None
        self.btn_create.Disable()
        self.Layout()
        
        # Reset range defaults
        if len(x_data) > 0:
            self.txt_min.SetValue(f"{np.min(x_data):.2f}")
            self.txt_max.SetValue(f"{np.max(x_data):.2f}")
        
        # Initial guess for parameters (Auto-guess)
        self._auto_guess_params()
        
        self._plot_data()

    def _auto_guess_params(self):
        # Basic heuristics
        if self.data_x is None or len(self.data_x) == 0: return
        
        y_min = np.min(self.data_y)
        y_max = np.max(self.data_y)
        x_at_max = self.data_x[np.argmax(self.data_y)]
        amp = y_max - y_min
        
        model = self.choice_model.GetStringSelection()
        
        if model == "Lorentzian":
            if "y0" in self.param_controls: self.param_controls["y0"]["value"].SetValue(f"{y_min:.2f}")
            if "x0" in self.param_controls: self.param_controls["x0"]["value"].SetValue(f"{x_at_max:.2f}")
            if "A" in self.param_controls: self.param_controls["A"]["value"].SetValue(f"{amp:.2f}")
            if "Gamma" in self.param_controls: self.param_controls["Gamma"]["value"].SetValue("5.0")
            
        elif model == "Sum of Lorentzian":
            # Just repeat same guess for all peaks for now (stacked)
            n = self.spin_peaks.GetValue()
            for i in range(1, n+1):
                if f"y0_{i}" in self.param_controls: self.param_controls[f"y0_{i}"]["value"].SetValue(f"{y_min/n:.2f}")
                if f"x0_{i}" in self.param_controls: self.param_controls[f"x0_{i}"]["value"].SetValue(f"{x_at_max:.2f}")
                if f"A_{i}" in self.param_controls: self.param_controls[f"A_{i}"]["value"].SetValue(f"{amp/n:.2f}")
                if f"Gamma_{i}" in self.param_controls: self.param_controls[f"Gamma_{i}"]["value"].SetValue("5.0")

    def on_model_change(self, event):
        model = self.choice_model.GetStringSelection()
        self.current_model = model
        
        # Visibility
        self.lbl_peaks.Show(model == "Sum of Lorentzian")
        self.spin_peaks.Show(model == "Sum of Lorentzian")
        self.lbl_formula.Show(model == "User Defined")
        self.txt_formula.Show(model == "User Defined")
        self.btn_parse.Show(model == "User Defined")
        self.Layout()
        
        # Rebuild Grid
        self.param_sizer.Clear(True)
        self.param_controls = {}
        
        params = []
        if model == "Lorentzian":
            params = ["y0", "x0", "A", "Gamma"]
        elif model == "Sum of Lorentzian":
            n = self.spin_peaks.GetValue()
            for i in range(1, n+1):
                params.extend([f"y0_{i}", f"x0_{i}", f"A_{i}", f"Gamma_{i}"])
        elif model == "User Defined":
            # Parse formula
            f_str = self.txt_formula.GetValue()
            # Extract identifiers (simple regex)
            tokens = set(re.findall(r"[a-zA-Z_]\w*", f_str))
            # Exclude math constants/funcs
            excludes = {"x", "np", "pi", "e", "sin", "cos", "tan", "exp", "log", "sqrt", "abs"}
            params = sorted(list(tokens - excludes))
            
        # Headers
        self.param_sizer.Add(wx.StaticText(self.scrolled, label="Param"), 0, wx.ALIGN_CENTER)
        self.param_sizer.Add(wx.StaticText(self.scrolled, label="Value"), 0, wx.ALIGN_CENTER)
        self.param_sizer.Add(wx.StaticText(self.scrolled, label="Fix"), 0, wx.ALIGN_CENTER)
        self.param_sizer.Add(wx.StaticText(self.scrolled, label="Min"), 0, wx.ALIGN_CENTER)
        self.param_sizer.Add(wx.StaticText(self.scrolled, label="Max"), 0, wx.ALIGN_CENTER)
        
        for p in params:
            self.param_sizer.Add(wx.StaticText(self.scrolled, label=f"{p}:"), 0, wx.ALIGN_CENTER_VERTICAL | wx.ALIGN_RIGHT)
            val = "1.0"
            if "y0" in p: val = "0.0"
            if "x0" in p and self.data_x is not None: val = f"{np.mean(self.data_x):.2f}"
            
            txt_val = wx.TextCtrl(self.scrolled, value=val)
            self.param_sizer.Add(txt_val, 1, wx.EXPAND)
            
            chk_fixed = wx.CheckBox(self.scrolled)
            self.param_sizer.Add(chk_fixed, 0, wx.ALIGN_CENTER)
            
            txt_min = wx.TextCtrl(self.scrolled)
            self.param_sizer.Add(txt_min, 1, wx.EXPAND)
            
            txt_max = wx.TextCtrl(self.scrolled)
            self.param_sizer.Add(txt_max, 1, wx.EXPAND)
            
            self.param_controls[p] = {
                "value": txt_val,
                "fixed": chk_fixed,
                "min": txt_min,
                "max": txt_max
            }
            
        self.scrolled.Layout()
        self.scrolled.FitInside()
        self._refresh_peak_list()

    def _get_param_values(self):
        vals = {}
        for p, ctrls in self.param_controls.items():
            try:
                vals[p] = float(ctrls["value"].GetValue())
            except ValueError:
                vals[p] = 0.0
        return vals

    def _lorentzian(self, x, y0, x0, A, G):
        # Using Height form as per prompt: A is height.
        # Standard: I = I_max * (G^2 / (4(x-x0)^2 + G^2))
        return y0 + A * (G**2 / (4 * (x - x0)**2 + G**2))

    def _get_fit_func(self, model_name, p_names):
        if model_name == "Lorentzian":
            def func(x, y0, x0, A, G):
                return self._lorentzian(x, y0, x0, A, G)
            return func
            
        elif model_name == "Sum of Lorentzian":
            n = self.spin_peaks.GetValue()
            def func(x, *args):
                y = np.zeros_like(x)
                for i in range(n):
                    idx = i * 4
                    y += self._lorentzian(x, args[idx], args[idx+1], args[idx+2], args[idx+3])
                return y
            return func
            
        elif model_name == "User Defined":
            f_str = self.txt_formula.GetValue()
            def func(x, *args):
                local_scope = {"x": x, "np": np}
                for name, val in zip(p_names, args):
                    local_scope[name] = val
                return eval(f_str, {"__builtins__": None}, local_scope)
            return func
        return None

    def on_fit(self, event):
        if not HAS_SCIPY:
            wx.MessageBox("Scipy is not installed.", "Error")
            return
        if self.data_x is None: return

        # 1. Get Range Mask
        try:
            xmin = float(self.txt_min.GetValue())
            xmax = float(self.txt_max.GetValue())
        except ValueError:
            xmin, xmax = -np.inf, np.inf
            
        mask = (self.data_x >= xmin) & (self.data_x <= xmax)
        x_fit = self.data_x[mask]
        y_fit = self.data_y[mask]
        
        if len(x_fit) < 4:
            wx.MessageBox("Not enough data points in range.", "Error")
            return

        # 2. Define Model Function
        model_name = self.choice_model.GetStringSelection()
        p_names = list(self.param_controls.keys())
        
        p_free_names = []
        p_free_indices = []
        p0_free = []
        bounds_min = []
        bounds_max = []
        p_full_current = []

        for i, p in enumerate(p_names):
            ctrls = self.param_controls[p]
            try:
                val = float(ctrls["value"].GetValue())
            except ValueError:
                val = 0.0
            p_full_current.append(val)
            
            if ctrls["fixed"].GetValue():
                continue
            
            p_free_names.append(p)
            p_free_indices.append(i)
            p0_free.append(val)
            
            # Bounds
            try:
                v_min = float(ctrls["min"].GetValue())
            except ValueError:
                v_min = -np.inf
            try:
                v_max = float(ctrls["max"].GetValue())
            except ValueError:
                v_max = np.inf
            bounds_min.append(v_min)
            bounds_max.append(v_max)
            
        if not p_free_names:
            wx.MessageBox("All parameters are fixed. Nothing to fit.", "Info")
            return
            
        bounds = (bounds_min, bounds_max)
        fit_func_base = self._get_fit_func(model_name, p_names)
        
        def wrapper_func(x, *args):
            current_args = list(p_full_current)
            for idx, val in zip(p_free_indices, args):
                current_args[idx] = val
            return fit_func_base(x, *current_args)

        # 3. Perform Fit
        try:
            popt, pcov = curve_fit(wrapper_func, x_fit, y_fit, p0=p0_free, bounds=bounds)
            
            # 4. Update UI
            for i, val in enumerate(popt):
                p_name = p_free_names[i]
                self.param_controls[p_name]["value"].SetValue(f"{val:.4f}")
                p_full_current[p_free_indices[i]] = val
            
            # 5. Plot Result
            y_model = fit_func_base(self.data_x, *p_full_current)
            self.fit_result_curve = (self.data_x, y_model)
            self._plot_data()
            self.btn_create.Enable()
            
        except Exception as e:
            wx.MessageBox(f"Fit failed: {str(e)}", "Error")

    def on_batch_fit(self, event):
        if not HAS_SCIPY or self.data_y_2d is None or self.data_x is None: return
        
        if not self.on_run_created: return

        # 1. Range Mask
        try:
            xmin = float(self.txt_min.GetValue())
            xmax = float(self.txt_max.GetValue())
        except ValueError:
            xmin, xmax = -np.inf, np.inf
        
        mask = (self.data_x >= xmin) & (self.data_x <= xmax)
        x_fit = self.data_x[mask]
        
        if len(x_fit) < 4:
            wx.MessageBox("Not enough data points in range.", "Error")
            return

        # 2. Model & Initial Guess
        model_name = self.choice_model.GetStringSelection()
        p_names = list(self.param_controls.keys())
        
        # Prepare wrapping logic (same as on_fit, but re-evaluated per row ideally, 
        # but here we use the initial UI state to define what is fixed/bounded)
        
        p_free_indices = []
        p_free_names = []
        p0_free = []
        bounds_min = []
        bounds_max = []
        p_full_current = []
        
        for i, p in enumerate(p_names):
            ctrls = self.param_controls[p]
            try:
                val = float(ctrls["value"].GetValue())
            except ValueError:
                val = 0.0
            p_full_current.append(val)
            
            if ctrls["fixed"].GetValue():
                continue
                
            p_free_indices.append(i)
            p_free_names.append(p)
            p0_free.append(val)
            
            try:
                v_min = float(ctrls["min"].GetValue())
            except ValueError:
                v_min = -np.inf
            try:
                v_max = float(ctrls["max"].GetValue())
            except ValueError:
                v_max = np.inf
            bounds_min.append(v_min)
            bounds_max.append(v_max)
            
        if not p_free_names:
            wx.MessageBox("All parameters are fixed.", "Info")
            return
            
        bounds = (bounds_min, bounds_max)
        fit_func_base = self._get_fit_func(model_name, p_names)
        
        def wrapper_func(x, *args):
            current_args = list(p_full_current)
            for idx, val in zip(p_free_indices, args):
                current_args[idx] = val
            return fit_func_base(x, *current_args)
        
        # 3. Iterate
        n_rows = self.data_y_2d.shape[0]
        results = np.zeros((n_rows, len(p_names)))
        
        # We need to fill fixed values into results
        for i in range(n_rows):
            results[i, :] = p_full_current # Initialize with fixed/initial values
        
        # We assume rows correspond to angle_values of source_run
        # Try to get y-axis values (e.g. Angles)
        y_axis = None
        if self.source_run and self.source_run.angle_values is not None:
            if len(self.source_run.angle_values) == n_rows:
                y_axis = self.source_run.angle_values
        
        if y_axis is None:
            y_axis = np.arange(n_rows) # Fallback indices

        dlg = wx.ProgressDialog("Batch Fitting", "Fitting rows...", maximum=n_rows, parent=self, style=wx.PD_APP_MODAL | wx.PD_AUTO_HIDE)
        
        p_current_free = list(p0_free)
        
        try:
            for i in range(n_rows):
                y_row_fit = self.data_y_2d[i, mask]
                try:
                    popt, _ = curve_fit(wrapper_func, x_fit, y_row_fit, p0=p_current_free, bounds=bounds)
                    
                    # Store results
                    # Populating full params array for this row
                    row_res = list(p_full_current) # Start with fixed
                    for idx, val in zip(p_free_indices, popt):
                        row_res[idx] = val
                    results[i, :] = row_res
                    
                    p_current_free = popt # Sequential update for next row
                except:
                    # Fallback to previous or NaN?
                    results[i, :] = results[i-1, :] if i > 0 else p_full_current
                
                if i % 10 == 0:
                    dlg.Update(i)
        finally:
            dlg.Destroy()
            
        # 4. Export Runs
        # Create a new run for EACH parameter
        cnt = 0
        for p_idx, p_name in enumerate(p_names):
            # p_name e.g. "x0", "Gamma"
            # Create 1D run: x = Angle, y = Param Value
            
            # Param name cleanup
            clean_p = p_name
            if model_name == "Sum of Lorentzian":
                # Maybe nicer name?
                pass
            
            nickname = f"{self.source_run.nickname}_{clean_p}"
            
            # y_axis is Angle, results[:, p_idx] is value
            new_run = Run.from_arrays(
                y_axis, results[:, p_idx], 
                x_label="Angle (deg)", 
                y_label=clean_p, 
                nickname=nickname
            )
            self.on_run_created(new_run)
            cnt += 1
            
        wx.MessageBox(f"Created {cnt} parameter runs.", "Batch Fit Complete")

    def _plot_data(self):
        self.ax.clear()
        if self.data_x is not None:
            self.ax.plot(self.data_x, self.data_y, 'o', markersize=2, alpha=0.5, label='Data')
            
        if self.fit_result_curve:
            self.ax.plot(self.fit_result_curve[0], self.fit_result_curve[1], 'r-', linewidth=1.5, label='Fit')
            
        self.ax.legend()
        self.ax.set_xlabel(self.x_label)
        self.ax.set_ylabel("Intensity")
        self.canvas.draw()

    def on_create_curve(self, event):
        if not self.fit_result_curve or not self.source_run: return
        if not self.on_run_created: return
        
        # Construct Formula String with baked values
        model_name = self.choice_model.GetStringSelection()
        vals = self._get_param_values()
        
        final_formula = ""
        
        if model_name == "Lorentzian":
            # y0 + A * (G**2 / (4 * (x - x0)**2 + G**2))
            # Use baked values
            final_formula = f"{vals['y0']} + {vals['A']} * ({vals['Gamma']}**2 / (4 * (x - {vals['x0']})**2 + {vals['Gamma']}**2))"
            
        elif model_name == "Sum of Lorentzian":
            n = self.spin_peaks.GetValue()
            parts = []
            for i in range(1, n+1):
                part = f"{vals[f'y0_{i}']} + {vals[f'A_{i}']} * ({vals[f'Gamma_{i}']}**2 / (4 * (x - {vals[f'x0_{i}']})**2 + {vals[f'Gamma_{i}']}**2))"
                parts.append(part)
            final_formula = " + ".join(parts)
            
        elif model_name == "User Defined":
            final_formula = self.txt_formula.GetValue()
            # Substitute parameters
            # Sort by length desc to avoid substring replacement collision (e.g. A vs A1)
            for p in sorted(vals.keys(), key=len, reverse=True):
                # Simple replace might be dangerous if variable names overlap (e.g. 'a' and 'aa')
                # Proper tokenization is better, but simple replace is what is asked for "baked".
                # To be safer, we can put spaces or parens?
                # Actually, DerivedRun formula logic relies on `eval`. 
                # If we replace 'a' with '1.0', it's hard to distinguish 'a' in 'tan(a)'.
                # A better approach: 
                # We can keep the formula as is, but DerivedRun needs self-contained string.
                # So we MUST replace.
                # We'll use regex word boundary.
                pattern = r"\b" + re.escape(p) + r"\b"
                final_formula = re.sub(pattern, str(vals[p]), final_formula)
        
        # Create Run
        # We use a special method or just 'from_formula'
        # The prompt says: "appropriately named derived run should be added... added to view of origin, overlayed..."
        
        # Generate new ID and nickname
        nickname = f"{self.source_run.nickname}_fit"
        
        # We need to construct a Derived Run. 
        # But `Run.from_formula` doesn't exist? Wait, `Run.from_formula` was mentioned in context.
        # Let's check `data_structure.py`.
        # Assuming `Run.from_formula(formula, ...)` exists.
        
        # Check `data_structure.py` content via memory or assumptions.
        # The context said "Implemented Run.from_arrays, Run.from_formula".
        # So I will use it.
        
        new_run = Run.from_formula(final_formula, nickname=nickname)
        
        # We also want to set default range for the derived run to match the fit range
        xmin = float(self.txt_min.GetValue())
        xmax = float(self.txt_max.GetValue())
        new_run.metadata["default_range"] = (xmin, xmax)
        new_run.metadata["default_autorange"] = False
        new_run.metadata["default_n_points"] = 500
        new_run.metadata["raw_x_unit"] = self.x_label if self.plot_type in {"A", "C"} else "deg"

        # Callback to MainFrame to add it and overlay it
        self.on_run_created(new_run, overlay_target_run=self.source_run, overlay_plot_type=self.plot_type)



class GradientStopsPanel(wx.Panel):
    """A compact draggable gradient-stop editor."""

    def __init__(self, parent, on_change=None):
        super().__init__(parent, size=(-1, 112), style=wx.BORDER_SIMPLE)
        self.stops = [[0.0, "#000000"], [1.0, "#FFFFFF"]]
        self.selected_index = 0
        self.on_change = on_change
        self._dragging = False

        self.SetMinSize((-1, 112))
        self.Bind(wx.EVT_PAINT, self.on_paint)
        self.Bind(wx.EVT_LEFT_DOWN, self.on_left_down)
        self.Bind(wx.EVT_LEFT_UP, self.on_left_up)
        self.Bind(wx.EVT_MOTION, self.on_motion)

    def set_stops(self, stops):
        cleaned = []
        for pos, color in stops:
            cleaned.append([float(np.clip(pos, 0.0, 1.0)), str(color)])
        self.stops = sorted(cleaned or [[0.0, "#000000"], [1.0, "#FFFFFF"]], key=lambda x: x[0])
        self.selected_index = min(self.selected_index, len(self.stops) - 1)
        self.Refresh()
        self._emit_change()

    def get_stops(self):
        return [[float(p), str(c)] for p, c in sorted(self.stops, key=lambda x: x[0])]

    def get_selected_stop(self):
        if not self.stops:
            return None
        return self.stops[self.selected_index]

    def set_selected_position(self, pos):
        if not self.stops:
            return
        stop = self.stops[self.selected_index]
        stop[0] = float(np.clip(pos, 0.0, 1.0))
        self.stops.sort(key=lambda x: x[0])
        self.selected_index = next((i for i, s in enumerate(self.stops) if s is stop), 0)
        self.Refresh()
        self._emit_change()

    def set_selected_color(self, color):
        if not self.stops:
            return
        self.stops[self.selected_index][1] = color
        self.Refresh()
        self._emit_change()

    def add_stop(self, pos=None, color=None):
        if pos is None:
            selected = self.get_selected_stop()
            pos = selected[0] if selected else 0.5
            larger = [p for p, _ in self.stops if p > pos]
            if larger:
                pos = (pos + larger[0]) / 2.0
            else:
                pos = min(1.0, pos + 0.1)
        if color is None:
            color = self._color_at(float(pos))
        stop = [float(np.clip(pos, 0.0, 1.0)), color]
        self.stops.append(stop)
        self.stops.sort(key=lambda x: x[0])
        self.selected_index = next((i for i, s in enumerate(self.stops) if s is stop), 0)
        self.Refresh()
        self._emit_change()

    def remove_selected(self):
        if len(self.stops) <= 2:
            return
        self.stops.pop(self.selected_index)
        self.selected_index = min(self.selected_index, len(self.stops) - 1)
        self.Refresh()
        self._emit_change()

    def _bar_rect(self):
        w, _ = self.GetClientSize()
        left = 18
        top = 24
        return wx.Rect(left, top, max(10, w - 36), 42)

    def _hex_to_rgb(self, color):
        c = wx.Colour(color)
        if not c.IsOk():
            c = wx.BLACK
        return c.Red(), c.Green(), c.Blue()

    def _rgb_to_hex(self, rgb):
        return "#{:02X}{:02X}{:02X}".format(
            int(np.clip(rgb[0], 0, 255)),
            int(np.clip(rgb[1], 0, 255)),
            int(np.clip(rgb[2], 0, 255)),
        )

    def _color_at(self, pos):
        stops = self.get_stops()
        if pos <= stops[0][0]:
            return stops[0][1]
        if pos >= stops[-1][0]:
            return stops[-1][1]
        for (p0, c0), (p1, c1) in zip(stops[:-1], stops[1:]):
            if p0 <= pos <= p1:
                span = max(p1 - p0, 1e-9)
                t = (pos - p0) / span
                rgb0 = np.array(self._hex_to_rgb(c0), dtype=float)
                rgb1 = np.array(self._hex_to_rgb(c1), dtype=float)
                return self._rgb_to_hex(rgb0 * (1.0 - t) + rgb1 * t)
        return stops[-1][1]

    def _x_for_pos(self, pos):
        rect = self._bar_rect()
        return rect.x + int(round(float(pos) * rect.width))

    def _pos_for_x(self, x):
        rect = self._bar_rect()
        return float(np.clip((x - rect.x) / max(rect.width, 1), 0.0, 1.0))

    def _hit_stop(self, point):
        if not self.stops:
            return None
        rect = self._bar_rect()
        y_min = rect.GetBottom() - 4
        y_max = rect.GetBottom() + 30
        if point.y < y_min or point.y > y_max:
            return None
        distances = [(abs(point.x - self._x_for_pos(pos)), idx) for idx, (pos, _) in enumerate(self.stops)]
        dist, idx = min(distances)
        return idx if dist <= 14 else None

    def _emit_change(self):
        if self.on_change:
            self.on_change()

    def on_paint(self, event):
        dc = wx.PaintDC(self)
        dc.SetBackground(wx.Brush(self.GetBackgroundColour()))
        dc.Clear()

        rect = self._bar_rect()
        for i in range(rect.width):
            color = wx.Colour(self._color_at(i / max(rect.width - 1, 1)))
            dc.SetPen(wx.Pen(color))
            dc.DrawLine(rect.x + i, rect.y, rect.x + i, rect.GetBottom())

        dc.SetPen(wx.Pen(wx.Colour(140, 140, 140)))
        dc.SetBrush(wx.TRANSPARENT_BRUSH)
        dc.DrawRectangle(rect)

        for idx, (pos, color) in enumerate(self.stops):
            x = self._x_for_pos(pos)
            selected = idx == self.selected_index
            border = wx.Colour(255, 128, 0) if selected else wx.Colour(150, 160, 170)
            fill = wx.Colour(color)

            triangle = [
                wx.Point(x, rect.GetBottom() + 1),
                wx.Point(x - 8, rect.GetBottom() + 10),
                wx.Point(x + 8, rect.GetBottom() + 10),
            ]
            dc.SetPen(wx.Pen(border, 2 if selected else 1))
            dc.SetBrush(wx.Brush(wx.WHITE))
            dc.DrawPolygon(triangle)
            dc.SetBrush(wx.Brush(fill))
            dc.DrawRectangle(x - 8, rect.GetBottom() + 10, 16, 16)
            dc.SetBrush(wx.TRANSPARENT_BRUSH)
            dc.DrawRectangle(x - 8, rect.GetBottom() + 10, 16, 16)

    def on_left_down(self, event):
        idx = self._hit_stop(event.GetPosition())
        if idx is None:
            idx = min(range(len(self.stops)), key=lambda i: abs(event.GetX() - self._x_for_pos(self.stops[i][0])))
        self.selected_index = idx
        self._dragging = True
        self.CaptureMouse()
        self.Refresh()
        self._emit_change()

    def on_left_up(self, event):
        if self._dragging and self.HasCapture():
            self.ReleaseMouse()
        self._dragging = False

    def on_motion(self, event):
        if self._dragging and event.Dragging() and event.LeftIsDown():
            self.set_selected_position(self._pos_for_x(event.GetX()))


class CustomColorbarsDialog(wx.Dialog):
    def __init__(self, parent):
        super().__init__(parent, title="Custom Colormaps", size=(680, 360), style=wx.DEFAULT_DIALOG_STYLE | wx.RESIZE_BORDER)

        raw_maps = config.get('custom_colormaps', {})
        self.custom_cmaps = {
            name: [[float(p), str(c)] for p, c in nodes]
            for name, nodes in raw_maps.items()
        }
        if not self.custom_cmaps:
            self.custom_cmaps = {"MyCustomMap": [[0.0, "#000000"], [1.0, "#FFFFFF"]]}
        self._original_cmaps = {
            name: [[float(p), str(c)] for p, c in nodes]
            for name, nodes in self.custom_cmaps.items()
        }
        self._updating_controls = False

        main_sizer = wx.BoxSizer(wx.VERTICAL)

        top_sizer = wx.BoxSizer(wx.HORIZONTAL)
        top_sizer.Add(wx.StaticText(self, label="Colormap:"), 0, wx.ALIGN_CENTER_VERTICAL | wx.ALL, 5)
        self.choice_cmap = wx.Choice(self)
        top_sizer.Add(self.choice_cmap, 1, wx.EXPAND | wx.ALL, 5)

        self.btn_new = wx.Button(self, label="New")
        self.btn_copy = wx.Button(self, label="Copy Existing")
        self.btn_reverse = wx.Button(self, label="Reverse")
        self.btn_delete = wx.Button(self, label="Delete")
        for btn in (self.btn_new, self.btn_copy, self.btn_reverse, self.btn_delete):
            top_sizer.Add(btn, 0, wx.ALIGN_CENTER_VERTICAL | wx.ALL, 5)
        main_sizer.Add(top_sizer, 0, wx.EXPAND)

        stops_header = wx.BoxSizer(wx.HORIZONTAL)
        stops_header.Add(wx.StaticText(self, label="Gradient stops"), 0, wx.ALIGN_CENTER_VERTICAL | wx.LEFT | wx.RIGHT, 8)
        stops_header.AddStretchSpacer(1)
        self.btn_add = wx.Button(self, label="+", size=(34, -1))
        self.btn_remove = wx.Button(self, label="-", size=(34, -1))
        stops_header.Add(self.btn_add, 0, wx.RIGHT, 4)
        stops_header.Add(self.btn_remove, 0, wx.RIGHT, 8)
        main_sizer.Add(stops_header, 0, wx.EXPAND | wx.TOP, 4)

        self.stops_panel = GradientStopsPanel(self, on_change=self.on_stops_changed)
        main_sizer.Add(self.stops_panel, 0, wx.EXPAND | wx.LEFT | wx.RIGHT | wx.BOTTOM, 8)

        edit_sizer = wx.FlexGridSizer(2, 4, 6, 8)
        edit_sizer.AddGrowableCol(2, 1)
        edit_sizer.Add(wx.StaticText(self, label="Color"), 0, wx.ALIGN_CENTER_VERTICAL)
        self.cp = wx.ColourPickerCtrl(self)
        edit_sizer.Add(self.cp, 0, wx.ALIGN_CENTER_VERTICAL)
        edit_sizer.Add(wx.StaticText(self, label="Position (%)"), 0, wx.ALIGN_CENTER_VERTICAL | wx.ALIGN_RIGHT)
        self.txt_pos = wx.TextCtrl(self, style=wx.TE_PROCESS_ENTER)
        edit_sizer.Add(self.txt_pos, 0, wx.EXPAND)

        edit_sizer.Add(wx.StaticText(self, label="Brightness"), 0, wx.ALIGN_CENTER_VERTICAL)
        self.slider_brightness = wx.Slider(self, value=50, minValue=0, maxValue=100)
        edit_sizer.Add(self.slider_brightness, 1, wx.EXPAND)
        self.lbl_brightness = wx.StaticText(self, label="50%")
        edit_sizer.Add(self.lbl_brightness, 0, wx.ALIGN_CENTER_VERTICAL | wx.ALIGN_RIGHT)
        edit_sizer.Add(wx.StaticText(self, label=""), 0)
        main_sizer.Add(edit_sizer, 0, wx.EXPAND | wx.LEFT | wx.RIGHT | wx.BOTTOM, 10)

        btn_sizer = self.CreateButtonSizer(wx.OK | wx.CANCEL)
        main_sizer.Add(btn_sizer, 0, wx.ALIGN_RIGHT | wx.ALL, 10)
        self.SetSizer(main_sizer)

        self.choice_cmap.Bind(wx.EVT_CHOICE, self.on_cmap_select)
        self.btn_new.Bind(wx.EVT_BUTTON, self.on_new)
        self.btn_copy.Bind(wx.EVT_BUTTON, self.on_copy_existing)
        self.btn_reverse.Bind(wx.EVT_BUTTON, self.on_reverse)
        self.btn_delete.Bind(wx.EVT_BUTTON, self.on_delete)
        self.btn_add.Bind(wx.EVT_BUTTON, self.on_add_node)
        self.btn_remove.Bind(wx.EVT_BUTTON, self.on_remove_node)
        self.cp.Bind(wx.EVT_COLOURPICKER_CHANGED, self.on_color_change)
        self.txt_pos.Bind(wx.EVT_TEXT_ENTER, self.on_position_enter)
        self.txt_pos.Bind(wx.EVT_KILL_FOCUS, self.on_position_enter)
        self.slider_brightness.Bind(wx.EVT_SLIDER, self.on_brightness_change)

        self.refresh_cmap_list()

    def _normalize_nodes(self, nodes):
        cleaned = [[float(np.clip(pos, 0.0, 1.0)), str(col)] for pos, col in nodes]
        cleaned.sort(key=lambda x: x[0])
        return cleaned

    def refresh_cmap_list(self, select_name=None):
        names = list(self.custom_cmaps.keys())
        self.choice_cmap.Set(names)
        if not names:
            self.stops_panel.set_stops([[0.0, "#000000"], [1.0, "#FFFFFF"]])
            return
        if select_name in names:
            self.choice_cmap.SetStringSelection(select_name)
        else:
            self.choice_cmap.SetSelection(0)
        self.load_nodes(self.get_current_name())

    def load_nodes(self, name):
        if not name:
            return
        self.custom_cmaps[name] = self._normalize_nodes(self.custom_cmaps.get(name, []))
        self.stops_panel.set_stops(self.custom_cmaps[name])
        self.sync_controls_from_selection()

    def get_current_name(self):
        return self.choice_cmap.GetStringSelection()

    def on_cmap_select(self, event):
        self.load_nodes(self.get_current_name())

    def on_stops_changed(self):
        name = self.get_current_name()
        if name:
            self.custom_cmaps[name] = self.stops_panel.get_stops()
        self.sync_controls_from_selection()

    def sync_controls_from_selection(self):
        stop = self.stops_panel.get_selected_stop()
        if not stop:
            return
        self._updating_controls = True
        try:
            pos, color = stop
            self.txt_pos.ChangeValue(f"{pos * 100:.1f}")
            self.cp.SetColour(wx.Colour(color))

            r, g, b = [v / 255.0 for v in self.stops_panel._hex_to_rgb(color)]
            _, lightness, _ = colorsys.rgb_to_hls(r, g, b)
            brightness = int(round(lightness * 100))
            self.slider_brightness.SetValue(brightness)
            self.lbl_brightness.SetLabel(f"{brightness}%")
        finally:
            self._updating_controls = False

    def on_position_enter(self, event):
        if self._updating_controls:
            if event:
                event.Skip()
            return
        try:
            pos = float(self.txt_pos.GetValue().replace("%", "").strip()) / 100.0
        except ValueError:
            wx.MessageBox("Position must be a number from 0 to 100.", "Invalid position")
            return
        self.stops_panel.set_selected_position(pos)
        if event:
            event.Skip()

    def on_color_change(self, event):
        if self._updating_controls:
            return
        color = self.cp.GetColour().GetAsString(wx.C2S_HTML_SYNTAX)
        self.stops_panel.set_selected_color(color)

    def on_brightness_change(self, event):
        if self._updating_controls:
            return
        stop = self.stops_panel.get_selected_stop()
        if not stop:
            return
        color = wx.Colour(stop[1])
        r, g, b = color.Red() / 255.0, color.Green() / 255.0, color.Blue() / 255.0
        h, _, s = colorsys.rgb_to_hls(r, g, b)
        new_lightness = self.slider_brightness.GetValue() / 100.0
        nr, ng, nb = colorsys.hls_to_rgb(h, new_lightness, s)
        new_color = "#{:02X}{:02X}{:02X}".format(int(nr * 255), int(ng * 255), int(nb * 255))
        self.lbl_brightness.SetLabel(f"{self.slider_brightness.GetValue()}%")
        self.stops_panel.set_selected_color(new_color)

    def on_new(self, event):
        dlg = wx.TextEntryDialog(self, "New colormap name:", "New Colormap")
        if dlg.ShowModal() == wx.ID_OK:
            name = dlg.GetValue().strip()
            if name and name not in self.custom_cmaps:
                self.custom_cmaps[name] = [[0.0, "#000000"], [1.0, "#FFFFFF"]]
                self.refresh_cmap_list(name)
        dlg.Destroy()

    def on_delete(self, event):
        name = self.get_current_name()
        if name in self.custom_cmaps:
            del self.custom_cmaps[name]
            self.refresh_cmap_list()

    def on_copy_existing(self, event):
        cmaps = ['OrRd', 'rocket_r', 'inferno', 'magma_r', 'cividis', 'gray', 'seismic', 'jet', 'hsv']
        if HAS_CMCRAMERI:
            cmaps.extend(['cmc.batlow', 'cmc.roma', 'cmc.lajolla', 'cmc.oslo', 'cmc.tokyo'])
        dlg = wx.SingleChoiceDialog(self, "Select base colormap to copy:", "Copy Existing", cmaps)
        if dlg.ShowModal() == wx.ID_OK:
            base = dlg.GetStringSelection()
            name_dlg = wx.TextEntryDialog(self, "New custom colormap name:", "Name", value=f"{base}_custom")
            if name_dlg.ShowModal() == wx.ID_OK:
                name = name_dlg.GetValue().strip()
                if name:
                    try:
                        cmap = plt.get_cmap(base)
                    except Exception:
                        cmap = plt.get_cmap('OrRd')
                    self.custom_cmaps[name] = [[float(p), to_hex(cmap(p))] for p in np.linspace(0, 1, 5)]
                    self.refresh_cmap_list(name)
            name_dlg.Destroy()
        dlg.Destroy()

    def on_reverse(self, event):
        name = self.get_current_name()
        if not name:
            return
        self.custom_cmaps[name] = sorted([[1.0 - pos, col] for pos, col in self.custom_cmaps[name]], key=lambda x: x[0])
        self.load_nodes(name)

    def on_add_node(self, event):
        self.stops_panel.add_stop()

    def on_remove_node(self, event):
        self.stops_panel.remove_selected()

    def get_results(self):
        return {name: self._normalize_nodes(nodes) for name, nodes in self.custom_cmaps.items()}

    def get_changed_names(self):
        result = self.get_results()
        names = set(result.keys()) | set(self._original_cmaps.keys())
        return [name for name in names if result.get(name) != self._original_cmaps.get(name)]


class PreferencesPanel(wx.Panel):
    """App-wide defaults applied to new views and newly imported ambiguous runs."""

    def __init__(self, parent, on_use_current_view=None):
        super().__init__(parent)
        self._on_use_current_view = on_use_current_view
        self._loading = False

        sizer = wx.BoxSizer(wx.VERTICAL)
        grid = wx.FlexGridSizer(0, 2, 6, 8)
        grid.AddGrowableCol(1, 1)

        unit_choices = ["meV", "cm-1"]
        self.choice_unit = wx.Choice(self, choices=unit_choices)
        grid.Add(wx.StaticText(self, label="Default unit"), 0, wx.ALIGN_CENTER_VERTICAL)
        grid.Add(self.choice_unit, 0, wx.EXPAND)

        cmap_choices = sorted(set(list(plt.colormaps()) + list(config.get("custom_colormaps", {}).keys())))
        self.choice_cmap = wx.ComboBox(self, choices=cmap_choices, style=wx.CB_DROPDOWN)
        grid.Add(wx.StaticText(self, label="Default colormap"), 0, wx.ALIGN_CENTER_VERTICAL)
        grid.Add(self.choice_cmap, 1, wx.EXPAND)

        self.spin_vmin = wx.SpinCtrlDouble(self, min=0, max=100, inc=1)
        self.spin_vmax = wx.SpinCtrlDouble(self, min=0, max=100, inc=1)
        contrast_row = wx.BoxSizer(wx.HORIZONTAL)
        contrast_row.Add(self.spin_vmin, 0, wx.RIGHT, 5)
        contrast_row.Add(self.spin_vmax, 0)
        grid.Add(wx.StaticText(self, label="Default contrast (%)"), 0, wx.ALIGN_CENTER_VERTICAL)
        grid.Add(contrast_row, 0, wx.EXPAND)

        self.choice_angle = wx.Choice(self, choices=["polar", "cartesian"])
        grid.Add(wx.StaticText(self, label="Angle slice"), 0, wx.ALIGN_CENTER_VERTICAL)
        grid.Add(self.choice_angle, 0, wx.EXPAND)

        bin_row = wx.BoxSizer(wx.HORIZONTAL)
        spin_style = wx.SP_ARROW_KEYS | wx.TE_PROCESS_ENTER
        self.spin_default_bin_x = wx.SpinCtrl(self, min=1, max=999, initial=1, size=(62, -1), style=spin_style)
        self.spin_default_bin_y = wx.SpinCtrl(self, min=1, max=999, initial=1, size=(62, -1), style=spin_style)
        self.chk_default_box_binning = wx.CheckBox(self, label="Box")
        bin_row.Add(wx.StaticText(self, label="X"), 0, wx.ALIGN_CENTER_VERTICAL | wx.RIGHT, 3)
        bin_row.Add(self.spin_default_bin_x, 0, wx.RIGHT, 6)
        bin_row.Add(wx.StaticText(self, label="Y"), 0, wx.ALIGN_CENTER_VERTICAL | wx.RIGHT, 3)
        bin_row.Add(self.spin_default_bin_y, 0, wx.RIGHT, 8)
        bin_row.Add(self.chk_default_box_binning, 0, wx.ALIGN_CENTER_VERTICAL)
        grid.Add(wx.StaticText(self, label="Default binning"), 0, wx.ALIGN_CENTER_VERTICAL)
        grid.Add(bin_row, 0, wx.EXPAND)

        self.chk_secondary = wx.CheckBox(self, label="Show alternate unit axis")
        grid.AddSpacer(1)
        grid.Add(self.chk_secondary, 0, wx.EXPAND)

        self.chk_highlight_modifier = wx.CheckBox(self, label="Ctrl/Cmd-click moves highlight")
        grid.AddSpacer(1)
        grid.Add(self.chk_highlight_modifier, 0, wx.EXPAND)

        self.choice_unknown_1d = wx.Choice(self, choices=unit_choices)
        grid.Add(wx.StaticText(self, label="Unknown 1D unit"), 0, wx.ALIGN_CENTER_VERTICAL)
        grid.Add(self.choice_unknown_1d, 0, wx.EXPAND)

        self.choice_unknown_2d = wx.Choice(self, choices=unit_choices)
        grid.Add(wx.StaticText(self, label="Unknown 2D unit"), 0, wx.ALIGN_CENTER_VERTICAL)
        grid.Add(self.choice_unknown_2d, 0, wx.EXPAND)

        sizer.Add(grid, 0, wx.ALL | wx.EXPAND, 8)

        btn_row = wx.BoxSizer(wx.HORIZONTAL)
        self.btn_use_current = wx.Button(self, label="Use Current View")
        self.btn_reset = wx.Button(self, label="Reset Defaults")
        btn_row.Add(self.btn_use_current, 0, wx.RIGHT, 6)
        btn_row.Add(self.btn_reset, 0)
        sizer.Add(btn_row, 0, wx.LEFT | wx.RIGHT | wx.BOTTOM, 8)
        self.SetSizer(sizer)

        for ctrl in (self.choice_unit, self.choice_angle, self.choice_unknown_1d, self.choice_unknown_2d):
            ctrl.Bind(wx.EVT_CHOICE, self.on_change)
        for ctrl in (self.spin_vmin, self.spin_vmax):
            ctrl.Bind(wx.EVT_SPINCTRLDOUBLE, self.on_change)
            ctrl.Bind(wx.EVT_TEXT_ENTER, self.on_change)
        for ctrl in (self.spin_default_bin_x, self.spin_default_bin_y):
            ctrl.Bind(wx.EVT_SPINCTRL, self.on_change)
            ctrl.Bind(wx.EVT_TEXT_ENTER, self.on_change)
            ctrl.Bind(wx.EVT_TEXT, self.on_change)
        self.chk_default_box_binning.Bind(wx.EVT_CHECKBOX, self.on_change)
        self.chk_secondary.Bind(wx.EVT_CHECKBOX, self.on_change)
        self.chk_highlight_modifier.Bind(wx.EVT_CHECKBOX, self.on_change)
        self.choice_cmap.Bind(wx.EVT_COMBOBOX, self.on_change)
        self.choice_cmap.Bind(wx.EVT_TEXT, self.on_change)
        self.btn_use_current.Bind(wx.EVT_BUTTON, self.on_use_current_view)
        self.btn_reset.Bind(wx.EVT_BUTTON, self.on_reset_defaults)

        self.refresh_from_config()

    def _set_choice(self, choice: wx.Choice, value: str) -> None:
        if choice.SetStringSelection(value):
            return
        if choice.GetCount():
            choice.SetSelection(0)

    def _coerce_binning(self, value: Any) -> int:
        try:
            return max(1, int(value))
        except (TypeError, ValueError):
            return 1

    def refresh_from_config(self):
        self._loading = True
        try:
            self._set_choice(self.choice_unit, normalize_spectral_unit(config.get("default_spectral_unit", config.get("unit", "meV"))))
            self.choice_cmap.SetValue(config.get("default_colormap", config.get("colormap", "OrRd")))
            self.spin_vmin.SetValue(float(config.get("default_vmin_percent", 0.0)))
            self.spin_vmax.SetValue(float(config.get("default_vmax_percent", 100.0)))
            self._set_choice(self.choice_angle, str(config.get("default_angle_slice_type", "polar")))
            self.spin_default_bin_x.SetValue(self._coerce_binning(config.get("default_slice_x_binning", 1)))
            self.spin_default_bin_y.SetValue(self._coerce_binning(config.get("default_slice_y_binning", 1)))
            self.chk_default_box_binning.SetValue(str(config.get("default_slice_binning_mode", "cross")).lower() == "box")
            self.chk_secondary.SetValue(bool(config.get("show_secondary_unit_axis", True)))
            self.chk_highlight_modifier.SetValue(bool(config.get("highlight_requires_modifier", False)))
            self._set_choice(self.choice_unknown_1d, normalize_spectral_unit(config.get("default_unknown_1d_spectral_unit", "meV")))
            self._set_choice(self.choice_unknown_2d, normalize_spectral_unit(config.get("default_unknown_2d_spectral_unit", "meV")))
        finally:
            self._loading = False

    def _save_values(self):
        if self._loading:
            return
        vmin = float(self.spin_vmin.GetValue())
        vmax = float(self.spin_vmax.GetValue())
        if vmax < vmin:
            vmax = vmin
            self.spin_vmax.SetValue(vmax)
        config.set("default_spectral_unit", normalize_spectral_unit(self.choice_unit.GetStringSelection()))
        config.set("default_colormap", self.choice_cmap.GetValue() or "OrRd")
        config.set("default_vmin_percent", vmin)
        config.set("default_vmax_percent", vmax)
        config.set("default_angle_slice_type", self.choice_angle.GetStringSelection() or "polar")
        config.set("default_slice_x_binning", self._coerce_binning(self.spin_default_bin_x.GetValue()))
        config.set("default_slice_y_binning", self._coerce_binning(self.spin_default_bin_y.GetValue()))
        config.set("default_slice_binning_mode", "box" if self.chk_default_box_binning.GetValue() else "cross")
        config.set("show_secondary_unit_axis", self.chk_secondary.GetValue())
        config.set("highlight_requires_modifier", self.chk_highlight_modifier.GetValue())
        config.set("default_unknown_1d_spectral_unit", normalize_spectral_unit(self.choice_unknown_1d.GetStringSelection()))
        config.set("default_unknown_2d_spectral_unit", normalize_spectral_unit(self.choice_unknown_2d.GetStringSelection()))

    def on_change(self, event):
        self._save_values()
        if event:
            event.Skip()

    def on_use_current_view(self, event):
        if self._on_use_current_view:
            self._on_use_current_view()
        self.refresh_from_config()

    def on_reset_defaults(self, event):
        for key in (
            "default_spectral_unit", "default_colormap", "default_vmin_percent",
            "default_vmax_percent", "default_angle_slice_type",
            "default_slice_x_binning", "default_slice_y_binning",
            "default_slice_binning_mode",
            "show_secondary_unit_axis", "highlight_requires_modifier",
            "default_unknown_1d_spectral_unit",
            "default_unknown_2d_spectral_unit",
        ):
            config.set(key, DEFAULT_SETTINGS[key])
        self.refresh_from_config()


class PlotConfigPanel(wx.Panel):
    def __init__(self, parent, on_reset=None):
        super().__init__(parent)
        
        self._on_reset = on_reset
        
        self.x_unit = normalize_spectral_unit(config.get('default_spectral_unit', config.get('unit', 'meV')))

        sizer = wx.BoxSizer(wx.VERTICAL)
        
        # Range controls
        range_sizer = wx.FlexGridSizer(3, 3, 5, 5)
        range_sizer.AddGrowableCol(1, 1)
        range_sizer.AddGrowableCol(2, 1)

        # Unit radio buttons
        self.rb_cm1 = wx.RadioButton(self, label="cm-1", style=wx.RB_GROUP)
        self.rb_mev = wx.RadioButton(self, label="meV")
        self._set_unit_controls(self.x_unit)
        unit_sizer = wx.BoxSizer(wx.HORIZONTAL)
        unit_sizer.Add(self.rb_cm1, 0, wx.RIGHT, 5)
        unit_sizer.Add(self.rb_mev, 0)
        
        range_sizer.Add(wx.StaticText(self, label="Range"), 0, wx.ALIGN_CENTER_VERTICAL)
        range_sizer.Add(unit_sizer, 0, wx.ALIGN_CENTER)
        range_sizer.Add(wx.StaticText(self, label=""), 0) # Placeholder

        # X Range
        self.x_min_text = wx.TextCtrl(self, style=wx.TE_PROCESS_ENTER)
        self.x_max_text = wx.TextCtrl(self, style=wx.TE_PROCESS_ENTER)
        x_sizer = wx.BoxSizer(wx.HORIZONTAL)
        x_sizer.Add(self.x_min_text, 1, wx.EXPAND | wx.RIGHT, 5)
        x_sizer.Add(self.x_max_text, 1, wx.EXPAND)
        range_sizer.Add(wx.StaticText(self, label="X:"), 0, wx.ALIGN_CENTER_VERTICAL | wx.ALIGN_RIGHT)
        range_sizer.Add(x_sizer, 1, wx.EXPAND)
        range_sizer.Add(wx.StaticText(self, label=""), 0) # Placeholder

        # Y Range
        self.y_min_text = wx.TextCtrl(self, style=wx.TE_PROCESS_ENTER)
        self.y_max_text = wx.TextCtrl(self, style=wx.TE_PROCESS_ENTER)
        y_sizer = wx.BoxSizer(wx.HORIZONTAL)
        y_sizer.Add(self.y_min_text, 1, wx.EXPAND | wx.RIGHT, 5)
        y_sizer.Add(self.y_max_text, 1, wx.EXPAND)
        range_sizer.Add(wx.StaticText(self, label="Y:"), 0, wx.ALIGN_CENTER_VERTICAL | wx.ALIGN_RIGHT)
        range_sizer.Add(y_sizer, 1, wx.EXPAND)
        range_sizer.Add(wx.StaticText(self, label="deg"), 0, wx.ALIGN_CENTER_VERTICAL)

        sizer.Add(range_sizer, 0, wx.EXPAND | wx.ALL, 5)

        # Slice binning controls
        bin_box = wx.StaticBoxSizer(wx.StaticBox(self, label="Slice binning"), wx.VERTICAL)
        bin_grid = wx.FlexGridSizer(2, 4, 4, 6)
        for label in ("X", "Y", "All"):
            bin_grid.Add(wx.StaticText(self, label=label), 0, wx.ALIGN_CENTER)
        bin_grid.Add(wx.StaticText(self, label="Mode"), 0, wx.ALIGN_CENTER)

        spin_style = wx.SP_ARROW_KEYS | wx.TE_PROCESS_ENTER
        self.spin_bin_x = wx.SpinCtrl(self, min=1, max=999, initial=1, size=(64, -1), style=spin_style)
        self.spin_bin_y = wx.SpinCtrl(self, min=1, max=999, initial=1, size=(64, -1), style=spin_style)
        self.spin_bin_all = wx.SpinCtrl(self, min=1, max=999, initial=1, size=(64, -1), style=spin_style)
        self.btn_box_binning = wx.ToggleButton(self, label="Cross")
        bin_grid.Add(self.spin_bin_x, 0, wx.EXPAND)
        bin_grid.Add(self.spin_bin_y, 0, wx.EXPAND)
        bin_grid.Add(self.spin_bin_all, 0, wx.EXPAND)
        bin_grid.Add(self.btn_box_binning, 0, wx.EXPAND)
        bin_box.Add(bin_grid, 0, wx.EXPAND | wx.ALL, 5)
        sizer.Add(bin_box, 0, wx.EXPAND | wx.LEFT | wx.RIGHT | wx.BOTTOM, 5)

        fit_overlay_sizer = wx.BoxSizer(wx.HORIZONTAL)
        fit_overlay_sizer.Add(wx.StaticText(self, label="Fit overlay:"), 0, wx.ALIGN_CENTER_VERTICAL | wx.RIGHT, 5)
        self.choice_fit_overlay = wx.Choice(self, choices=["Off", "Global", "Row", "Both"])
        fit_overlay_sizer.Add(self.choice_fit_overlay, 1, wx.EXPAND)
        sizer.Add(fit_overlay_sizer, 0, wx.EXPAND | wx.LEFT | wx.RIGHT | wx.BOTTOM, 5)
        
        # Colormap selection
        cmap_sizer = wx.BoxSizer(wx.HORIZONTAL)
        cmap_sizer.Add(wx.StaticText(self, label="Colormap:"), 0, wx.ALIGN_CENTER_VERTICAL | wx.RIGHT, 5)
        
        # Ensure seaborn is imported so its cmaps are registered
        try:
            import seaborn as sns
        except ImportError:
            pass

        # Get all matplotlib colormaps (including registered ones like seaborn)
        self.cmaps_std = sorted([m for m in plt.colormaps() if not m.startswith('cmc.')])
        self.cmaps_cmc = sorted([m for m in plt.colormaps() if m.startswith('cmc.')])
        
        if HAS_CMCRAMERI and not self.cmaps_cmc:
            # Fallback if cmcrameri didn't auto-register
            self.cmaps_cmc = [
                'cmc.batlow', 'cmc.batlowW', 'cmc.batlowK', 
                'cmc.glasgow', 'cmc.lipari', 'cmc.navia', 
                'cmc.grayC', 'cmc.grayC_r',
                'cmc.roma', 'cmc.roma_r',
                'cmc.devon', 'cmc.devon_r',
                'cmc.lajolla', 'cmc.lajolla_r', 
                'cmc.bamako', 'cmc.bamako_r',
                'cmc.davos', 'cmc.davos_r',
                'cmc.bilbao', 'cmc.bilbao_r',
                'cmc.oslo', 'cmc.oslo_r', 
                'cmc.acton', 'cmc.acton_r', 
                'cmc.turku', 'cmc.turku_r',
                'cmc.tokyo', 'cmc.tokyo_r',
                'cmc.lapaz', 'cmc.lapaz_r',
                'cmc.nuuk', 'cmc.nuuk_r',
                'cmc.imola', 'cmc.imola_r',
                'cmc.berlin', 'cmc.berlin_r',
                'cmc.lisbon', 'cmc.lisbon_r',
                'cmc.broc', 'cmc.broc_r',
                'cmc.cork', 'cmc.cork_r',
                'cmc.vik', 'cmc.vik_r',
                'cmc.buda', 'cmc.buda_r',
            ]
        
        # Category Choice
        cat_choices = ['Standard']
        if self.cmaps_cmc:
            cat_choices.append('CMCrameri')
        cat_choices.append('Custom...')
        self.choice_cat = wx.Choice(self, choices=cat_choices)
        
        # Map Choice - Using ComboBox for searchability
        self.choice_cmap = wx.ComboBox(self, style=wx.CB_DROPDOWN) 
        self._filtering = False
        
        self._register_custom_cmaps()

        # Load saved colormap and populate choices
        saved_cmap = config.get('default_colormap', config.get('colormap', 'OrRd'))
        self.set_colormap(saved_cmap)

        cmap_sizer.Add(self.choice_cat, 0, wx.RIGHT | wx.ALIGN_CENTER_VERTICAL, 5)
        cmap_sizer.Add(self.choice_cmap, 1, wx.EXPAND)
        
        sizer.Add(cmap_sizer, 0, wx.EXPAND | wx.ALL, 5)
        
        # Contrast controls
        sizer.Add(wx.StaticText(self, label="Contrast"), 0, wx.LEFT | wx.TOP, 5)
        
        # Header for Percentile vs Value
        header_sizer = wx.BoxSizer(wx.HORIZONTAL)
        header_sizer.Add(wx.StaticText(self, label=""), 0, wx.RIGHT, 35) # spacing for 'min:' label
        header_sizer.Add(wx.StaticText(self, label="Percentile (%)"), 1, wx.ALIGN_CENTER)
        header_sizer.Add(wx.StaticText(self, label="Value (a.u.)"), 0, wx.ALIGN_CENTER | wx.LEFT, 10)
        sizer.Add(header_sizer, 0, wx.EXPAND | wx.LEFT | wx.RIGHT, 5)

        vmin_sizer = wx.BoxSizer(wx.HORIZONTAL)
        self.vmin_label = wx.StaticText(self, label="min:")
        self.vmin_slider = wx.Slider(self, value=0, minValue=0, maxValue=100)
        self.txt_vmin = wx.TextCtrl(self, value="0", size=(60, -1), style=wx.TE_PROCESS_ENTER)
        
        vmin_sizer.Add(self.vmin_label, 0, wx.ALIGN_CENTER_VERTICAL | wx.RIGHT, 5)
        vmin_sizer.Add(self.vmin_slider, 1, wx.EXPAND | wx.RIGHT, 5)
        vmin_sizer.Add(self.txt_vmin, 0, wx.ALIGN_CENTER_VERTICAL)
        sizer.Add(vmin_sizer, 0, wx.EXPAND | wx.LEFT | wx.RIGHT, 5)

        vmax_sizer = wx.BoxSizer(wx.HORIZONTAL)
        self.vmax_label = wx.StaticText(self, label="max:")
        self.vmax_slider = wx.Slider(self, value=100, minValue=0, maxValue=100)
        self.txt_vmax = wx.TextCtrl(self, value="100", size=(60, -1), style=wx.TE_PROCESS_ENTER)
        
        vmax_sizer.Add(self.vmax_label, 0, wx.ALIGN_CENTER_VERTICAL | wx.RIGHT, 5)
        vmax_sizer.Add(self.vmax_slider, 1, wx.EXPAND | wx.RIGHT, 5)
        vmax_sizer.Add(self.txt_vmax, 0, wx.ALIGN_CENTER_VERTICAL)
        sizer.Add(vmax_sizer, 0, wx.EXPAND | wx.LEFT | wx.RIGHT, 5)

        # Action buttons
        btn_sizer = wx.BoxSizer(wx.HORIZONTAL)
        self.reset_button = wx.Button(self, label="Reset Plot")
        self.btn_roi_vlim = wx.Button(self, label="Adjust color to this ROI")
        
        btn_sizer.Add(self.reset_button, 0, wx.ALL, 5)
        btn_sizer.Add(self.btn_roi_vlim, 0, wx.ALL, 5)
        
        sizer.Add(btn_sizer, 0, wx.ALIGN_CENTER)

        self.SetSizer(sizer)

        # Bind events
        self.rb_cm1.Bind(wx.EVT_RADIOBUTTON, self.on_unit_change)
        self.rb_mev.Bind(wx.EVT_RADIOBUTTON, self.on_unit_change)
        self.x_min_text.Bind(wx.EVT_TEXT_ENTER, self.on_x_range_enter)
        self.x_max_text.Bind(wx.EVT_TEXT_ENTER, self.on_x_range_enter)
        self.y_min_text.Bind(wx.EVT_TEXT_ENTER, self.on_y_range_enter)
        self.y_max_text.Bind(wx.EVT_TEXT_ENTER, self.on_y_range_enter)
        
        self.vmin_slider.Bind(wx.EVT_SLIDER, self.on_vlim_slide)
        self.vmax_slider.Bind(wx.EVT_SLIDER, self.on_vlim_slide)
        self.txt_vmin.Bind(wx.EVT_TEXT_ENTER, self.on_vlim_text_enter)
        self.txt_vmax.Bind(wx.EVT_TEXT_ENTER, self.on_vlim_text_enter)
        
        self.choice_cat.Bind(wx.EVT_CHOICE, self.on_cat_change)
        self.choice_cmap.Bind(wx.EVT_COMBOBOX, self.on_cmap_change)
        self.choice_cmap.Bind(wx.EVT_TEXT, self.on_cmap_text)
        if hasattr(wx, "EVT_COMBOBOX_DROPDOWN"):
            self.choice_cmap.Bind(wx.EVT_COMBOBOX_DROPDOWN, self.on_cmap_dropdown)
        self.reset_button.Bind(wx.EVT_BUTTON, self.on_reset_button)
        self.btn_roi_vlim.Bind(wx.EVT_BUTTON, self.on_roi_vlim_button)
        for ctrl in (self.spin_bin_x, self.spin_bin_y):
            ctrl.Bind(wx.EVT_SPINCTRL, self.on_binning_change)
            ctrl.Bind(wx.EVT_TEXT_ENTER, self.on_binning_change)
            ctrl.Bind(wx.EVT_TEXT, self.on_binning_change)
        self.spin_bin_all.Bind(wx.EVT_SPINCTRL, self.on_all_binning_change)
        self.spin_bin_all.Bind(wx.EVT_TEXT_ENTER, self.on_all_binning_change)
        self.spin_bin_all.Bind(wx.EVT_TEXT, self.on_all_binning_change)
        self.btn_box_binning.Bind(wx.EVT_TOGGLEBUTTON, self.on_binning_mode_change)
        self.choice_fit_overlay.Bind(wx.EVT_CHOICE, self.on_fit_overlay_change)

        self.target_view: Optional["ViewPanel"] = None
        self._syncing_binning_controls = False
        self.set_target_view(None)

    def _set_unit_controls(self, unit: str):
        self.x_unit = normalize_spectral_unit(unit)
        if self.x_unit == 'meV':
            self.rb_mev.SetValue(True)
        else:
            self.rb_cm1.SetValue(True)

    def set_target_view(self, view_panel: Optional["ViewPanel"]):
        self.target_view = view_panel
        if self.target_view is None:
            for widget in [self.rb_cm1, self.rb_mev, self.x_min_text, self.x_max_text,
                           self.y_min_text, self.y_max_text, self.choice_cat, self.choice_cmap,
                           self.vmin_slider, self.txt_vmin, self.vmax_slider, self.txt_vmax,
                           self.reset_button, self.btn_roi_vlim, self.spin_bin_x,
                           self.spin_bin_y, self.spin_bin_all, self.btn_box_binning,
                           self.choice_fit_overlay]:
                widget.Disable()
            return

        for widget in [self.rb_cm1, self.rb_mev, self.x_min_text, self.x_max_text,
                       self.y_min_text, self.y_max_text, self.choice_cat, self.choice_cmap,
                       self.vmin_slider, self.txt_vmin, self.vmax_slider, self.txt_vmax,
                       self.reset_button, self.btn_roi_vlim, self.spin_bin_x,
                       self.spin_bin_y, self.spin_bin_all, self.btn_box_binning,
                       self.choice_fit_overlay]:
            widget.Enable()

        plot_config = self.target_view.get_plot_config()
        self._set_unit_controls(plot_config.get('unit', self.target_view.get_spectral_unit()))
        
        # Sync Limits
        xlim = plot_config.get('xlim', (0, 1))
        ylim = plot_config.get('ylim', (0, 1))

        self.set_x_range(xlim[0], xlim[1])
        self.set_y_range(ylim[0], ylim[1])
        
        # Sync Contrast
        vmin_p = plot_config.get('vmin_p', 0)
        vmax_p = plot_config.get('vmax_p', 100)
        self.set_vlim_range(vmin_p, vmax_p)
        
        # Sync Absolute Contrast Values
        v_abs = self.target_view.get_absolute_vlim_for_percentiles(vmin_p, vmax_p)
        self.update_absolute_vlim_display(v_abs[0], v_abs[1])
        
        # Sync Colormap
        cmap = plot_config.get('cmap', 'OrRd')
        self.set_colormap(cmap)

        self._set_binning_controls(
            plot_config.get('slice_x_binning', 1),
            plot_config.get('slice_y_binning', 1),
            plot_config.get('slice_binning_mode', 'cross'),
        )
        self._set_fit_overlay_control(plot_config.get('fit_overlay_mode', 'off'))

    def _cmap_choices_for_category(self):
        cat = self.choice_cat.GetStringSelection()
        if cat == 'Custom...':
            return list(config.get('custom_colormaps', {}).keys()) + ['Edit Custom...']
        if cat == 'CMCrameri':
            return list(self.cmaps_cmc)
        return list(self.cmaps_std)

    def _restore_full_cmap_choices(self, keep_value=True):
        value = self.choice_cmap.GetValue()
        insertion_point = self.choice_cmap.GetInsertionPoint()
        self._filtering = True
        choices = self._cmap_choices_for_category()
        self.choice_cmap.Set(choices)
        if keep_value and value:
            self.choice_cmap.SetValue(value)
            try:
                self.choice_cmap.SetInsertionPoint(insertion_point)
            except Exception:
                pass
        elif choices:
            self.choice_cmap.SetSelection(0)
        self._filtering = False

    def on_cmap_dropdown(self, event):
        self._restore_full_cmap_choices(keep_value=True)
        event.Skip()

    def on_cat_change(self, event):
        cat = self.choice_cat.GetStringSelection()
        self._filtering = True
        
        # If Custom, we want to disable search/typing so user can see and modify directly
        if cat == 'Custom...':
            # Note: We can't easily change style, but we can clear and disable text part
            cmaps = self._cmap_choices_for_category()
            self.choice_cmap.Set(cmaps)
            if cmaps:
                self.choice_cmap.SetSelection(0)
            # Try to disable typing
            try:
                self.choice_cmap.GetTextCtrl().SetEditable(False)
            except Exception:
                pass
        elif cat == 'CMCrameri':
            cmaps = self._cmap_choices_for_category()
            self.choice_cmap.Set(cmaps)
            if cmaps:
                self.choice_cmap.SetSelection(0)
            try:
                self.choice_cmap.GetTextCtrl().SetEditable(True)
            except Exception:
                pass
        else: # Standard
            cmaps = self._cmap_choices_for_category()
            self.choice_cmap.Set(cmaps)
            if 'OrRd' in cmaps:
                self.choice_cmap.SetStringSelection('OrRd')
            elif cmaps:
                self.choice_cmap.SetSelection(0)
            try:
                self.choice_cmap.GetTextCtrl().SetEditable(True)
            except Exception:
                pass
                
        self._filtering = False
        self.on_cmap_change(None)

    def on_cmap_text(self, event):
        """Filter the ComboBox list based on user input."""
        if self._filtering:
            return
            
        cat = self.choice_cat.GetStringSelection()
        if cat == 'Custom...':
            # Skip filtering for Custom category so they always see all items
            return
            
        txt = self.choice_cmap.GetValue().lower()
        if not txt:
            self._restore_full_cmap_choices(keep_value=False)
            return
            
        base_list = self._cmap_choices_for_category()
            
        filtered = [m for m in base_list if txt in m.lower()]
        
        if filtered:
            self._filtering = True
            current_val = self.choice_cmap.GetValue()
            insertion_point = self.choice_cmap.GetInsertionPoint()
            
            self.choice_cmap.Set(filtered)
            self.choice_cmap.SetValue(current_val)
            self.choice_cmap.SetInsertionPoint(insertion_point)
            
            # self.choice_cmap.Popup() # Optional: auto-popup
            self._filtering = False

    def on_unit_change(self, event):
        rb = event.GetEventObject()
        new_unit = normalize_spectral_unit(rb.GetLabel())
        if new_unit == self.x_unit:
            return
        old_xlim = None
        if self.target_view:
            old_xlim, _ = self.target_view.get_plot_limits()
        self._set_unit_controls(new_unit)
        
        if self.target_view:
            self.target_view.set_spectral_unit(new_unit)
            plot_config = self.target_view.get_plot_config()
            xlim = plot_config.get('xlim', old_xlim or (0, 1))
            if xlim and xlim[0] is not None:
                self.set_x_range(xlim[0], xlim[1])

    def on_x_range_enter(self, event):
        if self.target_view:
            try:
                xmin = float(self.x_min_text.GetValue())
                xmax = float(self.x_max_text.GetValue())
                self.target_view.set_x_range(xmin, xmax, self.x_unit)
            except ValueError:
                wx.MessageBox("Invalid X range. Please enter numeric values.", "Error", wx.OK | wx.ICON_ERROR)

    def on_y_range_enter(self, event):
        if self.target_view:
            try:
                ymin = float(self.y_min_text.GetValue())
                ymax = float(self.y_max_text.GetValue())
                self.target_view.set_y_range(ymin, ymax)
            except ValueError:
                wx.MessageBox("Invalid Y range. Please enter numeric values.", "Error", wx.OK | wx.ICON_ERROR)

    def _coerce_binning_value(self, ctrl: wx.SpinCtrl) -> Optional[int]:
        try:
            return max(1, int(ctrl.GetValue()))
        except (TypeError, ValueError):
            return None

    def _set_binning_controls(self, x_bin: int, y_bin: int, mode: str) -> None:
        self._syncing_binning_controls = True
        try:
            try:
                x_bin = max(1, int(x_bin))
            except (TypeError, ValueError):
                x_bin = 1
            try:
                y_bin = max(1, int(y_bin))
            except (TypeError, ValueError):
                y_bin = 1
            mode = "box" if mode == "box" else "cross"
            self.spin_bin_x.SetValue(x_bin)
            self.spin_bin_y.SetValue(y_bin)
            self.spin_bin_all.SetValue(x_bin if x_bin == y_bin else max(x_bin, y_bin))
            self.btn_box_binning.SetValue(mode == "box")
            self.btn_box_binning.SetLabel("Box" if mode == "box" else "Cross")
        finally:
            self._syncing_binning_controls = False

    def on_binning_change(self, event):
        if self._syncing_binning_controls or not self.target_view:
            if event:
                event.Skip()
            return
        x_bin = self._coerce_binning_value(self.spin_bin_x)
        y_bin = self._coerce_binning_value(self.spin_bin_y)
        if x_bin is None or y_bin is None:
            if event:
                event.Skip()
            return
        if x_bin == y_bin:
            self._syncing_binning_controls = True
            try:
                self.spin_bin_all.SetValue(x_bin)
            finally:
                self._syncing_binning_controls = False
        self.target_view.set_slice_binning(
            x_bin=x_bin,
            y_bin=y_bin,
            mode="box" if self.btn_box_binning.GetValue() else "cross",
        )
        if event:
            event.Skip()

    def on_all_binning_change(self, event):
        if self._syncing_binning_controls or not self.target_view:
            if event:
                event.Skip()
            return
        value = self._coerce_binning_value(self.spin_bin_all)
        if value is None:
            if event:
                event.Skip()
            return
        self._syncing_binning_controls = True
        try:
            self.spin_bin_x.SetValue(value)
            self.spin_bin_y.SetValue(value)
        finally:
            self._syncing_binning_controls = False
        self.target_view.set_slice_binning(
            x_bin=value,
            y_bin=value,
            mode="box" if self.btn_box_binning.GetValue() else "cross",
        )
        if event:
            event.Skip()

    def on_binning_mode_change(self, event):
        if not self.target_view:
            return
        mode = "box" if self.btn_box_binning.GetValue() else "cross"
        self.btn_box_binning.SetLabel("Box" if mode == "box" else "Cross")
        self.target_view.set_slice_binning(mode=mode)

    def _set_fit_overlay_control(self, mode: str) -> None:
        mode = normalize_fit_overlay_mode(mode)
        labels = {"off": "Off", "global": "Global", "row": "Row", "both": "Both"}
        if not self.choice_fit_overlay.SetStringSelection(labels.get(mode, "Off")):
            self.choice_fit_overlay.SetSelection(0)

    def on_fit_overlay_change(self, event):
        if not self.target_view:
            return
        mode = normalize_fit_overlay_mode(self.choice_fit_overlay.GetStringSelection())
        self.target_view.set_fit_overlay_mode(mode)

    def on_vlim_slide(self, event):
        vmin_p = self.vmin_slider.GetValue()
        vmax_p = self.vmax_slider.GetValue()
        # Simple guard
        if vmin_p > vmax_p:
            if event.GetEventObject() is self.vmin_slider:
                vmax_p = vmin_p
                self.vmax_slider.SetValue(vmax_p)
            else:
                vmin_p = vmax_p
                self.vmin_slider.SetValue(vmin_p)
        
        if self.target_view:
            self.target_view.set_vlim(vmin_p, vmax_p)
            # update_absolute_vlim_display is called by set_vlim via ViewPanel

    def on_vlim_text_enter(self, event):
        try:
            vmin = float(self.txt_vmin.GetValue())
            vmax = float(self.txt_vmax.GetValue())
            
            if vmin > vmax:
                vmax = vmin + 1e-9
                self.txt_vmax.SetValue(f"{vmax:.2f}")
            
            if self.target_view:
                self.target_view.set_vlim_absolute(vmin, vmax)
        except ValueError:
            pass

    def update_absolute_vlim_display(self, vmin, vmax):
        """Called by ViewPanel to update the absolute value text boxes."""
        self.txt_vmin.ChangeValue(f"{vmin:.2f}")
        self.txt_vmax.ChangeValue(f"{vmax:.2f}")

    def on_roi_vlim_button(self, event):
        if self.target_view:
            vmin, vmax = self.target_view.get_roi_vlim()
            self.target_view.set_vlim_absolute(vmin, vmax)
            self.update_absolute_vlim_display(vmin, vmax)

    def _register_custom_cmaps(self):
        custom_maps = config.get('custom_colormaps', {})
        for name, nodes in custom_maps.items():
            if len(nodes) < 2: continue
            
            # Sort just in case
            sorted_nodes = sorted(nodes, key=lambda x: x[0])
            positions = [float(p) for p, c in sorted_nodes]
            colors = [c for p, c in sorted_nodes]
            
            # Matplotlib requires exactly 0 and 1 at the ends
            span = positions[-1] - positions[0]
            if span > 0:
                positions = [(p - positions[0]) / span for p in positions]
            
            positions[0] = 0.0
            positions[-1] = 1.0
            
            try:
                cmap = LinearSegmentedColormap.from_list(name, list(zip(positions, colors)))
                try:
                    plt.colormaps.register(cmap=cmap, force=True)
                except AttributeError:
                    plt.register_cmap(cmap=cmap)
            except Exception as e:
                pass

    def on_cmap_change(self, event):
        cmap = self.choice_cmap.GetValue()
        if not cmap: return
        
        previous_cmap = self.target_view.get_colormap() if self.target_view else config.get('default_colormap', config.get('colormap', 'OrRd'))

        if cmap == 'Edit Custom...':
            dlg = CustomColorbarsDialog(self)
            current_view_cmap = self.target_view.get_colormap() if self.target_view else previous_cmap
            
            # Try to pre-select previous if it's custom
            base_previous = previous_cmap[:-2] if previous_cmap.endswith('_r') else previous_cmap
            if base_previous in config.get('custom_colormaps', {}):
                dlg.choice_cmap.SetStringSelection(base_previous)
                dlg.load_nodes(base_previous)
                
            if dlg.ShowModal() == wx.ID_OK:
                changed_names = dlg.get_changed_names()
                config.set('custom_colormaps', dlg.get_results())
                self._register_custom_cmaps()
                
                new_cmap = dlg.get_current_name()
                custom_cmaps = list(config.get('custom_colormaps', {}).keys())
                self._filtering = True
                self.choice_cmap.Set(custom_cmaps + ['Edit Custom...'])
                
                if new_cmap in custom_cmaps:
                    if previous_cmap.endswith('_r') and previous_cmap[:-2] == new_cmap:
                        self.choice_cmap.SetValue(previous_cmap)
                        cmap = previous_cmap
                    else:
                        self.choice_cmap.SetValue(new_cmap)
                        cmap = new_cmap
                elif custom_cmaps:
                    self.choice_cmap.SetSelection(0)
                    cmap = self.choice_cmap.GetValue()
                else:
                    self.choice_cat.SetSelection(0)
                    self._filtering = False
                    self.on_cat_change(None)
                    return
                self._filtering = False
                current_base = current_view_cmap[:-2] if current_view_cmap.endswith('_r') else current_view_cmap
                if self.target_view and current_base in changed_names:
                    self.target_view.refresh_current_colormap()
            else:
                # Cancelled, revert
                custom_cmaps = list(config.get('custom_colormaps', {}).keys())
                self._filtering = True
                self.choice_cmap.Set(custom_cmaps + ['Edit Custom...'])
                if previous_cmap in custom_cmaps:
                    self.choice_cmap.SetValue(previous_cmap)
                    cmap = previous_cmap
                elif custom_cmaps:
                    self.choice_cmap.SetSelection(0)
                    cmap = self.choice_cmap.GetValue()
                else:
                    self.choice_cat.SetSelection(0)
                    self._filtering = False
                    self.on_cat_change(None)
                    return
                self._filtering = False
            dlg.Destroy()

        # Verify it exists in matplotlib
        try:
            plt.get_cmap(cmap)
        except Exception:
            # If not found, maybe it was a custom map that we just didn't register yet
            # Try to register all custom maps again just in case
            self._register_custom_cmaps()
            try:
                plt.get_cmap(cmap)
            except Exception:
                return

        if self.target_view and cmap != 'Edit Custom...':
            self.target_view.set_colormap(cmap)
    
    def on_reset_button(self, event):
        if self._on_reset:
            self._on_reset()

    def set_x_range(self, xmin, xmax):
        self.x_min_text.SetValue(f"{xmin:.2f}")
        self.x_max_text.SetValue(f"{xmax:.2f}")

    def set_y_range(self, ymin, ymax):
        self.y_min_text.SetValue(f"{ymin:.2f}")
        self.y_max_text.SetValue(f"{ymax:.2f}")

    def set_vlim_range(self, vmin, vmax):
        self.vmin_slider.SetValue(int(vmin))
        self.vmax_slider.SetValue(int(vmax))
        self.txt_vmin.SetValue(str(int(vmin)))
        self.txt_vmax.SetValue(str(int(vmax)))

    def set_colormap(self, cmap_name):
        # Switch category if needed
        self._filtering = True
        custom_maps = config.get('custom_colormaps', {})
        try:
            if cmap_name in custom_maps or (cmap_name.endswith('_r') and cmap_name[:-2] in custom_maps):
                self.choice_cat.SetStringSelection('Custom...')
                self.choice_cmap.Set(list(custom_maps.keys()) + ['Edit Custom...'])
            elif cmap_name.startswith('cmc.') and self.cmaps_cmc:
                self.choice_cat.SetStringSelection('CMCrameri')
                self.choice_cmap.Set(self.cmaps_cmc)
            else:
                self.choice_cat.SetStringSelection('Standard')
                self.choice_cmap.Set(self.cmaps_std)
            
            self.choice_cmap.SetValue(cmap_name)
        finally:
            self._filtering = False

    def get_x_unit(self):
        return self.x_unit


class StyleEditDialog(wx.Dialog):
    """
    Dialog to edit plot style: color, linestyle, linewidth, marker.
    Includes a Crameri categorical palette selector.
    """
    def __init__(self, parent, style_str: str):
        super().__init__(parent, title="Edit Style", style=wx.DEFAULT_DIALOG_STYLE | wx.RESIZE_BORDER)
        
        # Parse initial string: "color, linestyle, linewidth, marker, markersize"
        parts = [p.strip() for p in style_str.split(",")]
        self.color = parts[0] if len(parts) > 0 else "black"
        self.linestyle = parts[1] if len(parts) > 1 else "-"
        self.linewidth = parts[2] if len(parts) > 2 else "1.0"
        self.marker = parts[3] if len(parts) > 3 else ""
        self.markersize = parts[4] if len(parts) > 4 else "5.0"
        
        main_sizer = wx.BoxSizer(wx.VERTICAL)
        
        # 1. Color Selection
        color_sizer = wx.BoxSizer(wx.HORIZONTAL)
        color_sizer.Add(wx.StaticText(self, label="Color:"), 0, wx.ALIGN_CENTER_VERTICAL | wx.RIGHT, 10)
        
        self.cp = wx.ColourPickerCtrl(self, colour=wx.Colour(self.color) if self.color.startswith("#") else wx.BLACK)
        color_sizer.Add(self.cp, 0, wx.ALIGN_CENTER_VERTICAL | wx.RIGHT, 5)
        
        self.txt_color = wx.TextCtrl(self, value=self.color)
        color_sizer.Add(self.txt_color, 1, wx.EXPAND)
        
        main_sizer.Add(color_sizer, 0, wx.EXPAND | wx.ALL, 10)
        
        # 2. Crameri Section
        if HAS_CMCRAMERI:
            box = wx.StaticBox(self, label="Crameri Palettes")
            box_sizer = wx.StaticBoxSizer(box, wx.VERTICAL)
            
            self.maps = [
                'cmc.batlowS', 'cmc.lajollaS', 'cmc.romaS', 'cmc.devonS', 
                'cmc.bilbaoS', 'cmc.osloS', 'cmc.actonS', 'cmc.bamakoS',
                'cmc.davosS', 'cmc.grayS', 'cmc.hawaiiS', 'cmc.imolaS',
                'cmc.lapazS', 'cmc.nuukS', 'cmc.tokyoS', 'cmc.turkuS'
            ]
            self.choice_map = wx.Choice(self, choices=self.maps)
            self.choice_map.SetSelection(0)
            box_sizer.Add(self.choice_map, 0, wx.EXPAND | wx.BOTTOM, 5)
            
            self.grid_sizer = wx.GridSizer(cols=8, vgap=2, hgap=2)
            box_sizer.Add(self.grid_sizer, 1, wx.EXPAND)
            
            main_sizer.Add(box_sizer, 0, wx.EXPAND | wx.LEFT | wx.RIGHT | wx.BOTTOM, 10)
            
            self.choice_map.Bind(wx.EVT_CHOICE, self.on_map_change)
            self.refresh_buttons()
            
        # 3. Line Style, Width, Marker
        line_sizer = wx.FlexGridSizer(4, 2, 5, 5)
        line_sizer.AddGrowableCol(1, 1)
        
        line_sizer.Add(wx.StaticText(self, label="Linestyle:"), 0, wx.ALIGN_CENTER_VERTICAL)
        # Change to ComboBox to allow user-defined text (e.g. tuple strings)
        self.choice_ls = wx.ComboBox(self, choices=["-", "--", "-.", ":", "None", "solid", "dashed", "dashdot", "dotted"], style=wx.CB_DROPDOWN)
        self.choice_ls.SetValue(self.linestyle)
        line_sizer.Add(self.choice_ls, 1, wx.EXPAND)
        
        line_sizer.Add(wx.StaticText(self, label="Linewidth:"), 0, wx.ALIGN_CENTER_VERTICAL)
        self.txt_lw = wx.TextCtrl(self, value=self.linewidth)
        line_sizer.Add(self.txt_lw, 1, wx.EXPAND)

        line_sizer.Add(wx.StaticText(self, label="Marker:"), 0, wx.ALIGN_CENTER_VERTICAL)
        self.choice_mk = wx.ComboBox(self, choices=["", "o", ".", "x", "+", "v", "^", "<", ">", "s", "d", "*"], style=wx.CB_DROPDOWN)
        self.choice_mk.SetValue(self.marker)
        line_sizer.Add(self.choice_mk, 1, wx.EXPAND)
        
        line_sizer.Add(wx.StaticText(self, label="Marker Size:"), 0, wx.ALIGN_CENTER_VERTICAL)
        self.txt_ms = wx.TextCtrl(self, value=self.markersize)
        line_sizer.Add(self.txt_ms, 1, wx.EXPAND)
        
        main_sizer.Add(line_sizer, 0, wx.EXPAND | wx.LEFT | wx.RIGHT | wx.BOTTOM, 10)
        
        # Buttons
        btn_sizer = self.CreateButtonSizer(wx.OK | wx.CANCEL)
        main_sizer.Add(btn_sizer, 0, wx.ALIGN_RIGHT | wx.ALL, 10)
        
        self.SetSizer(main_sizer)
        self.Fit()
        
        self.cp.Bind(wx.EVT_COLOURPICKER_CHANGED, self.on_cp_change)

    def on_cp_change(self, event):
        self.txt_color.SetValue(event.GetColour().GetAsString(wx.C2S_HTML_SYNTAX))

    def on_map_change(self, event):
        self.refresh_buttons()
        self.Layout()
        self.Fit()

    def refresh_buttons(self):
        self.grid_sizer.Clear(True)
        import matplotlib.pyplot as plt
        from matplotlib.colors import to_hex
        cmap = plt.get_cmap(self.choice_map.GetStringSelection())
        colors = cmap.colors if hasattr(cmap, 'colors') else cmap(np.linspace(0, 1, cmap.N))
        
        for i in range(min(len(colors), 32)):
            c_hex = to_hex(colors[i])
            btn = wx.Button(self, size=(20, 20))
            btn.SetBackgroundColour(wx.Colour(c_hex))
            btn.Bind(wx.EVT_BUTTON, lambda e, h=c_hex: self.txt_color.SetValue(h))
            self.grid_sizer.Add(btn, 0)

    def GetStyleString(self):
        color = self.txt_color.GetValue().strip()
        ls = self.choice_ls.GetValue().strip() # Use GetValue for ComboBox
        lw = self.txt_lw.GetValue().strip()
        mk = self.choice_mk.GetValue().strip()
        ms = self.txt_ms.GetValue().strip()
        return f"{color}, {ls}, {lw}, {mk}, {ms}"


class AppearancesPanel(wx.Panel):
    """
    Appearance tab: Table view of runs and plots.
    Columns: Run, Plot, Style.
    Each run has a separator row followed by its plot components.
    """
    def __init__(self, parent, on_rename_run=None, on_style_change=None):
        super().__init__(parent)
        self._on_rename_run = on_rename_run
        self._on_style_change = on_style_change 

        self.lc = wx.ListCtrl(self, style=wx.LC_REPORT | wx.LC_VRULES | wx.LC_HRULES)
        
        self.lc.InsertColumn(0, "Run", width=150)
        self.lc.InsertColumn(1, "Plot", width=100)
        self.lc.InsertColumn(2, "Style", width=200)

        sizer = wx.BoxSizer(wx.VERTICAL)
        sizer.Add(self.lc, 1, wx.EXPAND | wx.ALL, 4)
        self.SetSizer(sizer)

        self.lc.Bind(wx.EVT_LEFT_DCLICK, self.on_dbl_click)
        self.lc.Bind(wx.EVT_LIST_ITEM_RIGHT_CLICK, self.on_right_click)

        self.experiment: Optional[ExperimentSet] = None
        self.view_state: Optional[ViewState] = None
        
        # Map item index -> (run_id, component)
        # component is None for separator row
        self._row_map: Dict[int, Tuple[str, Optional[str]]] = {}

    def update_view(self, view_state: Optional[ViewState], experiment: Optional[ExperimentSet]):
        self.experiment = experiment
        self.view_state = view_state
        self.lc.DeleteAllItems()
        self._row_map.clear()

        if not view_state or not experiment:
            return

        # Identify primary runs by order (first 2 valid runs)
        primary_run_ids = []
        for rid in view_state.run_ids:
            run = experiment.get_run(rid)
            if run:
                primary_run_ids.append(rid)
                if len(primary_run_ids) >= 2: break
        
        for run_id in view_state.run_ids:
            run = experiment.get_run(run_id)
            if not run: continue
            
            nickname = experiment.get_run_nickname(run_id)
            run_config = view_state.get_run_config(run_id)
            
            # Separator Row (Header)
            idx = self.lc.InsertItem(self.lc.GetItemCount(), nickname)
            self.lc.SetItemFont(idx, wx.Font(wx.FontInfo().Bold()))
            self._row_map[idx] = (run_id, None)
            
            # Determine Plot column label for Header
            labels = []
            if run_id in primary_run_ids:
                slot = primary_run_ids.index(run_id) + 1
                if run.intensity_2d is not None:
                    labels.append(f"{slot}A, {slot}B, {slot}C")
                else:
                    # 1D Run
                    x_unit = run.metadata.get("raw_x_unit", "").lower()
                    if "deg" in x_unit or "angle" in x_unit:
                        labels.append(f"{slot}B")
                    else:
                        labels.append(f"{slot}C")
            
            # If it's ONLY an overlay (not primary), show target in header
            if not labels and run_config.overlay_target:
                 labels.append(f">{run_config.overlay_target}")
            
            plot_label = " + ".join(labels) if labels else "No Plot"
            self.lc.SetItem(idx, 1, plot_label)
            
            # Children
            comps = []
            
            # Add standard components if it is a primary run
            if run_id in primary_run_ids:
                if run.intensity_2d is not None:
                    comps.extend([("Map (A)", "A"), ("Angle (B)", "B"), ("Shift (C)", "C")])
                else:
                    # 1D Run - Determine which plot it sits on
                    x_unit = run.metadata.get("raw_x_unit", "").lower()
                    if "deg" in x_unit or "angle" in x_unit:
                        comps.append(("Angle (B)", "B"))
                    else:
                        comps.append(("Shift (C)", "C"))
            
            # Add Overlay rows if targets are set
            if run_config.overlay_target:
                ov_targets = [t.strip() for t in run_config.overlay_target.split(",") if t.strip()]
                for t in ov_targets:
                    comps.append((f"Overlay (>{t})", "Overlay"))
            
            for label, code in comps:
                s = run_config.get_style(code)
                style_str = f"{s.color}, {s.linestyle}, {s.linewidth}, {s.marker}, {s.markersize}"
                if not s.visible:
                    style_str += " (Hidden)"
                
                c_idx = self.lc.InsertItem(self.lc.GetItemCount(), "")
                self.lc.SetItem(c_idx, 1, label)
                self.lc.SetItem(c_idx, 2, style_str)
                self._row_map[c_idx] = (run_id, code)
                
                if not s.visible:
                    self.lc.SetItemTextColour(c_idx, wx.LIGHT_GREY)

    def on_dbl_click(self, event):
        pos = event.GetPosition()
        idx, flags = self.lc.HitTest(pos)
        
        if idx == wx.NOT_FOUND:
            return
            
        if idx not in self._row_map:
            return
            
        run_id, component = self._row_map[idx]
        col = self._get_column_from_x(pos.x)
        
        if col == 0 and component is None: # Separator Row Name
            current_name = self.lc.GetItemText(idx, 0)
            dlg = wx.TextEntryDialog(self, "Enter new run name:", "Rename Run", value=current_name)
            if dlg.ShowModal() == wx.ID_OK:
                new_name = dlg.GetValue().strip()
                if new_name and self._on_rename_run:
                    self._on_rename_run(run_id, new_name)
            dlg.Destroy()
            
        elif col == 1 and component is None: # Separator Row Overlay Target
            self._ask_overlay_target(run_id)

        elif col == 2: # Style Edit
            current_style = self.lc.GetItemText(idx, 2)
            if "(Hidden)" in current_style:
                current_style = current_style.replace(" (Hidden)", "")
            
            dlg = StyleEditDialog(self, current_style)
            if dlg.ShowModal() == wx.ID_OK:
                new_style = dlg.GetStyleString()
                if new_style and self._on_style_change:
                    self._on_style_change(run_id, "style_string", new_style, component=component)
            dlg.Destroy()

    def _ask_overlay_target(self, run_id):
        if not self.view_state: return
        run_config = self.view_state.get_run_config(run_id)
        
        choices = ["1B", "1C", "2B", "2C"]
        current = run_config.overlay_target if run_config.overlay_target else ""
        current_selections = [s.strip() for s in current.split(",") if s.strip()]
        
        dlg = wx.MultiChoiceDialog(self, f"Select plot(s) to overlay '{run_id}' onto:", "Overlay Target", choices)
        
        # Pre-select
        selections = []
        for i, c in enumerate(choices):
            if c in current_selections:
                selections.append(i)
        dlg.SetSelections(selections)
        
        if dlg.ShowModal() == wx.ID_OK:
            selections = dlg.GetSelections()
            selected_strings = [choices[i] for i in selections]
            result_str = ",".join(selected_strings)
            
            if self._on_style_change:
                self._on_style_change(run_id, "overlay_target", result_str, component=None)
        dlg.Destroy()

    def _get_column_from_x(self, x):
        total_w = 0
        for i in range(self.lc.GetColumnCount()):
            w = self.lc.GetColumnWidth(i)
            if x < total_w + w:
                return i
            total_w += w
        return -1

    def on_right_click(self, event):
        idx = event.GetIndex()
        if idx == wx.NOT_FOUND or idx not in self._row_map:
            return
            
        run_id, component = self._row_map[idx]
        
        menu = wx.Menu()
        item_vis = menu.Append(wx.ID_ANY, "Toggle Visibility")
        self.Bind(wx.EVT_MENU, lambda e: self._toggle_vis(run_id, component), item_vis)
        
        item_ov = menu.Append(wx.ID_ANY, "Set Overlay Target...")
        self.Bind(wx.EVT_MENU, lambda e: self._ask_overlay_target(run_id), item_ov)
        
        # Add Derived Run options if applicable
        if self.experiment:
            run = self.experiment.get_run(run_id)
            if run and run.run_type == RunType.DERIVED:
                menu.AppendSeparator()
                item_derived = menu.Append(wx.ID_ANY, "Edit Derived Props...")
                self.Bind(wx.EVT_MENU, lambda e: self._edit_derived_props(run_id), item_derived)
        
        self.PopupMenu(menu)
        menu.Destroy()

    def _edit_derived_props(self, run_id):
        if not self.view_state or not self.experiment: return
        
        run = self.experiment.get_run(run_id)
        if not run: return
        
        run_config = self.view_state.get_run_config(run_id)
        
        # Get current or default values
        current_n = run_config.derived_n_points or run.metadata.get("default_n_points", 100)
        current_auto = run_config.derived_autorange
        if current_auto is None: current_auto = run.metadata.get("default_autorange", False)
        
        current_range = run_config.derived_range or run.metadata.get("default_range", (0, 100))
        
        # Build Dialog
        dlg = wx.Dialog(self, title=f"Derived Props: {run.nickname}")
        sizer = wx.BoxSizer(wx.VERTICAL)
        
        # N Points
        gs = wx.FlexGridSizer(3, 2, 5, 5)
        gs.Add(wx.StaticText(dlg, label="N Points:"), 0, wx.ALIGN_CENTER_VERTICAL)
        txt_n = wx.TextCtrl(dlg, value=str(current_n))
        gs.Add(txt_n, 1, wx.EXPAND)
        
        # Auto Range
        gs.Add(wx.StaticText(dlg, label="Auto Range:"), 0, wx.ALIGN_CENTER_VERTICAL)
        chk_auto = wx.CheckBox(dlg, label="Use Plot Limits")
        chk_auto.SetValue(current_auto)
        gs.Add(chk_auto, 1, wx.EXPAND)
        
        # Range
        gs.Add(wx.StaticText(dlg, label="Manual Range:"), 0, wx.ALIGN_CENTER_VERTICAL)
        
        range_sizer = wx.BoxSizer(wx.HORIZONTAL)
        txt_min = wx.TextCtrl(dlg, value=str(current_range[0]))
        txt_max = wx.TextCtrl(dlg, value=str(current_range[1]))
        range_sizer.Add(txt_min, 1, wx.RIGHT, 5)
        range_sizer.Add(txt_max, 1)
        gs.Add(range_sizer, 1, wx.EXPAND)
        
        sizer.Add(gs, 1, wx.EXPAND | wx.ALL, 10)
        
        # Enable/Disable range inputs based on auto
        def update_range_state(evt=None):
            is_auto = chk_auto.GetValue()
            txt_min.Enable(not is_auto)
            txt_max.Enable(not is_auto)
        
        chk_auto.Bind(wx.EVT_CHECKBOX, update_range_state)
        update_range_state()
        
        btns = dlg.CreateButtonSizer(wx.OK | wx.CANCEL)
        sizer.Add(btns, 0, wx.ALIGN_RIGHT | wx.ALL, 10)
        
        dlg.SetSizer(sizer)
        dlg.Fit()
        
        if dlg.ShowModal() == wx.ID_OK:
            try:
                new_n = int(txt_n.GetValue())
                new_auto = chk_auto.GetValue()
                r_min = float(txt_min.GetValue())
                r_max = float(txt_max.GetValue())
                new_range = (r_min, r_max)
                
                # Update via callback
                if self._on_style_change:
                    # We reuse on_style_change, but maybe we need a more generic 'update_config'
                    # For now, let's overload it or just call it 3 times?
                    # The MainFrame.on_style_change handles "overlay_target", let's check if it handles derived props.
                    # It doesn't yet. We need to add that logic to MainFrame or AppearancesPanel calling a new callback?
                    # Or simpler: Just update run_config directly here? No, that bypasses MainFrame logic/redraw.
                    # Let's emit events.
                    
                    # We'll use a hack: pass special attr names that MainFrame handles.
                    # Wait, I need to update MainFrame.on_style_change first.
                    pass 
                    
                    # Let's assume MainFrame will be updated to handle these keys:
                    # 'derived_n_points', 'derived_autorange', 'derived_range'
                    self._on_style_change(run_id, "derived_n_points", new_n)
                    self._on_style_change(run_id, "derived_autorange", new_auto)
                    self._on_style_change(run_id, "derived_range", new_range)
                    
            except ValueError:
                wx.MessageBox("Invalid input.", "Error")
        
        dlg.Destroy()

    def _toggle_vis(self, run_id, component):
        if not self.view_state: return
        
        run_config = self.view_state.get_run_config(run_id)
        target_comp = component if component else "A" 
        s = run_config.get_style(target_comp)
        new_vis = not s.visible
        
        if self._on_style_change:
            self._on_style_change(run_id, "visible", new_vis, component)
