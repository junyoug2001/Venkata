import json
import os

import wx
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("WXAgg")
from matplotlib.backends.backend_wxagg import FigureCanvasWxAgg as FigureCanvas
from matplotlib.backends.backend_wxagg import NavigationToolbar2WxAgg as NavigationToolbar
from matplotlib.figure import Figure

from advanced_bg_json_generator import (
    CM1_PER_MEV,
    bridge_peak_region,
    common_grid,
    default_peak_center,
    fit_peak_for_window,
    fit_phi,
    map_quality,
    reconstruct_map,
    smooth_profile,
    solve_pol_profiles,
    spline_dict,
)


class SupervisedBgEngine:
    def __init__(self):
        self.xx_path = None
        self.yx_path = None
        self.unit = "meV"
        self.angles = None
        self.shifts = None
        self.xx = None
        self.yx = None
        self.phi = 0.0
        self.raw_profiles = {}
        self.cleaned_profiles = {}
        self.peak_fits = {}
        self.shared_center = None
        self.remove_half_width = None
        self.quality = {}

    def load(self, xx_path, yx_path, unit):
        self.xx_path = xx_path
        self.yx_path = yx_path
        self.unit = unit
        self.angles, self.shifts, self.xx, self.yx = common_grid(xx_path, yx_path)
        self.raw_profiles = {}
        self.cleaned_profiles = {}
        self.peak_fits = {}
        self.quality = {}

    def fit_b1g_phi(self, phi_initial, peak_center, peak_half_width, peak_weight):
        self.phi = fit_phi(self.angles, self.shifts, self.xx, self.yx, phi_initial, peak_center, peak_half_width, peak_weight)
        return self.phi

    def extract_profiles(self, smooth_window, smooth_poly):
        iso_xx, ang_xx = solve_pol_profiles(self.angles, self.xx, "parallel", self.phi)
        iso_yx, ang_yx = solve_pol_profiles(self.angles, self.yx, "cross", self.phi)
        self.raw_profiles = {
            "isotropic_xx": smooth_profile(iso_xx, smooth_window, smooth_poly),
            "isotropic_yx": smooth_profile(iso_yx, smooth_window, smooth_poly),
            "angular_xx": smooth_profile(ang_xx, smooth_window, smooth_poly),
            "angular_yx": smooth_profile(ang_yx, smooth_window, smooth_poly),
        }
        self.cleaned_profiles = {k: v.copy() for k, v in self.raw_profiles.items()}

    def locate_peak(self, peak_center, peak_half_width, gamma_guess):
        self.peak_fits = {}
        centers = []
        gammas = []
        for name, profile in self.raw_profiles.items():
            fit = fit_peak_for_window(self.shifts, profile, peak_center, peak_half_width, gamma_guess)
            self.peak_fits[name] = fit
            if fit and fit.get("success"):
                centers.append(fit["center"])
                gammas.append(abs(fit["gamma"]))
        self.shared_center = float(np.median(centers)) if centers else float(peak_center)
        return self.shared_center, gammas

    def bridge_peak(self, remove_half_width):
        self.remove_half_width = remove_half_width
        self.cleaned_profiles = {
            name: bridge_peak_region(self.shifts, profile, self.shared_center, remove_half_width)
            for name, profile in self.raw_profiles.items()
        }

    def evaluate(self):
        outside_peak = np.abs(self.shifts - self.shared_center) > self.remove_half_width
        rec_raw_xx = reconstruct_map(self.angles, self.raw_profiles["isotropic_xx"], self.raw_profiles["angular_xx"], "parallel", self.phi)
        rec_raw_yx = reconstruct_map(self.angles, self.raw_profiles["isotropic_yx"], self.raw_profiles["angular_yx"], "cross", self.phi)
        rec_clean_xx = reconstruct_map(self.angles, self.cleaned_profiles["isotropic_xx"], self.cleaned_profiles["angular_xx"], "parallel", self.phi)
        rec_clean_yx = reconstruct_map(self.angles, self.cleaned_profiles["isotropic_yx"], self.cleaned_profiles["angular_yx"], "cross", self.phi)
        self.quality = {
            "raw_profiles_vs_input_xx_all": map_quality(self.xx, rec_raw_xx, np.ones_like(self.shifts, dtype=bool)),
            "raw_profiles_vs_input_yx_all": map_quality(self.yx, rec_raw_yx, np.ones_like(self.shifts, dtype=bool)),
            "cleaned_profiles_vs_input_xx_outside_removed_peak": map_quality(self.xx, rec_clean_xx, outside_peak),
            "cleaned_profiles_vs_input_yx_outside_removed_peak": map_quality(self.yx, rec_clean_yx, outside_peak),
        }
        return self.quality

    def export_json(self, path, spline_smoothing=0.0):
        out = {
            "schema": "advanced_si_bg_v2",
            "unit": self.unit,
            "source": {"xx": os.path.abspath(self.xx_path), "yx": os.path.abspath(self.yx_path)},
            "b1g": {
                "phi_deg": float(self.phi),
                "basis_normalization": "mean_per_polarization",
                "fit_phi": True,
            },
            "removed_peak": {
                "nominal_center": float(default_peak_center(self.unit)),
                "shared_center": float(self.shared_center),
                "remove_half_width": float(self.remove_half_width),
                "method": "supervised_lorentzian_center_then_bridge_interpolation",
                "profiles": self.peak_fits,
            },
            "quality": {"reconstruction_checks": self.quality},
            "components": {
                name: spline_dict(self.shifts, profile, spline_smoothing)
                for name, profile in self.cleaned_profiles.items()
            },
        }
        with open(path, "w") as f:
            json.dump(out, f, indent=2)

    def export_profiles_csv(self, path):
        mask = np.abs(self.shifts - self.shared_center) <= self.remove_half_width
        out = {"Shift": self.shifts, "removed_peak_mask": mask.astype(int)}
        for name in ["isotropic_xx", "isotropic_yx", "angular_xx", "angular_yx"]:
            out[f"{name}_raw"] = self.raw_profiles[name]
            out[f"{name}_cleaned"] = self.cleaned_profiles[name]
        pd.DataFrame(out).to_csv(path, index=False)


class SupervisedBgFrame(wx.Frame):
    def __init__(self):
        super().__init__(None, title="Supervised Advanced BG JSON", size=(1300, 850))
        self.engine = SupervisedBgEngine()
        self.xx_path = None
        self.yx_path = None
        self._init_ui()

    def _init_ui(self):
        splitter = wx.SplitterWindow(self, style=wx.SP_LIVE_UPDATE | wx.SP_3D)
        left = wx.ScrolledWindow(splitter, style=wx.VSCROLL)
        left.SetScrollRate(0, 20)
        left_sizer = wx.BoxSizer(wx.VERTICAL)

        files_box = wx.StaticBoxSizer(wx.VERTICAL, left, "1. Input Files")
        btn_files = wx.Button(left, label="Select XX/YX BG CSVs")
        btn_files.Bind(wx.EVT_BUTTON, self.on_select_files)
        files_box.Add(btn_files, 0, wx.EXPAND | wx.ALL, 5)
        self.txt_xx = wx.TextCtrl(left, style=wx.TE_READONLY)
        self.txt_yx = wx.TextCtrl(left, style=wx.TE_READONLY)
        files_box.Add(wx.StaticText(left, label="XX:"), 0, wx.LEFT | wx.RIGHT, 5)
        files_box.Add(self.txt_xx, 0, wx.EXPAND | wx.ALL, 5)
        files_box.Add(wx.StaticText(left, label="YX:"), 0, wx.LEFT | wx.RIGHT, 5)
        files_box.Add(self.txt_yx, 0, wx.EXPAND | wx.ALL, 5)
        unit_row = wx.BoxSizer(wx.HORIZONTAL)
        unit_row.Add(wx.StaticText(left, label="Unit:"), 0, wx.ALIGN_CENTER_VERTICAL | wx.RIGHT, 5)
        self.choice_unit = wx.Choice(left, choices=["meV", "cm-1"])
        self.choice_unit.SetSelection(0)
        unit_row.Add(self.choice_unit, 1, wx.EXPAND)
        files_box.Add(unit_row, 0, wx.EXPAND | wx.ALL, 5)
        btn_load = wx.Button(left, label="Load Maps")
        btn_load.Bind(wx.EVT_BUTTON, self.on_load_maps)
        files_box.Add(btn_load, 0, wx.EXPAND | wx.ALL, 5)
        left_sizer.Add(files_box, 0, wx.EXPAND | wx.ALL, 5)

        step_box = wx.StaticBoxSizer(wx.VERTICAL, left, "2. Supervised Steps")
        grid = wx.FlexGridSizer(0, 2, 5, 5)
        grid.AddGrowableCol(1, 1)
        self.txt_phi_initial = wx.TextCtrl(left, value="0.0")
        self.txt_peak_center = wx.TextCtrl(left, value="64.4")
        self.txt_peak_fit_half_width = wx.TextCtrl(left, value="2.5")
        self.txt_peak_weight = wx.TextCtrl(left, value="0.02")
        self.txt_gamma_guess = wx.TextCtrl(left, value=f"{2.0 / CM1_PER_MEV:.6g}")
        self.txt_remove_half_width = wx.TextCtrl(left, value="2.5")
        self.spin_smooth = wx.SpinCtrl(left, min=1, max=101, initial=11)
        self.spin_poly = wx.SpinCtrl(left, min=1, max=5, initial=3)
        for label, ctrl in [
            ("Initial phi", self.txt_phi_initial),
            ("Peak center", self.txt_peak_center),
            ("Peak fit half-width", self.txt_peak_fit_half_width),
            ("Peak fit weight", self.txt_peak_weight),
            ("Gamma guess", self.txt_gamma_guess),
            ("Removal half-width", self.txt_remove_half_width),
            ("Smooth window", self.spin_smooth),
            ("Smooth poly", self.spin_poly),
        ]:
            grid.Add(wx.StaticText(left, label=label), 0, wx.ALIGN_CENTER_VERTICAL)
            grid.Add(ctrl, 1, wx.EXPAND)
        step_box.Add(grid, 0, wx.EXPAND | wx.ALL, 5)

        for label, handler in [
            ("Step 1: Fit Phi", self.on_fit_phi),
            ("Step 2: Extract Profiles", self.on_extract_profiles),
            ("Step 3: Locate 64.4 Peak", self.on_locate_peak),
            ("Step 4: Bridge Remove Peak", self.on_bridge_peak),
            ("Step 5: Check Reconstruction", self.on_check_reconstruction),
        ]:
            btn = wx.Button(left, label=label)
            btn.Bind(wx.EVT_BUTTON, handler)
            step_box.Add(btn, 0, wx.EXPAND | wx.ALL, 5)
        left_sizer.Add(step_box, 0, wx.EXPAND | wx.ALL, 5)

        export_box = wx.StaticBoxSizer(wx.VERTICAL, left, "3. Export")
        btn_export_json = wx.Button(left, label="Export JSON")
        btn_export_json.Bind(wx.EVT_BUTTON, self.on_export_json)
        btn_export_csv = wx.Button(left, label="Export Profile CSV")
        btn_export_csv.Bind(wx.EVT_BUTTON, self.on_export_profiles_csv)
        export_box.Add(btn_export_json, 0, wx.EXPAND | wx.ALL, 5)
        export_box.Add(btn_export_csv, 0, wx.EXPAND | wx.ALL, 5)
        left_sizer.Add(export_box, 0, wx.EXPAND | wx.ALL, 5)

        self.txt_status = wx.TextCtrl(left, style=wx.TE_MULTILINE | wx.TE_READONLY, size=(-1, 150))
        left_sizer.Add(self.txt_status, 0, wx.EXPAND | wx.ALL, 5)
        left.SetSizer(left_sizer)

        right = wx.Panel(splitter)
        right_sizer = wx.BoxSizer(wx.VERTICAL)
        self.nb = wx.Notebook(right)

        maps_panel = wx.Panel(self.nb)
        self.fig_maps = Figure()
        self.canvas_maps = FigureCanvas(maps_panel, -1, self.fig_maps)
        self.toolbar_maps = NavigationToolbar(self.canvas_maps)
        maps_sizer = wx.BoxSizer(wx.VERTICAL)
        maps_sizer.Add(self.toolbar_maps, 0, wx.EXPAND)
        maps_sizer.Add(self.canvas_maps, 1, wx.EXPAND)
        maps_panel.SetSizer(maps_sizer)

        prof_panel = wx.Panel(self.nb)
        self.fig_profiles = Figure()
        self.canvas_profiles = FigureCanvas(prof_panel, -1, self.fig_profiles)
        self.toolbar_profiles = NavigationToolbar(self.canvas_profiles)
        prof_sizer = wx.BoxSizer(wx.VERTICAL)
        prof_sizer.Add(self.toolbar_profiles, 0, wx.EXPAND)
        prof_sizer.Add(self.canvas_profiles, 1, wx.EXPAND)
        prof_panel.SetSizer(prof_sizer)

        quality_panel = wx.Panel(self.nb)
        self.fig_quality = Figure()
        self.canvas_quality = FigureCanvas(quality_panel, -1, self.fig_quality)
        self.toolbar_quality = NavigationToolbar(self.canvas_quality)
        quality_sizer = wx.BoxSizer(wx.VERTICAL)
        quality_sizer.Add(self.toolbar_quality, 0, wx.EXPAND)
        quality_sizer.Add(self.canvas_quality, 1, wx.EXPAND)
        quality_panel.SetSizer(quality_sizer)

        self.nb.AddPage(maps_panel, "Input Maps")
        self.nb.AddPage(prof_panel, "Profiles")
        self.nb.AddPage(quality_panel, "Reconstruction")
        right_sizer.Add(self.nb, 1, wx.EXPAND)
        right.SetSizer(right_sizer)

        splitter.SplitVertically(left, right, 360)
        self.CreateStatusBar()

    def log(self, msg):
        self.txt_status.AppendText(msg + "\n")
        self.SetStatusText(msg)

    def values(self):
        return {
            "phi_initial": float(self.txt_phi_initial.GetValue()),
            "peak_center": float(self.txt_peak_center.GetValue()),
            "peak_fit_half_width": float(self.txt_peak_fit_half_width.GetValue()),
            "peak_weight": float(self.txt_peak_weight.GetValue()),
            "gamma_guess": float(self.txt_gamma_guess.GetValue()),
            "remove_half_width": float(self.txt_remove_half_width.GetValue()),
            "smooth_window": int(self.spin_smooth.GetValue()),
            "smooth_poly": int(self.spin_poly.GetValue()),
        }

    def on_select_files(self, event):
        with wx.FileDialog(self, "Select XX/YX BG CSVs", wildcard="CSV files (*.csv)|*.csv", style=wx.FD_OPEN | wx.FD_MULTIPLE | wx.FD_FILE_MUST_EXIST) as dlg:
            if dlg.ShowModal() == wx.ID_CANCEL:
                return
            paths = dlg.GetPaths()
        if len(paths) != 2:
            wx.MessageBox("Select exactly two CSV files.", "Selection Error", wx.ICON_ERROR)
            return
        p0, p1 = paths
        lower0 = os.path.basename(p0).lower()
        lower1 = os.path.basename(p1).lower()
        if ("yx" in lower0 or "cross" in lower0 or "rl" in lower0) and not ("yx" in lower1 or "cross" in lower1 or "rl" in lower1):
            p0, p1 = p1, p0
        self.xx_path, self.yx_path = p0, p1
        self.txt_xx.SetValue(self.xx_path)
        self.txt_yx.SetValue(self.yx_path)

    def on_load_maps(self, event):
        if not self.xx_path or not self.yx_path:
            wx.MessageBox("Select input files first.", "Missing Files", wx.ICON_ERROR)
            return
        unit = self.choice_unit.GetStringSelection()
        self.engine.load(self.xx_path, self.yx_path, unit)
        center = default_peak_center(unit)
        self.txt_peak_center.SetValue(f"{center:.6g}")
        if unit == "cm-1":
            self.txt_peak_fit_half_width.SetValue("20")
            self.txt_remove_half_width.SetValue("20")
            self.txt_gamma_guess.SetValue("2")
        else:
            self.txt_peak_fit_half_width.SetValue("2.5")
            self.txt_remove_half_width.SetValue("2.5")
            self.txt_gamma_guess.SetValue(f"{2.0 / CM1_PER_MEV:.6g}")
        self.plot_maps()
        self.log(f"Loaded maps: {len(self.engine.angles)} angles, {len(self.engine.shifts)} shifts, unit={unit}")

    def on_fit_phi(self, event):
        if self.engine.xx is None:
            wx.MessageBox("Load maps first.", "Missing Step", wx.ICON_ERROR)
            return
        v = self.values()
        phi = self.engine.fit_b1g_phi(v["phi_initial"], v["peak_center"], v["peak_fit_half_width"], v["peak_weight"])
        self.txt_phi_initial.SetValue(f"{phi:.6g}")
        self.log(f"Fitted B1g phi = {phi:.4f} deg")

    def on_extract_profiles(self, event):
        if self.engine.xx is None:
            wx.MessageBox("Load maps first.", "Missing Step", wx.ICON_ERROR)
            return
        v = self.values()
        self.engine.extract_profiles(v["smooth_window"], v["smooth_poly"])
        self.plot_profiles()
        self.nb.SetSelection(1)
        self.log("Extracted four raw profiles.")

    def on_locate_peak(self, event):
        if not self.engine.raw_profiles:
            wx.MessageBox("Extract profiles first.", "Missing Step", wx.ICON_ERROR)
            return
        v = self.values()
        center, gammas = self.engine.locate_peak(v["peak_center"], v["peak_fit_half_width"], v["gamma_guess"])
        if gammas:
            suggested = max(v["peak_fit_half_width"], 6.0 * float(np.median(gammas)))
            self.txt_remove_half_width.SetValue(f"{suggested:.6g}")
        self.plot_profiles(show_fit=True)
        self.nb.SetSelection(1)
        ok = [k for k, fit in self.engine.peak_fits.items() if fit and fit.get("success")]
        self.log(f"Located shared peak center = {center:.4f}; successful profile fits: {', '.join(ok) if ok else 'none'}")

    def on_bridge_peak(self, event):
        if not self.engine.raw_profiles:
            wx.MessageBox("Extract profiles first.", "Missing Step", wx.ICON_ERROR)
            return
        v = self.values()
        if self.engine.shared_center is None:
            self.engine.shared_center = v["peak_center"]
        self.engine.bridge_peak(v["remove_half_width"])
        self.plot_profiles(show_fit=True)
        self.nb.SetSelection(1)
        self.log(f"Bridge-removed peak window: {self.engine.shared_center:.4f} +/- {v['remove_half_width']:.4f} {self.engine.unit}")

    def on_check_reconstruction(self, event):
        if not self.engine.cleaned_profiles or self.engine.shared_center is None or self.engine.remove_half_width is None:
            wx.MessageBox("Bridge-remove the peak first.", "Missing Step", wx.ICON_ERROR)
            return
        quality = self.engine.evaluate()
        self.plot_quality()
        self.nb.SetSelection(2)
        xx_rel = quality["cleaned_profiles_vs_input_xx_outside_removed_peak"]["relative_rmse"]
        yx_rel = quality["cleaned_profiles_vs_input_yx_outside_removed_peak"]["relative_rmse"]
        self.log(f"Outside removed peak relative RMSE: XX={xx_rel:.4g}, YX={yx_rel:.4g}")

    def plot_maps(self):
        self.fig_maps.clear()
        ax1 = self.fig_maps.add_subplot(121)
        ax2 = self.fig_maps.add_subplot(122, sharex=ax1, sharey=ax1)
        ax1.pcolormesh(self.engine.shifts, self.engine.angles, self.engine.xx, shading="auto", cmap="inferno")
        ax2.pcolormesh(self.engine.shifts, self.engine.angles, self.engine.yx, shading="auto", cmap="inferno")
        ax1.set_title("Input BG XX")
        ax2.set_title("Input BG YX")
        ax1.set_xlabel(f"Shift ({self.engine.unit})")
        ax2.set_xlabel(f"Shift ({self.engine.unit})")
        ax1.set_ylabel("Angle")
        self.fig_maps.tight_layout()
        self.canvas_maps.draw()

    def plot_profiles(self, show_fit=False):
        self.fig_profiles.clear()
        names = ["isotropic_xx", "isotropic_yx", "angular_xx", "angular_yx"]
        for i, name in enumerate(names, start=1):
            ax = self.fig_profiles.add_subplot(2, 2, i)
            if name in self.engine.raw_profiles:
                ax.plot(self.engine.shifts, self.engine.raw_profiles[name], color="0.35", lw=1.2, label="raw extracted")
            if name in self.engine.cleaned_profiles:
                ax.plot(self.engine.shifts, self.engine.cleaned_profiles[name], color="tab:red", lw=1.2, label="cleaned JSON")
            if self.engine.shared_center is not None:
                hw = self.values()["remove_half_width"]
                ax.axvspan(self.engine.shared_center - hw, self.engine.shared_center + hw, color="tab:orange", alpha=0.15)
                ax.axvline(self.engine.shared_center, color="tab:orange", lw=1)
            if show_fit and name in self.engine.peak_fits and self.engine.peak_fits[name]:
                fit = self.engine.peak_fits[name]
                if fit.get("success"):
                    ax.axvline(fit["center"], color="tab:blue", ls="--", lw=1)
                    ax.text(0.02, 0.94, f"x0={fit['center']:.3g}\ng={fit['gamma']:.3g}", transform=ax.transAxes, va="top")
            ax.set_title(name)
            ax.set_xlabel(f"Shift ({self.engine.unit})")
            ax.legend(fontsize=8)
        self.fig_profiles.tight_layout()
        self.canvas_profiles.draw()

    def plot_quality(self):
        self.fig_quality.clear()
        rec_xx = reconstruct_map(self.engine.angles, self.engine.cleaned_profiles["isotropic_xx"], self.engine.cleaned_profiles["angular_xx"], "parallel", self.engine.phi)
        rec_yx = reconstruct_map(self.engine.angles, self.engine.cleaned_profiles["isotropic_yx"], self.engine.cleaned_profiles["angular_yx"], "cross", self.engine.phi)
        panels = [
            ("XX input", self.engine.xx),
            ("XX cleaned rec", rec_xx),
            ("XX residual", self.engine.xx - rec_xx),
            ("YX input", self.engine.yx),
            ("YX cleaned rec", rec_yx),
            ("YX residual", self.engine.yx - rec_yx),
        ]
        for i, (title, data) in enumerate(panels, start=1):
            ax = self.fig_quality.add_subplot(2, 3, i)
            ax.pcolormesh(self.engine.shifts, self.engine.angles, data, shading="auto", cmap="inferno" if "residual" not in title else "coolwarm")
            if self.engine.shared_center is not None:
                ax.axvspan(self.engine.shared_center - self.engine.remove_half_width, self.engine.shared_center + self.engine.remove_half_width, color="white", alpha=0.2)
            ax.set_title(title)
            ax.set_xlabel(f"Shift ({self.engine.unit})")
            if i in [1, 4]:
                ax.set_ylabel("Angle")
        self.fig_quality.tight_layout()
        self.canvas_quality.draw()

    def on_export_json(self, event):
        if not self.engine.cleaned_profiles or self.engine.shared_center is None:
            wx.MessageBox("Create cleaned profiles before exporting.", "Missing Step", wx.ICON_ERROR)
            return
        if not self.engine.quality:
            self.engine.evaluate()
        with wx.FileDialog(self, "Save Advanced BG JSON", wildcard="JSON files (*.json)|*.json", style=wx.FD_SAVE | wx.FD_OVERWRITE_PROMPT, defaultFile="advanced_si_bg.json") as dlg:
            if dlg.ShowModal() == wx.ID_CANCEL:
                return
            path = dlg.GetPath()
        self.engine.export_json(path)
        self.log(f"Exported JSON: {path}")

    def on_export_profiles_csv(self, event):
        if not self.engine.cleaned_profiles or self.engine.shared_center is None:
            wx.MessageBox("Create cleaned profiles before exporting.", "Missing Step", wx.ICON_ERROR)
            return
        with wx.FileDialog(self, "Save Profile Diagnostic CSV", wildcard="CSV files (*.csv)|*.csv", style=wx.FD_SAVE | wx.FD_OVERWRITE_PROMPT, defaultFile="advanced_si_bg_profiles.csv") as dlg:
            if dlg.ShowModal() == wx.ID_CANCEL:
                return
            path = dlg.GetPath()
        self.engine.export_profiles_csv(path)
        self.log(f"Exported profile CSV: {path}")


if __name__ == "__main__":
    app = wx.App(False)
    frame = SupervisedBgFrame()
    frame.Show()
    app.MainLoop()
