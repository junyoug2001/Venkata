import wx
import wx.grid
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('WXAgg')
from matplotlib.backends.backend_wxagg import FigureCanvasWxAgg as FigureCanvas
from matplotlib.backends.backend_wxagg import NavigationToolbar2WxAgg as NavigationToolbar
from matplotlib.figure import Figure
from scipy.optimize import curve_fit
import re
import os
from scipy.interpolate import BSpline

# ==========================================
# 1. CORE LOGIC & PHYSICS
# ==========================================

def _deg2rad(x): return np.deg2rad(x)
def _abs2(x): return np.abs(x)**2

class SelectionRules:
    @staticmethod
    def D2h_Ag(theta, config, a, b, phi):
        th = _deg2rad(theta - phi)
        if config == 'parallel':
            return _abs2(a * np.cos(th)**2 + b * np.sin(th)**2)
        else:
            return _abs2(0.5 * (a - b) * np.sin(2 * th))

    @staticmethod
    def D2h_B1g(theta, config, d, phi):
        th = _deg2rad(theta - phi)
        if config == 'parallel':
            return _abs2(d * np.sin(2 * th))
        else:
            return _abs2(d * np.cos(2 * th))

    @staticmethod
    def D6h_A1g(theta, config, a):
        if config == 'parallel': return np.full_like(theta, a**2)
        return np.zeros_like(theta)

    @staticmethod
    def D6h_E2g(theta, config, d):
        return np.full_like(theta, d**2)

    @staticmethod
    def Linear_Background(x, theta, offset, slope_x, slope_theta):
        return offset + slope_x * x + slope_theta * theta

RULE_METADATA = {
    "D2h_Ag":  {"func": SelectionRules.D2h_Ag,  "params": ["a", "b", "phi"]},
    "D2h_B1g": {"func": SelectionRules.D2h_B1g, "params": ["d", "phi"]},
    "D6h_A1g": {"func": SelectionRules.D6h_A1g, "params": ["a"]},
    "D6h_E2g": {"func": SelectionRules.D6h_E2g, "params": ["d"]},
}

def lorentzian_normalized(x, x0, gamma):
    g = np.abs(gamma) + 1e-9
    return (1 / np.pi) * (g / ((x - x0)**2 + g**2))

def normalized_b1g_basis(theta, config, phi):
    raw = SelectionRules.D2h_B1g(theta, config, d=1.0, phi=phi)
    mean = np.mean(raw)
    if np.abs(mean) < 1e-12:
        return raw
    return raw / mean

class SafeBSpline:
    def __init__(self, t, c, k):
        self.bspline = BSpline(t, c, k)
        self.xmin = t[k]
        self.xmax = t[-k-1]

    def __call__(self, x):
        x_clipped = np.clip(x, self.xmin, self.xmax)
        return self.bspline(x_clipped)

# ==========================================
# 2. DATA & FITTING ENGINE
# ==========================================

class FittingEngine:
    def __init__(self):
        self.datasets = [
            {"label": "Parallel (XX)", "config": "parallel", "filename": "", "x": None, "ang": None, "z": None},
            {"label": "Cross (YX)",    "config": "cross",    "filename": "", "x": None, "ang": None, "z": None}
        ]
        self.peaks = [] 
        # Separate background for XX and YX
        self.bg_params = [
            {
                "offset": [0.0, -np.inf, np.inf],
                "slope_x": [0.0, -np.inf, np.inf],
                "slope_theta": [0.0, -np.inf, np.inf],
                "amp_si": [0.0, 0.0, np.inf],
                "amp_b1g_peak": [0.0, -np.inf, np.inf]
            }, # XX
            {
                "offset": [0.0, -np.inf, np.inf],
                "slope_x": [0.0, -np.inf, np.inf],
                "slope_theta": [0.0, -np.inf, np.inf],
                "amp_si": [0.0, 0.0, np.inf],
                "amp_b1g_peak": [0.0, -np.inf, np.inf]
            }  # YX
        ]
        self.x_min_limit = -np.inf
        self.x_max_limit = np.inf
        self.unit = "meV" # default when filenames do not identify the unit
        self.si_bg_unit = "meV"
        self.si_bg_mode = "none"
        self.si_bg_interp_xx = None
        self.si_bg_interp_yx = None
        self.si_bg_iso_xx = None
        self.si_bg_iso_yx = None
        self.si_bg_ang_xx = None
        self.si_bg_ang_yx = None
        self.si_bg_peak_source_center = None
        self.si_bg_peak_source_gamma = None
        self.si_bg_peak_ref_center = None
        self.si_bg_peak_params = {
            "x0": [64.4, 0.0, 5000.0],
            "gamma": [0.25, 0.001, 50.0],
            "phi": [0.0, -180.0, 180.0]
        }
        
    def load_si_bg(self, filepath):
        import json

        try:
            with open(filepath, 'r') as f:
                data = json.load(f)
                
            self.si_bg_unit = data.get('unit', 'meV')
            self.si_bg_mode = data.get("schema", "legacy")
            self.si_bg_interp_xx = None
            self.si_bg_interp_yx = None
            self.si_bg_iso_xx = None
            self.si_bg_iso_yx = None
            self.si_bg_ang_xx = None
            self.si_bg_ang_yx = None
            
            if self.si_bg_mode == "advanced_si_bg_v2":
                components = data.get("components", {})
                if "isotropic_xx" in components:
                    s = components["isotropic_xx"]
                    self.si_bg_iso_xx = SafeBSpline(s["t"], s["c"], s["k"])
                if "isotropic_yx" in components:
                    s = components["isotropic_yx"]
                    self.si_bg_iso_yx = SafeBSpline(s["t"], s["c"], s["k"])
                if "angular_xx" in components:
                    s = components["angular_xx"]
                    self.si_bg_ang_xx = SafeBSpline(s["t"], s["c"], s["k"])
                if "angular_yx" in components:
                    s = components["angular_yx"]
                    self.si_bg_ang_yx = SafeBSpline(s["t"], s["c"], s["k"])

                phi_guess = float(data.get("b1g", {}).get("phi_deg", 0.0))
                self.si_bg_peak_params["phi"][0] = phi_guess

                peak_meta = data.get("removed_peak", {})
                center = peak_meta.get("shared_center", peak_meta.get("nominal_center", None))
                gamma_guess = peak_meta.get("gamma_guess", None)
                if center is not None:
                    self.si_bg_peak_source_center = float(center)
                    self.si_bg_peak_ref_center = self.convert_between_units(float(center), self.si_bg_unit, self.unit)
                if gamma_guess is not None:
                    self.si_bg_peak_source_gamma = float(gamma_guess)
                self.refresh_si_peak_units()

                return True, f"Loaded advanced four-profile Si BG. (Unit: {self.si_bg_unit}, phi={phi_guess:.2f} deg)"

            if 'XX' in data:
                t, c, k = data['XX']['t'], data['XX']['c'], data['XX']['k']
                self.si_bg_interp_xx = SafeBSpline(t, c, k)
            if 'YX' in data:
                t, c, k = data['YX']['t'], data['YX']['c'], data['YX']['k']
                self.si_bg_interp_yx = SafeBSpline(t, c, k)
                
            return True, f"Loaded Si BG B-Spline profile successfully. (Unit: {self.si_bg_unit})"
        except Exception as e:
            return False, str(e)

    def _convert_bg_x(self, x_array):
        x_converted = np.copy(x_array)
        if self.unit == "meV" and self.si_bg_unit == "cm-1":
            x_converted = x_array * 8.065544
        elif self.unit == "cm-1" and self.si_bg_unit == "meV":
            x_converted = x_array / 8.065544
        return x_converted

    def refresh_si_peak_units(self):
        if self.si_bg_peak_source_center is not None:
            self.si_bg_peak_ref_center = self.convert_between_units(self.si_bg_peak_source_center, self.si_bg_unit, self.unit)
            self.si_bg_peak_params["x0"][0] = self.si_bg_peak_ref_center
        if self.si_bg_peak_source_gamma is not None:
            self.si_bg_peak_params["gamma"][0] = abs(self.convert_between_units(self.si_bg_peak_source_gamma, self.si_bg_unit, self.unit))

    @staticmethod
    def convert_between_units(value, from_unit, to_unit):
        if from_unit == to_unit:
            return value
        if from_unit == "cm-1" and to_unit == "meV":
            return value / 8.065544
        if from_unit == "meV" and to_unit == "cm-1":
            return value * 8.065544
        return value

    def get_si_profile_shift(self):
        if self.si_bg_peak_ref_center is None:
            return 0.0
        return self.si_bg_peak_params["x0"][0] - self.si_bg_peak_ref_center
            
    def evaluate_si_bg(self, x_array, config_mode):
        x_converted = self._convert_bg_x(x_array)
            
        if config_mode == "parallel" and self.si_bg_interp_xx is not None:
            return self.si_bg_interp_xx(x_converted)
        elif config_mode != "parallel" and self.si_bg_interp_yx is not None:
            return self.si_bg_interp_yx(x_converted)
        return np.zeros_like(x_array)

    def evaluate_advanced_si_bg(self, x_array, theta_array, config_mode, scale_bg, peak_amp, peak_x0, peak_gamma, phi):
        if self.si_bg_peak_ref_center is not None:
            profile_shift = peak_x0 - self.si_bg_peak_ref_center
        else:
            profile_shift = 0.0
        x_converted = self._convert_bg_x(x_array - profile_shift)
        iso = np.zeros_like(x_array, dtype=float)
        angular = np.zeros_like(x_array, dtype=float)
        if config_mode == "parallel":
            if self.si_bg_iso_xx is not None:
                iso = self.si_bg_iso_xx(x_converted)
            if self.si_bg_ang_xx is not None:
                angular = self.si_bg_ang_xx(x_converted) * normalized_b1g_basis(theta_array, config_mode, phi)
        else:
            if self.si_bg_iso_yx is not None:
                iso = self.si_bg_iso_yx(x_converted)
            if self.si_bg_ang_yx is not None:
                angular = self.si_bg_ang_yx(x_converted) * normalized_b1g_basis(theta_array, config_mode, phi)

        b1g_peak = normalized_b1g_basis(theta_array, config_mode, phi) * lorentzian_normalized(x_array, peak_x0, peak_gamma)
        return scale_bg * (iso + angular) + peak_amp * b1g_peak
        
    def load_files(self, filenames):
        if len(filenames) != 2:
            return False, "Please select exactly two files."
        
        f0 = filenames[0].lower()
        f1 = filenames[1].lower()
        
        # Detect unit. If filenames are ambiguous, use meV by default.
        if "mev" in f0 or "mev" in f1:
            self.unit = "meV"
        elif any(token in f0 or token in f1 for token in ["cm-1", "cm_1", "cm^-1", "wavenumber", "raman_shift"]):
            self.unit = "cm-1"
        else:
            self.unit = "meV"
        self.refresh_si_peak_units()
        
        swap = False
        if ("yx" in f0 or "cross" in f0) and not ("yx" in f1 or "cross" in f1): swap = True
        elif ("xx" in f1 or "para" in f1) and not ("xx" in f0 or "para" in f0): swap = True
            
        if swap: filenames = [filenames[1], filenames[0]]
            
        for i, fname in enumerate(filenames):
            try:
                df = pd.read_csv(fname, index_col=0)
                self.datasets[i]["filename"] = fname
                self.datasets[i]["ang"] = np.array(df.index.astype(float))
                self.datasets[i]["x"]   = np.array(df.columns.astype(float))
                self.datasets[i]["z"]   = df.values.astype(float)
            except Exception as e:
                return False, f"Error loading {fname}: {e}"
        
        self.x_min_limit = self.datasets[0]["x"].min()
        self.x_max_limit = self.datasets[0]["x"].max()
        
        return True, f"Loaded ({self.unit}):\n1. {os.path.basename(filenames[0])} (assumed XX)\n2. {os.path.basename(filenames[1])} (assumed YX)"

    def add_peak(self, name=None, rule_name="D2h_B1g", center=None):
        if name is None or name == "New Peak":
            name = rule_name
            auto_name = True
        else:
            auto_name = (name == rule_name)

        if center is None:
            if self.datasets[0]["x"] is not None:
                center = np.median(self.datasets[0]["x"])
            else:
                center = 300.0 if self.unit == "cm-1" else 30.0
            
        spec_params = { "x0": [center, 0.0, 5000.0], "gamma": [2.0 if self.unit == "cm-1" else 0.2, 0.01, 50.0] }
        ang_params = {}
        for p in RULE_METADATA[rule_name]["params"]:
            ang_params[p] = [10.0, -np.inf, np.inf]
            if p == "phi": ang_params[p] = [0.0, -180, 180]

        self.peaks.append({ "name": name, "rule": rule_name, "spec_params": spec_params, "ang_params": ang_params, "auto_name": auto_name })

    def get_mask(self, dataset_index):
        ds = self.datasets[dataset_index]
        if ds["x"] is None: return None
        return (ds["x"] >= self.x_min_limit) & (ds["x"] <= self.x_max_limit)

    def reconstruct(self, dataset_index):
        ds = self.datasets[dataset_index]
        if ds["z"] is None: return np.zeros((10, 10))
        XX, YY = np.meshgrid(ds["x"], ds["ang"])
        flat_params = self.flatten_params(which_val=0)
        z_flat = self._calc_single_config(XX.ravel(), YY.ravel(), ds["config"], flat_params)
        return z_flat.reshape(ds["z"].shape)

    def _calc_single_config(self, x_flat, theta_flat, config_mode, params):
        if self.si_bg_mode == "advanced_si_bg_v2":
            if config_mode == "parallel":
                bg_off, bg_slope_x, bg_slope_th = params[0], params[1], params[2]
                scale_bg, peak_amp = params[3], params[4]
            else:
                bg_off, bg_slope_x, bg_slope_th = params[5], params[6], params[7]
                scale_bg, peak_amp = params[8], params[9]
            intensity = SelectionRules.Linear_Background(x_flat, theta_flat, bg_off, bg_slope_x, bg_slope_th)
            peak_x0, peak_gamma, phi_bg = params[10], params[11], params[12]
            intensity += self.evaluate_advanced_si_bg(
                x_flat, theta_flat, config_mode, scale_bg, peak_amp, peak_x0, peak_gamma, phi_bg
            )
            idx = 13
        else:
            if config_mode == "parallel":
                bg_off, bg_slope_x, bg_slope_th, amp_si = params[0], params[1], params[2], params[3]
            else:
                bg_off, bg_slope_x, bg_slope_th, amp_si = params[4], params[5], params[6], params[7]
            intensity = SelectionRules.Linear_Background(x_flat, theta_flat, bg_off, bg_slope_x, bg_slope_th)
            intensity += amp_si * self.evaluate_si_bg(x_flat, config_mode)
            idx = 8

        for peak in self.peaks:
            x0, gamma = params[idx], params[idx+1]
            idx += 2
            rule_def = RULE_METADATA[peak["rule"]]
            n_ang = len(rule_def["params"])
            ang_p = params[idx : idx+n_ang]
            idx += n_ang
            I_val = rule_def["func"](theta_flat, config_mode, *ang_p)
            area = I_val / (np.abs(gamma) + 1e-9)
            intensity += area * lorentzian_normalized(x_flat, x0, gamma)
        return intensity

    def _joint_model_func(self, xy_tuple, *params):
        x1, th1, x2, th2 = xy_tuple
        z1 = self._calc_single_config(x1, th1, "parallel", params)
        z2 = self._calc_single_config(x2, th2, "cross", params)
        return np.concatenate([z1, z2])

    def flatten_params(self, which_val=0): 
        p = []
        if self.si_bg_mode == "advanced_si_bg_v2":
            p.extend([self.bg_params[0]["offset"][which_val], self.bg_params[0]["slope_x"][which_val], self.bg_params[0]["slope_theta"][which_val], self.bg_params[0]["amp_si"][which_val], self.bg_params[0]["amp_b1g_peak"][which_val]])
            p.extend([self.bg_params[1]["offset"][which_val], self.bg_params[1]["slope_x"][which_val], self.bg_params[1]["slope_theta"][which_val], self.bg_params[1]["amp_si"][which_val], self.bg_params[1]["amp_b1g_peak"][which_val]])
            p.extend([self.si_bg_peak_params["x0"][which_val], self.si_bg_peak_params["gamma"][which_val], self.si_bg_peak_params["phi"][which_val]])
        else:
            # Background XX
            p.extend([self.bg_params[0]["offset"][which_val], self.bg_params[0]["slope_x"][which_val], self.bg_params[0]["slope_theta"][which_val], self.bg_params[0]["amp_si"][which_val]])
            # Background YX
            p.extend([self.bg_params[1]["offset"][which_val], self.bg_params[1]["slope_x"][which_val], self.bg_params[1]["slope_theta"][which_val], self.bg_params[1]["amp_si"][which_val]])
        
        for peak in self.peaks:
            p.extend([peak["spec_params"]["x0"] [which_val], peak["spec_params"]["gamma"] [which_val]])
            for name in RULE_METADATA[peak["rule"]]["params"]:
                p.append(peak["ang_params"] [name] [which_val])
        return p

    def update_params_from_fit(self, popt):
        idx = 0
        if self.si_bg_mode == "advanced_si_bg_v2":
            self.bg_params[0]["offset"][0], self.bg_params[0]["slope_x"][0], self.bg_params[0]["slope_theta"][0], self.bg_params[0]["amp_si"][0], self.bg_params[0]["amp_b1g_peak"][0] = popt[0], popt[1], popt[2], popt[3], popt[4]
            self.bg_params[1]["offset"][0], self.bg_params[1]["slope_x"][0], self.bg_params[1]["slope_theta"][0], self.bg_params[1]["amp_si"][0], self.bg_params[1]["amp_b1g_peak"][0] = popt[5], popt[6], popt[7], popt[8], popt[9]
            self.si_bg_peak_params["x0"][0], self.si_bg_peak_params["gamma"][0], self.si_bg_peak_params["phi"][0] = popt[10], popt[11], popt[12]
            idx = 13
        else:
            self.bg_params[0]["offset"][0], self.bg_params[0]["slope_x"][0], self.bg_params[0]["slope_theta"][0], self.bg_params[0]["amp_si"][0] = popt[idx], popt[idx+1], popt[idx+2], popt[idx+3]
            self.bg_params[1]["offset"][0], self.bg_params[1]["slope_x"][0], self.bg_params[1]["slope_theta"][0], self.bg_params[1]["amp_si"][0] = popt[idx+4], popt[idx+5], popt[idx+6], popt[idx+7]
            idx = 8
        for peak in self.peaks:
            peak["spec_params"]["x0"] [0], peak["spec_params"]["gamma"] [0] = popt[idx], popt[idx+1]
            idx += 2
            for name in RULE_METADATA[peak["rule"]]["params"]:
                peak["ang_params"] [name] [0] = popt[idx]; idx += 1

    def sort_and_rename_peaks(self):
        self.peaks.sort(key=lambda p: p["spec_params"]["x0"][0])
        rule_groups = {}
        for p in self.peaks:
            rule_groups.setdefault(p["rule"], []).append(p)
            
        for rule, rule_peaks in rule_groups.items():
            if len(rule_peaks) > 1:
                idx = 1
                for p in rule_peaks:
                    if p.get("auto_name", False) or p["name"] == rule or re.match(rf"^{re.escape(rule)}\(\d+\)$", p["name"]):
                        p["name"] = f"{rule}({idx})"
                        p["auto_name"] = True
                        idx += 1
            else:
                for p in rule_peaks:
                    if p.get("auto_name", False) or re.match(rf"^{re.escape(rule)}\(\d+\)$", p["name"]):
                        p["name"] = rule
                        p["auto_name"] = True

    def run_optimization(self):
        if self.datasets[0]["z"] is None or self.datasets[1]["z"] is None:
            return False, "Data missing"
        
        d1 = self.datasets[0]
        XX1, YY1 = np.meshgrid(d1["x"], d1["ang"])
        mask1 = np.tile(self.get_mask(0), (len(d1["ang"]), 1)).ravel()
        x1_fit = XX1.ravel()[mask1]
        th1_fit = YY1.ravel()[mask1]
        z1_fit = d1["z"].ravel()[mask1]
        
        d2 = self.datasets[1]
        XX2, YY2 = np.meshgrid(d2["x"], d2["ang"])
        mask2 = np.tile(self.get_mask(1), (len(d2["ang"]), 1)).ravel()
        x2_fit = XX2.ravel()[mask2]
        th2_fit = YY2.ravel()[mask2]
        z2_fit = d2["z"].ravel()[mask2]
        
        if len(z1_fit) == 0 or len(z2_fit) == 0:
            return False, "No data points in range for one or both datasets."

        z_combined = np.concatenate([z1_fit, z2_fit])
        p0, lower, upper = self.flatten_params(0), self.flatten_params(1), self.flatten_params(2)
        
        try:
            popt, _ = curve_fit(self._joint_model_func, (x1_fit, th1_fit, x2_fit, th2_fit), z_combined, 
                                p0=p0, bounds=(lower, upper), maxfev=5000)
            self.update_params_from_fit(popt)
            self.sort_and_rename_peaks()
            return True, "Success"
        except Exception as e:
            return False, str(e)

    def export_global_fit_traces(self, folder):
        for ds_idx, ds in enumerate(self.datasets):
            if ds["z"] is None: continue
            
            rows_params = []
            headers = ["Angle", "BG_Const", "BG_Slope_X", "Amp_Si"]
            if self.si_bg_mode == "advanced_si_bg_v2":
                headers.extend(["Amp_B1g_Peak", "B1g_Center", "B1g_Gamma", "B1g_Phi", "Profile_X_Shift"])
            for i, p in enumerate(self.peaks):
                headers.extend([f"P{i+1}_{p['name']}_Area", f"P{i+1}_{p['name']}_Gamma", f"P{i+1}_{p['name']}_Height"])
            
            bg_off = self.bg_params[ds_idx]["offset"][0]
            bg_slope_x = self.bg_params[ds_idx]["slope_x"][0]
            bg_slope_th = self.bg_params[ds_idx]["slope_theta"][0]
            amp_si = self.bg_params[ds_idx]["amp_si"][0]
            
            for r in range(len(ds["ang"])):
                angle = ds["ang"][r]
                bg_const_eff = bg_off + bg_slope_th * angle
                row_res = [angle, bg_const_eff, bg_slope_x, amp_si]
                if self.si_bg_mode == "advanced_si_bg_v2":
                    row_res.extend([
                        self.bg_params[ds_idx]["amp_b1g_peak"][0],
                        self.si_bg_peak_params["x0"][0],
                        self.si_bg_peak_params["gamma"][0],
                        self.si_bg_peak_params["phi"][0],
                        self.get_si_profile_shift(),
                    ])
                
                for peak in self.peaks:
                    gamma = peak["spec_params"]["gamma"][0]
                    rule_def = RULE_METADATA[peak["rule"]]
                    ang_p = [peak["ang_params"][pn][0] for pn in rule_def["params"]]
                    
                    I_val = rule_def["func"](np.array([angle]), ds["config"], *ang_p)[0]
                    area = I_val / (np.abs(gamma) + 1e-9)
                    height = area * (1.0 / (np.pi * gamma))
                    
                    row_res.extend([area, gamma, height])
                rows_params.append(row_res)
            
            base_name = os.path.splitext(os.path.basename(ds["filename"]))[0]
            df_params = pd.DataFrame(rows_params, columns=headers)
            df_params.to_csv(os.path.join(folder, f"{base_name}_globalfit_params.csv"), index=False)

    def export_peak_reconstructions(self, folder):
        if self.datasets[0]["z"] is None: return False, "No Data"

        flat_params = self.flatten_params(which_val=0) 

        for ds_idx, ds in enumerate(self.datasets):
            if ds["z"] is None: continue
            
            XX, YY = np.meshgrid(ds["x"], ds["ang"])
            x_flat = XX.ravel()
            theta_flat = YY.ravel()
            
            # Background
            if self.si_bg_mode == "advanced_si_bg_v2":
                if ds["config"] == "parallel":
                    bg_off, bg_slope_x, bg_slope_th, amp_si = flat_params[0], flat_params[1], flat_params[2], flat_params[3]
                else:
                    bg_off, bg_slope_x, bg_slope_th, amp_si = flat_params[5], flat_params[6], flat_params[7], flat_params[8]
            else:
                if ds["config"] == "parallel":
                    bg_off, bg_slope_x, bg_slope_th, amp_si = flat_params[0], flat_params[1], flat_params[2], flat_params[3]
                else:
                    bg_off, bg_slope_x, bg_slope_th, amp_si = flat_params[4], flat_params[5], flat_params[6], flat_params[7]
                
            z_bg_flat = SelectionRules.Linear_Background(x_flat, theta_flat, bg_off, bg_slope_x, bg_slope_th)
            if self.si_bg_mode == "advanced_si_bg_v2":
                z_bg_flat += self.evaluate_advanced_si_bg(
                    x_flat, theta_flat, ds["config"],
                    self.bg_params[ds_idx]["amp_si"][0],
                    self.bg_params[ds_idx]["amp_b1g_peak"][0],
                    self.si_bg_peak_params["x0"][0],
                    self.si_bg_peak_params["gamma"][0],
                    self.si_bg_peak_params["phi"][0]
                )
            else:
                z_bg_flat += amp_si * self.evaluate_si_bg(x_flat, ds["config"])
            
            z_bg = z_bg_flat.reshape(ds["z"].shape)
            
            base_name = os.path.splitext(os.path.basename(ds["filename"]))[0]
            
            df_bg = pd.DataFrame(z_bg, index=ds["ang"], columns=ds["x"])
            df_bg.to_csv(os.path.join(folder, f"{base_name}_Background_reconstructed.csv"))

            current_idx = 13 if self.si_bg_mode == "advanced_si_bg_v2" else 8
            for i, peak in enumerate(self.peaks):
                x0 = flat_params[current_idx]
                gamma = flat_params[current_idx+1]
                current_idx += 2
                
                rule_def = RULE_METADATA[peak["rule"]]
                n_ang = len(rule_def["params"])
                ang_p = flat_params[current_idx : current_idx+n_ang]
                current_idx += n_ang
                
                I_val = rule_def["func"](theta_flat, ds["config"], *ang_p)
                area = I_val / (np.abs(gamma) + 1e-9)
                z_peak_flat = area * lorentzian_normalized(x_flat, x0, gamma)
                z_peak = z_peak_flat.reshape(ds["z"].shape)
                
                safe_name = "".join(x for x in peak["name"] if x.isalnum() or x in " _-")
                df_peak = pd.DataFrame(z_peak, index=ds["ang"], columns=ds["x"])
                df_peak.to_csv(os.path.join(folder, f"{base_name}_Peak{i+1}_{safe_name}_reconstructed.csv"))
        
        return True, "Peak-wise Export Complete"

    def export_background_subtracted(self, folder):
        if self.datasets[0]["z"] is None: return False, "No Data"

        flat_params = self.flatten_params(which_val=0)

        for ds in self.datasets:
            if ds["z"] is None: continue

            XX, YY = np.meshgrid(ds["x"], ds["ang"])
            x_flat = XX.ravel()
            theta_flat = YY.ravel()

            if self.si_bg_mode == "advanced_si_bg_v2":
                if ds["config"] == "parallel":
                    bg_off, bg_slope_x, bg_slope_th, amp_si = flat_params[0], flat_params[1], flat_params[2], flat_params[3]
                else:
                    bg_off, bg_slope_x, bg_slope_th, amp_si = flat_params[5], flat_params[6], flat_params[7], flat_params[8]
            else:
                if ds["config"] == "parallel":
                    bg_off, bg_slope_x, bg_slope_th, amp_si = flat_params[0], flat_params[1], flat_params[2], flat_params[3]
                else:
                    bg_off, bg_slope_x, bg_slope_th, amp_si = flat_params[4], flat_params[5], flat_params[6], flat_params[7]

            z_bg_flat = SelectionRules.Linear_Background(x_flat, theta_flat, bg_off, bg_slope_x, bg_slope_th)
            if self.si_bg_mode == "advanced_si_bg_v2":
                z_bg_flat += self.evaluate_advanced_si_bg(
                    x_flat, theta_flat, ds["config"],
                    self.bg_params[0 if ds["config"] == "parallel" else 1]["amp_si"][0],
                    self.bg_params[0 if ds["config"] == "parallel" else 1]["amp_b1g_peak"][0],
                    self.si_bg_peak_params["x0"][0],
                    self.si_bg_peak_params["gamma"][0],
                    self.si_bg_peak_params["phi"][0]
                )
            else:
                z_bg_flat += amp_si * self.evaluate_si_bg(x_flat, ds["config"])
            z_bg = z_bg_flat.reshape(ds["z"].shape)
            z_bg_subtracted = ds["z"] - z_bg

            base_name = os.path.splitext(os.path.basename(ds["filename"]))[0]
            df_bg_subtracted = pd.DataFrame(z_bg_subtracted, index=ds["ang"], columns=ds["x"])
            df_bg_subtracted.to_csv(os.path.join(folder, f"{base_name}_BG_subtracted.csv"))

        return True, "Background-subtracted CSV export complete"

    def generate_unpolarized_pdf(self, file_paths, x_min, x_max, output_path):
        raw_files = []
        recon_groups = {} # Key: Peak ID/Name, Value: list of files
        
        for f in file_paths:
            base = os.path.basename(f)
            if "_reconstructed.csv" in base:
                # Try to extract Peak ID
                match = re.search(r"_(Peak\d+|Background)[_.]", base)
                if match:
                    key = match.group(1)
                    if key not in recon_groups: recon_groups[key] = []
                    recon_groups[key].append(f)
                else:
                    if "Background" in base:
                        key = "Background"
                        if key not in recon_groups: recon_groups[key] = []
                        recon_groups[key].append(f)
                    else:
                        print(f"Skipping unrecognized reconstruction file: {base}")
            else:
                raw_files.append(f)
        
        if len(raw_files) != 2:
            return False, f"Expected 2 raw files, found {len(raw_files)}."
        
        # Load and Sum
        def sum_spectra(files, x_min, x_max):
            total_sum = None
            for f in files:
                try:
                    df = pd.read_csv(f, index_col=0)
                    cols = df.columns.astype(float)
                    mask = (cols >= x_min) & (cols <= x_max)
                    sub_df = df.loc[:, mask]
                    col_sum = sub_df.sum(axis=0)
                    if total_sum is None: total_sum = col_sum
                    else: total_sum = total_sum + col_sum
                except Exception as e:
                    print(f"Error processing {f}: {e}")
            return total_sum

        raw_sum = sum_spectra(raw_files, x_min, x_max)
        if raw_sum is None: return False, "Failed to process raw files."
        
        fig = Figure(figsize=(10, 6))
        ax = fig.add_subplot(111)
        
        ax.plot(raw_sum.index.astype(float), raw_sum.values, 'k-', lw=2, label='Raw (Unpolarized)')
        
        total_fit = None
        
        def sort_key(k):
            if "Background" in k: return (0, k)
            m = re.search(r"Peak(\d+)", k)
            if m: return (1, int(m.group(1)))
            return (2, k)
            
        sorted_keys = sorted(recon_groups.keys(), key=sort_key)
        
        for key in sorted_keys:
            files = recon_groups[key]
            comp_sum = sum_spectra(files, x_min, x_max)
            if comp_sum is not None:
                if total_fit is None: total_fit = comp_sum
                else: total_fit = total_fit + comp_sum
                
                lbl = key
                ax.plot(comp_sum.index.astype(float), comp_sum.values, label=lbl, alpha=0.8)
                if key == "Background":
                    ax.fill_between(comp_sum.index.astype(float), 0, comp_sum.values, alpha=0.1)
        
        if total_fit is not None:
            ax.plot(total_fit.index.astype(float), total_fit.values, 'r--', label='Total Fit')

        ax.set_xlabel(f"Wavenumber ({self.unit})")
        ax.set_ylabel("Integrated Intensity (Angle Sum)")
        ax.set_title("Unpolarized Raman Spectra")
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        try:
            fig.savefig(output_path)
            return True, f"Saved to {output_path}"
        except Exception as e:
            return False, str(e)

    def validate_row_by_row(self):
        if self.datasets[0]["z"] is None: return False, "No Data", None
        
        validation_results = []
        
        for ds_idx, ds in enumerate(self.datasets):
            rows_params = []
            rec_matrix = []
            
            config = ds["config"]
            x_full = ds["x"]
            mask = self.get_mask(ds_idx)
            x_fit = x_full[mask]
            
            headers = ["Angle", "BG_Const", "BG_Slope_X", "Amp_Si"]
            if self.si_bg_mode == "advanced_si_bg_v2":
                headers.extend(["Amp_B1g_Peak", "B1g_Center", "B1g_Gamma", "B1g_Phi", "Profile_X_Shift"])
            for i, p in enumerate(self.peaks):
                headers.extend([f"P{i+1}_{p['name']}_Area", f"P{i+1}_{p['name']}_Gamma", f"P{i+1}_{p['name']}_Height"])
            
            for r in range(len(ds["ang"])):
                angle = ds["ang"][r]
                z_row = ds["z"][r, :]
                z_fit_data = z_row[mask]
                
                if len(z_fit_data) == 0:
                    print(f"Row {r} (Angle {angle}): Empty data in range.")
                    bg_nan = [np.nan, np.nan, np.nan, np.nan]
                    if self.si_bg_mode == "advanced_si_bg_v2":
                        bg_nan.extend([np.nan, np.nan, np.nan, np.nan, np.nan])
                    rows_params.append(bg_nan + [np.nan]*(3*len(self.peaks)))
                    rec_matrix.append(np.zeros_like(x_full))
                    continue

                bg_const_guess = self.bg_params[ds_idx]["offset"][0] + self.bg_params[ds_idx]["slope_theta"][0] * angle
                bg_slope_x_guess = self.bg_params[ds_idx]["slope_x"][0]
                amp_si_guess = self.bg_params[ds_idx]["amp_si"][0]
                
                p0_row = [bg_const_guess, bg_slope_x_guess, amp_si_guess]
                bounds_low = [-np.inf, -np.inf, 0]
                bounds_high = [np.inf, np.inf, np.inf]
                
                for peak in self.peaks:
                    g_guess = peak["spec_params"]["gamma"][0]
                    g_min = peak["spec_params"]["gamma"][1]
                    g_max = peak["spec_params"]["gamma"][2]
                    
                    rule_def = RULE_METADATA[peak["rule"]]
                    ang_p = [peak["ang_params"][pn][0] for pn in rule_def["params"]]
                    I_val = rule_def["func"](np.array([angle]), config, *ang_p)[0]
                    area_guess = I_val / (np.abs(g_guess) + 1e-9)
                    
                    p0_row.extend([area_guess, g_guess])
                    bounds_low.extend([0, g_min])
                    bounds_high.extend([np.inf, g_max])
                
                def fit_func(x, *p):
                    y = p[0] + p[1] * x
                    amp_si_row = p[2]
                    if self.si_bg_mode == "advanced_si_bg_v2":
                        y += self.evaluate_advanced_si_bg(
                            x, np.full_like(x, angle, dtype=float), ds["config"],
                            self.bg_params[ds_idx]["amp_si"][0],
                            self.bg_params[ds_idx]["amp_b1g_peak"][0],
                            self.si_bg_peak_params["x0"][0],
                            self.si_bg_peak_params["gamma"][0],
                            self.si_bg_peak_params["phi"][0]
                        )
                    else:
                        y += amp_si_row * self.evaluate_si_bg(x, ds["config"])
                        
                    idx = 3
                    for k in range(len(self.peaks)):
                        area, gamma = p[idx], p[idx+1]
                        idx += 2
                        x0 = self.peaks[k]["spec_params"]["x0"][0] 
                        y += area * lorentzian_normalized(x, x0, gamma)
                    return y

                try:
                    popt, _ = curve_fit(fit_func, x_fit, z_fit_data, p0=p0_row, bounds=(bounds_low, bounds_high), maxfev=2000)
                    row_res = [angle, popt[0], popt[1], popt[2]]
                    if self.si_bg_mode == "advanced_si_bg_v2":
                        row_res.extend([
                            self.bg_params[ds_idx]["amp_b1g_peak"][0],
                            self.si_bg_peak_params["x0"][0],
                            self.si_bg_peak_params["gamma"][0],
                            self.si_bg_peak_params["phi"][0],
                            self.get_si_profile_shift(),
                        ])
                    idx = 3
                    for k in range(len(self.peaks)):
                        area, gamma = popt[idx], popt[idx+1]
                        idx += 2
                        height = area * (1.0 / (np.pi * gamma))
                        row_res.extend([area, gamma, height])
                    rows_params.append(row_res)
                    z_rec_row = fit_func(x_full, *popt)
                    rec_matrix.append(z_rec_row)
                except Exception as e:
                    print(f"Row {r} (Angle {angle}) fit failed: {e}")
                    row_res = [angle, np.nan, np.nan, np.nan]
                    if self.si_bg_mode == "advanced_si_bg_v2":
                        row_res.extend([np.nan, np.nan, np.nan, np.nan, np.nan])
                    row_res += [np.nan]*(3*len(self.peaks))
                    rows_params.append(row_res)
                    rec_matrix.append(np.zeros_like(x_full))
            
            validation_results.append({
                "label": ds["label"],
                "config": config,
                "x": ds["x"],
                "ang": ds["ang"],
                "z_raw": ds["z"],
                "z_rec": np.array(rec_matrix),
                "unit": self.unit,
                "rows_params": rows_params,
                "headers": headers
            })
            
        return True, "Validation Complete", validation_results

    def export_parameters(self, filename):
        with open(filename, 'w') as f: 
            f.write(f"# Joint Fit Export\n# Files: {os.path.basename(self.datasets[0]['filename'])}, {os.path.basename(self.datasets[1]['filename'])}\n")
            f.write(f"# Range: {self.x_min_limit} - {self.x_max_limit}\n")
            f.write(f"# Unit: {self.unit}\n")
            f.write(f"[Background XX]\nOffset: {self.bg_params[0]['offset'][0]} [{self.bg_params[0]['offset'][1]}, {self.bg_params[0]['offset'][2]}]\n")
            f.write(f"Slope_X: {self.bg_params[0]['slope_x'][0]} [{self.bg_params[0]['slope_x'][1]}, {self.bg_params[0]['slope_x'][2]}]\n")
            f.write(f"Slope_Theta: {self.bg_params[0]['slope_theta'][0]} [{self.bg_params[0]['slope_theta'][1]}, {self.bg_params[0]['slope_theta'][2]}]\n")
            f.write(f"Amp_Si: {self.bg_params[0]['amp_si'][0]} [{self.bg_params[0]['amp_si'][1]}, {self.bg_params[0]['amp_si'][2]}]\n")
            if self.si_bg_mode == "advanced_si_bg_v2":
                f.write(f"Amp_B1g_Peak: {self.bg_params[0]['amp_b1g_peak'][0]} [{self.bg_params[0]['amp_b1g_peak'][1]}, {self.bg_params[0]['amp_b1g_peak'][2]}]\n")
            f.write(f"[Background YX]\nOffset: {self.bg_params[1]['offset'][0]} [{self.bg_params[1]['offset'][1]}, {self.bg_params[1]['offset'][2]}]\n")
            f.write(f"Slope_X: {self.bg_params[1]['slope_x'][0]} [{self.bg_params[1]['slope_x'][1]}, {self.bg_params[1]['slope_x'][2]}]\n")
            f.write(f"Slope_Theta: {self.bg_params[1]['slope_theta'][0]} [{self.bg_params[1]['slope_theta'][1]}, {self.bg_params[1]['slope_theta'][2]}]\n")
            f.write(f"Amp_Si: {self.bg_params[1]['amp_si'][0]} [{self.bg_params[1]['amp_si'][1]}, {self.bg_params[1]['amp_si'][2]}]\n")
            if self.si_bg_mode == "advanced_si_bg_v2":
                f.write(f"Amp_B1g_Peak: {self.bg_params[1]['amp_b1g_peak'][0]} [{self.bg_params[1]['amp_b1g_peak'][1]}, {self.bg_params[1]['amp_b1g_peak'][2]}]\n\n")
            if self.si_bg_mode == "advanced_si_bg_v2":
                f.write("[Advanced Si BG]\n")
                f.write(f"B1g_Center: {self.si_bg_peak_params['x0'][0]} [{self.si_bg_peak_params['x0'][1]}, {self.si_bg_peak_params['x0'][2]}]\n")
                f.write(f"B1g_Gamma: {self.si_bg_peak_params['gamma'][0]} [{self.si_bg_peak_params['gamma'][1]}, {self.si_bg_peak_params['gamma'][2]}]\n")
                f.write(f"B1g_Phi: {self.si_bg_peak_params['phi'][0]} [{self.si_bg_peak_params['phi'][1]}, {self.si_bg_peak_params['phi'][2]}]\n")
                f.write(f"Profile_X_Shift: {self.get_si_profile_shift()}\n\n")
            for i, p in enumerate(self.peaks):
                f.write(f"[Peak {i+1}: {p['name']}]\nRule: {p['rule']}\n")
                f.write(f"Center (x0): {p['spec_params']['x0'][0]} [{p['spec_params']['x0'][1]}, {p['spec_params']['x0'][2]}]\n")
                f.write(f"Width (Gamma): {p['spec_params']['gamma'][0]} [{p['spec_params']['gamma'][1]}, {p['spec_params']['gamma'][2]}]\n")
                for k, v in p['ang_params'].items(): f.write(f"{k}: {v[0]} [{v[1]}, {v[2]}]\n")
                f.write("\n")

    def import_parameters(self, filename):
        try:
            with open(filename, 'r') as f: lines = f.readlines()
            self.peaks = []
            current = None
            for line in lines:
                line = line.strip()
                if not line or line.startswith("#"):
                    if "Range:" in line:
                        parts = line.split(":")[-1].split("-")
                        if len(parts) == 2: self.x_min_limit, self.x_max_limit = float(parts[0]), float(parts[1])
                    if "Unit:" in line:
                        self.unit = line.split(":")[-1].strip()
                    continue
                if "[Background XX]" in line: current = "BG0"
                elif "[Background YX]" in line: current = "BG1"
                elif "[Advanced Si BG]" in line:
                    current = "ADVBG"
                elif "[Peak" in line:
                    match = re.search(r": (.*) ]", line)
                    if not match: match = re.search(r": (.*) J", line)
                    name = match.group(1) if match else "Peak"
                    self.add_peak(name, "D2h_B1g")
                    current = self.peaks[-1]
                elif ":" in line:
                    k, v = [x.strip() for x in line.split(":", 1)]
                    
                    match_bounds = re.search(r"([-\d.eE]+)\s*\[([-\d.eE]+),\s*([-\d.eE]+)\]", v)
                    if match_bounds:
                        val, vmin, vmax = float(match_bounds.group(1)), float(match_bounds.group(2)), float(match_bounds.group(3))
                    else:
                        try:
                            val = float(v.split()[0])
                            vmin, vmax = -np.inf, np.inf
                        except: val = v; vmin, vmax = 0, 0
                        
                    if current == "BG0": 
                        self.bg_params[0][k.lower()] = [val, vmin, vmax]
                    elif current == "BG1":
                        self.bg_params[1][k.lower()] = [val, vmin, vmax]
                    elif current == "ADVBG":
                        key_map = {"B1g_Center": "x0", "B1g_Gamma": "gamma", "B1g_Phi": "phi"}
                        if k in key_map:
                            self.si_bg_peak_params[key_map[k]] = [val, vmin, vmax]
                    elif isinstance(current, dict):
                        if k == "Rule":
                            current["rule"] = val
                            current["ang_params"] = {pn: [10.0, -np.inf, np.inf] for pn in RULE_METADATA[val]["params"]}
                            # Ensure phi range if present
                            if "phi" in current["ang_params"]:
                                current["ang_params"]["phi"] = [current["ang_params"]["phi"][0], -180, 180]
                        elif k == "Center (x0)": current["spec_params"]["x0"] = [val, vmin, vmax]
                        elif k == "Width (Gamma)": current["spec_params"]["gamma"] = [val, vmin, vmax]
                        elif k in current["ang_params"]: current["ang_params"][k] = [val, vmin, vmax]
            return True, "Success"
        except Exception as e: return False, str(e)


# ==========================================
# 3. GUI
# ==========================================

class ParamsGrid(wx.grid.Grid):
    def __init__(self, parent):
        super().__init__(parent)
        self.CreateGrid(0, 4)
        for i, l in enumerate(["Param", "Value", "Min", "Max"]): self.SetColLabelValue(i, l)
        self.SetRowLabelSize(0)

    def load_data(self, labels, data_lists):
        if self.GetNumberRows() > 0: self.DeleteRows(0, self.GetNumberRows())
        for label, dlist in zip(labels, data_lists):
            row = self.GetNumberRows()
            self.InsertRows(row, 1)
            self.SetCellValue(row, 0, label)
            self.SetReadOnly(row, 0, True)
            for i in range(3): self.SetCellValue(row, i+1, f"{dlist[i]:.6g}")
            if label == "Profile_X_Shift":
                for i in range(3):
                    self.SetReadOnly(row, i+1, True)

    def save_to_lists(self, data_lists):
        for row in range(self.GetNumberRows()):
            for i in range(3):
                try: data_lists[row][i] = float(self.GetCellValue(row, i+1))
                except: pass

class ValidationFrame(wx.Frame):
    def __init__(self, results, engine, parent=None):
        super().__init__(parent, title="Row-by-Row Validation (Polar Areas)", size=(1200, 800))
        self.results = results
        self.engine = engine
        self.init_ui()
        self.update_plots()

    def init_ui(self):
        panel = wx.Panel(self)
        sizer = wx.BoxSizer(wx.VERTICAL)
        self.fig = Figure()
        self.canvas = FigureCanvas(panel, -1, self.fig)
        self.toolbar = NavigationToolbar(self.canvas)
        
        btn_export = wx.Button(panel, label="Export Areas")
        btn_export.Bind(wx.EVT_BUTTON, self.on_export)
        
        t_sizer = wx.BoxSizer(wx.HORIZONTAL)
        t_sizer.Add(self.toolbar, 1, wx.EXPAND)
        t_sizer.Add(btn_export, 0, wx.ALL|wx.ALIGN_CENTER_VERTICAL, 5)
        
        sizer.Add(t_sizer, 0, wx.EXPAND)
        sizer.Add(self.canvas, 1, wx.EXPAND)
        panel.SetSizer(sizer)

    def on_export(self, e):
        dlg = wx.DirDialog(self, "Choose Output Directory for Area Exports")
        if dlg.ShowModal() == wx.ID_OK:
            out_dir = dlg.GetPath()
            for ds_idx, res in enumerate(self.results):
                config = res["config"]
                label = res["label"].replace(" ", "_").replace("(", "").replace(")", "")
                df_params = pd.DataFrame(res["rows_params"], columns=res["headers"])
                angles = df_params["Angle"].values
                
                export_data = {"Angle": angles}
                
                for p_idx, peak in enumerate(self.engine.peaks):
                    area_col = f"P{p_idx+1}_{peak['name']}_Area"
                    if area_col in df_params.columns:
                        areas = df_params[area_col].values
                        export_data[f"{peak['name']}_RowFit_Area"] = areas
                        
                        gamma = peak["spec_params"]["gamma"][0]
                        rule_def = RULE_METADATA[peak["rule"]]
                        ang_p = [peak["ang_params"][pn][0] for pn in rule_def["params"]]
                        
                        I_val_global = rule_def["func"](angles, config, *ang_p)
                        area_global = I_val_global / (np.abs(gamma) + 1e-9)
                        export_data[f"{peak['name']}_GlobalFit_Area"] = area_global
                        
                        valid = ~np.isnan(areas) & np.isfinite(areas)
                        area_refit = np.full_like(areas, np.nan)
                        if np.sum(valid) > len(ang_p):
                            def fit_func(th_deg, *p):
                                return rule_def["func"](th_deg, config, *p)
                            
                            p0_scaled = list(ang_p)
                            scale_factor = 1.0 / np.sqrt(np.abs(gamma) + 1e-9) if gamma != 0 else 1.0
                            for i_param, param_name in enumerate(rule_def["params"]):
                                if param_name != "phi":
                                    p0_scaled[i_param] *= scale_factor
                            try:
                                popt, _ = curve_fit(fit_func, angles[valid], areas[valid], p0=p0_scaled, maxfev=5000)
                                area_refit = rule_def["func"](angles, config, *popt)
                            except:
                                pass
                        export_data[f"{peak['name']}_Refit_Area"] = area_refit
                
                df_export = pd.DataFrame(export_data)
                out_path = os.path.join(out_dir, f"Validation_Areas_{label}.csv")
                df_export.to_csv(out_path, index=False)
            wx.MessageBox(f"Exported successfully to:\n{out_dir}", "Export Complete")
        dlg.Destroy()

    def update_plots(self):
        self.fig.clear()
        
        num_peaks = len(self.engine.peaks)
        if num_peaks == 0:
            return
            
        num_datasets = len(self.results)
        
        gs = self.fig.add_gridspec(num_datasets, num_peaks, wspace=0.3, hspace=0.4)
        
        for ds_idx, res in enumerate(self.results):
            config = res["config"]
            df_params = pd.DataFrame(res["rows_params"], columns=res["headers"])
            angles = df_params["Angle"].values
            theta_rad = np.deg2rad(angles)
            
            for p_idx, peak in enumerate(self.engine.peaks):
                ax = self.fig.add_subplot(gs[ds_idx, p_idx], projection='polar')
                
                area_col = f"P{p_idx+1}_{peak['name']}_Area"
                if area_col in df_params.columns:
                    areas = df_params[area_col].values
                    valid = ~np.isnan(areas) & np.isfinite(areas)
                    ax.scatter(theta_rad[valid], areas[valid], label="Row Fit Area", color="black", s=15, zorder=3)
                    
                    gamma = peak["spec_params"]["gamma"][0]
                    rule_def = RULE_METADATA[peak["rule"]]
                    ang_p = [peak["ang_params"][pn][0] for pn in rule_def["params"]]
                    
                    theta_smooth = np.linspace(0, 360, 360)
                    th_smooth_rad = np.deg2rad(theta_smooth)
                    
                    I_val_global = rule_def["func"](theta_smooth, config, *ang_p)
                    area_global = I_val_global / (np.abs(gamma) + 1e-9)
                    ax.plot(th_smooth_rad, area_global, 'b--', label="Global Fit", zorder=2)
                    
                    if np.sum(valid) > len(ang_p):
                        def fit_func(th_deg, *p):
                            return rule_def["func"](th_deg, config, *p)
                        
                        # Guess parameters that absorb the gamma scale
                        p0_scaled = list(ang_p)
                        # The scaling factor is roughly sqrt(1/gamma) since I_val goes as a^2 or d^2
                        scale_factor = 1.0 / np.sqrt(np.abs(gamma) + 1e-9) if gamma != 0 else 1.0
                        for i_param, param_name in enumerate(rule_def["params"]):
                            if param_name != "phi":
                                p0_scaled[i_param] *= scale_factor
                                
                        try:
                            popt, _ = curve_fit(fit_func, angles[valid], areas[valid], p0=p0_scaled, maxfev=5000)
                            area_refit = rule_def["func"](theta_smooth, config, *popt)
                            ax.plot(th_smooth_rad, area_refit, 'r-', label="Refit Area", zorder=1)
                        except Exception as e:
                            print(f"Refit failed for {peak['name']} {config}: {e}")
                
                ax.set_title(f"{res['label']}\n{peak['name']}")
                if p_idx == 0 and ds_idx == 0:
                    ax.legend(bbox_to_anchor=(0, 1.15), loc='lower left', borderaxespad=0.)

        self.canvas.draw()

class MainFrame(wx.Frame):
    def __init__(self):
        super().__init__(None, title="Joint 2D Fitting (XX + YX)", size=(1600, 1000))
        self.engine = FittingEngine()
        self.last_sel = 0
        self.slice_angle_idx = 0
        self.slice_shift_idx = 0
        self.init_ui()
        
    def init_ui(self):
        main_splitter = wx.SplitterWindow(self)
        left_panel = wx.Panel(main_splitter)
        right_panel = wx.Panel(main_splitter)
        
        left_outer_sizer = wx.BoxSizer(wx.VERTICAL)
        left_inner_splitter = wx.SplitterWindow(left_panel, style=wx.SP_LIVE_UPDATE)
        left_top = wx.Panel(left_inner_splitter)
        left_bot = wx.Panel(left_inner_splitter)
        
        lt_sizer = wx.BoxSizer(wx.VERTICAL)
        
        f_box = wx.BoxSizer(wx.HORIZONTAL)
        for lbl, cb in [("Open 2 CSVs", self.on_open), ("Import Params", self.on_import), ("Load Si BG", self.on_load_si_bg)]:
            btn = wx.Button(left_top, label=lbl); btn.Bind(wx.EVT_BUTTON, cb)
            f_box.Add(btn, 1, wx.ALL, 2)
        lt_sizer.Add(f_box, 0, wx.EXPAND)

        range_box = wx.FlexGridSizer(2, 2, 5, 5)
        self.txt_min_x = wx.TextCtrl(left_top, value="", style=wx.TE_PROCESS_ENTER)
        self.txt_max_x = wx.TextCtrl(left_top, value="", style=wx.TE_PROCESS_ENTER)
        self.txt_min_x.Bind(wx.EVT_TEXT_ENTER, self.on_range_change)
        self.txt_max_x.Bind(wx.EVT_TEXT_ENTER, self.on_range_change)
        range_box.AddMany([wx.StaticText(left_top, label="Min X:"), self.txt_min_x, 
                           wx.StaticText(left_top, label="Max X:"), self.txt_max_x])
        lt_sizer.Add(range_box, 0, wx.ALL|wx.EXPAND, 5)

        self.peak_list = wx.ListBox(left_top)
        self.peak_list.Bind(wx.EVT_LISTBOX, self.on_peak_sel)
        self.peak_list.Bind(wx.EVT_LISTBOX_DCLICK, self.on_peak_rename_req)
        self.peak_list.Bind(wx.EVT_KEY_DOWN, self.on_peak_list_key)
        
        lt_sizer.Add(wx.StaticText(left_top, label="Peaks (Double-click/Enter/F2 to rename):"), 0, wx.ALL, 5)
        lt_sizer.Add(self.peak_list, 1, wx.EXPAND|wx.ALL, 5)
        
        p_box = wx.BoxSizer(wx.HORIZONTAL)
        for lbl, cb in [("Add Peak", self.on_add_peak), ("Remove", self.on_rem_peak)]:
            btn = wx.Button(left_top, label=lbl)
            btn.Bind(wx.EVT_BUTTON, cb)
            p_box.Add(btn, 1, wx.ALL, 2)
        lt_sizer.Add(p_box, 0, wx.EXPAND)

        # Peak Rename
        rename_box = wx.BoxSizer(wx.HORIZONTAL)
        rename_box.Add(wx.StaticText(left_top, label="Name:"), 0, wx.ALIGN_CENTER_VERTICAL|wx.ALL, 5)
        self.txt_peak_name = wx.TextCtrl(left_top)
        self.txt_peak_name.Bind(wx.EVT_TEXT, self.on_peak_name_change)
        rename_box.Add(self.txt_peak_name, 1, wx.EXPAND|wx.ALL, 5)
        lt_sizer.Add(rename_box, 0, wx.EXPAND)

        self.rule_combo = wx.ComboBox(left_top, choices=list(RULE_METADATA.keys()), style=wx.CB_READONLY)
        self.rule_combo.Bind(wx.EVT_COMBOBOX, self.on_rule_change)
        lt_sizer.Add(self.rule_combo, 0, wx.EXPAND|wx.ALL, 5)
        
        left_top.SetSizer(lt_sizer)

        lb_sizer = wx.BoxSizer(wx.VERTICAL)
        
        self.grid = ParamsGrid(left_bot)
        lb_sizer.Add(self.grid, 1, wx.EXPAND|wx.ALL, 5)

        v_box = wx.FlexGridSizer(2, 2, 5, 5)
        self.txt_vmin = wx.TextCtrl(left_bot, value="0", style=wx.TE_PROCESS_ENTER)
        self.txt_vmax = wx.TextCtrl(left_bot, value="1000", style=wx.TE_PROCESS_ENTER)
        self.txt_vmin.Bind(wx.EVT_TEXT_ENTER, self.on_viz); self.txt_vmax.Bind(wx.EVT_TEXT_ENTER, self.on_viz)
        v_box.AddMany([wx.StaticText(left_bot, label="VMin:"), self.txt_vmin, wx.StaticText(left_bot, label="VMax:"), self.txt_vmax])
        lb_sizer.Add(v_box, 0, wx.ALL, 5)

        act_box = wx.FlexGridSizer(4, 2, 5, 5)
        for lbl, cb in [("Preview", self.on_preview), ("FIT Global", self.on_fit), 
                        ("Validate (Row-by-Row)", self.on_validate), ("Export Params", self.on_export), 
                        ("Export Peak Maps", self.on_export_peak_maps), ("Export BG-Sub CSVs", self.on_export_bg_subtracted),
                        ("Export Unpolarized PDF", self.on_export_unpolar_pdf)]:
            btn = wx.Button(left_bot, label=lbl)
            btn.Bind(wx.EVT_BUTTON, cb)
            act_box.Add(btn, 1, wx.EXPAND)
        lb_sizer.Add(act_box, 0, wx.ALL|wx.EXPAND, 5)
        
        left_bot.SetSizer(lb_sizer)
        
        left_inner_splitter.SplitHorizontally(left_top, left_bot, 400)
        left_outer_sizer.Add(left_inner_splitter, 1, wx.EXPAND)
        left_panel.SetSizer(left_outer_sizer)

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

        main_splitter.SplitVertically(left_panel, right_panel, 350)
        self.refresh_ui()

    def on_load_si_bg(self, e):
        dlg = wx.FileDialog(self, "Load Si BG (B-Spline JSON)", wildcard="*.json", style=wx.FD_OPEN)
        if dlg.ShowModal() == wx.ID_OK:
            success, msg = self.engine.load_si_bg(dlg.GetPath())
            if success:
                wx.MessageBox("Si BG profile loaded successfully.")
                self.update_plots()
            else:
                wx.MessageBox(f"Failed to load Si BG: {msg}", "Error", wx.ICON_ERROR)

    def refresh_ui(self):
        curr = self.peak_list.GetSelection()
        self.peak_list.Clear()
        self.peak_list.Append("Background (XX)")
        self.peak_list.Append("Background (YX)")
        for i, p in enumerate(self.engine.peaks): self.peak_list.Append(f"{i+1}: {p['name']}")
        self.peak_list.SetSelection(curr if curr != -1 else 0)
        self.last_sel = self.peak_list.GetSelection()
        self.load_grid()
        self.txt_min_x.SetValue(f"{self.engine.x_min_limit:.6g}")
        self.txt_max_x.SetValue(f"{self.engine.x_max_limit:.6g}")

    def on_peak_sel(self, e):
        self.save_grid(self.last_sel)
        self.last_sel = self.peak_list.GetSelection()
        self.load_grid()

    def on_peak_name_change(self, e):
        sel = self.peak_list.GetSelection()
        if sel > 1:
            name = self.txt_peak_name.GetValue()
            self.engine.peaks[sel-2]["name"] = name
            self.engine.peaks[sel-2]["auto_name"] = False
            # Update listbox without full refresh if possible, or just refresh list labels
            self.peak_list.SetString(sel, f"{sel-1}: {name}")

    def on_peak_list_key(self, e):
        keycode = e.GetKeyCode()
        if keycode in [wx.WXK_RETURN, wx.WXK_NUMPAD_ENTER, wx.WXK_F2]:
            self.on_peak_rename_req(None)
        else:
            e.Skip()

    def on_peak_rename_req(self, e):
        sel = self.peak_list.GetSelection()
        if sel > 1:
            p = self.engine.peaks[sel-2]
            dlg = wx.TextEntryDialog(self, "Enter new name for peak:", "Rename Peak", p["name"])
            if dlg.ShowModal() == wx.ID_OK:
                new_name = dlg.GetValue()
                p["name"] = new_name
                p["auto_name"] = False
                self.peak_list.SetString(sel, f"{sel-1}: {new_name}")
                self.txt_peak_name.ChangeValue(new_name)
            dlg.Destroy()

    def load_grid(self):
        sel = self.peak_list.GetSelection()
        if sel == 0: # BG XX
            self.rule_combo.Disable()
            self.txt_peak_name.Disable()
            labels = ["Offset", "Slope_X", "Slope_Theta", "Amp_Si"]
            data = [self.engine.bg_params[0]["offset"], self.engine.bg_params[0]["slope_x"], self.engine.bg_params[0]["slope_theta"], self.engine.bg_params[0]["amp_si"]]
            if self.engine.si_bg_mode == "advanced_si_bg_v2":
                labels.extend(["Amp_B1g_Peak", "B1g_Center", "B1g_Gamma", "B1g_Phi", "Profile_X_Shift"])
                data.extend([self.engine.bg_params[0]["amp_b1g_peak"], self.engine.si_bg_peak_params["x0"], self.engine.si_bg_peak_params["gamma"], self.engine.si_bg_peak_params["phi"], [self.engine.get_si_profile_shift(), 0.0, 0.0]])
            self.grid.load_data(labels, data)
        elif sel == 1: # BG YX
            self.rule_combo.Disable()
            self.txt_peak_name.Disable()
            labels = ["Offset", "Slope_X", "Slope_Theta", "Amp_Si"]
            data = [self.engine.bg_params[1]["offset"], self.engine.bg_params[1]["slope_x"], self.engine.bg_params[1]["slope_theta"], self.engine.bg_params[1]["amp_si"]]
            if self.engine.si_bg_mode == "advanced_si_bg_v2":
                labels.extend(["Amp_B1g_Peak", "B1g_Center", "B1g_Gamma", "B1g_Phi", "Profile_X_Shift"])
                data.extend([self.engine.bg_params[1]["amp_b1g_peak"], self.engine.si_bg_peak_params["x0"], self.engine.si_bg_peak_params["gamma"], self.engine.si_bg_peak_params["phi"], [self.engine.get_si_profile_shift(), 0.0, 0.0]])
            self.grid.load_data(labels, data)
        else:
            self.rule_combo.Enable()
            self.txt_peak_name.Enable()
            p = self.engine.peaks[sel-2]
            self.txt_peak_name.ChangeValue(p["name"])
            self.rule_combo.SetValue(p["rule"])
            lbls = ["Center (x0)", "Width (G)"] + RULE_METADATA[p["rule"]]["params"]
            dst = [p["spec_params"]["x0"], p["spec_params"]["gamma"]] + [p["ang_params"].get(n, [0.0, -np.inf, np.inf]) for n in RULE_METADATA[p["rule"]]["params"]]
            self.grid.load_data(lbls, dst)

    def save_grid(self, idx):
        if idx == 0:
            data = [self.engine.bg_params[0]["offset"], self.engine.bg_params[0]["slope_x"], self.engine.bg_params[0]["slope_theta"], self.engine.bg_params[0]["amp_si"]]
            if self.engine.si_bg_mode == "advanced_si_bg_v2":
                data.extend([self.engine.bg_params[0]["amp_b1g_peak"], self.engine.si_bg_peak_params["x0"], self.engine.si_bg_peak_params["gamma"], self.engine.si_bg_peak_params["phi"], [self.engine.get_si_profile_shift(), 0.0, 0.0]])
            self.grid.save_to_lists(data)
        elif idx == 1:
            data = [self.engine.bg_params[1]["offset"], self.engine.bg_params[1]["slope_x"], self.engine.bg_params[1]["slope_theta"], self.engine.bg_params[1]["amp_si"]]
            if self.engine.si_bg_mode == "advanced_si_bg_v2":
                data.extend([self.engine.bg_params[1]["amp_b1g_peak"], self.engine.si_bg_peak_params["x0"], self.engine.si_bg_peak_params["gamma"], self.engine.si_bg_peak_params["phi"], [self.engine.get_si_profile_shift(), 0.0, 0.0]])
            self.grid.save_to_lists(data)
        elif idx > 1 and idx-2 < len(self.engine.peaks):
            p = self.engine.peaks[idx-2]
            self.grid.save_to_lists([p["spec_params"]["x0"], p["spec_params"]["gamma"]] + [p["ang_params"].get(n, [0.0, -np.inf, np.inf]) for n in RULE_METADATA[p["rule"]]["params"]])

    def on_add_peak(self, e): 
        # Guess center from current slice if possible
        center = None
        if self.engine.datasets[0]["x"] is not None:
            center = self.engine.datasets[0]["x"][self.slice_shift_idx]
        self.engine.add_peak(center=center)
        self.refresh_ui()
        self.peak_list.SetSelection(self.peak_list.GetCount()-1)
        self.on_peak_sel(None)

    def on_rem_peak(self, e):
        sel = self.peak_list.GetSelection()
        if sel > 1: self.engine.peaks.pop(sel-2); self.refresh_ui()

    def on_rule_change(self, e):
        sel = self.peak_list.GetSelection()
        if sel > 1:
            p = self.engine.peaks[sel-2]
            p["rule"] = self.rule_combo.GetValue()
            p["ang_params"] = {pn: [10.0, -np.inf, np.inf] for pn in RULE_METADATA[p["rule"]]["params"]}
            # Enforce phi range
            if "phi" in p["ang_params"]:
                p["ang_params"]["phi"] = [0.0, -180, 180]
            if p.get("auto_name", False):
                p["name"] = p["rule"]
                self.peak_list.SetString(sel, f"{sel-1}: {p['name']}")
                self.txt_peak_name.ChangeValue(p["name"])
            self.load_grid()

    def on_preview(self, e): self.save_grid(self.peak_list.GetSelection()); self.update_plots()
    def on_viz(self, e): self.update_plots()
    def on_range_change(self, e):
        try: self.engine.x_min_limit, self.engine.x_max_limit = float(self.txt_min_x.GetValue()), float(self.txt_max_x.GetValue()); self.update_plots()
        except: pass

    def on_fit(self, e):
        self.save_grid(self.peak_list.GetSelection())
        dlg = wx.ProgressDialog("Fitting", "Running Global Optimization...", style=wx.PD_APP_MODAL|wx.PD_ELAPSED_TIME)
        dlg.Pulse()
        success, msg = self.engine.run_optimization()
        dlg.Destroy()
        if success:
            self.refresh_ui()
            self.update_plots()
            wx.MessageBox("Global Fit Complete")
        else:
            wx.MessageBox(msg)

    def on_validate(self, e):
        prog = wx.ProgressDialog("Validating", "Running Row-by-Row Fit...", style=wx.PD_APP_MODAL|wx.PD_ELAPSED_TIME)
        prog.Pulse()
        success, msg, results = self.engine.validate_row_by_row()
        prog.Destroy()
        
        if success and results:
            vf = ValidationFrame(results, self.engine, self)
            vf.Show()
        else:
            wx.MessageBox(msg, "Validation Failed", wx.ICON_ERROR)

    def on_open(self, e):
        dlg = wx.FileDialog(self, "Select 2 CSV files", wildcard="*.csv", style=wx.FD_OPEN | wx.FD_MULTIPLE)
        if dlg.ShowModal() == wx.ID_OK:
            paths = dlg.GetPaths()
            success, msg = self.engine.load_files(paths)
            if success:
                self.slice_shift_idx = len(self.engine.datasets[0]["x"])//2 if self.engine.datasets[0]["x"] is not None else 0
                self.refresh_ui()
                self.update_plots()
                wx.MessageBox(msg, "Loaded")
            else:
                wx.MessageBox(msg, "Error", wx.ICON_ERROR)

    def on_import(self, e):
        dlg = wx.FileDialog(self, "Import Params", wildcard="*.txt", style=wx.FD_OPEN)
        if dlg.ShowModal() == wx.ID_OK:
            self.engine.import_parameters(dlg.GetPath()); self.refresh_ui(); self.update_plots()

    def on_export(self, e):
        dlg = wx.FileDialog(self, "Export Params", wildcard="*.txt", style=wx.FD_SAVE)
        if dlg.ShowModal() == wx.ID_OK: self.engine.export_parameters(dlg.GetPath())

    def on_export_peak_maps(self, e):
        dlg = wx.DirDialog(self, "Choose Output Directory for Peak Reconstructions")
        if dlg.ShowModal() == wx.ID_OK:
            success, msg = self.engine.export_peak_reconstructions(dlg.GetPath())
            wx.MessageBox(msg)

    def on_export_bg_subtracted(self, e):
        self.save_grid(self.peak_list.GetSelection())
        dlg = wx.DirDialog(self, "Choose Output Directory for Background-Subtracted CSVs")
        if dlg.ShowModal() == wx.ID_OK:
            success, msg = self.engine.export_background_subtracted(dlg.GetPath())
            if success:
                wx.MessageBox(msg)
            else:
                wx.MessageBox(msg, "Export Failed", wx.ICON_ERROR)

    def on_export_unpolar_pdf(self, e):
        dlg = wx.DirDialog(self, "Choose Directory containing Reconstructed CSVs")
        if dlg.ShowModal() == wx.ID_OK:
            folder = dlg.GetPath()
            files = [os.path.join(folder, f) for f in os.listdir(folder) if f.endswith(".csv")]
            raw_paths = [ds["filename"] for ds in self.engine.datasets if ds["filename"]]
            for rp in raw_paths:
                if rp not in files: files.append(rp)
            
            save_dlg = wx.FileDialog(self, "Save Unpolarized PDF", wildcard="*.pdf", style=wx.FD_SAVE)
            if save_dlg.ShowModal() == wx.ID_OK:
                success, msg = self.engine.generate_unpolarized_pdf(files, self.engine.x_min_limit, self.engine.x_max_limit, save_dlg.GetPath())
                wx.MessageBox(msg)

    def on_click(self, e):
        if self.toolbar.mode != '': return
        clicked_ax = e.inaxes
        ds_idx = -1
        if clicked_ax in [self.ax_raw1, self.ax_rec1]: ds_idx = 0
        elif clicked_ax in [self.ax_raw2, self.ax_rec2]: ds_idx = 1
        
        if ds_idx != -1:
            ds = self.engine.datasets[ds_idx]
            if ds["x"] is not None and e.xdata and e.ydata:
                self.slice_shift_idx = np.abs(ds["x"] - e.xdata).argmin()
                self.slice_angle_idx = np.abs(ds["ang"] - e.ydata).argmin()
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
            unit = self.engine.unit
            
            try: vmin, vmax = float(self.txt_vmin.GetValue()), float(self.txt_vmax.GetValue())
            except: vmin, vmax = np.nanpercentile(z_raw, [1, 99])
            
            ax_raw.clear(); ax_rec.clear()
            ax_raw.pcolormesh(x, ang, z_raw, vmin=vmin, vmax=vmax, cmap='inferno', shading='auto')
            ax_raw.set_title(f"{ds['label']} Raw")
            ax_raw.set_ylabel("Angle (°)")
            
            ax_rec.pcolormesh(x, ang, z_rec, vmin=vmin, vmax=vmax, cmap='inferno', shading='auto')
            ax_rec.set_title("Reconstruction")
            ax_rec.set_yticklabels([])
            
            for ax in [ax_raw, ax_rec]:
                if self.engine.x_min_limit > x.min(): ax.axvspan(x.min(), self.engine.x_min_limit, color='gray', alpha=0.5)
                if self.engine.x_max_limit < x.max(): ax.axvspan(self.engine.x_max_limit, x.max(), color='gray', alpha=0.5)
                
                if 0 <= self.slice_angle_idx < len(ang):
                    ax.axhline(ang[self.slice_angle_idx], color='w', ls='--', alpha=0.5)
                if 0 <= self.slice_shift_idx < len(x):
                    ax.axvline(x[self.slice_shift_idx], color='w', ls='--', alpha=0.5)

                trans = ax.get_xaxis_transform()
                for p in self.engine.peaks:
                    x0 = p["spec_params"]["x0"][0]
                    ax.plot(x0, 0, marker='^', color='red', markersize=8, transform=trans, clip_on=False)
                    ax.text(x0, -0.05, p["name"], color='red', ha='center', va='top', transform=trans, clip_on=False, fontsize=8)

            ax_spec.clear()
            if 0 <= self.slice_angle_idx < len(ang):
                cur_ang = ang[self.slice_angle_idx]
                ax_spec.plot(x, z_raw[self.slice_angle_idx,:], 'k', alpha=0.5)
                ax_spec.plot(x, z_rec[self.slice_angle_idx,:], 'r')
                ax_spec.set_title(f"Spec @ {cur_ang:.1f}°")
                ax_spec.set_xlabel(f"Shift ({unit})")
                if self.engine.x_min_limit > x.min(): ax_spec.axvspan(x.min(), self.engine.x_min_limit, color='gray', alpha=0.2)
                if self.engine.x_max_limit < x.max(): ax_spec.axvspan(self.engine.x_max_limit, x.max(), color='gray', alpha=0.2)

            ax_ang.clear()
            if 0 <= self.slice_shift_idx < len(x):
                cur_shift = x[self.slice_shift_idx]
                ax_ang.plot(ang, z_raw[:,self.slice_shift_idx], 'k', alpha=0.5)
                ax_ang.plot(ang, z_rec[:,self.slice_shift_idx], 'r')
                ax_ang.set_title(f"Ang @ {cur_shift:.1f} {unit}")
                ax_ang.set_xlabel("Angle (°)")

        plot_set(0, self.ax_raw1, self.ax_rec1, self.ax_spec1, self.ax_ang1)
        plot_set(1, self.ax_raw2, self.ax_rec2, self.ax_spec2, self.ax_ang2)
        
        if saved_xlim != (0.0, 1.0) and saved_ylim != (0.0, 1.0):
            self.ax_raw1.set_xlim(saved_xlim)
            self.ax_raw1.set_ylim(saved_ylim)
            self.ax_ang1.set_xlim(saved_ylim)
            self.ax_ang2.set_xlim(saved_ylim)
        
        self.canvas.draw()


if __name__ == "__main__":
    app = wx.App(); MainFrame().Show(); app.MainLoop()
