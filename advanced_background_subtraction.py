import wx
import numpy as np
from matplotlib.figure import Figure
from matplotlib.backends.backend_wxagg import FigureCanvasWxAgg as FigureCanvas
from matplotlib.backends.backend_wxagg import NavigationToolbar2WxAgg as NavigationToolbar
from typing import Dict, List, Tuple, Optional
from scipy.linalg import lstsq
from scipy.optimize import minimize
from scipy.signal import savgol_filter
import os
import copy
 
from data_structure import Run, parse_filename
from plotting import RamanPlotter2d

# -----------------------------------------------------------------------------
# Physics / Logic
# -----------------------------------------------------------------------------

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
    def Linear_Background(theta, offset, slope):
        return offset + slope * theta

class BackgroundReconstructor:
    def __init__(self):
        self.raw_xx: Optional[Run] = None
        self.raw_yx: Optional[Run] = None
        self.bg_xx: Optional[Run] = None
        self.bg_yx: Optional[Run] = None
        
        self.rec_bg_xx: Optional[np.ndarray] = None
        self.rec_bg_yx: Optional[np.ndarray] = None
        
        # Dictionary to store extracted spectral profiles: { 'A1g': array_1d, ... }
        self.profiles: Dict[str, np.ndarray] = {}
        self.raw_profiles: Dict[str, np.ndarray] = {}
        
        # Dictionary to store basis vectors for reconstruction: { 'A1g': (v_xx, v_yx), ... }
        self.basis_vectors: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
        
    def load_runs(self, raw_xx_path, raw_yx_path, bg_xx_path, bg_yx_path):
        try:
            self.raw_xx = Run.from_file(raw_xx_path)
            self.raw_yx = Run.from_file(raw_yx_path)
            self.bg_xx = Run.from_file(bg_xx_path)
            self.bg_yx = Run.from_file(bg_yx_path)
            return True, "Loaded successfully"
        except Exception as e:
            return False, str(e)

    def calculate_bg_scale_factor(self, shift_target: float, window_cm1: float = 20.0):
        """
        Calculates a scaling factor to match the reconstructed background peak height
        to the raw data peak height in a specified window.
        """
        if self.rec_bg_xx is None or self.raw_xx is None:
            raise ValueError("Both Raw XX data and reconstructed background must be available.")

        shifts = self.raw_xx.shift_cm1

        # Find window indices
        idx_start = np.argmin(np.abs(shifts - (shift_target - window_cm1 / 2)))
        idx_end = np.argmin(np.abs(shifts - (shift_target + window_cm1 / 2)))

        if idx_start >= idx_end:
            idx_start, idx_end = idx_end, idx_start # Ensure start < end
        if idx_start == idx_end: idx_end += 1

        # Find peak height in raw_xx (using mean spectrum in the window)
        mean_spec_raw = np.mean(self.raw_xx.intensity_2d, axis=0)
        peak_raw = np.max(mean_spec_raw[idx_start:idx_end])

        # Find peak height in reconstructed bg_xx
        mean_spec_rec_bg = np.mean(self.rec_bg_xx, axis=0)
        peak_rec_bg = np.max(mean_spec_rec_bg[idx_start:idx_end])
        if peak_rec_bg <= 1e-9:
            raise ValueError("Reconstructed background peak height is zero or negative. Cannot calculate scale factor.")

        scale_factor = peak_raw / peak_rec_bg
        return scale_factor

    def _get_basis_matrix(self, angles: np.ndarray, use_symmetries: List[str], phi: float, ag_ratio: float) -> Tuple[Optional[np.ndarray], List[str], Dict[str, Tuple[np.ndarray, np.ndarray]]]:
        """
        Constructs the basis matrix A for the linear least-squares fit.
        Returns (A, valid_symmetries, basis_vectors_unnormalized).
        """
        basis_cols = []
        basis_vectors = {}
        valid_symmetries = []

        def get_basis_vectors(sym_key):
            if sym_key == "D6h_A1g":
                vec_xx = SelectionRules.D6h_A1g(angles, 'parallel', a=1.0)
                vec_yx = SelectionRules.D6h_A1g(angles, 'cross', a=1.0)
                return vec_xx, vec_yx
            elif sym_key == "D6h_E2g":
                vec_xx = SelectionRules.D6h_E2g(angles, 'parallel', d=1.0)
                vec_yx = SelectionRules.D6h_E2g(angles, 'cross', d=1.0)
                return vec_xx, vec_yx
            elif sym_key == "D2h_B1g":
                vec_xx = SelectionRules.D2h_B1g(angles, 'parallel', d=1.0, phi=phi)
                vec_yx = SelectionRules.D2h_B1g(angles, 'cross', d=1.0, phi=phi)
                return vec_xx, vec_yx
            elif sym_key == "D2h_Ag":
                vec_xx = SelectionRules.D2h_Ag(angles, 'parallel', a=ag_ratio, b=1.0, phi=phi)
                vec_yx = SelectionRules.D2h_Ag(angles, 'cross', a=ag_ratio, b=1.0, phi=phi)
                return vec_xx, vec_yx
            return None, None

        for sym in use_symmetries:
            v_xx, v_yx = get_basis_vectors(sym)
            if v_xx is not None:
                basis_vectors[sym] = (v_xx.copy(), v_yx.copy())
                norm = np.max(np.abs(np.concatenate([v_xx, v_yx])))
                v_xx_norm = v_xx / norm if norm > 1e-9 else v_xx
                v_yx_norm = v_yx / norm if norm > 1e-9 else v_yx
                col = np.concatenate([v_xx_norm, v_yx_norm])
                basis_cols.append(col)
                valid_symmetries.append(sym)

        if not basis_cols:
            return None, [], {}

        A = np.column_stack(basis_cols)
        return A, valid_symmetries, basis_vectors

    def _objective_func(self, p_nonlinear: List[float], use_symmetries: List[str], param_map: List[str], initial_params: Dict[str, float]):
        """Objective function for non-linear optimization."""
        params = initial_params.copy()
        for name, val in zip(param_map, p_nonlinear):
            params[name] = val
        
        phi, ag_ratio = params['phi'], params['ratio']

        angles = self.bg_xx.angle_values
        A, _, _ = self._get_basis_matrix(angles, use_symmetries, phi, ag_ratio)

        if A is None:
            return np.inf

        I_xx = self.bg_xx.intensity_2d
        I_yx = self.bg_yx.intensity_2d
        Y = np.vstack([I_xx, I_yx])

        # Project Y onto the column space of A and find the sum of squared residuals
        C, residues, rank, s = lstsq(A, Y)

        if residues.size > 0:
            return np.sum(residues)
        else:
            # If the system is exactly determined, residues is empty.
            Y_rec = A @ C
            return np.sum((Y - Y_rec)**2)

    def reconstruct(self, use_symmetries: List[str], phi: float, ag_ratio: float, smooth_window: int = 11, smooth_poly: int = 3):
        """
        Decompose BG XX/YX into symmetry components and reconstruct smooth BGs.
        """
        phi_final, ag_ratio_final = self._perform_linear_fit(use_symmetries, phi, ag_ratio, smooth_window, smooth_poly)
        return phi_final, ag_ratio_final

    def reconstruct_with_fit(self, use_symmetries: List[str], phi_initial: float, ag_ratio_initial: float, smooth_window: int = 11, smooth_poly: int = 3):
        """
        Decompose BG XX/YX, fitting for phi and ag_ratio if necessary.
        """
        if not (self.bg_xx and self.bg_yx):
            raise ValueError("Background files not loaded")

        phi_final = phi_initial
        ag_ratio_final = ag_ratio_initial

        has_phi_param = any(s in ["D2h_B1g", "D2h_Ag"] for s in use_symmetries)
        has_ratio_param = "D2h_Ag" in use_symmetries

        if has_phi_param or has_ratio_param:
            initial_guess = []
            bounds = []
            param_map = []

            if has_phi_param:
                initial_guess.append(phi_initial)
                bounds.append((-180, 180))
                param_map.append('phi')
            if has_ratio_param:
                initial_guess.append(ag_ratio_initial)
                bounds.append((0.01, 100))
                param_map.append('ratio')

            initial_params = {'phi': phi_initial, 'ratio': ag_ratio_initial}

            result = minimize(
                self._objective_func,
                initial_guess,
                args=(use_symmetries, param_map, initial_params),
                bounds=bounds,
                method='L-BFGS-B'
            )

            if not result.success:
                print(f"Warning: Non-linear optimization may not have converged: {result.message}")

            final_params = initial_params.copy()
            for name, val in zip(param_map, result.x):
                final_params[name] = val
            
            phi_final, ag_ratio_final = final_params['phi'], final_params['ratio']

        return self._perform_linear_fit(use_symmetries, phi_final, ag_ratio_final, smooth_window, smooth_poly)

    def _perform_linear_fit(self, use_symmetries: List[str], phi: float, ag_ratio: float, smooth_window: int = 11, smooth_poly: int = 3):
        if not (self.bg_xx and self.bg_yx):
            raise ValueError("Background files not loaded")

        # Ensure compatible dimensions
        angles = self.bg_xx.angle_values
        shifts = self.bg_xx.shift_cm1
        
        # Check compatibility
        # For simplicity, we assume all runs assume same grid. 
        # In production, we should interpolate to a common grid.
        n_ang, n_shift = self.bg_xx.intensity_2d.shape

        A, valid_symmetries, self.basis_vectors = self._get_basis_matrix(angles, use_symmetries, phi, ag_ratio)
        if A is None:
            raise ValueError("No valid symmetries selected")
        
        # 2. Solve for profiles at each wavelength
        # Data Matrix Y: shape (2*n_ang, n_shift)
        I_xx = self.bg_xx.intensity_2d
        I_yx = self.bg_yx.intensity_2d
        
        Y = np.vstack([I_xx, I_yx])
        
        # Solve A * C = Y  => C = lstsq(A, Y)
        # C shape: (n_sym, n_shift)
        C, residues, rank, s = lstsq(A, Y)
        
        # 3. Store Profiles
        self.profiles = {}
        self.raw_profiles = {}
        for i, sym in enumerate(valid_symmetries):
            profile = C[i, :]
            self.raw_profiles[sym] = profile.copy()
            if smooth_window > 1:
                w = smooth_window if smooth_window % 2 == 1 else smooth_window + 1
                if w > len(profile): w = len(profile) if len(profile) % 2 == 1 else len(profile) - 1
                if w > smooth_poly:
                    profile = savgol_filter(profile, w, smooth_poly)
            self.profiles[sym] = profile
            
        # 4. Reconstruct Smoothed BG
        # Y_rec = A * C
        C_smoothed = np.array([self.profiles[sym] for sym in valid_symmetries])
        Y_rec = A @ C_smoothed # (2*n_ang, n_shift)
        
        self.rec_bg_xx = Y_rec[:n_ang, :]
        self.rec_bg_yx = Y_rec[n_ang:, :]

        return phi, ag_ratio
        
    def get_individual_component_map(self, sym_key: str):
        """
        Reconstructs the 2D map for a single symmetry component.
        Returns the physical profile and the XX/YX maps.
        """
        if sym_key not in self.profiles or sym_key not in self.basis_vectors:
            return None, None, None

        profile_scaled = self.profiles[sym_key]
        v_xx, v_yx = self.basis_vectors[sym_key]

        # Re-calculate the normalization factor used during the fit
        norm = np.max(np.abs(np.concatenate([v_xx, v_yx])))
        if norm < 1e-9: norm = 1.0

        v_xx_norm = v_xx / norm
        v_yx_norm = v_yx / norm

        map_xx = np.outer(v_xx_norm, profile_scaled)
        map_yx = np.outer(v_yx_norm, profile_scaled)
        physical_profile = profile_scaled / norm

        return physical_profile, map_xx, map_yx

    def get_subtracted(self, scale_factor: float):
        if self.rec_bg_xx is None or self.raw_xx is None:
            return None, None, None, None
        
        scaled_rec_bg_xx = self.rec_bg_xx * scale_factor
        scaled_rec_bg_yx = self.rec_bg_yx * scale_factor

        sub_xx = self.raw_xx.intensity_2d - scaled_rec_bg_xx
        sub_yx = self.raw_yx.intensity_2d - scaled_rec_bg_yx
        
        return sub_xx, sub_yx, scaled_rec_bg_xx, scaled_rec_bg_yx

# -----------------------------------------------------------------------------
# UI Components
# -----------------------------------------------------------------------------

class PlotPanel(wx.Panel):
    def __init__(self, parent, rows=1, cols=1):
        super().__init__(parent)
        self.figure = Figure()
        self.canvas = FigureCanvas(self, -1, self.figure)
        self.toolbar = NavigationToolbar(self.canvas)
        self.axes = self.figure.subplots(rows, cols, squeeze=False)
        self.rows = rows
        self.cols = cols
        
        sizer = wx.BoxSizer(wx.VERTICAL)
        sizer.Add(self.toolbar, 0, wx.EXPAND)
        sizer.Add(self.canvas, 1, wx.EXPAND)
        self.SetSizer(sizer)

    def get_ax(self, r, c):
        return self.axes[r, c]

    def draw(self, draw_now=True):
        self.figure.tight_layout()
        if draw_now:
            self.canvas.draw()

class MainFrame(wx.Frame):
    def __init__(self):
        super().__init__(None, title="Advanced Background Subtraction", size=(1200, 800))
        self.engine = BackgroundReconstructor()
        
        # Plotters
        self.plotters: Dict[str, RamanPlotter2d] = {}
        self.global_vmin = 0.0
        self.global_vmax = 1.0
        
        # File paths
        self.raw_xx_path: Optional[str] = None
        self.raw_yx_path: Optional[str] = None
        self.bg_xx_path: Optional[str] = None
        self.bg_yx_path: Optional[str] = None
        
        self._init_ui()
        
    def _init_ui(self):
        main_splitter = wx.SplitterWindow(self, style=wx.SP_LIVE_UPDATE | wx.SP_3D)
        
        # --- Left Control Panel (Scrollable) ---
        left_scrolled = wx.ScrolledWindow(main_splitter, style=wx.VSCROLL)
        left_scrolled.SetScrollRate(0, 20)
        left_scrolled_sizer = wx.BoxSizer(wx.VERTICAL)
        
        left_panel = wx.Panel(left_scrolled)
        left_sizer = wx.BoxSizer(wx.VERTICAL)
        
        # File Inputs
        fb_sizer = wx.StaticBoxSizer(wx.VERTICAL, left_panel, "1. Input Files")

        # Raw files
        raw_sizer = wx.BoxSizer(wx.HORIZONTAL)
        raw_sizer.Add(wx.StaticText(left_panel, label="Raw Files:"), 0, wx.ALIGN_CENTER_VERTICAL | wx.RIGHT, 5)
        btn_select_raw = wx.Button(left_panel, label="Select...")
        raw_sizer.Add(btn_select_raw, 1, wx.EXPAND)
        fb_sizer.Add(raw_sizer, 0, wx.EXPAND | wx.ALL, 5)

        self.txt_raw_xx = wx.TextCtrl(left_panel, style=wx.TE_READONLY)
        self.txt_raw_yx = wx.TextCtrl(left_panel, style=wx.TE_READONLY)
        raw_grid = wx.FlexGridSizer(2, 2, 5, 5)
        raw_grid.Add(wx.StaticText(left_panel, label="  XX:"), 0, wx.ALIGN_CENTER_VERTICAL)
        raw_grid.Add(self.txt_raw_xx, 1, wx.EXPAND)
        raw_grid.Add(wx.StaticText(left_panel, label="  YX:"), 0, wx.ALIGN_CENTER_VERTICAL)
        raw_grid.Add(self.txt_raw_yx, 1, wx.EXPAND)
        raw_grid.AddGrowableCol(1, 1)
        fb_sizer.Add(raw_grid, 0, wx.EXPAND | wx.ALL, 5)

        fb_sizer.Add(wx.StaticLine(left_panel), 0, wx.EXPAND | wx.ALL, 5)

        # BG files
        bg_sizer = wx.BoxSizer(wx.HORIZONTAL)
        bg_sizer.Add(wx.StaticText(left_panel, label="BG Files:"), 0, wx.ALIGN_CENTER_VERTICAL | wx.RIGHT, 5)
        btn_select_bg = wx.Button(left_panel, label="Select...")
        bg_sizer.Add(btn_select_bg, 1, wx.EXPAND)
        fb_sizer.Add(bg_sizer, 0, wx.EXPAND | wx.ALL, 5)

        self.txt_bg_xx = wx.TextCtrl(left_panel, style=wx.TE_READONLY)
        self.txt_bg_yx = wx.TextCtrl(left_panel, style=wx.TE_READONLY)
        bg_grid = wx.FlexGridSizer(2, 2, 5, 5)
        bg_grid.Add(wx.StaticText(left_panel, label="  XX:"), 0, wx.ALIGN_CENTER_VERTICAL)
        bg_grid.Add(self.txt_bg_xx, 1, wx.EXPAND)
        bg_grid.Add(wx.StaticText(left_panel, label="  YX:"), 0, wx.ALIGN_CENTER_VERTICAL)
        bg_grid.Add(self.txt_bg_yx, 1, wx.EXPAND)
        bg_grid.AddGrowableCol(1, 1)
        fb_sizer.Add(bg_grid, 0, wx.EXPAND | wx.ALL, 5)
        
        btn_load = wx.Button(left_panel, label="Load Data")
        fb_sizer.Add(btn_load, 0, wx.ALIGN_RIGHT | wx.ALL, 5)
        left_sizer.Add(fb_sizer, 0, wx.EXPAND | wx.ALL, 5)
        
        # Symmetries
        sym_sizer = wx.StaticBoxSizer(wx.VERTICAL, left_panel, "2. Reconstruction Model")
        
        self.cb_a1g = wx.CheckBox(left_panel, label="D6h A1g (Isotropic XX)")
        self.cb_a1g.SetValue(True)
        self.cb_e2g = wx.CheckBox(left_panel, label="D6h E2g (Unpolarized)")
        self.cb_e2g.SetValue(True)
        self.cb_b1g = wx.CheckBox(left_panel, label="D2h B1g (4-fold)")
        self.cb_ag  = wx.CheckBox(left_panel, label="D2h Ag (2-fold mixed)")
        
        sym_sizer.Add(self.cb_a1g, 0, wx.ALL, 2)
        sym_sizer.Add(self.cb_e2g, 0, wx.ALL, 2)
        sym_sizer.Add(self.cb_b1g, 0, wx.ALL, 2)
        sym_sizer.Add(self.cb_ag, 0, wx.ALL, 2)
        
        # Parameters
        param_grid = wx.FlexGridSizer(0, 2, 5, 5)
        param_grid.Add(wx.StaticText(left_panel, label="Phi (deg):"), 0, wx.ALIGN_CENTER_VERTICAL)
        self.spin_phi = wx.SpinCtrlDouble(left_panel, min=-180, max=180, initial=0, inc=1)
        param_grid.Add(self.spin_phi, 0, wx.EXPAND)
        
        param_grid.Add(wx.StaticText(left_panel, label="Ag Ratio (a/b):"), 0, wx.ALIGN_CENTER_VERTICAL)
        self.spin_ratio = wx.SpinCtrlDouble(left_panel, min=0.01, max=100, initial=1.0, inc=0.1)
        param_grid.Add(self.spin_ratio, 0, wx.EXPAND)
        
        self.cb_fit_params = wx.CheckBox(left_panel, label="Fit Phi & Ag Ratio")
        param_grid.Add(self.cb_fit_params, 0, wx.ALIGN_CENTER_VERTICAL | wx.TOP, 5)

        sym_sizer.Add(param_grid, 0, wx.EXPAND | wx.ALL, 5)
        
        sym_sizer.Add(wx.StaticLine(left_panel), 0, wx.EXPAND | wx.ALL, 5)

        smooth_sizer = wx.BoxSizer(wx.HORIZONTAL)
        smooth_sizer.Add(wx.StaticText(left_panel, label="Smoothing (window):"), 0, wx.ALIGN_CENTER_VERTICAL | wx.RIGHT, 5)
        self.spin_smooth = wx.SpinCtrl(left_panel, min=1, max=101, initial=11)
        smooth_sizer.Add(self.spin_smooth, 1, wx.EXPAND)
        sym_sizer.Add(smooth_sizer, 0, wx.EXPAND | wx.ALL, 5)

        sym_sizer.Add(wx.StaticLine(left_panel), 0, wx.EXPAND | wx.ALL, 5)
        
        viz_choice_sizer = wx.BoxSizer(wx.HORIZONTAL)
        viz_choice_sizer.Add(wx.StaticText(left_panel, label="Visualize Component:"), 0, wx.ALIGN_CENTER_VERTICAL | wx.RIGHT, 5)
        self.choice_sym_viz = wx.Choice(left_panel, choices=[])
        viz_choice_sizer.Add(self.choice_sym_viz, 1, wx.EXPAND)
        sym_sizer.Add(viz_choice_sizer, 0, wx.EXPAND | wx.ALL, 5)

        btn_rec = wx.Button(left_panel, label="Reconstruct Background")
        btn_rec.Bind(wx.EVT_BUTTON, self.on_reconstruct)
        sym_sizer.Add(btn_rec, 0, wx.EXPAND | wx.ALL, 5)
        
        left_sizer.Add(sym_sizer, 0, wx.EXPAND | wx.ALL, 5)
        
        # Scaling
        scale_sizer = wx.StaticBoxSizer(wx.VERTICAL, left_panel, "3. Scaling")
        norm_sizer = wx.BoxSizer(wx.HORIZONTAL)
        norm_sizer.Add(wx.StaticText(left_panel, label="Peak near:"), 0, wx.ALIGN_CENTER_VERTICAL | wx.RIGHT, 5)
        self.txt_norm_shift = wx.TextCtrl(left_panel, value="520", size=(60,-1))
        norm_sizer.Add(self.txt_norm_shift, 0, wx.ALIGN_CENTER_VERTICAL)
        norm_sizer.Add(wx.StaticText(left_panel, label="cm-1"), 0, wx.ALIGN_CENTER_VERTICAL | wx.LEFT, 2)
        self.btn_auto_scale = wx.Button(left_panel, label="Auto-Scale")
        norm_sizer.Add(self.btn_auto_scale, 0, wx.LEFT, 10)
        scale_sizer.Add(norm_sizer, 0, wx.EXPAND | wx.ALL, 5)

        scale_factor_sizer = wx.BoxSizer(wx.HORIZONTAL)
        scale_factor_sizer.Add(wx.StaticText(left_panel, label="Manual Scale Factor:"), 0, wx.ALIGN_CENTER_VERTICAL | wx.RIGHT, 5)
        self.spin_scale_factor = wx.SpinCtrlDouble(left_panel, min=0.01, max=100.0, initial=1.0, inc=0.01)
        scale_factor_sizer.Add(self.spin_scale_factor, 1, wx.EXPAND)
        scale_sizer.Add(scale_factor_sizer, 0, wx.EXPAND | wx.ALL, 5)

        left_sizer.Add(scale_sizer, 0, wx.EXPAND | wx.ALL, 5)

        # Visualization
        viz_box = wx.StaticBoxSizer(wx.VERTICAL, left_panel, "4. Visualization")
 
        # Colormap
        cmap_sizer = wx.BoxSizer(wx.HORIZONTAL)
        cmap_sizer.Add(wx.StaticText(left_panel, label="Colormap:"), 0, wx.ALIGN_CENTER_VERTICAL | wx.RIGHT, 5)
        self.choice_cmap = wx.Choice(left_panel, choices=['inferno', 'magma_r', 'cividis', 'coolwarm'])
        self.choice_cmap.SetSelection(0)
        cmap_sizer.Add(self.choice_cmap, 1, wx.EXPAND)
        viz_box.Add(cmap_sizer, 0, wx.EXPAND | wx.ALL, 5)

        # Contrast sliders
        contrast_sizer = wx.FlexGridSizer(2, 3, 5, 5)
        contrast_sizer.AddGrowableCol(1, 1)
        self.vmin_slider = wx.Slider(left_panel, value=0, minValue=0, maxValue=100)
        self.vmax_slider = wx.Slider(left_panel, value=100, minValue=0, maxValue=100)
        self.txt_vmin = wx.TextCtrl(left_panel, value="0.00", size=(60,-1), style=wx.TE_PROCESS_ENTER)
        self.txt_vmax = wx.TextCtrl(left_panel, value="1.00", size=(60,-1), style=wx.TE_PROCESS_ENTER)
        
        contrast_sizer.Add(wx.StaticText(left_panel, label="VMin:"), 0, wx.ALIGN_CENTER_VERTICAL)
        contrast_sizer.Add(self.vmin_slider, 1, wx.EXPAND)
        contrast_sizer.Add(self.txt_vmin, 0, wx.ALIGN_CENTER_VERTICAL)
        contrast_sizer.Add(wx.StaticText(left_panel, label="VMax:"), 0, wx.ALIGN_CENTER_VERTICAL)
        contrast_sizer.Add(self.vmax_slider, 1, wx.EXPAND)
        contrast_sizer.Add(self.txt_vmax, 0, wx.ALIGN_CENTER_VERTICAL)
        viz_box.Add(contrast_sizer, 0, wx.EXPAND | wx.ALL, 5)

        left_sizer.Add(viz_box, 0, wx.EXPAND | wx.ALL, 5)

        # Subtraction
        sub_sizer = wx.StaticBoxSizer(wx.VERTICAL, left_panel, "5. Subtraction")
        btn_sub = wx.Button(left_panel, label="Process Subtraction")
        btn_sub.Bind(wx.EVT_BUTTON, self.on_subtract)
        sub_sizer.Add(btn_sub, 0, wx.EXPAND | wx.ALL, 5)
        
        self.btn_export_csv = wx.Button(left_panel, label="Export Subtracted Results (CSV)")
        self.btn_export_csv.Bind(wx.EVT_BUTTON, self.on_export_csv)
        self.btn_export_csv.Disable()
        sub_sizer.Add(self.btn_export_csv, 0, wx.EXPAND | wx.ALL, 5)
        
        left_sizer.Add(sub_sizer, 0, wx.EXPAND | wx.ALL, 5)
        
        # 6. Export Smoothed 1D BG
        exp_bg_sizer = wx.StaticBoxSizer(wx.VERTICAL, left_panel, "6. Export 1D Smoothed BG")
        self.clb_export_syms = wx.CheckListBox(left_panel, choices=[])
        exp_bg_sizer.Add(self.clb_export_syms, 0, wx.EXPAND | wx.ALL, 5)
        
        btn_export_1d_bg = wx.Button(left_panel, label="Export 1D Smoothed BG (CSV)")
        btn_export_1d_bg.Bind(wx.EVT_BUTTON, self.on_export_1d_bg)
        exp_bg_sizer.Add(btn_export_1d_bg, 0, wx.EXPAND | wx.ALL, 5)
        left_sizer.Add(exp_bg_sizer, 0, wx.EXPAND | wx.ALL, 5)
        
        left_panel.SetSizer(left_sizer)
        
        # Scrolled window setup
        left_scrolled_sizer.Add(left_panel, 1, wx.EXPAND)
        left_scrolled.SetSizer(left_scrolled_sizer)
        
        # --- Right Visualization Panel ---
        self.nb = wx.Notebook(main_splitter)
        
        self.panel_rec = PlotPanel(self.nb, rows=3, cols=3)
        self.panel_sub = PlotPanel(self.nb, rows=3, cols=2) # Raw, BG, Subtracted (XX/YX cols)
        
        self.nb.AddPage(self.panel_rec, "1. Reconstruction")
        self.nb.AddPage(self.panel_sub, "2. Subtraction")
        
        main_splitter.SplitVertically(left_scrolled, self.nb, 350)
        
        self.CreateStatusBar()

        # Bind events
        btn_load.Bind(wx.EVT_BUTTON, self.on_load)
        btn_select_raw.Bind(wx.EVT_BUTTON, self.on_select_raw_files)
        btn_select_bg.Bind(wx.EVT_BUTTON, self.on_select_bg_files)
        self.btn_auto_scale.Bind(wx.EVT_BUTTON, self.on_auto_scale)
        self.choice_sym_viz.Bind(wx.EVT_CHOICE, self.on_sym_viz_change)
        self.vmin_slider.Bind(wx.EVT_SLIDER, self.on_vlim_slide)
        self.vmax_slider.Bind(wx.EVT_SLIDER, self.on_vlim_slide)
        self.txt_vmin.Bind(wx.EVT_TEXT_ENTER, self.on_vlim_text)
        self.txt_vmax.Bind(wx.EVT_TEXT_ENTER, self.on_vlim_text)
        self.choice_cmap.Bind(wx.EVT_CHOICE, self.on_cmap_change)

        # Initially disable scaling controls
        self.txt_norm_shift.Disable()
        self.btn_auto_scale.Disable()
        self.spin_scale_factor.Disable()

    def on_select_raw_files(self, event):
        with wx.FileDialog(
            self, "Select 2 Raw Data Files (XX and YX)",
            wildcard="Data files (*.csv;*.txt)|*.csv;*.txt|All files (*.*)|*.*",
            style=wx.FD_OPEN | wx.FD_MULTIPLE | wx.FD_FILE_MUST_EXIST
        ) as fileDialog:
            if fileDialog.ShowModal() == wx.ID_CANCEL:
                return
            paths = fileDialog.GetPaths()

        if len(paths) != 2:
            wx.MessageBox("Please select exactly two files.", "Selection Error", wx.OK | wx.ICON_ERROR)
            return

        self.raw_xx_path, self.raw_yx_path = self._assign_file_pair(paths)
        self.txt_raw_xx.SetValue(os.path.basename(self.raw_xx_path) if self.raw_xx_path else "")
        self.txt_raw_yx.SetValue(os.path.basename(self.raw_yx_path) if self.raw_yx_path else "")

    def on_select_bg_files(self, event):
        with wx.FileDialog(
            self, "Select 2 Background Files (XX and YX)",
            wildcard="Data files (*.csv;*.txt)|*.csv;*.txt|All files (*.*)|*.*",
            style=wx.FD_OPEN | wx.FD_MULTIPLE | wx.FD_FILE_MUST_EXIST
        ) as fileDialog:
            if fileDialog.ShowModal() == wx.ID_CANCEL:
                return
            paths = fileDialog.GetPaths()

        if len(paths) != 2:
            wx.MessageBox("Please select exactly two files.", "Selection Error", wx.OK | wx.ICON_ERROR)
            return

        self.bg_xx_path, self.bg_yx_path = self._assign_file_pair(paths)
        self.txt_bg_xx.SetValue(os.path.basename(self.bg_xx_path) if self.bg_xx_path else "")
        self.txt_bg_yx.SetValue(os.path.basename(self.bg_yx_path) if self.bg_yx_path else "")

    def _assign_file_pair(self, paths: List[str]) -> Tuple[str, str]:
        """
        Given two file paths, determines which is XX and which is YX based on filename.
        Returns (xx_path, yx_path).
        """
        path1, path2 = paths
        info1, info2 = parse_filename(path1), parse_filename(path2)
        pol1, pol2 = str(info1.get("pol", "")).lower(), str(info2.get("pol", "")).lower()
        xx_keys, yx_keys = ['xx', 'para'], ['yx', 'xy', 'cross']
        is1_xx, is1_yx = any(k in pol1 for k in xx_keys), any(k in pol1 for k in yx_keys)
        is2_xx, is2_yx = any(k in pol2 for k in xx_keys), any(k in pol2 for k in yx_keys)
        if (is1_xx and is2_yx) or (is2_xx and not is1_xx) or (is2_yx and not is1_yx): return path1, path2
        if (is1_yx and is2_xx) or (is1_xx and not is2_xx) or (is1_yx and not is2_yx): return path2, path1
        self.SetStatusText("Could not distinguish XX/YX from filenames, assuming first is XX.")
        return path1, path2

    def on_load(self, event):
        paths = [
            self.raw_xx_path,
            self.raw_yx_path,
            self.bg_xx_path,
            self.bg_yx_path
        ]
        if not all(paths):
            wx.MessageBox("Please select all 4 files.", "Error")
            return
            
        success, msg = self.engine.load_runs(*paths)
        if success:
            self.SetStatusText("Files loaded successfully.")
        else:
            wx.MessageBox(f"Error loading files: {msg}", "Error")

    def on_reconstruct(self, event):
        if not self.engine.bg_xx:
            wx.MessageBox("Load data first.", "Error")
            return
            
        syms = []
        if self.cb_a1g.GetValue(): syms.append("D6h_A1g")
        if self.cb_e2g.GetValue(): syms.append("D6h_E2g")
        if self.cb_b1g.GetValue(): syms.append("D2h_B1g")
        if self.cb_ag.GetValue(): syms.append("D2h_Ag")
        
        if not syms:
            wx.MessageBox("Select at least one symmetry.", "Error")
            return
            
        try:
            phi = self.spin_phi.GetValue()
            ratio = self.spin_ratio.GetValue()
            w = self.spin_smooth.GetValue()
            
            if self.cb_fit_params.GetValue():
                phi_final, ratio_final = self.engine.reconstruct_with_fit(syms, phi, ratio, smooth_window=w)
                self.spin_phi.SetValue(phi_final)
                self.spin_ratio.SetValue(ratio_final)
                self.SetStatusText(f"Fit complete. Optimal phi={phi_final:.2f}, ratio={ratio_final:.2f}")
            else:
                self.engine.reconstruct(syms, phi, ratio, smooth_window=w)
                self.SetStatusText("Background reconstructed.")

            self.choice_sym_viz.Clear()
            self.clb_export_syms.Clear()
            sym_keys = list(self.engine.profiles.keys())
            if sym_keys:
                self.choice_sym_viz.AppendItems(sym_keys)
                self.choice_sym_viz.SetSelection(0)
                self.clb_export_syms.AppendItems(sym_keys)
                for i in range(self.clb_export_syms.GetCount()):
                    self.clb_export_syms.Check(i, True)
                
            # Compute global limits
            all_data = [self.engine.bg_xx.intensity_2d, self.engine.bg_yx.intensity_2d, 
                        self.engine.rec_bg_xx, self.engine.rec_bg_yx]
            self.global_vmin = float(min(np.nanmin(d) for d in all_data))
            self.global_vmax = float(max(np.nanmax(d) for d in all_data))
            self.update_vlim_ui_from_scale()

            self.plot_reconstruction()
            self.nb.SetSelection(0)

            # Enable scaling controls and auto-calculate
            self.txt_norm_shift.Enable()
            self.btn_auto_scale.Enable()
            self.spin_scale_factor.Enable()
            self.on_auto_scale(None)

        except Exception as e:
            wx.MessageBox(f"Reconstruction failed: {str(e)}", "Error")

    def update_vlim_ui_from_scale(self):
        self.txt_vmin.SetValue(f"{self.global_vmin:.2f}")
        self.txt_vmax.SetValue(f"{self.global_vmax:.2f}")
        self.vmin_slider.SetValue(0)
        self.vmax_slider.SetValue(100)

    def on_auto_scale(self, event):
        if self.engine.rec_bg_xx is None:
            wx.MessageBox("Run reconstruction first.", "Error")
            return
        try:
            shift = float(self.txt_norm_shift.GetValue())
            factor = self.engine.calculate_bg_scale_factor(shift)
            self.spin_scale_factor.SetValue(factor)
            self.SetStatusText(f"Auto-scaling factor calculated: {factor:.4f}")
        except Exception as e:
            wx.MessageBox(f"Auto-scaling failed: {e}", "Error")

    def on_cmap_change(self, event):
        cmap = self.choice_cmap.GetStringSelection()
        for name, plotter in self.plotters.items():
            if 'residual' not in name:
                plotter.set_cmap(cmap)
        self.panel_rec.draw()
        self.panel_sub.draw()

    def on_vlim_slide(self, event):
        vp_min = self.vmin_slider.GetValue()
        vp_max = self.vmax_slider.GetValue()
        
        if vp_min >= vp_max:
            if event and event.GetEventObject() is self.vmin_slider:
                vp_max = min(100, vp_min + 1)
                self.vmax_slider.SetValue(vp_max)
                if vp_min >= vp_max: vp_min = vp_max - 1; self.vmin_slider.SetValue(vp_min)
            else:
                vp_min = max(0, vp_max - 1)
                self.vmin_slider.SetValue(vp_min)
                if vp_min >= vp_max: vp_max = vp_min + 1; self.vmax_slider.SetValue(vp_max)
                
        vmin = self.global_vmin + (self.global_vmax - self.global_vmin) * (vp_min / 100.0)
        vmax = self.global_vmin + (self.global_vmax - self.global_vmin) * (vp_max / 100.0)
        
        self.txt_vmin.SetValue(f"{vmin:.2f}")
        self.txt_vmax.SetValue(f"{vmax:.2f}")
        
        self.apply_vlim(vmin, vmax)

    def on_vlim_text(self, event):
        try:
            vmin = float(self.txt_vmin.GetValue())
            vmax = float(self.txt_vmax.GetValue())
            
            if self.global_vmax > self.global_vmin:
                vp_min = int(100 * (vmin - self.global_vmin) / (self.global_vmax - self.global_vmin))
                vp_max = int(100 * (vmax - self.global_vmin) / (self.global_vmax - self.global_vmin))
                self.vmin_slider.SetValue(max(0, min(100, vp_min)))
                self.vmax_slider.SetValue(max(0, min(100, vp_max)))
            
            self.apply_vlim(vmin, vmax)
        except ValueError:
            pass

    def apply_vlim(self, vmin, vmax):
        for name, plotter in self.plotters.items():
            if 'residual' not in name:
                plotter.set_clim(vmin, vmax)
        self.panel_rec.draw()
        self.panel_sub.draw()

    def on_sym_viz_change(self, event):
        self.plot_individual_component()
        self.panel_rec.draw()

    def plot_reconstruction(self):
        p = self.panel_rec
        p.figure.clear()
        
        # We want to share axes between 2D maps. 
        # (3, 3) grid. 
        # Row 0: 2D, 2D, 1D
        # Row 1: 2D, 2D, 2D
        # Row 2: 2D, 2D, 1D
        # Let's use subplots with sharex/sharey for the 2D ones.
        # It's easier to join them manually for specific axes.
        p.axes = p.figure.subplots(3, 3, squeeze=False)
        
        # Axes to synchronize (all 2D maps)
        two_d_axes = [
            p.get_ax(0, 0), p.get_ax(0, 1),
            p.get_ax(1, 0), p.get_ax(1, 1), p.get_ax(1, 2),
            p.get_ax(2, 0), p.get_ax(2, 1)
        ]
        base_ax = two_d_axes[0]
        for ax in two_d_axes[1:]:
            ax.sharex(base_ax)
            ax.sharey(base_ax)

        self.plotters = {}
        
        bg_xx = self.engine.bg_xx
        bg_yx = self.engine.bg_yx
        x = bg_xx.shift_cm1
        y = bg_xx.angle_values
        
        cmap = self.choice_cmap.GetStringSelection()
        vmin = float(self.txt_vmin.GetValue())
        vmax = float(self.txt_vmax.GetValue())

        plotter_bg_xx = RamanPlotter2d(p.get_ax(0, 0))
        plotter_bg_xx.render(x, y, bg_xx.intensity_2d, title="Input BG XX", cmap=cmap)
        plotter_bg_xx.set_clim(vmin, vmax)
        plotter_bg_xx.ax.set_ylabel("Angle")
        self.plotters['rec_bg_xx'] = plotter_bg_xx

        plotter_bg_yx = RamanPlotter2d(p.get_ax(0, 1))
        plotter_bg_yx.render(x, y, bg_yx.intensity_2d, title="Input BG YX", cmap=cmap)
        plotter_bg_yx.set_clim(vmin, vmax)
        self.plotters['rec_bg_yx'] = plotter_bg_yx
        
        ax = p.get_ax(0, 2)
        for sym, prof in self.engine.profiles.items():
            ax.plot(x, prof, label=sym)
        ax.legend(fontsize='small')
        ax.set_title("Extracted Symmetry Profiles")
        
        plotter_rec_xx = RamanPlotter2d(p.get_ax(1, 0))
        plotter_rec_xx.render(x, y, self.engine.rec_bg_xx, title="Total Reconstructed BG XX", cmap=cmap)
        plotter_rec_xx.set_clim(vmin, vmax)
        plotter_rec_xx.ax.set_ylabel("Angle")
        self.plotters['rec_total_xx'] = plotter_rec_xx

        plotter_rec_yx = RamanPlotter2d(p.get_ax(1, 1))
        plotter_rec_yx.render(x, y, self.engine.rec_bg_yx, title="Total Reconstructed BG YX", cmap=cmap)
        plotter_rec_yx.set_clim(vmin, vmax)
        self.plotters['rec_total_yx'] = plotter_rec_yx

        residual_xx = self.engine.bg_xx.intensity_2d - self.engine.rec_bg_xx
        plotter_res_xx = RamanPlotter2d(p.get_ax(1, 2))
        plotter_res_xx.render(x, y, residual_xx, title="Residual (Input - Rec)", cmap='coolwarm')
        res_vmax = float(np.nanpercentile(np.abs(residual_xx), 99))
        plotter_res_xx.set_clim(-res_vmax, res_vmax)
        self.plotters['rec_residual_xx'] = plotter_res_xx

        self.plot_individual_component()
        p.draw()
        
    def plot_individual_component(self):
        p = self.panel_rec
        sym_key = self.choice_sym_viz.GetStringSelection()
        if not sym_key:
            for i in range(3): 
                p.get_ax(2, i).clear()
                p.get_ax(2, i).set_axis_off()
            return

        profile, map_xx, map_yx = self.engine.get_individual_component_map(sym_key)
        if profile is None: return

        x = self.engine.bg_xx.shift_cm1
        y = self.engine.bg_xx.angle_values
        
        cmap = self.choice_cmap.GetStringSelection()
        vmin = float(self.txt_vmin.GetValue())
        vmax = float(self.txt_vmax.GetValue())

        p.get_ax(2, 0).clear()
        plotter_ind_xx = RamanPlotter2d(p.get_ax(2, 0))
        plotter_ind_xx.render(x, y, map_xx, title=f"{sym_key} XX Map", cmap=cmap)
        plotter_ind_xx.set_clim(vmin, vmax)
        plotter_ind_xx.ax.set_xlabel("Raman Shift (cm-1)")
        self.plotters['rec_ind_xx'] = plotter_ind_xx

        p.get_ax(2, 1).clear()
        plotter_ind_yx = RamanPlotter2d(p.get_ax(2, 1))
        plotter_ind_yx.render(x, y, map_yx, title=f"{sym_key} YX Map", cmap=cmap)
        plotter_ind_yx.set_clim(vmin, vmax)
        plotter_ind_yx.ax.set_xlabel("Raman Shift (cm-1)")
        self.plotters['rec_ind_yx'] = plotter_ind_yx

        ax_prof = p.get_ax(2, 2); ax_prof.clear()
        
        # Plot Raw vs Smoothed
        raw_profile = self.engine.raw_profiles.get(sym_key)
        if raw_profile is not None:
            # Physical normalization
            v_xx, v_yx = self.engine.basis_vectors[sym_key]
            norm = np.max(np.abs(np.concatenate([v_xx, v_yx])))
            if norm < 1e-9: norm = 1.0
            ax_prof.plot(x, raw_profile / norm, label='Raw', alpha=0.4, color='gray', linestyle='--')
            
        ax_prof.plot(x, profile, label='Smoothed', color='tab:blue', linewidth=1.5)
        ax_prof.legend(fontsize='small')
        ax_prof.set_title(f"{sym_key} Profile (Physical Scale)")
        ax_prof.set_xlabel("Raman Shift (cm-1)")

    def on_subtract(self, event):
        if self.engine.rec_bg_xx is None:
            wx.MessageBox("Run reconstruction first.", "Error")
            return
            
        scale_factor = self.spin_scale_factor.GetValue()
        sub_xx, sub_yx, scaled_rec_xx, scaled_rec_yx = self.engine.get_subtracted(scale_factor)
        if sub_xx is None: return
        
        all_data = [
            self.engine.raw_xx.intensity_2d,
            self.engine.raw_yx.intensity_2d,
            scaled_rec_xx,
            scaled_rec_yx,
            sub_xx,
            sub_yx
        ]
        self.global_vmin = float(min(np.nanmin(d) for d in all_data))
        self.global_vmax = float(max(np.nanmax(d) for d in all_data))
        self.update_vlim_ui_from_scale()

        self.plot_subtraction(sub_xx, sub_yx, scaled_rec_xx, scaled_rec_yx)
        self.nb.SetSelection(1)
        self.btn_export_csv.Enable()
        self.SetStatusText("Subtraction complete.")

    def on_export_csv(self, event):
        scale_factor = self.spin_scale_factor.GetValue()
        sub_xx, sub_yx, _, _ = self.engine.get_subtracted(scale_factor)
        if sub_xx is None: return

        with wx.DirDialog(self, "Choose output directory", "", wx.DD_DEFAULT_STYLE | wx.DD_DIR_MUST_EXIST) as dirDialog:
            if dirDialog.ShowModal() == wx.ID_CANCEL:
                return
            output_dir = dirDialog.GetPath()

        try:
            # Create a copy of the raw runs but replace the intensity data
            # Prepare XX export
            run_xx = copy.deepcopy(self.engine.raw_xx)
            run_xx.intensity_2d = sub_xx
            run_xx.nickname = f"{run_xx.nickname}_subtracted"
            run_xx.export_csv(output_dir=output_dir)

            # Prepare YX export
            run_yx = copy.deepcopy(self.engine.raw_yx)
            run_yx.intensity_2d = sub_yx
            run_yx.nickname = f"{run_yx.nickname}_subtracted"
            run_yx.export_csv(output_dir=output_dir)

            wx.MessageBox(f"Subtracted results exported to {output_dir}", "Export Successful")
        except Exception as e:
            wx.MessageBox(f"Export failed: {str(e)}", "Error", wx.ICON_ERROR)

    def on_export_1d_bg(self, event):
        chosen = self.clb_export_syms.GetCheckedStrings()
        if not chosen:
            wx.MessageBox("Select at least one symmetry to export.", "Error")
            return

        with wx.FileDialog(
            self, "Save 1D Smoothed BG CSV", 
            wildcard="CSV files (*.csv)|*.csv", 
            style=wx.FD_SAVE | wx.FD_OVERWRITE_PROMPT,
            defaultFile="smoothed_1d_bg.csv"
        ) as fileDialog:
            if fileDialog.ShowModal() == wx.ID_CANCEL:
                return
            output_path = fileDialog.GetPath()

        try:
            x = self.engine.bg_xx.shift_cm1
            bg_xx_1d = np.zeros_like(x)
            bg_yx_1d = np.zeros_like(x)

            for sym in chosen:
                profile = self.engine.profiles[sym]
                v_xx, v_yx = self.engine.basis_vectors[sym]
                
                # We calculate the max norm to be consistent with the physical scale extraction
                norm = np.max(np.abs(np.concatenate([v_xx, v_yx])))
                if norm < 1e-9: norm = 1.0
                
                # Here we compute the average contribution over all measured angles
                v_xx_norm = v_xx / norm
                v_yx_norm = v_yx / norm
                
                # The mean of angle dependency for parallel and cross
                mean_v_xx = np.mean(v_xx_norm)
                mean_v_yx = np.mean(v_yx_norm)
                
                bg_xx_1d += mean_v_xx * profile
                bg_yx_1d += mean_v_yx * profile

            import pandas as pd
            df = pd.DataFrame({
                "Shift": x,
                "BG_XX": bg_xx_1d,
                "BG_YX": bg_yx_1d
            })
            df.to_csv(output_path, index=False)
            wx.MessageBox(f"1D Smoothed BG exported to {output_path}", "Export Successful")
        except Exception as e:
            wx.MessageBox(f"Export failed: {str(e)}", "Error", wx.ICON_ERROR)

    def plot_subtraction(self, sub_xx, sub_yx, scaled_rec_xx, scaled_rec_yx):
        p = self.panel_sub
        p.figure.clear()
        
        # Share axes between all 2D maps in subtraction panel
        p.axes = p.figure.subplots(3, 2, squeeze=False)
        base_ax = p.get_ax(0, 0)
        for r in range(3):
            for c in range(2):
                if r == 0 and c == 0: continue
                p.get_ax(r, c).sharex(base_ax)
                p.get_ax(r, c).sharey(base_ax)
        
        x = self.engine.raw_xx.shift_cm1
        y = self.engine.raw_xx.angle_values
        
        raw_xx = self.engine.raw_xx.intensity_2d
        raw_yx = self.engine.raw_yx.intensity_2d
        
        cmap = self.choice_cmap.GetStringSelection()
        vmin = float(self.txt_vmin.GetValue())
        vmax = float(self.txt_vmax.GetValue())
        
        plotter_raw_xx = RamanPlotter2d(p.get_ax(0, 0))
        plotter_raw_xx.render(x, y, raw_xx, title="Raw XX", cmap=cmap)
        plotter_raw_xx.set_clim(vmin, vmax)
        self.plotters['sub_raw_xx'] = plotter_raw_xx
        
        plotter_raw_yx = RamanPlotter2d(p.get_ax(0, 1))
        plotter_raw_yx.render(x, y, raw_yx, title="Raw YX", cmap=cmap)
        plotter_raw_yx.set_clim(vmin, vmax)
        self.plotters['sub_raw_yx'] = plotter_raw_yx
        
        plotter_bg_xx = RamanPlotter2d(p.get_ax(1, 0))
        plotter_bg_xx.render(x, y, scaled_rec_xx, title="Scaled Rec BG XX", cmap=cmap)
        plotter_bg_xx.set_clim(vmin, vmax)
        self.plotters['sub_bg_xx'] = plotter_bg_xx
        
        plotter_bg_yx = RamanPlotter2d(p.get_ax(1, 1))
        plotter_bg_yx.render(x, y, scaled_rec_yx, title="Scaled Rec BG YX", cmap=cmap)
        plotter_bg_yx.set_clim(vmin, vmax)
        self.plotters['sub_bg_yx'] = plotter_bg_yx
        
        plotter_sub_xx = RamanPlotter2d(p.get_ax(2, 0))
        plotter_sub_xx.render(x, y, sub_xx, title="Subtracted XX", cmap=cmap)
        plotter_sub_xx.set_clim(vmin, vmax)
        self.plotters['sub_sub_xx'] = plotter_sub_xx
        
        plotter_sub_yx = RamanPlotter2d(p.get_ax(2, 1))
        plotter_sub_yx.render(x, y, sub_yx, title="Subtracted YX", cmap=cmap)
        plotter_sub_yx.set_clim(vmin, vmax)
        self.plotters['sub_sub_yx'] = plotter_sub_yx
        
        p.draw()


if __name__ == "__main__":
    app = wx.App(False)
    frame = MainFrame()
    frame.Show()
    app.MainLoop()