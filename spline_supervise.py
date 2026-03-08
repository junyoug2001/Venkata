import wx
import pandas as pd
import numpy as np
import json
from scipy.interpolate import splrep, BSpline
import matplotlib
matplotlib.use('WXAgg')
from matplotlib.backends.backend_wxagg import FigureCanvasWxAgg as FigureCanvas
from matplotlib.backends.backend_wxagg import NavigationToolbar2WxAgg as NavigationToolbar
from matplotlib.figure import Figure

class SplineSuperviseFrame(wx.Frame):
    def __init__(self):
        super().__init__(None, title="B-Spline Parametrization Supervisor", size=(900, 700))
        self.df = None
        self.spline_params = {'XX': None, 'YX': None}
        self.spline_funcs = {'XX': None, 'YX': None}
        self.init_ui()

    def init_ui(self):
        panel = wx.Panel(self)
        sizer = wx.BoxSizer(wx.VERTICAL)
        
        # Top controls
        ctrl_sizer = wx.BoxSizer(wx.HORIZONTAL)
        
        btn_load = wx.Button(panel, label="Load 1D BG CSV")
        btn_load.Bind(wx.EVT_BUTTON, self.on_load)
        ctrl_sizer.Add(btn_load, 0, wx.ALL | wx.ALIGN_CENTER_VERTICAL, 5)
        
        ctrl_sizer.Add(wx.StaticText(panel, label="Smoothing (s):"), 0, wx.ALL | wx.ALIGN_CENTER_VERTICAL, 5)
        
        # The scale of 's' depends heavily on the sum of squared residuals. 
        # A text control is better here to allow large ranges like 0 to 1e9
        self.txt_s = wx.TextCtrl(panel, value="0.0", style=wx.TE_PROCESS_ENTER)
        self.txt_s.Bind(wx.EVT_TEXT_ENTER, self.on_param_change)
        ctrl_sizer.Add(self.txt_s, 0, wx.ALL | wx.ALIGN_CENTER_VERTICAL, 5)
        
        ctrl_sizer.Add(wx.StaticText(panel, label="Unit:"), 0, wx.ALL | wx.ALIGN_CENTER_VERTICAL, 5)
        self.choice_unit = wx.Choice(panel, choices=["cm-1", "meV"])
        self.choice_unit.SetSelection(0)
        ctrl_sizer.Add(self.choice_unit, 0, wx.ALL | wx.ALIGN_CENTER_VERTICAL, 5)
        
        btn_update = wx.Button(panel, label="Update Fit")
        btn_update.Bind(wx.EVT_BUTTON, self.on_param_change)
        ctrl_sizer.Add(btn_update, 0, wx.ALL | wx.ALIGN_CENTER_VERTICAL, 5)

        btn_export = wx.Button(panel, label="Export B-Spline Params (JSON)")
        btn_export.Bind(wx.EVT_BUTTON, self.on_export)
        ctrl_sizer.Add(btn_export, 0, wx.ALL | wx.ALIGN_CENTER_VERTICAL, 5)
        
        sizer.Add(ctrl_sizer, 0, wx.EXPAND)
        
        # Info labels
        self.lbl_info_xx = wx.StaticText(panel, label="XX: 0 parameters")
        self.lbl_info_yx = wx.StaticText(panel, label="YX: 0 parameters")
        info_sizer = wx.BoxSizer(wx.HORIZONTAL)
        info_sizer.Add(self.lbl_info_xx, 1, wx.ALL, 5)
        info_sizer.Add(self.lbl_info_yx, 1, wx.ALL, 5)
        sizer.Add(info_sizer, 0, wx.EXPAND)
        
        # Plot area
        self.fig = Figure()
        self.canvas = FigureCanvas(panel, -1, self.fig)
        self.toolbar = NavigationToolbar(self.canvas)
        
        self.ax_xx = self.fig.add_subplot(211)
        self.ax_yx = self.fig.add_subplot(212, sharex=self.ax_xx)
        
        sizer.Add(self.toolbar, 0, wx.EXPAND)
        sizer.Add(self.canvas, 1, wx.EXPAND)
        
        panel.SetSizer(sizer)

    def on_load(self, event):
        with wx.FileDialog(self, "Open 1D Smoothed BG CSV", wildcard="CSV files (*.csv)|*.csv", style=wx.FD_OPEN | wx.FD_FILE_MUST_EXIST) as fd:
            if fd.ShowModal() == wx.ID_CANCEL: return
            path = fd.GetPath()
        
        try:
            self.df = pd.read_csv(path)
            if 'Shift' in self.df.columns:
                self.df = self.df.sort_values(by='Shift').drop_duplicates(subset=['Shift'])
            
            # Auto-estimate a good starting 's'
            # s defaults to len(x) if m is the number of points and variance is 1.
            # Let's set it to 0 initially (exact interpolation)
            self.txt_s.SetValue("0.0")
            self.update_splines()
        except Exception as e:
            wx.MessageBox(f"Failed to load: {e}", "Error", wx.ICON_ERROR)

    def on_param_change(self, event):
        if self.df is not None:
            self.update_splines()

    def update_splines(self):
        if self.df is None or 'Shift' not in self.df.columns: return
        
        x = self.df['Shift'].values
        try:
            s_val = float(self.txt_s.GetValue())
        except ValueError:
            s_val = 0.0
            self.txt_s.SetValue("0.0")

        self.spline_params = {'XX': None, 'YX': None}
        self.spline_funcs = {'XX': None, 'YX': None}
        
        for pol in ['XX', 'YX']:
            col = f'BG_{pol}'
            if col in self.df.columns:
                y = self.df[col].values
                try:
                    # B-spline representation: t (knots), c (coeffs), k (degree)
                    # s=0 interpolates through all points. s > 0 smooths.
                    t, c, k = splrep(x, y, s=s_val)
                    self.spline_params[pol] = {'t': t.tolist(), 'c': c.tolist(), 'k': int(k)}
                    self.spline_funcs[pol] = BSpline(t, c, k)
                    
                    if pol == 'XX':
                        self.lbl_info_xx.SetLabel(f"XX: {len(c)} parameters (knots + coeffs)")
                    else:
                        self.lbl_info_yx.SetLabel(f"YX: {len(c)} parameters (knots + coeffs)")
                except Exception as e:
                    print(f"Error fitting {pol}: {e}")
                    
        self.plot_data()

    def plot_data(self):
        self.ax_xx.clear()
        self.ax_yx.clear()
        
        if self.df is not None and 'Shift' in self.df.columns:
            x = self.df['Shift'].values
            x_dense = np.linspace(x.min(), x.max(), 5000)
            
            if 'BG_XX' in self.df.columns:
                y_xx = self.df['BG_XX'].values
                self.ax_xx.plot(x, y_xx, 'ko', label='Raw Data Points', alpha=0.4, markersize=4)
                if self.spline_funcs['XX'] is not None:
                    y_spline = self.spline_funcs['XX'](x_dense)
                    self.ax_xx.plot(x_dense, y_spline, 'r-', label='B-Spline Fit', linewidth=1.5)
                self.ax_xx.set_title("XX Background Spline")
                self.ax_xx.legend()
                self.ax_xx.grid(True, alpha=0.3)

            if 'BG_YX' in self.df.columns:
                y_yx = self.df['BG_YX'].values
                self.ax_yx.plot(x, y_yx, 'bo', label='Raw Data Points', alpha=0.4, markersize=4)
                if self.spline_funcs['YX'] is not None:
                    y_spline = self.spline_funcs['YX'](x_dense)
                    self.ax_yx.plot(x_dense, y_spline, 'g-', label='B-Spline Fit', linewidth=1.5)
                self.ax_yx.set_title("YX Background Spline")
                self.ax_yx.set_xlabel("Raman Shift (cm-1)")
                self.ax_yx.legend()
                self.ax_yx.grid(True, alpha=0.3)
                
        self.fig.tight_layout()
        self.canvas.draw()

    def on_export(self, event):
        if self.df is None or ('XX' not in self.spline_params and 'YX' not in self.spline_params): return
        
        with wx.FileDialog(self, "Save Spline Params JSON", wildcard="JSON files (*.json)|*.json", style=wx.FD_SAVE | wx.FD_OVERWRITE_PROMPT, defaultFile="si_bg_spline_params.json") as fd:
            if fd.ShowModal() == wx.ID_CANCEL: return
            path = fd.GetPath()
            
        out_data = {'unit': self.choice_unit.GetStringSelection()}
        if self.spline_params['XX'] is not None:
            out_data['XX'] = self.spline_params['XX']
        if self.spline_params['YX'] is not None:
            out_data['YX'] = self.spline_params['YX']
            
        with open(path, 'w') as f:
            json.dump(out_data, f, indent=4)
            
        wx.MessageBox(f"Exported B-Spline parameters successfully.\nXX parameters: {len(self.spline_params['XX']['c']) if self.spline_params['XX'] else 0}\nYX parameters: {len(self.spline_params['YX']['c']) if self.spline_params['YX'] else 0}", "Success")

if __name__ == '__main__':
    app = wx.App(False)
    frame = SplineSuperviseFrame()
    frame.Show()
    app.MainLoop()
