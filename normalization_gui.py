
import wx
import numpy as np
import matplotlib
from matplotlib.backends.backend_wxagg import FigureCanvasWxAgg as FigureCanvas
from matplotlib.backends.backend_wxagg import NavigationToolbar2WxAgg as NavigationToolbar
from matplotlib.figure import Figure
from matplotlib.widgets import RectangleSelector
from data_structure import Run, RunType, new_run_id

class NormalizeDialog(wx.Dialog):
    def __init__(self, parent, source_run: Run):
        super().__init__(parent, title=f"Normalize 2D Map: {source_run.nickname}", size=(900, 800),
                         style=wx.DEFAULT_DIALOG_STYLE | wx.RESIZE_BORDER | wx.MAXIMIZE_BOX)
        
        self.source_run = source_run
        self.result_run = None
        
        # Current reference values
        self.ref_min = None
        self.ref_max = None
        self.roi = None # (xmin, xmax, ymin, ymax)
        
        self.init_ui()
        self.plot_data()

    def init_ui(self):
        main_sizer = wx.BoxSizer(wx.VERTICAL)
        
        # Plot Area
        self.fig = Figure()
        self.canvas = FigureCanvas(self, -1, self.fig)
        self.toolbar = NavigationToolbar(self.canvas)
        
        main_sizer.Add(self.toolbar, 0, wx.EXPAND)
        main_sizer.Add(self.canvas, 1, wx.EXPAND | wx.ALL, 5)
        
        self.ax = self.fig.add_subplot(111)
        
        # ROI Rectangle Selector
        self.rs = RectangleSelector(self.ax, self.on_select,
                                    useblit=True,
                                    button=[1, 3],  # left/right button
                                    minspanx=5, minspany=5,
                                    interactive=True)
        
        # Controls
        ctrl_sizer = wx.BoxSizer(wx.HORIZONTAL)
        
        # Min/Max display
        grid = wx.FlexGridSizer(2, 2, 5, 5)
        grid.Add(wx.StaticText(self, label="Min Reference:"), 0, wx.ALIGN_CENTER_VERTICAL)
        self.txt_min = wx.TextCtrl(self, value="", style=wx.TE_READONLY)
        grid.Add(self.txt_min, 0, wx.EXPAND)
        
        grid.Add(wx.StaticText(self, label="Max Reference:"), 0, wx.ALIGN_CENTER_VERTICAL)
        self.txt_max = wx.TextCtrl(self, value="", style=wx.TE_READONLY)
        grid.Add(self.txt_max, 0, wx.EXPAND)
        
        ctrl_sizer.Add(grid, 0, wx.ALL, 10)
        
        # Buttons
        btn_box = wx.BoxSizer(wx.VERTICAL)
        self.btn_calc = wx.Button(self, label="Normalize to this region")
        self.btn_calc.Bind(wx.EVT_BUTTON, self.on_calc_roi)
        btn_box.Add(self.btn_calc, 0, wx.EXPAND | wx.BOTTOM, 5)
        
        info_txt = wx.StaticText(self, label="""Choose ROI with rectangle.
If no ROI, uses current view limits.""")
        btn_box.Add(info_txt, 0, wx.EXPAND)
        
        ctrl_sizer.Add(btn_box, 0, wx.ALL, 10)
        
        main_sizer.Add(ctrl_sizer, 0, wx.EXPAND)
        
        # Dialog Buttons
        btns = self.CreateButtonSizer(wx.OK | wx.CANCEL)
        main_sizer.Add(btns, 0, wx.ALIGN_RIGHT | wx.ALL, 10)
        
        self.SetSizer(main_sizer)
        
        self.Bind(wx.EVT_BUTTON, self.on_confirm, id=wx.ID_OK)

    def plot_data(self):
        if self.source_run.intensity_2d is None:
            return
            
        x = self.source_run.shift_cm1
        y = self.source_run.angle_values
        z = self.source_run.intensity_2d
        
        # Handle orientation if needed (angles vs shifts)
        if z.shape != (len(y), len(x)):
            if z.shape == (len(x), len(y)):
                z = z.T
            else:
                self.ax.text(0.5, 0.5, "Data shape mismatch", ha='center', transform=self.ax.transAxes)
                self.canvas.draw()
                return

        # Use centers_to_edges if available, but for visualization standard pcolormesh is fine
        # We'll just use the centers for simplicity in this dialog
        vmin, vmax = np.nanpercentile(z, [1, 99])
        self.mesh = self.ax.pcolormesh(x, y, z, vmin=vmin, vmax=vmax, cmap='viridis', shading='auto')
        self.ax.set_title(f"Source: {self.source_run.nickname}")
        self.ax.set_xlabel("Raman shift (cm-1)")
        self.ax.set_ylabel("Angle (deg)")
        
        self.fig.colorbar(self.mesh, ax=self.ax)
        self.canvas.draw()

    def on_select(self, eclick, erelease):
        """Callback for RectangleSelector."""
        ext = self.rs.extents # (xmin, xmax, ymin, ymax)
        self.roi = ext

    def on_calc_roi(self, event):
        """Calculate min/max from ROI or current view."""
        if self.source_run.intensity_2d is None:
            return
            
        x = self.source_run.shift_cm1
        y = self.source_run.angle_values
        z = self.source_run.intensity_2d
        
        if z.shape != (len(y), len(x)):
            if z.shape == (len(x), len(y)):
                z = z.T
            else:
                return

        # Check if RectangleSelector is visible and has a selection
        if self.rs.get_visible() and self.roi:
            xmin, xmax, ymin, ymax = self.roi
        else:
            # Use current xlim/ylim
            xmin, xmax = self.ax.get_xlim()
            ymin, ymax = self.ax.get_ylim()
            
        # Find indices
        ix = np.where((x >= xmin) & (x <= xmax))[0]
        iy = np.where((y >= ymin) & (y <= ymax))[0]
        
        if len(ix) == 0 or len(iy) == 0:
            wx.MessageBox("No data points in the selected region.", "Error", wx.OK | wx.ICON_ERROR)
            return
            
        # Extract sub-matrix
        sub_z = z[iy[0]:iy[-1]+1, ix[0]:ix[-1]+1]
        
        self.ref_min = np.nanmin(sub_z)
        self.ref_max = np.nanmax(sub_z)
        
        self.txt_min.SetValue(f"{self.ref_min:.6g}")
        self.txt_max.SetValue(f"{self.ref_max:.6g}")

    def on_confirm(self, event):
        """Perform normalization and create new Run."""
        # Auto-calculate if not done yet
        if self.ref_min is None or self.ref_max is None:
            self.on_calc_roi(None)
            if self.ref_min is None or self.ref_max is None:
                return

        if self.ref_max <= self.ref_min:
            wx.MessageBox("Max value must be greater than min value.", "Error", wx.OK | wx.ICON_ERROR)
            return

        z = self.source_run.intensity_2d.astype(float)
        # Normalize: (data - min) / (max - min)
        norm_z = (z - self.ref_min) / (self.ref_max - self.ref_min)
        
        new_nickname = f"{self.source_run.nickname}_norm"
        
        new_metadata = self.source_run.metadata.copy()
        new_metadata["nickname"] = new_nickname
        new_metadata["normalized"] = True
        new_metadata["norm_min"] = self.ref_min
        new_metadata["norm_max"] = self.ref_max
        new_metadata["source_run_id"] = self.source_run.id
        
        self.result_run = Run(
            id=new_run_id(prefix="norm"),
            source_path=self.source_run.source_path,
            source_mtime=self.source_run.source_mtime,
            wl_nm=self.source_run.wl_nm,
            shift_cm1=self.source_run.shift_cm1,
            energy_eV=self.source_run.energy_eV,
            intensity=None,
            intensity_2d=norm_z,
            angle_values=self.source_run.angle_values,
            intensity_unit="normalized",
            angle_unit=self.source_run.angle_unit,
            metadata=new_metadata,
            run_type=RunType.RUN_2D,
            raw_table=None
        )
        
        event.Skip()
