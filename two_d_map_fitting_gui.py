
import wx
import wx.grid
import wx.lib.dialogs
import numpy as np
import matplotlib
from matplotlib.backends.backend_wxagg import FigureCanvasWxAgg as FigureCanvas
from matplotlib.backends.backend_wxagg import NavigationToolbar2WxAgg as NavigationToolbar
from matplotlib.figure import Figure
import analysis
from data_structure import Run, RunType, new_run_id

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

    def save_to_lists(self, data_lists):
        for row in range(self.GetNumberRows()):
            for i in range(3):
                try: 
                    data_lists[row][i] = float(self.GetCellValue(row, i+1))
                except: 
                    pass

class ValidationFrame(wx.Frame):
    def __init__(self, results, parent=None):
        super().__init__(parent, title="Row-by-Row Validation Check", size=(1200, 800))
        self.results = results
        self.slice_angle_idx = 0
        self.slice_shift_idx = 0
        self.init_ui()
        self.update_plots()

    def init_ui(self):
        panel = wx.Panel(self)
        sizer = wx.BoxSizer(wx.VERTICAL)
        self.fig = Figure()
        self.canvas = FigureCanvas(panel, -1, self.fig)
        self.toolbar = NavigationToolbar(self.canvas)
        sizer.Add(self.toolbar, 0, wx.EXPAND)
        sizer.Add(self.canvas, 1, wx.EXPAND)
        panel.SetSizer(sizer)
        
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

    def on_click(self, e):
        if self.toolbar.mode != '': return
        ds_idx = -1
        if e.inaxes in [self.ax_raw1, self.ax_rec1]: ds_idx = 0
        elif e.inaxes in [self.ax_raw2, self.ax_rec2]: ds_idx = 1
        
        if ds_idx != -1:
            res = self.results[ds_idx]
            if e.xdata and e.ydata:
                self.slice_shift_idx = np.abs(res["x"] - e.xdata).argmin()
                self.slice_angle_idx = np.abs(res["ang"] - e.ydata).argmin()
                self.update_plots()

    def update_plots(self):
        saved_xlim = self.ax_raw1.get_xlim()
        saved_ylim = self.ax_raw1.get_ylim()
        
        def plot_res(idx, ax_raw, ax_rec, ax_spec, ax_ang):
            res = self.results[idx]
            x, ang, z_raw, z_rec = res["x"], res["ang"], res["z_raw"], res["z_rec"]
            vmin, vmax = np.nanpercentile(z_raw, [1, 99])
            
            ax_raw.clear(); ax_rec.clear()
            ax_raw.pcolormesh(x, ang, z_raw, vmin=vmin, vmax=vmax, cmap='inferno', shading='auto')
            ax_raw.set_title(f"{res['label']} Raw")
            ax_rec.pcolormesh(x, ang, z_rec, vmin=vmin, vmax=vmax, cmap='inferno', shading='auto')
            ax_rec.set_title("Row Fit")
            
            for ax in [ax_raw, ax_rec]:
                if 0 <= self.slice_angle_idx < len(ang):
                    ax.axhline(ang[self.slice_angle_idx], color='w', ls='--', alpha=0.5)
                if 0 <= self.slice_shift_idx < len(x):
                    ax.axvline(x[self.slice_shift_idx], color='w', ls='--', alpha=0.5)

            ax_spec.clear()
            if 0 <= self.slice_angle_idx < len(ang):
                ax_spec.plot(x, z_raw[self.slice_angle_idx,:], 'k', alpha=0.5)
                ax_spec.plot(x, z_rec[self.slice_angle_idx,:], 'r')
                ax_spec.set_title(f"Spec @ {ang[self.slice_angle_idx]:.1f}°")

            ax_ang.clear()
            if 0 <= self.slice_shift_idx < len(x):
                ax_ang.plot(ang, z_raw[:,self.slice_shift_idx], 'k', alpha=0.5)
                ax_ang.plot(ang, z_rec[:,self.slice_shift_idx], 'r')
                ax_ang.set_title(f"Ang @ {x[self.slice_shift_idx]:.1f} cm-1")

        plot_res(0, self.ax_raw1, self.ax_rec1, self.ax_spec1, self.ax_ang1)
        plot_res(1, self.ax_raw2, self.ax_rec2, self.ax_spec2, self.ax_ang2)
        
        if saved_xlim != (0.0, 1.0) and saved_ylim != (0.0, 1.0):
            self.ax_raw1.set_xlim(saved_xlim)
            self.ax_raw1.set_ylim(saved_ylim)
            self.ax_ang1.set_xlim(saved_ylim)
            self.ax_ang2.set_xlim(saved_ylim)
            
        self.canvas.draw()

class MapFittingDialog(wx.Dialog):
    def __init__(self, parent, run1: Run, run2: Run):
        super().__init__(parent, title=f"Joint 2D Fitting: {run1.nickname} & {run2.nickname}", size=(1400, 900),
                         style=wx.DEFAULT_DIALOG_STYLE | wx.RESIZE_BORDER | wx.MAXIMIZE_BOX)
        
        self.engine = analysis.MapFittingEngine()
        self.run1 = run1
        self.run2 = run2
        
        # Determine which is XX and which is YX based on nickname or pol metadata
        # Default: run1 is XX, run2 is YX unless detected otherwise
        r1_pol = str(run1.metadata.get("pol", "")).lower()
        r2_pol = str(run2.metadata.get("pol", "")).lower()
        
        swap = False
        if ("yx" in r1_pol or "xy" in r1_pol or "cross" in r1_pol) and not ("xx" in r2_pol or "para" in r2_pol):
            swap = True
        
        if swap:
            self.run1, self.run2 = run2, run1
            
        self.engine.set_data(0, self.run1.shift_cm1, self.run1.angle_values, self.run1.intensity_2d, self.run1.nickname)
        self.engine.set_data(1, self.run2.shift_cm1, self.run2.angle_values, self.run2.intensity_2d, self.run2.nickname)
        
        self.last_sel = 0
        self.slice_angle_idx = 0
        self.slice_shift_idx = len(self.engine.datasets[0]["x"]) // 2 if self.engine.datasets[0]["x"] is not None else 0
        
        self.result_runs = []
        
        self.init_ui()
        self.refresh_ui()
        self.update_plots()

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
        for lbl, cb in [("Preview", self.on_preview), ("FIT Global", self.on_fit), 
                        ("Validate (RowFit)", self.on_validate), ("Export Text", self.on_export_text)]:
            btn = wx.Button(left_panel, label=lbl)
            btn.Bind(wx.EVT_BUTTON, cb)
            act_box.Add(btn, 1, wx.EXPAND)
        left_sizer.Add(act_box, 0, wx.ALL|wx.EXPAND, 5)

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
        
        # Dialog buttons
        btn_sizer = self.CreateButtonSizer(wx.OK | wx.CANCEL)
        # Add "Export to Experiment" button
        self.btn_export_exp = wx.Button(self, label="Export Result Runs to Experiment")
        self.btn_export_exp.Bind(wx.EVT_BUTTON, self.on_export_to_experiment)
        
        h_sizer = wx.BoxSizer(wx.HORIZONTAL)
        h_sizer.Add(self.btn_export_exp, 0, wx.ALL | wx.ALIGN_CENTER_VERTICAL, 10)
        h_sizer.AddStretchSpacer()
        h_sizer.Add(btn_sizer, 0, wx.ALL, 10)
        
        main_sizer.Add(h_sizer, 0, wx.EXPAND)
        
        self.SetSizer(main_sizer)

    def refresh_ui(self):
        curr = self.peak_list.GetSelection()
        self.peak_list.Clear()
        self.peak_list.Append("Background")
        for i, p in enumerate(self.engine.peaks): 
            self.peak_list.Append(f"{i+1}: {p['name']}")
        self.peak_list.SetSelection(curr if curr != -1 else 0)
        self.last_sel = self.peak_list.GetSelection()
        self.load_grid()
        self.txt_min_x.SetValue(f"{self.engine.x_min_limit:.6g}")
        self.txt_max_x.SetValue(f"{self.engine.x_max_limit:.6g}")

    def on_peak_sel(self, e):
        self.save_grid(self.last_sel)
        self.last_sel = self.peak_list.GetSelection()
        self.load_grid()

    def load_grid(self):
        sel = self.peak_list.GetSelection()
        if sel == 0:
            self.rule_combo.Disable()
            self.grid.load_data(["Offset", "Slope"], [self.engine.bg_params["offset"], self.engine.bg_params["slope"]])
        else:
            self.rule_combo.Enable()
            p = self.engine.peaks[sel-1]
            self.rule_combo.SetValue(p["rule"])
            lbls = ["Center (x0)", "Width (G)"] + analysis.RULE_METADATA[p["rule"]]["params"]
            dst = [p["spec_params"]["x0"], p["spec_params"]["gamma"]] + [p["ang_params"].get(n, [0.0, -np.inf, np.inf]) for n in analysis.RULE_METADATA[p["rule"]]["params"]]
            self.grid.load_data(lbls, dst)

    def save_grid(self, idx):
        if idx == 0: 
            self.grid.save_to_lists([self.engine.bg_params["offset"], self.engine.bg_params["slope"]])
        elif idx > 0 and idx-1 < len(self.engine.peaks):
            p = self.engine.peaks[idx-1]
            self.grid.save_to_lists([p["spec_params"]["x0"], p["spec_params"]["gamma"]] + [p["ang_params"].get(n, [0.0, -np.inf, np.inf]) for n in analysis.RULE_METADATA[p["rule"]]["params"]])

    def on_add_peak(self, e): 
        # Get x-value at current highlight position
        x_val = None
        ds = self.engine.datasets[0]
        if ds["x"] is not None and 0 <= self.slice_shift_idx < len(ds["x"]):
            x_val = float(ds["x"][self.slice_shift_idx])
            
        self.engine.add_peak(center=x_val)
        self.refresh_ui()
        
    def on_rem_peak(self, e):
        sel = self.peak_list.GetSelection()
        if sel > 0: 
            self.engine.peaks.pop(sel-1)
            self.refresh_ui()

    def on_rule_change(self, e):
        sel = self.peak_list.GetSelection()
        if sel > 0:
            p = self.engine.peaks[sel-1]
            p["rule"] = self.rule_combo.GetValue()
            p["ang_params"] = {pn: [10.0, -np.inf, np.inf] for pn in analysis.RULE_METADATA[p["rule"]]["params"]}
            self.load_grid()

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

    def on_fit(self, e):
        self.save_grid(self.peak_list.GetSelection())
        dlg = wx.ProgressDialog("Fitting", "Running Global Optimization...", parent=self, style=wx.PD_APP_MODAL|wx.PD_ELAPSED_TIME)
        dlg.Pulse()
        success, msg = self.engine.run_optimization()
        dlg.Destroy()
        if success: 
            self.load_grid()
            self.update_plots()
            wx.MessageBox("Global Fit Complete", "Success")
        else: 
            wx.MessageBox(msg, "Fit Failed", wx.ICON_ERROR)

    def on_validate(self, e):
        prog = wx.ProgressDialog("Validating", "Running Row-by-Row Fit...", parent=self, style=wx.PD_APP_MODAL|wx.PD_ELAPSED_TIME)
        prog.Pulse()
        success, msg, results = self.engine.validate_row_by_row()
        prog.Destroy()
        
        if success and results:
            vf = ValidationFrame(results, self)
            vf.Show()
        else:
            wx.MessageBox(msg, "Validation Failed", wx.ICON_ERROR)

    def on_export_text(self, e):
        text = self.engine.export_parameters_text()
        dlg = wx.lib.dialogs.ScrolledMessageDialog(self, text, "Exported Parameters")
        dlg.ShowModal()
        dlg.Destroy()

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
                
                if 0 <= self.slice_angle_idx < len(ang):
                    ax.axhline(ang[self.slice_angle_idx], color='w', ls='--', alpha=0.5)
                if 0 <= self.slice_shift_idx < len(x):
                    ax.axvline(x[self.slice_shift_idx], color='w', ls='--', alpha=0.5)

            ax_spec.clear()
            if 0 <= self.slice_angle_idx < len(ang):
                cur_ang = ang[self.slice_angle_idx]
                ax_spec.plot(x, z_raw[self.slice_angle_idx,:], 'k', alpha=0.5)
                ax_spec.plot(x, z_rec[self.slice_angle_idx,:], 'r')
                ax_spec.set_title(f"Spec @ {cur_ang:.1f}°")
                if self.engine.x_min_limit > x.min(): 
                    ax_spec.axvspan(x.min(), self.engine.x_min_limit, color='gray', alpha=0.2)
                if self.engine.x_max_limit < x.max(): 
                    ax_spec.axvspan(self.engine.x_max_limit, x.max(), color='gray', alpha=0.2)

            ax_ang.clear()
            if 0 <= self.slice_shift_idx < len(x):
                cur_shift = x[self.slice_shift_idx]
                ax_ang.plot(ang, z_raw[:,self.slice_shift_idx], 'k', alpha=0.5)
                ax_ang.plot(ang, z_rec[:,self.slice_shift_idx], 'r')
                ax_ang.set_title(f"Ang @ {cur_shift:.1f} cm-1")

        plot_set(0, self.ax_raw1, self.ax_rec1, self.ax_spec1, self.ax_ang1)
        plot_set(1, self.ax_raw2, self.ax_rec2, self.ax_spec2, self.ax_ang2)
        
        if saved_xlim != (0.0, 1.0) and saved_ylim != (0.0, 1.0):
            self.ax_raw1.set_xlim(saved_xlim)
            self.ax_raw1.set_ylim(saved_ylim)
            self.ax_ang1.set_xlim(saved_ylim)
            self.ax_ang2.set_xlim(saved_ylim)
        
        self.canvas.draw()

    def on_export_to_experiment(self, e):
        """Build Run objects from reconstructions and store them."""
        self.save_grid(self.peak_list.GetSelection())
        
        recons = self.engine.get_peak_reconstructions()
        if not recons:
            wx.MessageBox("No reconstruction data available. Run fitting first.", "Info")
            return
            
        self.result_runs = []
        
        for rec in recons:
            ds_idx = rec["dataset_idx"]
            orig_run = self.run1 if ds_idx == 0 else self.run2
            name = rec["name"]
            matrix = rec["matrix"]
            
            new_nickname = f"{orig_run.nickname}_{name}_fit"
            
            new_metadata = orig_run.metadata.copy()
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
                angle_values=orig_run.angle_values,
                intensity_unit=orig_run.intensity_unit,
                angle_unit=orig_run.angle_unit,
                metadata=new_metadata,
                run_type=RunType.RUN_2D,
                raw_table=None
            )
            self.result_runs.append(new_run)
            
        # Also add the TOTAL reconstruction for each dataset
        for i in range(2):
            orig_run = self.run1 if i == 0 else self.run2
            total_rec = self.engine.reconstruct(i)
            
            new_nickname = f"{orig_run.nickname}_TotalFit"
            new_metadata = orig_run.metadata.copy()
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
                angle_values=orig_run.angle_values,
                intensity_unit=orig_run.intensity_unit,
                angle_unit=orig_run.angle_unit,
                metadata=new_metadata,
                run_type=RunType.RUN_2D,
                raw_table=None
            )
            self.result_runs.append(new_run)
            
        wx.MessageBox(f"Created {len(self.result_runs)} reconstruction runs.", "Export Successful")
        # In a real dialog, we might want to close here or just stay open.
        # If we are in a Modal dialog, the caller will check self.result_runs.
