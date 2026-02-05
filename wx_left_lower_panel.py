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
    EV_PER_CM1,
    new_experiment_id,
    new_view_id,
    RunType
)
from plotting import RamanPlotter2d, AngularPlotter, SlicePlotter

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


class CurveFitPanel(wx.Panel):
    """
    Curve Fit tab: placeholder for future curve-fitting controls.
    """

    def __init__(self, parent):
        super().__init__(parent)
        # For now just an empty white panel (like "no active curve" state)
        self.SetBackgroundColour(wx.Colour(255, 255, 255))


class PlotConfigPanel(wx.Panel):
    def __init__(self, parent, on_reset=None):
        super().__init__(parent)
        
        self._on_reset = on_reset
        
        self.x_unit = 'cm-1' # Default unit

        sizer = wx.BoxSizer(wx.VERTICAL)
        
        # Range controls
        range_sizer = wx.FlexGridSizer(3, 3, 5, 5)
        range_sizer.AddGrowableCol(1, 1)
        range_sizer.AddGrowableCol(2, 1)

        # Unit radio buttons
        self.rb_cm1 = wx.RadioButton(self, label="cm-1", style=wx.RB_GROUP)
        self.rb_mev = wx.RadioButton(self, label="meV")
        self.rb_cm1.SetValue(True)
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
        
        # Colormap selection
        cmap_sizer = wx.BoxSizer(wx.HORIZONTAL)
        cmap_sizer.Add(wx.StaticText(self, label="Colormap:"), 0, wx.ALIGN_CENTER_VERTICAL | wx.RIGHT, 5)
        self.choice_cmap = wx.Choice(self, choices=['OrRd', 'plasma', 'inferno', 'magma', 'cividis', 'gray', 'seismic', 'jet', 'hsv'])
        self.choice_cmap.SetStringSelection('OrRd')
        cmap_sizer.Add(self.choice_cmap, 1, wx.EXPAND)
        sizer.Add(cmap_sizer, 0, wx.EXPAND | wx.ALL, 5)
        
        # Contrast controls
        sizer.Add(wx.StaticText(self, label="Contrast (percentiles)"), 0, wx.LEFT | wx.TOP, 5)
        
        vmin_sizer = wx.BoxSizer(wx.HORIZONTAL)
        self.vmin_label = wx.StaticText(self, label="min:")
        self.vmin_slider = wx.Slider(self, value=0, minValue=0, maxValue=100)
        self.txt_vmin = wx.TextCtrl(self, value="0", size=(40, -1), style=wx.TE_PROCESS_ENTER)
        
        vmin_sizer.Add(self.vmin_label, 0, wx.ALIGN_CENTER_VERTICAL | wx.RIGHT, 5)
        vmin_sizer.Add(self.vmin_slider, 1, wx.EXPAND | wx.RIGHT, 5)
        vmin_sizer.Add(self.txt_vmin, 0, wx.ALIGN_CENTER_VERTICAL)
        sizer.Add(vmin_sizer, 0, wx.EXPAND | wx.LEFT | wx.RIGHT, 5)

        vmax_sizer = wx.BoxSizer(wx.HORIZONTAL)
        self.vmax_label = wx.StaticText(self, label="max:")
        self.vmax_slider = wx.Slider(self, value=100, minValue=0, maxValue=100)
        self.txt_vmax = wx.TextCtrl(self, value="100", size=(40, -1), style=wx.TE_PROCESS_ENTER)
        
        vmax_sizer.Add(self.vmax_label, 0, wx.ALIGN_CENTER_VERTICAL | wx.RIGHT, 5)
        vmax_sizer.Add(self.vmax_slider, 1, wx.EXPAND | wx.RIGHT, 5)
        vmax_sizer.Add(self.txt_vmax, 0, wx.ALIGN_CENTER_VERTICAL)
        sizer.Add(vmax_sizer, 0, wx.EXPAND | wx.LEFT | wx.RIGHT, 5)

        # Reset button
        self.reset_button = wx.Button(self, label="Reset Plot")
        sizer.Add(self.reset_button, 0, wx.ALIGN_CENTER | wx.ALL, 5)

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
        
        self.choice_cmap.Bind(wx.EVT_CHOICE, self.on_cmap_change)
        self.reset_button.Bind(wx.EVT_BUTTON, self.on_reset_button)

        self.target_view: Optional["ViewPanel"] = None
        self.set_target_view(None)

    def set_target_view(self, view_panel: Optional["ViewPanel"]):
        self.target_view = view_panel
        if self.target_view is None:
            for widget in [self.rb_cm1, self.rb_mev, self.x_min_text, self.x_max_text,
                           self.y_min_text, self.y_max_text, self.choice_cmap,
                           self.vmin_slider, self.txt_vmin, self.vmax_slider, self.txt_vmax,
                           self.reset_button]:
                widget.Disable()
            return

        for widget in [self.rb_cm1, self.rb_mev, self.x_min_text, self.x_max_text,
                       self.y_min_text, self.y_max_text, self.choice_cmap,
                       self.vmin_slider, self.txt_vmin, self.vmax_slider, self.txt_vmax,
                       self.reset_button]:
            widget.Enable()

        config = self.target_view.get_plot_config()
        
        # Sync Limits
        xlim = config.get('xlim', (0, 1))
        ylim = config.get('ylim', (0, 1))
        
        current_unit = self.get_x_unit()
        if current_unit == 'meV':
             xlim = (xlim[0] * EV_PER_CM1 * 1000, xlim[1] * EV_PER_CM1 * 1000)
        
        self.set_x_range(xlim[0], xlim[1])
        self.set_y_range(ylim[0], ylim[1])
        
        # Sync Contrast
        vmin_p = config.get('vmin_p', 0)
        vmax_p = config.get('vmax_p', 100)
        self.set_vlim_range(vmin_p, vmax_p)
        
        # Sync Colormap
        cmap = config.get('cmap', 'OrRd')
        self.set_colormap(cmap)

    def on_unit_change(self, event):
        rb = event.GetEventObject()
        new_unit = rb.GetLabel()
        if new_unit == self.x_unit:
            return
        self.x_unit = new_unit
        
        if self.target_view:
            xlim_cm1, _ = self.target_view.get_plot_limits()
            if xlim_cm1 and xlim_cm1[0] is not None:
                if new_unit == 'meV':
                    xlim_display = (xlim_cm1[0] * EV_PER_CM1 * 1000, xlim_cm1[1] * EV_PER_CM1 * 1000)
                else: # cm-1
                    xlim_display = xlim_cm1
                self.set_x_range(xlim_display[0], xlim_display[1])

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

    def on_vlim_slide(self, event):
        vmin = self.vmin_slider.GetValue()
        vmax = self.vmax_slider.GetValue()
        # Simple guard
        if vmin > vmax:
            if event.GetEventObject() is self.vmin_slider:
                vmax = vmin
                self.vmax_slider.SetValue(vmax)
            else:
                vmin = vmax
                self.vmin_slider.SetValue(vmin)
        
        self.txt_vmin.SetValue(str(vmin))
        self.txt_vmax.SetValue(str(vmax))
        
        if self.target_view:
            self.target_view.set_vlim(vmin, vmax)

    def on_vlim_text_enter(self, event):
        try:
            vmin = int(float(self.txt_vmin.GetValue()))
            vmax = int(float(self.txt_vmax.GetValue()))
            
            vmin = max(0, min(100, vmin))
            vmax = max(0, min(100, vmax))
            
            if vmin > vmax:
                vmax = vmin
            
            self.vmin_slider.SetValue(vmin)
            self.vmax_slider.SetValue(vmax)
            self.txt_vmin.SetValue(str(vmin))
            self.txt_vmax.SetValue(str(vmax))
            
            if self.target_view:
                self.target_view.set_vlim(vmin, vmax)
        except ValueError:
            pass

    def on_cmap_change(self, event):
        cmap = self.choice_cmap.GetStringSelection()
        if self.target_view:
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
        self.choice_cmap.SetStringSelection(cmap_name)

    def get_x_unit(self):
        return self.x_unit


class AppearancesPanel(wx.Panel):
    """
    Appearance tab: Lists runs in the current view and their display settings.
    """
    def __init__(self, parent, on_rename_run=None, on_style_change=None):
        super().__init__(parent)
        self._on_rename_run = on_rename_run
        self._on_style_change = on_style_change

        self.list_ctrl = wx.ListCtrl(self, style=wx.LC_REPORT | wx.LC_VRULES | wx.LC_HRULES)
        self.list_ctrl.InsertColumn(0, "Run", width=120)
        self.list_ctrl.InsertColumn(1, "Type", width=80)
        self.list_ctrl.InsertColumn(2, "Plots", width=80)
        self.list_ctrl.InsertColumn(3, "Vis.", width=40)
        self.list_ctrl.InsertColumn(4, "Style", width=150)

        sizer = wx.BoxSizer(wx.VERTICAL)
        sizer.Add(self.list_ctrl, 1, wx.EXPAND | wx.ALL, 4)
        self.SetSizer(sizer)

        self.list_ctrl.Bind(wx.EVT_LEFT_DCLICK, self.on_double_click)

        # State storage
        self.experiment: Optional[ExperimentSet] = None
        self.view_state: Optional[ViewState] = None
        
        # Edit helpers
        self._edit_ctrl: Optional[wx.TextCtrl] = None
        self._edit_item_idx: int = -1
        self._edit_col_idx: int = -1

    def update_view(self, view_state: Optional[ViewState], experiment: Optional[ExperimentSet]):
        self.experiment = experiment
        self.view_state = view_state
        self.list_ctrl.DeleteAllItems()
        if not view_state or not experiment:
            return

        for i, run_id in enumerate(view_state.run_ids):
            run = experiment.get_run(run_id)
            if not run:
                continue
            
            nickname = experiment.get_run_nickname(run_id)
            # Handle RunType enum or string fallback
            if isinstance(run.run_type, RunType):
                run_type = run.run_type.value
            else:
                run_type = str(run.run_type)
            
            # Determine which plots (Logic from ViewPanel is: Run1 -> A1,B1,C1; Run2 -> A2,B2,C2)
            if i == 0:
                plots = "A1, B1, C1"
            elif i == 1:
                plots = "A2, B2, C2"
            else:
                plots = "None"
            
            # Style summary
            # Use 'A' component style as representative
            run_config = view_state.get_run_config(run_id)
            s = run_config.get_style("A")
            visible_str = "Yes" if s.visible else "No"
            style_str = f"{s.color}, {s.linestyle}, {s.linewidth}"
            
            idx = self.list_ctrl.InsertItem(self.list_ctrl.GetItemCount(), nickname)
            self.list_ctrl.SetItem(idx, 1, run_type)
            self.list_ctrl.SetItem(idx, 2, plots)
            self.list_ctrl.SetItem(idx, 3, visible_str)
            self.list_ctrl.SetItem(idx, 4, style_str)
            
            # Store run_id as item data for retrieval
            self.list_ctrl.SetItemData(idx, i) # Store index in run_ids list

    def on_double_click(self, event):
        pt = event.GetPosition()
        idx, flags = self.list_ctrl.HitTest(pt)
        if idx == wx.NOT_FOUND:
            return

        col_idx = self._get_column_from_point(pt.x)
        if col_idx == 0: # Rename
            self._start_edit(idx, 0)
        elif col_idx == 4: # Style
            self._start_edit(idx, 4)
            
    def _get_column_from_point(self, x: int) -> int:
        total_w = 0
        for i in range(self.list_ctrl.GetColumnCount()):
            w = self.list_ctrl.GetColumnWidth(i)
            if x < total_w + w:
                return i
            total_w += w
        return -1

    def _start_edit(self, item_idx: int, col_idx: int):
        # Clean up existing editor
        if self._edit_ctrl:
            self._edit_ctrl.Destroy()
            self._edit_ctrl = None
            
        self._edit_item_idx = item_idx
        self._edit_col_idx = col_idx
        
        # Get Item Rect
        rect = self.list_ctrl.GetItemRect(item_idx)
        
        # Calculate x offset and width for the specific column
        x_offset = 0
        for i in range(col_idx):
            x_offset += self.list_ctrl.GetColumnWidth(i)
        col_width = self.list_ctrl.GetColumnWidth(col_idx)
        
        rect.x += x_offset
        rect.width = col_width
        
        # Get current text
        item = self.list_ctrl.GetItem(item_idx, col_idx)
        text = item.GetText()
        
        self._edit_ctrl = wx.TextCtrl(self.list_ctrl, value=text, pos=(rect.x, rect.y), size=(rect.width, rect.height), style=wx.TE_PROCESS_ENTER)
        self._edit_ctrl.SetFocus()
        self._edit_ctrl.SelectAll()
        
        self._edit_ctrl.Bind(wx.EVT_TEXT_ENTER, self._on_edit_commit)
        self._edit_ctrl.Bind(wx.EVT_KILL_FOCUS, self._on_edit_cancel)

    def _on_edit_commit(self, event):
        if not self._edit_ctrl:
            return
            
        new_text = self._edit_ctrl.GetValue()
        
        # Retrieve run_id
        run_idx = self.list_ctrl.GetItemData(self._edit_item_idx)
        if self.view_state and 0 <= run_idx < len(self.view_state.run_ids):
            run_id = self.view_state.run_ids[run_idx]
            
            if self._edit_col_idx == 0: # Rename
                if self._on_rename_run:
                    self._on_rename_run(run_id, new_text)
            elif self._edit_col_idx == 4: # Style
                if self._on_style_change:
                    self._on_style_change(run_id, new_text)
        
        self._edit_ctrl.Destroy()
        self._edit_ctrl = None

    def _on_edit_cancel(self, event):
        # If we just click away, maybe we should commit? 
        # Usually inline edit commits on focus loss.
        self._on_edit_commit(event)
