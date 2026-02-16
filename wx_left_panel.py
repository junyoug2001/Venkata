
import os
import analysis
from merge_runs_gui import MergeRunsDialog
from typing import List, Optional, Dict, Any, Tuple

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


class DerivedRunFormulaDialog(wx.Dialog):
    def __init__(self, parent, formula: str, params: Dict[str, float]):
        super().__init__(parent, title="Edit Derived Run Formula", style=wx.DEFAULT_DIALOG_STYLE | wx.RESIZE_BORDER)
        
        self.formula = formula
        self.params = params.copy()
        
        sizer = wx.BoxSizer(wx.VERTICAL)
        
        # Formula
        sizer.Add(wx.StaticText(self, label="Formula (use 'x' as variable, 'np' for numpy):"), 0, wx.ALL, 5)
        self.txt_formula = wx.TextCtrl(self, value=self.formula, size=(400, -1))
        sizer.Add(self.txt_formula, 0, wx.EXPAND | wx.ALL, 5)
        
        # Parameters
        sizer.Add(wx.StaticText(self, label="Parameters:"), 0, wx.ALL, 5)
        self.param_grid = wx.FlexGridSizer(cols=2, hgap=5, vgap=5)
        self.param_ctrls = {}
        
        self._build_param_grid()
        sizer.Add(self.param_grid, 1, wx.EXPAND | wx.ALL, 5)
        
        # Add/Remove Param Buttons
        btn_sizer = wx.BoxSizer(wx.HORIZONTAL)
        btn_add = wx.Button(self, label="Add Param")
        btn_add.Bind(wx.EVT_BUTTON, self.on_add_param)
        btn_sizer.Add(btn_add, 0, wx.RIGHT, 5)
        sizer.Add(btn_sizer, 0, wx.ALL, 5)
        
        # Dialog Buttons
        btns = self.CreateButtonSizer(wx.OK | wx.CANCEL)
        sizer.Add(btns, 0, wx.ALIGN_RIGHT | wx.ALL, 10)
        
        self.SetSizer(sizer)
        self.Fit()
        
    def _build_param_grid(self):
        self.param_grid.Clear(True)
        self.param_ctrls.clear()
        
        for name, val in self.params.items():
            lbl = wx.StaticText(self, label=f"{name}:")
            txt = wx.TextCtrl(self, value=str(val))
            self.param_grid.Add(lbl, 0, wx.ALIGN_CENTER_VERTICAL)
            self.param_grid.Add(txt, 1, wx.EXPAND)
            self.param_ctrls[name] = txt
            
        self.Layout()
        
    def on_add_param(self, event):
        dlg = wx.TextEntryDialog(self, "Enter parameter name:", "New Parameter")
        if dlg.ShowModal() == wx.ID_OK:
            name = dlg.GetValue().strip()
            if name and name not in self.params:
                self.params[name] = 1.0
                self._build_param_grid()
        dlg.Destroy()
        
    def get_values(self):
        # Update params from text ctrls
        for name, txt in self.param_ctrls.items():
            try:
                self.params[name] = float(txt.GetValue())
            except ValueError:
                pass
        return self.txt_formula.GetValue(), self.params


class DerivedRunPropsDialog(wx.Dialog):
    def __init__(self, parent, n_points: int, autorange: bool, x_range: Tuple[float, float]):
        super().__init__(parent, title="Edit Derived Run Properties", style=wx.DEFAULT_DIALOG_STYLE | wx.RESIZE_BORDER)
        
        self.n_points = n_points
        self.autorange = autorange
        self.x_range = x_range
        
        sizer = wx.BoxSizer(wx.VERTICAL)
        gs = wx.FlexGridSizer(3, 2, 5, 5)
        
        # N Points
        gs.Add(wx.StaticText(self, label="N Points:"), 0, wx.ALIGN_CENTER_VERTICAL)
        self.txt_n = wx.TextCtrl(self, value=str(n_points))
        gs.Add(self.txt_n, 1, wx.EXPAND)
        
        # Auto Range
        gs.Add(wx.StaticText(self, label="Auto Range:"), 0, wx.ALIGN_CENTER_VERTICAL)
        self.chk_auto = wx.CheckBox(self, label="Use Plot Limits")
        self.chk_auto.SetValue(autorange)
        gs.Add(self.chk_auto, 1, wx.EXPAND)
        
        # Range
        gs.Add(wx.StaticText(self, label="Default Range:"), 0, wx.ALIGN_CENTER_VERTICAL)
        range_sizer = wx.BoxSizer(wx.HORIZONTAL)
        self.txt_min = wx.TextCtrl(self, value=str(x_range[0]))
        self.txt_max = wx.TextCtrl(self, value=str(x_range[1]))
        range_sizer.Add(self.txt_min, 1, wx.RIGHT, 5)
        range_sizer.Add(self.txt_max, 1)
        gs.Add(range_sizer, 1, wx.EXPAND)
        
        sizer.Add(gs, 1, wx.EXPAND | wx.ALL, 10)
        
        # Enable/Disable range inputs based on auto
        self.chk_auto.Bind(wx.EVT_CHECKBOX, self.update_range_state)
        self.update_range_state()
        
        btns = self.CreateButtonSizer(wx.OK | wx.CANCEL)
        sizer.Add(btns, 0, wx.ALIGN_RIGHT | wx.ALL, 10)
        
        self.SetSizer(sizer)
        self.Fit()
        
    def update_range_state(self, evt=None):
        is_auto = self.chk_auto.GetValue()
        self.txt_min.Enable(not is_auto)
        self.txt_max.Enable(not is_auto)
        
    def get_values(self):
        try:
            n = int(self.txt_n.GetValue())
            auto = self.chk_auto.GetValue()
            r_min = float(self.txt_min.GetValue())
            r_max = float(self.txt_max.GetValue())
            return n, auto, (r_min, r_max)
        except ValueError:
            return self.n_points, self.autorange, self.x_range


class FilesPanel(wx.Panel):
    """
    Simple file browser for the Files tab.

    - Uses wx.GenericDirCtrl to provide a more native tree/file view.
    - On activation:
        * Directory  -> default behavior (expand/collapse).
        * File       -> notifies the main frame via callback
                        (used to update the Preview tab and log).
    - On right-click (file):
        * "Import Run" context action, if provided.
    """

    def __init__(self, parent, on_file_activated, on_import_run=None):
        super().__init__(parent)
        self._on_file_activated = on_file_activated
        self._on_import_run = on_import_run

        sizer = wx.BoxSizer(wx.VERTICAL)

        # GenericDirCtrl already includes a path field, tree, and filter control
        self.dir_ctrl = wx.GenericDirCtrl(
            self,
            style=wx.DIRCTRL_SHOW_FILTERS,
        )
        sizer.Add(self.dir_ctrl, 1, wx.EXPAND | wx.ALL, 4)

        self.SetSizer(sizer)

        # Bind activation (double-click / Enter on an item)
        tree = self.dir_ctrl.GetTreeCtrl()
        tree.Bind(wx.EVT_TREE_ITEM_ACTIVATED, self.on_item_activated)
        tree.Bind(wx.EVT_TREE_ITEM_RIGHT_CLICK, self.on_item_right_click)

    def set_directory(self, path: str) -> None:
        """Set the directory to display in the Files tab."""
        if os.path.isdir(path):
            self.dir_ctrl.SetPath(os.path.abspath(path))

    def on_item_activated(self, event): 
        """Handle activation in the directory tree."""
        item = event.GetItem()
        if not item.IsOk():
            event.Skip()
            return

        path = self.dir_ctrl.GetPath()
        if not path:
            event.Skip()
            return

        if os.path.isdir(path):
            # Let GenericDirCtrl handle expanding/collapsing
            event.Skip()
            return

        # It's a file: notify the main frame
        if self._on_file_activated is not None:
            self._on_file_activated(path)

    def on_item_right_click(self, event):
        """Right-click context menu for a file item: Import Run."""
        if self._on_import_run is None:
            return

        item = event.GetItem()
        if not item.IsOk():
            return

        # Ensure the clicked item becomes the selection, so GetPath() is correct.
        tree = self.dir_ctrl.GetTreeCtrl()
        try:
            tree.SelectItem(item)
        except Exception:
            pass

        path = self.dir_ctrl.GetPath()
        if not path or os.path.isdir(path):
            return

        menu = wx.Menu()
        item_import = menu.Append(wx.ID_ANY, "Import Run")
        self.Bind(wx.EVT_MENU, lambda evt, p=path: self._on_import_run(p), item_import)
        self.PopupMenu(menu)
        menu.Destroy()


class RunsPanel(wx.Panel):
    """
    Experiment tab (formerly Runs):

    - Upper section: list of runs in the current ExperimentSet.
        * Columns: Run Name, ID, Description.
        * Supports multiple selection (Shift/Cmd click).
    - Lower section: list of views and which runs are attached to each view.
    - A splitter between runs and views allows adjusting their relative heights.
    - Right-click on the runs list:
        * "Add to current view"
        * "Add to new view"
    - Right-click on the views list:
        * "Remove view"
        * "Duplicate view"
        * "Delete {run_id} from view" (for each run in the selected view)
    """

    def __init__(
        self,
        parent,
        on_add_to_current_view=None,
        on_add_to_new_view=None,
        on_view_remove=None,
        on_view_duplicate=None,
        on_view_remove_run=None,
        on_rename_run=None,
        on_export_run=None,
        on_update_from_file=None,
        on_remove_run=None,
        on_edit_formula=None,
    ):
        super().__init__(parent)
        self._run_ids: List[str] = []
        self._view_ids: List[str] = []
        self._view_run_ids: List[List[str]] = []

        self._on_add_to_current_view = on_add_to_current_view
        self._on_add_to_new_view = on_add_to_new_view
        self._on_view_remove = on_view_remove
        self._on_view_duplicate = on_view_duplicate
        self._on_view_remove_run = on_view_remove_run
        self._on_rename_run = on_rename_run
        self._on_export_run = on_export_run
        self._on_update_from_file = on_update_from_file
        self._on_remove_run = on_remove_run
        self._on_edit_formula = on_edit_formula

        # Reference to the current ExperimentSet (for nicknames etc.).
        self._experiment: Optional[ExperimentSet] = None
        # For inline editing state (for renaming runs)
        self._edit_ctrl: Optional[wx.TextCtrl] = None
        self._editing_index: Optional[int] = None

        root_sizer = wx.BoxSizer(wx.VERTICAL)
        self.SetSizer(root_sizer)

        # Splitter between runs (top) and views (bottom)
        self.splitter = wx.SplitterWindow(
            self, style=wx.SP_LIVE_UPDATE | wx.SP_3D
        )
        root_sizer.Add(self.splitter, 1, wx.EXPAND | wx.ALL, 4)

        # Top panel: runs
        top_panel = wx.Panel(self.splitter)
        top_sizer = wx.BoxSizer(wx.VERTICAL)
        top_panel.SetSizer(top_sizer)

        label_runs = wx.StaticText(top_panel, label="Runs in current experiment:")
        top_sizer.Add(label_runs, 0, wx.ALL, 4)

        self.run_list = wx.ListCtrl(
            top_panel, 
            style=wx.LC_REPORT | wx.LC_VRULES | wx.LC_HRULES
        )
        self.run_list.InsertColumn(0, "Run Name", width=100)
        self.run_list.InsertColumn(1, "ID", width=120)
        self.run_list.InsertColumn(2, "Description", width=200)

        top_sizer.Add(self.run_list, 1, wx.EXPAND | wx.ALL, 4)

        # Bottom panel: views
        bottom_panel = wx.Panel(self.splitter)
        bottom_sizer = wx.BoxSizer(wx.VERTICAL)
        bottom_panel.SetSizer(bottom_sizer)

        label_views = wx.StaticText(bottom_panel, label="Views:")
        bottom_sizer.Add(label_views, 0, wx.ALL, 4)

        self.views_list = wx.ListBox(bottom_panel, choices=[])
        bottom_sizer.Add(self.views_list, 1, wx.EXPAND | wx.ALL, 4)

        # Configure splitter: initial 60% height for runs, 40% for views
        self.splitter.SplitHorizontally(top_panel, bottom_panel)
        self.splitter.SetMinimumPaneSize(60)
        self.splitter.SetSashGravity(0.6)
        wx.CallAfter(self._set_initial_sash)

        # Context menus
        self.run_list.Bind(wx.EVT_LIST_ITEM_RIGHT_CLICK, self.on_context_menu_runs)
        self.views_list.Bind(wx.EVT_CONTEXT_MENU, self.on_context_menu_views)

        # Key bindings for run list
        self.run_list.Bind(wx.EVT_KEY_DOWN, self.on_run_list_key_down)

    def _set_initial_sash(self):
        """Set an initial runs:views split of about 60%:40%."""
        size = self.splitter.GetClientSize()
        if size.height > 0:
            self.splitter.SetSashPosition(int(size.height * 0.6))

    def on_run_list_key_down(self, event):
        """Handle key events on the runs list."""
        key = event.GetKeyCode()

        # Enter -> Rename
        if key in [wx.WXK_RETURN, wx.WXK_NUMPAD_ENTER]:
            self._on_context_rename_run(event)
            return
        
        # Delete or Cmd+Backspace -> Remove
        if key == wx.WXK_DELETE or (key == wx.WXK_BACK and event.CmdDown()):
            self._on_context_remove_run(event)
            return

        event.Skip()

    # --- public API ---

    def refresh_from_experiment(self, exp: ExperimentSet) -> None:
        """Rebuild the runs and views lists from the given ExperimentSet."""
        self._experiment = exp
        # Runs
        self.run_list.DeleteAllItems()
        self._run_ids.clear()

        for run_id, run in exp.runs.items():
            nickname = exp.get_run_nickname(run_id)
            base = os.path.basename(run.source_path)
            
            idx = self.run_list.InsertItem(self.run_list.GetItemCount(), nickname)
            self.run_list.SetItem(idx, 1, run_id)
            self.run_list.SetItem(idx, 2, base)
            
            self._run_ids.append(run_id)

        # Views
        self.views_list.Clear()
        self._view_ids.clear()
        self._view_run_ids.clear()

        if exp.views:
            for view_id, view in exp.views.items():
                self._view_ids.append(view_id)
                self._view_run_ids.append(list(view.run_ids))

                title = getattr(view, "title", view_id)
                if view.run_ids:
                    nicknames = [
                        exp.get_run_nickname(rid) for rid in view.run_ids
                    ]
                    runs_str = ", ".join(nicknames)
                else:
                    runs_str = "(no runs)"
                self.views_list.Append(f"{title}: {runs_str}")
        else:
            self.views_list.Append("(no views)")

    # --- internal helpers ---

    def _get_selected_run_ids(self) -> List[str]:
        ids: List[str] = []
        item = self.run_list.GetFirstSelected()
        while item != -1:
            if item < len(self._run_ids):
                ids.append(self._run_ids[item])
            item = self.run_list.GetNextSelected(item)
        return ids

    def _get_selected_view_info(self):
        """Return (view_id, run_ids) for the currently selected view, or (None, [])."""
        idx = self.views_list.GetSelection()
        if idx == wx.NOT_FOUND:
            return None, []
        if idx >= len(self._view_ids):
            return None, []
        view_id = self._view_ids[idx]
        run_ids = self._view_run_ids[idx] if idx < len(self._view_run_ids) else []
        return view_id, run_ids

    # --- context menu handlers for runs ---

    def on_context_menu_runs(self, event):
        menu = wx.Menu()
        item_current = menu.Append(wx.ID_ANY, "Add to current view")
        item_new = menu.Append(wx.ID_ANY, "Add to new view")
        item_rename = menu.Append(wx.ID_ANY, "Rename run")
        item_update = menu.Append(wx.ID_ANY, "Update from file")
        item_export = menu.Append(wx.ID_ANY, "Export Run...")
        
        # Check if derived run is selected
        run_ids = self._get_selected_run_ids()
        if len(run_ids) == 1 and self._experiment:
            run = self._experiment.get_run(run_ids[0])
            if run and run.run_type == RunType.DERIVED:
                item_formula = menu.Append(wx.ID_ANY, "Edit Formula...")
                self.Bind(wx.EVT_MENU, self._on_context_edit_formula, item_formula)

        menu.AppendSeparator()
        item_remove = menu.Append(wx.ID_ANY, "Remove run")

        self.Bind(wx.EVT_MENU, self._on_context_add_to_current, item_current)
        self.Bind(wx.EVT_MENU, self._on_context_add_to_new, item_new)
        self.Bind(wx.EVT_MENU, self._on_context_rename_run, item_rename)
        self.Bind(wx.EVT_MENU, self._on_context_update_from_file, item_update)
        self.Bind(wx.EVT_MENU, self._on_context_export_run, item_export)
        self.Bind(wx.EVT_MENU, self._on_context_remove_run, item_remove)

        self.PopupMenu(menu)
        menu.Destroy()

    def _on_context_edit_formula(self, event):
        if self._on_edit_formula is None:
            return
        run_ids = self._get_selected_run_ids()
        if len(run_ids) != 1:
            return

        run_id = run_ids[0]
        if not self._experiment:
            return

        run = self._experiment.get_run(run_id)
        if not run or run.run_type != RunType.DERIVED:
            return

        current_formula = run.metadata.get("formula", "")
        current_params = run.metadata.get("formula_params", {})

        dlg = DerivedRunFormulaDialog(self, current_formula, current_params)
        if dlg.ShowModal() == wx.ID_OK:
            new_formula, new_params = dlg.get_values()
            self._on_edit_formula(run_id, new_formula, new_params)
        dlg.Destroy()

    def _on_context_remove_run(self, event):
        if self._on_remove_run is None:
            return
        
        run_ids = self._get_selected_run_ids()
        
        if run_ids:
            # Confirm removal
            msg = f"Are you sure you want to remove {len(run_ids)} run(s)?\nThis will remove them from all views."
            dlg = wx.MessageDialog(self, msg, "Remove Run(s)", wx.YES_NO | wx.NO_DEFAULT | wx.ICON_WARNING)
            if dlg.ShowModal() == wx.ID_YES:
                self._on_remove_run(run_ids)
            dlg.Destroy()

    def _on_context_export_run(self, event):
        if self._on_export_run is None:
            return

        run_ids = self._get_selected_run_ids()
        if not run_ids:
            wx.MessageBox("Please select one or more runs to export.", "No Runs Selected", wx.OK | wx.ICON_INFORMATION)
            return

        self._on_export_run(run_ids)

    def _on_context_update_from_file(self, event):
        if self._on_update_from_file is None:
            return
        
        run_ids = self._get_selected_run_ids()
        if run_ids:
            self._on_update_from_file(run_ids)

    def _on_context_rename_run(self, event):
        """
        Request a run-rename operation for the selected run.
        The actual renaming is handled inline via a TextCtrl overlay.
        """
        if self._on_rename_run is None:
            return

        idx = self.run_list.GetFirstSelected()
        if idx == -1:
            return

        if idx >= len(self._run_ids):
            return

        self._begin_inline_rename(idx)

    def _begin_inline_rename(self, index: int) -> None:
        """
        Start inline renaming of the run at the given index by overlaying
        a TextCtrl on top of the corresponding run item.
        """
        if index < 0 or index >= len(self._run_ids):
            return

        # If we are already editing something, finish it first.
        if self._edit_ctrl is not None:
            self._finish_inline_rename(commit=True)

        self._editing_index = index

        # Determine the current nickname to prefill the editor.
        run_id = self._run_ids[index]
        if self._experiment is not None:
            current_name = self._experiment.get_run_nickname(run_id)
        else:
            current_name = self.run_list.GetItemText(index)

        # Get item rectangle for column 0 (Run Name)
        rect = self.run_list.GetItemRect(index, wx.LIST_RECT_LABEL)
        
        # Adjust position slightly for better visual alignment
        x = rect.x
        y = rect.y
        w = rect.width
        h = rect.height

        # Create a TextCtrl overlayed roughly on top of the item area.
        self._edit_ctrl = wx.TextCtrl(
            self.run_list,
            style=wx.TE_PROCESS_ENTER,
        )
        self._edit_ctrl.SetValue(current_name)
        self._edit_ctrl.SetSelection(-1, -1)  # select all text
        self._edit_ctrl.SetPosition((x, y))
        self._edit_ctrl.SetSize((w, h))
        self._edit_ctrl.SetFocus()

        # Commit on Enter, and also when the editor loses focus.
        self._edit_ctrl.Bind(wx.EVT_TEXT_ENTER, self._on_edit_enter)
        self._edit_ctrl.Bind(wx.EVT_KILL_FOCUS, self._on_edit_kill_focus)

    def _finish_inline_rename(self, commit: bool) -> None:
        """
        Complete the inline rename operation.

        If commit is True, the new nickname is sent to the main frame via
        the on_rename_run callback. In all cases, the editor is destroyed.
        """
        if self._edit_ctrl is None or self._editing_index is None:
            return

        index = self._editing_index
        new_name = self._edit_ctrl.GetValue().strip()

        self._edit_ctrl.Destroy()
        self._edit_ctrl = None
        self._editing_index = None

        if not commit:
            return
        if not new_name:
            return
        if index < 0 or index >= len(self._run_ids):
            return
        if self._on_rename_run is None:
            return

        run_id = self._run_ids[index]
        # Delegate actual model update to the main frame.
        self._on_rename_run(run_id, new_name)

    def _on_edit_enter(self, event):
        """Handle Enter key inside the inline editor."""
        self._finish_inline_rename(commit=True)

    def _on_edit_kill_focus(self, event):
        """
        Handle loss of focus for the inline editor.

        We treat this as a commit as well, to avoid leaving half-edited names.
        """
        self._finish_inline_rename(commit=True)

    def _on_context_add_to_current(self, event):
        if self._on_add_to_current_view is None:
            return
        run_ids = self._get_selected_run_ids()
        if run_ids:
            self._on_add_to_current_view(run_ids)

    def _on_context_add_to_new(self, event):
        if self._on_add_to_new_view is None:
            return
        run_ids = self._get_selected_run_ids()
        if run_ids:
            self._on_add_to_new_view(run_ids)

    # --- context menu handlers for views ---

    def on_context_menu_views(self, event):
        view_id, run_ids = self._get_selected_view_info()
        if view_id is None:
            return

        menu = wx.Menu()
        item_remove_view = menu.Append(wx.ID_ANY, "Remove view")
        item_duplicate_view = menu.Append(wx.ID_ANY, "Duplicate view")

        self.Bind(
            wx.EVT_MENU,
            lambda evt, vid=view_id: self._handle_view_remove(vid),
            item_remove_view,
        )
        self.Bind(
            wx.EVT_MENU,
            lambda evt, vid=view_id: self._handle_view_duplicate(vid),
            item_duplicate_view,
        )

        if run_ids:
            menu.AppendSeparator()
            for rid in run_ids:
                # Build label as nickname(id) if experiment is known
                if self._experiment is not None:
                    label_nick = self._experiment.get_run_nickname(rid)
                    label = f"{label_nick} ({rid})"
                else:
                    label = rid
                item = menu.Append(wx.ID_ANY, f"Delete {label} from view")
                self.Bind(
                    wx.EVT_MENU,
                    lambda evt, vid=view_id, rrid=rid: self._handle_view_remove_run(
                        vid, rrid
                    ),
                    item,
                )

        self.PopupMenu(menu)
        menu.Destroy()

    def _handle_view_remove(self, view_id: str) -> None:
        if self._on_view_remove is not None:
            self._on_view_remove(view_id)

    def _handle_view_duplicate(self, view_id: str) -> None:
        if self._on_view_duplicate is not None:
            self._on_view_duplicate(view_id)

    def _handle_view_remove_run(self, view_id: str, run_id: str) -> None:
        if self._on_view_remove_run is not None:
            self._on_view_remove_run(view_id, run_id)


class ExperimentPanel(wx.Panel):
    """
    Experiment tab: tree view of ExperimentSet metadata and runs.
    """

    def __init__(self, parent, on_edit_formula=None, on_edit_derived_props=None):
        super().__init__(parent)
        self._on_edit_formula = on_edit_formula
        self._on_edit_derived_props = on_edit_derived_props
        self._experiment: Optional[ExperimentSet] = None
        
        sizer = wx.BoxSizer(wx.VERTICAL)

        self.tree = wx.TreeCtrl(self, style=wx.TR_HAS_BUTTONS | wx.TR_DEFAULT_STYLE)
        sizer.Add(self.tree, 1, wx.EXPAND | wx.ALL, 4)

        self.SetSizer(sizer)
        
        self.tree.Bind(wx.EVT_TREE_ITEM_RIGHT_CLICK, self.on_tree_right_click)

    def refresh_from_experiment(self, exp: ExperimentSet) -> None:
        """Rebuild the tree from the given ExperimentSet."""
        self._experiment = exp
        self.tree.DeleteAllItems()

        root = self.tree.AddRoot(f"Experiment {exp.id}")

        # Experiment-level metadata
        md_node = self.tree.AppendItem(root, "Metadata")
        if exp.metadata:
            for k, v in exp.metadata.items():
                self.tree.AppendItem(md_node, f"{k}: {v}")
        else:
            self.tree.AppendItem(md_node, "(no metadata)")

        # Runs subtree: RunNN as node label, with id / file / metadata as children
        runs_node = self.tree.AppendItem(root, "Runs")
        if exp.runs:
            for run_id, run in exp.runs.items():
                nickname = exp.get_run_nickname(run_id)
                run_item = self.tree.AppendItem(runs_node, nickname)
                self.tree.SetItemData(run_item, run_id) # Store run_id for context menu

                base = os.path.basename(run.source_path) if run.source_path else "(derived)"
                self.tree.AppendItem(run_item, f"id: {run_id}")
                self.tree.AppendItem(run_item, f"file: {base}")

                md = run.metadata or {}
                if md:
                    md_node_run = self.tree.AppendItem(run_item, "metadata")
                    for k, v in md.items():
                        # nickname already used as node label
                        if k == "nickname":
                            continue
                        self.tree.AppendItem(md_node_run, f"{k}: {v}")
                else:
                    self.tree.AppendItem(run_item, "(no metadata)")
        else:
            self.tree.AppendItem(runs_node, "(no runs)")

        # Views subtree: each view with id and attached run nicknames
        views_node = self.tree.AppendItem(root, "Views")
        if exp.views:
            for view_id, view in exp.views.items():
                title = getattr(view, "title", view_id)
                view_item = self.tree.AppendItem(views_node, title)
                self.tree.AppendItem(view_item, f"id: {view_id}")

                if view.run_ids:
                    nicknames = [exp.get_run_nickname(rid) for rid in view.run_ids]
                    runs_str = ", ".join(nicknames)
                    self.tree.AppendItem(view_item, f"runs: {runs_str}")
                else:
                    self.tree.AppendItem(view_item, "runs: (no runs)")
        else:
            self.tree.AppendItem(views_node, "(no views)")

        self.tree.ExpandAll()

    def on_tree_right_click(self, event):
        item = event.GetItem()
        if not item.IsOk():
            return
            
        run_id = self.tree.GetItemData(item)
        if not run_id or not self._experiment:
            return
            
        run = self._experiment.get_run(run_id)
        if not run or run.run_type != RunType.DERIVED:
            return
            
        menu = wx.Menu()
        item_formula = menu.Append(wx.ID_ANY, "Edit Formula...")
        item_props = menu.Append(wx.ID_ANY, "Edit Derived Props...")
        
        self.Bind(wx.EVT_MENU, lambda e: self._on_menu_edit_formula(run_id), item_formula)
        self.Bind(wx.EVT_MENU, lambda e: self._on_menu_edit_props(run_id), item_props)
        
        self.PopupMenu(menu)
        menu.Destroy()

    def _on_menu_edit_formula(self, run_id):
        run = self._experiment.get_run(run_id)
        current_formula = run.metadata.get("formula", "")
        current_params = run.metadata.get("formula_params", {})
        
        dlg = DerivedRunFormulaDialog(self, current_formula, current_params)
        if dlg.ShowModal() == wx.ID_OK:
            new_formula, new_params = dlg.get_values()
            if self._on_edit_formula:
                self._on_edit_formula(run_id, new_formula, new_params)
        dlg.Destroy()

    def _on_menu_edit_props(self, run_id):
        run = self._experiment.get_run(run_id)
        # Use metadata values as base
        current_n = run.metadata.get("default_n_points", 100)
        current_auto = run.metadata.get("default_autorange", False)
        current_range = run.metadata.get("default_range", (0, 100))
        
        dlg = DerivedRunPropsDialog(self, current_n, current_auto, current_range)
        if dlg.ShowModal() == wx.ID_OK:
            new_n, new_auto, new_range = dlg.get_values()
            if self._on_edit_derived_props:
                self._on_edit_derived_props(run_id, new_n, new_auto, new_range)
        dlg.Destroy()


class LogPanel(wx.Panel):
    """
    Log tab: shows log messages from the application.
    """

    def __init__(self, parent):
        super().__init__(parent)
        sizer = wx.BoxSizer(wx.VERTICAL)
        self.text = wx.TextCtrl(
            self, style=wx.TE_MULTILINE | wx.TE_READONLY | wx.TE_RICH2
        )
        sizer.Add(self.text, 1, wx.EXPAND | wx.ALL, 4)
        self.SetSizer(sizer)

    def append_log(self, msg: str) -> None:
        self.text.AppendText(msg + "\n")

