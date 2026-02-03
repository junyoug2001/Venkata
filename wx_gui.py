"""
Minimal VESTA-like GUI skeleton using wxPython, now wired to the
data_structure ExperimentSet / Run model for early visualization tests.

Usage
-----
python wx_gui.py file1.csv file2.csv

- Creates a single ExperimentSet.
- Builds up to two Run objects from the provided files.
- Populates:
    * Files tab   : simple directory browser + file preview.
    * Runs tab    : list of runs in the current ExperimentSet.
    * Experiment  : tree view of experiment metadata and runs.
    * Log         : textual log of operations.

Layout
------
- Main horizontal splitter:
    [ Left controls ] | [ Right view notebook ]

- Left controls:
    - Vertical splitter:
        * Top:   Notebook (Files / Runs / Experiment / Log)
        * Bottom Notebook (Preview / Curve Fit)

- Right view notebook:
    - Multiple "View N" tabs (initially one tab).
    - Menu: View -> New View Tab (adds a new tab).

Notes
-----
- This is still a structural GUI; plotting is not implemented yet.
- The purpose is to validate the early pipeline:
    CLI args -> ExperimentSet / Runs -> GUI panels.
"""

from __future__ import annotations

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
)
from plotting import RamanPlotter2d, AngularPlotter, SlicePlotter


# -----------------------------
# Left-top notebook pages
# -----------------------------


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

    - Upper section: list of runs in the current ExperimentSet (with checkboxes).
    - Lower section: list of views and which runs are attached to each view.
    - A splitter between runs and views allows adjusting their relative heights.
    - Right-click on the runs checklist:
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

        self.checklist = wx.CheckListBox(top_panel, choices=[])
        top_sizer.Add(self.checklist, 1, wx.EXPAND | wx.ALL, 4)

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
        self.checklist.Bind(wx.EVT_CONTEXT_MENU, self.on_context_menu_runs)
        self.views_list.Bind(wx.EVT_CONTEXT_MENU, self.on_context_menu_views)

    def _set_initial_sash(self):
        """Set an initial runs:views split of about 60%:40%."""
        size = self.splitter.GetClientSize()
        if size.height > 0:
            self.splitter.SetSashPosition(int(size.height * 0.6))

    # --- public API ---

    def refresh_from_experiment(self, exp: ExperimentSet) -> None:
        """Rebuild the runs and views lists from the given ExperimentSet."""
        self._experiment = exp
        # Runs
        self.checklist.Clear()
        self._run_ids.clear()

        for run_id, run in exp.runs.items():
            base = os.path.basename(run.source_path)
            nickname = exp.get_run_nickname(run_id)
            # Representative text: nickname (id) | filename
            label = f"{nickname} ({run_id})  |  {base}"
            self.checklist.Append(label)
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

    def _get_checked_run_ids(self) -> List[str]:
        ids: List[str] = []
        for i in range(self.checklist.GetCount()):
            if self.checklist.IsChecked(i) and i < len(self._run_ids):
                ids.append(self._run_ids[i])
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

        self.Bind(wx.EVT_MENU, self._on_context_add_to_current, item_current)
        self.Bind(wx.EVT_MENU, self._on_context_add_to_new, item_new)
        self.Bind(wx.EVT_MENU, self._on_context_rename_run, item_rename)
        self.Bind(wx.EVT_MENU, self._on_context_update_from_file, item_update)
        self.Bind(wx.EVT_MENU, self._on_context_export_run, item_export)

        self.PopupMenu(menu)
        menu.Destroy()

    def _on_context_export_run(self, event):
        if self._on_export_run is None:
            return

        checked_run_ids = self._get_checked_run_ids()
        if not checked_run_ids:
            wx.MessageBox("Please check one or more runs to export.", "No Runs Checked", wx.OK | wx.ICON_INFORMATION)
            return

        self._on_export_run(checked_run_ids)

    def _on_context_update_from_file(self, event):
        if self._on_update_from_file is None:
            return
        
        run_ids = self._get_checked_run_ids()
        if not run_ids:
            # Fallback: use the selected item if any
            idx = self.checklist.GetSelection()
            if idx != wx.NOT_FOUND and idx < len(self._run_ids):
                run_ids = [self._run_ids[idx]]
        
        if run_ids:
            self._on_update_from_file(run_ids)

    def _on_context_rename_run(self, event):
        """
        Request a run-rename operation for the selected (or first checked) run.
        The actual renaming is handled inline via a TextCtrl overlay.
        """
        if self._on_rename_run is None:
            return

        # Prefer the currently selected item; if none, fall back to first checked.
        idx = self.checklist.GetSelection()
        if idx == wx.NOT_FOUND:
            for i in range(self.checklist.GetCount()):
                if self.checklist.IsChecked(i):
                    idx = i
                    break

        if idx == wx.NOT_FOUND or idx >= len(self._run_ids):
            return

        self._begin_inline_rename(idx)

    def _begin_inline_rename(self, index: int) -> None:
        """
        Start inline renaming of the run at the given index by overlaying
        a TextCtrl on top of the corresponding checklist item.
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
            # Fallback: parse from the label before " (" if available.
            label = self.checklist.GetString(index)
            current_name = label.split("(", 1)[0].strip()

        # Approximate the item rectangle using the item index and font metrics,
        # since wx.CheckListBox does not provide GetItemRect on all platforms.
        client_rect = self.checklist.GetClientRect()
        item_height = self.checklist.GetCharHeight() + 4  # small padding

        x = client_rect.x + 2
        y = client_rect.y + 2 + index * item_height
        w = max(client_rect.width - 4, 40)
        h = item_height

        # Create a TextCtrl overlayed roughly on top of the item area.
        self._edit_ctrl = wx.TextCtrl(
            self.checklist,
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
        run_ids = self._get_checked_run_ids()
        if run_ids:
            self._on_add_to_current_view(run_ids)

    def _on_context_add_to_new(self, event):
        if self._on_add_to_new_view is None:
            return
        run_ids = self._get_checked_run_ids()
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

    def __init__(self, parent):
        super().__init__(parent)
        sizer = wx.BoxSizer(wx.VERTICAL)

        self.tree = wx.TreeCtrl(self, style=wx.TR_HAS_BUTTONS | wx.TR_DEFAULT_STYLE)
        sizer.Add(self.tree, 1, wx.EXPAND | wx.ALL, 4)

        self.SetSizer(sizer)

    def refresh_from_experiment(self, exp: ExperimentSet) -> None:
        """Rebuild the tree from the given ExperimentSet."""
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

                base = os.path.basename(run.source_path)
                self.tree.AppendItem(run_item, f"id: {run_id}")
                self.tree.AppendItem(run_item, f"file: {base}")

                md = run.metadata or {}
                if md:
                    md_node_run = self.tree.AppendItem(run_item, "metadata")
                    for k, v in md.items():
                        # nickname은 이미 노드 이름으로 쓰였으므로 중복하지 않는다.
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


# -----------------------------
# Left-bottom notebook pages
# -----------------------------


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
    def __init__(self, parent, on_x_range_changed=None, on_y_range_changed=None, on_vlim_changed=None, on_reset=None, on_unit_changed=None, on_cmap_changed=None):
        super().__init__(parent)
        
        self._on_x_range_changed = on_x_range_changed
        self._on_y_range_changed = on_y_range_changed
        self._on_vlim_changed = on_vlim_changed
        self._on_reset = on_reset
        self._on_unit_changed = on_unit_changed
        self._on_cmap_changed = on_cmap_changed
        
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

    def on_unit_change(self, event):
        rb = event.GetEventObject()
        new_unit = rb.GetLabel()
        if new_unit == self.x_unit:
            return

        self.x_unit = new_unit
        # self.x_unit_label.SetLabel(self.x_unit) # removed label

        if self._on_unit_changed:
            self._on_unit_changed(new_unit)

    def on_x_range_enter(self, event):
        if self._on_x_range_changed:
            try:
                xmin = float(self.x_min_text.GetValue())
                xmax = float(self.x_max_text.GetValue())
                self._on_x_range_changed(xmin, xmax, self.x_unit)
            except ValueError:
                wx.MessageBox("Invalid X range. Please enter numeric values.", "Error", wx.OK | wx.ICON_ERROR)

    def on_y_range_enter(self, event):
        if self._on_y_range_changed:
            try:
                ymin = float(self.y_min_text.GetValue())
                ymax = float(self.y_max_text.GetValue())
                self._on_y_range_changed(ymin, ymax)
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
        
        if self._on_vlim_changed:
            self._on_vlim_changed(vmin, vmax)

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
            
            if self._on_vlim_changed:
                self._on_vlim_changed(vmin, vmax)
        except ValueError:
            pass

    def on_cmap_change(self, event):
        cmap = self.choice_cmap.GetStringSelection()
        if self._on_cmap_changed:
            self._on_cmap_changed(cmap)
    
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


# -----------------------------
# Right-side view tab content
# -----------------------------


class RamanToolbar(NavigationToolbar):
    """
    Thin wrapper around NavigationToolbar so we can hook the Home button.

    When the user presses Home, we first let matplotlib restore the original
    view (axis limits etc.), and then we ask the owning ViewPanel to
    re-apply the last chosen slice so that Plot B/C are not left blank.
    """
    def __init__(self, canvas, owner_panel: "ViewPanel"):
        self._owner_panel = owner_panel
        super().__init__(canvas)

    def home(self, *args, **kwargs):
        super().home(*args, **kwargs)
        if hasattr(self._owner_panel, "go_home"):
            self._owner_panel.go_home()
        if hasattr(self._owner_panel, "_refresh_after_home"):
            self._owner_panel._refresh_after_home()


class ViewPanel(wx.Panel):
    """
    One 'View' tab, showing up to one active Run as three horizontal plots:

    - Plot A: 2D color map (angle vs Raman shift).
    - Plot B: Angular slice from Plot A.
    - Plot C: Spectral slice from Plot A.

    Refactored to use plotting.py classes.
    """

    def __init__(self, parent, view_label: str, on_limits_changed=None, on_vlim_changed=None):
        super().__init__(parent)

        self.view_label = view_label
        self.on_limits_changed = on_limits_changed
        self.on_vlim_changed = on_vlim_changed
        self.current_run_id: Optional[str] = None
        self.current_run: Optional[Run] = None
        self._experiment: Optional[ExperimentSet] = None

        # Matplotlib figure and canvas
        self.figure = Figure(figsize=(9, 4.5))
        self.canvas = FigureCanvas(self, -1, self.figure)

        # Plotter instances (initialized in _draw_runs)
        self.plotterA1: Optional[RamanPlotter2d] = None
        self.plotterB1: Optional[AngularPlotter] = None
        self.plotterC1: Optional[SlicePlotter] = None
        
        self.plotterA2: Optional[RamanPlotter2d] = None
        self.plotterB2: Optional[AngularPlotter] = None
        self.plotterC2: Optional[SlicePlotter] = None

        # Click event connection id
        self._cid_click = None
        self._cid_resize = None

        # Highlight mode: "none", "click", "line_profile", "peak_fit"
        self.highlight_mode: str = "click"

        # Angle slice rendering type for Plot B: "polar" or "cartesian"
        self.angle_slice_type: str = "polar"
        self._last_drawn_runs: List[Run] = []

        # Guard flag to avoid recursive callbacks when syncing zoom between axes
        self._syncing_limits: bool = False
        self._limit_cb_ids: list[tuple[object, int]] = []

        # Persistent selection state (indices)
        self._sel_idx1: Optional[tuple[int, int]] = None
        self._sel_idx2: Optional[tuple[int, int]] = None
        
        # Track contrast state
        self._contrast_percent = (0.0, 100.0)
        self.current_cmap = "OrRd"

        # Optional second run to visualize in the bottom row
        self._second_run: Optional[Run] = None

        # Layout
        sizer = wx.BoxSizer(wx.VERTICAL)
        label = wx.StaticText(self, label=view_label)
        label.SetForegroundColour(wx.Colour(50, 50, 50))
        sizer.Add(label, 0, wx.ALL, 4)

        self.toolbar = RamanToolbar(self.canvas, self)
        self.toolbar.Realize()
        sizer.Add(self.toolbar, 0, wx.EXPAND | wx.LEFT | wx.RIGHT, 4)

        sizer.Add(self.canvas, 1, wx.EXPAND | wx.ALL, 4)
        self.SetSizer(sizer)

        self._init_empty_figure()
        self.canvas.Bind(wx.EVT_CONTEXT_MENU, self._on_context_menu)

    def _refresh_after_home(self) -> None:
        self._apply_selection_from_indices(reason="home")
    
    def go_home(self):
        # Let the plotters handle standard view reset if needed,
        # but standard Toolbar Home usually works on the axes stack.
        # We might need to reset contrast or re-apply limits if they were manually set.
        pass

    # --- Public API for external controls ---

    def get_plot_limits(self):
        if self.plotterA1 and self.plotterA1.ax:
            return self.plotterA1.ax.get_xlim(), self.plotterA1.ax.get_ylim()
        return (None, None), (None, None)

    def set_x_range(self, xmin, xmax, unit):
        # We delegate unit conversion handling to the caller or do it here.
        # The plotter expects whatever unit it was rendered with (usually cm-1).
        if not self.plotterA1 or not self.plotterA1.ax:
            return
        
        # If input is meV, convert to cm-1 if that's the base unit
        if unit == 'meV':
            xmin = xmin / EV_PER_CM1 / 1000.0
            xmax = xmax / EV_PER_CM1 / 1000.0
        
        self.plotterA1.ax.set_xlim(xmin, xmax)
        self.canvas.draw_idle()

    def set_y_range(self, ymin, ymax):
        if self.plotterA1 and self.plotterA1.ax:
            self.plotterA1.ax.set_ylim(ymin, ymax)
            self.canvas.draw_idle()

    def set_vlim(self, vmin, vmax):
        self._contrast_percent = (vmin, vmax)
        if self.plotterA1:
            self.plotterA1.set_contrast(vmin, vmax)
        if self.plotterA2:
            self.plotterA2.set_contrast(vmin, vmax)

    def set_colormap(self, cmap_name: str):
        if self.plotterA1 and self.plotterA1.mesh:
            self.plotterA1.mesh.set_cmap(cmap_name)
        if self.plotterA2 and self.plotterA2.mesh:
            self.plotterA2.mesh.set_cmap(cmap_name)
        self.canvas.draw_idle()

    def get_colormap(self) -> str:
        if self.plotterA1 and self.plotterA1.mesh:
            return self.plotterA1.mesh.get_cmap().name
        return 'OrRd'

    def get_plot_config(self) -> Dict[str, Any]:
        xlim, ylim = self.get_plot_limits()
        if xlim[0] is None: xlim = (0, 1)
        if ylim[0] is None: ylim = (0, 1)
        
        return {
            'xlim': xlim,
            'ylim': ylim,
            'vmin_p': self._contrast_percent[0],
            'vmax_p': self._contrast_percent[1],
            'cmap': self.get_colormap()
        }

    # --- Internal Logic ---

    def _notify_limits_changed(self):
        if self.on_limits_changed and self.plotterA1 and self.plotterA1.ax:
            self.on_limits_changed(self.plotterA1.ax.get_xlim(), self.plotterA1.ax.get_ylim())

    def _disconnect_limit_sync_callbacks(self) -> None:
        if not getattr(self, "_limit_cb_ids", None):
            return
        for registry, cid in self._limit_cb_ids:
            try:
                registry.disconnect(cid)
            except Exception:
                pass
        self._limit_cb_ids.clear()

    def _rebind_limit_sync_callbacks(self) -> None:
        """
        Re-attach axis synchronization callbacks. 
        Note: We access .ax directly from the plotters.
        """
        self._disconnect_limit_sync_callbacks()
        
        # Helpers to get axes safely
        axA1 = self.plotterA1.ax if self.plotterA1 else None
        axB1 = self.plotterB1.ax if self.plotterB1 else None
        axC1 = self.plotterC1.ax if self.plotterC1 else None
        
        axA2 = self.plotterA2.ax if self.plotterA2 else None
        axB2 = self.plotterB2.ax if self.plotterB2 else None
        axC2 = self.plotterC2.ax if self.plotterC2 else None

        # ---------------------------------------------------------
        # Sync Logic Groups
        # ---------------------------------------------------------
        
        # Group 1: Shift (X-axis) -> A1, C1, A2, C2
        def sync_shift(source_ax):
            if self._syncing_limits: return
            self._syncing_limits = True
            try:
                xlim = source_ax.get_xlim()
                
                # Apply to A1, C1
                if axA1 and axA1 != source_ax: axA1.set_xlim(xlim)
                if axC1 and axC1 != source_ax: axC1.set_xlim(xlim)
                
                # Apply to A2, C2
                if axA2 and axA2 != source_ax: axA2.set_xlim(xlim)
                if axC2 and axC2 != source_ax: axC2.set_xlim(xlim)
                
            finally: 
                self._syncing_limits = False
            self._notify_limits_changed()

        # Group 2: Angle -> A1(y), A2(y), B1(x), B2(x)
        def sync_angle(source_ax, is_y_axis=True):
            if self._syncing_limits: return
            self._syncing_limits = True
            try:
                # Get the canonical angle range
                if is_y_axis:
                    angle_lim = source_ax.get_ylim()
                else:
                    angle_lim = source_ax.get_xlim()

                # Apply to A1 (Y), A2 (Y)
                if axA1 and axA1 != source_ax: axA1.set_ylim(angle_lim)
                if axA2 and axA2 != source_ax: axA2.set_ylim(angle_lim)

                # Apply to B1 (X), B2 (X) - ONLY if not polar
                if self.angle_slice_type != "polar":
                    if axB1 and axB1 != source_ax: axB1.set_xlim(angle_lim)
                    if axB2 and axB2 != source_ax: axB2.set_xlim(angle_lim)
            
            finally:
                self._syncing_limits = False
            self._notify_limits_changed()

        # ---------------------------------------------------------
        # Connect Callbacks
        # ---------------------------------------------------------

        # Run 1
        if axA1:
            # A1 X -> Shift
            cid = axA1.callbacks.connect("xlim_changed", sync_shift)
            self._limit_cb_ids.append((axA1.callbacks, cid))
            # A1 Y -> Angle
            cid = axA1.callbacks.connect("ylim_changed", lambda ax: sync_angle(ax, is_y_axis=True))
            self._limit_cb_ids.append((axA1.callbacks, cid))

        if axC1:
            # C1 X -> Shift
            cid = axC1.callbacks.connect("xlim_changed", sync_shift)
            self._limit_cb_ids.append((axC1.callbacks, cid))
        
        if axB1 and self.angle_slice_type != "polar":
            # B1 X -> Angle
            cid = axB1.callbacks.connect("xlim_changed", lambda ax: sync_angle(ax, is_y_axis=False))
            self._limit_cb_ids.append((axB1.callbacks, cid))

        # Run 2
        if axA2:
            # A2 X -> Shift
            cid = axA2.callbacks.connect("xlim_changed", sync_shift)
            self._limit_cb_ids.append((axA2.callbacks, cid))
            # A2 Y -> Angle
            cid = axA2.callbacks.connect("ylim_changed", lambda ax: sync_angle(ax, is_y_axis=True))
            self._limit_cb_ids.append((axA2.callbacks, cid))
        
        if axC2:
            # C2 X -> Shift
            cid = axC2.callbacks.connect("xlim_changed", sync_shift)
            self._limit_cb_ids.append((axC2.callbacks, cid))
            
        if axB2 and self.angle_slice_type != "polar":
            # B2 X -> Angle
            cid = axB2.callbacks.connect("xlim_changed", lambda ax: sync_angle(ax, is_y_axis=False))
            self._limit_cb_ids.append((axB2.callbacks, cid))


    def _init_empty_figure(self, message: str = "No data") -> None:
        self.figure.clf()
        ax = self.figure.add_subplot(1, 1, 1)
        ax.text(0.5, 0.5, message, ha="center", va="center", transform=ax.transAxes)
        ax.set_axis_off()
        
        self.plotterA1 = self.plotterB1 = self.plotterC1 = None
        self.plotterA2 = self.plotterB2 = self.plotterC2 = None
        
        self._disconnect_limit_sync_callbacks()
        
        if self._cid_click:
            self.canvas.mpl_disconnect(self._cid_click)
            self._cid_click = None
        if self._cid_resize:
            self.canvas.mpl_disconnect(self._cid_resize)
            self._cid_resize = None

    def _apply_selection_from_indices(self, *, reason: str = "") -> None:
        if not self.plotterA1 or self._sel_idx1 is None:
            return

        # Prevent sync callbacks from propagating the "reset" that happens during render()
        self._syncing_limits = True
        try:
            def _update_run_plots(run: Run, ix: int, iy: int, pA: RamanPlotter2d, pB: AngularPlotter, pC: SlicePlotter):
                # 1. Get Data
                shift = np.asarray(run.shift_cm1, dtype=float)
                angles = np.asarray(run.angle_values, dtype=float)
                I = np.asarray(run.intensity_2d, dtype=float)
                
                # Ensure orientation (transpose if needed to match axes)
                if I.shape != (angles.size, shift.size):
                    if I.shape == (shift.size, angles.size):
                        I = I.T
                    else:
                        return # Data mismatch

                # 2. Get coords from index
                x_sel, y_sel = pA.get_coords_from_index(ix, iy)
                
                # 3. Update Highlights
                pA.set_highlight(x_sel, y_sel, visible=True)
                pB.set_highlight(y_sel, visible=True)
                pC.set_highlight(x_sel, visible=True)
                
                # 4. Update Slice Data
                # Plot B: Intensity vs Angle at selected Shift (column ix)
                if ix < I.shape[1]:
                    pB.render(angles, I[:, ix], mode=self.angle_slice_type, 
                              title=f"Angular Slice @ {x_sel:.1f} cm$^{{-1}}$")
                
                # Plot C: Intensity vs Shift at selected Angle (row iy)
                if iy < I.shape[0]:
                    # SlicePlotter creates a new line in render(), which is efficient enough here
                    pC.render(shift, I[iy, :], 
                              title=f"Spectral Slice @ {y_sel:.1f} deg")

                # 5. Restore Zoom (Sync limits from A to B/C)
                # Since render() cleared the axes, we must re-apply the current zoom from pA.
                if pA.ax:
                    # Sync C (Shift)
                    if pC.ax:
                        pC.ax.set_xlim(pA.ax.get_xlim())
                    
                    # Sync B (Angle) if Cartesian
                    if self.angle_slice_type != "polar" and pB.ax:
                        pB.ax.set_xlim(pA.ax.get_ylim())

            # Update Run 1
            _update_run_plots(self.current_run, self._sel_idx1[0], self._sel_idx1[1], 
                              self.plotterA1, self.plotterB1, self.plotterC1)

            # Update Run 2
            if self._second_run and self.plotterA2:
                # Sync indices if needed
                if self._sel_idx2 is None:
                    # Naive sync: same indices (assuming same grid)
                    # Or physical sync? Let's do physical sync logic
                    # For now, let's reuse idx1 if idx2 is missing, assuming close grids
                    self._sel_idx2 = self._sel_idx1
                
                _update_run_plots(self._second_run, self._sel_idx2[0], self._sel_idx2[1],
                                  self.plotterA2, self.plotterB2, self.plotterC2)
        
        finally:
            self._syncing_limits = False

        self.figure.subplots_adjust(top=0.862)
        self.canvas.draw_idle()

    def _draw_runs(self, runs: List[Run]) -> None:
        if not runs:
            self._init_empty_figure(message="No runs to draw.")
            return

        runs = runs[:2]
        self._last_drawn_runs = list(runs)
        self.figure.clf()

        # Layout
        gs = self.figure.add_gridspec(
            2, 3, height_ratios=[1.0, 1.0], width_ratios=[2.0, 1.0, 1.0],
            hspace=0.6, wspace=0.5, left=0.1, right=0.95
        )

        # Helper for secondary axis (cm-1 -> meV)
        def cm_to_mev(x): return x * EV_PER_CM1 * 1000
        def mev_to_cm(x): return x / (EV_PER_CM1 * 1000)
        
        # --- Run 1 ---
        run0 = runs[0]
        axA1 = self.figure.add_subplot(gs[0, 0])
        axB1 = self.figure.add_subplot(gs[0, 1], projection="polar" if self.angle_slice_type == "polar" else None)
        axC1 = self.figure.add_subplot(gs[0, 2])

        self.plotterA1 = RamanPlotter2d(axA1)
        self.plotterB1 = AngularPlotter(axB1)
        self.plotterC1 = SlicePlotter(axC1)

        nickname0 = self._experiment.get_run_nickname(run0.id) if self._experiment else run0.nickname
        self.plotterA1.render(
            run0.shift_cm1, run0.angle_values, run0.intensity_2d,
            title=f"{nickname0}: 2D map",
            cmap=self.current_cmap,
            x_unit_conversion=(cm_to_mev, mev_to_cm)
        )
        self.plotterA1.set_highlight_mode(self.highlight_mode)

        # --- Run 2 ---
        if len(runs) > 1:
            run1 = runs[1]
            axA2 = self.figure.add_subplot(gs[1, 0])
            axB2 = self.figure.add_subplot(gs[1, 1], projection="polar" if self.angle_slice_type == "polar" else None)
            axC2 = self.figure.add_subplot(gs[1, 2])

            self.plotterA2 = RamanPlotter2d(axA2)
            self.plotterB2 = AngularPlotter(axB2)
            self.plotterC2 = SlicePlotter(axC2)

            nickname1 = self._experiment.get_run_nickname(run1.id) if self._experiment else run1.nickname
            self.plotterA2.render(
                run1.shift_cm1, run1.angle_values, run1.intensity_2d,
                title=f"{nickname1}: 2D map",
                cmap=self.current_cmap,
                x_unit_conversion=(cm_to_mev, mev_to_cm)
            )
            self.plotterA2.set_highlight_mode(self.highlight_mode)
        else:
            self.plotterA2 = self.plotterB2 = self.plotterC2 = None
            # Fill empty space
            self.figure.add_subplot(gs[1, 0]).set_axis_off()
            self.figure.add_subplot(gs[1, 1]).set_axis_off()
            self.figure.add_subplot(gs[1, 2]).set_axis_off()

        # Connect Callbacks
        self._rebind_limit_sync_callbacks()
        
        if self._cid_click: self.canvas.mpl_disconnect(self._cid_click)
        self._cid_click = self.canvas.mpl_connect("button_press_event", self._on_canvas_click)
        
        self.figure.subplots_adjust(top=0.862)
        self.canvas.draw_idle()

    def _on_canvas_click(self, event):
        if event.button == 3: # Right click
            if (self.plotterB1 and event.inaxes == self.plotterB1.ax) or \
               (self.plotterB2 and event.inaxes == self.plotterB2.ax):
                self._popup_angle_slice_type_menu(event)
            return

        # Determine which plotter was clicked
        clicked_plotter = None
        is_run1 = False
        
        if self.plotterA1 and event.inaxes == self.plotterA1.ax:
            clicked_plotter = self.plotterA1
            is_run1 = True
        elif self.plotterA2 and event.inaxes == self.plotterA2.ax:
            clicked_plotter = self.plotterA2
            is_run1 = False
        
        if not clicked_plotter:
            return

        # Get indices from plotter
        indices = clicked_plotter.get_index_at(event.xdata, event.ydata)
        if not indices:
            return
        ix, iy = indices

        # Update Indices state
        if is_run1:
            self._sel_idx1 = (ix, iy)
            # Optional: Map to idx2 via physical coords if needed
            self._sel_idx2 = (ix, iy) # Simple sync for now
        else:
            self._sel_idx2 = (ix, iy)
            self._sel_idx1 = (ix, iy) # Simple sync

        self._apply_selection_from_indices(reason="click")

    def _popup_angle_slice_type_menu(self, mpl_event) -> None:
        menu = wx.Menu()
        sub = wx.Menu()
        item_cart = sub.AppendRadioItem(wx.ID_ANY, "Cartesian")
        item_polar = sub.AppendRadioItem(wx.ID_ANY, "Polar")
        
        if self.angle_slice_type == "polar": item_polar.Check(True)
        else: item_cart.Check(True)

        self.Bind(wx.EVT_MENU, lambda e: self._set_angle_slice_type("cartesian"), item_cart)
        self.Bind(wx.EVT_MENU, lambda e: self._set_angle_slice_type("polar"), item_polar)
        menu.AppendSubMenu(sub, "Angle slice type")
        
        self.PopupMenu(menu)
        menu.Destroy()

    def _set_angle_slice_type(self, mode: str) -> None:
        if mode == self.angle_slice_type: return
        self.angle_slice_type = mode
        # Re-draw everything because axes projection needs to change
        self._draw_runs(self._last_drawn_runs)
        self._apply_selection_from_indices()

    def _on_context_menu(self, event):
        menu = wx.Menu()
        highlight_menu = wx.Menu()
        
        modes = ["none", "click", "line_profile", "peak_fit"]
        for m in modes:
            item = highlight_menu.AppendRadioItem(wx.ID_ANY, m.capitalize())
            if self.highlight_mode == m: item.Check(True)
            self.Bind(wx.EVT_MENU, lambda e, mode=m: self._set_highlight_mode(mode), item)
            
        menu.AppendSubMenu(highlight_menu, "Highlight")
        self.PopupMenu(menu)
        menu.Destroy()

    def _set_highlight_mode(self, mode: str):
        self.highlight_mode = mode
        if self.plotterA1: self.plotterA1.set_highlight_mode(mode)
        if self.plotterA2: self.plotterA2.set_highlight_mode(mode)
        self.canvas.draw_idle()

    def set_view_model(self, experiment: ExperimentSet, view_state: ViewState) -> None:
        self._experiment = experiment
        self.current_run_id = None
        self.current_run = None
        self._second_run = None
        self._sel_idx1 = (0, 0)
        self._sel_idx2 = (0, 0)

        runs: List[Run] = []
        for rid in view_state.run_ids:
            r = experiment.runs.get(rid)
            if r and r.intensity_2d is not None:
                runs.append(r)
                if len(runs) == 2: break

        if not runs:
            self._init_empty_figure("No runs with 2D data.")
            self.canvas.draw_idle()
            return

        self.current_run = runs[0]
        self.current_run_id = runs[0].id
        self._second_run = runs[1] if len(runs) > 1 else None

        self._draw_runs(runs)



# -----------------------------
# Main frame
# -----------------------------


class MainFrame(wx.Frame):
    def __init__(self, initial_files: Optional[List[str]] = None):
        super().__init__(
            None,
            title="Venkata - GUI Based Raman Analysis",
            size=(1400, 800),
        )

        # Core experiment model
        self.experiment = ExperimentSet(id=new_experiment_id())

        # Simple counter for naming view tabs
        self._next_view_index = 1

        # Map notebook page index -> view_id in ExperimentSet.views
        self._view_page_to_id = {}
        # Map view_id -> ViewPanel instance
        self._view_id_to_panel = {}

        self._build_menu_bar()
        self._build_layout()

        # Status bar
        self.CreateStatusBar()

        # Load any initial files passed from CLI
        if initial_files:
            self.load_initial_files(initial_files)

        self.Centre()
        self.Show()

    # -------- menu --------

    def _build_menu_bar(self):
        menubar = wx.MenuBar()

        # File menu
        file_menu = wx.Menu()
        item_open = file_menu.Append(wx.ID_OPEN, "Open Experiment...\tCtrl-O")
        item_save = file_menu.Append(wx.ID_SAVE, "Save Experiment...\tCtrl-S")
        file_menu.AppendSeparator()
        item_import = file_menu.Append(wx.ID_ANY, "Import Run...\tCtrl-I")
        item_merge = file_menu.Append(wx.ID_ANY, "Merge Run...\tCtrl-Shift-M")
        file_menu.AppendSeparator()
        item_quit = file_menu.Append(wx.ID_EXIT, "Quit\tCtrl-Q")
        
        self.Bind(wx.EVT_MENU, self.on_open_experiment, item_open)
        self.Bind(wx.EVT_MENU, self.on_save_experiment, item_save)
        self.Bind(wx.EVT_MENU, self.on_import_run_dialog, item_import)
        self.Bind(wx.EVT_MENU, self.on_merge_run_dialog, item_merge)
        self.Bind(wx.EVT_MENU, self.on_quit, item_quit)
        menubar.Append(file_menu, "&File")

        # View menu
        view_menu = wx.Menu()
        item_new_view = view_menu.Append(wx.ID_ANY, "New View Tab\tCtrl+T")
        self.Bind(wx.EVT_MENU, self.on_new_view, item_new_view)
        menubar.Append(view_menu, "&View")

        self.SetMenuBar(menubar)

    def _refresh_left_panels(self) -> None:
        """Refresh left-side panels from the current experiment model."""
        try:
            self.runs_panel.refresh_from_experiment(self.experiment)
        except Exception:
            pass
        try:
            self.experiment_panel.refresh_from_experiment(self.experiment)
        except Exception:
            pass

    def on_merge_run_dialog(self, event):
        """Open the merge-run dialog.

        Note: wx.FileDialog cannot embed extra checkboxes in a cross-platform way.
        Therefore merge options are collected inside the subsequent MergeRunsDialog.
        """
        with wx.FileDialog(
            self,
            message="Select a 1D run file to seed merging",
            wildcard="Data files (*.csv;*.txt)|*.csv;*.txt|All files (*.*)|*.*",
            style=wx.FD_OPEN | wx.FD_FILE_MUST_EXIST,
        ) as dlg:
            if dlg.ShowModal() != wx.ID_OK:
                return
            seed_path = dlg.GetPath()

        md = MergeRunsDialog(self, [seed_path], log_cb=self.log_panel.append_log)
        try:
            if md.ShowModal() == wx.ID_OK and md.result_run:
                new_run = md.result_run
                self.experiment.add_run(new_run)
                nickname = self.experiment.get_run_nickname(new_run.id)
                self.log_panel.append_log(
                    f"Merged run created as {nickname} ({new_run.id})."
                )
                self._refresh_left_panels()
                self._refresh_all_view_panels() # To update any open views
            else:
                self.log_panel.append_log("Merge operation cancelled.")
        finally:
            md.Destroy()

    # -------- layout --------

    def _build_layout(self):
        # root sizer for the frame
        root_sizer = wx.BoxSizer(wx.VERTICAL)
        self.SetSizer(root_sizer)

        # main horizontal splitter: left controls | right views
        self.main_splitter = wx.SplitterWindow(
            self, style=wx.SP_LIVE_UPDATE | wx.SP_3D
        )

        root_sizer.Add(self.main_splitter, 1, wx.EXPAND)

        # left side: panel containing its own vertical splitter
        left_panel = wx.Panel(self.main_splitter)
        left_sizer = wx.BoxSizer(wx.VERTICAL)
        left_panel.SetSizer(left_sizer)

        self.left_splitter = wx.SplitterWindow(
            left_panel, style=wx.SP_LIVE_UPDATE | wx.SP_3D
        )
        left_sizer.Add(self.left_splitter, 1, wx.EXPAND)

        # --- top notebook (Files / Runs / Experiment / Log) ---
        top_notebook = wx.Notebook(self.left_splitter, style=wx.NB_TOP)

        self.files_panel = FilesPanel(
            top_notebook,
            on_file_activated=self.on_file_activated,
            on_import_run=self.on_import_run_from_files_panel,
        )
        self.runs_panel = RunsPanel(
            top_notebook,
            on_add_to_current_view=self.on_add_runs_to_current_view,
            on_add_to_new_view=self.on_add_runs_to_new_view,
            on_view_remove=self.on_remove_view,
            on_view_duplicate=self.on_duplicate_view,
            on_view_remove_run=self.on_remove_run_from_view,
            on_rename_run=self.on_rename_run,
            on_export_run=self.on_export_run,
            on_update_from_file=self.on_update_run_from_file,
        )
        self.experiment_panel = ExperimentPanel(top_notebook)
        self.log_panel = LogPanel(top_notebook)

        top_notebook.AddPage(self.files_panel, "Files")
        top_notebook.AddPage(self.runs_panel, "Experiment")
        top_notebook.AddPage(self.experiment_panel, "Metadata")
        top_notebook.AddPage(self.log_panel, "Log")

        # --- bottom notebook (Preview / Curve Fit) ---
        bottom_notebook = wx.Notebook(self.left_splitter, style=wx.NB_TOP)

        self.preview_panel = PreviewPanel(bottom_notebook)
        self.curvefit_panel = CurveFitPanel(bottom_notebook)
        self.plot_config_panel = PlotConfigPanel(
            bottom_notebook,
            on_x_range_changed=self.on_x_range_changed,
            on_y_range_changed=self.on_y_range_changed,
            on_vlim_changed=self.on_vlim_changed,
            on_reset=self.on_plot_reset,
            on_unit_changed=self.on_plot_config_unit_changed,
            on_cmap_changed=self.on_view_cmap_changed,
        )

        bottom_notebook.AddPage(self.preview_panel, "Preview")
        bottom_notebook.AddPage(self.plot_config_panel, "Plot Config.")
        bottom_notebook.AddPage(self.curvefit_panel, "Curve Fit")

        # Put the two notebooks into the left vertical splitter
        # Initial sash position: ~40% of left panel height.
        self.left_splitter.SplitHorizontally(top_notebook, bottom_notebook)
        self.left_splitter.SetMinimumPaneSize(80)
        self.left_splitter.SetSashGravity(0.4)

        # --- right: notebook of Views ---
        self.view_notebook = wx.Notebook(
            self.main_splitter,
            style=wx.NB_TOP,
        )

        # view tabs will be created after splitter/layout is set up

        # split left and right; adjust sash after layout based on real splitter width
        self.main_splitter.SplitVertically(left_panel, self.view_notebook)
        self.main_splitter.SetMinimumPaneSize(200)
        # SashGravity controls how extra space is distributed when resizing.
        # 1/3 means the left pane tends to stay about one third of the width when resizing.
        self.main_splitter.SetSashGravity(1.0 / 3.0)

        # Defer initial sash positioning until after the first layout pass,
        # so GetClientSize() returns a meaningful width.
        wx.CallAfter(self._set_initial_split_ratio)

        # After layout, create the initial view tab and view state
        wx.CallAfter(self.add_view_tab)
        
        # Bind notebook page changed
        self.view_notebook.Bind(wx.EVT_NOTEBOOK_PAGE_CHANGED, self.on_view_page_changed)

    def _set_initial_split_ratio(self):
        """
        Set initial left:right split to approximately 1/3 : 2/3,
        using the actual client width of the splitter after layout.
        """
        width = self.main_splitter.GetClientSize().width
        if width <= 0:
            return
        sash_pos = int(width * (1.0 / 3.0))
        self.main_splitter.SetSashPosition(sash_pos)

    # -------- experiment / runs --------

    def load_initial_files(self, file_paths: List[str]) -> None:
        """
        Build Run objects from the given file paths and add them to the ExperimentSet.
        """
        self.import_runs_from_paths(file_paths, set_files_dir=True)

    def import_runs_from_paths(self, file_paths: List[str], *, set_files_dir: bool = True) -> None:
        """Import runs from file paths using the same pipeline as CLI startup."""
        if not file_paths:
            return

        loaded_dirs = []

        for path in file_paths:
            if not path:
                continue
            try:
                run = Run.from_file(path)
            except Exception as e:
                self.log_panel.append_log(f"Failed to load {path}: {e}")
                continue

            self.experiment.add_run(run)
            nickname = self.experiment.get_run_nickname(run.id)
            self.log_panel.append_log(
                f"Loaded file '{os.path.basename(path)}' as run {nickname}({run.id})"
            )
            loaded_dirs.append(os.path.dirname(os.path.abspath(path)))

        # Update simple experiment-level metadata
        self.experiment.metadata["num_runs"] = len(self.experiment.runs)

        # Refresh tabs
        self.runs_panel.refresh_from_experiment(self.experiment)
        self.experiment_panel.refresh_from_experiment(self.experiment)

        # Initialize Files tab directory to the first loaded directory
        if set_files_dir and loaded_dirs:
            self.files_panel.set_directory(loaded_dirs[0])

    def on_import_run_from_files_panel(self, path: str) -> None:
        """Called from FilesPanel context menu: import a single file as a run."""
        if not path:
            return
        self.import_runs_from_paths([path], set_files_dir=True)

    def on_import_run_dialog(self, event) -> None:
        """File -> Import Run... opens file picker and imports as runs."""
        wildcard = "Data files (*.csv;*.txt;*.tvf)|*.csv;*.txt;*.tvf|All files (*.*)|*.*"
        dlg = wx.FileDialog(
            self,
            message="Import Run",
            wildcard=wildcard,
            style=wx.FD_OPEN | wx.FD_FILE_MUST_EXIST | wx.FD_MULTIPLE,
        )
        try:
            if dlg.ShowModal() != wx.ID_OK:
                return
            paths = dlg.GetPaths()
        finally:
            dlg.Destroy()

        self.import_runs_from_paths(list(paths), set_files_dir=True)


    def on_export_run(self, run_ids: List[str]):
        if not run_ids:
            return

        # Ask the user for a directory ONCE.
        with wx.DirDialog(
            self,
            f"Choose directory to save {len(run_ids)} exported runs",
            style=wx.DD_DEFAULT_STYLE | wx.DD_DIR_MUST_EXIST,
        ) as dlg:
            if dlg.ShowModal() != wx.ID_OK:
                return
            
            output_dir = dlg.GetPath()

        exported_count = 0
        failed_count = 0

        for run_id in run_ids:
            run = self.experiment.get_run(run_id)
            if run is None:
                self.log_panel.append_log(f"Error: Run {run_id} not found for export.")
                failed_count += 1
                continue

            if not run.is_2d:
                self.log_panel.append_log(f"Skipping export: Run '{run.nickname}' is not 2D.")
                continue

            try:
                run.export_csv(output_dir=output_dir)
                exported_count += 1
            except Exception as e:
                self.log_panel.append_log(f"Failed to export run {run.nickname}: {e}")
                failed_count += 1
        
        self.log_panel.append_log(f"Finished export operation to {output_dir}.")
        self.log_panel.append_log(f"Successfully exported {exported_count} runs.")
        if failed_count > 0:
            self.log_panel.append_log(f"Failed to export {failed_count} runs.")

    # -------- view-tab helpers --------

    def add_view_tab(self, run_ids: Optional[List[str]] = None):
        """
        Create a new View tab and corresponding ViewState in the ExperimentSet.

        Parameters
        ----------
        run_ids : list of str, optional
            If provided, these run IDs will be attached to the new view.
        """
        idx = len(self.experiment.views) + 1
        view_id = new_view_id()
        title = f"View {idx}"

        view_state = ViewState(id=view_id, title=title, run_ids=run_ids or [])
        self.experiment.add_view(view_state)

        panel = ViewPanel(
            self.view_notebook,
            view_label=title,
            on_limits_changed=self.on_view_limits_changed,
            on_vlim_changed=self.on_view_contrast_reset
        )
        self.view_notebook.AddPage(panel, title, select=True)

        page_index = self.view_notebook.GetPageCount() - 1
        self._view_page_to_id[page_index] = view_id
        self._view_id_to_panel[view_id] = panel

        # Initialize plot for this view
        panel.set_view_model(self.experiment, view_state)

        # Update the Experiment tab's views list and metadata tree
        self.runs_panel.refresh_from_experiment(self.experiment)
        self.experiment_panel.refresh_from_experiment(self.experiment)
    # -------- view management helpers --------

    def _get_current_view_id(self) -> Optional[str]:
        """Return the view_id for the currently selected view tab, if any."""
        page_index = self.view_notebook.GetSelection()
        if page_index == wx.NOT_FOUND:
            return None
        return self._view_page_to_id.get(page_index)

    def _update_view_plot(self, view_id: str) -> None:
        """
        Refresh the plotting for the specified view, if its panel exists.
        """
        view_state = self.experiment.get_view(view_id)
        if view_state is None:
            return
        panel = self._view_id_to_panel.get(view_id)
        if panel is None:
            return
        panel.set_view_model(self.experiment, view_state)

    def get_current_view_panel(self) -> Optional[ViewPanel]:
        view_id = self._get_current_view_id()
        if view_id:
            return self._view_id_to_panel.get(view_id)
        return None

    def on_x_range_changed(self, xmin, xmax, unit):
        view_panel = self.get_current_view_panel()
        if view_panel:
            view_panel.set_x_range(xmin, xmax, unit)

    def on_y_range_changed(self, ymin, ymax):
        view_panel = self.get_current_view_panel()
        if view_panel:
            view_panel.set_y_range(ymin, ymax)

    def on_vlim_changed(self, vmin, vmax):
        view_panel = self.get_current_view_panel()
        if view_panel:
            view_panel.set_vlim(vmin, vmax)

    def on_plot_reset(self):
        view_id = self._get_current_view_id()
        if view_id:
            self._update_view_plot(view_id)

    def on_plot_config_unit_changed(self, new_unit):
        view_panel = self.get_current_view_panel()
        if view_panel:
            xlim_cm1, _ = view_panel.get_plot_limits()
            if xlim_cm1 and xlim_cm1[0] is not None:
                if new_unit == 'meV':
                    xlim_display = (xlim_cm1[0] * EV_PER_CM1 * 1000, xlim_cm1[1] * EV_PER_CM1 * 1000)
                else: # cm-1
                    xlim_display = xlim_cm1
                self.plot_config_panel.set_x_range(xlim_display[0], xlim_display[1])

    def on_view_page_changed(self, event):
        view_panel = self.get_current_view_panel()
        if not view_panel:
            event.Skip()
            return
            
        config = view_panel.get_plot_config()
        
        # Sync Limits
        xlim = config.get('xlim', (0, 1))
        ylim = config.get('ylim', (0, 1))
        
        current_unit = self.plot_config_panel.get_x_unit()
        if current_unit == 'meV':
             xlim = (xlim[0] * EV_PER_CM1 * 1000, xlim[1] * EV_PER_CM1 * 1000)
        
        self.plot_config_panel.set_x_range(xlim[0], xlim[1])
        self.plot_config_panel.set_y_range(ylim[0], ylim[1])
        
        # Sync Contrast
        vmin_p = config.get('vmin_p', 0)
        vmax_p = config.get('vmax_p', 100)
        self.plot_config_panel.set_vlim_range(vmin_p, vmax_p)
        
        # Sync Colormap
        cmap = config.get('cmap', 'OrRd')
        self.plot_config_panel.set_colormap(cmap)
        
        event.Skip()

    def on_view_cmap_changed(self, cmap_name):
        view_panel = self.get_current_view_panel()
        if view_panel:
            view_panel.set_colormap(cmap_name)

    def on_view_limits_changed(self, xlim, ylim):
        current_unit = self.plot_config_panel.get_x_unit()
        if xlim and xlim[0] is not None:
            if current_unit == 'meV':
                xlim = (xlim[0] * EV_PER_CM1 * 1000, xlim[1] * EV_PER_CM1 * 1000)
            self.plot_config_panel.set_x_range(xlim[0], xlim[1])
        if ylim and ylim[0] is not None:
            self.plot_config_panel.set_y_range(ylim[0], ylim[1])

    def on_view_contrast_reset(self, vmin, vmax):
        self.plot_config_panel.vmin_slider.SetValue(vmin)
        self.plot_config_panel.vmax_slider.SetValue(vmax)
            
    def on_add_runs_to_current_view(self, run_ids: List[str]) -> None:
        """
        Attach the given runs to the currently selected view.
        If no view exists yet, a new view is created.
        """
        if not run_ids:
            return

        view_id = self._get_current_view_id()
        if view_id is None:
            # No view yet: create a new one with these runs
            self.add_view_tab(run_ids=run_ids)
            labels = [
                f"{self.experiment.get_run_nickname(rid)}({rid})"
                for rid in run_ids
            ]
            self.log_panel.append_log(
                f"Created new view with runs: {', '.join(labels)}"
            )
            return

        view_state = self.experiment.get_view(view_id)
        if view_state is None:
            return

        # Extend unique run IDs
        for rid in run_ids:
            if rid not in view_state.run_ids:
                view_state.run_ids.append(rid)

        labels = [
            f"{self.experiment.get_run_nickname(rid)}({rid})"
            for rid in run_ids
        ]
        self.log_panel.append_log(
            f"Added runs {', '.join(labels)} to view {view_state.title}"
        )

        # Refresh the views listing and update plotting
        self.runs_panel.refresh_from_experiment(self.experiment)
        self._update_view_plot(view_id)
        self.experiment_panel.refresh_from_experiment(self.experiment)

    def on_add_runs_to_new_view(self, run_ids: List[str]) -> None:
        """
        Create a new view and attach the given runs to it.
        """
        if not run_ids:
            return

        self.add_view_tab(run_ids=run_ids)
        labels = [
            f"{self.experiment.get_run_nickname(rid)}({rid})"
            for rid in run_ids
        ]
        self.log_panel.append_log(
            f"Created new view with runs: {', '.join(labels)}"
        )

    def on_remove_view(self, view_id: str) -> None:
        """
        Remove a view from the ExperimentSet and its corresponding notebook tab.
        """
        view_state = self.experiment.get_view(view_id)
        if view_state is None:
            return

        title = view_state.title
        # Remove panel mapping if present
        if view_id in self._view_id_to_panel:
            del self._view_id_to_panel[view_id]

        # Find the notebook page index corresponding to this view
        index_to_remove = None
        for idx, vid in self._view_page_to_id.items():
            if vid == view_id:
                index_to_remove = idx
                break

        if index_to_remove is not None and 0 <= index_to_remove < self.view_notebook.GetPageCount():
            self.view_notebook.DeletePage(index_to_remove)

            # Rebuild the page-index -> view_id mapping after deletion
            old_map = self._view_page_to_id
            new_map = {}
            for old_idx, vid in old_map.items():
                if vid == view_id:
                    continue
                if old_idx > index_to_remove:
                    new_idx = old_idx - 1
                else:
                    new_idx = old_idx
                new_map[new_idx] = vid
            self._view_page_to_id = new_map

        # Remove from the experiment
        if view_id in self.experiment.views:
            del self.experiment.views[view_id]

        self.runs_panel.refresh_from_experiment(self.experiment)
        self.experiment_panel.refresh_from_experiment(self.experiment)
        self.log_panel.append_log(f"Removed view {title}")

    def on_duplicate_view(self, view_id: str) -> None:
        """
        Duplicate an existing view (including its run list) into a new view/tab.
        """
        view_state = self.experiment.get_view(view_id)
        if view_state is None:
            return

        run_ids = list(view_state.run_ids)
        self.add_view_tab(run_ids=run_ids)

        if run_ids:
            labels = [
                f"{self.experiment.get_run_nickname(rid)}({rid})"
                for rid in run_ids
            ]
            runs_str = ", ".join(labels)
        else:
            runs_str = "(no runs)"

        self.log_panel.append_log(
            f"Duplicated view {view_state.title} with runs: {runs_str}"
        )
        self.runs_panel.refresh_from_experiment(self.experiment)
        self.experiment_panel.refresh_from_experiment(self.experiment)

    def on_remove_run_from_view(self, view_id: str, run_id: str) -> None:
        """
        Remove a single run from the specified view.
        """
        view_state = self.experiment.get_view(view_id)
        if view_state is None:
            return

        if run_id in view_state.run_ids:
            view_state.run_ids = [rid for rid in view_state.run_ids if rid != run_id]
            label = f"{self.experiment.get_run_nickname(run_id)}({run_id})"
            self.log_panel.append_log(
                f"Removed run {label} from view {view_state.title}"
            )
            self.runs_panel.refresh_from_experiment(self.experiment)
            self._update_view_plot(view_id)
            self.experiment_panel.refresh_from_experiment(self.experiment)

    def on_rename_run(self, run_id: str, new_name: str) -> None:
        """
        Apply a new nickname for the given run and update the panels.

        The nickname is stored in run.metadata['nickname'] and managed by
        the ExperimentSet. This method is called from RunsPanel after an
        inline rename operation.
        """
        new_name = str(new_name).strip()
        if not new_name:
            return

        run = self.experiment.get_run(run_id)
        if run is None:
            return

        # Update the model and refresh views that display the nickname.
        self.experiment.set_run_nickname(run_id, new_name)
        self.runs_panel.refresh_from_experiment(self.experiment)
        self.experiment_panel.refresh_from_experiment(self.experiment)
        self._refresh_all_view_panels()
        self.log_panel.append_log(f"Renamed run {run_id} to {new_name}")

    def on_update_run_from_file(self, run_ids: List[str]) -> None:
        """
        Reload data for the specified runs from their source files.
        """
        updated_any = False
        for rid in run_ids:
            run = self.experiment.get_run(rid)
            if run:
                try:
                    run.reload_data()
                    self.log_panel.append_log(f"Reloaded data for {run.nickname} ({run.source_path})")
                    updated_any = True
                except Exception as e:
                    self.log_panel.append_log(f"Failed to reload {run.nickname}: {e}")
        
        if updated_any:
            self._refresh_left_panels()
            self._refresh_all_view_panels()

    def on_save_experiment(self, event=None) -> None:
        """
        Save the current ExperimentSet to an HDF5 file.
        """
        wildcard = "HDF5 files (*.h5;*.hdf5)|*.h5;*.hdf5|All files (*.*)|*.*"
        with wx.FileDialog(
            self, message="Save Experiment",
            wildcard=wildcard,
            style=wx.FD_SAVE | wx.FD_OVERWRITE_PROMPT
        ) as dlg:
            if dlg.ShowModal() == wx.ID_OK:
                path = dlg.GetPath()
                try:
                    self.experiment.export_hdf5(path)
                    self.log_panel.append_log(f"Experiment saved to {path}")
                except Exception as e:
                    wx.MessageBox(f"Failed to save experiment: {e}", "Error", wx.OK | wx.ICON_ERROR)

    def on_open_experiment(self, event=None) -> None:
        """
        Load an ExperimentSet from an HDF5 file.
        """
        wildcard = "HDF5 files (*.h5;*.hdf5)|*.h5;*.hdf5|All files (*.*)|*.*"
        with wx.FileDialog(
            self, message="Open Experiment",
            wildcard=wildcard,
            style=wx.FD_OPEN | wx.FD_FILE_MUST_EXIST
        ) as dlg:
            if dlg.ShowModal() == wx.ID_OK:
                path = dlg.GetPath()
                try:
                    new_exp = ExperimentSet.from_hdf5(path)
                    self.experiment = new_exp
                    self._refresh_left_panels()
                    self._recreate_view_tabs_from_experiment()
                    self.log_panel.append_log(f"Experiment loaded from {path}")
                except Exception as e:
                    wx.MessageBox(f"Failed to load experiment: {e}", "Error", wx.OK | wx.ICON_ERROR)

    def _recreate_view_tabs_from_experiment(self):
        # Clear existing tabs
        while self.view_notebook.GetPageCount() > 0:
            self.view_notebook.DeletePage(0)
        self._view_page_to_id.clear()
        self._view_id_to_panel.clear()

        # Add tabs back
        for vid, vstate in self.experiment.views.items():
            panel = ViewPanel(
                self.view_notebook,
                view_label=vstate.title,
                on_limits_changed=self.on_view_limits_changed,
                on_vlim_changed=self.on_view_contrast_reset
            )
            self.view_notebook.AddPage(panel, vstate.title)
            page_index = self.view_notebook.GetPageCount() - 1
            self._view_page_to_id[page_index] = vid
            self._view_id_to_panel[vid] = panel
            panel.set_view_model(self.experiment, vstate)

    def _refresh_all_view_panels(self) -> None:
        for view_id in list(self.experiment.views.keys()):
            self._update_view_plot(view_id)


    # -------- callbacks from panels --------

    def on_file_activated(self, path: str) -> None:
        """
        Called by FilesPanel when a file is double-clicked.

        For now:
        - Reads a small chunk of the file and shows it in the Preview tab.
        - Logs the action.
        """
        try:
            with open(path, "r", encoding="utf-8", errors="replace") as f:
                text = f.read(8000)  # ~8 kB is enough for a quick preview
        except OSError as e:
            msg = f"Failed to open file for preview: {path} ({e})"
            self.log_panel.append_log(msg)
            self.preview_panel.show_text(msg)
            return

        self.preview_panel.show_text(text)
        self.log_panel.append_log(f"Previewed file: {path}")

    # -------- event handlers --------

    def on_new_view(self, event):
        self.add_view_tab()

    def on_quit(self, event):
        self.Close()



# -----------------------------
# App entry
# -----------------------------


class RamanApp(wx.App):
    def __init__(self, filenames: Optional[List[str]] = None):
        self._filenames = filenames or []
        super().__init__(False)

    def OnInit(self):
        self.frame = MainFrame(initial_files=self._filenames)
        self.SetTopWindow(self.frame)
        return True


if __name__ == "__main__":
    import sys

    filenames = sys.argv[1:]
    app = RamanApp(filenames)
    app.MainLoop()

