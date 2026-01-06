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
        item_export = menu.Append(wx.ID_ANY, "Export Run...")

        self.Bind(wx.EVT_MENU, self._on_context_add_to_current, item_current)
        self.Bind(wx.EVT_MENU, self._on_context_add_to_new, item_new)
        self.Bind(wx.EVT_MENU, self._on_context_rename_run, item_rename)
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
        # Standard Home behavior
        super().home(*args, **kwargs)
        # Then refresh the slice plots, if the panel has that helper
        if hasattr(self._owner_panel, "_refresh_after_home"):
            self._owner_panel._refresh_after_home()


class ViewPanel(wx.Panel):
    def _refresh_after_home(self) -> None:
        """Called by RamanToolbar after the Home button is pressed.

        Re-apply the last clicked data coordinates (in the same imshow data
        coordinate system) so that Plot A/B/C and all slices are restored to
        exactly the same point as before pressing Home.
        """
        # IMPORTANT: Home must NOT "re-select" by re-running click-to-index
        # inference. That causes cumulative drift (rounding/extent effects).
        # Home should only re-render B/C + highlights from the stored indices.
        self._apply_selection_from_indices(reason="home")
    """
    One 'View' tab, showing up to one active Run as three horizontal plots:

    - Plot A: 2D color map (angle vs Raman shift) with secondary x-axis in eV.
              Mouse click = choose (shift, angle) and draw faint red crosshair.
    - Plot B: Angular slice from Plot A (vertical slice; intensity vs angle).
    - Plot C: Spectral slice from Plot A (horizontal slice; intensity vs Raman shift).

    For now, each ViewPanel visualizes at most the first Run attached to its ViewState.
    """

    def __init__(self, parent, view_label: str):
        super().__init__(parent)

        self.view_label = view_label
        self.current_run_id: Optional[str] = None
        self.current_run: Optional[Run] = None
        # Back-reference to the ExperimentSet currently driving this view.
        # Used mainly for retrieving run nicknames for titles, logs, etc.
        self._experiment: Optional[ExperimentSet] = None

        # Matplotlib figure and canvas
        self.figure = Figure(figsize=(9, 4.5))
        self.canvas = FigureCanvas(self, -1, self.figure)

        # Axes placeholders for the first run (top row)
        self.axA = None
        self.axB = None
        self.axC = None
        self._imA = None  # image handle for Plot A (run 1)
        self._vline = None
        self._hline = None
        self._lineB = None
        self._lineC = None
        self._vlineB = None
        self._vlineC = None

        # Axes placeholders for the second run (bottom row, optional)
        self.axA2 = None
        self.axB2 = None
        self.axC2 = None
        self._imA2 = None  # image handle for Plot A (run 2)
        self._vline2 = None
        self._hline2 = None
        self._lineB2 = None
        self._lineC2 = None
        self._vlineB2 = None
        self._vlineC2 = None

        # Click event connection id
        self._cid_click = None
        self._cid_resize = None

        # Highlight mode: one of {"none", "click", "line_profile", "peak_fit"}
        self.highlight_mode: str = "click"

        # Angle slice rendering type for Plot B: "polar" or "cartesian"
        self.angle_slice_type: str = "polar"
        self._last_drawn_runs: List[Run] = []

        # Guard flag to avoid recursive callbacks when syncing zoom between axes
        self._syncing_limits: bool = False

        # ------------------------------------------------------------------
        # Callback binding management (IMPORTANT)
        #
        # This GUI frequently redraws by calling figure.clf() and recreating
        # Axes objects (e.g., run count changes 2→1, polar/cartesian toggles).
        #
        # Matplotlib callbacks are attached to Axes instances. If we connect
        # callbacks every redraw without disconnecting, they can accumulate
        # on stale Axes, causing double-trigger bugs and hard-to-debug drift.
        # Therefore we always disconnect previous callback IDs before rebinding.
        # ------------------------------------------------------------------
        self._limit_cb_ids: list[tuple[object, int]] = []  # (callbacks registry, cid)

        # Last selected slice coordinates in data (imshow) coordinates.
        # Stored as (x, y) in the same coordinate system as event.xdata/ydata.
        self._last_coords: Optional[tuple[float, float]] = None

        # Persistent selection state (index-based, drift-free).
        # These are the ONLY authoritative "selection" variables.
        # - Run1 selection index in its (shift, angle) grids: (ix1, iy1)
        # - Run2 selection index in its grids: (ix2, iy2) (nearest-mapped from run1)
        self._sel_idx1: Optional[tuple[int, int]] = None
        self._sel_idx2: Optional[tuple[int, int]] = None

        # Optional second run to visualize in the bottom row
        self._second_run: Optional[Run] = None

        # Layout: label + toolbar + canvas
        sizer = wx.BoxSizer(wx.VERTICAL)
        label = wx.StaticText(self, label=view_label)
        label.SetForegroundColour(wx.Colour(50, 50, 50))
        sizer.Add(label, 0, wx.ALL, 4)

        # Use RamanToolbar so that Home also refreshes slices
        self.toolbar = RamanToolbar(self.canvas, self)
        self.toolbar.Realize()
        sizer.Add(self.toolbar, 0, wx.EXPAND | wx.LEFT | wx.RIGHT, 4)

        sizer.Add(self.canvas, 1, wx.EXPAND | wx.ALL, 4)
        self.SetSizer(sizer)

        # Initialize empty figure
        self._init_empty_figure()

        # Right-click on the canvas to configure highlight mode
        self.canvas.Bind(wx.EVT_CONTEXT_MENU, self._on_context_menu)

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
        self._disconnect_limit_sync_callbacks()

        if self.axA is not None:
            self._limit_cb_ids.append(
                (
                    self.axA.callbacks,
                    self.axA.callbacks.connect("xlim_changed", self._on_xlim_changed_A),
                )
            )
            # Always keep A1<->A2 ylim synchronized, regardless of Plot B mode.
            # Plot B angle-domain synchronization remains conditional inside
            # `_on_ylim_changed_A()`.
            self._limit_cb_ids.append(
                (
                    self.axA.callbacks,
                    self.axA.callbacks.connect("ylim_changed", self._on_ylim_changed_A),
                )
            )
        if self.axC is not None:
            self._limit_cb_ids.append(
                (
                    self.axC.callbacks,
                    self.axC.callbacks.connect("xlim_changed", self._on_xlim_changed_C),
                )
            )
        if self.angle_slice_type != "polar" and self.axB is not None:
            self._limit_cb_ids.append(
                (
                    self.axB.callbacks,
                    self.axB.callbacks.connect("xlim_changed", self._on_xlim_changed_B),
                )
            )

        if self.axA2 is not None:
            self._limit_cb_ids.append(
                (
                    self.axA2.callbacks,
                    self.axA2.callbacks.connect(
                        "xlim_changed", self._on_xlim_changed_A2
                    ),
                )
            )
            # Always keep A2<->A1 ylim synchronized, regardless of Plot B mode.
            # Plot B angle-domain synchronization remains conditional inside
            # `_on_ylim_changed_A2()`.
            self._limit_cb_ids.append(
                (
                    self.axA2.callbacks,
                    self.axA2.callbacks.connect("ylim_changed", self._on_ylim_changed_A2),
                )
            )
        if self.axC2 is not None:
            self._limit_cb_ids.append(
                (
                    self.axC2.callbacks,
                    self.axC2.callbacks.connect(
                        "xlim_changed", self._on_xlim_changed_C2
                    ),
                )
            )
        if self.angle_slice_type != "polar" and self.axB2 is not None:
            self._limit_cb_ids.append(
                (
                    self.axB2.callbacks,
                    self.axB2.callbacks.connect(
                        "xlim_changed", self._on_xlim_changed_B2
                    ),
                )
            )

    def _on_xlim_changed_A(self, ax) -> None:
        if self._syncing_limits:
            return
        if self.axA is None or self.axC is None:
            return
        self._syncing_limits = True
        try:
            xlim = self.axA.get_xlim()
            self.axC.set_xlim(xlim)
            if self.axA2 is not None:
                self.axA2.set_xlim(xlim)
            if self.axC2 is not None:
                self.axC2.set_xlim(xlim)
        finally:
            self._syncing_limits = False

    def _on_xlim_changed_C(self, ax) -> None:
        if self._syncing_limits:
            return
        if self.axA is None or self.axC is None:
            return
        self._syncing_limits = True
        try:
            xlim = self.axC.get_xlim()
            self.axA.set_xlim(xlim)
            if self.axA2 is not None:
                self.axA2.set_xlim(xlim)
            if self.axC2 is not None:
                self.axC2.set_xlim(xlim)
        finally:
            self._syncing_limits = False

    def _on_ylim_changed_A(self, ax) -> None:
        if self._syncing_limits:
            return
        if self.axA is None:
            return
        self._syncing_limits = True
        try:
            ylim = self.axA.get_ylim()
            if self.axA2 is not None:
                self.axA2.set_ylim(ylim)
            if self.angle_slice_type != "polar":
                if self.axB is not None:
                    self.axB.set_xlim(ylim)
                if self.axB2 is not None:
                    self.axB2.set_xlim(ylim)
        finally:
            self._syncing_limits = False

    def _on_xlim_changed_B(self, ax) -> None:
        if self._syncing_limits:
            return
        if self.axB is None:
            return
        self._syncing_limits = True
        try:
            xlim = self.axB.get_xlim()
            if self.axA is not None:
                self.axA.set_ylim(xlim)
            if self.axA2 is not None:
                self.axA2.set_ylim(xlim)
            if self.axB2 is not None:
                self.axB2.set_xlim(xlim)
        finally:
            self._syncing_limits = False

    def _on_xlim_changed_A2(self, ax) -> None:
        if self._syncing_limits:
            return
        if self.axA2 is None:
            return
        self._syncing_limits = True
        try:
            xlim = self.axA2.get_xlim()
            if self.axC2 is not None:
                self.axC2.set_xlim(xlim)
            if self.axA is not None:
                self.axA.set_xlim(xlim)
            if self.axC is not None:
                self.axC.set_xlim(xlim)
        finally:
            self._syncing_limits = False

    def _on_xlim_changed_C2(self, ax) -> None:
        if self._syncing_limits:
            return
        if self.axC2 is None:
            return
        self._syncing_limits = True
        try:
            xlim = self.axC2.get_xlim()
            if self.axA2 is not None:
                self.axA2.set_xlim(xlim)
            if self.axA is not None:
                self.axA.set_xlim(xlim)
            if self.axC is not None:
                self.axC.set_xlim(xlim)
        finally:
            self._syncing_limits = False

    def _on_ylim_changed_A2(self, ax) -> None:
        if self._syncing_limits:
            return
        if self.axA2 is None:
            return
        self._syncing_limits = True
        try:
            ylim = self.axA2.get_ylim()
            if self.axA is not None:
                self.axA.set_ylim(ylim)
            if self.angle_slice_type != "polar":
                if self.axB2 is not None:
                    self.axB2.set_xlim(ylim)
                if self.axB is not None:
                    self.axB.set_xlim(ylim)
        finally:
            self._syncing_limits = False

    def _on_xlim_changed_B2(self, ax) -> None:
        if self._syncing_limits:
            return
        if self.axB2 is None:
            return
        self._syncing_limits = True
        try:
            xlim = self.axB2.get_xlim()
            if self.axA2 is not None:
                self.axA2.set_ylim(xlim)
            if self.axA is not None:
                self.axA.set_ylim(xlim)
            if self.axB is not None:
                self.axB.set_xlim(xlim)
        finally:
            self._syncing_limits = False
    def _on_context_menu(self, event):
        """
        Right-click context menu on the canvas.

        Provides a 'Highlight' submenu with mutually exclusive options:
        - None
        - Click (default)
        - Add line profile (placeholder)
        - Add peak fit (placeholder)
        """
        menu = wx.Menu()

        highlight_menu = wx.Menu()

        item_none = highlight_menu.AppendRadioItem(wx.ID_ANY, "None")
        item_click = highlight_menu.AppendRadioItem(wx.ID_ANY, "Click")
        item_line = highlight_menu.AppendRadioItem(wx.ID_ANY, "Add line profile (yet)")
        item_peak = highlight_menu.AppendRadioItem(wx.ID_ANY, "Add peak fit (yet)")

        # Set current check state
        mode = self.highlight_mode
        if mode == "none":
            item_none.Check(True)
        elif mode == "click":
            item_click.Check(True)
        elif mode == "line_profile":
            item_line.Check(True)
        elif mode == "peak_fit":
            item_peak.Check(True)

        # Bind handlers
        self.Bind(wx.EVT_MENU, lambda evt: self._set_highlight_mode("none"), item_none)
        self.Bind(wx.EVT_MENU, lambda evt: self._set_highlight_mode("click"), item_click)
        self.Bind(
            wx.EVT_MENU, lambda evt: self._set_highlight_mode("line_profile"), item_line
        )
        self.Bind(
            wx.EVT_MENU, lambda evt: self._set_highlight_mode("peak_fit"), item_peak
        )

        menu.AppendSubMenu(highlight_menu, "Highlight")

        self.PopupMenu(menu)
        menu.Destroy()

    def _set_highlight_mode(self, mode: str) -> None:
        """
        Set the current highlight mode and update visibility of overlays.

        For now:
        - 'click'  : show red crosshair overlay when clicking.
        - others   : hide the crosshair but keep slice position updates.
        """
        self.highlight_mode = mode

        # Hide crosshair overlays if mode is not 'click'
        if mode != "click":
            if self._vline is not None:
                self._vline.set_visible(False)
            if self._hline is not None:
                self._hline.set_visible(False)
            if self._vline2 is not None:
                self._vline2.set_visible(False)
            if self._hline2 is not None:
                self._hline2.set_visible(False)
            self.canvas.draw_idle()
        else:
            # 'click' mode: crosshair will be (re-)drawn on the next click.
            pass

    def _popup_angle_slice_type_menu(self, mpl_event) -> None:
        """
        Show a context menu for selecting Plot B angle slice type.
        Triggered by right-click on axB/axB2.
        """
        menu = wx.Menu()
        sub = wx.Menu()

        item_cart = sub.AppendRadioItem(wx.ID_ANY, "Cartesian")
        item_polar = sub.AppendRadioItem(wx.ID_ANY, "Polar")

        if self.angle_slice_type == "polar":
            item_polar.Check(True)
        else:
            item_cart.Check(True)

        self.Bind(
            wx.EVT_MENU,
            lambda evt: self._set_angle_slice_type("cartesian"),
            item_cart,
        )
        self.Bind(
            wx.EVT_MENU,
            lambda evt: self._set_angle_slice_type("polar"),
            item_polar,
        )

        menu.AppendSubMenu(sub, "Angle slice type")

        pt = None
        ge = getattr(mpl_event, "guiEvent", None)
        if ge is not None and hasattr(ge, "GetPosition"):
            pt = ge.GetPosition()

        if pt is not None:
            self.PopupMenu(menu, pt)
        else:
            self.PopupMenu(menu)

        menu.Destroy()

    def _set_angle_slice_type(self, mode: str) -> None:
        """
        Set slice type and re-render plots.
        mode: "polar" or "cartesian"
        """
        mode = str(mode).lower().strip()
        if mode not in ("polar", "cartesian"):
            return
        if self.angle_slice_type == mode:
            return

        self.angle_slice_type = mode

        if not self._last_drawn_runs:
            self.canvas.draw_idle()
            return

        self._draw_runs(self._last_drawn_runs)

        if self._last_coords is not None and self.axA is not None:
            x_sel, y_sel = self._last_coords

            class _DummyEvent:
                def __init__(self, ax, x, y):
                    self.inaxes = ax
                    self.xdata = x
                    self.ydata = y
                    self.button = 1
                    self.guiEvent = None

            dummy = _DummyEvent(self.axA, x_sel, y_sel)
            self._on_canvas_click(dummy)

        self.canvas.draw_idle()

    # ---------- public API ----------

    def set_view_model(self, experiment: ExperimentSet, view_state: ViewState) -> None:
        """
        Update this panel to reflect the given ViewState and ExperimentSet.

        Up to two runs are visualized in a 2×3 layout:
        - Top row : first run in view_state.run_ids (interactive A/B/C).
        - Bottom row: second run (if present), linked to the same (x, y) slice.
        """
        self._experiment = experiment
        self.current_run_id = None
        self.current_run = None
        self._second_run = None
        self._last_coords = None
        self._sel_idx1 = None
        self._sel_idx2 = None

        # Collect up to two runs with valid 2D data
        runs: List[Run] = []
        for rid in view_state.run_ids:
            r = experiment.runs.get(rid)
            if r is None:
                continue
            if (
                r.intensity_2d is None
                or r.shift_cm1 is None
                or r.angle_values is None
            ):
                continue
            runs.append(r)
            if len(runs) == 2:
                break

        if not runs:
            self._init_empty_figure(
                message="No runs with 2D data attached to this view."
            )
            self.canvas.draw_idle()
            return

        # First run is the active one for interaction
        self.current_run = runs[0]
        self.current_run_id = runs[0].id
        self._second_run = runs[1] if len(runs) > 1 else None

        self._draw_runs(runs)
        self.canvas.draw_idle()

    # ---------- internal helpers ----------

    def _adaptive_polar_locator(self, ax) -> None:
        if ax is None or self.angle_slice_type != "polar":
            return
        if getattr(ax, "name", "") != "polar":
            return

        bbox = ax.get_window_extent()
        height_px = float(bbox.height) if bbox is not None else 0.0
        if not np.isfinite(height_px) or height_px <= 0.0:
            height_px = 240.0

        target_ticks = int(round(height_px / 120.0))
        target_ticks = max(2, min(4, target_ticks))
        ax.yaxis.set_major_locator(mticker.MaxNLocator(nbins=target_ticks))

    def _on_mpl_resize(self, event) -> None:
        if self.angle_slice_type != "polar":
            return
        if self.axB is not None:
            self._adaptive_polar_locator(self.axB)
        if self.axB2 is not None:
            self._adaptive_polar_locator(self.axB2)
        self.canvas.draw_idle()

    def _init_empty_figure(self, message: str = "No data") -> None:
        """Clear the figure and show a simple text message."""
        self.figure.clf()
        ax = self.figure.add_subplot(1, 1, 1)
        ax.text(
            0.5,
            0.5,
            message,
            ha="center",
            va="center",
            transform=ax.transAxes,
        )
        ax.set_axis_off()
        self.axA = self.axB = self.axC = None
        self._imA = None
        self._vline = None
        self._hline = None
        self._lineB = None
        self._lineC = None
        self._vlineB = None
        self._vlineC = None

        self.axA2 = self.axB2 = self.axC2 = None
        self._imA2 = None
        self._vline2 = None
        self._hline2 = None
        self._lineB2 = None
        self._lineC2 = None
        self._vlineB2 = None
        self._vlineC2 = None

        self._last_coords = None
        self._second_run = None
        self._last_drawn_runs = []
        self._sel_idx1 = None
        self._sel_idx2 = None

        if self._cid_click is not None:
            self.canvas.mpl_disconnect(self._cid_click)
            self._cid_click = None

        self._disconnect_limit_sync_callbacks()
        if self._cid_resize is not None:
            self.canvas.mpl_disconnect(self._cid_resize)
            self._cid_resize = None

    def _apply_selection_from_indices(self, *, reason: str = "") -> None:
        """
        Re-render crosshair highlights and B/C slices from persistent index state.

        This method is the core of the "drift-free" design:
        - It never calls click-to-index inference.
        - It never mutates (_sel_idx1, _sel_idx2) except to lazily initialize idx2
          from idx1 when needed.
        - It re-computes only display coordinates (x_hl, y_hl) from current extent,
          so zoom/home/redraw stays visually consistent without changing selection.
        """
        if self.current_run is None or self.axA is None or self._imA is None:
            return
        if self._sel_idx1 is None:
            return

        def _prepare_run_data(run: Run):
            shift = np.asarray(run.shift_cm1, dtype=float)
            angles = np.asarray(run.angle_values, dtype=float)
            I = np.asarray(run.intensity_2d, dtype=float)
            if I.shape != (angles.size, shift.size):
                if I.shape == (shift.size, angles.size):
                    I = I.T
                else:
                    return None
            return shift, angles, I

        def _hl_from_index(im, I, ix, iy):
            # Render highlight at the CENTER of the selected cell.
            x0, x1, y0, y1 = im.get_extent()
            ny, nx = I.shape
            # Guard against degenerate extents
            if nx <= 0 or ny <= 0 or x1 == x0 or y1 == y0:
                return None
            x_hl = x0 + (ix + 0.5) * (x1 - x0) / nx
            y_hl = y0 + (iy + 0.5) * (y1 - y0) / ny
            return float(x_hl), float(y_hl)

        def _update_crosshair(ax, vline, hline, x_hl, y_hl):
            if ax is None:
                return vline, hline
            if self.highlight_mode == "click":
                if vline is None:
                    vline = ax.axvline(x_hl, color="red", alpha=0.3, linewidth=1.0)
                else:
                    vline.set_xdata([x_hl, x_hl])
                    vline.set_visible(True)
                if hline is None:
                    hline = ax.axhline(y_hl, color="red", alpha=0.3, linewidth=1.0)
                else:
                    hline.set_ydata([y_hl, y_hl])
                    hline.set_visible(True)
            else:
                if vline is not None:
                    vline.set_visible(False)
                if hline is not None:
                    hline.set_visible(False)
            return vline, hline

        def _update_slice_b(ax, line, vline, angles, prof, title, vline_x_deg):
            if ax is None:
                return line, vline
            if self.angle_slice_type == "polar":
                th = np.deg2rad(np.asarray(angles, float))
                ax.clear()
                ax.set_theta_zero_location("N")
                ax.set_theta_direction(-1)
                ax.plot(th, prof, "o", ms=3, linestyle="None")
                vals = np.asarray(prof, float)
                vals = vals[np.isfinite(vals)]
                rmax = float(vals.max()) if vals.size else 1.0
                if not np.isfinite(rmax) or rmax <= 0:
                    rmax = 1.0
                pad = 0.05 * rmax
                ax.set_rlim(0.0, rmax + pad)
                ax.set_title(title)
                theta_sel = np.deg2rad(float(vline_x_deg))
                ax.axvline(theta_sel, color="red", alpha=0.3, linewidth=1.0)
                self._adaptive_polar_locator(ax)
                return None, None
            if line is None:
                line, = ax.plot([], [], "-k", linewidth=1.2)
            line.set_data(angles, prof)
            ax.set_title(title)
            ax.set_xlabel("Angle (deg)")
            ax.set_ylabel("Intensity (a.u.)")
            if vline is None:
                vline = ax.axvline(vline_x_deg, color="red", alpha=0.3, linewidth=1.0)
            else:
                vline.set_xdata([vline_x_deg, vline_x_deg])
                vline.set_visible(True)
            ax.relim()
            ax.autoscale_view(scalex=False, scaley=True)
            return line, vline

        def _update_slice_c(ax, line, vline, shift, prof, title, vline_x_cm1):
            if ax is None:
                return line, vline
            if line is None:
                line, = ax.plot([], [], "-k", linewidth=1.2)
            line.set_data(shift, prof)
            ax.set_title(title)
            ax.set_xlabel("Raman shift (cm$^{-1}$)")
            ax.set_ylabel("Intensity (a.u.)")
            if vline is None:
                vline = ax.axvline(vline_x_cm1, color="red", alpha=0.3, linewidth=1.0)
            else:
                vline.set_xdata([vline_x_cm1, vline_x_cm1])
                vline.set_visible(True)
            ax.relim()
            ax.autoscale_view(scalex=False, scaley=True)
            return line, vline

        # ---- Run 1 ----
        data1 = _prepare_run_data(self.current_run)
        if data1 is None:
            return
        shift1, angles1, I1 = data1
        ix1, iy1 = self._sel_idx1
        ix1 = int(np.clip(ix1, 0, shift1.size - 1))
        iy1 = int(np.clip(iy1, 0, angles1.size - 1))
        self._sel_idx1 = (ix1, iy1)
        x_sel1 = float(shift1[ix1])
        y_sel1 = float(angles1[iy1])
        hl1 = _hl_from_index(self._imA, I1, ix1, iy1)
        if hl1 is not None:
            self._vline, self._hline = _update_crosshair(
                self.axA, self._vline, self._hline, hl1[0], hl1[1]
            )
        # B/C slices from indices (not from last_coords)
        self._lineB, self._vlineB = _update_slice_b(
            self.axB,
            self._lineB,
            self._vlineB,
            angles1,
            I1[:, ix1],
            f"Angular Slice @ {x_sel1:.1f} cm$^{{-1}}$",
            y_sel1,
        )
        self._lineC, self._vlineC = _update_slice_c(
            self.axC,
            self._lineC,
            self._vlineC,
            shift1,
            I1[iy1, :],
            f"Spectral Slice @ {y_sel1:.1f} deg",
            x_sel1,
        )

        # ---- Run 2 (optional) ----
        if (
            self._second_run is not None
            and self.axA2 is not None
            and self._imA2 is not None
        ):
            data2 = _prepare_run_data(self._second_run)
            if data2 is not None:
                shift2, angles2, I2 = data2
                # Lazily initialize sel_idx2 from sel_idx1 physical coords
                if self._sel_idx2 is None:
                    ix2 = int(np.argmin(np.abs(shift2 - x_sel1)))
                    iy2 = int(np.argmin(np.abs(angles2 - y_sel1)))
                    self._sel_idx2 = (ix2, iy2)
                ix2, iy2 = self._sel_idx2
                ix2 = int(np.clip(ix2, 0, shift2.size - 1))
                iy2 = int(np.clip(iy2, 0, angles2.size - 1))
                self._sel_idx2 = (ix2, iy2)
                x_sel2 = float(shift2[ix2])
                y_sel2 = float(angles2[iy2])
                hl2 = _hl_from_index(self._imA2, I2, ix2, iy2)
                if hl2 is not None:
                    self._vline2, self._hline2 = _update_crosshair(
                        self.axA2, self._vline2, self._hline2, hl2[0], hl2[1]
                    )
                self._lineB2, self._vlineB2 = _update_slice_b(
                    self.axB2,
                    self._lineB2,
                    self._vlineB2,
                    angles2,
                    I2[:, ix2],
                    f"Angular Slice @ {x_sel2:.1f} cm$^{{-1}}$",
                    y_sel2,
                )
                self._lineC2, self._vlineC2 = _update_slice_c(
                    self.axC2,
                    self._lineC2,
                    self._vlineC2,
                    shift2,
                    I2[iy2, :],
                    f"Spectral Slice @ {y_sel2:.1f} deg",
                    x_sel2,
                )

        # No tight_layout here; keep existing overall adjustments.
        self.figure.subplots_adjust(top=0.862)
        self.canvas.draw_idle()

    def _draw_runs(self, runs: List[Run]) -> None:
        """
        Draw up to two runs as a 2×3 grid.

        - Top row: run 1 (interactive A/B/C).
        - Bottom row: run 2 (if present). Slices and highlights are driven by
          the same (x, y) selection used for run 1.
        If only one run is present, the bottom row is kept as empty axes so
        that the single run occupies roughly the top half of the view.
        """
        if not runs:
            self._init_empty_figure(message="No runs to draw.")
            return

        runs = runs[:2]  # at most two runs
        self._last_drawn_runs = list(runs)

        self.figure.clf()

        # Always allocate two rows so that one run uses about half the height
        gs = self.figure.add_gridspec(
            2,
            3,
            height_ratios=[1.0, 1.0],
            width_ratios=[2.0, 1.0, 1.0],
            hspace=0.6,
            wspace=0.5,
            left=0.1,
            right=0.95
        )

        # Reset sync guard and crosshair handles
        self._syncing_limits = False
        self._vline = None
        self._hline = None
        self._vline2 = None
        self._hline2 = None
        self._lineB = None
        self._lineC = None
        self._vlineB = None
        self._vlineC = None
        self._lineB2 = None
        self._lineC2 = None
        self._vlineB2 = None
        self._vlineC2 = None

        # --- helper for eV conversion ---
        def cm_to_ev(x):
            return x * EV_PER_CM1

        def ev_to_cm(x):
            return x / EV_PER_CM1

        # ========================
        # Top row: first run
        # ========================
        run0 = runs[0]
        shift0 = np.asarray(run0.shift_cm1, dtype=float)
        angles0 = np.asarray(run0.angle_values, dtype=float)
        I0 = np.asarray(run0.intensity_2d, dtype=float)

        # Human-friendly label for the first run
        if self._experiment is not None:
            nickname0 = self._experiment.get_run_nickname(run0.id)
        else:
            nickname0 = run0.nickname

        if I0.shape != (angles0.size, shift0.size):
            if I0.shape == (shift0.size, angles0.size):
                I0 = I0.T
            else:
                self._init_empty_figure(
                    message=(
                        f"Shape mismatch (run 1): intensity_2d {I0.shape} vs "
                        f"angle_values {angles0.shape} and shift_cm1 {shift0.shape}."
                    )
                )
                return

        self.axA = self.figure.add_subplot(gs[0, 0])
        if self.angle_slice_type == "polar":
            self.axB = self.figure.add_subplot(gs[0, 1], projection="polar")
        else:
            self.axB = self.figure.add_subplot(gs[0, 1])
        self.axC = self.figure.add_subplot(gs[0, 2])

        # --- axis limit synchronization for run 1 (and run 2 if present) ---

        def on_xlim_changed_A(ax):
            # A1.x (shift) <-> C1.x (shift); also keep run 2 in sync if present
            if self._syncing_limits:
                return
            if self.axA is None or self.axC is None:
                return
            self._syncing_limits = True
            try:
                xlim = self.axA.get_xlim()
                self.axC.set_xlim(xlim)
                if self.axA2 is not None:
                    self.axA2.set_xlim(xlim)
                if self.axC2 is not None:
                    self.axC2.set_xlim(xlim)
            finally:
                self._syncing_limits = False

        def on_xlim_changed_C(ax):
            # C1.x (shift) <-> A1.x (shift)
            if self._syncing_limits:
                return
            if self.axA is None or self.axC is None:
                return
            self._syncing_limits = True
            try:
                xlim = self.axC.get_xlim()
                self.axA.set_xlim(xlim)
                if self.axA2 is not None:
                    self.axA2.set_xlim(xlim)
                if self.axC2 is not None:
                    self.axC2.set_xlim(xlim)
            finally:
                self._syncing_limits = False

        def on_ylim_changed_A(ax):
            # A1.y (angle) <-> B1.x (angle); keep run 2 angle view in sync
            if self._syncing_limits:
                return
            if self.axA is None or self.axB is None:
                return
            self._syncing_limits = True
            try:
                ylim = self.axA.get_ylim()
                self.axB.set_xlim(ylim)
                if self.axA2 is not None:
                    self.axA2.set_ylim(ylim)
                if self.axB2 is not None:
                    self.axB2.set_xlim(ylim)
            finally:
                self._syncing_limits = False

        def on_xlim_changed_B(ax):
            # B1.x (angle) <-> A1.y (angle)
            if self._syncing_limits:
                return
            if self.axA is None or self.axB is None:
                return
            self._syncing_limits = True
            try:
                xlim = self.axB.get_xlim()
                self.axA.set_ylim(xlim)
                if self.axA2 is not None:
                    self.axA2.set_ylim(xlim)
                if self.axB2 is not None:
                    self.axB2.set_xlim(xlim)
            finally:
                self._syncing_limits = False

        # --- symmetric callbacks for run 2 (so zooming on 2A/2C also syncs run 1) ---
        def on_xlim_changed_A2(ax):
            # A2.x (shift) <-> C2.x (shift); also keep run 1 in sync
            if self._syncing_limits:
                return
            if self.axA2 is None:
                return
            self._syncing_limits = True
            try:
                xlim = self.axA2.get_xlim()
                if self.axC2 is not None:
                    self.axC2.set_xlim(xlim)
                if self.axA is not None:
                    self.axA.set_xlim(xlim)
                if self.axC is not None:
                    self.axC.set_xlim(xlim)
                if self.angle_slice_type != "polar":
                    if self.axB is not None:
                        self.axB.set_xlim(xlim)
                    if self.axB2 is not None:
                        self.axB2.set_xlim(xlim)
            finally:
                self._syncing_limits = False

        def on_xlim_changed_C2(ax):
            # C2.x (shift) <-> A2.x (shift); also keep run 1 in sync
            if self._syncing_limits:
                return
            if self.axC2 is None:
                return
            self._syncing_limits = True
            try:
                xlim = self.axC2.get_xlim()
                if self.axA2 is not None:
                    self.axA2.set_xlim(xlim)
                if self.axA is not None:
                    self.axA.set_xlim(xlim)
                if self.axC is not None:
                    self.axC.set_xlim(xlim)
            finally:
                self._syncing_limits = False

        def on_ylim_changed_A2(ax):
            # self.log_panel.append_log(f"A2 ylim changed")
            # A2.y (angle) <-> B2.x (angle) for cartesian; also keep run 1 in sync
            if self._syncing_limits:
                return
            if self.axA2 is None:
                return
            self._syncing_limits = True
            try:
                ylim = self.axA2.get_ylim()
                # keep A1 y in sync
                if self.axA is not None:
                    self.axA.set_ylim(ylim)
                # for cartesian angle-slice plots, keep B x-range in sync
                if self.angle_slice_type != "polar":
                    if self.axB2 is not None:
                        self.axB2.set_xlim(ylim)
                    if self.axB is not None:
                        self.axB.set_xlim(ylim)
            finally:
                self._syncing_limits = False

        def on_xlim_changed_B2(ax):
            # B2.x (angle) <-> A2.y (angle) for cartesian; also keep run 1 in sync
            if self._syncing_limits:
                return
            if self.axB2 is None:
                return
            self._syncing_limits = True
            try:
                xlim = self.axB2.get_xlim()
                if self.axA2 is not None:
                    self.axA2.set_ylim(xlim)
                if self.axA is not None:
                    self.axA.set_ylim(xlim)
                if self.axB is not None:
                    self.axB.set_xlim(xlim)
            finally:
                self._syncing_limits = False

        self.axA.callbacks.connect("xlim_changed", on_xlim_changed_A)
        self.axC.callbacks.connect("xlim_changed", on_xlim_changed_C)
        if self.angle_slice_type != "polar":
            self.axA.callbacks.connect("ylim_changed", on_ylim_changed_A)
            self.axB.callbacks.connect("xlim_changed", on_xlim_changed_B)

        # IMPORTANT: connect callbacks for run 2 axes too, so zoom/pan on 2A/2C works.
        # (Without this, run2 zoom changes never trigger synchronization.)
        # These axes exist only if the second row is allocated (which it always is),
        # but some may be "off" if no run2; guard by checking None.
        if self.axA2 is not None:
            self.axA2.callbacks.connect("xlim_changed", on_xlim_changed_A2)
            if self.angle_slice_type != "polar":
                self.axA2.callbacks.connect("ylim_changed", on_ylim_changed_A2)
        if self.axC2 is not None:
            self.axC2.callbacks.connect("xlim_changed", on_xlim_changed_C2)
        if self.angle_slice_type != "polar" and self.axB2 is not None:
            self.axB2.callbacks.connect("xlim_changed", on_xlim_changed_B2)

        # --- Plot A1: 2D color map ---
        extent0 = [shift0.min(), shift0.max(), angles0.min(), angles0.max()]
        self._imA = self.axA.imshow(
            I0,
            origin="lower",
            aspect="auto",
            extent=extent0,
        )
        self.figure.colorbar(self._imA, ax=self.axA, label="Intensity")

        self.axA.set_xlabel("Raman shift (cm$^{-1}$)")
        self.axA.set_ylabel("Angle (deg)")
        self.axA.set_title(f"{nickname0}: 2D map (Plot A)")

        secax0 = self.axA.secondary_xaxis("top", functions=(cm_to_ev, ev_to_cm))
        secax0.set_xlabel("Energy shift (eV)")

        # B1/C1: initially empty; will be filled on first click
        self.axB.set_title(f"angle slice")
        if self.angle_slice_type == "polar":
            self._adaptive_polar_locator(self.axB)
        if self.angle_slice_type != "polar":
            self.axB.set_xlabel("Angle (deg)")
            self.axB.set_ylabel("Intensity (a.u.)")
        self._lineB, = self.axB.plot([], [], "-k", linewidth=1.2)
        self._vlineB = self.axB.axvline(
            0.0, color="red", alpha=0.3, linewidth=1.0, visible=False
        )

        self.axC.set_title(f"spectral slice")
        self.axC.set_xlabel("Raman shift (cm$^{-1}$)")
        self.axC.set_ylabel("Intensity (a.u.)")
        secax_c0 = self.axC.secondary_xaxis("top", functions=(cm_to_ev, ev_to_cm))
        secax_c0.set_xlabel("Energy shift (eV)")
        self._lineC, = self.axC.plot([], [], "-k", linewidth=1.2)
        self._vlineC = self.axC.axvline(
            0.0, color="red", alpha=0.3, linewidth=1.0, visible=False
        )

        # ========================
        # Bottom row: second run (optional)
        # ========================
        if len(runs) > 1:
            run1 = runs[1]
            shift1 = np.asarray(run1.shift_cm1, dtype=float)
            angles1 = np.asarray(run1.angle_values, dtype=float)
            I1 = np.asarray(run1.intensity_2d, dtype=float)

            # Human-friendly label for the second run
            if self._experiment is not None:
                nickname1 = self._experiment.get_run_nickname(run1.id)
            else:
                nickname1 = run1.nickname

            self.axA2 = self.figure.add_subplot(gs[1, 0])
            if self.angle_slice_type == "polar":
                self.axB2 = self.figure.add_subplot(gs[1, 1], projection="polar")
            else:
                self.axB2 = self.figure.add_subplot(gs[1, 1])
            self.axC2 = self.figure.add_subplot(gs[1, 2])

            valid_second = True
            if I1.shape != (angles1.size, shift1.size):
                if I1.shape == (shift1.size, angles1.size):
                    I1 = I1.T
                else:
                    valid_second = False

            if valid_second:
                extent1 = [shift1.min(), shift1.max(), angles1.min(), angles1.max()]
                self._imA2 = self.axA2.imshow(
                    I1,
                    origin="lower",
                    aspect="auto",
                    extent=extent1,
                )
                self.figure.colorbar(self._imA2, ax=self.axA2, label="Intensity")

                self.axA2.set_xlabel("Raman shift (cm$^{-1}$)")
                self.axA2.set_ylabel("Angle (deg)")
                self.axA2.set_title(f"{nickname1}: 2D map (Plot A)")

                secax1 = self.axA2.secondary_xaxis("top", functions=(cm_to_ev, ev_to_cm))
                secax1.set_xlabel("Energy shift (eV)")

                self.axB2.set_title(f"angle slice")
                if self.angle_slice_type == "polar":
                    self._adaptive_polar_locator(self.axB2)
                if self.angle_slice_type != "polar":
                    self.axB2.set_xlabel("Angle (deg)")
                    self.axB2.set_ylabel("Intensity (a.u.)")
                self._lineB2, = self.axB2.plot([], [], "-k", linewidth=1.2)
                self._vlineB2 = self.axB2.axvline(
                    0.0, color="red", alpha=0.3, linewidth=1.0, visible=False
                )

                self.axC2.set_title(f"spectral slice")
                self.axC2.set_xlabel("Raman shift (cm$^{-1}$)")
                self.axC2.set_ylabel("Intensity (a.u.)")
                secax_c2 = self.axC2.secondary_xaxis(
                    "top", functions=(cm_to_ev, ev_to_cm)
                )
                secax_c2.set_xlabel("Energy shift (eV)")
                self._lineC2, = self.axC2.plot([], [], "-k", linewidth=1.2)
                self._vlineC2 = self.axC2.axvline(
                    0.0, color="red", alpha=0.3, linewidth=1.0, visible=False
                )
            else:
                self.axA2.text(
                    0.5,
                    0.5,
                    "Run 2: invalid 2D shape",
                    ha="center",
                    va="center",
                    transform=self.axA2.transAxes,
                )
                self.axA2.set_axis_off()
                self.axB2.set_axis_off()
                self.axC2.set_axis_off()
                self._imA2 = None
        else:
            # No second run: blank bottom row (keeps overall layout)
            self.axA2 = self.figure.add_subplot(gs[1, 0])
            self.axB2 = self.figure.add_subplot(gs[1, 1])
            self.axC2 = self.figure.add_subplot(gs[1, 2])
            self.axA2.set_axis_off()
            self.axB2.set_axis_off()
            self.axC2.set_axis_off()
            self._imA2 = None

        self._rebind_limit_sync_callbacks()

        if self.angle_slice_type != "polar":
            if self.axA is not None and self.axB is not None:
                self.axB.set_xlim(self.axA.get_ylim())
        if self.axA is not None and self.axC is not None:
            self.axC.set_xlim(self.axA.get_xlim())
        if self.angle_slice_type != "polar":
            if self.axA2 is not None and self.axB2 is not None:
                self.axB2.set_xlim(self.axA2.get_ylim())
        if self.axA2 is not None and self.axC2 is not None:
            self.axC2.set_xlim(self.axA2.get_xlim())

        # Click event on Plot A (run 1) only
        if self._cid_click is not None:
            self.canvas.mpl_disconnect(self._cid_click)
        self._cid_click = self.canvas.mpl_connect(
            "button_press_event", self._on_canvas_click
        )
        if self._cid_resize is not None:
            self.canvas.mpl_disconnect(self._cid_resize)
        self._cid_resize = self.canvas.mpl_connect(
            "resize_event", self._on_mpl_resize
        )

        # self.figure.tight_layout()
        self.figure.subplots_adjust(top=0.862)

    def _on_canvas_click(self, event):
        """Handle mouse clicks in Plot A: update crosshair and slices (syncs both runs)."""
        if (
            getattr(event, "button", None) == 3
            and (event.inaxes is self.axB or event.inaxes is self.axB2)
        ):
            self._popup_angle_slice_type_menu(event)
            return
        if event.inaxes is not self.axA and event.inaxes is not self.axA2:
            return
        if event.xdata is None or event.ydata is None:
            return

        def _get_nickname(run: Run) -> str:
            if self._experiment is not None:
                return self._experiment.get_run_nickname(run.id)
            return run.nickname

        def _prepare_run_data(run: Run):
            shift = np.asarray(run.shift_cm1, dtype=float)
            angles = np.asarray(run.angle_values, dtype=float)
            I = np.asarray(run.intensity_2d, dtype=float)
            if I.shape != (angles.size, shift.size):
                if I.shape == (shift.size, angles.size):
                    I = I.T
                else:
                    return None
            return shift, angles, I

        def _select_from_click(shift, angles, I, im, x, y):
            x0, x1, y0, y1 = im.get_extent()
            ny, nx = I.shape
            if x1 == x0 or y1 == y0:
                return None
            j_float = (x - x0) / (x1 - x0) * nx - 0.5
            i_float = (y - y0) / (y1 - y0) * ny - 0.5
            ix = int(np.clip(np.round(j_float), 0, nx - 1))
            iy = int(np.clip(np.round(i_float), 0, ny - 1))
            # Do NOT return x_sel/y_sel as authoritative state.
            # Index (ix,iy) is the persistent selection; physical coords are derived later.
            x_hl = x0 + (ix + 0.5) * (x1 - x0) / nx
            y_hl = y0 + (iy + 0.5) * (y1 - y0) / ny
            return ix, iy, float(x_hl), float(y_hl)

        def _select_nearest(shift, angles, I, im, x_sel, y_sel):
            ix = int(np.argmin(np.abs(shift - x_sel)))
            iy = int(np.argmin(np.abs(angles - y_sel)))
            x_sel2 = shift[ix]
            y_sel2 = angles[iy]
            x0, x1, y0, y1 = im.get_extent()
            ny, nx = I.shape
            if x1 != x0 and y1 != y0:
                x_hl = x0 + (ix + 0.5) * (x1 - x0) / nx
                y_hl = y0 + (iy + 0.5) * (y1 - y0) / ny
            else:
                x_hl = x_sel2
                y_hl = y_sel2
            return ix, iy, x_sel2, y_sel2, x_hl, y_hl

        def _update_crosshair(ax, vline, hline, x_hl, y_hl):
            if self.highlight_mode == "click":
                if vline is None:
                    vline = ax.axvline(
                        x_hl, color="red", alpha=0.3, linewidth=1.0
                    )
                else:
                    vline.set_xdata([x_hl, x_hl])
                    vline.set_visible(True)
                if hline is None:
                    hline = ax.axhline(
                        y_hl, color="red", alpha=0.3, linewidth=1.0
                    )
                else:
                    hline.set_ydata([y_hl, y_hl])
                    hline.set_visible(True)
            else:
                if vline is not None:
                    vline.set_visible(False)
                if hline is not None:
                    hline.set_visible(False)
            return vline, hline

        def _update_slice_b(ax, line, vline, angles, prof, title, vline_x):
            if ax is None:
                return line, vline
            if self.angle_slice_type == "polar":
                th = np.deg2rad(np.asarray(angles, float))

                ax.clear()
                ax.set_theta_zero_location("N")
                ax.set_theta_direction(-1)

                ax.plot(th, prof, "o", ms=3, linestyle="None")

                vals = np.asarray(prof, float)
                vals = vals[np.isfinite(vals)]
                rmax = float(vals.max()) if vals.size else 1.0
                if not np.isfinite(rmax) or rmax <= 0:
                    rmax = 1.0
                pad = 0.05 * rmax
                ax.set_rlim(0.0, rmax + pad)

                ax.set_title(title)

                theta_sel = np.deg2rad(float(vline_x))
                ax.axvline(theta_sel, color="red", alpha=0.3, linewidth=1.0)
                self._adaptive_polar_locator(ax)
                return None, None
            if line is None:
                line, = ax.plot([], [], "-k", linewidth=1.2)
            line.set_data(angles, prof)
            ax.set_title(title)
            ax.set_xlabel("Angle (deg)")
            ax.set_ylabel("Intensity (a.u.)")
            if vline is None:
                vline = ax.axvline(
                    vline_x, color="red", alpha=0.3, linewidth=1.0
                )
            else:
                vline.set_xdata([vline_x, vline_x])
                vline.set_visible(True)
            ax.relim()
            ax.autoscale_view(scalex=False, scaley=True)
            return line, vline

        def _update_slice_c(ax, line, vline, shift, prof, title, vline_x):
            if ax is None:
                return line, vline
            if line is None:
                line, = ax.plot([], [], "-k", linewidth=1.2)
            line.set_data(shift, prof)
            ax.set_title(title)
            ax.set_xlabel("Raman shift (cm$^{-1}$)")
            ax.set_ylabel("Intensity (a.u.)")
            if vline is None:
                vline = ax.axvline(
                    vline_x, color="red", alpha=0.3, linewidth=1.0
                )
            else:
                vline.set_xdata([vline_x, vline_x])
                vline.set_visible(True)
            ax.relim()
            ax.autoscale_view(scalex=False, scaley=True)
            return line, vline

        # Determine which Plot A was clicked; infer indices on that run only.
        clicked_first = (event.inaxes is self.axA)

        if clicked_first:
            if self.current_run is None or self._imA is None:
                return
            base_run = self.current_run
            base_im = self._imA
        else:
            if self._second_run is None or self._imA2 is None:
                return
            base_run = self._second_run
            base_im = self._imA2

        base_data = _prepare_run_data(base_run)
        if base_data is None:
            return
        base_shift, base_angles, base_I = base_data

        sel = _select_from_click(
            base_shift,
            base_angles,
            base_I,
            base_im,
            float(event.xdata),
            float(event.ydata),
        )
        if sel is None:
            return
        ix, iy, x_hl, y_hl = sel

        # Persist selection as INDICES (drift-free).
        if clicked_first:
            self._sel_idx1 = (ix, iy)
            # Invalidate idx2; it will be re-derived from idx1 during apply.
            self._sel_idx2 = None
        else:
            # If clicked on run2, we set idx2 directly, and also derive idx1
            # from run2 physical coords (nearest on run1) for a consistent "linked" selection.
            self._sel_idx2 = (ix, iy)
            if self.current_run is not None:
                d1 = _prepare_run_data(self.current_run)
                if d1 is not None:
                    shift1, angles1, I1 = d1
                    x_sel2 = float(base_shift[ix])
                    y_sel2 = float(base_angles[iy])
                    ix1 = int(np.argmin(np.abs(shift1 - x_sel2)))
                    iy1 = int(np.argmin(np.abs(angles1 - y_sel2)))
                    self._sel_idx1 = (ix1, iy1)

        # Optional: keep _last_coords as an informational/debug variable only.
        # It MUST NOT drive home/refresh anymore.
        self._last_coords = (float(base_shift[ix]), float(base_angles[iy]))

        # Apply selection to both runs' highlights and B/C plots without reselecting.
        self._apply_selection_from_indices(reason="click")


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
        item_import = file_menu.Append(wx.ID_ANY, "Import Run...\tCtrl-I")
        item_merge = file_menu.Append(wx.ID_ANY, "Merge Run...\tCtrl-Shift-M")
        file_menu.AppendSeparator()
        item_quit = file_menu.Append(wx.ID_EXIT, "Quit\tCtrl-Q")
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

        bottom_notebook.AddPage(self.preview_panel, "Preview")
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

        panel = ViewPanel(self.view_notebook, view_label=title)
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
