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
    new_run_id,
    RunType
)
from plotting import RamanPlotter2d, AngularPlotter, SlicePlotter

from wx_left_panel import FilesPanel, RunsPanel, ExperimentPanel, LogPanel
from wx_left_lower_panel import PreviewPanel, CurveFitPanel, PlotConfigPanel, AppearancesPanel
from wx_right_panel import RamanToolbar, ViewPanel


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
        item_new_formula = file_menu.Append(wx.ID_ANY, "New Formula Run...")
        item_merge = file_menu.Append(wx.ID_ANY, "Merge Run...\tCtrl-Shift-I")
        item_calc = file_menu.Append(wx.ID_ANY, "Calculate from run...")
        file_menu.AppendSeparator()
        item_quit = file_menu.Append(wx.ID_EXIT, "Quit\tCtrl-Q")
        
        self.Bind(wx.EVT_MENU, self.on_open_experiment, item_open)
        self.Bind(wx.EVT_MENU, self.on_save_experiment, item_save)
        self.Bind(wx.EVT_MENU, self.on_import_run_dialog, item_import)
        self.Bind(wx.EVT_MENU, self.on_new_formula_run, item_new_formula)
        self.Bind(wx.EVT_MENU, self.on_merge_run_dialog, item_merge)
        self.Bind(wx.EVT_MENU, self.on_calculate_from_run, item_calc)
        self.Bind(wx.EVT_MENU, self.on_quit, item_quit)
        menubar.Append(file_menu, "&File")

        # View menu
        view_menu = wx.Menu()
        item_new_view = view_menu.Append(wx.ID_ANY, "New View Tab\tCtrl+T")
        item_close_view = view_menu.Append(wx.ID_ANY, "Close View Tab\tCtrl+W")
        
        self.Bind(wx.EVT_MENU, self.on_new_view, item_new_view)
        self.Bind(wx.EVT_MENU, self.on_close_current_view, item_close_view)
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
        
        # Refresh appearances panel if a view is active
        try:
            view_id = self._get_current_view_id()
            if view_id:
                view_state = self.experiment.get_view(view_id)
                self.appearances_panel.update_view(view_state, self.experiment)
            else:
                self.appearances_panel.update_view(None, None)
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

    def on_calculate_from_run(self, event):
        """
        Open a dialog to select a run, then perform a calculation (currently angle summation).
        """
        # Get list of runs
        run_ids = list(self.experiment.runs.keys())
        if not run_ids:
            wx.MessageBox("No runs available.", "Error", wx.OK | wx.ICON_ERROR)
            return

        choices = [f"{self.experiment.get_run_nickname(rid)} ({rid})" for rid in run_ids]
        
        with wx.SingleChoiceDialog(self, "Select a run to calculate from:", "Calculate from Run", choices) as dlg:
            if dlg.ShowModal() != wx.ID_OK:
                return
            selection = dlg.GetSelection()
            run_id = run_ids[selection]
        
        run = self.experiment.get_run(run_id)
        if not run:
             return

        # For now, we only support Angle Summation on 2D runs.
        if not run.is_2d:
             wx.MessageBox("Selected run is not a 2D run. Only angle summation of 2D runs is currently supported.", "Info", wx.OK | wx.ICON_INFORMATION)
             return

        # Perform summation
        # intensity_2d shape is (angles, shifts)
        # Sum along axis 0 to get (shifts,)
        if run.intensity_2d is None:
             return

        summed_intensity = np.sum(run.intensity_2d, axis=0)
        
        new_nickname = f"{run.nickname}_summed"
        
        new_metadata = run.metadata.copy()
        new_metadata["nickname"] = new_nickname
        new_metadata["calculation"] = "angle_sum"
        new_metadata["source_run_id"] = run.id
        new_metadata["raw_dim"] = "1d"

        new_run = Run(
            id=new_run_id(prefix="calc"),
            source_path=run.source_path, 
            source_mtime=run.source_mtime,
            wl_nm=run.wl_nm,
            shift_cm1=run.shift_cm1,
            energy_eV=run.energy_eV,
            intensity=summed_intensity,
            intensity_2d=None,
            angle_values=None,
            intensity_unit=run.intensity_unit,
            angle_unit=run.angle_unit,
            metadata=new_metadata,
            run_type=RunType.RUN_1D,
            raw_table=None
        )

        self.experiment.add_run(new_run)
        self.log_panel.append_log(f"Created summed run: {new_nickname}")
        self._refresh_left_panels()

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
            on_remove_run=self.on_remove_run,
            on_edit_formula=self.on_edit_formula,
        )
        self.experiment_panel = ExperimentPanel(
            top_notebook,
            on_edit_formula=self.on_edit_formula,
            on_edit_derived_props=self.on_edit_derived_props
        )
        self.log_panel = LogPanel(top_notebook)

        top_notebook.AddPage(self.files_panel, "Files")
        top_notebook.AddPage(self.runs_panel, "Experiment")
        top_notebook.AddPage(self.experiment_panel, "Metadata")
        top_notebook.AddPage(self.log_panel, "Log")

        # --- bottom notebook (Preview / Curve Fit) ---
        bottom_notebook = wx.Notebook(self.left_splitter, style=wx.NB_TOP)

        self.preview_panel = PreviewPanel(bottom_notebook)
        self.curvefit_panel = CurveFitPanel(bottom_notebook, on_run_created=self.on_run_created)
        self.plot_config_panel = PlotConfigPanel(bottom_notebook, 
                                                 on_reset=self.on_plot_reset)
        self.appearances_panel = AppearancesPanel(
            bottom_notebook, 
            on_rename_run=self.on_rename_run, 
            on_style_change=self.on_style_change
        )

        bottom_notebook.AddPage(self.preview_panel, "Preview")
        bottom_notebook.AddPage(self.plot_config_panel, "Plot Config.")
        bottom_notebook.AddPage(self.appearances_panel, "Appearance")
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
        wx.CallAfter(self._init_view_tab)
        
        # Bind notebook page changed
        self.view_notebook.Bind(wx.EVT_NOTEBOOK_PAGE_CHANGED, self.on_view_page_changed)

    def _init_view_tab(self):
        self.add_view_tab()
        # Manually trigger target setting for the initial view
        panel = self.get_current_view_panel()
        if panel:
            self.plot_config_panel.set_target_view(panel)

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
        self._refresh_left_panels()

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


    def on_new_formula_run(self, event):
        """Create a new derived run with a default formula."""
        # Create a default sine wave
        new_run = Run.from_formula(
            formula="np.sin(x/10.0 * freq) * amp",
            params={"amp": 100.0, "freq": 1.0},
            n_points=200,
            x_range=(0, 500),
            nickname="New Formula Run"
        )
        self.experiment.add_run(new_run)
        self.log_panel.append_log(f"Created new formula run: {new_run.id}")
        self._refresh_left_panels()
        
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

    def on_edit_formula(self, run_id: str, new_formula: str, new_params: Dict[str, float]) -> None:
        """
        Update the formula and parameters for a Derived Run and refresh views.
        """
        run = self.experiment.get_run(run_id)
        if not run or run.run_type != RunType.DERIVED:
            return

        run.metadata["formula"] = new_formula
        run.metadata["formula_params"] = new_params
        
        nickname = self.experiment.get_run_nickname(run_id)
        self.log_panel.append_log(f"Updated formula for derived run '{nickname}' ({run_id})")

        # Refresh all views that might be displaying this run
        for view_id, view_state in self.experiment.views.items():
            if run_id in view_state.run_ids:
                self._update_view_plot(view_id, preserve_state=True)
        
        # Also refresh metadata tree to show updated formula
        self.experiment_panel.refresh_from_experiment(self.experiment)

    def on_edit_derived_props(self, run_id: str, n_points: int, autorange: bool, x_range: Tuple[float, float]) -> None:
        """
        Update derived run properties and refresh views.
        """
        run = self.experiment.get_run(run_id)
        if not run or run.run_type != RunType.DERIVED:
            return
            
        run.metadata["default_n_points"] = n_points
        run.metadata["default_autorange"] = autorange
        run.metadata["default_range"] = x_range
        
        self.log_panel.append_log(f"Updated properties for derived run {run.nickname}")
        
        # Refresh views
        for view_id, view_state in self.experiment.views.items():
            if run_id in view_state.run_ids:
                # Update run_config override if it exists, or just clear it to use metadata defaults?
                # The logic in ViewPanel._update_run_plots uses run_config override OR metadata default.
                # If run_config has an override, editing metadata won't change the plot unless we clear the override.
                # For now, let's just refresh. If the user previously set a View-specific override in Appearance panel,
                # that should probably persist. If they want to reset, they'd use Appearance panel.
                # Here we are editing the "global" default for the Run.
                self._update_view_plot(view_id, preserve_state=True)

        self.experiment_panel.refresh_from_experiment(self.experiment)

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
            view_label=title, plot_config_panel=self.plot_config_panel,
            on_run_created=self.on_run_created,
            on_fit_request=self.on_curve_fit_request
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
    
    def on_run_created(self, new_run: Run, overlay_target_run: Optional[Run] = None, overlay_plot_type: str = None) -> None:
        """Callback from ViewPanel or CurveFitPanel when a new run is created."""
        self.experiment.add_run(new_run)
        nickname = self.experiment.get_run_nickname(new_run.id)
        self.log_panel.append_log(
            f"Created new run '{nickname}' ({new_run.id})."
        )
        
        # Handle automatic overlay if requested
        if overlay_target_run and overlay_plot_type:
            view_id = self._get_current_view_id()
            if view_id:
                view_state = self.experiment.get_view(view_id)
                # Check if target run is in this view
                slot_prefix = None
                if len(view_state.run_ids) > 0 and view_state.run_ids[0] == overlay_target_run.id:
                    slot_prefix = "1"
                elif len(view_state.run_ids) > 1 and view_state.run_ids[1] == overlay_target_run.id:
                    slot_prefix = "2"
                
                if slot_prefix:
                    target_code = f"{slot_prefix}{overlay_plot_type}"
                    
                    # Add new run to view if not present (it shouldn't be yet)
                    if new_run.id not in view_state.run_ids:
                        view_state.run_ids.append(new_run.id)
                    
                    # Set overlay config
                    cfg = view_state.get_run_config(new_run.id)
                    cfg.overlay_target = target_code
                    
                    # Also default the style color to red or something distinct?
                    # For now default black is fine, or random.
                    
                    self.log_panel.append_log(f"Overlaying {nickname} onto {target_code}")

        self._refresh_left_panels()
        # If we modified the view, we need to refresh it
        if overlay_target_run:
            self._refresh_all_view_panels()

    def on_curve_fit_request(self, source_run: Run, plot_type: str, x_data: np.ndarray, y_data: np.ndarray):
        """
        Handle request to fit a curve to the given data.
        Switches to CurveFitPanel and loads data.
        """
        # 1. Switch Left Splitter to Bottom
        # The sash position might hide it, but we can't easily force it open without sizing.
        # Assuming user has it visible or we just focus the notebook.
        
        # 2. Select CurveFit tab (index 3 currently: Preview, PlotConfig, Appearance, CurveFit)
        # Better to find by name or instance
        for i in range(self.curvefit_panel.GetParent().GetPageCount()):
            if self.curvefit_panel.GetParent().GetPage(i) == self.curvefit_panel:
                self.curvefit_panel.GetParent().SetSelection(i)
                break
        
        # 3. Load Data
        self.curvefit_panel.set_data(source_run, plot_type, x_data, y_data)
        self.log_panel.append_log(f"Started curve fit for {source_run.nickname} ({plot_type})")

    
    # -------- view management helpers --------

    def _get_current_view_id(self) -> Optional[str]:
        """Return the view_id for the currently selected view tab, if any."""
        page_index = self.view_notebook.GetSelection()
        if page_index == wx.NOT_FOUND:
            return None
        return self._view_page_to_id.get(page_index)

    def _update_view_plot(self, view_id: str, preserve_state: bool = False) -> None:
        """
        Refresh the plotting for the specified view, if its panel exists.
        """
        view_state = self.experiment.get_view(view_id)
        if view_state is None:
            return
        panel = self._view_id_to_panel.get(view_id)
        if panel is None:
            return
        panel.set_view_model(self.experiment, view_state, preserve_state=preserve_state)
        # Also update appearances tab if relevant
        if view_id == self._get_current_view_id():
            self.appearances_panel.update_view(view_state, self.experiment)

    def get_current_view_panel(self) -> Optional[ViewPanel]:
        view_id = self._get_current_view_id()
        if view_id:
            return self._view_id_to_panel.get(view_id)
        return None

    def on_plot_reset(self):
        view_id = self._get_current_view_id()
        if view_id:
            self._update_view_plot(view_id, preserve_state=False)

    def on_view_page_changed(self, event):
        view_panel = self.get_current_view_panel()
        self.plot_config_panel.set_target_view(view_panel)
        
        # Sync Appearances Panel
        view_id = self._get_current_view_id()
        if view_id:
            view_state = self.experiment.get_view(view_id)
            self.appearances_panel.update_view(view_state, self.experiment)
        else:
            self.appearances_panel.update_view(None, None)
        
        event.Skip()

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

    def on_style_change(self, run_id: str, attr: str, value: Any, component: Optional[str] = None) -> None:
        """
        Update run style based on edits in AppearancesPanel.
        attr: 'visible', 'color', 'linestyle', 'linewidth', 'style_string'
        value: bool or str
        component: "A", "B", "C" or None (apply to all)
        """
        view_id = self._get_current_view_id()
        if not view_id:
            return
        
        view_state = self.experiment.get_view(view_id)
        if not view_state:
            return

        run_config = view_state.get_run_config(run_id)
        
        # Determine targets: specific component or all
        targets = [component] if component else ["A", "B", "C"]
        
        for comp in targets:
            style = run_config.get_style(comp)
            if attr == "style_string":
                parts = [p.strip() for p in str(value).split(",")]
                if len(parts) > 0 and parts[0]: style.color = parts[0]
                if len(parts) > 1 and parts[1]: style.linestyle = parts[1]
                if len(parts) > 2 and parts[2]:
                    try:
                        style.linewidth = float(parts[2])
                    except ValueError: pass
                if len(parts) > 3: style.marker = parts[3]
                if len(parts) > 4 and parts[4]:
                    try:
                        style.markersize = float(parts[4])
                    except ValueError: pass
            elif attr == "visible":
                style.visible = bool(value)
            elif attr == "color":
                style.color = str(value)
            elif attr == "linestyle":
                style.linestyle = str(value)
            elif attr == "linewidth":
                try:
                    style.linewidth = float(value)
                except ValueError:
                    pass
            elif attr == "overlay_target":
                run_config.overlay_target = str(value) if str(value) != "None" else None
            elif attr == "derived_n_points":
                run_config.derived_n_points = int(value)
            elif attr == "derived_autorange":
                run_config.derived_autorange = bool(value)
            elif attr == "derived_range":
                if isinstance(value, (tuple, list)) and len(value) == 2:
                    run_config.derived_range = tuple(value)
        
        self.log_panel.append_log(f"Updated style '{attr}' for run {run_id} (comp={component or 'all'})")
        # If overlay_target changed, we might need a full redraw because the primary run list might change
        if attr in ["overlay_target", "derived_n_points", "derived_autorange", "derived_range"]:
            self._update_view_plot(view_id, preserve_state=True)
        else:
            self._update_view_plot(view_id, preserve_state=True)

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

    def on_remove_run(self, run_ids: List[str]) -> None:
        """
        Remove the specified runs from the experiment.
        """
        if not run_ids:
            return

        for run_id in run_ids:
            run = self.experiment.get_run(run_id)
            if run is None:
                continue
            
            nickname = run.nickname
            self.experiment.remove_run(run_id)
            self.log_panel.append_log(f"Removed run {nickname} ({run_id})")

        self.runs_panel.refresh_from_experiment(self.experiment)
        self.experiment_panel.refresh_from_experiment(self.experiment)
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
                plot_config_panel=self.plot_config_panel,
                on_run_created=self.on_run_created,
                on_fit_request=self.on_curve_fit_request
            )
            self.view_notebook.AddPage(panel, vstate.title)
            page_index = self.view_notebook.GetPageCount() - 1
            self._view_page_to_id[page_index] = vid
            self._view_id_to_panel[vid] = panel
            panel.set_view_model(self.experiment, vstate)
        
        # Ensure panels are connected to the current (first) view
        if self.view_notebook.GetPageCount() > 0:
            # Manually trigger updates as if the page changed
            self.view_notebook.SetSelection(0)
            self.on_view_page_changed(wx.BookCtrlEvent(wx.EVT_NOTEBOOK_PAGE_CHANGED.typeId, 0, 0))

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

    def on_close_current_view(self, event):
        """Close the currently selected view tab, with confirmation if it has runs."""
        view_id = self._get_current_view_id()
        if not view_id:
            return

        view_state = self.experiment.get_view(view_id)
        if not view_state:
            return

        # If the view has attached runs, ask for confirmation
        if view_state.run_ids:
            msg = (
                f"View '{view_state.title}' has {len(view_state.run_ids)} run(s) attached.\n"
                "Are you sure you want to close it?"
            )
            dlg = wx.MessageDialog(self, msg, "Close View", wx.YES_NO | wx.NO_DEFAULT | wx.ICON_WARNING)
            if dlg.ShowModal() != wx.ID_YES:
                dlg.Destroy()
                return
            dlg.Destroy()

        self.on_remove_view(view_id)

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


