# Venkata: Raman GUI for 1D / 2D / Angular Maps

Venkata is a small, experiment-driven Raman spectroscopy GUI aimed at day-to-day use by physicists.

It is designed to:

- Load 1D and 2D / angular Raman data from simple text / CSV / Excel files.
- Organize measurements into an **ExperimentSet** (runs + views + metadata).
- Inspect up to **two runs side-by-side** in a VESTA-like layout with synchronized interactions.
- Provide a simple workflow for merging multiple 1D spectra into a 2D angular map.
- Export 2D data to publication-ready CSV files.
- Keep the GUI relatively simple and lightweight, while pushing heavy analysis
  (cosmic rays, curve fitting, etc.) into a separate non-GUI layer.

The codebase is split into a few cooperating layers:

- `data_structure.py` – core data model (Run / ExperimentSet / ViewState), persistence, CSV/Igor export, and filename/unit helpers.
- `analysis.py` – higher-level analysis (cosmic rays, curve fitting, plotting-related data preparation).
- `plotting.py` – matplotlib plotter classes used by the wx views.
- `wx_gui.py` – main frame, menus, event wiring, and application control.
- `wx_left_panel.py`, `wx_left_lower_panel.py`, `wx_right_panel.py` – file/run panels, plot controls, appearance controls, and view rendering.
- `merge_runs_gui.py` – merge-preview dialog and merge-specific plotting controls.
- `plot_export_cli.py` – headless plot export from saved experiment files.

The intent is that each file remains small and well-scoped so that another person (or an AI assistant)
can safely modify one layer without breaking the others.

---

## 1. Running the application

From the `Venkata` directory:

```bash
python wx_gui.py
```

While you can still provide initial files as command-line arguments, the primary way to load data is through the `File` menu.

- **`File -> Import Run(s)...`**: Opens a dialog to select one or more data files to import as individual runs.
- **`File -> Merge Run...`**: Starts the process of combining multiple 1D spectra into a 2D run.
- **`File -> Export to Igor Pro...`**: Writes all numeric runs in the current experiment to Igor Text (`.itx`) without requiring `igor2`.

Saved experiments can also be exported from the command line:

```bash
python wx_gui.py export-plot experiment.h5 --output plot.png
python wx_gui.py export-plot experiment.h5 --list
```

---

## 2. Modules and responsibilities

### 2.1 wx_gui.py (GUI / controller)

- Owns the wxPython application and main event loop.
- Builds and manages the VESTA-like window layout.
- Acts as the controller between GUI widgets, the data model, and analysis helpers.
- Handles file import/merge/export actions, experiment and view management, and logging.
- Saves live plot panel state back into `ViewState` before experiment save/close.

### 2.2 data_structure.py (core model / basic I/O)

Defines the experiment-side data model:

- **Run**
  - Represents a single Raman measurement (1D or 2D angular map).
  - Stores source file path, axes, intensity data, and metadata.
  - Metadata includes a user-facing nickname (Run01, Run02, …) and filename-parsed fields.

- **ExperimentSet**
  - Collects runs and views.
  - Manages unique run nicknames and experiment-level metadata.

- **ViewState**
  - Describes what is plotted in one view tab (up to two runs).
  - Keeps legacy whole-view plot fields (`xlim`, `ylim`, `vmin`, `vmax`, `cmap`) for compatibility.
  - Stores newer per-panel graph settings in `graph_configs` keyed by `1A`, `1B`, `1C`, `2A`, `2B`, and `2C`.

- **GraphViewConfig**
  - Stores per-panel axis limits, colormap, percentile contrast, and optional absolute color limits.

This module contains no GUI code.

### 2.3 analysis.py (higher-level analysis)

Provides numerical and I/O helpers without GUI dependencies:

- **Merge Preview**: Discovers and previews how multiple 1D runs will be combined into a 2D map.
- **Cosmic Ray Correction**: Placeholder for cosmic ray rejection algorithms.
- Planned extensions: baseline correction, curve fitting.

---

## 3. GUI and Feature Summary

### Main Window
- **Left Pane**: Contains notebooks for file browsing, experiment management (listing runs and views), metadata display, and logging.
- **Right Pane**: Contains one or more "View" tabs for data visualization.
- **Plots**: Each view shows up to two runs in a 2x3 layout: 2D map (Plot A), angular slice (Plot B), and spectral slice (Plot C).

### Key Features and Interactions

- **Data Import**:
  - Load individual runs via `File -> Import Run(s)...`.
  - Runs are listed in the "Experiment" panel.

- **Merging 1D Spectra**:
  - Initiate a merge via `File -> Merge Run...`.
  - A dialog appears to preview the merge, allowing for cosmic ray correction and selection of the final X-axis units (`cm-1` or `meV`).
  - The newly created 2D run is added to the experiment.

- **Data Export**:
  - In the "Experiment" panel, right-click on one or more runs and select "Export Run...".
  - All selected runs will be exported.
  - A directory dialog will open, allowing you to choose where to save the files.
  - For each exported run, two CSV files are created (e.g., `..._angular_matrix_cm-1.csv` and `..._angular_matrix_meV.csv`). The filenames are generated automatically from the run's metadata.
  - `File -> Export to Igor Pro...` writes the current experiment to `.itx`, including companion axis waves for Raman shift, wavelength, energy, or angle data when available.
  - `python wx_gui.py export-plot ...` exports saved views or runs headlessly with optional `--view`, `--run`, `--panel`, `--xlim`, `--ylim`, `--cmap`, `--clim`, and slice-position overrides.
  - `python wx_gui.py agent ...` provides agent-friendly listing, plot export, data export, and raw-file experiment creation commands for saved `ExperimentSet` files such as `RamanData.dat`.

- **Curve Fitting**:
  - Right-click on any 1D trace (Angular slice or Spectral slice) and select **"Curve Fit..."**.
  - The "Curve Fit" tab (bottom-left) will activate with the selected data loaded.
  - Choose a model (Lorentzian, Sum of Lorentzian, or User Defined), adjust parameters, and click **Fit**.
  - Click **Create Curve** to save the fit result as a new "Derived Run", which will automatically overlay on the original plot.

- **Plot Interaction**:
  - Clicking on a 2D map selects a Raman shift and angle, updating the slice plots.
  - Plot Config includes per-view X/Y/All slice binning controls. Cross binning averages only the orthogonal direction; Box binning also smooths along the slice direction.
  - Plot Config includes a per-view fit overlay mode (`Off`, `Global`, `Row`, `Both`) for fitted 2D map runs.
  - Zoom and pan are synchronized across all relevant plots.
  - The 2D maps are rendered with `pcolormesh` for accurate visualization of non-uniform data.
  - Right-click on a plot panel to copy the visible data region as a bitmap; 1D panels also offer a lightweight single-panel vector copy for editing lines and axes in illustration tools.
  - Spectral axes are dual-unit (`meV` or `cm-1`) with `meV` as the default display unit for new views.
  - The Preferences tab stores app-wide defaults for new views and ambiguous imports, including unit, colormap, contrast, angle-slice type, slice binning, secondary-axis behavior, and optional Ctrl/Cmd-click-only highlight movement.
  - Plot range and colormap settings are saved per graph panel, while older experiment files still load through the legacy whole-view fields.

---

## 4. Design notes

- **Clear separation of concerns**:
  - Model and IO in `data_structure.py`.
  - Numerical analysis in `analysis.py`.
  - Rendering helpers in `plotting.py`.
  - Main wx event wiring in `wx_gui.py`.
  - Panel-specific widgets in `wx_left_panel.py`, `wx_left_lower_panel.py`, and `wx_right_panel.py`.
- Run nicknames are first-class, user-visible labels and propagate across the UI.
- The design favors correctness and maintainability over visual complexity.

## 5. Code structure and maintenance notes

- `Run` is the source of truth for loaded numeric data. Prefer adding data import/export logic there or in small helpers that do not import wx.
- `ExperimentSet` owns persistence. When adding persistent view fields, make the loader tolerant of older HDF5/JSON dicts and preserve existing legacy fields.
- `ViewState.run_configs` controls per-run style and overlay behavior. `ViewState.graph_configs` controls per-panel plot appearance and limits.
- `ViewPanel` owns live matplotlib axes. It should push live axis/color state back into `ViewState` before save/close and after user-driven range or colormap changes.
- `PlotConfigPanel` edits the currently targeted `ViewPanel`; it should not directly mutate experiments except through the view panel API.
- Custom colormaps live in `settings.json` under `custom_colormaps` and are registered into matplotlib before plotting or CLI export.
- `plot_export_cli.py` intentionally uses the Agg backend and should stay wx-free so it can run in scripts and tests.
- Igor export writes plain `.itx` text directly. Keep this dependency-free unless there is a strong reason to add and vendor an Igor writer.
- For GUI bugs, first inspect `wx_gui.py` for event routing, then the relevant left/right panel file for widget state, then `plotting.py` for matplotlib behavior.

## 6. Development Roadmap

For a detailed list of planned features, bug fixes, and completed work, please see `TODO.md`.
