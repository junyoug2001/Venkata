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

The codebase is split into three main modules:

- `wx_gui.py` – GUI, event wiring, and application control.
- `data_structure.py` – core data model (Run / ExperimentSet / ViewState) and basic filename / IO / unit helpers.
- `analysis.py` – higher-level analysis (cosmic rays, curve fitting, plotting-related data preparation).

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

---

## 2. Modules and responsibilities

### 2.1 wx_gui.py (GUI / controller)

- Owns the wxPython application and main event loop.
- Builds and manages the VESTA-like window layout.
- Acts as the controller between GUI widgets, the data model, and analysis helpers.
- Handles file import/merge/export actions, experiment and view management, and logging.

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
  - The export is **checkbox-based**: all checked runs will be exported.
  - A directory dialog will open, allowing you to choose where to save the files.
  - For each exported run, two CSV files are created (e.g., `..._angular_matrix_cm-1.csv` and `..._angular_matrix_meV.csv`). The filenames are generated automatically from the run's metadata.

- **Plot Interaction**:
  - Clicking on a 2D map selects a Raman shift and angle, updating the slice plots.
  - Zoom and pan are synchronized across all relevant plots.
  - The 2D maps are rendered with `pcolormesh` for accurate visualization of non-uniform data.

---

## 4. Design notes

- **Clear separation of concerns**:
  - Model and IO in `data_structure.py`.
  - Numerical analysis in `analysis.py`.
  - Rendering and interaction in `wx_gui.py`.
- Run nicknames are first-class, user-visible labels and propagate across the UI.
- The design favors correctness and maintainability over visual complexity.

## 5. Development Roadmap

For a detailed list of planned features, bug fixes, and completed work, please see `TODO.md`.
