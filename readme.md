# Venkata: Raman GUI for 1D / 2D / Angular Maps

Venkata is a small, experiment-driven Raman spectroscopy GUI aimed at day-to-day use by physicists.

It is designed to:

- Load 1D and 2D / angular Raman data from simple text / CSV / Excel files.
- Organize measurements into an **ExperimentSet** (runs + views + metadata).
- Inspect up to **two runs side-by-side** in a VESTA-like layout with synchronized interactions.
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
python wx_gui.py file1.csv file2.csv
```

Current behavior:

- One `ExperimentSet` is created.
- Each file path is loaded into a `Run` via helpers in `data_structure.py`.
- An initial `ViewState` is created and attached to a first tab.
- The first one or two runs are attached to this view and shown in the 2×3 plot layout.

---

## 2. Modules and responsibilities

### 2.1 wx_gui.py (GUI / controller)

- Owns the wxPython application and main event loop.
- Builds and manages the VESTA-like window layout.
- Acts as the controller between GUI widgets, the data model, and analysis helpers.
- Handles file open/export actions, experiment and view management, and logging.

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

- File loading and inference (1D vs 2D).
- Filename-based metadata parsing.
- 2D construction from compatible 1D runs.
- Planned extensions: cosmic-ray handling, baseline correction, curve fitting.

---

## 3. GUI layout and interaction summary

- Left pane:
  - Files, Experiment, Metadata, Log tabs.
  - Preview and Curve Fit panels.
- Right pane:
  - Notebook of views.
  - Each view shows up to two runs in a 2×3 layout:
    - A/A2: 2D maps.
    - B/B2: vertical (angle) slices.
    - C/C2: horizontal (Raman shift) slices.

### Interaction highlights

- Clicking on a 2D map selects a Raman shift and angle.
- Both runs update synchronously (crosshair + slices).
- Highlight modes control whether crosshairs are shown.
- Zoom and Home are synchronized so slices remain consistent.

---

## 4. Design notes

- Clear separation of concerns:
  - Model and IO in `data_structure.py`.
  - Numerical analysis in `analysis.py`.
  - Rendering and interaction in `wx_gui.py`.
- Run nicknames are first-class, user-visible labels and propagate across the UI.
- The design favors correctness and maintainability over visual complexity.

## 5. Planned TODO / Roadmap
1. Editable Metadata tab
2. View axis controls
3. Interactive curve fitting
4. Run type specialization
5. Plotting for 1D / Calculated Runs
6. Import run from menu
7. Merge 1D runs into 2D run