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
- `qe_raman.py`, `qe_raman_gui.py` – wx-free Quantum ESPRESSO `ph.x` parsing/spectrum generation and its desktop import dialog.
- `plot_export_cli.py` – headless plot export from saved experiment files.
- `agent_cli.py` – agent-friendly command line wrapper for listing runs, exporting maps/slices, exporting data, and creating temporary experiments.
- `fit_overlay.py`, `polar_area_fitting.py` – fit reconstruction overlays and polar peak-area fitting exports.
- `config_manager.py`, `settings.example.json` – application defaults, example custom colormaps, and local preference persistence.

The intent is that each file remains small and well-scoped so that another person (or an AI assistant)
can safely modify one layer without breaking the others.

---

## 1. Running the application

Venkata is tested with Python 3.12. From the `Venkata` directory, install the
runtime dependencies and launch the application:

```bash
python -m pip install -r requirements.txt
python wx_gui.py
```

Optional Seaborn and Crameri colormaps are listed separately:

```bash
python -m pip install -r requirements-optional.txt
```

The application runs with defaults when `settings.json` is absent and creates
that local file as preferences are changed. To start from the included custom
colormap examples, copy `settings.example.json` to `settings.json`. Runtime
`settings.json` is intentionally ignored because it contains machine-specific
paths and window state. The command-line exporters use the same project modules,
although they render with Matplotlib's headless Agg backend.

While you can still provide initial files as command-line arguments, the primary way to load data is through the `File` menu.

- **`File -> Import Run(s)...`**: Opens a dialog to select one or more data files to import as individual runs.
- **`File -> Import Quantum ESPRESSO Raman...`**: Reads a `ph.x` output and creates paired frequency-only `<name>_DFT_xx` / `<name>_DFT_yx` runs, with optional relative heights and widths from a saved Fit Parameters run.
- **`File -> Merge Run...`**: Starts the process of combining multiple 1D spectra into a 2D run.
- **`File -> Export to Igor Pro...`**: Writes all numeric runs in the current experiment to Igor Text (`.itx`) without requiring `igor2`.

Saved experiments can also be exported from the command line:

```bash
python wx_gui.py export-plot RamanData.dat --list
python wx_gui.py export-plot RamanData.dat --view "View 3" --panel full --output view3.pdf
python wx_gui.py export-plot RamanData.dat --run CPS_bulk_xx --panel A --unit meV --xlim 0.62 50 --output cps_bulk_xx_map.pdf
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
- **Cosmic Ray Correction**: Detects and tracks candidate single-row spikes during merge workflows.
- **Curve and Map Fitting**: Supplies Lorentzian helpers and map-fitting engines used by 1D, row-by-row, and overlay workflows.
- Planned extensions: baseline correction and broader peak-finding/model support.

### 2.4 Command-line modules

- `plot_export_cli.py` exports saved views, individual panels, single runs, and polar peak-area tensor-fit PDFs without launching the GUI.
- `agent_cli.py` exposes a smaller, search-oriented interface for AI agents and scripts. It resolves natural-language-ish run queries against run ids, nicknames, source paths, and metadata.
- Both CLI paths load saved `ExperimentSet` files such as `.dat`, `.h5`, or `.hdf5`, register custom colormaps from `settings.json`, and write normal matplotlib outputs selected by the output extension.

---

## 3. GUI and Feature Summary

### Main Window
- **Left Pane**: Contains notebooks for file browsing, experiment management (listing runs and views), metadata display, and logging.
- **Right Pane**: Contains one or more "View" tabs for data visualization.
- **Plots**: Each view shows up to two runs in a 2x3 layout: 2D map (Plot A), angular slice (Plot B), and spectral slice (Plot C).

### Key Features and Interactions

- **Data Import**:
  - Load individual runs via `File -> Import Run(s)...`.
  - Load Quantum ESPRESSO phonon results via `File -> Import Quantum ESPRESSO Raman...`. The editable mapping table can pair positive Raman-active QE modes with fitted Venkata peaks, then previews and creates native XX/YX 1D runs.
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
  - `python wx_gui.py export-plot ... --polar-area-fit --target-peaks ...` exports vector PDFs of polar peak-area tensor fits from selected 2D runs or fit-parameter runs.

- **2D Map Fitting and Fit Review**:
  - Use **Tools -> Make 2D Fit...** or right-click a 2D map to open the 2D fitting popup.
  - The manually selectable `D6h_E2g(anisotropy)` tensor rule fits `a`, `b`, and an in-plane orientation `phi`. It incoherently sums `diag(a,b)` with an off-diagonal E2g partner of strength `(a-b)/2`, giving `I_XX = a² cos²(theta-phi) + b² sin²(theta-phi)` and the constant `I_YX = (a-b)²/4`. The ordinary `D6h_E2g` response is recovered when `b = -a`.
  - Linear background `Slope_Theta` is evaluated against each row's original pre-rotation acquisition angle, so its zero remains at the raw 0° measurement after cyclic or unwrapped dataset rotation. Raman tensor rules continue to use the displayed/rotated angle; `Slope_X` continues to use the spectral coordinate.
  - **Validate (RowFit)** first opens a setup table for per-polarization starting values, bounds, and fixed parameters. Enter `global` to inherit the frozen global-fit value for each row, or enter a numeric override. Peak `Gamma` bounds initially span 0.5x to 1.5x the corresponding frozen global-fit value and remain editable.
  - The row-fit inspector mirrors the main 2D fitting layout: controls and editable row parameters are on the left, while each polarization's Data and complete Row-Fit Reconstruction maps are shown on the right. The selected spectral and angular slices overlay Data, frozen Global Fit, and Row Fit. The parameter table reports row/global ratios for peak areas and `Gamma`; a `Gamma` row is highlighted yellow when the row fit reaches either configured bound. Failed rows, RMSE, and parameters at bounds are shown beside the row table; settings can be changed and all rows or only the selected row can be fit again.
  - Peak areas greater than twice the frozen global-fit area at the same angle are marked in the row status and highlighted in the Row/Global/Ratio table; **Next Anomaly** navigates through them without drawing stripes over the reconstruction. **Polar Area Fit...** opens a linked, peak-selectable polar popup showing row areas, a tensor fit to those areas, the frozen global tensor curve, and starred anomaly points. Clicking any row-area marker moves the inspection cursor to that marker's polarization, angular row, and peak center while keeping the polar plot open.
  - **Tools -> Generate Polar Area Fittings...** exports those saved inspection areas and `Gamma` values to a vector PDF and companion CSV when **Use saved inspection row-fit cache** is selected (the default). Choose `cm-1` or `meV` for target entry, PDF peak labels, and CSV frequency/`Gamma` columns; when runs are taken from the current view, its spectral unit becomes the default. Both polarization fits use solid lines and remain distinguishable by color and marker. After editing or refitting rows, choose **Save Caches** in the inspector to stage the compact row table, then commit the main fitting dialog. Clear the cache option only when an independent raw-map row refit is intentionally desired.
  - Validation works from a read-only snapshot of the 2D popup's global parameters. Row fitting and manual row edits never replace the global-fit solution.
  - The lower-left **Fit Results** tab is read-only and summarizes stored fit parameters and compact polar area visuals.
  - Use **Tools -> Rotate 360° Dataset...** to store non-destructive angular rotation/window settings for 2D maps. Rotation preserves every selected measurement row. Angular windows are inclusive, so a 0°-360° window retains both endpoint measurements; only rows deliberately outside the selected window are excluded.
  - Rotation keeps saved FitParams tensor angles synchronized by applying only the change in cyclic coordinate offset. Reapplying the same rotation is idempotent, disabling rotation restores the previous `phi`, and non-cyclic/unwrapped rotation leaves `phi` unchanged. Every run belonging to a common XX/YX fit must be rotated together with the same offset change; otherwise Venkata warns and leaves that common FitParams state unchanged. Rotation invalidates old compact row results, so run **Validate -> Save Caches** and commit again before cache-backed exports.
  - **FIT Global** now performs only the global optimization by default. Enable **Also fit all rows after global fit (slow)** only when an immediate row-cache rebuild is wanted; otherwise use **Validate (RowFit)** when you are ready to inspect or rebuild row fits.
  - New row-cache setup defaults keep every background parameter and every peak `Gamma` fixed to its inherited global-fit value while leaving peak areas free. The setup popup provides **Fix all backgrounds**, **Fix all areas**, **Fix all Gammas**, and **Fix all parameters** controls that apply to every active polarization dataset at once. Explicit choices stored in an older FitParams state remain respected.
  - The fitting dialog is transactional. **OK** updates the loaded FitParams in place, **Save as new params** creates a separate FitParams and closes, and **Cancel** discards fits, validation changes, and staged exports. Automatic global/row reconstruction matrices are not saved: FitParams stores only global state plus compact per-row parameters and statuses, and maps are rebuilt in memory. Committing also migrates exact legacy row tables and removes automatic legacy matrix-cache runs; explicitly exported maps are retained.
  - `python wx_gui.py agent validate-rowfit INPUT.dat --params-query QUERY ...` is the headless equivalent of **Validate (RowFit)**. Use `--all-fit-params` for a batch, `--fix-background`, `--fix-gamma`, `--free-area`, or the corresponding inverse switches for group settings, and `--row-config rules.json` for per-FitParams/dataset/parameter initial values and bounds. `--output UPDATED.dat` saves the resolved configuration and compact row results copy-on-write, without hidden reconstruction matrices; omit it for report-only operation. `--csv-dir`, `--report-dir`, and `--polar-dir` export status-rich row tables, diagnostic PDFs, and paginated vector polar-area PDFs from the same validation result. Batch processing saves successful fits and reports skipped failures by default; `--strict` suppresses the `.dat` output when any selected FitParams fails.

Example row-rule file (later matching rules override earlier ones):

```json
{
  "rules": [
    {"fitparams": "FitParams_RPS-*", "dataset": "*", "parameter": "*_Gamma", "fixed": true},
    {"fitparams": "FitParams_RPS-bi*", "dataset": "cross", "parameter": "P2_*_Area", "initial": "global", "min": 0, "max": 5000, "fixed": false}
  ]
}
```

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

### Headless and agent CLI examples

List saved views and runs:

```bash
python wx_gui.py agent list RamanData.dat
```

Export publication-style maps, slices, or full views:

```bash
python wx_gui.py agent plot RamanData.dat --kind map --run-query CPS_bulk_xx --unit meV --xlim 0.62 50 --output CPS_bulk_xx_2dmap.pdf
python wx_gui.py agent plot RamanData.dat --kind slice-b --run-query "CPS bulk xx" --unit meV --x-value 35 --angle-slice polar --output CPS_bulk_xx_35meV_polar.pdf
python wx_gui.py agent plot RamanData.dat --kind map --run-query RPS_mono_xx --data-only --output RPS_mono_bitmap.png
python wx_gui.py export-plot RamanData.dat --view "View 3" --panel full --output view3.pdf
```

Export selected data:

```bash
python wx_gui.py agent export-data RamanData.dat --run-query CPS_bulk_xx --run-query CPS_bulk_yx --csv-dir out --igor out/CPS_bulk.itx
```

Create a temporary saved experiment from raw files:

```bash
python wx_gui.py agent create-experiment --input run_xx.csv --input run_yx.csv --nickname CPS_bulk_xx --nickname CPS_bulk_yx --unit meV --default-cmap Standard-bone --output imported.dat
```

Import Quantum ESPRESSO `ph.x` frequencies as paired XX/YX runs:

```bash
python wx_gui.py agent import-qe-raman ph.out --nickname SAMPLE --angle 0 --fwhm 4 --match-tolerance 15 --output qe_raman.dat
python wx_gui.py agent import-qe-raman ph.out --nickname SAMPLE --append experiment.dat --fit-params-query "sample fit parameters" --output updated.dat
```

This importer uses QE frequencies for peak centers. Without Fit Parameters it assigns equal unit peak heights and the default FWHM to both polarizations. With Fit Parameters it evaluates Venkata's saved tensor rules at the requested in-plane rotation and uses their relative XX/YX heights and fitted linewidths. Neither path is an exact ab-initio QE polarization calculation; that requires companion dynamical-matrix post-processing (for example `fildyn` with `dynmat.x`). Generated runs keep the QE parser result and reconstruction settings in metadata and must be regenerated through the specialized importer rather than generic **Update from file**.

When a requested spectral range mixes units, choose one output unit and convert the bounds before calling the CLI. For example, `5 cm-1` to `50 meV` can be exported as `--unit meV --xlim 0.61992099 50`.

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
- `agent_cli.py` should stay conservative and scriptable: list first, resolve run names through `--run-query`, and print created output paths.
- Fit overlays are centralized in `fit_overlay.py`; use `--fit-overlay global`, `row`, or `both` only when a run has compatible fit state or reconstruction data.
- Polar peak-area fitting is centralized in `polar_area_fitting.py` and is exposed through `export-plot --polar-area-fit`.
- Igor export writes plain `.itx` text directly. Keep this dependency-free unless there is a strong reason to add and vendor an Igor writer.
- For GUI bugs, first inspect `wx_gui.py` for event routing, then the relevant left/right panel file for widget state, then `plotting.py` for matplotlib behavior.

## 6. Development Roadmap

For a detailed list of planned features, bug fixes, and completed work, please see `TODO.md`.
