# Venkata Development Roadmap

This document lists proposed enhancements and refactoring tasks for the Venkata Raman‑analysis GUI.

---

## In Progress / To Do

1.  **Import/Distinguish and Plot 1D/Calculated Runs**
    *   **Goal**: Explicitly distinguish between run categories for clarity and improved plotting behavior, and provide dedicated 1D plotting options.
    *   **Partially Complete**: The application now implicitly handles 1D (for merging) and 2D runs (for plotting/export). Merged runs act as a form of "Calculated Run". 1D Curve Fitting is implemented.
    *   **Tasks**:
        *   Add an explicit `run_type` field to the `Run` dataclass (e.g., '1D Spectrum', '2D Map', 'Merged Map'). (Partially done with `RunType` enum).
        *   Adjust the logic in `ViewPanel` to choose the appropriate plotting layout based on run type (e.g., overlay plots for multiple 1D runs).
        *   For 1D runs, implement overlay plots that can stack multiple spectra in the same axes.
        *   Allow toggling individual curves on/off within the overlay via a legend or checkboxes.

2.  **Plot Overlay Feature**
    *   **Goal**: Allow users to overlay slices (B or C) from multiple runs onto a single axis (e.g., Overlay Run 2's Angle Slice onto Plot 1B).
    *   **Status**: **Partially Complete**.
    *   **Implementation**: Curve fitting results (Derived Runs) now automatically overlay on their source plots. The infrastructure for general overlays exists in `ViewPanel`.
    *   **Remaining Tasks**:
        *   Expose a generic UI (e.g. context menu or drag-and-drop) to overlay *any* arbitrary run onto another.
        *   Fix issues with modifying the 'Plot' column in the Appearance panel.
        *   Improve UI responsiveness and visual feedback when an overlay is active.

3.  **Improved Peak Finding**
    *   **Goal**: Automate the initialization of fit parameters.
    *   **Tasks**:
        *   Implement peak detection algorithms (e.g. `scipy.signal.find_peaks`) to automatically guess `x0` and `y0` for multiple peaks.
        *   Integrate this into the `CurveFitPanel` "auto-guess" logic.

4.  **Integrate 2D Map Global Fitting**
    *   **Goal**: Merge the functionality of `2d_map_fitting_gui.py` into the main application.
    *   **Tasks**:
        *   Create a new Panel or Dialog within the main GUI to host the Global Fitting workflow.
        *   Port the `SelectionRules`, `FittingEngine`, and GUI components.
        *   Ensure it can operate on any currently loaded 2D Run in the Experiment.

---

## Completed

-   **Interactive Curve Fitting on the Curve Fit Tab**
    *   **Status**: Done.
    *   **Implementation**:
        *   Implemented `CurveFitPanel` with support for Lorentzian, Sum of Lorentzian, and User-defined models.
        *   Integrated `scipy.optimize.curve_fit` for interactive fitting.
        *   Added "Curve Fit..." context menu to 1D plots.
        *   Implemented creation of "Derived Runs" from fit results, which automatically overlay on the source data.

-   **Import Run via Menu**
    *   **Status**: Done.
    *   **Implementation**: Added `File -> Import Run(s)...` and `File -> Merge Run...` menu items. This replaces the need to load files via command-line arguments.

-   **Merge Multiple 1D Runs into a 2D Run**
    *   **Status**: Done.
    *   **Implementation**: The `File -> Merge Run...` dialog allows a user to select a seed 1D file, which then discovers other mergeable files in the same directory. The dialog provides options for cosmic ray removal and axis unit selection. The merged data is added to the experiment as a new 2D `Run`.

-   **Data Export for 2D Runs**
    *   **Status**: Done.
    *   **Implementation**:
        *   Added "Export Run..." to the context menu in the Experiment panel.
        *   Supports selecting multiple runs via checkboxes for batch export.
        *   Uses a directory dialog for selecting the output location.
        *   Automatically generates filenames based on run metadata (e.g., `..._angular_matrix_cm-1.csv`).
        *   Exports data to two separate CSV files: one with Raman shift in `cm-1` and another in `meV`.