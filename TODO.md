# Venkata Development Roadmap

This document lists proposed enhancements and refactoring tasks for the Venkata Raman‑analysis GUI.

---

## In Progress / To Do

1.  **Import/Distinguish and Plot 1D/Calculated Runs**
    *   **Goal**: Explicitly distinguish between run categories for clarity and improved plotting behavior, and provide dedicated 1D plotting options.
    *   **Partially Complete**: The application now implicitly handles 1D (for merging) and 2D runs (for plotting/export). Merged runs act as a form of "Calculated Run".
    *   **Tasks**:
        *   Add an explicit `run_type` field to the `Run` dataclass (e.g., '1D Spectrum', '2D Map', 'Merged Map').
        *   Adjust the logic in `ViewPanel` to choose the appropriate plotting layout based on run type (e.g., overlay plots for multiple 1D runs).
        *   For 1D runs, implement overlay plots that can stack multiple spectra in the same axes.
        *   Allow toggling individual curves on/off within the overlay via a legend or checkboxes.
2.  **Interactive Curve Fitting on the Curve Fit Tab**
    *   **Goal**: Provide an integrated peak‑fitting workflow.
    *   **Tasks**:
        *   Build a `CurveFitPanel` class that replaces the current placeholder panel.
        *   Display selected 1D spectra and allow the user to click on peaks to set initial positions and widths.
        *   Implement a simple peak‑fit engine (e.g., Gaussian/Lorentzian) that calls an analysis function with user‑supplied initial parameters and updates the plot with fitted curves.
        *   Display fit parameters (positions, widths, amplitudes) and goodness‑of‑fit metrics in a summary area.


---

## Completed

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

