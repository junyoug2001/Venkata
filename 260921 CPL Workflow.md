# 260921 Circular-Polarization Workflow Update

## Scope

This update separates circular-polarization (CPL) merge discovery from ordinary linear-polarization and angular-map filename handling. It also makes opposite-chirality comparison a persistent view feature rather than a manual overlay setup.

## Filename Discovery

- The indexed PfDirk CPL form is matched by `pfdirk_cpl_indexed_txt`, including names such as `26091019_mono_30sx2_1200g_50mu_514.37nm_0,3mW_RR_300K_0002.txt`.
- The first six digits are the measurement date. The final two digits are the acquisition ID and may differ between `RR`, `RL`, `LL`, and `LR` files.
- Sample, integration, grating, slit, laser, power, temperature, extension, and date must still match.
- A dated legacy CPL fallback applies the same acquisition-ID rule to otherwise unrecognized legacy names.
- CPL row order is `RR`, `RL`, `LL`, `LR`; display positions are `0`, `90`, `180`, `270` degrees.

## View Behavior

- Adding one or two complete CPL maps to a view creates or reuses one opposite-chirality virtual map per source.
- Virtual row data are exchanged as `RR` <-> `LL` and `RL` <-> `LR` while the displayed row labels stay aligned with the source.
- The virtual maps target `1C` or `2C`, use a red `0.6 pt` line, and do not occupy primary Plot A/B/C slots.
- Existing source-only views are backfilled when opened in the GUI.
- Virtual runs and their overlay configuration persist in HDF5 files.

## Lifecycle And Safety

- Reattaching or reloading a source synchronizes its existing virtual arrays instead of reusing stale data.
- Removing a source also removes its generated CPL virtual run and stale view configuration.
- New merged runs record the parameters needed to rebuild the complete family with **Update from file**.
- A legacy merge without a reload recipe is not silently collapsed to the seed spectrum.
- A GUI merge with interactively selected cosmic corrections is marked non-reloadable because the exact selection is not currently stored as a deterministic recipe. Re-run **Merge Run...** to rebuild that case.

## Verification

Regression coverage includes cross-acquisition filename discovery, 90-degree CPL coordinates, chirality permutation, overlay slot/style persistence, source-to-virtual synchronization, dependent-run deletion, merged-family reload, and headless `agent merge` output.
