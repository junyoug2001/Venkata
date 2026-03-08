---
name: venkata-raman
description: Use when exporting Raman maps, slices, pure bitmap data images, CSV/Igor data, or temporary ExperimentSet files from the Venkata Raman codebase with the agent CLI.
---

# Venkata Raman Agent Workflow

Use the repo CLI from the Venkata project root:

```bash
python wx_gui.py agent list RamanData.dat
```

Start with `agent list` unless the user gives exact run IDs. Resolve natural language phrases to `--run-query` strings using nickname/source metadata shown by the list command.

## Plot Exports

Use `agent plot`:

```bash
python wx_gui.py agent plot RamanData.dat --kind map --run-query "CPS bulk" --unit cm-1 --xlim -15 600 --output cps_bulk_map.pdf
python wx_gui.py agent plot RamanData.dat --kind slice-b --run-query "CPtS mono" --unit meV --x-value 35 --angle-slice polar --output cpts_mono_35meV.pdf
python wx_gui.py agent plot RamanData.dat --kind map --run-query "RPS mono" --ylim 0 180 --data-only --output rps_mono_bitmap.png
```

Guidelines:

- Prefer PDF for publication/vector plots.
- Use PNG plus `--data-only` for pure bitmap map images with no axes, titles, or colorbar.
- Use `--unit cm-1` when the user gives cm-1 limits; use `--unit meV` for meV positions.
- Use `--fit-overlay global`, `--fit-overlay row`, or `--fit-overlay both` only when the user asks to show fit results.
- For angular polar plots, use `--kind slice-b --angle-slice polar --x-value VALUE`.
- For spectral slices, use `--kind slice-c --y-value ANGLE`.

## Data Exports

Use `agent export-data`:

```bash
python wx_gui.py agent export-data RamanData.dat --run-query "CNS bi xx" --run-query "CNA bi yx" --csv-dir out --igor out/export.itx
```

Notes:

- CSV export uses Venkata's existing dual-unit export behavior for 2D runs.
- Igor export writes Igor Text `.itx`. If a `.pxp` path is requested, report the actual `.itx` output path.

## Create Temporary Experiments

Use `agent create-experiment` for raw CSV imports:

```bash
python wx_gui.py agent create-experiment --input file1.csv --input file2.csv --nickname "Run A" --nickname "Run B" --default-cmap Standard-bone --output imported.dat
```

Report all created output paths and the run IDs/nicknames printed by the command.

