"""wx dialog for previewing and applying Raman-axis recalibration."""

from __future__ import annotations

from typing import Optional

import numpy as np
import wx
from matplotlib.backends.backend_wxagg import FigureCanvasWxAgg as FigureCanvas
from matplotlib.figure import Figure

from data_structure import ExperimentSet
from raman_axis_correction import (
    RamanAxisCorrectionPlan,
    build_raman_axis_correction_plan,
    infer_applied_laser_nm,
    recover_raw_wavelength_axis,
)


class RamanAxisCorrectionDialog(wx.Dialog):
    def __init__(self, parent: wx.Window, experiment: ExperimentSet, run_id: str):
        super().__init__(
            parent,
            title="Redo Raman-shift conversion",
            style=wx.DEFAULT_DIALOG_STYLE | wx.RESIZE_BORDER,
        )
        self.experiment = experiment
        self.run_id = str(run_id)
        self.plan: Optional[RamanAxisCorrectionPlan] = None
        run = experiment.get_run(self.run_id)
        if run is None:
            raise ValueError(f"Run '{run_id}' was not found.")
        wavelength = recover_raw_wavelength_axis(run)
        self.current_laser_nm = infer_applied_laser_nm(run, wavelength)

        self.SetMinSize((720, 560))
        self.SetSize((900, 680))
        root = wx.BoxSizer(wx.VERTICAL)
        self.SetSizer(root)

        heading = wx.StaticText(
            self,
            label=(
                f"Run: {run.nickname}\n"
                "Only the wavelength-to-Raman-shift conversion will change; merged intensities are preserved."
            ),
        )
        root.Add(heading, 0, wx.EXPAND | wx.ALL, 10)

        controls = wx.BoxSizer(wx.HORIZONTAL)
        controls.Add(wx.StaticText(self, label="Current laser:"), 0, wx.ALIGN_CENTER_VERTICAL | wx.RIGHT, 5)
        controls.Add(wx.StaticText(self, label=f"{self.current_laser_nm:.9g} nm"), 0, wx.ALIGN_CENTER_VERTICAL | wx.RIGHT, 20)
        controls.Add(wx.StaticText(self, label="Corrected laser (nm):"), 0, wx.ALIGN_CENTER_VERTICAL | wx.RIGHT, 5)
        self.txt_laser = wx.TextCtrl(self, value=f"{self.current_laser_nm:.9g}", style=wx.TE_PROCESS_ENTER)
        controls.Add(self.txt_laser, 0, wx.ALIGN_CENTER_VERTICAL | wx.RIGHT, 8)
        self.btn_preview = wx.Button(self, label="Preview")
        controls.Add(self.btn_preview, 0, wx.ALIGN_CENTER_VERTICAL)
        root.Add(controls, 0, wx.EXPAND | wx.LEFT | wx.RIGHT | wx.BOTTOM, 10)

        self.lbl_summary = wx.StaticText(self, label="")
        self.lbl_summary.Wrap(850)
        root.Add(self.lbl_summary, 0, wx.EXPAND | wx.LEFT | wx.RIGHT | wx.BOTTOM, 10)

        self.figure = Figure(figsize=(8, 5))
        self.canvas = FigureCanvas(self, -1, self.figure)
        root.Add(self.canvas, 1, wx.EXPAND | wx.LEFT | wx.RIGHT, 10)

        buttons = wx.StdDialogButtonSizer()
        self.btn_apply = wx.Button(self, wx.ID_OK, "Apply conversion")
        self.btn_cancel = wx.Button(self, wx.ID_CANCEL, "Cancel")
        buttons.AddButton(self.btn_apply)
        buttons.AddButton(self.btn_cancel)
        buttons.Realize()
        root.Add(buttons, 0, wx.ALIGN_RIGHT | wx.ALL, 10)

        self.btn_preview.Bind(wx.EVT_BUTTON, self._on_preview)
        self.txt_laser.Bind(wx.EVT_TEXT_ENTER, self._on_preview)
        self.txt_laser.Bind(wx.EVT_TEXT, self._on_laser_changed)
        self.btn_apply.Enable(False)
        self._refresh_preview(show_errors=False)

    def _on_laser_changed(self, event) -> None:
        self.plan = None
        self.btn_apply.Enable(False)
        self.lbl_summary.SetLabel("Enter the corrected laser wavelength, then select Preview.")
        event.Skip()

    def _on_preview(self, event) -> None:
        self._refresh_preview(show_errors=True)

    def _refresh_preview(self, *, show_errors: bool) -> None:
        try:
            laser_nm = float(self.txt_laser.GetValue().strip())
            plan = build_raman_axis_correction_plan(self.experiment, self.run_id, laser_nm)
            correction = plan.correction_for(self.run_id)
        except Exception as exc:
            self.plan = None
            self.btn_apply.Enable(False)
            self.lbl_summary.SetLabel(f"Preview unavailable: {exc}")
            if show_errors:
                wx.MessageBox(str(exc), "Redo Raman-shift conversion", wx.OK | wx.ICON_ERROR, self)
            return

        self.plan = plan
        self.btn_apply.Enable(True)
        old_min, old_max = float(np.nanmin(correction.old_shift_cm1)), float(np.nanmax(correction.old_shift_cm1))
        new_min, new_max = float(np.nanmin(correction.new_shift_cm1)), float(np.nanmax(correction.new_shift_cm1))
        affected_names = ", ".join(item.nickname for item in plan.corrections)
        fit_note = f" Saved fits to translate: {len(plan.fit_owner_ids)}." if plan.fit_owner_ids else ""
        self.lbl_summary.SetLabel(
            f"Axis: {old_min:.6g}…{old_max:.6g} → {new_min:.6g}…{new_max:.6g} cm-1; "
            f"shift {correction.delta_cm1:+.6g} cm-1. "
            f"Affected run(s): {affected_names}.{fit_note}"
        )
        self.lbl_summary.Wrap(max(400, self.GetClientSize().width - 40))
        self._draw_preview(correction.new_shift_cm1)
        self.Layout()

    def _draw_preview(self, corrected_shift: np.ndarray) -> None:
        run = self.experiment.get_run(self.run_id)
        x = np.asarray(corrected_shift, dtype=float)
        y = np.asarray(run.angle_values, dtype=float)
        matrix = np.asarray(run.intensity_2d, dtype=float)
        if matrix.shape != (y.size, x.size):
            if matrix.T.shape == (y.size, x.size):
                matrix = matrix.T
            else:
                raise ValueError("Merged intensity shape no longer matches its axes.")
        order = np.argsort(x)
        x = x[order]
        matrix = matrix[:, order]
        row_index = max(0, min(matrix.shape[0] // 2, matrix.shape[0] - 1))

        self.figure.clear()
        grid = self.figure.add_gridspec(2, 1, height_ratios=(2.4, 1.0), hspace=0.38)
        map_axis = self.figure.add_subplot(grid[0, 0])
        slice_axis = self.figure.add_subplot(grid[1, 0])
        map_axis.pcolormesh(x, y, matrix, shading="auto")
        map_axis.axhline(y[row_index], color="white", linewidth=0.8, alpha=0.8)
        map_axis.set_title("Corrected-axis preview")
        map_axis.set_xlabel("Raman shift (cm-1)")
        map_axis.set_ylabel("Merged row axis")
        slice_axis.plot(x, matrix[row_index], color="black", linewidth=0.9)
        slice_axis.set_title(f"Spectral slice, row {row_index + 1}")
        slice_axis.set_xlabel("Raman shift (cm-1)")
        slice_axis.set_ylabel("Intensity")
        self.figure.subplots_adjust(left=0.10, right=0.97, bottom=0.10, top=0.94)
        self.canvas.draw_idle()

