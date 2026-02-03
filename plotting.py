
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.figure import Figure
from matplotlib.axes import Axes
from matplotlib.collections import QuadMesh
from typing import Optional, Tuple, Callable, List, Dict, Any, Union

# Constants needed for secondary axis conversions
EV_PER_CM1 = 1.0 / 8065.544


def centers_to_edges(x: np.ndarray) -> np.ndarray:
    """
    Convert 1D center coordinates to bin edges.
    Supports non-uniform grids.
    """
    x = np.asarray(x, dtype=float)
    if x.ndim != 1 or x.size < 2:
        if x.size == 1:
            return np.array([x[0] - 0.5, x[0] + 0.5], dtype=float)
        return np.array([0.0, 1.0], dtype=float)
    dx = np.diff(x)
    edges = np.empty(x.size + 1, dtype=float)
    edges[1:-1] = x[:-1] + 0.5 * dx
    edges[0] = x[0] - 0.5 * dx[0]
    edges[-1] = x[-1] + 0.5 * dx[-1]
    return edges


class BasePlotter:
    """
    Base class for plotters handling common tasks like clearing axes
    and managing simple artists.
    """
    def __init__(self, ax: Axes):
        self.ax = ax
        self.canvas = ax.figure.canvas if ax and ax.figure else None

    def clear(self):
        """Clear the axes and reset internal state."""
        self.ax.clear()

    def draw(self):
        """Trigger a canvas redraw."""
        if self.canvas:
            self.canvas.draw_idle()

    def set_title(self, title: str):
        self.ax.set_title(title)

    def set_labels(self, xlabel: Optional[str] = None, ylabel: Optional[str] = None):
        if xlabel:
            self.ax.set_xlabel(xlabel)
        if ylabel:
            self.ax.set_ylabel(ylabel)


class RamanPlotter2d(BasePlotter):
    """
    Plot A: 2D map (e.g. Angle vs Raman Shift).
    
    Features:
    - pcolormesh rendering with non-uniform grid support.
    - Contrast adjustment (vmin/vmax perceniles).
    - Secondary x-axis (eV).
    - Highlight crosshair (vertical/horizontal lines).
    - Click-to-select logic abstraction.
    """
    def __init__(self, ax: Axes):
        super().__init__(ax)
        self.mesh: Optional[QuadMesh] = None
        self.cbar = None
        self.secax = None
        
        # Data Cache
        self._x_centers = None
        self._y_centers = None
        self._x_edges = None
        self._y_edges = None
        self._data = None # 2D array
        
        # Highlight Artists
        self._vline = None
        self._hline = None
        
        # Config
        self._highlight_mode = "click" # "none", "click"
        self._x_unit_conversion = None # (func_forward, func_inverse) for secondary axis

    def render(
        self, 
        x: np.ndarray, 
        y: np.ndarray, 
        data: np.ndarray, 
        title: str = "",
        xlabel: str = "Raman shift (cm$^{-1}$)",
        ylabel: str = "Angle (deg)",
        cmap: str = "OrRd",
        x_unit_conversion: Optional[Tuple[Callable, Callable]] = None
    ):
        self.clear()
        
        # 1. Prepare Data
        self._x_centers = np.asarray(x, dtype=float)
        self._y_centers = np.asarray(y, dtype=float)
        self._data = np.asarray(data, dtype=float)
        
        # Handle shape mismatch (transpose if needed)
        ny, nx = self._data.shape
        if nx != self._x_centers.size or ny != self._y_centers.size:
            # Try transposing
            if nx == self._y_centers.size and ny == self._x_centers.size:
                self._data = self._data.T
            else:
                # Fallback or error
                self.ax.text(0.5, 0.5, f"Shape Mismatch: Data {self._data.shape} vs Axes", 
                             ha='center', va='center', transform=self.ax.transAxes)
                return

        # 2. Grid Edges
        self._x_edges = centers_to_edges(self._x_centers)
        self._y_edges = centers_to_edges(self._y_centers)

        # 3. Plot
        self.mesh = self.ax.pcolormesh(
            self._x_edges,
            self._y_edges,
            self._data,
            shading="auto",
            cmap=cmap
        )
        
        self.ax.set_xlim(float(self._x_edges[0]), float(self._x_edges[-1]))
        self.ax.set_ylim(float(self._y_edges[0]), float(self._y_edges[-1]))
        
        # 4. Labels & Title
        self.set_title(title)
        self.set_labels(xlabel, ylabel)
        
        # 5. Colorbar
        if self.cbar:
            try: self.cbar.remove()
            except: pass
        self.cbar = self.ax.figure.colorbar(self.mesh, ax=self.ax, label="Intensity")

        # 6. Secondary Axis
        if x_unit_conversion:
            self._x_unit_conversion = x_unit_conversion
            self.secax = self.ax.secondary_xaxis("top", functions=x_unit_conversion)
            self.secax.set_xlabel("Energy shift (meV)") # assumption, can be parameterized

        # Reset artists
        self._vline = None
        self._hline = None

        self.draw()

    def set_cmap(self, cmap_name: str):
        """Update the colormap of the 2D mesh."""
        if self.mesh:
            self.mesh.set_cmap(cmap_name)
            self.draw()

    def set_contrast(self, vmin_p: float, vmax_p: float):
        """Set contrast based on percentiles [0, 100]."""
        if self.mesh is None or self._data is None:
            return
        
        # Flatten and remove NaNs/Infs for percentile calc
        valid_data = self._data[np.isfinite(self._data)]
        if valid_data.size == 0:
            return

        vmin_val = np.percentile(valid_data, vmin_p)
        vmax_val = np.percentile(valid_data, vmax_p)
        
        if vmax_val <= vmin_val:
            vmax_val = vmin_val + 1e-9
            
        self.mesh.set_clim(vmin_val, vmax_val)
        self.draw()

    def set_highlight(self, x_val: float, y_val: float, visible: bool = True):
        """Draw or update the crosshair at data coordinates (x_val, y_val)."""
        if self._highlight_mode == "none":
            visible = False

        if not visible:
            if self._vline: self._vline.set_visible(False)
            if self._hline: self._hline.set_visible(False)
            self.draw()
            return

        # Create artists if missing
        if self._vline is None:
            self._vline = self.ax.axvline(x_val, color="red", alpha=0.3, linewidth=1.0)
        else:
            self._vline.set_xdata([x_val, x_val])
            self._vline.set_visible(True)

        if self._hline is None:
            self._hline = self.ax.axhline(y_val, color="red", alpha=0.3, linewidth=1.0)
        else:
            self._hline.set_ydata([y_val, y_val])
            self._hline.set_visible(True)
            
        self.draw()

    def set_highlight_mode(self, mode: str):
        self._highlight_mode = mode
        if mode == "none":
            self.set_highlight(0, 0, visible=False)

    def get_index_at(self, x: float, y: float) -> Optional[Tuple[int, int]]:
        """
        Convert click coordinates to array indices.
        Returns (col_index, row_index).
        """
        if self._x_edges is None or self._y_edges is None:
            return None
        
        # searchsorted returns insertion index. 
        # For edges e[i] <= x < e[i+1], searchsorted gives i+1.
        ix = int(np.searchsorted(self._x_edges, x, side="right") - 1)
        iy = int(np.searchsorted(self._y_edges, y, side="right") - 1)
        
        # Clip to valid range
        nx = self._x_centers.size
        ny = self._y_centers.size
        ix = max(0, min(ix, nx - 1))
        iy = max(0, min(iy, ny - 1))
        
        return ix, iy

    def get_coords_from_index(self, ix: int, iy: int) -> Tuple[float, float]:
        """Return center coordinates for given indices."""
        if self._x_centers is None or self._y_centers is None:
            return 0.0, 0.0
        return self._x_centers[ix], self._y_centers[iy]


class AngularPlotter(BasePlotter):
    """
    Plot B: Angular Slice (Intensity vs Angle).
    
    Features:
    - Switchable Cartesian vs Polar projection.
    - Highlight selected angle.
    - Synced x-axis (in Cartesian) with Plot A's y-axis.
    """
    def __init__(self, ax: Axes):
        super().__init__(ax)
        self._mode = "cartesian" # or "polar"
        self._line = None
        self._vline = None # Highlight for Cartesian
        self._vline_polar = None # Highlight for Polar

    def render(self, angles: np.ndarray, intensity: np.ndarray, mode: str = "cartesian", title: str = ""):
        # Note: Switching projections usually requires clearing/recreating the Axes in Matplotlib.
        # This class assumes 'ax' has the correct projection already, or handles rendering logic
        # that fits the current ax projection. The GUI layer is responsible for recreating ax if 
        # projection changes from rectilinear to polar.
        
        self.clear()
        self._mode = mode
        
        if self._mode == "polar":
            self._render_polar(angles, intensity, title)
        else:
            self._render_cartesian(angles, intensity, title)
        
        self.draw()

    def _render_cartesian(self, angles, intensity, title):
        self._line, = self.ax.plot(angles, intensity, "-k", linewidth=1.2)
        self.set_title(title)
        self.set_labels("Angle (deg)", "Intensity (a.u.)")
        self._vline = self.ax.axvline(0, color="red", alpha=0.3, linewidth=1.0, visible=False)

    def _render_polar(self, angles, intensity, title):
        theta = np.deg2rad(angles)
        self.ax.set_theta_zero_location("N")
        self.ax.set_theta_direction(-1) # Clockwise
        
        self.ax.plot(theta, intensity, "o", ms=3, linestyle="None")
        
        # Adaptive R-limits
        valid = intensity[np.isfinite(intensity)]
        if valid.size > 0:
            rmax = valid.max()
            if rmax > 0:
                self.ax.set_rlim(0, rmax * 1.05)
                
        self.set_title(title)
        # Create persistent highlight line for polar
        self._vline_polar = self.ax.axvline(0, color="red", alpha=0.3, linewidth=1.0, visible=False)
        self._adaptive_ticks()

    def _adaptive_ticks(self):
        # Heuristic for cleaner polar grid
        bbox = self.ax.get_window_extent()
        if bbox.height > 10:
            target = max(2, min(4, int(bbox.height / 120)))
            self.ax.yaxis.set_major_locator(mticker.MaxNLocator(nbins=target))

    def set_highlight(self, angle_deg: float, visible: bool = True):
        if not visible:
            if self._vline: self._vline.set_visible(False)
            if self._vline_polar: self._vline_polar.set_visible(False)
            self.draw()
            return

        if self._mode == "polar":
            angle_rad = np.deg2rad(angle_deg)
            if self._vline_polar:
                self._vline_polar.set_xdata([angle_rad, angle_rad])
                self._vline_polar.set_visible(True)
        else:
            if self._vline:
                self._vline.set_xdata([angle_deg, angle_deg])
                self._vline.set_visible(True)
        
        self.draw()


class SlicePlotter(BasePlotter):
    """
    Plot C: Spectral Slice (Intensity vs Shift/Wavelength).
    
    Features:
    - Cartesian line plot.
    - Multiple traces (e.g. Raw + Processed).
    - Highlight selected x-value.
    - Secondary x-axis (eV).
    """
    def __init__(self, ax: Axes):
        super().__init__(ax)
        self._traces: Dict[str, Any] = {} # label -> line artist
        self._vline = None
        self._secax = None

    def render(
        self, 
        x: np.ndarray, 
        intensity: np.ndarray, 
        title: str = "",
        xlabel: str = "Raman shift (cm$^{-1}$)",
        ylabel: str = "Intensity (a.u.)",
        x_unit_conversion: Optional[Tuple[Callable, Callable]] = None
    ):
        self.clear()
        self._traces = {} # Reset traces
        
        # Main trace
        line, = self.ax.plot(x, intensity, "-k", linewidth=1.2, label="Signal")
        self._traces["main"] = line
        
        self.set_title(title)
        self.set_labels(xlabel, ylabel)
        
        self._vline = self.ax.axvline(0, color="red", alpha=0.3, linewidth=1.0, visible=False)

        if x_unit_conversion:
            self._secax = self.ax.secondary_xaxis("top", functions=x_unit_conversion)
            self._secax.set_xlabel("Energy shift (meV)")

        self.draw()

    def add_trace(self, x: np.ndarray, y: np.ndarray, color="blue", style="-", alpha=1.0, label=None):
        """Add an additional trace (e.g. raw data, fit)."""
        line, = self.ax.plot(x, y, color=color, linestyle=style, alpha=alpha, label=label)
        key = label if label else f"trace_{len(self._traces)}"
        self._traces[key] = line
        self.draw()

    def set_highlight(self, x_val: float, visible: bool = True):
        if not self._vline:
            return
            
        if visible:
            self._vline.set_xdata([x_val, x_val])
            self._vline.set_visible(True)
        else:
            self._vline.set_visible(False)
        self.draw()


