import json
import os
from typing import Any, Dict

DEFAULT_SETTINGS = {
    "default_spectral_unit": "meV",
    "default_colormap": "OrRd",
    "default_vmin_percent": 0.0,
    "default_vmax_percent": 100.0,
    "default_angle_slice_type": "polar",
    "default_slice_x_binning": 1,
    "default_slice_y_binning": 1,
    "default_slice_binning_mode": "cross",
    "show_secondary_unit_axis": True,
    "highlight_requires_modifier": False,
    "default_unknown_1d_spectral_unit": "meV",
    "default_unknown_2d_spectral_unit": "meV",
    # Legacy keys kept as fallbacks for older settings files.
    "colormap": "OrRd",
    "unit": "meV",
    "dark_value": 600.0,
    "cosmic_threshold": 1200.0,
    "cosmic_ratio": 20.0,
    "window_size": [1400, 800],
    "last_directory": ""
}

class ConfigManager:
    def __init__(self, config_path: str = "settings.json"):
        self.config_path = config_path
        self.settings = DEFAULT_SETTINGS.copy()
        self.load()

    def load(self):
        if os.path.exists(self.config_path):
            try:
                with open(self.config_path, "r") as f:
                    loaded = json.load(f)
                    # Update defaults with loaded values to ensure all keys exist
                    self.settings.update(loaded)
                    self._migrate_legacy_keys(loaded)
            except Exception as e:
                print(f"Failed to load settings: {e}")

    def _migrate_legacy_keys(self, loaded: Dict[str, Any] | None = None):
        loaded = loaded or {}
        if "default_spectral_unit" not in loaded and "unit" in loaded:
            self.settings["default_spectral_unit"] = self.settings["unit"]
        if "default_colormap" not in loaded and "colormap" in loaded:
            self.settings["default_colormap"] = self.settings["colormap"]
        self.settings["unit"] = self.settings.get("default_spectral_unit", self.settings.get("unit", "meV"))
        self.settings["colormap"] = self.settings.get("default_colormap", self.settings.get("colormap", "OrRd"))

    def save(self):
        try:
            with open(self.config_path, "w") as f:
                json.dump(self.settings, f, indent=4)
        except Exception as e:
            print(f"Failed to save settings: {e}")

    def get(self, key: str, default: Any = None) -> Any:
        return self.settings.get(key, default)

    def set(self, key: str, value: Any):
        self.settings[key] = value
        if key == "default_spectral_unit":
            self.settings["unit"] = value
        elif key == "unit":
            self.settings["default_spectral_unit"] = value
        elif key == "default_colormap":
            self.settings["colormap"] = value
        elif key == "colormap":
            self.settings["default_colormap"] = value
        # Auto-save on set might be too frequent, but for now it's simple
        self.save()

# Global instance
config = ConfigManager()
