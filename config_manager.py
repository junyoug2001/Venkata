import json
import os
from typing import Any, Dict

DEFAULT_SETTINGS = {
    "colormap": "OrRd",
    "unit": "cm-1",
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
            except Exception as e:
                print(f"Failed to load settings: {e}")

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
        # Auto-save on set might be too frequent, but for now it's simple
        self.save()

# Global instance
config = ConfigManager()
