from pathlib import Path
import yaml


class PipelineConfig:
    """Thin YAML-backed configuration wrapper for the pipeline and modules."""
    def __init__(self, yaml_path=None):
        if yaml_path is not None:
            self.yaml_path = yaml_path
            if Path(yaml_path).exists():
                with open(yaml_path) as f:
                    self.config = yaml.load(f, Loader=yaml.FullLoader)
                    pass
        else:
            self.config = {}

    def __getitem__(self, keys):
        """Read a nested key or top-level key as list or string."""
        keys = self._validate_keys(keys)
        d = self.config
        for key in keys:
            d = d[key]
        return d

    def __setitem__(self, keys, value):
        """Set a nested key, creating intermediate dicts when necessary."""
        keys = self._validate_keys(keys)
        d = self.config
        for key in keys[:-1]:
            d = d.setdefault(key, {})
        d[keys[-1]] = value

    def __delitem__(self, keys):
        """Delete a nested key."""
        keys = self._validate_keys(keys)
        d = self.config
        for key in keys[:-1]:
            d = d[key]
        del d[keys[-1]]

    def __str__(self):
        """Dump configuration to a YAML string."""
        return yaml.dump(self.config, default_flow_style=False, sort_keys=False)

    def _validate_keys(self, keys):
        """Ensure keys are a list; accept single string for convenience."""
        if isinstance(keys, str):
            return [keys]
        return keys

    def save(self, path=None):
        """Save configuration to YAML file (default: original yaml_path)."""
        if path is None and self.yaml_path is None:
            raise ValueError("No path specified")
        if path is None:
            path = self.yaml_path
        with open(path, 'w') as f:
            yaml.dump(self.config, f, default_flow_style=False, sort_keys=False)
