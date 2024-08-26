from pathlib import Path
import yaml


class PipelineConfig:
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
        keys = self._validate_keys(keys)
        d = self.config
        for key in keys:
            d = d[key]
        return d

    def __setitem__(self, keys, value):
        keys = self._validate_keys(keys)
        d = self.config
        for key in keys[:-1]:
            d = d.setdefault(key, {})
        d[keys[-1]] = value

    def __delitem__(self, keys):
        keys = self._validate_keys(keys)
        d = self.config
        for key in keys[:-1]:
            d = d[key]
        del d[keys[-1]]

    def __str__(self):
        return yaml.dump(self.config, default_flow_style=False, sort_keys=False)

    def _validate_keys(self, keys):
        if isinstance(keys, str):
            return [keys]
        return keys

    def save(self, path=None):
        if path is None and self.yaml_path is None:
            raise ValueError("No path specified")
        if path is None:
            path = self.yaml_path
        with open(path, 'w') as f:
            yaml.dump(self.config, f, default_flow_style=False, sort_keys=False)
