"""EasyConfig adapted from PointDGMamba/OpenPoints; YAML and CLI overrides only."""
from ast import literal_eval
from pathlib import Path
import yaml


class EasyConfig(dict):
    def __getattr__(self, key):
        try:
            return self[key]
        except KeyError as exc:
            raise AttributeError(key) from exc

    def update(self, other):
        if isinstance(other, (list, tuple)):
            index = 0
            while index < len(other):
                opt = other[index].removeprefix('--')
                if '=' in opt:
                    key, value = opt.split('=', 1)
                    index += 1
                else:
                    if index+1 >= len(other):
                        raise ValueError('Override requires a value: '+opt)
                    key, value = opt, other[index+1]
                    index += 2
                try:
                    value = literal_eval(value)
                except (ValueError, SyntaxError):
                    pass
                current = self
                parts = key.split('.')
                for part in parts[:-1]:
                    current = current.setdefault(part, EasyConfig())
                current.update({parts[-1]: value})
        else:
            for key, value in other.items():
                if isinstance(value, dict):
                    if key not in self or not isinstance(self[key], EasyConfig):
                        self[key] = EasyConfig()
                    self[key].update(value)
                else:
                    self[key] = value

    def load(self, fpath, *, recursive=False):
        path = Path(fpath).resolve()
        if not path.is_file():
            raise FileNotFoundError(path)
        paths = [path]
        if recursive:
            paths = [parent/('default'+path.suffix) for parent in reversed(path.parents)] + paths
        for candidate in paths:
            if candidate.is_file():
                values = yaml.safe_load(candidate.read_text(encoding='utf-8'))
                if not isinstance(values, dict):
                    raise ValueError('Configuration must be a mapping: '+str(candidate))
                self.update(values)

    def dict(self):
        return {key: value.dict() if isinstance(value, EasyConfig) else value
                for key, value in self.items()}
