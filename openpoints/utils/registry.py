"""Single-scope subset of the OpenPoints (MMCV-derived) registry.

Removed child registries, deprecated APIs and introspection helpers.
"""
import copy
import inspect


class Registry:
    def __init__(self, name):
        self.name = name
        self._module_dict = {}

    def get(self, key):
        return self._module_dict.get(key)

    def build(self, cfg, **kwargs):
        return build_from_cfg(cfg, self, **kwargs)

    def register_module(self, name=None):
        def register(cls):
            if not inspect.isclass(cls):
                raise TypeError('Only classes may be registered')
            key = name or cls.__name__
            if key in self._module_dict:
                raise KeyError('Duplicate registry entry: '+key)
            self._module_dict[key] = cls
            return cls
        return register


def build_from_cfg(cfg, registry, default_args=None):
    """Build a module from config dict.
    Args:
        cfg (edict): Config dict. It should at least contain the key "NAME".
        registry (:obj:`Registry`): The registry to search the type from.
    Returns:
        object: The constructed object.
    """
    if not isinstance(cfg, dict):
        raise TypeError(f'cfg must be a dict, but got {type(cfg)}')
    if not isinstance(registry, Registry):
        raise TypeError('registry must be a Registry object, '
                        f'but got {type(registry)}')

    if not (isinstance(default_args, dict) or default_args is None):
        raise TypeError('default_args must be a dict or None, '
                        f'but got {type(default_args)}')

    obj_cfg = copy.deepcopy(cfg)
    if default_args is not None:
        obj_cfg.update(default_args)
    if 'NAME' not in obj_cfg:
        raise KeyError('Configuration requires NAME')
    obj_type = obj_cfg.pop('NAME')

    if isinstance(obj_type, str):
        obj_cls = registry.get(obj_type)
        if obj_cls is None:
            raise KeyError(
                f'{obj_type} is not in the {registry.name} registry')
    elif inspect.isclass(obj_type):
        obj_cls = obj_type
    else:
        raise TypeError(
            f'type must be a str or valid type, but got {type(obj_type)}')
    try:
        return obj_cls(**obj_cfg)
    except Exception as e:
        # Normal TypeError does not print class name.
        raise type(e)(f'{obj_cls.__name__}: {e}')
