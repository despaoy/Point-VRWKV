"""OpenPoints NAME/default_args dataset construction; unused collators removed."""
from openpoints.utils.registry import Registry

DATASETS = Registry('dataset')


def build_dataset_from_cfg(cfg, default_args=None):
    return DATASETS.build(cfg, default_args=default_args)
