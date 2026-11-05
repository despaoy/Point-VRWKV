"""Source-only domain streams and explicit, single-view target inputs."""
import math
import random
from pathlib import Path

import numpy as np
import torch

from openpoints.cpp.fps import fps
from .transforms import normalize
from .build import DATASETS

CLASSES = {
    'pointda': ['bathtub', 'bed', 'bookshelf', 'cabinet', 'chair', 'lamp',
                'monitor', 'plant', 'sofa', 'table'],
    'pointdg': ['cabinet', 'chair', 'lamp', 'sofa', 'table'],
}


def load_split(root, domain, split, num_classes):
    path = Path(root) / domain
    points = np.array(np.load(path / f'{split}_pts.npy', allow_pickle=False)[..., :3],
                      dtype=np.float32, copy=True)
    raw_labels = np.load(path / f'{split}_label.npy', allow_pickle=False).reshape(-1)
    if (points.ndim != 3 or points.shape[-1] != 3 or len(points) != len(raw_labels)
            or not len(points) or not np.isfinite(points).all()
            or not np.isfinite(raw_labels).all()
            or not np.equal(raw_labels, np.round(raw_labels)).all()):
        raise ValueError(f'Invalid points/labels: {domain}/{split}')
    labels = raw_labels.astype(np.int64)
    if labels.min() < 0 or labels.max() >= num_classes:
        raise ValueError(f'Class mapping outside [0,{num_classes}): {domain}/{split}')
    return points, labels


@DATASETS.register_module()
class SourceStream:
    """Equal objects per source, shuffled without replacement inside each domain."""
    def __init__(self, root, config):
        if config['target'] in config['sources'] or len(set(config['sources'])) != len(config['sources']):
            raise ValueError('Target must be excluded from distinct source domains')
        if config['batch'] % len(config['sources']):
            raise ValueError('Batch must be divisible by source-domain count')
        self.per_domain = config['batch'] // len(config['sources'])
        self.generic = config['augmentation'] == 'generic'
        self.clouds, self.labels = [], []
        self.rng = random.Random(config['sampling_seed'])
        self.point_rng = np.random.RandomState(config['sampling_seed'])
        for domain in config['sources']:
            x, y = load_split(root, domain, 'train', config['model']['num_classes'])
            if not self.generic:
                if x.shape[1] == 1024:
                    x = np.repeat(x, 2, axis=1)
                if x.shape[1] != 2048:
                    raise ValueError('Standard/mild recipes expect 1024 or 2048 source points')
                for start in range(0, len(x), 256):
                    x[start:start+256] = normalize(torch.from_numpy(x[start:start+256])).numpy()
            elif x.shape[1] < 1024:
                raise ValueError('Generic recipe requires at least 1024 source points')
            self.clouds.append(x)
            self.labels.append(y)
        self.pools = [list(range(len(x))) for x in self.clouds]
        for pool in self.pools:
            self.rng.shuffle(pool)
        self.positions = [0] * len(self.pools)
        self.epoch_steps = math.ceil(max(map(len, self.clouds)) / self.per_domain)

    def next(self):
        selected = []
        for d, pool in enumerate(self.pools):
            for _ in range(self.per_domain):
                if self.positions[d] == len(pool):
                    self.rng.shuffle(pool)
                    self.positions[d] = 0
                selected.append((d, pool[self.positions[d]]))
                self.positions[d] += 1
        self.rng.shuffle(selected)
        clouds = []
        for d, i in selected:
            x = self.clouds[d][i]
            if self.generic:
                ids = np.arange(len(x)) if len(x) == 1024 else self.point_rng.choice(len(x), 1024, replace=False)
                x = x[ids].copy()
                x -= x.mean(axis=0, keepdims=True)
                radius = np.sqrt(np.square(x).sum(axis=1)).max()
                if radius <= 1e-12:
                    raise ValueError('Degenerate source cloud')
                x /= radius
            clouds.append(x)
        labels = torch.tensor([self.labels[d][i] for d, i in selected]).long()
        return torch.from_numpy(np.stack(clouds)), labels

    def state_dict(self):
        state = self.point_rng.get_state()
        return dict(pools=self.pools, positions=self.positions, rng=self.rng.getstate(),
                    point_rng=(state[0], state[1].tolist(), *state[2:]))

    def load_state_dict(self, state):
        self.pools, self.positions = state['pools'], state['positions']
        self.rng.setstate(state['rng'])
        name, values, pos, gaussian, cached = state['point_rng']
        self.point_rng.set_state((name, np.asarray(values, dtype=np.uint32), pos, gaussian, cached))


@torch.no_grad()
def target_inputs(root, config, eval_points=None):
    """Explicit NPY1024 or raw XYZ -> original input-FPS kernel -> normalize."""
    raw, labels = load_split(root, config['target'], 'test', config['model']['num_classes'])
    if eval_points:
        raw = np.array(np.load(eval_points, allow_pickle=False), dtype=np.float32, copy=True)
        if raw.shape != (len(labels), 1024, 3) or not np.isfinite(raw).all():
            raise ValueError('Explicit evaluation points must be finite [samples,1024,3]')
    rows = []
    for start in range(0, len(raw), 24):
        x = torch.from_numpy(raw[start:start+24]).cuda()
        if x.shape[1] > 1024:
            x = x.gather(1, fps(x, 1024)[:, :, None].expand(-1, -1, 3))
        if x.shape[1] != 1024:
            raise ValueError('Target requires at least 1024 points')
        rows.append(normalize(x).cpu())
    return torch.cat(rows), labels
