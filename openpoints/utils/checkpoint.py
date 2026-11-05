from pathlib import Path
import random

import torch


def save_checkpoint(path, model, optimizer, step, config, stream, aug_rng, trans_rng, best):
    """Save only tensors and plain Python values, loadable with weights_only=True."""
    path = Path(path)
    temporary = path.with_suffix('.tmp')
    torch.save(dict(model=model.state_dict(), optimizer=optimizer.state_dict(), step=step,
                    config=config, stream=stream.state_dict(), best=best,
                    augmentation_rng=aug_rng.get_state(), translation_rng=trans_rng.get_state(),
                    python_rng=random.getstate(), torch_rng=torch.get_rng_state(),
                    cuda_rng=torch.cuda.get_rng_state_all()), temporary)
    temporary.replace(path)


def restore_training(payload, model, optimizer, stream, aug_rng, trans_rng):
    model.load_state_dict(payload['model'], strict=True)
    optimizer.load_state_dict(payload['optimizer'])
    stream.load_state_dict(payload['stream'])
    aug_rng.set_state(payload['augmentation_rng'])
    trans_rng.set_state(payload['translation_rng'])
    random.setstate(payload['python_rng'])
    torch.set_rng_state(payload['torch_rng'])
    torch.cuda.set_rng_state_all(payload['cuda_rng'])
