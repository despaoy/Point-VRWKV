"""Source-only training; target-selected checkpoint exploration is explicit."""
import argparse
import json
import math
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from openpoints.dataset.domains import CLASSES, target_inputs
from openpoints.dataset.transforms import fps_augment, generic_augment
from openpoints.models import build_model_from_cfg
from openpoints.dataset import build_dataset_from_cfg
from openpoints.optim import build_optimizer_from_cfg
from openpoints.utils import EasyConfig
import yaml
from openpoints.utils.checkpoint import restore_training, save_checkpoint
from openpoints.utils.metrics import evaluate


def main():
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cfg', required=True)
    parser.add_argument('--data-root', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--eval-points', help='Explicit same-order, unnormalized NPY1024 input')
    parser.add_argument('--resume', help='Checkpoint produced by this release')
    parser.add_argument('--initial-weights', help='Trusted, strict model state_dict for a fresh run')
    args, opts = parser.parse_known_args()
    if args.resume and args.initial_weights:
        raise ValueError('Use resume or initial weights, not both')
    cfg = EasyConfig()
    cfg.load(args.cfg, recursive=True)
    cfg.update(opts)
    config = cfg.dict()
    config['data_root'] = str(Path(args.data_root).resolve())
    config['eval_points'] = str(Path(args.eval_points).resolve()) if args.eval_points else None
    expected_domains = {'modelnet', 'shapenet', 'scannet'} if config['benchmark']=='pointda' else set('ABCD')
    if (set(config['sources']) | {config['target']} != expected_domains
            or config['model']['num_classes'] != len(CLASSES[config['benchmark']])):
        raise ValueError('Invalid benchmark/domain/class configuration')
    if config.get('epochs') is not None:
        valid_schedule = 0 <= config.get('warmup_epochs', 5) < config['epochs']
    else:
        valid_schedule = 0 <= config['warmup'] < config['steps']
    if not valid_schedule or config['precision'] not in ('fp32', 'bf16'):
        raise ValueError('Invalid schedule or precision')
    if config['augmentation'] not in ('standard', 'mild', 'generic'):
        raise ValueError('Unknown augmentation recipe')
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=bool(args.resume))
    if not torch.cuda.is_available():
        raise RuntimeError('A CUDA GPU and toolkit are required')
    torch.set_num_threads(4)
    random.seed(config['seed'])
    torch.manual_seed(config['seed'])
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True)
    model = build_model_from_cfg(config['model']).cuda()
    if args.initial_weights:
        model.load_state_dict(torch.load(args.initial_weights, map_location='cpu', weights_only=True), strict=True)
    optimizer = build_optimizer_from_cfg(model, lr=config['lr'], **config['optimizer'])
    stream = build_dataset_from_cfg(config['dataset'], default_args=dict(root=args.data_root, config=config))
    if config.get('epochs') is not None:
        config['steps'] = int(config['epochs']) * stream.epoch_steps
        config['warmup'] = int(config.get('warmup_epochs', 5)) * stream.epoch_steps
        if not 0 <= config['warmup'] < config['steps']:
            raise ValueError('Warmup must be shorter than training')
        config['epoch_steps'] = stream.epoch_steps
    aug_rng = torch.Generator().manual_seed(config['seed'])
    trans_rng = torch.Generator().manual_seed(config['seed']+101)
    # This is evaluation-only data; it never enters the optimization loop.
    points, target_labels = target_inputs(args.data_root, config, config['eval_points'])
    np.savez_compressed(output/'evaluation_inputs.npz', points=points.numpy(), labels=target_labels)
    first_step, best = 1, None
    if args.resume:
        payload = torch.load(args.resume, map_location='cpu', weights_only=True)
        if payload['config'] != config:
            raise ValueError('Resume configuration differs; use the original config/data paths')
        restore_training(payload, model, optimizer, stream, aug_rng, trans_rng)
        first_step, best = payload['step']+1, payload['best']
    else:
        save_checkpoint(output/'initial.pt', model, optimizer, 0, config, stream, aug_rng, trans_rng, best)
    (output/'cfg.yaml').write_text(yaml.safe_dump(config, sort_keys=False))
    history_path = output/'target.jsonl'
    history = ([json.loads(line) for line in history_path.read_text().splitlines() if line]
               if args.resume and history_path.exists() else [])
    started = time.monotonic()
    stop_reason = 'budget_exhausted'
    for step in range(first_step, config['steps']+1):
        model.train()
        raw, labels = stream.next()
        if config['augmentation']=='generic':
            inputs = generic_augment(raw, aug_rng, translation_generator=trans_rng).cuda()
        else:
            inputs = fps_augment(raw, aug_rng, mild=config['augmentation']=='mild')
        labels = labels.cuda()
        warmup = config['warmup']
        lr = (config['lr']*step/warmup if step<=warmup else config['min_lr'] +
              (config['lr']-config['min_lr'])*.5*(1+math.cos(math.pi*(step-warmup)/(config['steps']-warmup))))
        for group in optimizer.param_groups:
            group['lr'] = lr
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast('cuda', dtype=torch.bfloat16, enabled=config['precision']=='bf16'):
            logits = model(inputs).float()
        losses = F.cross_entropy(logits, labels, label_smoothing=config['smoothing'], reduction='none')
        if config['class_balance']:
            counts = F.one_hot(labels, num_classes=logits.shape[1]).sum(dim=0)
            losses = losses / counts[labels]
        loss = losses.mean()
        if not torch.isfinite(loss):
            raise FloatingPointError(f'Nonfinite loss at {step}')
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 10.)
        if not torch.isfinite(norm):
            raise FloatingPointError(f'Nonfinite gradient at {step}')
        optimizer.step()
        row = dict(step=step, epoch=step/stream.epoch_steps, loss=float(loss.detach()),
                   lr=lr, elapsed_seconds=time.monotonic()-started)
        if step==first_step or step%100==0:
            print(json.dumps(row), flush=True)
            with (output/'train.jsonl').open('a') as handle:
                handle.write(json.dumps(row)+'\n')
        should_evaluate = (step%stream.epoch_steps==0 or step==config['steps']
                           or (config['eval_every']>0 and step%config['eval_every']==0))
        if should_evaluate:
            score = dict(step=step, epoch=step/stream.epoch_steps, target_selected=True, **evaluate(
                model, points, target_labels, config['model']['num_classes'], output/f'predictions_{step}'))
            score['source_completed_passes'] = dict(zip(config['sources'],
                [step*stream.per_domain//len(x) for x in stream.clouds]))
            if best is None or score['correct']>best['correct']:
                best = score
                save_checkpoint(output/'best_target.pt', model, optimizer, step, config, stream, aug_rng, trans_rng, best)
            with (output/'target.jsonl').open('a') as handle:
                handle.write(json.dumps(score)+'\n')
            (output/'result.json').write_text(json.dumps(dict(latest=score, best=best), indent=2)+'\n')
            print(json.dumps(dict(event='evaluation', **score)), flush=True)
            history.append(score)
        if should_evaluate or step%1000==0:
            save_checkpoint(output/'last.pt', model, optimizer, step, config, stream, aug_rng, trans_rng, best)
        if should_evaluate and config['stop_oa'] is not None and score['oa_percent']>=config['stop_oa']:
            stop_reason = 'target_threshold'
            break
        policy = config.get('early_stopping')
        if should_evaluate and policy and step % stream.epoch_steps == 0:
            reason = early_stop_reason(history, policy)
            if reason:
                stop_reason = reason
                break
    (output/'COMPLETE.json').write_text(json.dumps(dict(
        step=step, epoch=step/stream.epoch_steps, stop_reason=stop_reason,
        elapsed_seconds=time.monotonic()-started, best=best, latest=history[-1] if history else None), indent=2))


def early_stop_reason(history, policy):
    """Require both OA and MCA to stall; only evaluate complete source epochs."""
    if not history or history[-1]['epoch'] < policy['min_epochs']:
        return None
    best_oa, best_mca = -1., -1.
    oa_epoch = mca_epoch = 0.
    for row in history:
        if row['oa_percent'] > best_oa + 1e-9:
            best_oa, oa_epoch = row['oa_percent'], row['epoch']
        if row['mca_percent'] > best_mca + policy['min_mca_gain']:
            best_mca, mca_epoch = row['mca_percent'], row['epoch']
    recent = history[-5:]
    if len(recent) == 5:
        for metric in ('oa_percent', 'mca_percent'):
            changes = [b[metric]-a[metric] for a, b in zip(recent, recent[1:])]
            if sum(changes) >= 0.2 and sum(delta > 0 for delta in changes) >= 3:
                return None
    stale = history[-1]['epoch'] - max(oa_epoch, mca_epoch)
    if stale >= policy['patience_epochs']:
        return 'oa_and_mca_plateau'
    recent = history[-policy['decline_epochs']:]
    if (stale >= policy['decline_patience_epochs']
            and len(recent) == policy['decline_epochs']
            and all(r['oa_percent'] <= best_oa-policy['decline_pp']
                    and r['mca_percent'] <= best_mca-policy['decline_pp'] for r in recent)):
        return 'sustained_oa_and_mca_decline'
    return None


if __name__ == '__main__':
    main()
