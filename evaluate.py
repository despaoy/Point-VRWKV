import argparse
import json
import os
from pathlib import Path

import torch

from openpoints.dataset.domains import target_inputs
from openpoints.models import build_model_from_cfg
from openpoints.utils.metrics import evaluate


def main():
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    parser = argparse.ArgumentParser(description='Evaluate a release checkpoint, FP32, one view')
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--data-root', required=True)
    parser.add_argument('--eval-points')
    parser.add_argument('--output', required=True, help='New directory for predictions and metrics')
    args = parser.parse_args()
    payload = torch.load(args.checkpoint, map_location='cpu', weights_only=True)
    config = payload['config']
    # Never silently replace a historical explicit input with another protocol.
    eval_points = args.eval_points or config.get('eval_points')
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    model = build_model_from_cfg(config['model']).cuda()
    model.load_state_dict(payload['model'], strict=True)
    points, labels = target_inputs(args.data_root, config, eval_points)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    score = evaluate(model, points, labels, config['model']['num_classes'], output/'predictions')
    score.update(step=payload['step'], target=config['target'], single_view=True,
                 input_protocol='explicit_npy1024' if eval_points else 'raw_fps1024',
                 target_selected=True)
    (output/'metrics.json').write_text(json.dumps(score, indent=2)+'\n')
    print(json.dumps(score, indent=2))


if __name__ == '__main__':
    main()
