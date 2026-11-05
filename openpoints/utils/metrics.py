import numpy as np
import torch


@torch.no_grad()
def evaluate(model, points, labels, num_classes, output):
    """Single view, FP32; save predictions and confusion at the given prefix."""
    was_training = model.training
    model.eval()
    try:
        logits = np.concatenate([model(x.cuda()).float().cpu().numpy()
                                 for x in points.split(24)])
    finally:
        model.train(was_training)
    if not np.isfinite(logits).all():
        raise FloatingPointError('Nonfinite evaluation logits')
    prediction = logits.argmax(1)
    confusion = np.zeros((num_classes, num_classes), dtype=np.int64)
    np.add.at(confusion, (labels, prediction), 1)
    support = confusion.sum(1)
    per_class = np.divide(confusion.diagonal(), support, out=np.zeros(num_classes), where=support>0)
    np.savez_compressed(str(output)+'.npz', logits=logits, labels=labels,
                        predictions=prediction, indices=np.arange(len(labels)), confusion=confusion)
    return dict(samples=len(labels), correct=int((prediction==labels).sum()),
                oa_percent=float(100*(prediction==labels).mean()),
                mca_percent=float(100*per_class[support>0].mean()),
                per_class_percent=(100*per_class).tolist(), support=support.tolist(),
                confusion=confusion.tolist())
