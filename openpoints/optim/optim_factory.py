"""AdamW-only OpenPoints optimizer-factory adaptation.

Retains its filter_bias_and_bn option; unsupported optimizers are intentionally
excluded. Group order is retained from the released baseline for resume safety.
"""
import torch


def build_optimizer_from_cfg(model, NAME, lr, weight_decay, filter_bias_and_bn=True):
    if NAME.lower() != 'adamw':
        raise ValueError('This baseline supports AdamW only')
    parameters = list(model.parameters())
    if filter_bias_and_bn:
        decay, exempt = [], []
        for name, value in model.named_parameters():
            (exempt if value.ndim==1 or name.endswith('.bias') else decay).append(value)
        parameters = [dict(params=decay, weight_decay=weight_decay),
                      dict(params=exempt, weight_decay=0.)]
    return torch.optim.AdamW(parameters, lr=lr, weight_decay=weight_decay)
