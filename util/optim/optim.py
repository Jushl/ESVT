import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import re


__all__ = ['AdamW', 'SGD', 'Adam', 'MultiStepLR', 'CosineAnnealingLR', 'OneCycleLR', 'LambdaLR']


SGD = optim.SGD
Adam = optim.Adam
AdamW = optim.AdamW


MultiStepLR = lr_scheduler.MultiStepLR
CosineAnnealingLR = lr_scheduler.CosineAnnealingLR
OneCycleLR = lr_scheduler.OneCycleLR
LambdaLR = lr_scheduler.LambdaLR


def get_optim_params(model):
    cfg_params = [
        {'params': '^(?=.*backbone)(?!.*norm|bn).*$', 'lr': 0.00005},
        {'params': '^(?=.*backbone)(?=.*norm|bn).*$', 'lr': 0.00005, 'weight_decay': 0.0},
        {'params': '^(?=.*(?:encoder|decoder))(?=.*(?:norm|bn|bias)).*$', 'weight_decay': 0.0}
    ]
    param_groups = []
    visited = []
    for pg in cfg_params:
        pattern = pg['params']
        params = {k: v for k, v in model.named_parameters() if v.requires_grad and len(re.findall(pattern, k)) > 0}
        pg['params'] = params.values()
        param_groups.append(pg)
        visited.extend(list(params.keys()))

    names = [k for k, v in model.named_parameters() if v.requires_grad]

    if len(visited) < len(names):
        unseen = set(names) - set(visited)
        params = {k: v for k, v in model.named_parameters() if v.requires_grad and k in unseen}
        param_groups.append({'params': params.values()})
        visited.extend(list(params.keys()))

    assert len(visited) == len(names), ''
    return param_groups


def build_optim(model, args):
    params = get_optim_params(model)

    if args.optimizer == 'AdamW':
        return AdamW(params=params, lr=args.lr, weight_decay=args.weight_decay)
    elif args.optimizer == 'Adam':
        return Adam(params=params, lr=args.lr, weight_decay=args.weight_decay)
    elif args.optimizer == 'SGD':
        return SGD(params=params, lr=args.lr, )
    else:
        assert args.optimizer in ['AdamW', 'Adam', 'SGD']

