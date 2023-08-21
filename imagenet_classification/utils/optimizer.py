import torch


def get_optimizer(args, model):
    print(args.optimizer)
    if args.optimizer == 'SGD':
        if args.weigh_decay_apply_on_all:
            return torch.optim.SGD(model.parameters(), args.lr,
                                   momentum=args.momentum,
                                   weight_decay=args.weight_decay,
                                   nesterov=args.nesterov)
        else: # apply weight decay regularization only on conv/fc weight
            return torch.optim.SGD(get_parameters(model), args.lr,
                                   momentum=args.momentum,
                                   weight_decay=args.weight_decay,
                                   nesterov=args.nesterov)
    elif args.optimizer == 'RMSprop':
        return torch.optim.RMSprop(model.parameters(),
                                   args.lr, alpha=0.9, #
                                   momentum=args.momentum,
                                   weight_decay=args.weight_decay)
    else:
        raise NotImplementedError("The optimizer {} is not implemented!".format(args.optimizer))


def get_parameters(model):
    group_no_weight_decay = []
    group_weight_decay = []
    for pname, p in model.named_parameters():
        if pname.find('weight') >= 0 and len(p.size()) > 1:
            print('include ', pname, p.size())
            group_weight_decay.append(p)
        else:
            print('not include ', pname, p.size())
            group_no_weight_decay.append(p)
    assert len(list(model.parameters())) == len(group_weight_decay) + len(group_no_weight_decay)
    groups = [dict(params=group_weight_decay), dict(params=group_no_weight_decay, weight_decay=0.)]
    return groups