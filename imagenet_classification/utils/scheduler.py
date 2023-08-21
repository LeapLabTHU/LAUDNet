import math
import numpy as np


def get_scheduler(args):
    if args.scheduler in ['multistep', 'cosine', 'linear', 'exponential', 'uneven_multistep', 'rmsprop_step']:
        return LrScheduler(args)
    else:
        raise NotImplementedError("The scheduler {} is not implemented! Please "
                                  "choose from [multistep, cosine, linear, "
                                  "exponential]".format(args.scheduler))


class LrScheduler(object):

    def __init__(self, args):
        self.args = args
        self.type = args.scheduler

    def step(self, optimizer, epoch, batch=None, nBatch=None):
        if self.type == 'multistep':
            if self.args.warmup_epoch:
                T_warmup_epoch = self.args.warmup_epoch
                if epoch < T_warmup_epoch:
                    lr = self.args.warmup_lr + (self.args.lr - self.args.warmup_lr) * (
                                (epoch * nBatch + batch) / (T_warmup_epoch * nBatch))
                else:
                    lr = self.args.lr * (self.args.lr_decay_rate ** (epoch // self.args.lr_decay_step))
            else: 
                lr = self.args.lr * (self.args.lr_decay_rate ** (epoch // self.args.lr_decay_step))
        elif self.type == 'uneven_multistep':
            lr = self.args.lr * (self.args.lr_decay_rate ** np.sum(np.array(self.args.lr_milestone) <= epoch))
        elif self.type == 'rmsprop_step':
            lr = self.args.lr * (self.args.lr_decay_rate ** ((epoch * nBatch + batch) // (
                    self.args.lr_decay_step * nBatch)))
        elif self.type == 'cosine':
            if self.args.warmup_epoch:
                T_warmup_epoch = self.args.warmup_epoch
                if epoch < T_warmup_epoch:
                    lr = self.args.warmup_lr + (self.args.lr - self.args.warmup_lr) * (
                                (epoch * nBatch + batch) / (T_warmup_epoch * nBatch))
                else:
                    T_total = (self.args.epochs - self.args.warmup_epoch) * nBatch
                    T_cur = (epoch - self.args.warmup_epoch) * nBatch + batch
                    lr = self.args.lr_min + 0.5 * (self.args.lr-self.args.lr_min) * (1 + math.cos(math.pi * T_cur / T_total))
            else:
                T_total = self.args.epochs * nBatch
                T_cur = (epoch % self.args.epochs) * nBatch + batch
                lr = 0.5 * self.args.lr * (1 + math.cos(math.pi * T_cur / T_total))
        elif self.type == 'linear':
            T_total = self.args.epochs * nBatch
            T_cur = (epoch % self.args.epochs) * nBatch + batch
            lr = self.args.lr * (1 - T_cur / T_total)
        elif self.type == 'exponential':
            
            lr = self.args.lr * (self.args.lr_decay_rate ** (epoch // self.args.lr_decay_step))
        else:
            pass
        
        if 'dyn' in self.args.arch and '_y' not in self.args.arch and 'mob' not in self.args.arch :
            for param_group in optimizer.param_groups:
                if param_group['name'] == 'backbone_params':
                    param_group['lr'] = lr
                else:
                    param_group['lr'] = lr * self.args.lr_mult
        else:
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr

        return lr