from __future__ import print_function

import math
import random

import numpy as np
import torch
import torch.optim as optim
from functools import partial

from torch.optim.lr_scheduler import LambdaLR
from transformers.optimization import _get_linear_schedule_with_warmup_lr_lambda


class TwoCropTransform:
    """Create two crops of the same image"""

    def __init__(self, transform):
        self.transform = transform

    def __call__(self, x):
        return [self.transform(x), self.transform(x)]


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


def adjust_learning_rate(args, optimizer, epoch):
    lr = args.learning_rate
    if args.cosine:
        eta_min = lr * (args.lr_decay_rate ** 3)
        lr = eta_min + (lr - eta_min) * (
                1 + math.cos(math.pi * epoch / args.epochs)) / 2
    else:
        steps = np.sum(epoch > np.asarray(args.lr_decay_epochs))
        if steps > 0:
            lr = lr * (args.lr_decay_rate ** steps)

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def get_linear_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps, last_epoch=-1):
    lr_lambda = partial(
        _get_linear_schedule_with_warmup_lr_lambda,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps,
    )
    return LambdaLR(optimizer, lr_lambda, last_epoch)


def warmup_learning_rate(args, epoch, batch_id, total_batches, optimizer):
    if args.warm and epoch <= args.warm_epochs:
        p = (batch_id + (epoch - 1) * total_batches) / \
            (args.warm_epochs * total_batches)
        lr = args.warmup_from + p * (args.warmup_to - args.warmup_from)

        for param_group in optimizer.param_groups:
            param_group['lr'] = lr


def set_optimizer(opt, model):
    optimizer = optim.Adam(model.parameters(),
                           lr=opt.learning_rate, eps=1e-6)
    return optimizer


def save_model(model, optimizer, opt, epoch, save_file):
    print('==> Saving...')
    torch.save(model.state_dict(), save_file)


import torch
from torch.utils.data import DataLoader, Sampler

class CustomBalanceSampler(Sampler):
    def __init__(self, data_source, batch_size, shuffle=True):
        self.data_source = data_source
        self.batch_size = batch_size
        self.shuffle = shuffle

        # Troviamo gli indici degli elementi con label 0 e label 1
        self.indices_label_0 = [i for i, label in enumerate(data_source["hard_label"]) if label == 0]
        self.indices_label_1 = [i for i, label in enumerate(data_source["hard_label"]) if label == 1]

    def __iter__(self):
        # Shuffle degli indici se richiesto
        if self.shuffle:
            random.shuffle(self.indices_label_0)
            random.shuffle(self.indices_label_1)

        # Calcoliamo il numero di batch necessari per bilanciare le label 0 e label 1
        num_batches = min(len(self.indices_label_0), len(self.indices_label_1)) // (self.batch_size // 2)

        # Creiamo gli iteratori sugli indici delle label 0 e label 1
        iter_label_0 = iter(self.indices_label_0)
        iter_label_1 = iter(self.indices_label_1)

        # Creiamo i batch bilanciati alternando tra le label 0 e label 1
        for _ in range(num_batches):
            batch = []
            for _ in range(self.batch_size // 2):
                try:
                    batch.append(next(iter_label_0))
                    batch.append(next(iter_label_1))
                except StopIteration:
                    break

            yield batch

    def __len__(self):
        return len(self.data_source)





import logging
import logging.config


logging_level_dict = {
    0: logging.WARNING,
    1: logging.INFO,
    2: logging.DEBUG
}


DEFAULT_CONFIG = {
    'version': 1,
    'disable_existing_loggers': False,
    'formatters': {
        'simple': {
            'format': "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        }
    },
    'handlers': {
        'console': {
            'class': 'logging.StreamHandler',
            'level': 'DEBUG',
            'formatter': 'simple',
            'stream': 'ext://sys.stdout'
        }
    }
}


def setup_logging(config=DEFAULT_CONFIG):
    """Setup logging configuration"""
    logging.config.dictConfig(config)


def setup_logger(cls, name='', verbose=0):
    logger = logging.getLogger(name)
    if verbose not in logging_level_dict:
        raise KeyError(f'Verbose option {verbose} for {name} not valid. '
                        'Valid options are {logging_level_dict.keys()}.')
    logger.setLevel(logging_level_dict[verbose])
    return logger


setup_logging()


