from deepml import train
from deepml.models import CNNs
import torch
import numpy as np
from torchvision import models
from deepml import losses

import argparse
import random
import torch.backends.cudnn as cudnn
import warnings


model_names = sorted(
    name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name])
)

model_losses = sorted(
    name for name in losses.__dict__
    if callable(losses.__dict__[name])
)


def main(args):

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True

    num_device = torch.cuda.device_count()
    device = torch.device('cuda:%d'.format(0 if args.gpu is None else args.gpu)
                          if num_device > 0 else 'cpu')

    model = CNNs(out_dim=args.outdim, arch=args.arch)
    criterion = losses.__dict__[args.loss]

    model.to(device)
    train()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Deep metric learning')

    parser.add_argument('--data', default='cub', required=True,
                        help='name of Data Set')

    parser.add_argument('-a', '--arch', metavar='ARCH', default='bnincepnet',
                        choices=model_names,
                        help='model architecture: ' +
                        ' | '.join(model_names) +
                        ' (default: bnincepnet)')
    parser.add_argument('-l', '--loss', metavar='LOSS', default='contrastive',
                        choices=model_losses,
                        help='model loss: | '.join(model_losses) +
                        ' (default: contrastive)')
    parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                        help='number of data loading workers (default: 4)')
    parser.add_argument('--epochs', default=90, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('--outdim', default=128, type=int, metavar='N',
                        help='number of features')
    parser.add_argument('-b', '--batch-size', default=4, type=int,
                        metavar='N',
                        help='mini-batch size (default: 256), this is the'
                        'total batch size of all GPUs on the current node'
                        'when using Data Parallel or Distributed Data Parallel'
                        )
    parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                        metavar='LR', help='initial learning rate', dest='lr')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                        help='momentum')
    parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                        metavar='W', help='weight decay (default: 1e-4)',
                        dest='weight_decay')
    parser.add_argument('-e', '--evaluate', dest='evaluate',
                        action='store_true',
                        help='evaluate model on validation set')
    parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                        help='use pre-trained model')
    parser.add_argument('--seed', default=None, type=int,
                        help='seed for initializing training. ')
    parser.add_argument('--gpu', default=None, type=int, help='GPU id to use.')
    args = parser.parse_args()

    main(args)
    print('Everything is OK')


# print(models.__dict__)

"""
model = CNNs(out_dim=10, arch='bnincepnet')
input_shape = (3, 224, 224)

x = torch.rand(1, *input_shape)

y = model(x)
print(y)

"""
