import argparse
import random

import torch
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from torchvision import models, transforms

from deepml import datasets, losses, test, train
from deepml.models import CNNs

# list of all arquitechtures
MODEL_NAMES = sorted(
    name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name])
)
# list of all losses
MODEL_LOSSES = sorted(
    name for name in losses.__dict__
    if callable(losses.__dict__[name]) and not name.startswith("__")
)
# list of all data sets
MODEL_DATASETS = sorted(
    name for name in datasets.__dict__
    if callable(datasets.__dict__[name]) and not name.startswith("__")
)
# list of data paths
DATA_PATHS = {
    'Cub': '/home/kunkun220189/Documents/Python/deepml/data/cub_200_2011',
    'Stand': '/home/kunkun220189/Documents/Python/deepml/data/stanford',
    'Car': '/home/kunkun220189/Documents/Python/deepml/data/cars196'
}


def main(args):
    """Perform train

    Args:
        args ([type]): [description]
    """

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True

    # setup GPUs or CPUs
    num_device = torch.cuda.device_count()
    device = torch.device('cuda:%d'.format(0 if args.gpu is None else args.gpu)
                          if num_device > 0 else 'cpu')

    # setup data augmentation
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
    train_transforms = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(args.img_size),
        transforms.ToTensor(),
        normalize
    ])
    test_transforms = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(args.img_size),
        transforms.ToTensor(),
        normalize
    ])

    # build model
    model = CNNs(
        out_dim=args.outdim,
        arch=args.arch,
        pretrained=args.pretrained
    ).to(device)

    args.device = device
    # setup loss function
    criterion = losses.__dict__[args.loss]()
    # setup data set
    data = datasets.__dict__[args.data](DATA_PATHS[args.data])

    train_loader = DataLoader(
        data.get_train_loader(train_transforms),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=num_device > 0
    )

    valid_loader = DataLoader(
        data.get_train_loader(test_transforms),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=num_device > 0
    )

    test_loader = DataLoader(
        data.get_test_loader(test_transforms),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=num_device > 0
    )

    # setup the optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr,
                                 weight_decay=args.weight_decay)

    # train the model
    train(train_loader, valid_loader, model, criterion, optimizer, args)

    # test the model
    test(test_loader, model, criterion, optimizer, args)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Deep metric learning')

    parser.add_argument('--data', default='Cub', required=True,
                        help='name of the dataset')
    parser.add_argument('-a', '--arch', metavar='ARCH', default='bnincepnet',
                        choices=MODEL_NAMES,
                        help='model architecture: ' +
                        ' | '.join(MODEL_NAMES) +
                        ' (default: bnincepnet)')
    parser.add_argument('-l', '--loss', metavar='LOSS',
                        default='ContrastiveLoss',
                        choices=MODEL_LOSSES,
                        help='model loss: | '.join(MODEL_LOSSES) +
                        ' (default: ContrastiveLoss)')
    parser.add_argument('-img_size', default=227, type=int,
                        help='image shape (default: 227)')
    parser.add_argument('-j', '--workers', default=8, type=int, metavar='N',
                        help='number of data loading workers (default: 4)')
    parser.add_argument('--epochs', default=90, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                        help='manual epoch number (useful on restarts)')
    parser.add_argument('--outdim', default=128, type=int, metavar='N',
                        help='number of features')
    parser.add_argument('-b', '--batch-size', default=4, type=int,
                        metavar='N',
                        help='mini-batch size (default: 256), this is the'
                        'total batch size of all GPUs on the current node'
                        'when using Data Parallel or Distributed Data Parallel'
                        )
    parser.add_argument('-p', '--print-freq', default=10, type=int,
                        metavar='N', help='print frequency (default: 10)')
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
    parser.add_argument('--pretrained', dest='pretrained', default=True,
                        action='store_true', help='use pre-trained model')
    parser.add_argument('--seed', default=None, type=int,
                        help='seed for initializing training. ')
    parser.add_argument('--gpu', default=None, type=int, help='GPU id to use.')
    args = parser.parse_args()

    main(args)
    print('Everything is OK')
