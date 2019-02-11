import argparse
import random
import os
import torch
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader

from deepml import datasets, losses
from deepml.models import CNNs
from deepml.utils import libs
import deepml

# list of data paths
DATA_PATHS = {
    'Cub': './data/cub_200_2011',
    'Stand': './data/stanford',
    'Car': './data/cars196'
}


def main(args):
    """Perform train."""
    print(args)
    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True

    # setup GPUs or CPUs
    num_device = torch.cuda.device_count()
    gpu_id = torch.cuda.current_device() if num_device > 0 else -1
    device = torch.device('cuda:%d'.format(gpu_id) if gpu_id >= 0 else 'cpu')

    # build model
    model = CNNs(
        out_dim=args.outdim,
        arch=args.arch,
        pretrained=args.pretrained
    ).to(device)

    # setup loss function
    criterion = losses.__dict__[args.loss]()
    # setup data set
    data_path = os.path.abspath(DATA_PATHS[args.data])
    data = datasets.__dict__[args.data](data_path)

    train_loader = DataLoader(
        data.get_train_loader(
            libs.get_data_augmentation(args.img_size, 'train')
        ),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=num_device > 0
    )

    valid_loader = DataLoader(
        data.get_train_loader(
            libs.get_data_augmentation(args.img_size, 'valid')
        ),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=num_device > 0
    )

    # setup the optimizer
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay
    )

    # setup device and print frequency
    args.device = device
    args.print_freq = len(train_loader) // 10

    # train the model
    libs.train(train_loader, valid_loader, model, criterion, optimizer, args)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Deep metric learning')

    parser.add_argument('--data', default='Cub', required=True,
                        help='name of the dataset')
    parser.add_argument('-a', '--arch', metavar='ARCH', default='bnincepnet',
                        choices=deepml.MODEL_NAMES,
                        help='model architecture: ' +
                        ' | '.join(deepml.MODEL_NAMES) +
                        ' (default: bnincepnet)')
    parser.add_argument('-l', '--loss', metavar='LOSS',
                        default='ContrastiveLoss',
                        choices=deepml.MODEL_LOSSES,
                        help='model loss: | '.join(deepml.MODEL_LOSSES) +
                        ' (default: ContrastiveLoss)')
    parser.add_argument('-img_size', default=227, type=int,
                        help='image shape (default: 227)')
    parser.add_argument('-j', '--workers', default=8, type=int, metavar='N',
                        help='number of data loading workers (default: 8)')
    parser.add_argument('--epochs', default=200, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                        help='manual epoch number (useful on restarts)')
    parser.add_argument('--outdim', default=128, type=int, metavar='N',
                        help='number of features')
    parser.add_argument('-b', '--batch-size', default=256, type=int,
                        metavar='N',
                        help='mini-batch size (default: 256)')
    parser.add_argument('--lr', '--learning-rate', default=0.0001, type=float,
                        metavar='LR', help='initial learning rate', dest='lr')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                        help='momentum')
    parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                        metavar='W', help='weight decay (default: 1e-4)',
                        dest='weight_decay')
    parser.add_argument('--pretrained', dest='pretrained', default=False,
                        action='store_true', help='use pre-trained model')
    parser.add_argument('--seed', default=None, type=int,
                        help='seed for initializing training. ')

    args = parser.parse_args()

    main(args)
    print('Training was done!')
