import argparse
import os
import random

import pretrainedmodels
import torch
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader

import deepml
from deepml import datasets, losses
from deepml.models import CNNs
from deepml.utils import libs

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
    device = torch.device(('cuda:%d' % gpu_id) if gpu_id >= 0 else 'cpu')

    # build model
    model = CNNs(
        out_dim=args.outdim,
        arch=args.arch,
        pretrained=args.pretrained,
        normalized=args.normalized
    ).to(device)

    # setup loss function
    criterion = losses.__dict__[args.loss]()
    # setup data set
    data_path = os.path.abspath(DATA_PATHS[args.data])
    data = datasets.__dict__[args.data](data_path)

    inverted = (model.base.input_space == 'BGR')
    train_loader = DataLoader(
        data.get_dataloader(
            ttype='train',
            inverted=inverted,
            transform=libs.get_data_augmentation(
                img_size=args.img_size,
                mean=model.base.mean,
                std=model.base.std,
                ttype='train'
            )
        ),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=gpu_id >= 0
    )

    valid_loader = DataLoader(
        data.get_dataloader(
            ttype='valid',
            inverted=inverted,
            transform=libs.get_data_augmentation(
                img_size=args.img_size,
                mean=model.base.mean,
                std=model.base.std,
                ttype='valid'
            )
        ),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=gpu_id >= 0
    )

    # setup the optimizer
    linear_params = model.base.last_linear.parameters()
    ignored_params = list(map(id, linear_params))
    base_params = filter(lambda p: id(p) not in ignored_params,
                         model.parameters())
    optimizer = torch.optim.Adam([
        {'params': base_params},
        {'params': linear_params, 'lr': args.lr}
    ], lr=args.lr*0.1, weight_decay=args.weight_decay)
    # Decay LR by a factor of 0.1 every 10 epochs
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=10, gamma=0.1)

    # setup device and print frequency
    args.device = device
    args.print_freq = len(train_loader) // 10

    # train the model
    libs.train(train_loader, valid_loader, model,
               criterion, optimizer, scheduler, args)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Deep metric learning')

    parser.add_argument('--data', default='Cub', required=True,
                        help='name of the dataset')
    parser.add_argument('-a', '--arch', metavar='ARCH', default='bninception',
                        choices=pretrainedmodels.model_names,
                        help='model architecture: ' +
                        ' | '.join(pretrainedmodels.model_names) +
                        ' (default: bninception)')
    parser.add_argument('-p', '--pretrained', metavar='PRET',
                        default='imagenet',
                        help='use pre-trained model')
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
    parser.add_argument('--lr', '--learning-rate', default=1e-5, type=float,
                        metavar='LR', help='initial learning rate', dest='lr')
    parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                        metavar='W', help='weight decay (default: 1e-4)',
                        dest='weight_decay')
    parser.add_argument('--seed', default=None, type=int,
                        help='seed for initializing training. ')
    parser.add_argument('--normalized', dest='normalized', default=True,
                        action='store_true', help='normalize the last layer')

    args = parser.parse_args()

    main(args)
    print('Training was done!')
