import argparse
import os
import random

import torch
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader

import deepml
import pretrainedmodels
from deepml import datasets, losses
from deepml.datasets.loader import DeepMLDataLoader
from deepml.models import CNNs
from deepml.utils import libs, runner

# list of data paths
DATA_PATHS = {
    'Cub': './data/cub_200_2011',
    'Stanford': './data/stanford',
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

    temp_data = data.get_dataset(
        ttype='train',
        inverted=inverted,
        transform=libs.get_data_augmentation(
            img_size=args.img_size,
            mean=model.base.mean,
            std=model.base.std,
            ttype='test'
        )
    )
    # create train loader
    train_loader = DeepMLDataLoader(
        data.get_dataset(
            ttype='train',
            inverted=inverted,
            transform=libs.get_data_augmentation(
                img_size=args.img_size,
                mean=model.base.mean,
                std=model.base.std,
                ttype='train'
            )
        ),
        temp_data,
        batch_size=args.batch_size,
        shuffle=False,  # must be False to avoid problem
        n_targets=args.n_targets,
        num_workers=args.workers,
        pin_memory=gpu_id >= 0
    )

    # create test loader
    test_loader = DataLoader(
        data.get_dataset(
            ttype='test',
            inverted=inverted,
            transform=libs.get_data_augmentation(
                img_size=args.img_size,
                mean=model.base.mean,
                std=model.base.std,
                ttype='test'
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
        {'params': linear_params, 'lr': args.lr * 10}
    ], lr=args.lr, weight_decay=args.weight_decay)

    # Decay LR by a factor of 0.5 every 5 epochs
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 'min', factor=0.5, patience=5, verbose=True)

    # setup device and print frequency
    args.device = device
    args.print_freq = 10

    # train the model
    runner.train(train_loader, test_loader, model,
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
    parser.add_argument('-img_size', default=224, type=int,
                        help='image shape (default: 224)')
    parser.add_argument('-n_targets', default=5, type=int,
                        help='number of targets (default: 5)')
    parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                        help='number of data loading workers (default: 8)')
    parser.add_argument('--epochs', default=150, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('--outdim', default=512, type=int, metavar='N',
                        help='number of features')
    parser.add_argument('-b', '--batch-size', default=256, type=int,
                        metavar='N',
                        help='mini-batch size (default: 256)')
    parser.add_argument('--lr', '--learning-rate', default=1e-4, type=float,
                        metavar='LR', help='initial learning rate', dest='lr')
    parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                        metavar='W', help='weight decay (default: 1e-4)',
                        dest='weight_decay')
    parser.add_argument('--seed', default=None, type=int,
                        help='seed for initializing training. ')
    parser.add_argument('--normalized', dest='normalized',
                        action='store_true', help='normalize the last layer')

    args = parser.parse_args()

    main(args)
    print('Training was done!')
