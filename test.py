import argparse
import os
import random

import torch
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader

import pretrainedmodels
from deepml import datasets
from deepml.models import CNNs
from deepml.utils import libs

# list of data paths
DATA_PATHS = {
    'Cub': './data/cub_200_2011',
    'Stand': './data/stanford',
    'Car': './data/cars196'
}


def main(args):
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
    model = CNNs(out_dim=args.outdim, arch=args.arch,
                 normalized=args.normalized)
    if os.path.isfile(args.checkpoint):
        checkpoint = torch.load(
            args.checkpoint,
            map_location=lambda storage, loc: storage)
        print("=> Loaded checkpoint '{}'".format(args.checkpoint))
        model.load_state_dict(checkpoint['state_dict'])
    else:
        raise ValueError(
            "=> No checkpoint found at '{}'".format(args.checkpoint))
    model = model.to(device)
    inverted = (model.base.input_space == 'BGR')
    # setup data set
    data_path = os.path.abspath(DATA_PATHS[args.data])
    data = datasets.__dict__[args.data](data_path)

    test_loader = DataLoader(
        data.get_dataloader(
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
        pin_memory=num_device > 0
    )

    args.device = device
    args.print_freq = len(test_loader) // 10

    # test the model
    libs.test(test_loader, model, args)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Deep metric learning')

    parser.add_argument('--data', default='Cub', required=True,
                        help='name of the dataset')
    parser.add_argument('-a', '--arch', metavar='ARCH', default='bninception',
                        choices=pretrainedmodels.model_names,
                        help='model architecture: ' +
                        ' | '.join(pretrainedmodels.model_names) +
                        ' (default: bninception)')
    parser.add_argument('-c', '--checkpoint', type=str,
                        default='./output/model_best.pth.tar', metavar='PATH')
    parser.add_argument('-img_size', default=227, type=int,
                        help='image shape (default: 227)')
    parser.add_argument('-j', '--workers', default=8, type=int, metavar='N',
                        help='number of data loading workers (default: 8)')
    parser.add_argument('--outdim', default=128, type=int, metavar='N',
                        help='number of features')
    parser.add_argument('-b', '--batch-size', default=256, type=int,
                        metavar='N',
                        help='mini-batch size (default: 256)')
    parser.add_argument('--seed', default=None, type=int,
                        help='seed for initializing training. ')
    parser.add_argument('--normalized', dest='normalized', default=True,
                        action='store_true', help='normalize the last layer')
    args = parser.parse_args()

    main(args)
    print('Test was done!')
