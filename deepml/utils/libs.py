import os
import shutil
from tqdm import tqdm

import numpy as np
import pandas as pd
import torch
from torchvision import transforms

from ..evals import nmi_clustering, recall_at_k
from .early_stopping import EarlyStopping


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


def adjust_learning_rate(optimizer, epoch, args):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs.
    """
    lr = args.lr * (0.1 ** (epoch // 30))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def compute_feature(data_loader, model, args):
    """Compute the features

    Args:
        data_loader ([type]): [description]
        model ([type]): [description]
        args ([type]): [description]

    Returns:
        features: The features
        labels: The corresponding labels
    """

    # switch to evaluate mode
    model.eval()
    features, labels = [], []

    with torch.no_grad():
        for i, (input, target) in enumerate(tqdm(data_loader)):

            # place input tensors on the device
            input = input.to(args.device)
            target = target.to(args.device)

            # compute output
            features.append(model(input).cpu().numpy())
            labels.append(target.cpu().numpy().reshape(-1, 1))

    return np.vstack(features), np.vstack(labels)


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    filename = os.path.join('output', filename)
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, os.path.join('output', 'model_best.pth.tar'))


def get_data_augmentation(img_size, mean, std, ttype):
    """Get data augmentation

    Args:
        img_size (int): The desired output dim.
        ttype (str, optional): Defaults to 'train'. The type
            of data augmenation.

    Returns:
        Transform: A transform.
    """
    # setup data augmentation
    mean = np.array(mean).astype(np.float)
    std = np.array(std).astype(np.float)
    # fix the error
    if (mean[0] > 1):
        mean = mean / 255.0
        std = std / 255.0
    normalize = transforms.Normalize(mean=mean, std=std)

    if ttype == 'train':
        return transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop((img_size, img_size)),
            transforms.ToTensor(),
            normalize
        ])

    return transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.CenterCrop((img_size, img_size)),
        transforms.ToTensor(),
        normalize
    ])
