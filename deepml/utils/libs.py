import os
import shutil

import numpy as np
import numpy.matlib as matl

import torch

from sklearn.metrics import pairwise_distances
from torchvision import transforms
from tqdm import tqdm


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


def _generate_triplet(inds, tars, imps):
    k1 = tars.shape[0]
    k2 = imps.shape[0]
    n = inds.shape[0]
    T = np.zeros((3, n*k1*k2), dtype=np.int)
    T[0] = matl.repmat(inds.reshape(-1, 1), 1, k1 * k2).flatten()
    T[1] = matl.repmat(tars.T.flatten().reshape(-1, 1), 1, k2).flatten()
    T[2] = matl.repmat(imps.reshape(-1, 1), k1 * n, 1).flatten()
    return T


def build_triplets(X, y, n_targets=3):
    """Compute all triplet constraints.

    Args:
        X (np.array, shape = [n_samples, n_features]): The input data.
        y (np.array, shape = (n_samples,) ): The labels.
        n_target (int, optional): Defaults to 3. The number of targets.

    Returns:
        (np.array, shape = [3, n_triplets]): The triplet index
    """
    dist = pairwise_distances(X, X)
    np.fill_diagonal(dist, np.inf)
    # list of triplets
    Triplets = list()
    for label in np.unique(y):
        targets = np.where(label == y)[0]
        imposters = np.where(label != y)[0]
        # remove group of examples with a few targets or no imposters
        if len(targets) > 1 and len(imposters) > 0:
            # compute the targets
            true_n_targets = min(n_targets, len(targets) - 1)
            index = np.argsort(dist[targets, :][:, targets], axis=0)[
                0:true_n_targets]
            Triplets.append(_generate_triplet(
                targets, targets[index], imposters))

    # if set of triplet is not empty
    if len(Triplets) > 0:
        Triplets = np.hstack(Triplets)

    return Triplets


def _check_triplets(T, X, y):
    assert T.shape == np.unique(T, axis=1).shape
    for t in range(T.shape[1]):
        i, j, k = T[:, t]
        assert(y[i] == y[j] and y[i] != y[k])


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
            # transforms.RandomCrop((img_size, img_size)),
            transforms.RandomResizedCrop(scale=(0.16, 1), size=img_size),
            transforms.ToTensor(),
            normalize
        ])

    return transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.CenterCrop((img_size, img_size)),
        transforms.ToTensor(),
        normalize
    ])
