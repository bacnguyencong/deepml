import os
import shutil
import time

import numpy as np
import pandas as pd

import torch
from torchvision import transforms

from ..evals import nmi_clustering, recall_at_k


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
        for i, (input, target) in enumerate(data_loader):

            # place input tensors on the device
            input = input.to(args.device)
            target = target.to(args.device)

            # compute output
            features.append(model(input).cpu().numpy())
            labels.append(target.cpu().numpy().reshape(-1, 1))

            if i % args.print_freq == 0:
                print('Test: [{0}/{1}]'.format(i, len(data_loader)))

    return np.vstack(features), np.vstack(labels)


def train(train_loader, val_loader, model, criterion, optimizer, args):
    """Train the model.

    Args:
        train_loader (DataLoader): The training data loader.
        val_loader (DataLoader): The validation data loader.
        model (CNNs): The model.
        criterion (Loss): The loss function.
        optimizer (Optimizer): The optimizer.
        args (Any): The input arguments.
    """
    losses = []
    best_loss = np.inf
    for epoch in range(args.start_epoch, args.epochs):
        # adjust the learning rate
        adjust_learning_rate(optimizer, epoch, args)
        # run an epoch
        run_epoch(train_loader, model, criterion, optimizer, epoch, args)
        # compute the valiation
        loss = validate(val_loader, model, criterion, args)
        # update the best parameters
        save_checkpoint({
            'epoch': epoch + 1,
            'arch': args.arch,
            'state_dict': model.state_dict(),
            'best_loss': best_loss,
            'optimizer': optimizer.state_dict(),
        }, loss < best_loss)
        # update best accuracy
        best_loss = min(best_loss, loss)
        # keep tracking
        losses.append(loss)

    # write the output
    pd.DataFrame({
        'epch': range(args.start_epoch, args.epochs),
        'loss': losses
    }).to_csv(os.path.join('output', 'train_loss.csv'), index=False)


def run_epoch(train_loader, model, criterion, optimizer, epoch, args):
    """Run one epoch.

    Args:
        train_loader ([type]): [description]
        model ([type]): [description]
        criterion ([type]): [description]
        optimizer ([type]): [description]
        epoch ([type]): [description]
        args ([type]): [description]
    """
    batch_time = AverageMeter()
    losses = AverageMeter()
    # switch to train mode
    model.train()

    end = time.time()
    for i, (input, target) in enumerate(train_loader):

        # place input tensors on the device
        input = input.to(args.device)
        target = target.to(args.device)

        # compute output
        output = model(input)
        loss = criterion(output, target)

        # measure accuracy and record loss
        losses.update(loss.item(), input.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})'.format(
                      epoch, i, len(train_loader),
                      batch_time=batch_time, loss=losses))


def validate(val_loader, model, criterion, args):
    """Validate the model.

    Args:
        val_loader ([type]): [description]
        model ([type]): [description]
        criterion ([type]): [description]
        args ([type]): [description]

    Returns:
        [type]: [description]
    """

    batch_time = AverageMeter()
    losses = AverageMeter()

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        end = time.time()
        for i, (input, target) in enumerate(val_loader):

            # place input tensors on the device
            input = input.to(args.device)
            target = target.to(args.device)

            # compute output
            output = model(input)
            loss = criterion(output, target)

            # measure accuracy and record loss
            losses.update(loss.item(), input.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                print('Validate: [{0}/{1}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})'.format(
                          i, len(val_loader),
                          batch_time=batch_time, loss=losses))

    return losses.avg


def test(test_loader, model, args):
    """Validate the model.

    Args:
        test_loader (DataLoader): The data set loader.
        model: The network.
        criterion (Loss): The loss function.
        args: The hyperparameter arguments.
    """
    features, labels = compute_feature(test_loader, model, args)
    topk = np.arange(1, 101, 1)
    result = recall_at_k(features, labels, topk)
    recalls = pd.DataFrame({'k': topk, 'recall': result})
    nmi = nmi_clustering(features, labels)

    # write the recall@k results
    recalls.to_csv(os.path.join('output', 'recall.csv'), index=False)
    # write the clustering results
    file = open(os.path.join('output', 'nmi.txt'), 'w')
    file.write('%.8f' % nmi)
    file.close()


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    filename = os.path.join('output', filename)
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, os.path.join('output', 'model_best.pth.tar'))


def get_data_augmentation(img_size, ttype='train'):
    """Get data augmentation

    Args:
        img_size (int): The desired output dim.
        ttype (str, optional): Defaults to 'train'. The type
            of data augmenation.

    Returns:
        Transform: A transform.
    """
    if ttype == 'valid':
        ttype = 'test'
        # setup data augmentation
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
    trans = dict()
    trans['train'] = transforms.Compose([
        transforms.Resize(256),
        transforms.RandomCrop(img_size),
        transforms.ToTensor(),
        normalize
    ])
    trans['test'] = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        normalize
    ])
    return trans[ttype]
