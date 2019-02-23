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


def train(train_loader,
          val_loader,
          test_loader,
          model, criterion,
          optimizer,
          scheduler,
          args):
    """Train the model.

    Args:
        train_loader (DataLoader): The training data loader.
        val_loader (DataLoader): The validation data loader.
        test_loader (DataLoader): The test data loader.
        model (CNNs): The model.
        criterion (Loss): The loss function.
        optimizer (Optimizer): The optimizer.
        args (Any): The input arguments.
    """
    losses, acces = list(), list()
    best_acc = -np.inf
    topk = ([1, 5])
    tests = list()

    # setup early stopping
    early_stop = EarlyStopping(mode='max', patience=10)

    for epoch in range(args.start_epoch, args.epochs):

        # build the triplets
        print('Rebuiding the targets and triplets...')
        X, y = compute_feature(train_loader.standard_loader, model, args)
        train_loader.generate_batches(X, y)

        # run an epoch
        loss = run_epoch(train_loader, model, criterion,
                         optimizer, epoch, args)

        # compute the valiation
        acc = validate(val_loader, model, args, topk)
        is_best = acc[0] > best_acc

        # update best accuracy
        best_acc = max(best_acc, acc[0])

        # update the best parameters
        save_checkpoint({
            'epoch': epoch + 1,
            'arch': args.arch,
            'state_dict': model.state_dict(),
            'best_acc': best_acc,
            'optimizer': optimizer.state_dict(),
        }, is_best)

        # keep tracking
        acces.append(acc)
        losses.append(loss)
        print('Loss=%.4f\tRecall\t@1=%.4f\t@5=%.4f' %
              (loss, acc[0], acc[1]))

        # adjust the learning rate
        scheduler.step(acc[0])

        # early stopping
        if early_stop.step(acc[0]):
            print('Early stopping reached! Stop running...')
            break

        if test_loader is not None:
            acc = validate(test_loader, model, args, topk)
            print('Test\tRecall\t@1=%.4f\t@5=%.4f' % (acc[0], acc[1]))
            tests.append(acc)

    # --------------------------------------------------------------------#
    # write the output
    tab = pd.DataFrame({
        'epoch': range(args.start_epoch + 1, args.epochs + 1),
        'loss': np.array(losses)
    })
    # write the valid results
    acces = np.vstack(acces)
    for i, k in enumerate(topk):
        tab['train_recall_at_{}'.format(k)] = acces[:, i]

    # write the test results
    if test_loader is not None:
        tests = np.vstack(tests)
        for i, k in enumerate(topk):
            tab['test_recall_at_{}'.format(k)] = tests[:, i]

    tab.to_csv(os.path.join('output', 'train_track.csv'), index=False)
    # --------------------------------------------------------------------#


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
    losses = AverageMeter()
    # switch to train mode
    model.train()

    for i, data in enumerate(train_loader):

        # place input tensors on the device
        input = data[0].to(args.device)
        target = data[1].to(args.device)

        # compute output
        output = model(input)
        if len(data) > 2:
            loss = criterion(output, target, data[2])
        else:
            loss = criterion(output, target)

        # measure accuracy and record loss
        losses.update(loss.item(), input.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if i % args.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})'.format(
                      epoch, i, len(train_loader), loss=losses))
    return losses.avg


def validate(val_loader, model, args, topk):
    """Validate the model.

    Args:
        val_loader ([type]): [description]
        model ([type]): [description]
        args ([type]): [description]

    Returns:
        [type]: [description]
    """
    features, labels = compute_feature(val_loader, model, args)
    return recall_at_k(features, labels, topk)


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
    recalls.to_csv(os.path.join('output', 'test_recall.csv'), index=False)
    # write the clustering results
    file = open(os.path.join('output', 'test_nmi.txt'), 'w')
    file.write('%.8f' % nmi)
    file.close()
    # write the features and labels
    pd.DataFrame(features).to_csv(os.path.join(
        'output', 'test_features.csv'), header=False, index=False)
    pd.DataFrame(labels).to_csv(os.path.join(
        'output', 'test_labels.csv'), header=False, index=False)


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
