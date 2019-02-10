import os
import shutil
import time

import numpy as np
import torch


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


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified
        values of k.

    """
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


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
    best_acc = 0
    for epoch in range(args.start_epoch, args.epochs):
        # adjust the learning rate
        adjust_learning_rate(optimizer, epoch, args)
        # run an epoch
        run_epoch(train_loader, model, criterion, optimizer, epoch, args)
        # compute the valiation
        acc = validate(val_loader, model, criterion, args)
        # update the best parameters
        save_checkpoint({
            'epoch': epoch + 1,
            'arch': args.arch,
            'state_dict': model.state_dict(),
            'best_acc': best_acc,
            'optimizer': optimizer.state_dict(),
        }, acc > best_acc)
        # update best accuracy
        best_acc = max(best_acc, acc)


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
    acces = AverageMeter()
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
        acc = accuracy(output, target)
        losses.update(loss.item(), input.size(0))
        acces.update(acc, input.size(0))

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
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Acc@ {acc.val:.3f} ({acc.avg:.3f})'.format(
                      epoch, i, len(train_loader), batch_time=batch_time,
                      loss=losses, acc=acces))


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
    acces = AverageMeter()

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
            acc = accuracy(output, target)
            losses.update(loss.item(), input.size(0))
            acces.update(acc, input.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                print('Test: [{0}/{1}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Acc@1 {acc.val:.3f} ({acc.avg:.3f})'.format(
                          i, len(val_loader), batch_time=batch_time,
                          loss=losses, acc=acces))

        print(' * Acc {acc.avg:.3f}'.format(acc=acces))

    return acces.avg


def test(test_loader, model, criterion, args):
    """Validate the model.

    Args:
        test_loader ([type]): [description]
        model ([type]): [description]
        criterion ([type]): [description]
        args ([type]): [description]

    Returns:
        [type]: [description]
    """

    features, labels = compute_feature(test_loader, model, args)

    return


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    filename = os.path.join('output', filename)
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, os.path.join('output', 'model_best.pth.tar'))
