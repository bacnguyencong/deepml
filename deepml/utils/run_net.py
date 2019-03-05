import os
import time

import numpy as np
import pandas as pd

from ..evals import nmi_clustering, recall_at_k
from .early_stopping import EarlyStopping
from .libs import AverageMeter, compute_feature, save_checkpoint


def train(train_loader,
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
    losses = list()
    best_acc = -np.inf
    topk = ([1, 5])
    tests = list()

    # setup early stopping
    early_stop = EarlyStopping(mode='min', patience=15)

    for epoch in range(0, args.epochs):
        # Rebuiding mini-batchs
        X, y = compute_feature(train_loader.standard_loader, model, args)
        end = time.time()
        train_loader.generate_batches(X, y)
        print('Recomputed batches...%.8f' % (time.time() - end))

        # run an epoch
        loss = run_epoch(train_loader, model, criterion,
                         optimizer, epoch, args)

        is_best = False
        # compute the valiation
        if test_loader is not None:
            t_acc = validate(test_loader, model, args, topk)
            print('Test\tRecall\t@1=%.4f\t@5=%.4f' % (t_acc[0], t_acc[1]))
            tests.append(t_acc)
            is_best = t_acc[0] > best_acc
            # update best accuracy
            best_acc = max(best_acc, t_acc[0])

        # update the best parameters
        save_checkpoint({
            'epoch': epoch + 1,
            'arch': args.arch,
            'state_dict': model.state_dict(),
            'best_acc': best_acc,
            'optimizer': optimizer.state_dict(),
        }, is_best)

        # keep tracking
        losses.append(loss)
        print('Loss=%.4f' % loss)

        # adjust the learning rate
        scheduler.step(loss)

        # early stopping
        if early_stop.step(loss):
            print('Early stopping reached! Stop running...')
            break

    # --------------------------------------------------------------------#
    # write the output
    tab = pd.DataFrame({
        'epoch': range(1, len(losses) + 1),
        'loss': np.array(losses)
    })

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
    batch_time = AverageMeter()

    # switch to train mode
    model.train()

    for i, data in enumerate(train_loader):

        # place input tensors on the device
        input = data[0].to(args.device)
        target = data[1].to(args.device)

        # start measuring time
        end = time.time()

        # compute output
        output = model(input)
        if len(data) > 2:
            loss = criterion(output, target, data[2])
        else:
            loss = criterion(output, target)

        # update loss and time
        losses.update(loss.item(), input.size(0))
        batch_time.update(time.time() - end)

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if i % args.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})'.format(
                      epoch, i, len(train_loader),
                      batch_time=batch_time, loss=losses))

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

    # write the features and labels
    pd.DataFrame(features).to_csv(os.path.join(
        'output', 'test_features.csv'), header=False, index=False)
    pd.DataFrame(labels).to_csv(os.path.join(
        'output', 'test_labels.csv'), header=False, index=False)

    result = recall_at_k(features, labels, topk)
    print('Recall@k is computed ...')
    recalls = pd.DataFrame({'k': topk, 'recall': result})
    # write the recall@k results
    recalls.to_csv(os.path.join('output', 'test_recall.csv'), index=False)

    nmi = nmi_clustering(features, labels)
    print('NMI is computed ...')
    # write the clustering results
    file = open(os.path.join('output', 'test_nmi.txt'), 'w')
    file.write('%.8f' % nmi)
    file.close()
