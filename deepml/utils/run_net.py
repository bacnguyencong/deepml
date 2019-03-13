import time

import numpy as np
import pandas as pd

from ..evals import nmi_clustering, recall_at_k
from .early_stopping import EarlyStopping
from .libs import AverageMeter, compute_feature, save_checkpoint


def train(train_loader,
          model,
          criterion,
          optimizer,
          scheduler,
          args):
    """Train the model.

    Args:
        train_loader (DataLoader): The training data loader.
        model (CNNs): The model.
        criterion (Loss): The loss function.
        optimizer (Optimizer): The optimizer.
        scheduler: Reduce learning rate scheduler.
        args (Any): The input arguments.

    Returns:
        The trained model.
        A list containing loss values.

    """
    losses = list()
    best_loss = np.inf

    # setup early stopping
    early_stop = EarlyStopping(mode='min', patience=15)

    for epoch in range(0, args.epochs):
        # Rebuiding mini-batchs
        X, y = compute_feature(train_loader.standard_loader, model, args)
        end = time.time()
        train_loader.generate_batches(X, y, n_jobs=args.workers)
        print('Recomputed batches...%.8f' % (time.time() - end))

        # run an epoch
        loss = run_epoch(train_loader, model, criterion,
                         optimizer, epoch, args)
        # update the best parameters
        save_checkpoint({
            'epoch': epoch + 1,
            'arch': args.arch,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
        }, loss < best_loss)

        # keep tracking
        best_loss = min(best_loss, loss)
        losses.append(loss)

        print('Loss=%.4f' % loss)

        # adjust the learning rate
        scheduler.step(loss)

        # early stopping
        if early_stop.step(loss):
            print('Early stopping reached! Stop running...')
            break

    return model, losses


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


def test(test_loader, model, args):
    """Test the model.

    Args:
        test_loader (DataLoader): The data set loader.
        model: The network.
        criterion (Loss): The loss function.
        args: The hyperparameter arguments.

    Returns:
        The NMI score.
        The recall@k results.

    """
    features, labels = compute_feature(test_loader, model, args)
    topk = np.arange(1, 101, 1)
    # compute the recall
    results = recall_at_k(features, labels, topk)
    recalls = pd.DataFrame({'k': topk, 'recall': results})
    # compute the NMI score
    nmi = nmi_clustering(features, labels)

    return nmi, recalls
