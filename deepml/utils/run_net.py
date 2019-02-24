import os
import numpy as np
import pandas as pd
from ..evals import nmi_clustering, recall_at_k
from .early_stopping import EarlyStopping
from .libs import compute_feature, save_checkpoint, AverageMeter


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
    early_stop = EarlyStopping(mode='max', patience=15)

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

        if test_loader is not None:
            t_acc = validate(test_loader, model, args, topk)
            print('Test\tRecall\t@1=%.4f\t@5=%.4f' % (t_acc[0], t_acc[1]))
            tests.append(t_acc)

        # adjust the learning rate
        scheduler.step(acc[0])

        # early stopping
        if early_stop.step(acc[0]):
            print('Early stopping reached! Stop running...')
            break

    # --------------------------------------------------------------------#
    # write the output
    tab = pd.DataFrame({
        'epoch': range(1, len(losses) + 1),
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
