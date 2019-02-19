import torch
import torch.nn as nn


class TripletLoss(nn.Module):
    """Implement the triplet loss function [bac]_.

    Args:
        margin (float, optional): The loss margin. Defaults to 0.5.

    .. bac:
    """

    def __init__(self, margin=0.5, **kwargs):
        super(TripletLoss, self).__init__()
        self.margin = margin

    def forward(self, inputs, targets, *args):
        T = args[0]
        n = T.shape[1]
        maxBlocks = 100
        loss = list()  # total loss
        for i in range(0, n, maxBlocks):
            addBlocks = min(maxBlocks, n - i)
            ancs = inputs[T[0][i:i+addBlocks]]
            tars = inputs[T[1][i:i+addBlocks]]
            imps = inputs[T[2][i:i+addBlocks]]
            pos = torch.sum(torch.pow(tars - ancs, 2), dim=1, keepdim=True)
            neg = torch.sum(torch.pow(imps - ancs, 2), dim=1, keepdim=True)
            loss.append(torch.sum(torch.clamp(
                pos + self.margin - neg, 0),  dim=0, keepdim=True)/n)
        return torch.sum(torch.cat(loss))
