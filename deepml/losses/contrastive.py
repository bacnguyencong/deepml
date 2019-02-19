import torch
import torch.nn as nn


class ContrastiveLoss(nn.Module):
    """Implement the contrastive loss function [Hadsell]_.

    Args:
        margin (float, optional): The loss margin. Defaults to 0.5.

    .. [Hadsell] Hadsell, R., Chopra, S. and LeCun, Y. "Dimensionality
        reduction by learning an invariant mapping." CVPR, 2006, pp. 1735-1742.
    """

    def __init__(self, margin=0.5, **kwargs):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, inputs, targets, *args):
        n = inputs.size(0)  # number of inputs
        loss = list()  # total loss
        for i in range(n):
            dist = torch.pairwise_distance(inputs[i], inputs, keepdim=True)
            dist = torch.pow(dist, 2)
            # adding positive
            pos = torch.masked_select(dist, targets == targets[i])
            if pos.size(0) > 0:
                loss.append(torch.sum(pos, dim=0, keepdim=True))
            # adding negative
            neg = self.margin-torch.masked_select(dist, targets != targets[i])
            if neg.size(0) > 0:
                loss.append(torch.sum(torch.clamp(
                    neg, 0.0), dim=0, keepdim=True))
        return torch.sum(torch.cat(loss)) / n
