import torch
import torch.nn as nn

from ..utils import libs


class TripletLoss(nn.Module):
    """Implement the triplet loss function.

    Args:
        margin (float, optional): The loss margin. Defaults to 0.5.

    """

    def __init__(self, margin=0.2, n_targets=5, **kwargs):
        super(TripletLoss, self).__init__()
        self.margin = margin
        self.n_targets = n_targets

    def forward(self, inputs, targets, *args):
        T = libs.build_triplets(
            inputs.cpu().detach().numpy(),
            targets.cpu().detach().numpy(),
            n_target=self.n_targets
        )
        if len(T) == 0:
            return torch.zeros(1, requires_grad=True)
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
                pos + self.margin - neg, 0),  dim=0, keepdim=True))
        return sum(loss)/n
