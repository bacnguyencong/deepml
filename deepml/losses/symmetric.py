import torch
import torch.nn as nn

from ..utils.libs import build_triplets, _check_triplets


class SymTripLoss(nn.Module):
    """Implement the symmetric loss

    Args:
        margin (float, optional): The loss margin.
    .. bac:
    """

    def __init__(self, n_targets, verbose=False, **kwargs):
        super(SymTripLoss, self).__init__()
        self.n_targets = n_targets
        self.verbose = verbose

    def forward(self, inputs, targets, *args):
        if len(args) == 0:
            # compute the triplets
            if self.verbose:
                print('Online computing triplets')
            T = build_triplets(
                X=inputs.cpu().detach().numpy(),
                y=targets.cpu().detach().numpy(),
                n_targets=self.n_targets
            )
        else:
            T = args[0]

        if len(T) == 0:
            return torch.zeros(1, requires_grad=True)
        if self.verbose:
            _check_triplets(T, inputs.cpu().detach().numpy(),
                            targets.cpu().detach().numpy())
        # block size
        maxBlocks = 500
        n = T.shape[1]
        loss = list()  # total loss
        for i in range(0, n, maxBlocks):
            addBlocks = min(maxBlocks, n - i)
            ancs = inputs[T[0][i:i+addBlocks]]
            tars = inputs[T[1][i:i+addBlocks]]
            imps = inputs[T[2][i:i+addBlocks]]
            pos = torch.sum(torch.pow(tars - ancs, 2), dim=1, keepdim=True)
            neg_a = torch.sum(torch.pow(imps - ancs, 2), dim=1, keepdim=True)
            neg_b = torch.sum(torch.pow(imps - tars, 2), dim=1, keepdim=True)
            # cat into tensor of two columns
            out = torch.cat([-pos, -0.5*(neg_a + neg_b)], dim=1)
            logs = pos + torch.logsumexp(out, dim=1, keepdim=True)
            loss.append(torch.sum(logs, dim=0, keepdim=True))
        return torch.sum(torch.cat(loss)) / n
