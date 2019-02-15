import torch.nn as nn
import torch


class Jeffrey(nn.Module):
    def __init__(self, **kwargs):
        super(Jeffrey, self).__init__()

    def forward(self, inputs, targets):
        dist = torch.pairwise_distance(inputs, torch.cat(
            (inputs[-1:], inputs[:-1])), keepdim=True)
        dist = torch.pow(dist, 2)
        pos = neg = 1e-6
        selected = torch.masked_select(dist, targets == torch.cat(
            (targets[-1:], targets[:-1])))
        nonselec = torch.masked_select(dist, targets != torch.cat(
            (targets[-1:], targets[:-1])))
        if selected.size(0) > 0:
            pos = pos + torch.mean(selected)
        else:
            print('no pos found')
        if nonselec.size(0) > 0:
            neg = neg + torch.mean(nonselec)
        else:
            print('no neg found')
        return -pos / neg - neg / pos
