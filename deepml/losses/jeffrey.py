import torch.nn as nn
import torch


class Jeffrey(nn.Module):
    def __init__(self, **kwargs):
        super(Jeffrey, self).__init__()

    def forward(self, inputs, targets):
        n = inputs.size(0)  # number of inputs
        sim = list()  # total loss
        dis = list()
        for i in range(n):
            dist = torch.pairwise_distance(inputs[i], inputs, keepdim=True)
            dist = torch.pow(dist, 2)
            # adding positive
            pos = torch.masked_select(dist, targets == targets[i])
            if pos.size(0) > 0:
                sim.append(pos)
            # adding negative
            neg = torch.masked_select(dist, targets != targets[i])
            if neg.size(0) > 0:
                dis.append(neg)
        a = torch.mean(torch.cat(sim, dim=0))
        b = torch.mean(torch.cat(dis, dim=0))
        return -a/b - b/a
        """
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
        """
