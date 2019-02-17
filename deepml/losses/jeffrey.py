import torch.nn as nn
import torch


class Jeffrey(nn.Module):
    def __init__(self, **kwargs):
        super(Jeffrey, self).__init__()
        self.loss = nn.BCEWithLogitsLoss()

    def cov(self, inputs, eps):
        """Compute the covariance of differences."""
        diff = inputs.detach()
        cov = torch.sum(torch.pow(diff, 2), 0, keepdim=True) + eps
        return cov / diff.size(0)

    def forward(self, inputs, targets):
        n = inputs.size(0)  # number of inputs
        sim = list()  # total loss
        dis = list()
        for i in range(n):
            # adding positive
            idx = targets.squeeze() == targets[i]
            idx[i] = 0  # remove i-th sample
            pos = inputs[idx]
            if pos.size(0) > 0:
                sim.append(inputs[i] - pos)

            # adding negative
            neg = inputs[targets.squeeze() != targets[i]]
            if neg.size(0) > 0:
                dis.append(inputs[i] - neg)

        pos_dif = torch.cat(sim, dim=0)
        neg_dif = torch.cat(dis, dim=0)
        pos_tag = torch.ones((pos_dif.size(0), 1), dtype=torch.float).cuda()
        neg_tag = torch.zeros((neg_dif.size(0), 1), dtype=torch.float).cuda()

        # compute covariance matrices
        sigma0 = self.cov(neg_dif, 1e-1)
        sigma1 = self.cov(pos_dif, 1e-1)

        # print('0 - Min: {}, Max: {}'.format(torch.min(sigma0), torch.max(sigma0)))
        # print('1 - Min: {}, Max: {}'.format(torch.min(sigma1), torch.max(sigma1)))
        # print(sigma0)
        # print(sigma1)

        weights = torch.pow(sigma0, -1) - torch.pow(sigma1, -1)
        # compute (x_i - x_j)^T * (Sigma^{-1}_1 - Simga^{-1}_0) * (x_i - x_j)
        a = torch.sum(torch.pow(pos_dif, 2) * weights, dim=1, keepdim=True)
        b = torch.sum(torch.pow(neg_dif, 2) * weights, dim=1, keepdim=True)

        coef = torch.sum(torch.log(sigma0) - torch.log(sigma1))
        """
        print('{} {}'.format(torch.mean(a) * 0.5 +
                             coef, torch.mean(b) * 0.5 + coef))
        """
        in_logits = torch.cat([a, b], dim=0)  # * 0.5 + coef
        out_logits = torch.cat([pos_tag, neg_tag])

        # compute the loss
        return self.loss(in_logits, out_logits)
        """
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
        print(a, b)
        return - 128.0*(b/a) - torch.log(a) + torch.log(b)
        """
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
