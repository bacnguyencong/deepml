import torch.nn as nn
import torch


class ContrastiveLoss(nn.Module):
    """Implement the contrastive loss function [Hadsell]_.

    Args:
        margin (float, optional): The loss margin. Defaults to 0.5.

    .. [Hadsell] Hadsell, R., Chopra, S. and LeCun, Y. "Dimensionality reduction by learning an invariant mapping." CVPR, 2006, pp. 1735-1742.
    """

    def __init__(self, margin=0.5, **kwargs):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, inputs, targets):
        n = inputs.size(0)  # number of inputs
        loss = torch.Tensor([0])  # total loss
        for i in range(n):
            dist = torch.pairwise_distance(inputs[i], inputs, keepdim=True)
            dist = torch.pow(dist, 2)
            # adding positive
            loss += torch.sum(torch.masked_select(dist, targets == targets[i]))
            # adding negative
            loss += torch.sum(torch.clamp(self.margin - torch.masked_select(
                dist, targets != targets[i]), 0.0))
        return loss / n


if __name__ == "__main__":
    data_size = 32
    input_dim = 3
    output_dim = 2
    num_class = 4
    x = torch.rand(data_size, input_dim)
    w = torch.rand(input_dim, output_dim)
    inputs = x.mm(w)
    targets = torch.randint(0, num_class, (data_size, 1))

    print(ContrastiveLoss()(inputs, targets))
