import torch

from ..contrastive import ContrastiveLoss

data_size = 32
input_dim = 3
output_dim = 2
num_class = 4


def test_ContrastiveLoss():
    # margin = 0.5
    x = torch.rand(data_size, input_dim)
    w = torch.rand(input_dim, output_dim)
    inputs = x.mm(w)
    targets = torch.randint(0, num_class, (data_size, 1))
    assert(print(ContrastiveLoss()(inputs, targets)) != 0)
