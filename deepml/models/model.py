import torch.nn as nn
import pretrainedmodels
import torch


class CNNs(nn.Module):
    """Build a convolutional neural network.

    Args:
        out_dim: The number of desired dimension.
        arch: The architechture name.
        pretrained: If use a pretrained network.
        normalized: If normalize the last layer.
    """

    def __init__(self, arch, out_dim, pretrained=None, normalized=True):
        super(CNNs, self).__init__()
        model = pretrainedmodels.__dict__[arch](pretrained=pretrained)
        model.last_linear = nn.Linear(model.last_linear.in_features, out_dim)
        self.arch = arch
        self.base = model
        self.normalized = normalized

    def forward(self, x):
        out = self.base(x)
        if self.normalized:
            out = torch.nn.functional.normalize(out, p=2, dim=1)
        return out
