# TODO: Several model will be develop
import torch.nn as nn
import torchvision.models as models
from .embedding import Embedding
import torch.nn.functional as F


class CNNs(nn.Module):

    def __init__(self,
                 out_dim,
                 arch='resnet18',
                 pretrained=True,
                 dropout=None,
                 normalized=True):
        super(CNNs, self).__init__()

        if pretrained:
            print("=> using pre-trained model '{}'\n".format(arch))
            model = models.__dict__[arch](pretrained=True)
        else:
            print("=> creating model '{}'\n".format(arch))
            model = models.__dict__[arch]()

        if arch.startswith('alexnet'):
            self.features = model.features
            self.classifier = nn.Sequential(
                nn.Dropout(),
                Embedding(256 * 6 * 6, out_dim, dropout, normalized)
            )
        elif arch.startswith('resnet'):
            # Everything except the last linear layer
            self.features = nn.Sequential(
                *list(model.children())[:-1])
            self.classifier = Embedding(
                model.fc.in_features, out_dim, dropout, normalized
            )
        elif arch.startswith('densenet'):
            self.features = model.features
            self.classifier = Embedding(
                model.classifier.in_features, out_dim, dropout, normalized
            )
        elif arch.startswith('vgg'):
            self.features = model.features
            self.classifier = nn.Sequential(
                nn.Dropout(),
                Embedding(25088, out_dim, dropout, normalized)
            )
        else:
            raise ("Finetuning not supported on this architecture yet")

        self.modelName = arch

    def forward(self, x):
        f = self.features(x)
        if self.modelName.startswith('densenet'):
            out = F.relu(f, inplace=True)
            out = F.avg_pool2d(out, kernel_size=7).view(f.size(0), -1)
            y = self.classifier(out)
        else:
            f = f.view(f.size(0), -1)
            y = self.classifier(f)
        return y
