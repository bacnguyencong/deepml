from deepml.model import CNNs
import torch
import numpy as np

from torchvision import models

# print(models.__dict__)

model = CNNs(out_dim=10, arch='resnet18')
input_shape = (3, 224, 224)

x = torch.rand(1, *input_shape)

y = model(x)
print(y)
