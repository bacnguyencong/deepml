import numpy as np
import torch
import os
import matplotlib.pyplot as plt
#from deepml.models import CNNs

from deepml.datasets import Car
from PIL import Image

"""
model = CNNs(out_dim=12, arch='bninception')
input_224 = torch.from_numpy(np.random.randn(3, 3, 224, 224)).float()
x = model(input_224)
print(x)
"""
data_path = os.path.abspath('./data/cars196')
data = Car(data_path)
train = data.get_dataloader(ttype='train', transform=None, inverted=False)
