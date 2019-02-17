import numpy as np
import torch
import os
# from deepml.models import CNNs

from deepml.datasets import Stanford
from deepml.utils import FastRandomIdentitySampler
from PIL import Image

"""
model = CNNs(out_dim=12, arch='bninception')
input_224 = torch.from_numpy(np.random.randn(3, 3, 224, 224)).float()
x = model(input_224)
print(x)
"""
data_path = os.path.abspath('./data/cars196')
data_path = os.path.abspath('./data/stanford')
data = Stanford(data_path)

FastRandomIdentitySampler(data.get_dataloader('train', None, False), 2)


"""
same = data.data_df['train']['label'] == 765
go = data.data_df['train'][same].reset_index()

for i in range(4):
    Image.open(go['img'][i]).show()
"""
