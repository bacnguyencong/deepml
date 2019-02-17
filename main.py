import numpy as np
import torch
import os
# from deepml.models import CNNs

from deepml.utils import libs
from torch.utils.data import DataLoader

from deepml.datasets import Stanford, Cub
from deepml.utils import RandomIdentitySampler
from PIL import Image

"""
model = CNNs(out_dim=12, arch='bninception')
input_224 = torch.from_numpy(np.random.randn(3, 3, 224, 224)).float()
x = model(input_224)
print(x)
"""
data_path = os.path.abspath('./data/cars196')
data_path = os.path.abspath('./data/stanford')
data_path = os.path.abspath('./data/cub_200_2011')
data = Cub(data_path)

dloader = data.get_dataloader(
    ttype='train',
    inverted=False,
    transform=libs.get_data_augmentation(
        img_size=224,
        mean=[1, 1, 1],
        std=[0, 0, 0],
        ttype='train'
    )
)
sampler = RandomIdentitySampler(dloader, 8)
train_loader = DataLoader(
    dloader,
    sampler=sampler,
    batch_size=128,
    drop_last=True,
    # shuffle=True,
    num_workers=1,
    pin_memory=0
)

print(len(train_loader), len(sampler))


"""
same = data.data_df['train']['label'] == 765
go = data.data_df['train'][same].reset_index()

for i in range(4):
    Image.open(go['img'][i]).show()
"""
