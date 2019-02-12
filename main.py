from deepml.models import CNNs
import torch
import numpy as np

model = CNNs(out_dim=12, arch='bninception')
input_224 = torch.from_numpy(np.random.randn(3, 3, 224, 224)).float()
x = model(input_224)
print(x)
