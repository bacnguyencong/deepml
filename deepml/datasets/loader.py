
from collections import defaultdict

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from ..utils.libs import build_batches


class DeepMLDataLoader(object):

    def __init__(self, dataset, temp_data, batch_size=128, shuffle=False,
                 n_targets=3, num_workers=8, pin_memory=False):
        self.batch_size = batch_size
        self.n_targets = n_targets
        self.dataset = dataset
        self.batches = None
        self.standard_loader = DataLoader(
            temp_data,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=pin_memory
        )

    def generate_batches(self, X, y):
        self.batches = build_batches(X, y, self.n_targets, self.batch_size)

    def __iter__(self):
        """Returns a generator containing inputs, targets, triplets."""
        for batch in self.batches:
            inputs, targets = [], []
            for i in batch[0]:
                inputs.append(self.dataset[i][0])
                targets.append(self.dataset[i][1])
            targets = torch.from_numpy(np.array(targets).reshape(-1, 1))
            yield (torch.stack(inputs), targets, batch[1])

    def __len__(self):
        return 0 if self.batches is None else len(self.batches)


class DeepMLDataset(Dataset):
    """Dataset for deep metric learning

    Args:
        df_data (Dataframe): A dataframe contains two columns
            img: the path of each image
            label: the labels
    """

    def __init__(self, df_data, inverted=False, transform=None):
        super(DeepMLDataset, self).__init__()
        self.df_data = df_data
        self.transform = transform
        self.is_test = 'label' in df_data.columns
        self.inverted = inverted
        # compute an Index dictionary for every label
        self.Index = defaultdict(list)
        for i, pid in enumerate(df_data['label']):
            self.Index[pid].append(i)

    def __getitem__(self, index):
        img_path = self.df_data['img'][index]
        img = Image.open(img_path).convert('RGB')
        if self.inverted:
            r, g, b = img.split()
            img = Image.merge("RGB", (b, g, r))
        label = self.df_data['label'][index] if self.is_test else -1
        if self.transform is not None:
            img = self.transform(img)
        return img, label

    def __len__(self):
        return len(self.df_data)
