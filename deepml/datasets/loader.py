
from collections import defaultdict

import numpy as np
import torch
from PIL import Image
from sklearn.cluster import KMeans
from torch.utils.data import DataLoader, Dataset
from ..utils.libs import build_triplets


class DeepMLDataLoader(object):

    def __init__(self, dataset, batch_size=128, shuffle=False,
                 n_targets=None, num_workers=8, pin_memory=False):
        self.batch_size = batch_size
        self.dataset = dataset
        self.batches = None
        self.n_targets = n_targets
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.standard_loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=pin_memory
        )

    def generate_batches(self, X, y):
        """Build the batches from training data.

        Args:
            X ([type]): [description]
            y ([type]): [description]
            n_target (int, optional): Defaults to 3. Number of targets.
            batch_size (int, optional): Defaults to 128.

        Returns:
            List((indices, triplets)): A list of indices and the corresponding
                triplet constraints.
        """
        assert len(X) == len(y)
        # compute the clusters using kmeans
        n_clusters = max(1, X.shape[0] // self.batch_size)
        model = KMeans(n_clusters=n_clusters).fit(X)
        # if offline
        offline = self.n_targets is not None
        # generate all triplet constraints
        self.batches = list()
        for label in np.unique(model.labels_):
            index = np.where(model.labels_ == label)[0]
            # if the number of examples is larger than requires
            if len(index) > self.batch_size:
                index = np.random.choice(index, self.batch_size, replace=False)
            if offline:
                triplets = build_triplets(
                    X[index], y[index], n_targets=self.n_targets)

            if len(triplets) > 0:
                self.batches.append((index, triplets))

            if (not offline) and len(index) > 0:
                self.batches.append(index)

    def __iter__(self):
        """Returns a generator containing inputs, targets."""
        for batch in self.batches:
            inputs, targets = [], []
            for i in batch:
                inputs.append(self.dataset[i][0])
                targets.append(self.dataset[i][1])
            targets = torch.from_numpy(np.array(targets).reshape(-1, 1))
            yield (torch.stack(inputs), targets)

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
