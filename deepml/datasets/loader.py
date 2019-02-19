
from PIL import Image
from torch.utils.data import Dataset
from collections import defaultdict
from sklearn.cluster import KMeans
import numpy as np
import numpy.matlib as matl
import torch
from sklearn.metrics import pairwise_distances


def _generate_triplet(inds, tars, imps):
    k1 = tars.shape[0]
    k2 = imps.shape[0]
    n = inds.shape[0]
    """ Check
    for i in range(k1):
        assert n == len(np.unique(np.hstack([inds, tars[i, :]])))
    assert k2 == len(np.unique(imps))
    assert n == len(np.unique(inds))
    """
    T = np.zeros((3, n*k1*k2), dtype=np.int)
    T[0] = matl.repmat(inds.reshape(-1, 1), 1, k1 * k2).flatten()
    T[1] = matl.repmat(tars.T.flatten().reshape(-1, 1), 1, k2).flatten()
    T[2] = matl.repmat(imps.reshape(-1, 1), k1 * n, 1).flatten()
    return T


def build_triplets(X, y, n_target=3):
    """Compute all triplet constraints.

    Args:
        X (np.array, shape = [n_samples, n_features]): The input data.
        y (np.array, shape = (n_samples,) ): The labels.
        n_target (int, optional): Defaults to 3. The number of targets.

    Returns:
        (np.array, shape = [3, n_triplets]): The triplet index
    """
    dist = pairwise_distances(X, X)
    np.fill_diagonal(dist, np.inf)
    # list of triplets
    Triplets = list()
    for label in np.unique(y):
        targets = np.where(label == y)[0]
        imposters = np.where(label != y)[0]
        # remove group of examples with a few targets or no imposters
        if len(targets) > n_target and len(imposters) > 0:
            # compute the targets
            index = np.argsort(dist[targets, :][:, targets], axis=0)[
                0:n_target]
            Triplets.append(_generate_triplet(
                targets, targets[index], imposters))
    if len(Triplets) > 0:
        Triplets = np.hstack(Triplets)
    return Triplets


def _check_triplets(T, X, y):
    for t in range(T.shape[1]):
        i, j, k = T[:, t]
        assert(y[i] == y[j] and y[i] != y[k])


def build_batches(X, y, n_target=3, batch_size=128):
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
    n_clusters = max(1, X.shape[0] // batch_size)
    model = KMeans(n_clusters=n_clusters).fit(X)

    # generate all triplet constraints
    batches = list()
    for label in np.unique(model.labels_):
        index = np.where(model.labels_ == label)[0]
        # if the number of examples is larger than requires
        if len(index) > batch_size:
            index = np.random.choice(index, batch_size, replace=False)
        triplets = build_triplets(X[index], y[index], n_target=n_target)
        if len(triplets) > 0:
            batches.append((index, triplets))

    return batches


def _check_batches(batches, X, y):
    """Verify the generate batches are correct."""
    for batch in batches:
        index = batch[0]
        _check_triplets(batch[1], X[index], y[index])
        assert len(batch[1]) == len(np.unique(batch[1], axis=1))


class DeepMLDataLoader(object):

    def __init__(self, dataset, batch_size=128, n_targets=3):
        self.batch_size = batch_size
        self.n_targets = n_targets
        self.dataset = dataset
        self.batches = None

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
