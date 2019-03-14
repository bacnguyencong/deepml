import numpy as np


def recall_at_k(features, labels, topk=([1, 2, 4])):
    n = features.shape[0]
    assert(max(topk) <= n and min(topk) > 0)
    # compute the distances
    squared_dist = np.sum(features**2, axis=1, keepdims=True)
    dist = squared_dist + squared_dist.T - 2 * np.dot(features, features.T)
    np.fill_diagonal(dist, np.inf)
    # compute the top smallest values
    idx = np.argsort(dist, axis=1)
    ret = np.zeros_like(topk).astype(np.float)

    for i, k in enumerate(topk):
        match = labels[idx[:, :k]].reshape(-1, k) == labels
        ret[i] = np.sum(match.any(axis=1)) / n
    return ret
