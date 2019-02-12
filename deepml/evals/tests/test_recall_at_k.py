import numpy as np

from ..recall_at_k import recall_at_k

num_examples = 100
num_dim = 5
num_labels = 3
topk = (1, 3, 5)


def compute_recall(features, labels, k):
    n = features.shape[0]
    dist = np.zeros((n, 1))
    ret = 0
    for i in range(n):
        x = features[i]
        for j in range(n):
            y = features[j]
            dist[j] = np.sum((x - y)*(x-y))
        dist[i] = np.inf
        match = np.argsort(dist, axis=0)
        for t in range(k):
            if labels[match[t]] == labels[i]:
                ret += 1
                break
    return ret / n


def test_recall_at_k():
    features = np.random.randn(num_examples, num_dim)
    labels = np.random.randint(num_labels, size=(num_examples, 1))
    ret = recall_at_k(features, labels, topk)
    for i, x in enumerate(ret):
        assert abs(x - compute_recall(features, labels, topk[i])) < 1e-8
